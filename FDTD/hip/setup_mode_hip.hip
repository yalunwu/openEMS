/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "setup_mode_hip.h"
#include "FDTD/gpu_setup_mode_export.h"
#include "FDTD/gpu_setup_metrics.h"
#include "FDTD/gpu_setup_lookup.h"
#include "FDTD/gpu_setup_memory.h"
#include "FDTD/operator.h"
#include "tools/setup_timer.h"

namespace {
struct ModeTable : Engine_SetupTable
{
	std::shared_ptr<HIP_Context> context;
	double* values=nullptr;size_t count=0;bool diagnostics=false;
	explicit ModeTable(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	~ModeTable() {if (values) hipFree(values);}
	size_t Count() const override {return count;}
	void Read(double* output,size_t length) const override
	{
		if (length!=count || !output) throw std::out_of_range("requested mode table size");
		HIP_Check(hipMemcpyAsync(output,values,count*sizeof(double),hipMemcpyDeviceToHost,context->stream),"requested mode table readback");
		HIP_Check(hipStreamSynchronize(context->stream),"requested mode table completion");
		if (diagnostics) std::cout << "GPU requested mode table readback: bytes=" << count*sizeof(double) << std::endl;
	}
};
struct ModeInputs
{
	GPU_SetupMetrics metrics;
	GPU_SetupExpressionView expressions;
	GPU_SetupFileWeight file;
	const double* file_data;
	uint32_t programs[2],start[3],normal,count,width;
	bool magnetic,has_file;
	double origin[3];
};
struct TemporaryBuffers
{
	hipStream_t stream;std::vector<void*> buffers;size_t bytes=0,uploaded=0;
	explicit TemporaryBuffers(hipStream_t s) : stream(s) {}
	~TemporaryBuffers() {for (auto* buffer:buffers) hipFree(buffer);}
	template<class T> T* Add(size_t count,const T* host=nullptr)
	{
		const size_t size=GPU_SetupMultiply((std::max)(count,size_t(1)),sizeof(T));T* buffer=nullptr;
		const auto status=HIP_AllocateDevice(stream,&buffer,size);
		if (status==hipErrorOutOfMemory) {hipGetLastError();throw GPU_SetupUnavailable("insufficient device memory for mode preparation");}
		HIP_Check(status,"mode input allocation");
		try {buffers.push_back(buffer);} catch (...) {hipFree(buffer);throw;}
		bytes=GPU_SetupAdd(bytes,size);
		if (host && count) {HIP_Check(hipMemcpyAsync(buffer,host,count*sizeof(T),hipMemcpyHostToDevice,stream),"mode input upload");uploaded=GPU_SetupAdd(uploaded,count*sizeof(T));}
		else HIP_Check(hipMemsetAsync(buffer,0,size,stream),"mode input initialization");
		return buffer;
	}
};
__global__ void InitializeModeGrid(ModeInputs input)
{
	const uint32_t n=blockIdx.y,p=blockIdx.x*blockDim.x+threadIdx.x;if (p>=input.metrics.mesh.dims[n]) return;
	GPU_SetupInitializeLine(const_cast<double*>(input.metrics.mesh.grid)+input.metrics.mesh.offsets[n],input.metrics.mesh.dims[n],p,input.metrics.unit,input.metrics.closed_alpha && n==1);
}
__global__ void BuildModeSamples(ModeInputs input,double* values,double* squared)
{
	const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;if (id>=input.count) return;
	const int a=(input.normal+1)%3,b=(input.normal+2)%3;
	int p[3]={int(input.start[0]),int(input.start[1]),int(input.start[2])};p[a]+=int(id/input.width);p[b]+=int(id%input.width);
	double coord[7]={input.metrics.Line(0,p[0],input.magnetic),input.metrics.Line(1,p[1],input.magnetic),input.metrics.Line(2,p[2],input.magnetic)};
	if (input.metrics.cylindrical) {const double radius=coord[0],angle=coord[1];coord[0]=radius*cos(angle);coord[1]=radius*sin(angle);}
	for (int n=0;n<3;++n) coord[n]-=input.origin[n];
	coord[3]=sqrt(coord[0]*coord[0]+coord[1]*coord[1]);coord[4]=atan2(coord[1],coord[0]);
	coord[5]=sqrt(coord[0]*coord[0]+coord[1]*coord[1]+coord[2]*coord[2]);coord[6]=asin(1.0)-atan(coord[2]/coord[3]);
	double sum=0;
	for (int n=0;n<2;++n)
	{
		int error=0;double value=input.has_file ? GPU_SetupInterpolateFile(input.file,input.file_data,n==0 ? a : b,coord) :
			GPU_SetupEvaluate(input.expressions.programs[input.programs[n]],input.expressions.instructions,input.expressions.constants,coord,
			                  input.expressions.stack+size_t(id)*input.expressions.capacity,input.expressions.capacity,error);
		if (error || !isfinite(value)) value=0;
		values[size_t(n)*input.count+id]=value;sum+=value*value;
	}
	squared[id]=sum*input.metrics.NodeArea(input.normal,p,input.magnetic);
}
__global__ void ReduceModeNorm(const double* input,uint32_t count,double* output)
{
	__shared__ double values[256];const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x;
	values[lane]=id<count ? input[id] : 0;__syncthreads();
	for (uint32_t stride=128;stride;stride/=2) {if (lane<stride) values[lane]+=values[lane+stride];__syncthreads();}
	if (!lane) output[blockIdx.x]=values[0];
}
__global__ void NormalizeModes(double* values,uint32_t count,const double* squared,uint32_t* error)
{
	const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;if (id>=count) return;
	const double norm=sqrt(*squared);if (!(norm>0) || !isfinite(norm)) {atomicOr(error,1u);return;}
	values[id]/=norm;
}
}

std::shared_ptr<Engine_SetupTable> HIP_BuildModeTable(GPU_Backend_HIP::Impl* backend,const Operator* op,const Engine_ModeTableInput& raw)
{
	if (!backend->setup_result) return {};
	SetupTimer timer("GPU processing construction");
	if (raw.normal>=3) throw std::invalid_argument("invalid mode normal");
	if (raw.start[raw.normal]!=raw.stop[raw.normal]) throw std::invalid_argument("mode sampling requires a plane");
	ModeInputs input={};input.normal=raw.normal;input.magnetic=raw.magnetic;std::copy(raw.origin,raw.origin+3,input.origin);
	const int a=(raw.normal+1)%3,b=(raw.normal+2)%3;size_t count=1;std::vector<double> grid;uint32_t max_lines=0;
	for (int n=0;n<3;++n)
	{
		if (raw.start[n]>raw.stop[n] || raw.stop[n]>=op->GetNumberOfLines(n,true)) throw std::out_of_range("mode sampling bounds");
		input.start[n]=raw.start[n];input.metrics.mesh.dims[n]=op->GetNumberOfLines(n,true);input.metrics.mesh.offsets[n]=uint32_t(grid.size());
		max_lines=(std::max)(max_lines,input.metrics.mesh.dims[n]);
		for (uint32_t p=0;p<input.metrics.mesh.dims[n];++p) {grid.push_back(op->GetDiscLine(n,p));for (int k=0;k<5;++k) grid.push_back(0);}
		if (n!=int(raw.normal)) count=GPU_SetupMultiply(count,size_t(raw.stop[n])-raw.start[n]+1);
	}
	if (count>(UINT32_MAX-255u)/2) throw GPU_SetupUnavailable("mode sampling index range exceeded");
	input.count=uint32_t(count);input.width=raw.stop[b]-raw.start[b]+1;
	input.metrics.unit=op->GetGridDelta();input.metrics.cylindrical=backend->setup_result->cylindrical;input.metrics.closed_alpha=backend->setup_result->closed_alpha;
	const auto model=GPU_ExportModeInput(raw);input.has_file=model.file!=0;
	if (input.has_file) input.file=model.data.file_weights.at(model.file);
	std::copy(model.expressions,model.expressions+2,input.programs);
	const auto& bank=model.data.expressions;
	size_t estimate=GPU_SetupAdd(4096,GPU_SetupMultiply(4*count,sizeof(double)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(GPU_SetupMultiply(count,bank.stack_size),sizeof(double)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(grid.size()+model.data.geometry.size()+bank.constants.size(),sizeof(double)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupAdd(GPU_SetupMultiply(bank.programs.size(),sizeof(GPU_SetupExpression)),GPU_SetupMultiply(bank.instructions.size(),sizeof(GPU_SetupInstruction))));
	size_t free=0,total=0;HIP_Check(hipMemGetInfo(&free,&total),"mode memory query");
	const size_t cap=GPU_SetupMemoryLimit(),live=backend->ctx->setup_start_free>free ? backend->ctx->setup_start_free-free : 0;
	if (estimate>free || live>cap || estimate>cap-live) throw GPU_SetupUnavailable("insufficient device memory for mode tables");
	TemporaryBuffers temporary(backend->Stream());input.metrics.mesh.grid=temporary.Add(grid.size(),grid.data());
	input.expressions.programs=temporary.Add(bank.programs.size(),bank.programs.data());
	input.expressions.instructions=temporary.Add(bank.instructions.size(),bank.instructions.data());
	input.expressions.constants=temporary.Add(bank.constants.size(),bank.constants.data());
	input.expressions.capacity=bank.stack_size;input.expressions.stack=temporary.Add<double>(GPU_SetupMultiply(count,bank.stack_size));
	input.file_data=temporary.Add(model.data.geometry.size(),model.data.geometry.data());
	auto result=std::make_shared<ModeTable>(backend->ctx);result->count=2*count;result->diagnostics=backend->diagnostics;
	const auto status=HIP_AllocateDevice(backend->Stream(),&result->values,GPU_SetupMultiply(2*count,sizeof(double)));
	if (status==hipErrorOutOfMemory) {hipGetLastError();throw GPU_SetupUnavailable("insufficient device memory for mode tables");}HIP_Check(status,"mode table allocation");
	double* squared=temporary.Add<double>(count);double* workspace=temporary.Add<double>((count+255)/256);
	uint32_t* error=temporary.Add<uint32_t>(1);
	InitializeModeGrid<<<dim3((max_lines+255)/256,3),256,0,backend->Stream()>>>(input);HIP_Check(hipGetLastError(),"mode grid metrics");
	BuildModeSamples<<<(input.count+255)/256,256,0,backend->Stream()>>>(input,result->values,squared);HIP_Check(hipGetLastError(),"mode sampling construction");
	double* source=squared;double* target=workspace;uint32_t remaining=input.count;
	while (remaining>1) {const uint32_t blocks=(remaining+255)/256;ReduceModeNorm<<<blocks,256,0,backend->Stream()>>>(source,remaining,target);HIP_Check(hipGetLastError(),"mode norm reduction");remaining=blocks;std::swap(source,target);}
	NormalizeModes<<<(2*input.count+255)/256,256,0,backend->Stream()>>>(result->values,2*input.count,source,error);HIP_Check(hipGetLastError(),"mode normalization");
	uint32_t invalid=0;HIP_Check(hipMemcpyAsync(&invalid,error,sizeof(invalid),hipMemcpyDeviceToHost,backend->Stream()),"mode status readback");backend->Flush();
	if (invalid) throw std::runtime_error("mode matching template has zero or invalid norm");
	std::cout << "HIP mode table construction active" << std::endl;
	if (backend->diagnostics) std::cout << "GPU mode preparation: H2D=" << temporary.uploaded << " D2H=" << sizeof(invalid) << " device_bytes=" << result->count*sizeof(double) << " peak=" << temporary.bytes+result->count*sizeof(double) << std::endl;
	timer.Report("mode matching tables");return result;
}

std::shared_ptr<Engine_SetupTable> GPU_Backend_HIP::BuildModeMatch(const Operator* op,const Engine_ModeTableInput& input)
{
	return HIP_BuildModeTable(d,op,input);
}
