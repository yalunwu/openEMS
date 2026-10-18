/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "setup_sampling_hip.h"
#include "setup_processing_hip.h"
#include "FDTD/gpu_setup_gather.h"
#include "FDTD/gpu_setup_memory.h"
#include "FDTD/operator.h"
#include "hip_gather.hiph"
#include "tools/setup_timer.h"
#include <limits>

HIP_FieldGather::~HIP_FieldGather() {if (entries) hipFree(entries);if (values) hipFree(values);}
bool GPU_Backend_HIP::HasDeviceConstruction() const {return bool(d->setup_result);}
void HIP_FieldGather::Read(std::vector<Engine_GatherEntry>& output) const
{
	output.resize(count);
	HIP_Check(hipMemcpyAsync(output.data(),entries,count*sizeof(GPU_GatherEntry),hipMemcpyDeviceToHost,context->stream),"requested sampling map readback");
	HIP_Check(hipStreamSynchronize(context->stream),"requested sampling map synchronization");
	if (diagnostics) std::cout << "GPU requested sampling map readback: bytes=" << count*sizeof(GPU_GatherEntry) << std::endl;
}

__global__ static void EvaluateSamples(const float* fields,const GPU_GatherEntry* entries,uint32_t first,uint32_t count,float* output)
{
	const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if (i<count) output[i]=gather_value(fields,entries[first+i]);
}
bool HIP_FieldGather::Evaluate(float* output,size_t size) const
{
	if (size!=count || !backend || (size && !output)) throw std::invalid_argument("requested field sampling dimensions differ");
	const size_t capacity=(std::min)(count,size_t(65536)),nodes=count/3;
	std::vector<float> samples(capacity);
	for (size_t first=0;first<count;first+=capacity)
	{
		const uint32_t part=uint32_t((std::min)(capacity,count-first));
		EvaluateSamples<<<(part+255u)/256,256,0,context->stream>>>(magnetic ? backend->curr : backend->volt,entries,uint32_t(first),part,values);
		HIP_Check(hipGetLastError(),"requested field sampling");
		HIP_Check(hipMemcpyAsync(samples.data(),values,part*sizeof(float),hipMemcpyDeviceToHost,context->stream),"requested sampled fields readback");backend->Flush();
		// Reorder the requested output for its host file encoder; interpolation
		// and metrics have already been evaluated on the device.
		for (uint32_t i=0;i<part;++i) {const size_t index=first+i;output[(index%3)*nodes+index/3]=samples[i];}
	}
	if (diagnostics) std::cout << "GPU requested sampled fields: d2h_bytes=" << count*sizeof(float) << std::endl;
	return true;
}

struct SamplingInputs
{
	GPU_SetupMetrics metrics;
	uint32_t count,interpolation,lines[3],offsets[3];
	const uint32_t* positions;
	bool magnetic;
};
__global__ static void PrepareSamplingGrid(SamplingInputs input)
{
	const uint32_t n=blockIdx.y,p=blockIdx.x*blockDim.x+threadIdx.x;
	if (p>=input.metrics.mesh.dims[n]) return;
	GPU_SetupInitializeLine(const_cast<double*>(input.metrics.mesh.grid)+input.metrics.mesh.offsets[n],input.metrics.mesh.dims[n],p,
	                        input.metrics.unit,input.metrics.closed_alpha && n==1);
}
__global__ static void BuildSamplingEntries(SamplingInputs input,GPU_GatherEntry* output)
{
	const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;if (id>=input.count) return;
	uint32_t q=id/3;int p[3];
	for (int n=2;n>=0;--n) {p[n]=int(input.positions[input.offsets[n]+q%input.lines[n]]);q/=input.lines[n];}
	GPU_SetupGatherBuilder builder={input.metrics,int(id%3),input.magnetic};output[id]=builder.Build(p,input.interpolation);
}

std::shared_ptr<GPU_FieldGather> GPU_Backend_HIP::BuildFieldGather(const Operator* op,unsigned int interpolation,bool magnetic,
                                       const unsigned int lines[3],unsigned int* const positions[3],const std::shared_ptr<Engine_SamplingMesh>& prepared_mesh)
{
	if (!d->setup_result) return {};
	SetupTimer timer("GPU processing construction");SamplingInputs input={};
	input.interpolation=interpolation;input.magnetic=magnetic;input.metrics.unit=op->GetGridDelta();
	if (interpolation>2) throw std::invalid_argument("invalid sampling interpolation");
	input.metrics.cylindrical=d->setup_result->cylindrical;
	input.metrics.closed_alpha=d->setup_result->closed_alpha;input.metrics.axis=d->setup_result->axis;
	const auto mesh=std::dynamic_pointer_cast<HIP_SamplingMesh>(prepared_mesh);
	if (prepared_mesh && (!mesh || mesh->context!=d->ctx)) throw std::logic_error("sampling mesh context mismatch");
	std::vector<double> grid;std::vector<uint32_t> raw_positions;size_t count=3;uint32_t max_lines=0;
	for (int n=0;n<3;++n)
	{
		input.lines[n]=lines[n];count=GPU_SetupMultiply(count,lines[n]);
		if (mesh)
		{
			if (lines[n]!=mesh->lines[n]) throw std::logic_error("sampling mesh dimensions differ");
			input.offsets[n]=mesh->offsets[n];continue;
		}
		input.metrics.mesh.dims[n]=op->GetNumberOfLines(n,true);input.metrics.mesh.offsets[n]=uint32_t(grid.size());
		max_lines=(std::max)(max_lines,input.metrics.mesh.dims[n]);
		for (uint32_t p=0;p<input.metrics.mesh.dims[n];++p) {grid.push_back(op->GetDiscLine(n,p));for (int k=0;k<5;++k) grid.push_back(0);}
		input.offsets[n]=uint32_t(raw_positions.size());
		for (uint32_t p=0;p<lines[n];++p)
		{
			if (positions[n][p]>=input.metrics.mesh.dims[n]) throw std::out_of_range("sampling position outside mesh");
			raw_positions.push_back(positions[n][p]);
		}
	}
	if (!count || count>UINT32_MAX-255u) throw GPU_SetupUnavailable("sampling map index range exceeded");
	input.count=uint32_t(count);
	const size_t map_bytes=GPU_SetupMultiply(count,sizeof(GPU_GatherEntry));
	const size_t value_bytes=GPU_SetupMultiply((std::min)(count,size_t(65536)),sizeof(float));
	const size_t scratch_bytes=GPU_SetupAdd(GPU_SetupMultiply(grid.size(),sizeof(double)),GPU_SetupMultiply(raw_positions.size(),sizeof(uint32_t)));
	size_t free_bytes=0,total_bytes=0;HIP_Check(hipMemGetInfo(&free_bytes,&total_bytes),"sampling memory query");
	const size_t needed=GPU_SetupAdd(GPU_SetupAdd(map_bytes,value_bytes),scratch_bytes);
	const size_t cap=GPU_SetupMemoryLimit(),live=d->ctx->setup_start_free>free_bytes ? d->ctx->setup_start_free-free_bytes : 0;
	if (needed>free_bytes || live>cap || needed>cap-live) throw GPU_SetupUnavailable("insufficient device memory for sampling maps");
	const auto result=std::make_shared<HIP_FieldGather>(d->ctx);result->count=count;result->diagnostics=d->diagnostics;
	result->backend=d;result->magnetic=magnetic;result->mesh=prepared_mesh;
	double* device_grid=nullptr;uint32_t* device_positions=nullptr;
	struct FreeInputs {double*& grid;uint32_t*& positions;~FreeInputs() {if (grid) hipFree(grid);if (positions) hipFree(positions);}} cleanup={device_grid,device_positions};
	auto allocate=[this](void** output,size_t bytes) {
		const auto status=HIP_AllocateDevice(d->Stream(),output,bytes);
		if (status==hipErrorOutOfMemory) {hipGetLastError();throw GPU_SetupUnavailable("insufficient device memory for sampling maps");}
		HIP_Check(status,"sampling allocation");
	};
	allocate(reinterpret_cast<void**>(&result->entries),map_bytes);allocate(reinterpret_cast<void**>(&result->values),value_bytes);
	if (mesh) {input.metrics=mesh->metrics;input.positions=mesh->positions;}
	else
	{
		allocate(reinterpret_cast<void**>(&device_grid),grid.size()*sizeof(double));
		allocate(reinterpret_cast<void**>(&device_positions),raw_positions.size()*sizeof(uint32_t));
		HIP_Check(hipMemcpyAsync(device_grid,grid.data(),grid.size()*sizeof(double),hipMemcpyHostToDevice,d->Stream()),"sampling grid upload");
		HIP_Check(hipMemcpyAsync(device_positions,raw_positions.data(),raw_positions.size()*sizeof(uint32_t),hipMemcpyHostToDevice,d->Stream()),"sampling positions upload");
		input.metrics.mesh.grid=device_grid;input.positions=device_positions;
		PrepareSamplingGrid<<<dim3((max_lines+255)/256,3),256,0,d->Stream()>>>(input);HIP_Check(hipGetLastError(),"sampling metrics construction");
	}
	BuildSamplingEntries<<<(input.count+255)/256,256,0,d->Stream()>>>(input,result->entries);HIP_Check(hipGetLastError(),"sampling map construction");d->Flush();
	if (d->diagnostics) std::cout << "HIP sampling map construction active: entries=" << count << " H2D=" << scratch_bytes << " D2H=0 device_bytes=" << map_bytes+value_bytes << " peak=" << needed << std::endl;
	timer.Report("sampling maps");return result;
}

bool GPU_Backend_HIP::SetPreparedSnapshotGather(const std::vector<std::shared_ptr<GPU_FieldGather>>& volt,
                                               const std::vector<std::shared_ptr<GPU_FieldGather>>& curr)
{
	size_t nv=0,count=0;
	for (const auto& table:volt) nv=GPU_SetupAdd(nv,table->Count());count=nv;
	for (const auto& table:curr) count=GPU_SetupAdd(count,table->Count());
	if (!count || count>UINT32_MAX) return false;
	HIP_Check(hipStreamSynchronize(d->ctx->copy_stream),"sampling snapshot synchronization");d->FreeSnapshots();
	if (HIP_AllocateDevice(d->Stream(),&d->snap_entries,GPU_SetupMultiply(count,sizeof(GPU_GatherEntry)))!=hipSuccess ||
	    HIP_AllocateDevice(d->Stream(),&d->snap_dev,GPU_SetupMultiply(count,sizeof(float)))!=hipSuccess ||
	    HIP_AllocateHost(d->Stream(),&d->snap_host,GPU_SetupMultiply(count,sizeof(float)))!=hipSuccess)
	{
		hipGetLastError();d->FreeSnapshots();return false;
	}
	size_t offset=0;
	for (const auto* list:{&volt,&curr}) for (const auto& table:*list)
	{
		const auto hip=std::dynamic_pointer_cast<HIP_FieldGather>(table);
		if (!hip || hip->context!=d->ctx) throw std::logic_error("sampling table context mismatch");
		HIP_Check(hipMemcpyAsync(d->snap_entries+offset,hip->entries,hip->count*sizeof(GPU_GatherEntry),hipMemcpyDeviceToDevice,d->Stream()),"snapshot sampling map adoption");offset+=hip->count;
	}
	d->snap_nv=uint32_t(nv);d->snap_n=uint32_t(count);d->Flush();return true;
}

int GPU_Backend_HIP::AddPreparedFieldDFT(const std::shared_ptr<GPU_FieldGather>& table,bool currents,unsigned int count)
{
	const auto hip=std::dynamic_pointer_cast<HIP_FieldGather>(table);
	if (!hip || hip->context!=d->ctx) throw std::logic_error("DFT sampling table context mismatch");
	if (!count || !hip->count) return -1;
	Impl::FieldDFT output={};output.n=uint32_t(hip->count);output.count=count;output.currents=currents;
	output.entries=hip->entries;output.prepared=table;
	const size_t values=GPU_SetupMultiply(hip->count,count),bytes=GPU_SetupMultiply(values,sizeof(float2));
	size_t available=0,total=0;HIP_Check(hipMemGetInfo(&available,&total),"DFT memory query");
	const size_t cap=GPU_SetupMemoryLimit(),live=d->ctx->setup_start_free>available ? d->ctx->setup_start_free-available : 0;
	if (bytes>available || live>cap || bytes>cap-live) throw GPU_SetupUnavailable("insufficient device memory for frequency-domain sampling");
	output.sums=d->Alloc<float2>(values);d->field_dfts.push_back(output);
	if (d->diagnostics) std::cout << "HIP sampling map adopted by DFT: entries=" << hip->count << " history_bytes=" << bytes << " H2D=0 D2H=0" << std::endl;
	return int(d->field_dfts.size()-1);
}
