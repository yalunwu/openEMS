/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "setup_processing_hip.h"
#include "FDTD/operator.h"
#include "FDTD/gpu_setup_processing.h"
#include "FDTD/gpu_setup_memory.h"
#include "tools/setup_timer.h"
#include <cmath>

namespace
{
std::shared_ptr<HIP_Context> Context(GPU_SetupResult* result,GPU_Backend* backend)
{
	if (auto* hip=dynamic_cast<HIP_SetupResult*>(result)) return hip->context;
	if (auto* hip=dynamic_cast<GPU_Backend_HIP*>(backend)) return hip->GetImpl()->ctx;
	return {};
}
void Budget(const std::shared_ptr<HIP_Context>& context,size_t needed)
{
	size_t available=0,total=0;HIP_Check(hipMemGetInfo(&available,&total),"processing mesh memory query");
	const size_t limit=GPU_SetupMemoryLimit(),live=context->setup_start_free>available ? context->setup_start_free-available : 0;
	if (needed>available || live>limit || needed>limit-live) throw GPU_SetupUnavailable("insufficient device memory for processing mesh");
}
void Allocate(const std::shared_ptr<HIP_Context>& context,void** output,size_t size)
{
	const auto status=HIP_AllocateDevice(context->stream,output,size);
	if (status==hipErrorOutOfMemory) {hipGetLastError();throw GPU_SetupUnavailable("insufficient device memory for processing mesh");}
	HIP_Check(status,"processing mesh allocation");
}
__global__ void Initialize(GPU_SetupMetrics metrics)
{
	const uint32_t n=blockIdx.y,p=blockIdx.x*blockDim.x+threadIdx.x;
	if (p<metrics.mesh.dims[n]) GPU_SetupInitializeLine(const_cast<double*>(metrics.mesh.grid)+metrics.mesh.offsets[n],metrics.mesh.dims[n],p,
	    metrics.unit,metrics.closed_alpha && n==1);
}
std::shared_ptr<HIP_SamplingMesh> Grid(const std::shared_ptr<HIP_Context>& context,const Operator_Base* op,size_t extra)
{
	auto mesh=std::make_shared<HIP_SamplingMesh>(context);
	mesh->metrics.unit=op->GetGridDelta();
	const char* diagnostic_setting=std::getenv("OPENEMS_GPU_DIAGNOSTICS");mesh->diagnostics=diagnostic_setting && std::atoi(diagnostic_setting)!=0;
	const unsigned int flags=GPU_SetupCoordinateFlags(op);
	mesh->metrics.cylindrical=(flags&1u)!=0;mesh->metrics.closed_alpha=(flags&2u)!=0;mesh->metrics.axis=(flags&4u)!=0;
	std::vector<double> raw;uint32_t maximum=0;
	for (int n=0;n<3;++n)
	{
		const uint32_t count=op->GetNumberOfLines(n,true);
		if (count<2 || count>UINT32_MAX/18 || raw.size()>UINT32_MAX-6ull*count) throw GPU_SetupUnavailable("processing mesh index range exceeded");
		mesh->metrics.mesh.dims[n]=count;mesh->metrics.mesh.offsets[n]=uint32_t(raw.size());maximum=(std::max)(maximum,count);
		for (uint32_t p=0;p<count;++p) {raw.push_back(op->GetDiscLine(n,p));for (int k=0;k<5;++k) raw.push_back(0);}
	}
	const size_t bytes=GPU_SetupMultiply(raw.size(),sizeof(double));Budget(context,GPU_SetupAdd(bytes,extra));
	Allocate(context,reinterpret_cast<void**>(&mesh->grid),bytes);mesh->metrics.mesh.grid=mesh->grid;
	HIP_Check(hipMemcpyAsync(mesh->grid,raw.data(),bytes,hipMemcpyHostToDevice,context->stream),"raw processing mesh upload");
	Initialize<<<dim3((maximum+255)/256,3),256,0,context->stream>>>(mesh->metrics);HIP_Check(hipGetLastError(),"processing mesh metrics");
	// The upload uses a temporary host descriptor. Keep it alive until consumed.
	HIP_Check(hipStreamSynchronize(context->stream),"processing mesh initialization");
	return mesh;
}
struct Selection
{
	uint32_t start[3],stop[3],step[3],offsets[3];double resolution[3];bool dual,optimal;
};
__global__ void Select(GPU_SetupMeshView mesh,Selection input,uint32_t* positions,double* coordinates,uint32_t* counts)
{
	const int n=threadIdx.x;if (n>=3) return;
	uint32_t count=0;const uint32_t offset=input.offsets[n],first=input.start[n],last=input.stop[n];
	if (!input.optimal)
	{
		for (uint32_t p=first;;)
		{
			positions[offset+count++]=p;
			if (input.step[n]>last-p) break;
			p+=input.step[n];
		}
	}
	else
	{
		positions[offset+count++]=first;double previous=mesh.Line(n,first,input.dual ? 1 : 0);
		if (!last) positions[offset+count++]=last; // Preserve the CPU's singleton-at-zero convention.
		else for (uint32_t p=first+1;p<last;++p)
			if (mesh.Line(n,p+1,input.dual ? 1 : 0)-previous>=input.resolution[n])
			{positions[offset+count++]=p;previous=mesh.Line(n,p,input.dual ? 1 : 0);}
		if (first!=last) positions[offset+count++]=last;
	}
	for (uint32_t p=0;p<count;++p) coordinates[offset+p]=mesh.Line(n,positions[offset+p],input.dual ? 1 : 0);
	counts[n]=count;
}
struct SnapInput {double start[3],stop[3];bool dual,full,point,cylindrical,closed;int method;};
struct SnapOutput {uint32_t first[3],last[3],first_inside[3],last_inside[3];int dimension,normal;};
__global__ void Snap(GPU_SetupMeshView mesh,SnapInput input,SnapOutput* output)
{
	bool first_inside[3]={},last_inside[3]={};
	if (input.point)
	{
		GPU_SetupMeshView range=mesh;if (input.closed && !input.full) range.dims[1]-=2;
		output->dimension=0;
		for (int n=0;n<3;++n)
		{
			const double value=n==1 && input.cylindrical ? GPU_SetupFitAlpha(mesh,input.start[n],input.closed) : input.start[n];
			first_inside[n]=value>=range.Line(n,0) && value<=range.Line(n,range.dims[n]-1);
			last_inside[n]=first_inside[n];output->dimension+=first_inside[n] ? 1 : 0;
			output->first[n]=output->last[n]=range.Snap(n,value,input.dual);
		}
	}
	else output->dimension=GPU_SetupSnapBox(mesh,input.start,input.stop,output->first,output->last,input.dual,input.method,
	    input.cylindrical,input.closed,input.full,first_inside,last_inside);
	for (int n=0;n<3;++n) {output->first_inside[n]=first_inside[n];output->last_inside[n]=last_inside[n];}
}
__global__ void AdjustIndices(GPU_SetupMeshView mesh,bool mode_match,bool dual,bool closed,SnapInput input,SnapOutput* output)
{
	if (closed) mesh.dims[1]-=2;
	output->dimension=0;
	if (mode_match)
	{
		output->normal=-1;
		for (int n=0;n<3;++n)
		{
			if (output->first[n]>output->last[n]) {const uint32_t value=output->first[n];output->first[n]=output->last[n];output->last[n]=value;}
			if (!output->first[n]) ++output->first[n];
			if (output->last[n]==mesh.dims[n]-1) --output->last[n];
			else if (dual && output->last[n]>output->first[n]+1) --output->last[n];
			if (output->first[n]!=output->last[n]) ++output->dimension;else output->normal=n;
		}
	}
	else if (output->normal<0 || output->normal>2)
	{
		for (int n=0;n<3;++n) if (output->last[n]!=output->first[n]) ++output->dimension;
		if (output->dimension==2) for (int n=0;n<3;++n) if (output->last[n]==output->first[n]) output->normal=n;
	}
	else for (int n=0;n<3;++n)
	{
		if (n==output->normal) continue;
		if (input.start[n]==input.stop[n])
		{
			if (mesh.Line(n,output->first[n],1)>input.start[n] && output->first[n]>0) --output->first[n];
			if (mesh.Line(n,output->first[n],1)<input.start[n] && output->last[n]<mesh.dims[n]-1) ++output->last[n];
		}
		if (output->last[n]>output->first[n]) ++output->dimension;
	}
}
}

HIP_SamplingMesh::~HIP_SamplingMesh() {if (coordinates) hipFree(coordinates);if (positions) hipFree(positions);if (grid) hipFree(grid);}
void HIP_SamplingMesh::Read(unsigned int* output_positions[3],double* output_coordinates[3]) const
{
	size_t bytes=0;
	for (int n=0;n<3;++n)
	{
		HIP_Check(hipMemcpyAsync(output_positions[n],positions+offsets[n],lines[n]*sizeof(uint32_t),hipMemcpyDeviceToHost,context->stream),"requested output mesh indices");
		HIP_Check(hipMemcpyAsync(output_coordinates[n],coordinates+offsets[n],lines[n]*sizeof(double),hipMemcpyDeviceToHost,context->stream),"requested output mesh coordinates");
		bytes+=lines[n]*(sizeof(uint32_t)+sizeof(double));
	}
	HIP_Check(hipStreamSynchronize(context->stream),"requested output mesh synchronization");
	if (diagnostics) std::cout << "GPU requested output mesh metadata: d2h_bytes=" << bytes << std::endl;
}

std::shared_ptr<Engine_SamplingMesh> HIP_SetupSamplingMesh(GPU_SetupResult* result,GPU_Backend* backend,const Operator_Base* op,
    const unsigned int* start,const unsigned int* stop,bool dual,bool optimal,const unsigned int* step,const double* resolution)
{
	const auto context=Context(result,backend);if (!context) return {};
	SetupTimer timer("GPU processing construction");Selection input={};input.dual=dual;input.optimal=optimal;size_t capacity=0;
	for (int n=0;n<3;++n)
	{
		if (start[n]>stop[n] || stop[n]>=op->GetNumberOfLines(n,true) || !step[n] || !std::isfinite(resolution[n])) throw std::invalid_argument("invalid processing mesh selection");
		input.start[n]=start[n];input.stop[n]=stop[n];input.step[n]=step[n];input.resolution[n]=resolution[n];input.offsets[n]=uint32_t(capacity);
		capacity=GPU_SetupAdd(capacity,size_t(stop[n])-start[n]+2);
		if (capacity>UINT32_MAX) throw GPU_SetupUnavailable("processing selection index range exceeded");
	}
	const size_t bytes=GPU_SetupMultiply(capacity,sizeof(uint32_t)+sizeof(double));
	auto mesh=Grid(context,op,GPU_SetupAdd(bytes,3*sizeof(uint32_t)));std::copy(input.offsets,input.offsets+3,mesh->offsets);
	Allocate(context,reinterpret_cast<void**>(&mesh->positions),capacity*sizeof(uint32_t));
	Allocate(context,reinterpret_cast<void**>(&mesh->coordinates),capacity*sizeof(double));
	uint32_t* counts=nullptr;struct Cleanup {uint32_t*& p;~Cleanup() {if (p) hipFree(p);}} cleanup={counts};
	Allocate(context,reinterpret_cast<void**>(&counts),3*sizeof(uint32_t));
	Select<<<1,3,0,context->stream>>>(mesh->metrics.mesh,input,mesh->positions,mesh->coordinates,counts);HIP_Check(hipGetLastError(),"processing mesh selection");
	HIP_Check(hipMemcpyAsync(mesh->lines,counts,sizeof(mesh->lines),hipMemcpyDeviceToHost,context->stream),"processing mesh counts");
	HIP_Check(hipStreamSynchronize(context->stream),"processing selection synchronization");
	std::cout << "HIP processing mesh selection active" << std::endl;timer.Report("sampling mesh selection");return mesh;
}

bool HIP_SetupProcessingSnap(GPU_SetupResult* result,GPU_Backend* backend,const Operator_Base* op,const double* start,const double* stop,
    unsigned int* first,unsigned int* last,bool dual,bool full,int method,bool point,bool* first_inside,bool* last_inside,int& dimension)
{
	const auto context=Context(result,backend);if (!context) return false;
	SnapInput input={};input.dual=dual;input.full=full;input.method=method;input.point=point;
	for (int n=0;n<3;++n) {if (!std::isfinite(start[n]) || !std::isfinite(stop[n])) throw std::invalid_argument("non-finite snapping coordinate");input.start[n]=start[n];input.stop[n]=stop[n];}
	auto mesh=Grid(context,op,sizeof(SnapOutput));input.cylindrical=mesh->metrics.cylindrical;input.closed=mesh->metrics.closed_alpha;
	SnapOutput* output=nullptr;struct Cleanup {SnapOutput*& p;~Cleanup() {if (p) hipFree(p);}} cleanup={output};
	Allocate(context,reinterpret_cast<void**>(&output),sizeof(SnapOutput));HIP_Check(hipMemsetAsync(output,0,sizeof(SnapOutput),context->stream),"snapping result initialization");
	Snap<<<1,1,0,context->stream>>>(mesh->metrics.mesh,input,output);HIP_Check(hipGetLastError(),"processing snapping");
	SnapOutput host={};HIP_Check(hipMemcpyAsync(&host,output,sizeof(host),hipMemcpyDeviceToHost,context->stream),"processing snapping scalar readback");
	HIP_Check(hipStreamSynchronize(context->stream),"processing snapping synchronization");dimension=host.dimension;
	for (int n=0;n<3;++n) {first[n]=host.first[n];last[n]=host.last[n];if (first_inside) first_inside[n]=host.first_inside[n]!=0;if (last_inside) last_inside[n]=host.last_inside[n]!=0;}
	if (mesh->diagnostics) std::cout << "HIP processing snapping active" << std::endl;
	return true;
}

bool HIP_SetupProcessingIndices(GPU_SetupResult* result,GPU_Backend* backend,const Operator_Base* op,bool mode_match,bool dual,const double* start,const double* stop,
    unsigned int* first,unsigned int* last,int& normal,int& dimension)
{
	const auto context=Context(result,backend);if (!context) return false;
	auto mesh=Grid(context,op,sizeof(SnapOutput));SnapOutput host={};host.normal=normal;
	SnapInput input={};for (int n=0;n<3;++n) {host.first[n]=first[n];host.last[n]=last[n];if (start) input.start[n]=start[n];if (stop) input.stop[n]=stop[n];}
	SnapOutput* output=nullptr;struct Cleanup {SnapOutput*& p;~Cleanup() {if (p) hipFree(p);}} cleanup={output};
	Allocate(context,reinterpret_cast<void**>(&output),sizeof(SnapOutput));
	HIP_Check(hipMemcpyAsync(output,&host,sizeof(host),hipMemcpyHostToDevice,context->stream),"processing scalar registration inputs");
	AdjustIndices<<<1,1,0,context->stream>>>(mesh->metrics.mesh,mode_match,dual,mesh->metrics.closed_alpha,input,output);HIP_Check(hipGetLastError(),"processing index preparation");
	HIP_Check(hipMemcpyAsync(&host,output,sizeof(host),hipMemcpyDeviceToHost,context->stream),"processing scalar registration result");
	HIP_Check(hipStreamSynchronize(context->stream),"processing registration synchronization");normal=host.normal;dimension=host.dimension;
	for (int n=0;n<3;++n) {first[n]=host.first[n];last[n]=host.last[n];}
	return true;
}
