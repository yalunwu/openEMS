/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_internal.h"
#include "FDTD/gpu_setup_memory.h"
void HIP_PrepareUPMLFusion(GPU_Backend_HIP::Impl*);

void HIP_FinishInitialization(GPU_Backend* backend)
{
	if (auto* hip=dynamic_cast<GPU_Backend_HIP*>(backend))
	{
		auto* impl=hip->GetImpl();HIP_PrepareUPMLFusion(impl);impl->DecideFusedStep();
		impl->Flush();
	}
}

__global__ static void FusionBounds(const GPU_UPMLRegion* regions,uint32_t count,HIP_GridDim dims,uint32_t cells,uint32_t* output)
{
	__shared__ uint32_t values[8][256];const uint32_t lane=threadIdx.x,id=blockIdx.x*blockDim.x+lane;
	const uint32_t p[3]={id/(dims.ny*dims.nz),(id/dims.nz)%dims.ny,id%dims.nz},size[3]={dims.nx,dims.ny,dims.nz};
	uint32_t covers=0;
	if (id<cells) for (uint32_t r=0;r<count;++r)
	{
		bool inside=true;for (int n=0;n<3;++n) inside&=p[n]>=regions[r].start[n] && p[n]-regions[r].start[n]<regions[r].size[n];
		covers+=inside ? 1 : 0;
	}
	const bool empty=id<cells && !covers;
	for (int n=0;n<3;++n) {values[n][lane]=empty ? p[n] : size[n];values[3+n][lane]=empty ? p[n]+1 : 0;}
	values[6][lane]=empty ? 1 : 0;values[7][lane]=covers>1 ? 1 : 0;__syncthreads();
	for (uint32_t step=128;step;step/=2)
	{
		if (lane<step)
		{
			for (int n=0;n<3;++n) {values[n][lane]=min(values[n][lane],values[n][lane+step]);values[3+n][lane]=max(values[3+n][lane],values[3+n][lane+step]);}
			values[6][lane]+=values[6][lane+step];values[7][lane]|=values[7][lane+step];
		}
		__syncthreads();
	}
	if (!lane)
	{
		for (int n=0;n<3;++n) {atomicMin(output+n,values[n][0]);atomicMax(output+3+n,values[3+n][0]);}
		atomicAdd(output+6,values[6][0]);atomicOr(output+7,values[7][0]);
	}
}
__global__ static void ValidateFusionBounds(uint32_t* output)
{
	uint64_t volume=1;for (int n=0;n<3;++n) volume*=output[3+n]-output[n];
	if (!output[6] || volume!=output[6]) output[7]=1;
}

bool HIP_SetupUPMLFusion(GPU_Backend_HIP::Impl* backend,const std::vector<GPU_UPMLRegion>& regions,const unsigned int* dims,unsigned int* first,unsigned int* last)
{
	if (regions.empty()) return false;
	size_t cells=1;for (int n=0;n<3;++n) cells=GPU_SetupMultiply(cells,dims[n]);
	if (!cells || cells>UINT32_MAX-255u || regions.size()>UINT32_MAX) throw GPU_SetupUnavailable("UPML fusion index range exceeded");
	for (const auto& region:regions) for (int n=0;n<3;++n) if (region.start[n]>dims[n] || region.size[n]>dims[n]-region.start[n]) return false;
	const size_t bytes=GPU_SetupMultiply(regions.size(),sizeof(GPU_UPMLRegion)),needed=GPU_SetupAdd(bytes,8*sizeof(uint32_t));
	size_t free=0,total=0;HIP_Check(hipMemGetInfo(&free,&total),"UPML fusion memory query");
	const size_t cap=GPU_SetupMemoryLimit(),live=backend->ctx->setup_start_free>free ? backend->ctx->setup_start_free-free : 0;
	if (needed>free || live>cap || needed>cap-live) throw GPU_SetupUnavailable("insufficient device memory for UPML fusion");
	GPU_UPMLRegion* device_regions=nullptr;uint32_t* output=nullptr;
	struct Cleanup {GPU_UPMLRegion*& regions;uint32_t*& output;~Cleanup() {if (regions) hipFree(regions);if (output) hipFree(output);}} cleanup={device_regions,output};
	auto allocate=[&](void** target,size_t size) {
		const auto status=HIP_AllocateDevice(backend->Stream(),target,size);
		if (status==hipErrorOutOfMemory) {hipGetLastError();throw GPU_SetupUnavailable("insufficient device memory for UPML fusion");}
		HIP_Check(status,"UPML fusion allocation");
	};
	allocate(reinterpret_cast<void**>(&device_regions),bytes);allocate(reinterpret_cast<void**>(&output),8*sizeof(uint32_t));
	uint32_t host[8]={dims[0],dims[1],dims[2],0,0,0,0,0};
	HIP_Check(hipMemcpyAsync(device_regions,regions.data(),bytes,hipMemcpyHostToDevice,backend->Stream()),"UPML fusion region descriptors");
	HIP_Check(hipMemcpyAsync(output,host,sizeof(host),hipMemcpyHostToDevice,backend->Stream()),"UPML fusion reduction initialization");
	const HIP_GridDim size={dims[0],dims[1],dims[2]};
	FusionBounds<<<(uint32_t(cells)+255)/256,256,0,backend->Stream()>>>(device_regions,uint32_t(regions.size()),size,uint32_t(cells),output);
	HIP_Check(hipGetLastError(),"UPML fusion bounds reduction");ValidateFusionBounds<<<1,1,0,backend->Stream()>>>(output);HIP_Check(hipGetLastError(),"UPML fusion topology");
	HIP_Check(hipMemcpyAsync(host,output,sizeof(host),hipMemcpyDeviceToHost,backend->Stream()),"UPML fusion scalar bounds");backend->Flush();
	if (backend->diagnostics) std::cout << "HIP UPML fusion metadata active: d2h_bytes=" << sizeof(host) << std::endl;
	if (host[7]) return false;
	for (int n=0;n<3;++n) {first[n]=host[n];last[n]=host[3+n];}
	return true;
}
