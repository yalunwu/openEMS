/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_internal.h"
#include "FDTD/gpu_setup_memory.h"
#include <cmath>
#include <iostream>

__global__ static void FindSplit(const double* lines,uint32_t count,double radius,uint32_t* output)
{
	uint32_t lo=0,hi=count;
	while (lo<hi) {const uint32_t mid=lo+(hi-lo)/2;if (radius<lines[mid]) hi=mid;else lo=mid+1;}
	*output=lo<count ? lo : 0;
}

bool HIP_SetupMultiGridSplit(GPU_Backend* backend,const double* lines,unsigned int count,double radius,unsigned int& split)
{
	auto* hip=dynamic_cast<GPU_Backend_HIP*>(backend);if (!hip) return false;
	if (!lines || !count || !std::isfinite(radius)) throw std::invalid_argument("invalid multigrid split descriptor");
	auto* impl=hip->GetImpl();const auto context=impl->ctx;
	const size_t bytes=GPU_SetupMultiply(count,sizeof(double)),needed=GPU_SetupAdd(bytes,sizeof(uint32_t));
	size_t available=0,total=0;HIP_Check(hipMemGetInfo(&available,&total),"multigrid planning memory query");
	if (!context->setup_start_free) context->setup_start_free=available;
	const size_t cap=GPU_SetupMemoryLimit(),live=context->setup_start_free>available ? context->setup_start_free-available : 0;
	if (needed>available || live>cap || needed>cap-live) throw GPU_SetupUnavailable("insufficient device memory for multigrid planning");
	double* device_lines=nullptr;uint32_t* output=nullptr;
	struct Cleanup {double*& lines;uint32_t*& output;~Cleanup() {if (lines) hipFree(lines);if (output) hipFree(output);}} cleanup={device_lines,output};
	auto allocate=[&](void** target,size_t size) {
		const auto status=HIP_AllocateDevice(context->stream,target,size);
		if (status==hipErrorOutOfMemory) {hipGetLastError();throw GPU_SetupUnavailable("insufficient device memory for multigrid planning");}
		HIP_Check(status,"multigrid planning allocation");
	};
	allocate(reinterpret_cast<void**>(&device_lines),bytes);allocate(reinterpret_cast<void**>(&output),sizeof(uint32_t));
	HIP_Check(hipMemcpyAsync(device_lines,lines,bytes,hipMemcpyHostToDevice,context->stream),"multigrid raw radial mesh upload");
	FindSplit<<<1,1,0,context->stream>>>(device_lines,count,radius,output);HIP_Check(hipGetLastError(),"multigrid split selection");
	HIP_Check(hipMemcpyAsync(&split,output,sizeof(split),hipMemcpyDeviceToHost,context->stream),"multigrid split scalar readback");impl->Flush();
	std::cout << "HIP multigrid split planning active: index=" << split << std::endl;
	return true;
}
