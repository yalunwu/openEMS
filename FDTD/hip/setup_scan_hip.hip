/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_internal.h"
#include "setup_scan_hip.h"
#include "FDTD/gpu_setup_memory.h"

namespace
{
__global__ void ScanBlocks(uint64_t* values, uint32_t count, uint64_t* sums)
{
	__shared__ uint64_t prefix[256];
	const uint32_t lane=threadIdx.x,index=blockIdx.x*blockDim.x+lane;
	const uint64_t value=index<count ? values[index] : 0;
	prefix[lane]=value; __syncthreads();
	for (uint32_t stride=1;stride<256;stride*=2)
	{
		const uint64_t add=lane>=stride ? prefix[lane-stride] : 0;
		__syncthreads(); prefix[lane]+=add; __syncthreads();
	}
	if (index<count) values[index]=prefix[lane]-value;
	if (lane==255) sums[blockIdx.x]=prefix[lane];
}

__global__ void AddBlocks(uint64_t* values, uint32_t count, const uint64_t* offsets)
{
	const uint32_t index=blockIdx.x*blockDim.x+threadIdx.x;
	if (index<count) values[index]+=offsets[blockIdx.x];
}
}

size_t HIP_SetupScanWorkspace(size_t count)
{
	size_t size=0;
	do {count=(count+255)/256;size=GPU_SetupAdd(size,count);} while (count>1);
	return size;
}

void HIP_SetupExclusiveScan(uint64_t* values, size_t count, uint64_t* workspace, hipStream_t stream)
{
	if (!count) return;
	if (count>UINT32_MAX-255u) throw GPU_SetupUnavailable("scan dispatch index range exceeded");
	const uint32_t groups=static_cast<uint32_t>((count+255)/256);
	ScanBlocks<<<groups,256,0,stream>>>(values,static_cast<uint32_t>(count),workspace);
	HIP_Check(hipGetLastError(),"construction block scan");
	if (groups>1)
	{
		HIP_SetupExclusiveScan(workspace,groups,workspace+groups,stream);
		AddBlocks<<<groups,256,0,stream>>>(values,static_cast<uint32_t>(count),workspace);
		HIP_Check(hipGetLastError(),"construction scan offsets");
	}
}
