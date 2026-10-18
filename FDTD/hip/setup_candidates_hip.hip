/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_internal.h"
#include "setup_candidates_hip.h"
#include "FDTD/gpu_setup_candidates.h"
#include "FDTD/gpu_setup_memory.h"

namespace
{
__device__ bool CandidateRow(HIP_SetupCandidateInput s, uint32_t row, uint32_t& which, double bounds[6])
{
	const uint32_t xy=s.dims[0]*s.dims[1];
	int p[3]={-1,-1,-1};
	if (row<2*(xy+1))
	{
		which=row/(xy+1); row%=xy+1;
		if (row==xy) return false;
		p[0]=row/s.dims[1];p[1]=row%s.dims[1];
	}
	else
	{
		which=2;row-=2*(xy+1);
		if (row==s.dims[1]*s.dims[2]) return false;
		p[2]=row/s.dims[1];p[1]=row%s.dims[1];
	}
	for (int n=0;n<3;++n)
	{
		const uint32_t lo=p[n]<0 ? 0 : max(0,p[n]-1);
		const uint32_t hi=p[n]<0 ? s.dims[n]-1 : min(int(s.dims[n])-1,p[n]+1);
		bounds[2*n]=s.grid[s.grid_offsets[n]+6*lo];
		bounds[2*n+1]=s.grid[s.grid_offsets[n]+6*hi];
	}
	return true;
}

__global__ void Count(HIP_SetupCandidateInput s, uint32_t rows, uint64_t* counts)
{
	const uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;
	if (row>rows) return;
	uint64_t count=0;
	uint32_t which=0;double bounds[6];
	if (row<rows && CandidateRow(s,row,which,bounds))
		for (uint32_t i=s.order_offsets[which];i<s.order_offsets[which+1];++i)
			count+=GPU_SetupCandidate(s.shapes[s.order[i]],bounds) ? 1 : 0;
	counts[row]=count;
}

__global__ void Fill(HIP_SetupCandidateInput s, uint32_t rows, const uint64_t* offsets, uint32_t* output)
{
	const uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;
	if (row>=rows) return;
	uint32_t offset=rows+static_cast<uint32_t>(offsets[row]);
	output[row]=offset;
	uint32_t which=0;double bounds[6];
	if (CandidateRow(s,row,which,bounds))
		for (uint32_t i=s.order_offsets[which];i<s.order_offsets[which+1];++i)
			if (GPU_SetupCandidate(s.shapes[s.order[i]],bounds)) output[offset++]=s.order[i];
}
}

size_t HIP_SetupCandidateRows(const uint32_t dims[3])
{
	return GPU_SetupAdd(GPU_SetupMultiply(2,GPU_SetupAdd(GPU_SetupMultiply(dims[0],dims[1]),1)),
	                    GPU_SetupAdd(GPU_SetupMultiply(dims[1],dims[2]),1));
}

void HIP_CountSetupCandidates(HIP_SetupCandidateInput input, uint64_t* counts, hipStream_t stream)
{
	const size_t rows=HIP_SetupCandidateRows(input.dims);
	if (rows>UINT32_MAX-256u) throw GPU_SetupUnavailable("candidate row dispatch range exceeded");
	Count<<<static_cast<uint32_t>((rows+256)/256),256,0,stream>>>(input,static_cast<uint32_t>(rows),counts);
	HIP_Check(hipGetLastError(),"candidate row counting");
}

void HIP_FillSetupCandidates(HIP_SetupCandidateInput input, const uint64_t* offsets, uint32_t* candidates, hipStream_t stream)
{
	const uint32_t rows=static_cast<uint32_t>(HIP_SetupCandidateRows(input.dims));
	Fill<<<(rows+255u)/256,256,0,stream>>>(input,rows,offsets,candidates);
	HIP_Check(hipGetLastError(),"candidate row filling");
}
