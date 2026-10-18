/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "FDTD/hip/hip_internal.h"
#include "FDTD/hip/setup_candidates_hip.h"
#include "FDTD/hip/setup_scan_hip.h"
#include "openems_global.h"
#include <iostream>
#include <numeric>

namespace
{
struct Buffers
{
	std::vector<void*> allocations;
	~Buffers() {for (void* p:allocations) hipFree(p);}
	template<class T> T* Allocate(size_t count, const T* input=nullptr)
	{
		T* p=nullptr;HIP_Check(hipMalloc(&p,std::max(size_t(1),count)*sizeof(T)),"test allocation");
		allocations.push_back(p);
		if (input && count)
		{
			HIP_Check(hipMemcpy(p,input,count*sizeof(T),hipMemcpyHostToDevice),"test upload");
			HIP_Check(hipDeviceSynchronize(),"test input completion");
		}
		return p;
	}
};

void CompareScan(const std::shared_ptr<HIP_Context>& context, uint32_t count)
{
	Buffers buffers;
	std::vector<uint64_t> input(count),expected(count),actual(count);
	uint64_t sum=0;
	for (uint32_t i=0;i<count;++i) {input[i]=i%7 ? i%13 : UINT32_MAX;expected[i]=sum;sum+=input[i];}
	uint64_t* values=buffers.Allocate(input.size(),input.data());
	uint64_t* workspace=buffers.Allocate<uint64_t>(HIP_SetupScanWorkspace(count));
	HIP_SetupExclusiveScan(values,count,workspace,context->stream);
	HIP_Check(hipStreamSynchronize(context->stream),"test scan completion");
	HIP_Check(hipMemcpy(actual.data(),values,actual.size()*sizeof(uint64_t),hipMemcpyDeviceToHost),"test scan readback");
	for (uint32_t i=0;i<count;++i) if (actual[i]!=expected[i])
		throw std::runtime_error("device exclusive scan differs: count="+std::to_string(count)+" index="+std::to_string(i)+
		                         " actual="+std::to_string(actual[i])+" expected="+std::to_string(expected[i]));
}

void CompareCandidates(const std::shared_ptr<HIP_Context>& context)
{
	Buffers buffers;
	HIP_SetupCandidateInput input={};input.dims[0]=33;input.dims[1]=19;input.dims[2]=17;
	std::vector<double> grid;
	for (int n=0;n<3;++n)
	{
		input.grid_offsets[n]=static_cast<uint32_t>(grid.size());
		for (uint32_t p=0;p<input.dims[n];++p)
			for (int k=0;k<6;++k) grid.push_back(-2.0+p*p*0.03125);
	}
	std::vector<GPU_SetupPrimitive> shapes(21);
	for (size_t i=1;i<shapes.size();++i)
	{
		shapes[i].candidate_bounds=i%5!=0;
		for (int n=0;n<3;++n) {shapes[i].bounds[2*n]=-3+i*0.25+n;shapes[i].bounds[2*n+1]=i*0.75+n;}
		if (i%3==0) std::swap(shapes[i].bounds[0],shapes[i].bounds[1]);
	}
	std::vector<uint32_t> order={9,3,18,7,13,5,1,19,2,8,4,6,20,11,10,16,12,15,14,17,20,10,5,1,3,8};
	input.order_offsets[0]=0;input.order_offsets[1]=7;input.order_offsets[2]=20;input.order_offsets[3]=26;
	const size_t rows=HIP_SetupCandidateRows(input.dims),xy=input.dims[0]*input.dims[1];
	std::vector<uint32_t> expected(rows);
	for (size_t row=0;row<rows;++row)
	{
		expected[row]=static_cast<uint32_t>(expected.size());
		uint32_t which=0;size_t offset=row;int p[3]={-1,-1,-1};
		if (row<2*(xy+1))
		{
			which=static_cast<uint32_t>(row/(xy+1));offset=row%(xy+1);if (offset==xy) continue;
			p[0]=static_cast<int>(offset/input.dims[1]);p[1]=static_cast<int>(offset%input.dims[1]);
		}
		else {which=2;offset=row-2*(xy+1);if (offset==size_t(input.dims[1])*input.dims[2]) continue;p[2]=static_cast<int>(offset/input.dims[1]);p[1]=static_cast<int>(offset%input.dims[1]);}
		for (uint32_t i=input.order_offsets[which];i<input.order_offsets[which+1];++i)
		{
			bool keep=true;const auto& shape=shapes[order[i]];
			for (int n=0;n<3 && shape.candidate_bounds;++n)
			{
				const int lo=p[n]<0 ? 0 : std::max(0,p[n]-1),hi=p[n]<0 ? input.dims[n]-1 : std::min(int(input.dims[n])-1,p[n]+1);
				const double lower=grid[input.grid_offsets[n]+6*lo],upper=grid[input.grid_offsets[n]+6*hi];
				if (upper<std::min(shape.bounds[2*n],shape.bounds[2*n+1]) || lower>std::max(shape.bounds[2*n],shape.bounds[2*n+1])) keep=false;
			}
			if (keep) expected.push_back(order[i]);
		}
	}
	input.grid=buffers.Allocate(grid.size(),grid.data());input.shapes=buffers.Allocate(shapes.size(),shapes.data());input.order=buffers.Allocate(order.size(),order.data());
	uint64_t* counts=buffers.Allocate<uint64_t>(rows+1);
	uint64_t* scan=buffers.Allocate<uint64_t>(HIP_SetupScanWorkspace(rows+1));
	HIP_CountSetupCandidates(input,counts,context->stream);HIP_SetupExclusiveScan(counts,rows+1,scan,context->stream);
	HIP_Check(hipStreamSynchronize(context->stream),"test candidate count completion");
	uint64_t count=0;HIP_Check(hipMemcpy(&count,counts+rows,sizeof(count),hipMemcpyDeviceToHost),"test candidate count");
	if (count!=expected.size()-rows) throw std::runtime_error("device candidate count differs");
	uint32_t* candidates=buffers.Allocate<uint32_t>(expected.size());
	HIP_FillSetupCandidates(input,counts,candidates,context->stream);
	HIP_Check(hipStreamSynchronize(context->stream),"test candidate completion");
	std::vector<uint32_t> actual(expected.size());
	HIP_Check(hipMemcpy(actual.data(),candidates,actual.size()*sizeof(uint32_t),hipMemcpyDeviceToHost),"test candidate readback");
	if (actual!=expected) throw std::runtime_error("device candidate order/bounds differs");
}
}

extern "C" OPENEMS_EXPORT int openEMS_TestGPUSetupCandidates()
{
	try
	{
		std::unique_ptr<GPU_Backend_HIP> backend(GPU_Backend_HIP::New());
		if (!backend) throw std::runtime_error("required HIP backend unavailable");
		const auto context=backend->GetImpl()->ctx;
		for (uint32_t count:{1u,255u,256u,257u,65537u,600001u}) CompareScan(context,count);
		CompareCandidates(context);
		std::cout << "PASS 64-bit hierarchical scan and stable candidate count/scan/fill" << std::endl;
		return 0;
	}
	catch (const std::exception& error) {std::cerr << error.what() << std::endl;return 1;}
}
