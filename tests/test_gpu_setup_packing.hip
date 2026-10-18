/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "FDTD/hip/hip_internal.h"
#include "FDTD/hip/hip_setup_result.h"
#include "FDTD/gpu_setup_multigrid_plan.h"
#include "openems_global.h"
#include <cstring>
#include <iostream>
bool HIP_SetupUPMLFusion(GPU_Backend_HIP::Impl*,const std::vector<GPU_UPMLRegion>&,const unsigned int*,unsigned int*,unsigned int*);

static void CheckFusionBounds(GPU_Backend_HIP::Impl* backend)
{
	const unsigned int dims[3]={7,9,11};
	const std::vector<GPU_UPMLRegion> shell={{{0,0,0},{2,9,11}},{{5,0,0},{2,9,11}},{{2,0,0},{3,2,11}},{{2,7,0},{3,2,11}},{{2,2,0},{3,5,2}},{{2,2,9},{3,5,2}}};
	auto overlapping=shell;overlapping.push_back(shell.front());
	const std::vector<std::vector<GPU_UPMLRegion>> cases={shell,{shell.front()},overlapping,{{{2,2,2},{1,1,1}}},{{{0,0,0},{7,9,11}}},{{{7,0,0},{1,9,11}}},{},{{{0,0,0},{0,9,11}}}};
	for (const auto& regions:cases)
	{
		bool expected=!regions.empty();std::vector<unsigned int> mask(7*9*11,0);unsigned int low[3]={7,9,11},high[3]={},empty=0;
		for (const auto& region:regions)
		{
			bool valid=true;for (int n=0;n<3;++n) valid&=region.start[n]<=dims[n] && region.size[n]<=dims[n]-region.start[n];
			if (!valid) {expected=false;break;}
			for (unsigned int x=region.start[0];x<region.start[0]+region.size[0];++x) for (unsigned int y=region.start[1];y<region.start[1]+region.size[1];++y) for (unsigned int z=region.start[2];z<region.start[2]+region.size[2];++z)
				if (++mask[(x*9+y)*11+z]>1) expected=false;
		}
		for (unsigned int x=0;x<7;++x) for (unsigned int y=0;y<9;++y) for (unsigned int z=0;z<11;++z) if (!mask[(x*9+y)*11+z])
		{
			++empty;const unsigned int p[3]={x,y,z};for (int n=0;n<3;++n) {low[n]=(std::min)(low[n],p[n]);high[n]=(std::max)(high[n],p[n]+1);}
		}
		expected&=empty && empty==(high[0]-low[0])*(high[1]-low[1])*(high[2]-low[2]);unsigned int first[3]={},last[3]={};
		if (HIP_SetupUPMLFusion(backend,regions,dims,first,last)!=expected) throw std::runtime_error("device UPML fusion topology differs");
		if (expected) for (int n=0;n<3;++n) if (first[n]!=low[n] || last[n]!=high[n]) throw std::runtime_error("device UPML fusion bounds differ");
	}
	std::cout << "PASS device UPML fusion bounds, overlaps, holes and full coverage" << std::endl;
}

static void Compare(const std::shared_ptr<HIP_Context>& context, unsigned int cells, unsigned int distinct, unsigned int mode)
{
	HIP_SetupResult result(context);
	result.dims[0]=cells; result.dims[1]=result.dims[2]=1;
	std::vector<float> input[4];
	for (int k=0;k<4;++k)
	{
		input[k].resize(3ull*cells);
		for (unsigned int node=0;node<cells;++node) for (unsigned int n=0;n<3;++n)
		{
			// Exactly representable values, including signed zero as distinct bits.
			float value=float((node%distinct)*12+3*k+n);
			if (node%distinct==1 && n==0 && k==0) value=-0.0f;
			input[k][n*cells+node]=value;
		}
		HIP_Check(hipMalloc(&result.coefficients[k],input[k].size()*sizeof(float)),"test coefficient allocation");
		HIP_Check(hipMemcpyAsync(result.coefficients[k],input[k].data(),input[k].size()*sizeof(float),hipMemcpyHostToDevice,context->stream),"test coefficient upload");
	}
	HIP_PackSetupCoefficients(result);
	if (result.mode!=mode || result.sets!=distinct) throw std::runtime_error("coefficient palette compression decision changed");
	std::vector<float> table;
	std::vector<unsigned int> index(cells);
	if (mode)
	{
		table.resize(size_t(distinct)*12);
		HIP_Check(hipMemcpy(table.data(),result.palette,table.size()*sizeof(float),hipMemcpyDeviceToHost),"test palette readback");
		if (mode==1)
		{
			std::vector<unsigned short> packed(cells);
			HIP_Check(hipMemcpy(packed.data(),result.index,packed.size()*sizeof(unsigned short),hipMemcpyDeviceToHost),"test 16 bit index readback");
			index.assign(packed.begin(),packed.end());
		}
		else HIP_Check(hipMemcpy(index.data(),result.index,index.size()*sizeof(unsigned int),hipMemcpyDeviceToHost),"test 32 bit index readback");
	}
	for (int k=0;k<4;++k)
	{
		std::vector<float> output(3ull*cells);
		if (mode)
			for (unsigned int node=0;node<cells;++node) for (unsigned int n=0;n<3;++n)
			{
				if (index[node]>=distinct) throw std::runtime_error("coefficient palette index out of bounds");
				output[n*cells+node]=table[12*index[node]+3*k+n];
			}
		else HIP_Check(hipMemcpy(output.data(),result.coefficients[k],output.size()*sizeof(float),hipMemcpyDeviceToHost),"test dense readback");
		if (std::memcmp(input[k].data(),output.data(),output.size()*sizeof(float))) throw std::runtime_error("device-packed coefficient bits changed");
		result.ReadCoefficients(k,0,output.size(),output.data());
		if (std::memcmp(input[k].data(),output.data(),output.size()*sizeof(float))) throw std::runtime_error("requested coefficient materialization bits changed");
		float scalar=0;result.ReadCoefficients(k,cells,1,&scalar);
		if (std::memcmp(&scalar,&input[k][cells],sizeof(float))) throw std::runtime_error("scalar coefficient getter differs");
	}
	result.WriteCoefficient(0,cells,1.25f);input[0][cells]=1.25f;
	for (int k=0;k<4;++k)
	{
		std::vector<float> output(3ull*cells);result.ReadCoefficients(k,0,output.size(),output.data());
		if (std::memcmp(input[k].data(),output.data(),output.size()*sizeof(float))) throw std::runtime_error("coefficient setter changed unrelated values");
	}
	float* adopted=result.coefficients[0];result.engine_adopted=true;
	result.WriteCoefficient(0,cells,-2.5f);
	float scalar=0;HIP_Check(hipMemcpy(&scalar,adopted+cells,sizeof(float),hipMemcpyDeviceToHost),"adopted coefficient snapshot read");
	if (scalar!=1.25f) throw std::runtime_error("operator setter changed an adopted engine snapshot");
	result.ReadCoefficients(0,cells,1,&scalar);
	if (scalar!=-2.5f) throw std::runtime_error("operator setter not visible to getter");
	std::cout << "PASS device palette and setters " << cells << " nodes, " << distinct << " sets, mode " << mode << std::endl;
}

static void CompareUPML(const std::shared_ptr<HIP_Context>& context,unsigned int cells,unsigned int distinct,unsigned int mode)
{
	HIP_SetupUPML result(context);result.dims[0]=cells;result.dims[1]=result.dims[2]=1;
	std::vector<float> input(18ull*cells),output(input.size()),table;
	for (unsigned int k=0;k<6;++k) for (unsigned int n=0;n<3;++n) for (unsigned int node=0;node<cells;++node)
	{
		float value=float((node%distinct)*18+3*k+n);
		if (node%distinct==1 && k==0 && n==0) value=-0.0f;
		input[(3*k+n)*cells+node]=value;
	}
	HIP_Check(hipMalloc(&result.values,input.size()*sizeof(float)),"UPML test input allocation");
	HIP_Check(hipMemcpyAsync(result.values,input.data(),input.size()*sizeof(float),hipMemcpyHostToDevice,context->stream),"UPML test input upload");
	HIP_PackSetupUPML(result);
	if (result.mode!=mode || result.sets!=distinct) throw std::runtime_error("UPML palette decision changed");
	std::vector<unsigned int> index(cells);
	if (mode)
	{
		table.resize(18ull*distinct);
		HIP_Check(hipMemcpy(table.data(),result.palette,table.size()*sizeof(float),hipMemcpyDeviceToHost),"UPML test palette readback");
		if (mode==1)
		{
			std::vector<unsigned short> packed(cells);HIP_Check(hipMemcpy(packed.data(),result.index,cells*sizeof(unsigned short),hipMemcpyDeviceToHost),"UPML test 16-bit readback");
			index.assign(packed.begin(),packed.end());
		}
		else HIP_Check(hipMemcpy(index.data(),result.index,cells*sizeof(unsigned int),hipMemcpyDeviceToHost),"UPML test 32-bit readback");
		for (unsigned int k=0;k<18;++k) for (unsigned int node=0;node<cells;++node) output[k*cells+node]=table[18*index[node]+k];
	}
	else HIP_Check(hipMemcpy(output.data(),result.values,output.size()*sizeof(float),hipMemcpyDeviceToHost),"UPML test dense readback");
	if (std::memcmp(input.data(),output.data(),output.size()*sizeof(float))) throw std::runtime_error("UPML palette changed coefficient bits");
	float* arrays[6];for (int k=0;k<6;++k) arrays[k]=output.data()+3ull*k*cells;
	result.Materialize(arrays,false);
	if (std::memcmp(input.data(),output.data(),output.size()*sizeof(float))) throw std::runtime_error("requested UPML materialization changed bits");
	std::cout << "PASS device UPML palette " << cells << " nodes, " << distinct << " sets, mode " << mode << std::endl;
}

static void CheckDiagnostics(const std::shared_ptr<HIP_Context>& context)
{
	if (!context->setup_diagnostics.enabled) {context->setup_diagnostics.enabled=true;HIP_RegisterDiagnostics(context.get());}
	const auto before=context->setup_diagnostics;
	float* values=nullptr;HIP_Check(HIP_AllocateDevice(context->stream,&values,32*sizeof(float)),"diagnostic test allocation");
	float input[4]={1,2,3,4},output[4]={};
	HIP_Check(hipMemcpyAsync(values,input,sizeof(input),hipMemcpyHostToDevice,context->stream),"diagnostic test upload");
	HIP_Check(hipMemcpyAsync(values+4,values,sizeof(input),hipMemcpyDeviceToDevice,context->stream),"diagnostic test device copy");
	HIP_Check(hipStreamSynchronize(context->stream),"diagnostic test completion");
	HIP_Check(hipMemcpy(output,values+4,sizeof(output),hipMemcpyDeviceToHost),"diagnostic test readback");
	HIP_Check(hipFree(values),"diagnostic test release");
	const auto after=context->setup_diagnostics;
	if (after.h2d-before.h2d!=sizeof(input) || after.d2h-before.d2h!=sizeof(output) || after.d2d-before.d2d!=sizeof(input) ||
	    after.live!=before.live || after.peak<before.live+32*sizeof(float) || std::memcmp(input,output,sizeof(input)))
		throw std::runtime_error("whole-context diagnostics missed allocation, offset transfer or release");
	std::cout << "PASS whole-context allocation peak and synchronous/asynchronous transfer counters" << std::endl;
}

extern "C" OPENEMS_EXPORT int openEMS_TestGPUSetupPacking()
{
	try
	{
		std::unique_ptr<GPU_Backend_HIP> backend(GPU_Backend_HIP::New());
		if (!backend) throw std::runtime_error("required HIP backend unavailable");
		std::shared_ptr<HIP_Context> context=backend->GetImpl()->ctx;
		CheckFusionBounds(backend->GetImpl());
		const double radial[]={0,0.25,1,2,4,7,11,20};
		for (double radius:{-1.0,0.0,0.25,0.5,4.0,19.0,20.0,21.0})
		{
			unsigned int expected=0,actual=0;for (unsigned int i=0;i<8;++i) if (radius<radial[i]) {expected=i;break;}
			if (!GPU_SetupMultiGridSplit(backend.get(),radial,8,radius,actual) || actual!=expected)
				throw std::runtime_error("device multigrid split differs at a boundary or outside the mesh");
		}
		Compare(context,17,1,1);
		Compare(context,200000,65536,1);
		Compare(context,200000,65537,2);
		Compare(context,200000,200000,0);
		CompareUPML(context,17,1,1);CompareUPML(context,200000,65537,2);CompareUPML(context,200000,200000,0);
		CheckDiagnostics(context);
		return 0;
	}
	catch (const std::exception& error) {std::cerr << error.what() << std::endl;return 1;}
}
