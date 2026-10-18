/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "FDTD/hip/hip_internal.h"
#include "FDTD/gpu_setup_expression_export.h"
#include "FDTD/gpu_setup_upml.h"
#include "CSFunctionParser.h"
#include "openems_global.h"
#include <iostream>

namespace
{
__global__ void ExpressionSamples(GPU_SetupExpression d, const GPU_SetupInstruction* code, const double* constants,
                                  const double* inputs, unsigned int count, double* stack, double* values, int* errors)
{
	const unsigned int sample=blockIdx.x*blockDim.x+threadIdx.x;
	if (sample<count)
	{
		int error=0;
		const double value=GPU_SetupEvaluate(d,code,constants,inputs+4*sample,stack+size_t(sample)*d.stack_size,d.stack_size,error);
		values[sample]=value; errors[sample]=error;
	}
}

struct ExpressionBuffers
{
	std::vector<void*> buffers;
	~ExpressionBuffers() {hipDeviceSynchronize();for (void* buffer:buffers) hipFree(buffer);}
	template<class T> T* Allocate(size_t count, const T* data=NULL)
	{
		T* buffer=NULL;
		HIP_Check(hipMalloc(&buffer,std::max(size_t(1),count)*sizeof(T)),"expression test allocation");
		try {buffers.push_back(buffer);} catch (...) {hipFree(buffer);throw;}
		if (data && count) HIP_Check(hipMemcpy(buffer,data,count*sizeof(T),hipMemcpyHostToDevice),"expression test upload");
		return buffer;
	}
};
}

void GPU_TestExpressionSamples(const GPU_SetupExpression& d, const GPU_SetupExpressionBank& program,
                               const std::vector<double>& inputs, std::vector<double>& values, std::vector<int>& errors)
{
	ExpressionBuffers buffers;
	auto* code=buffers.Allocate(program.instructions.size(),program.instructions.data());
	auto* constants=buffers.Allocate(program.constants.size(),program.constants.data());
	auto* samples=buffers.Allocate(inputs.size(),inputs.data());
	auto* stack=buffers.Allocate<double>(values.size()*d.stack_size);
	auto* output=buffers.Allocate<double>(values.size());
	auto* status=buffers.Allocate<int>(errors.size());
	ExpressionSamples<<<static_cast<unsigned int>((values.size()+255)/256),256>>>(d,code,constants,samples,static_cast<unsigned int>(values.size()),stack,output,status);
	HIP_Check(hipGetLastError(),"HIP expression evaluation kernel");
	HIP_Check(hipMemcpy(values.data(),output,values.size()*sizeof(double),hipMemcpyDeviceToHost),"expression test value readback");
	HIP_Check(hipMemcpy(errors.data(),status,errors.size()*sizeof(int),hipMemcpyDeviceToHost),"expression test error readback");
}

__global__ void DefaultGradingSamples(const double* inputs,unsigned int count,double* output)
{
	const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i<count) output[i]=GPU_SetupDefaultGradingValue(inputs[4*i],inputs[4*i+1],inputs[4*i+2],inputs[4*i+3]);
}
void GPU_TestDefaultGrading(const std::vector<double>& inputs,std::vector<double>& output)
{
	ExpressionBuffers buffers;
	auto* data=buffers.Allocate(inputs.size(),inputs.data());
	auto* values=buffers.Allocate<double>(output.size());
	DefaultGradingSamples<<<1,256>>>(data,static_cast<unsigned int>(output.size()),values);
	HIP_Check(hipGetLastError(),"default PML test launch");
	HIP_Check(hipMemcpy(output.data(),values,output.size()*sizeof(double),hipMemcpyDeviceToHost),"default PML test readback");
}

#ifdef OPENEMS_STANDALONE_SETUP_TEST
extern "C" int openEMS_TestGPUSetupExpressionDevice();
#else
extern "C" OPENEMS_EXPORT int openEMS_TestGPUSetupExpressionDevice();
#endif
#define OPENEMS_DEVICE_EXPRESSION_TEST
#define main openEMS_TestGPUSetupExpressionDevice
#include "test_gpu_setup_expression.cpp"
#undef main
