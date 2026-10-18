/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_internal.h"
#include "hip_setup_result.h"
#include "FDTD/excitation.h"
#include "FDTD/gpu_setup_waveform_export.h"
#include "FDTD/gpu_setup_memory.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <cstdlib>

namespace
{
struct WaveformWorkspace
{
	std::shared_ptr<HIP_Context> context;
	std::vector<void*> buffers;
	size_t bytes=0;
	explicit WaveformWorkspace(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	~WaveformWorkspace() {hipStreamSynchronize(context->stream);for (void* buffer:buffers) hipFree(buffer);}
	template<class T> T* Allocate(size_t count, const T* data=nullptr)
	{
		const size_t size=GPU_SetupMultiply(std::max(size_t(1),count),sizeof(T));
		T* buffer=nullptr;
		const hipError_t status=HIP_AllocateDevice(context->stream,&buffer,size);
		if (status==hipErrorOutOfMemory) {hipGetLastError();throw GPU_SetupUnavailable("insufficient device memory for waveform construction");}
		HIP_Check(status,"waveform workspace allocation");
		try {buffers.push_back(buffer);} catch (...) {hipFree(buffer);throw;}
		bytes=GPU_SetupAdd(bytes,size);
		if (data && count) HIP_Check(hipMemcpyAsync(buffer,data,count*sizeof(T),hipMemcpyHostToDevice,context->stream),"waveform input upload");
		else HIP_Check(hipMemsetAsync(buffer,0,size,context->stream),"waveform workspace clear");
		return buffer;
	}
};

__global__ void BuildWaveform(GPU_SetupWaveform d, GPU_SetupExpressionView expressions, GPU_SetupExpression custom,
                              float* voltage, float* current, unsigned int* status, unsigned int* peaks)
{
	__shared__ float amplitudes[256];
	__shared__ unsigned int indices[256];
	const unsigned int lane=threadIdx.x,index=blockIdx.x*blockDim.x+lane;
	float amplitude=-1;
	if (index<d.length)
	{
		int error=0;
		expressions.stack+=size_t(index)*expressions.capacity; expressions.error=&error;
		voltage[index]=GPU_SetupWaveformSample(d,index,false,expressions,custom);
		if (error) atomicOr(status,1u);
		current[index]=GPU_SetupWaveformSample(d,index,true,expressions,custom);
		if (error || !isfinite(voltage[index]) || !isfinite(current[index])) atomicOr(status,1u);
		amplitude=fabsf(voltage[index]);
	}
	amplitudes[lane]=amplitude; indices[lane]=index; __syncthreads();
	for (unsigned int stride=128;stride;stride/=2)
	{
		if (lane<stride && (amplitudes[lane+stride]>amplitudes[lane] ||
		    (amplitudes[lane+stride]==amplitudes[lane] && indices[lane+stride]<indices[lane])))
		{amplitudes[lane]=amplitudes[lane+stride];indices[lane]=indices[lane+stride];}
		__syncthreads();
	}
	if (!lane) peaks[blockIdx.x]=indices[0];
}

__global__ void ReduceWaveformPeaks(const float* signal, const unsigned int* input, unsigned int count, unsigned int* output)
{
	__shared__ float amplitudes[256];
	__shared__ unsigned int indices[256];
	const unsigned int lane=threadIdx.x,index=blockIdx.x*blockDim.x+lane;
	indices[lane]=index<count ? input[index] : UINT32_MAX;
	amplitudes[lane]=index<count ? fabsf(signal[indices[lane]]) : -1; __syncthreads();
	for (unsigned int stride=128;stride;stride/=2)
	{
		if (lane<stride && (amplitudes[lane+stride]>amplitudes[lane] ||
		    (amplitudes[lane+stride]==amplitudes[lane] && indices[lane+stride]<indices[lane])))
		{amplitudes[lane]=amplitudes[lane+stride];indices[lane]=indices[lane+stride];}
		__syncthreads();
	}
	if (!lane) output[blockIdx.x]=indices[0];
}
}

class HIP_SetupWaveform
{
public:
	static bool Build(HIP_SetupResult& result, Excitation& excitation, unsigned int maxTS, bool diagnostics)
	{
		if (result.waveform[0] || result.waveform[1]) throw std::logic_error("device waveform already constructed");
		const GPU_SetupWaveform d=GPU_DescribeSetupWaveform(excitation,maxTS);

		GPU_SetupExpression custom={};
		const unsigned int groups=(d.length+255u)/256;
		size_t estimate=GPU_SetupMultiply(d.length,2*sizeof(float));
		estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(groups,2*sizeof(unsigned int)));
		const size_t cells=size_t(result.dims[0])*result.dims[1]*result.dims[2];
		const size_t future=result.context->setup_future_bytes ? result.context->setup_future_bytes : GPU_SetupMultiply(cells,12*sizeof(float));
		estimate=GPU_SetupAdd(estimate,GPU_SetupAdd(future,64ull*1024*1024));
		size_t available=0,total=0; HIP_Check(hipMemGetInfo(&available,&total),"waveform memory budget");
		if (const char* limit=std::getenv("OPENEMS_GPU_SETUP_MEMORY_LIMIT"))
		{
			const unsigned long long cap=std::strtoull(limit,nullptr,10);
			const size_t live=result.context->setup_start_free>available ? result.context->setup_start_free-available : 0;
			available=static_cast<size_t>(std::min<unsigned long long>(available,cap>live ? cap-live : 0));
		}
		if (estimate>available-available/5) throw GPU_SetupUnavailable("insufficient device memory budget for waveform construction");
		WaveformWorkspace workspace(result.context);
		for (int kind=0;kind<2;++kind)
		{
			float* signal=nullptr;
			const hipError_t status=HIP_AllocateDevice(result.context->stream,&signal,GPU_SetupMultiply(d.length,sizeof(float)));
			if (status==hipErrorOutOfMemory) {hipGetLastError();throw GPU_SetupUnavailable("insufficient device memory for waveform output");}
			HIP_Check(status,"waveform output allocation");
			const auto context=result.context;
			result.waveform_owner[kind]=std::shared_ptr<float>(signal,[context](float* values) {
				hipStreamSynchronize(context->stream); hipFree(values);
			});
			result.waveform[kind]=signal;
		}



		auto* status=workspace.Allocate<unsigned int>(1);
		unsigned int* peaks=workspace.Allocate<unsigned int>(groups);
		unsigned int* temporary=workspace.Allocate<unsigned int>(groups);
		GPU_SetupExpressionView expressions={};
		BuildWaveform<<<groups,256,0,result.context->stream>>>(d,expressions,custom,result.waveform[0],result.waveform[1],status,peaks);
		HIP_Check(hipGetLastError(),"waveform construction kernel");
		for (unsigned int count=groups;count>1;)
		{
			const unsigned int next=(count+255u)/256;
			ReduceWaveformPeaks<<<next,256,0,result.context->stream>>>(result.waveform[0],peaks,count,temporary);
			HIP_Check(hipGetLastError(),"waveform peak reduction");
			std::swap(peaks,temporary); count=next;
		}
		unsigned int error=0,peak=0;
		HIP_Check(hipMemcpyAsync(&error,status,sizeof(error),hipMemcpyDeviceToHost,result.context->stream),"waveform status readback");
		HIP_Check(hipMemcpyAsync(&peak,peaks,sizeof(peak),hipMemcpyDeviceToHost,result.context->stream),"waveform peak readback");
		HIP_Check(hipStreamSynchronize(result.context->stream),"waveform completion");
		if (error) throw std::runtime_error("HIP waveform construction: invalid numerical expression result");
		const auto context=result.context;
		const auto voltage=result.waveform_owner[0],current=result.waveform_owner[1];
		excitation.m_MaterializeSignal=[context,voltage,current,diagnostics](bool magnetic,float* host,unsigned int count) {
			const size_t bytes=GPU_SetupMultiply(count,sizeof(float));
			HIP_Check(hipMemcpyAsync(host,magnetic ? current.get() : voltage.get(),bytes,hipMemcpyDeviceToHost,context->stream),"requested waveform materialization");
			HIP_Check(hipStreamSynchronize(context->stream),"waveform materialization completion");
			if (diagnostics) std::cout << "GPU requested host waveform: current=" << magnetic << " d2h_bytes=" << bytes << std::endl;
		};
		delete[] excitation.m_signal_volt; delete[] excitation.m_signal_curr;
		excitation.m_signal_volt=nullptr; excitation.m_signal_curr=nullptr;
		excitation.m_length=d.length; excitation.m_nyquistTS=d.nyquist;
		excitation.m_f_max=d.maximum_frequency; excitation.m_foi=d.frequency_of_interest;
		excitation.m_HasDevicePeak=true; excitation.m_DevicePeak=peak;
		result.waveform_source=&excitation; result.waveform_length=d.length;
		context->waveform_source=&excitation;context->waveform_length=d.length;
		for (unsigned int k=0;k<2;++k) context->waveform[k]=result.waveform_owner[k];
		std::cout << "HIP waveform construction active: " << d.length << " samples" << std::endl;
		if (diagnostics) std::cout << "GPU waveform diagnostics: h2d_bytes=" << 0
		                           << " d2h_bytes=" << 2*sizeof(unsigned int)
		                           << " workspace_bytes=" << workspace.bytes << " output_bytes=" << 2ull*d.length*sizeof(float) << std::endl;
		return true;
	}
};

bool HIP_BuildSetupWaveform(HIP_SetupResult& result, Excitation& excitation, unsigned int maxTS, bool diagnostics)
{
	try {return HIP_SetupWaveform::Build(result,excitation,maxTS,diagnostics);}
	catch (const std::bad_alloc&) {throw GPU_SetupUnavailable("insufficient host memory for waveform construction metadata");}
}
