/*
*	Copyright (C) 2026 Sean Mollet (sean@malmoset.com)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "hip/hip_runtime.h"
#include <cmath>
#include <iostream>

#include "hip_internal.h"
#include "FDTD/engine_gpu.h"
#include "FDTD/engine_interface_fdtd.h"
#include "FDTD/extensions/operator_ext_steadystate.h"
#include "FDTD/extensions/engine_ext_steadystate.h"

// Probe records and period comparisons stay on the device. The host receives
// only the difference/validity scalars used by the shared stopping criterion.
struct SSDParam { unsigned int count; unsigned int length; unsigned int rel_pos; };

HIP_SetupSteadyState::~HIP_SetupSteadyState()
{
	hipStreamSynchronize(context->stream);
	if (indices) hipFree(indices);if (records) hipFree(records);
}

__global__ void ssd_record(const float* volt, float* records, const size_t* index, SSDParam P)
{
	const unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;
	if (n<P.count)
		records[n*P.length + P.rel_pos] = volt[index[n]];
}

__global__ void ssd_power(const float* records,unsigned int count,unsigned int period,unsigned int relative,double2* power)
{
	const unsigned int q=blockIdx.x*blockDim.x+threadIdx.x;
	if (q>=count) return;
	const unsigned int first=relative<=period ? 0 : period,old=period-first;
	const float* values=records+size_t(q)*2*period;
	double current=0,difference=0;
	for (unsigned int i=0;i<period;++i)
	{
		const double a=values[first+i],b=values[old+i];
		current+=a*a;difference+=(b-a)*(b-a);
	}
	power[q]=make_double2(current,difference);
}

__global__ void ssd_difference(const double2* power,unsigned int count,double* result)
{
	double maximum=0,difference=0,valid=0;
	for (unsigned int i=0;i<count;++i)
	{
		if (!isfinite(power[i].x) || !isfinite(power[i].y)) {result[0]=0;result[1]=-1;return;}
		maximum=fmax(maximum,power[i].x);
	}
	for (unsigned int i=0;i<count;++i) if (power[i].x>maximum*1e-2)
	{
		difference=fmax(difference,power[i].y/power[i].x);valid=1;
	}
	result[0]=difference;result[1]=valid;
}


__global__ void widen_steady_indices(const unsigned int* input,size_t* output,unsigned int count)
{
	const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i<count) output[i]=input[i];
}

class HIP_Ext_SteadyState : public GPU_Extension
{
public:
	HIP_Ext_SteadyState(GPU_Backend_HIP::Impl* impl, Operator_Ext_SteadyState* op_ext, Engine_Ext_SteadyState* eng_ext, Engine_GPU* eng);

	virtual void Apply2Voltages();

protected:
	void CheckPeriod(unsigned int rel_pos);

	GPU_Backend_HIP::Impl* d;
	Engine_Ext_SteadyState* m_Eng_SSD;
	Engine_GPU* m_Eng;
	unsigned int m_Period;       //!< timesteps per period
	unsigned int m_Count;        //!< number of probes
	size_t* m_Index;             //!< flat NIJK index of the probed voltages
	float* m_Records;            //!< [probe][2*period]: the two periods CheckPeriod() compares
	double2* m_Power;
	double* m_Difference;
};

HIP_Ext_SteadyState::HIP_Ext_SteadyState(GPU_Backend_HIP::Impl* impl, Operator_Ext_SteadyState* op_ext, Engine_Ext_SteadyState* eng_ext, Engine_GPU* eng)
{
	d = impl;
	m_Eng_SSD = eng_ext;
	m_Eng = eng;
	m_Period = op_ext->m_TS_period;
	m_Count = op_ext->m_E_probe_dir.size();
	m_Power=d->Alloc<double2>(std::max(m_Count,1u));m_Difference=d->Alloc<double>(2);
	if (d->setup_result)
	{
		const auto prepared=d->setup_result->steady_state.find(op_ext);
		if (prepared!=d->setup_result->steady_state.end())
		{
			m_Index=d->Alloc<size_t>(m_Count);m_Records=prepared->second->records;
			HIP_Launch(d,"widen_steady_indices",widen_steady_indices,m_Count,1,1,static_cast<const unsigned int*>(prepared->second->indices),m_Index,m_Count);
			std::cout << "HIP steady-state construction result adopted" << std::endl;return;
		}
	}

	std::vector<size_t> index(std::max(m_Count, 1u), 0);
	for (unsigned int n=0; n<m_Count; ++n)
		index[n] = (((size_t)op_ext->m_E_probe_dir.at(n)*d->dim.nx + op_ext->m_E_probe_pos[0].at(n))*d->dim.ny
		            + op_ext->m_E_probe_pos[1].at(n))*d->dim.nz + op_ext->m_E_probe_pos[2].at(n);
	m_Index = d->Alloc<size_t>(index.size(), index.data());
	m_Records = d->Alloc<float>(std::max(m_Count,1u)*2*size_t(m_Period));
}

void HIP_Ext_SteadyState::Apply2Voltages()
{
	const unsigned int p = m_Period;
	const unsigned int TS = m_Eng->GetNumberOfTimesteps();
	const unsigned int rel_pos = TS%(2*p);

	if (m_Count>0)
	{
		SSDParam param = {m_Count, 2*p, rel_pos};
		HIP_Launch(d, "ssd_record", ssd_record, m_Count, 1, 1, (const float*)d->volt, m_Records, (const size_t*)m_Index, param);
	}

	if ((TS%p==0) && (TS>=2*p))
		CheckPeriod(rel_pos);
}

// Reduce the recorded periods on the device; read only the comparison scalars.
void HIP_Ext_SteadyState::CheckPeriod(unsigned int rel_pos)
{
	const unsigned int p = m_Period;

	HIP_Launch(d,"ssd_power",ssd_power,m_Count,1,1,static_cast<const float*>(m_Records),m_Count,p,rel_pos,m_Power);
	ssd_difference<<<1,1,0,d->Stream()>>>(m_Power,m_Count,m_Difference);
	HIP_Check(hipGetLastError(),"steady-state period reduction");
	double difference[2];
	HIP_Check(hipMemcpyAsync(difference,m_Difference,sizeof(difference),hipMemcpyDeviceToHost,d->Stream()),"steady-state result metadata");d->Flush();
	if (difference[1]<0) throw std::runtime_error("non-finite steady-state records");
	m_Eng_SSD->SetPeriodDifference(difference[0],difference[1]!=0);

}

GPU_Extension* HIP_CreateExt_SteadyState(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng)
{
	Engine_Ext_SteadyState* ssd_ext = dynamic_cast<Engine_Ext_SteadyState*>(eng_ext);
	if (!ssd_ext)
		return NULL;
	Operator_Ext_SteadyState* op_ext = dynamic_cast<Operator_Ext_SteadyState*>(eng_ext->GetOperatorExtension());
	if (!op_ext)
		return NULL;
	Engine_GPU* gpu_eng = dynamic_cast<Engine_GPU*>(eng);
	if (!gpu_eng)
		return NULL;
	return new HIP_Ext_SteadyState(d, op_ext, ssd_ext, gpu_eng);
}
