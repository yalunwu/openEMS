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
#include <set>
#include <iostream>

#include "hip_internal.h"
#include "FDTD/engine.h"
#include "FDTD/excitation.h"
#include "FDTD/extensions/operator_ext_excitation.h"
#include "FDTD/extensions/engine_ext_excitation.h"

// Soft voltage/current excitation, see Engine_Ext_Excitation.
// One thread per excitation point, or a single thread if points share an edge.
struct ExcitationParam { int numTS; int period; int length; unsigned int count; };

__device__ __forceinline__ void excite(float* field, const size_t* index, const float* amp,
                                       const unsigned int* delay, const float* signal, const ExcitationParam& P, unsigned int n)
{
	int exc_pos = P.numTS - (int)delay[n];
	exc_pos *= (exc_pos>0);
	exc_pos %= P.period;
	exc_pos *= (exc_pos<P.length);
	field[index[n]] = field[index[n]] + amp[n]*signal[exc_pos];
}

__global__ void excitation_apply(float* field, const size_t* index, const float* amp,
                                 const unsigned int* delay, const float* signal, ExcitationParam P)
{
	const unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;
	if (n<P.count)
		excite(field, index, amp, delay, signal, P, n);
}

// same, in the order of the host, for excitation points sharing an edge
__global__ void excitation_apply_serial(float* field, const size_t* index, const float* amp,
                                        const unsigned int* delay, const float* signal, ExcitationParam P)
{
	if (blockIdx.x*blockDim.x + threadIdx.x != 0)
		return;
	for (unsigned int n=0; n<P.count; ++n)
		excite(field, index, amp, delay, signal, P, n);
}

__global__ void excitation_apply_prepared(float* field,const GPU_SetupSourcePoint* points,const float* signal,ExcitationParam P,bool serial)
{
	const unsigned int first=blockIdx.x*blockDim.x+threadIdx.x;
	if (serial && first) return;
	const unsigned int stop=serial ? P.count : first+1;
	for (unsigned int n=first;n<stop && n<P.count;++n)
	{
		const GPU_SetupSourcePoint point=points[n];
		int position=P.numTS-static_cast<int>(point.delay);
		position*=position>0;position%=P.period;position*=position<P.length;
		field[point.index]=field[point.index]+point.amplitude*signal[position];
	}
}

__global__ void source_boundary_delay(const GPU_SetupSourcePoint* points,unsigned int count,HIP_GridDim dims,
                                      unsigned int normal,unsigned int line,unsigned int* delay)
{
	const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i>=count) return;
	const GPU_SetupSourcePoint point=points[i];
	const unsigned int cells=dims.nx*dims.ny*dims.nz,node=point.index%cells;
	const unsigned int p[3]={node/(dims.ny*dims.nz),(node/dims.nz)%dims.ny,node%dims.nz};
	if (point.index/cells!=normal && p[normal]==line) atomicMax(delay,point.delay+1u);
}

HIP_SetupSources::~HIP_SetupSources()
{
	hipStreamSynchronize(context->stream);
	for (auto* buffer:points) if (buffer) hipFree(buffer);
}

void HIP_SetupSources::Materialize(unsigned int current,std::vector<GPU_SetupSourcePoint>& output,bool diagnostics) const
{
	output.resize(count[current]);
	const size_t bytes=output.size()*sizeof(GPU_SetupSourcePoint);
	if (bytes) HIP_Check(hipMemcpyAsync(output.data(),points[current],bytes,hipMemcpyDeviceToHost,context->stream),"requested source tables");
	HIP_Check(hipStreamSynchronize(context->stream),"requested source completion");
	if (diagnostics) std::cout << "GPU requested host sources: d2h_bytes=" << bytes << std::endl;
}

int HIP_SetupSources::BoundaryDelay(unsigned int normal,unsigned int line,const unsigned int dims[3],bool diagnostics) const
{
	if (normal>=3) throw std::out_of_range("source boundary normal");
	if (!count[0]) return -1;
	unsigned int* temporary=nullptr;
	HIP_Check(HIP_AllocateDevice(context->stream,&temporary,sizeof(unsigned int)),"source boundary delay allocation");
	try
	{
		HIP_Check(hipMemsetAsync(temporary,0,sizeof(unsigned int),context->stream),"source boundary delay reset");
		source_boundary_delay<<<(count[0]+255u)/256,256,0,context->stream>>>(points[0],count[0],HIP_GridDim{dims[0],dims[1],dims[2]},normal,line,temporary);
		HIP_Check(hipGetLastError(),"source boundary delay reduction");
		unsigned int delay=0;
		HIP_Check(hipMemcpyAsync(&delay,temporary,sizeof(delay),hipMemcpyDeviceToHost,context->stream),"source boundary delay metadata");
		HIP_Check(hipStreamSynchronize(context->stream),"source boundary delay completion");
		HIP_Check(hipFree(temporary),"source boundary delay release");
		if (diagnostics) std::cout << "GPU source boundary metadata: d2h_bytes=" << sizeof(delay) << std::endl;
		return delay ? static_cast<int>(delay-1) : -1;
	}
	catch (...) {hipStreamSynchronize(context->stream);hipFree(temporary);throw;}
}

class HIP_Ext_Excitation : public GPU_Extension
{
public:
	HIP_Ext_Excitation(GPU_Backend_HIP::Impl* impl, Operator_Ext_Excitation* op_ext, Engine* eng);

	virtual void Apply2Voltages() {Apply(m_Volt, d->volt);}
	virtual void Apply2Current() {Apply(m_Curr, d->curr);}

protected:
	struct Sources
	{
		unsigned int count;
		bool serial;            //!< points share an edge, apply them one after the other
		size_t* index;          //!< flat NIJK index of the excited edge
		float* amp;
		unsigned int* delay;
		float* signal;
		const GPU_SetupSourcePoint* points;
	};

	void Setup(Sources& s, unsigned int count, unsigned int* const index[3], const unsigned short* dir,
	           const FDTD_FLOAT* amp, const unsigned int* delay, const FDTD_FLOAT* signal, bool current);
	void Apply(Sources& s, float* field);

	GPU_Backend_HIP::Impl* d;
	Operator_Ext_Excitation* m_Op_Exc;
	Engine* m_Eng;

	Sources m_Volt;
	Sources m_Curr;
};

HIP_Ext_Excitation::HIP_Ext_Excitation(GPU_Backend_HIP::Impl* impl, Operator_Ext_Excitation* op_ext, Engine* eng)
{
	d = impl;
	m_Op_Exc = op_ext;
	m_Eng = eng;
	if (d->setup_result)
	{
		const auto prepared=d->setup_result->sources.find(op_ext);
		if (prepared!=d->setup_result->sources.end())
		{
			const auto& result=*prepared->second;
			Sources* tables[2]={&m_Volt,&m_Curr};
			for (unsigned int current=0;current<2;++current)
			{
				tables[current]->count=result.count[current];tables[current]->serial=result.serial[current];
				tables[current]->points=result.points[current];tables[current]->signal=d->waveform[current];
			}
			d->device_volt_modified.push_back({reinterpret_cast<const unsigned int*>(result.points[0]),result.count[0],sizeof(GPU_SetupSourcePoint)/sizeof(unsigned int)});
			std::cout << "HIP excitation construction result adopted: " << result.count[0] << " voltage, " << result.count[1] << " current contributions" << std::endl;
			return;
		}
	}
	const bool signal_ready=d->waveform_source==op_ext->m_Exc && d->waveform_length==op_ext->m_Exc->GetLength();
	Setup(m_Volt, op_ext->Volt_Count, op_ext->Volt_index, op_ext->Volt_dir, op_ext->Volt_amp, op_ext->Volt_delay, signal_ready ? nullptr : op_ext->m_Exc->GetVoltageSignal(),false);
	Setup(m_Curr, op_ext->Curr_Count, op_ext->Curr_index, op_ext->Curr_dir, op_ext->Curr_amp, op_ext->Curr_delay, signal_ready ? nullptr : op_ext->m_Exc->GetCurrentSignal(),true);

	// fused step (see update_fused): the currents next to the excited voltages are recomputed;
	// excited currents on the last mesh lines (never updated) would differ between its buffers
	for (unsigned int n=0; n<op_ext->Volt_Count; ++n)
		d->volt_modified.push_back((((size_t)op_ext->Volt_dir[n]*d->dim.nx + op_ext->Volt_index[0][n])*d->dim.ny + op_ext->Volt_index[1][n])*d->dim.nz + op_ext->Volt_index[2][n]);
	for (unsigned int n=0; n<op_ext->Curr_Count; ++n)
		if ((op_ext->Curr_index[0][n]+1>=d->dim.nx) || (op_ext->Curr_index[1][n]+1>=d->dim.ny) || (op_ext->Curr_index[2][n]+1>=d->dim.nz))
			d->fused_blockers++;
}

void HIP_Ext_Excitation::Setup(Sources& s, unsigned int count, unsigned int* const index[3], const unsigned short* dir,
                                const FDTD_FLOAT* amp, const unsigned int* delay, const FDTD_FLOAT* signal, bool current)
{
	s.count = count;
	s.serial = false;
	s.points = nullptr;
	if (count==0)
		return;

	std::vector<size_t> flat(count);
	std::set<size_t> edges;
	for (unsigned int n=0; n<count; ++n)
	{
		flat[n] = (((size_t)dir[n]*d->dim.nx + index[0][n])*d->dim.ny + index[1][n])*d->dim.nz + index[2][n];
		if (!edges.insert(flat[n]).second)
			s.serial = true;
	}
	s.index = d->Alloc<size_t>(count, flat.data());
	s.amp = d->Alloc<float>(count, amp);
	s.delay = d->Alloc<unsigned int>(count, delay);
	if (d->waveform_source==m_Op_Exc->m_Exc && d->waveform_length==m_Op_Exc->m_Exc->GetLength()) s.signal=d->waveform[current ? 1 : 0];
	else s.signal = d->Alloc<float>(m_Op_Exc->m_Exc->GetLength(), signal);
}

void HIP_Ext_Excitation::Apply(Sources& s, float* field)
{
	if (s.count==0)
		return;

	ExcitationParam param;
	param.numTS = m_Eng->GetNumberOfTimesteps();
	param.period = param.numTS+1;
	if (m_Op_Exc->m_Exc->GetSignalPeriod()>0)
		param.period = int(m_Op_Exc->m_Exc->GetSignalPeriod()/m_Op_Exc->m_Exc->GetTimestep());
	param.length = m_Op_Exc->m_Exc->GetLength();
	param.count = s.count;
	if (s.points)
	{
		HIP_Launch(d,"excitation_apply_prepared",excitation_apply_prepared,s.serial ? 1 : s.count,1,1,
		           field,s.points,static_cast<const float*>(s.signal),param,s.serial);
		return;
	}

	if (s.serial)
		HIP_Launch(d, "excitation_apply_serial", excitation_apply_serial, 1, 1, 1,
		            field, (const size_t*)s.index, (const float*)s.amp, (const unsigned int*)s.delay, (const float*)s.signal, param);
	else
		HIP_Launch(d, "excitation_apply", excitation_apply, s.count, 1, 1,
		            field, (const size_t*)s.index, (const float*)s.amp, (const unsigned int*)s.delay, (const float*)s.signal, param);
}

GPU_Extension* HIP_CreateExt_Excitation(GPU_Backend_HIP::Impl* d, Engine_Extension* eng_ext, Engine* eng)
{
	if (!dynamic_cast<Engine_Ext_Excitation*>(eng_ext))
		return NULL;
	Operator_Ext_Excitation* op_ext = dynamic_cast<Operator_Ext_Excitation*>(eng_ext->GetOperatorExtension());
	if (!op_ext)
		return NULL;
	return new HIP_Ext_Excitation(d, op_ext, eng);
}
