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

#include <stdexcept>
#include <vector>

#include "engine_interface_gpu_fdtd.h"
#include "engine_field_gather.h"
#include "engine_gpu.h"

Engine_Interface_GPU_FDTD::Engine_Interface_GPU_FDTD(Operator* op) : Engine_Interface_FDTD(op)
{
	m_Eng_GPU = dynamic_cast<Engine_GPU*>(m_Eng);
	if (m_Eng_GPU==NULL)
		throw std::runtime_error("Engine_Interface_GPU_FDTD::Engine_Interface_GPU_FDTD: Error: GPU-Engine is not set!");
}

Engine_Interface_GPU_FDTD::~Engine_Interface_GPU_FDTD()
{
	m_Eng_GPU = NULL;
}

double Engine_Interface_GPU_FDTD::CalcFastEnergy() const
{
	// the end criteria asks for this every Nyquist period, so it is worth the
	// device implementation; without one the host mirror has to be brought up to
	// date, it is not written at the end of a batch
	unsigned int numNodes[3];
	for (int n=0; n<3; ++n)
		numNodes[n] = m_Op->GetNumberOfLines(n)-1;
	double E_energy=0.0, H_energy=0.0;
	if (m_Eng_GPU->CalcFastEnergy(numNodes, E_energy, H_energy))
		return EPS0*E_energy + MUE0*H_energy;
	m_Eng_GPU->UpdateHostMirror();
	return Engine_Interface_FDTD::CalcFastEnergy();
}

void Engine_Interface_GPU_FDTD::PrepareFieldAccess()
{
	// several threads read the fields next; reading them from the device value by value is not thread-safe
	m_Eng_GPU->UpdateHostMirror();
}

bool Engine_Interface_GPU_FDTD::TakeFieldSnapshot(const float* &volt, const float* &curr)
{
	return m_Eng_GPU->SnapshotFields(volt, curr);
}

void Engine_Interface_GPU_FDTD::WaitFieldSnapshot() const
{
	m_Eng_GPU->WaitSnapshot();
}

bool Engine_Interface_GPU_FDTD::PrepareSnapshotGather(Engine_Field_Gather* gather)
{
	if (!m_Eng_GPU->CanSnapshotGather())
		return true;   // plain field snapshots, if the engine has any: evaluated on the host
	// the engine only snapshots the raw values the dump reads, so it has to know them beforehand
	if (auto* prepared=dynamic_cast<Prepared_Field_Gather_FDTD*>(gather))
	{
		size_t offset=0;
		if (!m_Eng_GPU->AddPreparedSnapshotGather(prepared->HField(),prepared->prepared,offset)) return false;
		prepared->SetSnapshotEvaluated(offset);return true;
	}
	Field_Gather_FDTD* g = dynamic_cast<Field_Gather_FDTD*>(gather);
	size_t offset = 0;
	if (!g || !m_Eng_GPU->AddSnapshotGather(g->HField(), g->region, offset))
		return false;
	g->SetSnapshotRegion(offset);
	return true;
}

Engine_Field_Gather* Engine_Interface_GPU_FDTD::CreateFieldGather(bool h_field, const unsigned int numLines[3], unsigned int* const posLines[3],const std::shared_ptr<Engine_SamplingMesh>& mesh) const
{
	if (auto table=m_Eng_GPU->GetBackend()->BuildFieldGather(m_Op,m_InterpolType,h_field,numLines,posLines,mesh))
	{
		auto* gather=new Prepared_Field_Gather_FDTD(m_Eng_GPU->FieldArray(false),m_Eng_GPU->FieldArray(true),h_field,numLines[1],numLines[2],!m_Eng_GPU->HostDumpsOnly());
		gather->prepared=table;return gather;
	}
	return new Field_Gather_FDTD(this, m_Op, m_Eng_GPU->FieldArray(false), m_Eng_GPU->FieldArray(true), h_field, numLines, posLines);
}

int Engine_Interface_GPU_FDTD::CreateFieldDFT(const Engine_Field_Gather* gather, unsigned int count)
{
	if (const auto* prepared=dynamic_cast<const Prepared_Field_Gather_FDTD*>(gather))
	{
		const int id=m_Eng_GPU->AddPreparedFieldDFT(prepared->HField(),prepared->prepared,count);
		if (id>=0) m_PreparedFieldDFT[id]=prepared;
		return id;
	}
	const Field_Gather_FDTD* g = dynamic_cast<const Field_Gather_FDTD*>(gather);
	if (!g || (g->region.Size()==0))
		return -1;
	const int id = m_Eng_GPU->AddFieldDFT(g->HField(), g->region, count);
	if (id>=0)
		m_FieldDFT[id] = g;
	return id;
}

void Engine_Interface_GPU_FDTD::AccumulateFieldDFT(int id, const std::vector<std::complex<float>>& weights)
{
	m_Eng_GPU->AccumulateFieldDFT(id, weights);
}

bool Engine_Interface_GPU_FDTD::ReadFieldDFT(int id, std::vector<ArrayLib::ArrayNIJK<std::complex<float>>*>& fields)
{
	const auto prepared=m_PreparedFieldDFT.find(id);
	if (prepared!=m_PreparedFieldDFT.end())
	{
		std::vector<std::complex<float>> sums;
		for (size_t f=0;f<fields.size();++f)
		{
			if (!m_Eng_GPU->ReadFieldDFT(id,f,sums) || sums.size()!=prepared->second->EntryCount()) return false;
			prepared->second->EvaluateSums(sums.data(),*fields[f]);
		}
		return true;
	}
	std::map<int, const Field_Gather_FDTD*>::const_iterator it = m_FieldDFT.find(id);
	if (it==m_FieldDFT.end())
		return false;
	// the sums of the raw values, one frequency at a time: interpolated here, like the dump does it with the fields
	std::vector<std::complex<float>> sums;
	for (size_t f=0; f<fields.size(); ++f)
	{
		if (!m_Eng_GPU->ReadFieldDFT(id, f, sums) || (sums.size()!=it->second->region.Size()))
			return false;
		it->second->EvaluateSums(sums.data(), *fields[f]);
	}
	return true;
}

std::shared_ptr<Engine_SetupTable> Engine_Interface_GPU_FDTD::BuildModeMatch(const Engine_ModeTableInput& input)
{return m_Eng_GPU->GetBackend()->BuildModeMatch(m_Op,input);}
std::shared_ptr<Engine_CellMaterialTable> Engine_Interface_GPU_FDTD::BuildCellMaterials(const unsigned int lines[3],unsigned int* const positions[3],const std::shared_ptr<Engine_SamplingMesh>& mesh)
{return m_Eng_GPU->GetBackend()->BuildCellMaterials(lines,positions,mesh);}
