// CPU sampling oracle retained from the pre-port implementation.
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

#ifndef REFERENCE_FIELD_GATHER_H
#define REFERENCE_FIELD_GATHER_H

#include <vector>
#include <memory>
#include <mutex>
#include "Common/engine_interface_base.h"
#include "tools/constants.h"
#include "tools/arraylib/array_nijk.h"

class Operator;

//! One component of a dumped node: the (interpolated) E or H field from up to four voltages or currents
/*!
  idx[k] indexes the voltages or currents in the flat ArrayNIJK layout,
  ((n*numLines[0] + x)*numLines[1] + y)*numLines[2] + z.
  raw(k) = field[idx[k]] / delta[k] (0 if delta[k] is 0), and the value is
  - ZERO: 0
  - RAW: raw(0)
  - LERP: raw(0)*(1-rel) + raw(1)*rel
  - AVG4: (raw(0) + raw(1) + raw(2) + raw(3)) / 4
  in double precision, rounded to float. See Reference_Field_Gather_FDTD.
  */
#include "FDTD/engine_field_gather.h"
class Reference_Field_Gather_FDTD : public Engine_Field_Gather
{
public:
	//! one component of a node: raw(k) = value(idx[k])/delta[k] (0 if delta is 0), see GetRawField().
	//! The layout a backend evaluates on the device, see Engine::AddSnapshotGather().
	typedef Engine_GatherEntry Entry;
	typedef Engine_GatherEntry G;

	Reference_Field_Gather_FDTD(const ArrayLib::ArrayNIJK<FDTD_FLOAT>* volt, const ArrayLib::ArrayNIJK<FDTD_FLOAT>* curr,
	                  bool h_field, unsigned int nj, unsigned int nk)
		: m_Volt(volt), m_Curr(curr), m_H(h_field), m_nj(nj), m_nk(nk), m_Evaluated(false), m_Offset(0) {}

	//! The evaluation of GetEField() (\a h_field false) or GetHField() of Engine_Interface_FDTD with the interpolation \a interpol at the nodes posLines, reading \a volt or \a curr
	static Reference_Field_Gather_FDTD* New(const Operator* op, Engine_Interface_Base::InterpolationType interpol,
	                              const ArrayLib::ArrayNIJK<FDTD_FLOAT>* volt, const ArrayLib::ArrayNIJK<FDTD_FLOAT>* curr,
	                              bool h_field, const unsigned int numLines[3], unsigned int* const posLines[3],const std::shared_ptr<Engine_SamplingMesh>& mesh={});

	mutable std::vector<Entry> entries;   //!< [line][k][n], materialized only for host evaluation
	std::shared_ptr<GPU_FieldGather> prepared;
	size_t EntryCount() const {return prepared ? prepared->Count() : entries.size();}
	void EnsureHost() const {if (prepared) std::call_once(m_HostRead,[this]() {prepared->Read(entries);});}
	bool EvaluateDevice(ArrayLib::ArrayNIJK<float>& field) const override {return prepared && prepared->Evaluate(field.data(),field.size());}

	bool HField() const {return m_H;}
	unsigned int NumLinesJ() const {return m_nj;}
	unsigned int NumLinesK() const {return m_nk;}
	//! The snapshots hold the values of these entries from \a offset on, see PrepareSnapshotGather()
	void SetSnapshotEvaluated(size_t offset) {m_Evaluated = true; m_Offset = offset;}

	virtual void Evaluate(size_t line_start, size_t line_stop, ArrayLib::ArrayNIJK<float> &field, const float* src=NULL) const
	{
		// the device evaluated the entries into the snapshot, only unpack them
		if (src && m_Evaluated)
		{
			for (size_t l=line_start; l<line_stop; ++l)
			{
				const float* v = src + m_Offset + l*m_nk*3;
				const unsigned int i = l/m_nj;
				const unsigned int j = l%m_nj;
				for (unsigned int k=0; k<m_nk; ++k)
					for (int n=0; n<3; ++n)
						field(n, i, j, k) = v[k*3 + n];
			}
			return;
		}
		EnsureHost();
		const float* f = src ? src : (m_H ? m_Curr->data() : m_Volt->data());
		for (size_t l=line_start; l<line_stop; ++l)
		{
			const unsigned int i = l/m_nj;
			const unsigned int j = l%m_nj;
			for (unsigned int k=0; k<m_nk; ++k)
				for (int n=0; n<3; ++n)
				{
					const Entry& e = entries[(l*m_nk + k)*3 + n];
					double out = 0;
					switch (e.form)
					{
					case G::ZERO:
						break;
					case G::RAW:
						out = Raw(f, e, 0);
						break;
					case G::LERP:
						out = Raw(f, e, 0)*(1.0-e.rel) + Raw(f, e, 1)*e.rel;
						break;
					case G::AVG4:
						out = Raw(f, e, 0);
						out+= Raw(f, e, 1);
						out+= Raw(f, e, 2);
						out+= Raw(f, e, 3);
						out/=4;
						break;
					}
					field(n, i, j, k) = out;
				}
		}
	}

protected:
	static inline double Raw(const float* f, const Entry& e, int k)
	{
		double value = f[e.idx[k]];
		if (e.delta[k])
			return value/e.delta[k];
		return 0.0;
	}

	const ArrayLib::ArrayNIJK<FDTD_FLOAT>* m_Volt;
	mutable std::once_flag m_HostRead;
	const ArrayLib::ArrayNIJK<FDTD_FLOAT>* m_Curr;
	bool m_H;
	unsigned int m_nj, m_nk;
	bool m_Evaluated;   //!< a snapshot holds the evaluated entries, see SetSnapshotEvaluated()
	size_t m_Offset;
};

#include "FDTD/operator.h"
#include "FDTD/operator_cylinder.h"
inline Reference_Field_Gather_FDTD* Reference_Field_Gather_FDTD::New(const Operator* op, Engine_Interface_Base::InterpolationType interpol,
                                          const ArrayLib::ArrayNIJK<FDTD_FLOAT>* volt, const ArrayLib::ArrayNIJK<FDTD_FLOAT>* curr,
                                          bool h_field, const unsigned int numLines[3], unsigned int* const posLines[3],const std::shared_ptr<Engine_SamplingMesh>& mesh)
{
	unsigned int N[3];
	for (int n=0; n<3; ++n)
		N[n] = op->GetNumberOfLines(n, true);
	auto index = [&](int n, const unsigned int* p) -> unsigned int {return ((n*N[0] + p[0])*N[1] + p[1])*N[2] + p[2];};
	auto delta = [&](int n, const unsigned int* p) -> double {return op->GetEdgeLength(n, p, h_field);};

	typedef Engine_GatherEntry G;
	Reference_Field_Gather_FDTD* gather = new Reference_Field_Gather_FDTD(volt, curr, h_field, numLines[1], numLines[2]);
	gather->entries.resize((size_t)numLines[0]*numLines[1]*numLines[2]*3);
	size_t e_idx = 0;
	for (unsigned int i=0; i<numLines[0]; ++i)
		for (unsigned int j=0; j<numLines[1]; ++j)
			for (unsigned int k=0; k<numLines[2]; ++k)
			{
				const unsigned int pos[3] = {posLines[0][i], posLines[1][j], posLines[2][k]};
				for (int n=0; n<3; ++n)
				{
					G& e = gather->entries[e_idx++];
					e.form = G::ZERO;
					e.rel = 0;
					for (int m=0; m<4; ++m) {e.idx[m] = 0; e.delta[m] = 0;}
					unsigned int p[3] = {pos[0], pos[1], pos[2]};
					const int nP = (n+1)%3;
					const int nPP = (n+2)%3;
					auto set = [&](int m) {e.idx[m] = index(n, p); e.delta[m] = delta(n, p);};
					const bool upper = (pos[0]==N[0]-1) || (pos[1]==N[1]-1) || (pos[2]==N[2]-1);
					switch (interpol)
					{
					default:
					case Engine_Interface_Base::NO_INTERPOLATION:
						e.form = G::RAW;
						set(0);
						break;
					case Engine_Interface_Base::NODE_INTERPOLATE:
						if (!h_field)
						{
							// see GetRawInterpolatedField()
							if (pos[n]==N[n]-1)
							{
								--p[n];
								e.form = G::RAW;
								set(0);
								break;
							}
							const double d = delta(n, p);
							if (d==0)
								break;   // ZERO
							e.form = G::RAW;
							set(0);
							if (pos[n]==0)
								break;
							--p[n];
							const double d_down = delta(n, p);
							e.form = G::LERP;
							e.rel = d / (d+d_down);
							set(1);
						}
						else
						{
							// see GetRawInterpolatedDualField()
							if (upper || (pos[nP]==0) || (pos[nPP]==0))
								break;   // ZERO
							e.form = G::AVG4;
							set(0);
							--p[nP];
							set(1);
							--p[nPP];
							set(2);
							++p[nP];
							set(3);
						}
						break;
					case Engine_Interface_Base::CELL_INTERPOLATE:
						if (!h_field)
						{
							// see GetRawInterpolatedField()
							if (upper)
								break;   // ZERO
							e.form = G::AVG4;
							set(0);
							++p[nP];
							set(1);
							++p[nPP];
							set(2);
							--p[nP];
							set(3);
						}
						else
						{
							// see GetRawInterpolatedDualField()
							if (pos[n]>=N[n]-1)
								break;   // ZERO
							const double d = delta(n, p);
							e.form = G::LERP;
							set(0);
							++p[n];
							const double d_up = delta(n, p);
							e.rel = d / (d+d_up);
							set(1);
						}
						break;
					}
				}
			}
	return gather;
}

#endif // REFERENCE_FIELD_GATHER_H
