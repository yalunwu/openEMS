/*
*	Copyright (C) 2015 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#ifndef OPERATOR_EXT_STEADYSTATE_H
#define OPERATOR_EXT_STEADYSTATE_H

#include "operator_extension.h"
#include "FDTD/operator.h"

class Engine_Ext_SteadyState;
class EngineVulkan;

class OPENEMS_EXPORT Operator_Ext_SteadyState : public Operator_Extension
{
	friend class Engine_Ext_SteadyState;
	friend class EngineVulkan;
public:
	Operator_Ext_SteadyState(Operator* op, double period);
	virtual ~Operator_Ext_SteadyState();

	virtual Operator_Extension* Clone(Operator* op);

	virtual bool BuildExtension();
	virtual Engine_Extension* CreateEngineExtention();

	virtual bool IsCylinderCoordsSave(bool closedAlpha, bool R0_included) const {UNUSED(closedAlpha); UNUSED(R0_included); return true;}
	virtual bool IsCylindricalMultiGridSave(bool child) const {UNUSED(child); return true;}
	virtual bool IsMPISave() const {return true;}

	virtual std::string GetExtensionName() const
	{
		return std::string("Steady-State Detection Extension");
	}

	virtual void ShowStat(std::ostream &ostr) const;

	virtual void Reset();

	bool Add_E_Probe(unsigned int pos[3], int dir);
	bool Add_H_Probe(unsigned int pos[3], int dir);

	unsigned int GetTSPeriod() const { return m_TS_period; }
	size_t GetNumberOfEProbes() const { return m_E_probe_dir.size(); }
	unsigned int GetEProbeDir(size_t idx) const { return m_E_probe_dir.at(idx); }
	void GetEProbePos(size_t idx, unsigned int pos[3]) const
	{
		pos[0] = m_E_probe_pos[0].at(idx);
		pos[1] = m_E_probe_pos[1].at(idx);
		pos[2] = m_E_probe_pos[2].at(idx);
	}
	size_t GetNumberOfHProbes() const { return m_H_probe_dir.size(); }
	unsigned int GetHProbeDir(size_t idx) const { return m_H_probe_dir.at(idx); }
	void GetHProbePos(size_t idx, unsigned int pos[3]) const
	{
		pos[0] = m_H_probe_pos[0].at(idx);
		pos[1] = m_H_probe_pos[1].at(idx);
		pos[2] = m_H_probe_pos[2].at(idx);
	}

protected:
	Operator_Ext_SteadyState(Operator* op, Operator_Ext_SteadyState* op_ext);
	double m_T_period;
	unsigned int m_TS_period;
	std::vector<unsigned int> m_E_probe_pos[3];
	std::vector<unsigned int> m_E_probe_dir;
	std::vector<unsigned int> m_H_probe_pos[3];
	std::vector<unsigned int> m_H_probe_dir;
};

#endif // OPERATOR_EXT_STEADYSTATE_H
