/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "FDTD/operator_gpu.h"
#include "FDTD/operator_cylinder.h"
#include "FDTD/extensions/operator_ext_cylinder.h"
#include "FDTD/extensions/engine_ext_cylinder.h"
#include "FDTD/gpu_setup_adapters.h"
#include "FDTD/extensions/operator_ext_upml.h"
#include "FDTD/extensions/operator_ext_excitation.h"
#include "FDTD/extensions/operator_ext_tfsf.h"
#include "FDTD/extensions/operator_ext_mur_abc.h"
#include "FDTD/extensions/operator_ext_absorbing_bc.h"
#include "FDTD/extensions/operator_ext_lumpedRLC.h"
#include "FDTD/extensions/operator_ext_lorentzmaterial.h"
#include "FDTD/extensions/operator_ext_debyematerial.h"
#include "FDTD/extensions/operator_ext_conductingsheet.h"
#include "FDTD/extensions/operator_ext_steadystate.h"
#include "FDTD/extensions/engine_ext_upml.h"
#include "FDTD/extensions/engine_ext_excitation.h"
#include "FDTD/extensions/engine_ext_tfsf.h"
#include "FDTD/extensions/engine_ext_mur_abc.h"
#include "FDTD/extensions/engine_ext_absorbing_bc.h"
#include "FDTD/extensions/engine_ext_lumpedRLC.h"
#include "FDTD/extensions/engine_ext_lorentzmaterial.h"
#include "FDTD/extensions/engine_ext_debyematerial.h"
#include "FDTD/extensions/engine_ext_steadystate.h"
#include "CSPropExcitation.h"
#include "CSPropLorentzMaterial.h"
#include "CSPropDebyeMaterial.h"
#include "CSPropLumpedElement.h"
#include "CSPropConductingSheet.h"
#include "CSPropAbsorbingBC.h"
#include "CSPrimBox.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

namespace
{
struct Tables
{
	std::map<std::string,std::vector<unsigned int>> discrete;
	std::map<std::string,std::vector<double>> numeric;
	template<class T> void D(const std::string& name,const T* p,size_t count) {if (count) discrete[name]=std::vector<unsigned int>(p,p+count);else discrete[name]={};}
	template<class T> void N(const std::string& name,const T* p,size_t count) {if (count) numeric[name]=std::vector<double>(p,p+count);else numeric[name]={};}
	void Compare(const Tables& actual) const
	{
		if (discrete!=actual.discrete) throw std::runtime_error("extension discrete snapshot differs");
		if (numeric.size()!=actual.numeric.size()) throw std::runtime_error("extension snapshot inventory differs");
		for (const auto& table:numeric)
		{
			const auto& values=actual.numeric.at(table.first);if (table.second.size()!=values.size()) throw std::runtime_error("extension snapshot dimensions differ: "+table.first);
			double scale=0;for (double value:table.second) scale=(std::max)(scale,std::abs(value));
			for (size_t n=0;n<values.size();++n)
				if (!std::isfinite(values[n]) || std::abs(values[n]-table.second[n])>2e-6*scale)
					throw std::runtime_error("extension coefficient snapshot differs: "+table.first+" at "+std::to_string(n));
		}
	}
};
struct Source : Operator_Ext_Excitation
{
	explicit Source(Operator* op):Operator_Ext_Excitation(op) {}
	void Read(Tables& out)
	{
		EnsureHostConstruction();
		for (int current=0;current<2;++current)
		{
			const auto count=current ? Curr_Count : Volt_Count;const std::string key=current ? "H" : "E";
			out.D(key+"count",&count,1);out.D(key+"dir",current ? Curr_dir : Volt_dir,count);
			out.D(key+"delay",current ? Curr_delay : Volt_delay,count);out.N(key+"amp",current ? Curr_amp : Volt_amp,count);
			for (int n=0;n<3;++n) out.D(key+std::to_string(n),current ? Curr_index[n] : Volt_index[n],count);
		}
	}
};
struct UPML : Operator_Ext_UPML
{
	explicit UPML(Operator* op):Operator_Ext_UPML(op) {}
	void Read(Tables& out)
	{
		EnsureHostConstruction();out.D("start",m_StartPos,3);out.D("size",m_numLines,3);
		ArrayLib::ArrayNIJK<float>* arrays[]={&vv,&vvfo,&vvfn,&ii,&iifo,&iifn};
		for (int k=0;k<6;++k) out.N(std::to_string(k),arrays[k]->data(),arrays[k]->size());
	}
};
struct Mur : Operator_Ext_Mur_ABC
{
	explicit Mur(Operator* op):Operator_Ext_Mur_ABC(op) {}
	void Read(Tables& out) {EnsureHostConstruction();out.D("size",m_numLines,2);out.N("a",m_Mur_Coeff_nyP.data(),m_Mur_Coeff_nyP.size());out.N("b",m_Mur_Coeff_nyPP.data(),m_Mur_Coeff_nyPP.size());}
};
struct Absorber : Operator_Ext_Absorbing_BC
{
	explicit Absorber(Operator* op):Operator_Ext_Absorbing_BC(op) {}
	void Read(Tables& out)
	{
		EnsureHostConstruction();out.D("start",m_sheetX0,3);out.D("stop",m_sheetX1,3);
		ArrayLib::ArrayIJ<float>* arrays[]={&m_K1_nyP,&m_K1_nyPP,&m_K2_nyP,&m_K2_nyPP};
		for (int k=0;k<4;++k) out.N(std::to_string(k),arrays[k]->data(),arrays[k]->size());
	}
};
struct RLC : Operator_Ext_LumpedRLC
{
	explicit RLC(Operator* op):Operator_Ext_LumpedRLC(op) {}
	void Read(Tables& out)
	{
		EnsureHostConstruction();out.D("count",&RLC_count,1);out.D("dir",v_RLC_dir,RLC_count);
		for (int n=0;n<3;++n) out.D("pos"+std::to_string(n),v_RLC_pos[n],RLC_count);
		float* arrays[]={v_RLC_ilv,v_RLC_i2v,v_RLC_vv2,v_RLC_vj1,v_RLC_vj2,v_RLC_vvd,v_RLC_ib0,v_RLC_b1,v_RLC_b2};
		for (int k=0;k<9;++k) out.N(std::to_string(k),arrays[k],RLC_count);
	}
};
template<class Base> struct Lorentz : Base
{
	template<class... Args> explicit Lorentz(Args&&... args):Base(std::forward<Args>(args)...) {}
	void Read(Tables& out)
	{
		this->EnsureHostConstruction();out.D("count",this->m_LM_Count.data(),this->m_LM_Count.size());
		out.D("v",this->m_volt_ADE_On,this->m_Order);out.D("i",this->m_curr_ADE_On,this->m_Order);
		out.D("vl",this->m_volt_Lor_ADE_On,this->m_Order);out.D("il",this->m_curr_Lor_ADE_On,this->m_Order);
		for (int k=0;k<this->m_Order;++k) for (int n=0;n<3;++n)
		{
			const auto key=std::to_string(k)+":"+std::to_string(n);const auto count=this->m_LM_Count[k];
			out.D(key+"pos",this->m_LM_pos[k][n],count);
			if (this->m_volt_ADE_On[k]) {out.N(key+"vi",this->v_int_ADE[k][n],count);out.N(key+"ve",this->v_ext_ADE[k][n],count);}
			if (this->m_curr_ADE_On[k]) {out.N(key+"ii",this->i_int_ADE[k][n],count);out.N(key+"ie",this->i_ext_ADE[k][n],count);}
			if (this->m_volt_Lor_ADE_On[k]) out.N(key+"vl",this->v_Lor_ADE[k][n],count);
			if (this->m_curr_Lor_ADE_On[k]) out.N(key+"il",this->i_Lor_ADE[k][n],count);
		}
	}
};
struct Debye : Operator_Ext_DebyeMaterial
{
	explicit Debye(Operator* op):Operator_Ext_DebyeMaterial(op) {}
	void Read(Tables& out)
	{
		EnsureHostConstruction();out.D("count",m_LM_Count.data(),m_LM_Count.size());
		for (int n=0;n<3;++n)
		{
			const auto key=std::to_string(n);const auto count=m_LM_Count[0];out.D(key+"pos",m_LM_pos[0][n],count);out.N(key+"solve",v_solve_ADE[n],count);
			for (int k=0;k<m_PoleCount;++k) {out.N(key+"r"+std::to_string(k),v_relax_ADE[k][n],count);out.N(key+"d"+std::to_string(k),v_drive_ADE[k][n],count);}
		}
	}
};
struct TFSF : Operator_Ext_TFSF
{
	explicit TFSF(Operator* op):Operator_Ext_TFSF(op) {}
	void Read(Tables& out)
	{
		EnsureHostConstruction();out.D("start",m_Start,3);out.D("stop",m_Stop,3);out.D("delay",&m_maxDelay,1);
		out.N("direction",m_PropDir,3);out.N("E",m_E_Amp,3);out.N("H",m_H_Amp,3);out.N("velocity",&m_PhVel,1);
		for (int n=0;n<3;++n) for (int side=0;side<2;++side) if (m_ActiveDir[n][side])
		{
			const auto count=m_numLines[(n+1)%3]*m_numLines[(n+2)%3];
			for (int c=0;c<2;++c)
			{
				const auto key=std::to_string(n)+":"+std::to_string(side)+":"+std::to_string(c);
				out.D(key+"vdelay",m_VoltDelay[n][side][c],count);out.D(key+"idelay",m_CurrDelay[n][side][c],count);
				out.N(key+"vdelta",m_VoltDelayDelta[n][side][c],count);out.N(key+"idelta",m_CurrDelayDelta[n][side][c],count);
				out.N(key+"vamp",m_VoltAmp[n][side][c],count);out.N(key+"iamp",m_CurrAmp[n][side][c],count);
			}
		}
	}
};
struct Steady : Operator_Ext_SteadyState
{
	explicit Steady(Operator* op):Operator_Ext_SteadyState(op,1e-9) {}
	void Read(Tables& out) {out.D("period",&m_TS_period,1);out.N("time",&m_T_period,1);out.D("e",m_E_probe_dir.data(),m_E_probe_dir.size());out.D("h",m_H_probe_dir.data(),m_H_probe_dir.size());for (int n=0;n<3;++n) {out.D("e"+std::to_string(n),m_E_probe_pos[n].data(),m_E_probe_pos[n].size());out.D("h"+std::to_string(n),m_H_probe_pos[n].data(),m_H_probe_pos[n].size());}}
};
struct Cylinder : Operator_Ext_Cylinder
{
	explicit Cylinder(Operator_Cylinder* op):Operator_Ext_Cylinder(op) {}
	void Read(Tables& out) {EnsureHostConstruction();const unsigned int count=CC_R0_included ? m_Op->GetNumberOfLines(2,true) : 0;out.N("vv",vv_R0,count);out.N("vi",vi_R0,count);}
};
struct TestCylinder : Operator_Cylinder
{
	explicit TestCylinder(bool gpu) {Init();setNumThreads(1);SetGPUBackend("auto",nullptr,gpu ? "gpu" : "cpu");}
	using Operator_Cylinder::CalcECOperator;
	void SetAxis(Cylinder* view) {DeleteExtension(m_Cyl_Ext);m_Cyl_Ext=view;AddExtension(view);}
};
struct TestOperator : Operator_GPU {explicit TestOperator(bool gpu):Operator_GPU("auto",1,gpu ? "gpu" : "cpu") {Init();}};
CSPrimBox* Box(ContinuousStructure& csx,CSProperties* prop,const double start[3],const double stop[3])
{
	auto* box=new CSPrimBox(csx.GetParameterSet(),prop);for (int n=0;n<3;++n) {box->SetCoord(2*n,start[n]);box->SetCoord(2*n+1,stop[n]);}return box;
}
template<class View,class Base,class Engine> struct Register
{
	Register() {GPU_RegisterExtensionAdapter(typeid(View),typeid(Base),typeid(Engine),"snapshot",[](Operator_Extension&,std::string&) {return true;});}
	~Register() {GPU_UnregisterExtensionAdapter(typeid(View));}
};
template<class View,class Base,class Engine,class Factory> void Pair(ContinuousStructure& csx,Factory factory)
{
	Register<View,Base,Engine> registration;Tables expected,actual;
	for (bool gpu:{false,true})
	{
#ifndef OPENEMS_CONTRACT_HIP
		if (gpu) continue;
#endif
		Excitation waveform;waveform.SetupGaussianPulse(1e9,0.5e9);TestOperator op(gpu);op.SetExcitationSignal(&waveform);
		for (int n=0;n<6;++n) if (op.GetBCSize(n)!=0) throw std::runtime_error("default operator boundary width is not zero");
		if (!op.SetGeometryCSX(&csx)) throw std::runtime_error("extension snapshot grid invalid");
		View* view=factory(op,gpu);op.AddExtension(view);
		if (op.CalcECOperator()!=0 || op.GetNumberOfExtentions()!=1 || (gpu && (!op.GetConstruction() || !view->HasDeviceConstruction()))) throw std::runtime_error("extension snapshot skipped construction");
		view->Read(gpu ? actual : expected);
	}
#ifdef OPENEMS_CONTRACT_HIP
	expected.Compare(actual);
#endif
	std::cout << "PASS extension table snapshot " << typeid(Base).name() << std::endl;
}
}

void openEMS_TestGPUExtensionSnapshots()
{
	ContinuousStructure csx;csx.GetGrid()->SetDeltaUnit(1e-3);for (int n=0;n<3;++n) for (int p=-6;p<=6;++p) csx.GetGrid()->AddDiscLine(n,p);
	auto* source=new CSPropExcitation(csx.GetParameterSet());csx.AddProperty(source);source->SetName("source");source->SetExcitType(0);source->SetExcitation(1,2);source->SetWeightFunction("1/2",2);source->SetDelay(2e-12);
	const double s0[3]={0,0,-2},s1[3]={0,0,2};Box(csx,source,s0,s1);
	auto* magnetic=new CSPropExcitation(csx.GetParameterSet());csx.AddProperty(magnetic);magnetic->SetName("magnetic");magnetic->SetExcitType(2);magnetic->SetExcitation(.5,1);Box(csx,magnetic,s0,s1);
	auto* plane=new CSPropExcitation(csx.GetParameterSet());csx.AddProperty(plane);plane->SetName("plane");plane->SetExcitType(10);plane->SetPropagationDir(2,0);plane->SetPropagationDir(-1,1);plane->SetExcitation(1,2);plane->SetExcitation(.2,0);plane->SetFrequency(1e9);
	const double low[3]={-3,-3,-3},high[3]={3,3,3};Box(csx,plane,low,high);
	auto* lorentz=new CSPropLorentzMaterial(csx.GetParameterSet());csx.AddProperty(lorentz);lorentz->SetName("lorentz");lorentz->SetDispersionOrder(1);lorentz->SetEpsilon(2);lorentz->SetEpsPlasmaFreq(0,2e9);lorentz->SetEpsLorPoleFreq(0,1e9);lorentz->SetEpsRelaxTime(0,2e-11);lorentz->SetMuePlasmaFreq(0,1e9);lorentz->SetMueLorPoleFreq(0,.5e9);lorentz->SetMueRelaxTime(0,2e-11);
	const double l0[3]={-3,-3,-3},l1[3]={-1,3,3};Box(csx,lorentz,l0,l1);
	auto* debye=new CSPropDebyeMaterial(csx.GetParameterSet());csx.AddProperty(debye);debye->SetName("debye");debye->SetDispersionOrder(2);debye->SetEpsilon(2);debye->SetEpsDelta(0,1.5);debye->SetEpsRelaxTime(0,1e-11);debye->SetEpsDelta(1,.5);debye->SetEpsRelaxTime(1,3e-11);
	const double d0[3]={1,-3,-3},d1[3]={3,3,3};Box(csx,debye,d0,d1);
	auto* sheet=new CSPropConductingSheet(csx.GetParameterSet());csx.AddProperty(sheet);sheet->SetName("sheet");sheet->SetConductivity(5e6);sheet->SetThickness(1e-5);const double c0[3]={-2,-2,0},c1[3]={2,2,0};Box(csx,sheet,c0,c1);
	for (int kind=0;kind<2;++kind) {auto* lumped=new CSPropLumpedElement(csx.GetParameterSet());csx.AddProperty(lumped);lumped->SetName("lumped"+std::to_string(kind));lumped->SetDirection(2);lumped->SetResistance(50);lumped->SetInductance(1e-9);lumped->SetCapacity(2e-12);lumped->SetLEtype(static_cast<CSPropLumpedElement::LEtype>(kind));const double a[3]={double(kind*4-2),-2,1},b[3]={double(kind*4-1),-1,3};Box(csx,lumped,a,b);}
	auto* absorbing=new CSPropAbsorbingBC(csx.GetParameterSet());csx.AddProperty(absorbing);absorbing->SetName("absorber");absorbing->SetPhaseVelocity(C0);absorbing->SetAbsorbingBoundaryType(CSPropAbsorbingBC::MUR_1ST_SA);const double a0[3]={-4,-4,4},a1[3]={4,4,4};auto* absorbing_box=Box(csx,absorbing,a0,a1);
	if (!csx.Update().empty()) throw std::runtime_error("extension snapshot geometry invalid");
	const double sheet_probe[3]={.5,.5,0};
	auto* sheet_winner=csx.GetPropertyByCoordPriority(sheet_probe,static_cast<CSProperties::PropertyType>(CSProperties::MATERIAL|CSProperties::METAL));
	if (sheet_winner!=sheet) throw std::runtime_error("conducting sheet fixture lost its material priority");
	if (sheet_winner->ToConductingSheet()!=sheet || dynamic_cast<CSPropConductingSheet*>(sheet_winner)!=sheet) throw std::runtime_error("conducting sheet RTTI differs across geometry/solver libraries");
	Pair<Source,Operator_Ext_Excitation,Engine_Ext_Excitation>(csx,[](Operator& op,bool) {return new Source(&op);});
	Pair<TFSF,Operator_Ext_TFSF,Engine_Ext_TFSF>(csx,[](Operator& op,bool) {return new TFSF(&op);});
	Pair<UPML,Operator_Ext_UPML,Engine_Ext_UPML>(csx,[](Operator& op,bool) {auto* view=new UPML(&op);const int bc[6]={3,0,0,0,0,0};const unsigned int sizes[6]={2,0,0,0,0,0},start[3]={0,0,0},stop[3]={2,12,12};view->SetBoundaryCondition(bc,sizes);view->SetRange(start,stop);return view;});
	Pair<Mur,Operator_Ext_Mur_ABC,Engine_Ext_Mur_ABC>(csx,[](Operator& op,bool) {auto* view=new Mur(&op);view->SetDirection(2,true);return view;});
	Pair<Absorber,Operator_Ext_Absorbing_BC,Engine_Ext_Absorbing_BC>(csx,[&](Operator& op,bool gpu) {auto* view=new Absorber(&op);if (!view->SetInitParams(absorbing_box,absorbing,gpu)) throw std::runtime_error("absorber snapshot input invalid");return view;});
	Pair<RLC,Operator_Ext_LumpedRLC,Engine_Ext_LumpedRLC>(csx,[](Operator& op,bool) {return new RLC(&op);});
	Pair<Lorentz<Operator_Ext_LorentzMaterial>,Operator_Ext_LorentzMaterial,Engine_Ext_LorentzMaterial>(csx,[](Operator& op,bool) {return new Lorentz<Operator_Ext_LorentzMaterial>(&op);});
	Pair<Debye,Operator_Ext_DebyeMaterial,Engine_Ext_DebyeMaterial>(csx,[](Operator& op,bool) {return new Debye(&op);});
	Pair<Lorentz<Operator_Ext_ConductingSheet>,Operator_Ext_ConductingSheet,Engine_Ext_LorentzMaterial>(csx,[](Operator& op,bool) {return new Lorentz<Operator_Ext_ConductingSheet>(&op,3e9);});
	Pair<Steady,Operator_Ext_SteadyState,Engine_Ext_SteadyState>(csx,[](Operator& op,bool) {auto* view=new Steady(&op);unsigned int p[3]={5,5,5};view->Add_E_Probe(p,0);view->Add_H_Probe(p,2);return view;});
	Register<Cylinder,Operator_Ext_Cylinder,Engine_Ext_Cylinder> registration;Tables expected,actual;
	ContinuousStructure cylinder;cylinder.SetCoordInputType(CYLINDRICAL);cylinder.GetGrid()->SetDeltaUnit(1e-3);
	for (double r:{0.0,1.0,2.5,4.0,6.0}) cylinder.GetGrid()->AddDiscLine(0,r);
	for (int n=0;n<=12;++n) cylinder.GetGrid()->AddDiscLine(1,2*acos(-1.0)*n/12);
	for (double z:{-3.0,-1.0,0.0,2.0,4.0}) cylinder.GetGrid()->AddDiscLine(2,z);
	for (bool gpu:{false,true})
	{
#ifndef OPENEMS_CONTRACT_HIP
		if (gpu) continue;
#endif
		TestCylinder op(gpu);Excitation waveform;waveform.SetupGaussianPulse(1e9,.5e9);op.SetExcitationSignal(&waveform);
		if (!op.SetGeometryCSX(&cylinder) || op.GetNumberOfExtentions()!=1) throw std::runtime_error("axis snapshot input invalid");
		auto* view=new Cylinder(&op);op.SetAxis(view);
		if (op.CalcECOperator()!=0 || (gpu && !view->HasDeviceConstruction())) throw std::runtime_error("axis snapshot skipped construction");
		view->Read(gpu ? actual : expected);
	}
#ifdef OPENEMS_CONTRACT_HIP
	expected.Compare(actual);
#endif
	std::cout << "PASS cylindrical axis table snapshot" << std::endl;
#ifdef OPENEMS_CONTRACT_HIP
	// Custom grading must fail before publishing device construction, and the
	// same operator must remain usable after restoring the default function.
	Register<UPML,Operator_Ext_UPML,Engine_Ext_UPML> pml_registration;
	TestOperator custom_pml(true);Excitation pml_signal;pml_signal.SetupGaussianPulse(1e9,.5e9);custom_pml.SetExcitationSignal(&pml_signal);
	if (!custom_pml.SetGeometryCSX(&csx)) throw std::runtime_error("custom PML grid invalid");
	auto* pml=new UPML(&custom_pml);custom_pml.AddExtension(pml);
	const int pml_bc[6]={3,0,0,0,0,0};const unsigned int pml_sizes[6]={2,0,0,0,0,0},pml_start[3]={0,0,0},pml_stop[3]={2,12,12};
	pml->SetBoundaryCondition(pml_bc,pml_sizes);pml->SetRange(pml_start,pml_stop);
	const std::string default_grading=pml->GetGradingFunction();
	if (!pml->SetGradingFunction("D/dl")) throw std::runtime_error("custom PML fixture invalid");
	bool pml_rejected=false;
	try {custom_pml.CalcECOperator();} catch (const std::runtime_error& error) {pml_rejected=std::string(error.what()).find("custom PML grading requires CPU operator construction")!=std::string::npos;}
	if (!pml_rejected || custom_pml.GetConstruction()) throw std::runtime_error("custom PML published device construction");
	if (!pml->SetGradingFunction(default_grading) || custom_pml.CalcECOperator()!=0 || !custom_pml.GetConstruction()) throw std::runtime_error("PML reuse after unsupported grading failed");
	std::cout << "PASS custom PML rejection and reuse with default grading" << std::endl;

	Register<Absorber,Operator_Ext_Absorbing_BC,Engine_Ext_Absorbing_BC> absorber_registration;
	absorbing_box->SetPrimitiveUsed(false);TestOperator cancelled(true);Excitation waveform;waveform.SetupGaussianPulse(1e9,.5e9);cancelled.SetExcitationSignal(&waveform);
	if (!cancelled.SetGeometryCSX(&csx)) throw std::runtime_error("absorber cancellation grid invalid");
	auto* absorber_view=new Absorber(&cancelled);absorber_view->SetInitParams(absorbing_box,absorbing,true);cancelled.AddExtension(absorber_view);
	cancelled.SetSetupCancellation([&]() {return absorbing_box->GetPrimitiveUsed();});bool rejected=false;
	try {cancelled.CalcECOperator();} catch (const std::runtime_error& error) {rejected=std::string(error.what())=="FDTD setup cancelled";}
	if (!rejected || absorbing_box->GetPrimitiveUsed() || cancelled.GetConstruction()) throw std::runtime_error("absorber cancellation leaked non-material primitive state");
	cancelled.SetSetupCancellation({});
	if (cancelled.CalcECOperator()!=0 || !absorbing_box->GetPrimitiveUsed() || !absorber_view->HasDeviceConstruction()) throw std::runtime_error("absorber cancellation reuse failed");
	std::cout << "PASS non-material primitive cancellation rollback and repeated absorber preparation" << std::endl;
#endif
}
