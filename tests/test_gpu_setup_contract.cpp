/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "openems_global.h"
extern "C" OPENEMS_EXPORT int openEMS_TestGPUSetupContract();
#ifndef OPENEMS_NATIVE_SETUP_TESTS
int main() {return openEMS_TestGPUSetupContract();}
#else
#include "FDTD/operator_gpu.h"
#include "FDTD/gpu_setup_controller.h"
#include "FDTD/gpu_setup_processing.h"
#include "ContinuousStructure.h"
#include "CSPropMaterial.h"
#include "CSPropMetal.h"
#include "CSPrimBox.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <chrono>
#include "vtkNew.h"
#include "vtkXMLRectilinearGridReader.h"
#include "vtkRectilinearGrid.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
void openEMS_TestGPUExtensionSnapshots();

namespace
{
class SnapshotOperator : public Operator_GPU
{
public:

	explicit SnapshotOperator(bool gpu) : Operator_GPU("auto",1,gpu ? "gpu" : "cpu") {Init();}
	using Operator::DumpMaterial2File;
	std::vector<float> ec[4];
	std::vector<unsigned int> PEC() const {return {m_Nr_PEC[0],m_Nr_PEC[1],m_Nr_PEC[2]};}
protected:
	void Init_EC() override {Operator::Init_EC();} // Explicit test snapshots.
	void CalcCoefficients() override
	{
		float** arrays[]={EC_C,EC_G,EC_L,EC_R};
		const size_t cells=size_t(numLines[0])*numLines[1]*numLines[2];
		for (int k=0;k<4;++k) for (int n=0;n<3;++n) ec[k].insert(ec[k].end(),arrays[k][n],arrays[k][n]+cells);
		Operator_GPU::CalcCoefficients();
	}
};
void Check(bool value,const char* reason) {if (!value) throw std::runtime_error(reason);}
void Compare(const std::vector<float>& expected,const float* actual,const char* reason)
{
	float scale=0;for (float value:expected) scale=std::max(scale,std::abs(value));
	for (size_t n=0;n<expected.size();++n)
		Check(std::isfinite(actual[n]) && std::abs(double(actual[n])-expected[n])<=2e-6*scale,reason);
}
void ProcessingSnapshots(const Operator& cpu,const Operator& gpu)
{
	for (bool dual:{false,true}) for (bool full:{false,true})
	{
		for (double value:{-8.0,-5.0,-4.25,0.0,4.0,5.0,9.0})
		{
			double a[3]={value,-value,value/2},b[3]={3.25,1.5,-2.75};
			for (int method=0;method<3;++method)
			{
				unsigned int first[3]={},last[3]={},expected_first[3]={},expected_last[3]={};bool inside_first[3]={},inside_last[3]={},expected_a[3]={},expected_b[3]={};
				const int dimension=cpu.SnapBox2Mesh(a,b,expected_first,expected_last,dual,full,method,expected_a,expected_b);
				Check(gpu.SnapBox2Mesh(a,b,first,last,dual,full,method,inside_first,inside_last)==dimension,"processing box dimension differs");
				if (dimension>=0) for (int n=0;n<3;++n) Check(first[n]==expected_first[n] && last[n]==expected_last[n] && inside_first[n]==expected_a[n] && inside_last[n]==expected_b[n],"processing box indices/inside flags differ");
			}
			unsigned int actual[3],expected[3];bool actual_inside[3],expected_inside[3];
			Check(gpu.SnapToMesh(a,actual,dual,full,actual_inside)==cpu.SnapToMesh(a,expected,dual,full,expected_inside),"processing point domain differs");
			for (int n=0;n<3;++n) Check(actual[n]==expected[n] && actual_inside[n]==expected_inside[n],"processing point index/inside differs");
		}
	}
	for (bool dual:{false,true}) for (bool optimal:{false,true}) for (bool singleton:{false,true})
	{
		const unsigned int first[3]={0,1,2},last[3]={singleton ? 0u : 6u,singleton ? 1u : 5u,singleton ? 2u : 4u},step[3]={2,3,2};const double resolution[3]={2.5,3.0,1.1};
		auto mesh=GPU_SetupSamplingMesh(&gpu,first,last,dual,optimal,step,resolution);Check(bool(mesh),"processing mesh skipped HIP");
		std::vector<unsigned int> indices[3];std::vector<double> coordinates[3];unsigned int* index_pointers[3];double* coordinate_pointers[3];
		for (int n=0;n<3;++n) {indices[n].resize(mesh->Lines(n));coordinates[n].resize(mesh->Lines(n));index_pointers[n]=indices[n].data();coordinate_pointers[n]=coordinates[n].data();}
		mesh->Read(index_pointers,coordinate_pointers);
		for (int n=0;n<3;++n)
		{
			std::vector<unsigned int> expected;
			if (!optimal) for (unsigned int p=first[n];p<=last[n];p+=step[n]) expected.push_back(p);
			else
			{
				expected.push_back(first[n]);double previous=cpu.GetDiscLine(n,first[n],dual);
				if (!last[n]) expected.push_back(last[n]);
				else for (unsigned int p=first[n]+1;p<last[n];++p)
					if (cpu.GetDiscLine(n,p+1,dual)-previous>=resolution[n]) {expected.push_back(p);previous=cpu.GetDiscLine(n,p,dual);}
				if (first[n]!=last[n]) expected.push_back(last[n]);
			}
			Check(expected==indices[n],"sampling selection indices differ");
			for (size_t p=0;p<expected.size();++p) Check(coordinates[n][p]==cpu.GetDiscLine(n,expected[p],dual),"sampling output mesh coordinates differ");
		}
	}
}
}

int openEMS_TestGPUSetupContract()
{
	try
	{
		ContinuousStructure csx;csx.GetGrid()->SetDeltaUnit(1e-3);
		for (int n=0;n<3;++n) for (double p:{-5.0,-3.5,-2.0,0.0,1.0,3.0,5.0}) csx.GetGrid()->AddDiscLine(n,p);
		auto* material=new CSPropMaterial(csx.GetParameterSet());csx.AddProperty(material);material->SetName("graded");
		material->SetEpsilon(2.5);material->SetKappa(0.02);material->SetEpsilonWeightFunction("1/2",0);
		auto* volume=new CSPrimBox(csx.GetParameterSet(),material);
		for (int n=0;n<3;++n) {volume->SetCoord(2*n,-3.5);volume->SetCoord(2*n+1,3.0);}
		auto* metal=new CSPropMetal(csx.GetParameterSet());csx.AddProperty(metal);metal->SetName("metal");
		auto* plate=new CSPrimBox(csx.GetParameterSet(),metal);
		for (int n=0;n<3;++n) {plate->SetCoord(2*n,n==2 ? 1.0 : -2.0);plate->SetCoord(2*n+1,n==2 ? 1.0 : 1.0);}
		Check(csx.Update().empty(),"snapshot geometry invalid");
		Excitation signal;signal.SetupGaussianPulse(1e9,0.5e9);
		SnapshotOperator cpu(false);cpu.SetExcitationSignal(&signal);Check(cpu.SetGeometryCSX(&csx),"snapshot CPU mesh invalid");
		Check(cpu.CalcECOperator()==0,"snapshot CPU construction failed");
		std::vector<bool> used;for (auto* primitive:csx.GetAllPrimitives()) {used.push_back(primitive->GetPrimitiveUsed());primitive->SetPrimitiveUsed(false);}
		const float original=cpu.GetVV(0,2,2,2);cpu.SetVV(0,2,2,2,0.25f);Check(cpu.GetVV(0,2,2,2)==0.25f,"CPU setter failed");cpu.SetVV(0,2,2,2,original);
		openEMS_TestGPUExtensionSnapshots();
#ifdef OPENEMS_CONTRACT_HIP
		SnapshotOperator gpu(true);gpu.SetExcitationSignal(&signal);Check(gpu.SetGeometryCSX(&csx),"snapshot GPU mesh invalid");
		Check(gpu.CalcECOperator()==0 && gpu.GetConstruction(),"strict snapshot construction skipped HIP");
		for (int k=0;k<4;++k) Compare(cpu.ec[k],gpu.ec[k].data(),"EC snapshot differs");
		gpu.GetConstruction()->MaterializeCoefficients();
		ArrayLib::ArrayNIJK<float>* actual[]={gpu.vv_ptr,gpu.vi_ptr,gpu.ii_ptr,gpu.iv_ptr};
		ArrayLib::ArrayNIJK<float>* expected[]={cpu.vv_ptr,cpu.vi_ptr,cpu.ii_ptr,cpu.iv_ptr};
		for (int k=0;k<4;++k)
		{
			std::vector<float> snapshot(expected[k]->data(),expected[k]->data()+expected[k]->size());
			Compare(snapshot,actual[k]->data(),"coefficient snapshot differs");
			for (size_t n=0;n<snapshot.size();++n) Check((snapshot[n]==0)==(actual[k]->data()[n]==0),"PEC coefficient mask differs");
		}
		Check(cpu.PEC()==gpu.PEC(),"PEC counts differ");
		Check(std::abs(cpu.GetTimestep()-gpu.GetTimestep())<=1e-12*cpu.GetTimestep(),"timestep snapshot differs");
		const auto primitives=csx.GetAllPrimitives();for (size_t n=0;n<used.size();++n) Check(used[n]==primitives[n]->GetPrimitiveUsed(),"primitive used flags differ");
		gpu.SetVV(0,2,2,2,0.25f);Check(gpu.GetVV(0,2,2,2)==0.25f,"GPU public setter failed");
		ProcessingSnapshots(cpu,gpu);
		const std::string debug_name="gpu_setup_contract_debug_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
		cpu.DumpMaterial2File(debug_name+"_cpu");gpu.DumpMaterial2File(debug_name+"_gpu");
		vtkNew<vtkXMLRectilinearGridReader> cpu_dump,gpu_dump;
		cpu_dump->SetFileName((debug_name+"_cpu.vtr").c_str());gpu_dump->SetFileName((debug_name+"_gpu.vtr").c_str());cpu_dump->Update();gpu_dump->Update();
		for (const char* name:{"epsilon","mue","kappa","sigma"})
		{
			auto* expected=cpu_dump->GetOutput()->GetPointData()->GetArray(name);auto* actual=gpu_dump->GetOutput()->GetPointData()->GetArray(name);
			Check(expected && actual && expected->GetNumberOfTuples()==343 && actual->GetNumberOfTuples()==343,"late public material dump dimensions differ");
			Check(expected->GetNumberOfComponents()==3 && actual->GetNumberOfComponents()==3,"late public material dump components differ");
			for (vtkIdType p=0;p<343;++p) for (int n=0;n<3;++n) Check(expected->GetComponent(p,n)==actual->GetComponent(p,n),"late public material debug dump differs");
		}
		std::remove((debug_name+"_cpu.vtr").c_str());std::remove((debug_name+"_gpu.vtr").c_str());
		// Cancel after geometry upload and material construction, then repeat on
		// the same operator. No partial device result or used flags may survive.
		for (auto* primitive:primitives) primitive->SetPrimitiveUsed(false);
		SnapshotOperator cancelled(true);cancelled.SetExcitationSignal(&signal);Check(cancelled.SetGeometryCSX(&csx),"cancellation mesh invalid");
		unsigned int checks=0;cancelled.SetSetupCancellation([&checks]() {return ++checks==6;});
		bool rejected=false;try {cancelled.CalcECOperator();} catch (const std::runtime_error& error) {rejected=std::string(error.what())=="FDTD setup cancelled";}
		Check(rejected && !cancelled.GetConstruction(),"cancelled construction published a result");
		for (auto* primitive:primitives) Check(!primitive->GetPrimitiveUsed(),"cancelled construction published used flags");
		cancelled.SetSetupCancellation({});cancelled.ec[0].clear();cancelled.ec[1].clear();cancelled.ec[2].clear();cancelled.ec[3].clear();
		Check(cancelled.CalcECOperator()==0 && cancelled.GetConstruction(),"operator reuse after cancellation failed");
		std::cout << "PASS HIP EC/coefficient/PEC/timestep/used snapshots, setters and cancellation/reuse" << std::endl;
#else
		std::cout << "PASS CPU construction snapshot and coefficient setter (HIP unavailable in this build)" << std::endl;
#endif
		return 0;
	}
	catch (const std::exception& error) {std::cerr << error.what() << std::endl;return 1;}
}
#endif
