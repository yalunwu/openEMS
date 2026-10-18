#include "field_gather_reference.h"
/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "openems_global.h"
extern "C" OPENEMS_EXPORT int openEMS_TestGPUSetupPath();

#ifndef OPENEMS_NATIVE_SETUP_TESTS
int main() {return openEMS_TestGPUSetupPath();}
#else
#include "FDTD/gpu_setup_path.h"
#include "FDTD/gpu_setup_phase_velocity.h"
#include "FDTD/gpu_setup_metrics.h"
#include "FDTD/gpu_setup_multigrid.h"
#include "FDTD/gpu_setup_adapters.h"
#include "FDTD/gpu_setup_gather.h"
#include "FDTD/gpu_operator_setup.h"
#include "FDTD/gpu_backend.h"
#include "FDTD/extensions/operator_ext_steadystate.h"
#include "FDTD/extensions/engine_ext_steadystate.h"
#include "CSPropMaterial.h"
#include "CSPropMetal.h"
#include "FDTD/operator.h"
#include "FDTD/operator_cylinder.h"
#include "FDTD/operator_cylindermultigrid.h"
#include "ContinuousStructure.h"
#include <iostream>
#include <memory>
#include <vector>

struct Path
{
	std::vector<unsigned short> directions;
	std::vector<unsigned int> positions;
	void operator()(int direction, const uint32_t* p)
	{
		directions.push_back(direction);
		positions.insert(positions.end(),p,p+3);
	}
};

static void CheckAdapters(Operator& op)
{
	struct CustomMaterial : CSPropMaterial {explicit CustomMaterial(ParameterSet* p) : CSPropMaterial(p) {}};
	struct CustomSteady : Operator_Ext_SteadyState {explicit CustomSteady(Operator* p) : Operator_Ext_SteadyState(p,1e-9) {}};
	struct CustomEngine : Engine_Ext_SteadyState {explicit CustomEngine(Operator_Ext_SteadyState* p) : Engine_Ext_SteadyState(p) {}};
	ParameterSet parameters;CustomMaterial material(&parameters);CustomSteady steady(&op);
	auto rejected=[](const std::function<void()>& check) {
		try {check();} catch (const std::runtime_error&) {return;}
		throw std::runtime_error("custom adapter rejection was skipped");
	};
	rejected([&]() {GPU_PreparePropertyAdapter(material);});
	rejected([&]() {GPU_PrepareExtensionAdapter(steady);});
	bool allowed=false;
	GPU_RegisterPropertyAdapter(typeid(CustomMaterial),typeid(CSPropMaterial),"test material",
		[&allowed](CSProperties&,std::string& reason) {reason="disabled test adapter";return allowed;});
	GPU_RegisterExtensionAdapter(typeid(CustomSteady),typeid(Operator_Ext_SteadyState),typeid(Engine_Ext_SteadyState),"test steady state",
		[&allowed](Operator_Extension&,std::string& reason) {reason="disabled test adapter";return allowed;});
	struct Cleanup {~Cleanup() {GPU_UnregisterPropertyAdapter(typeid(CustomMaterial));GPU_UnregisterExtensionAdapter(typeid(CustomSteady));}} cleanup;
	rejected([&]() {GPU_PreparePropertyAdapter(material);});
	rejected([&]() {GPU_PrepareExtensionAdapter(steady);});allowed=true;
	GPU_PreparePropertyAdapter(material);
	if (GPU_PrepareExtensionAdapter(steady)!=GPU_SetupExtensionKind::SteadyState) throw std::runtime_error("extension semantics changed");
	Engine_Ext_SteadyState matching(&steady);GPU_CheckEngineAdapter(matching);
	CustomEngine unexpected(&steady);rejected([&]() {GPU_CheckEngineAdapter(unexpected);});
	GPU_UnregisterExtensionAdapter(typeid(CustomSteady));
	GPU_RegisterExtensionAdapter(typeid(CustomSteady),typeid(Operator_Ext_SteadyState),typeid(CustomEngine),"test custom engine",
		[](Operator_Extension&,std::string&) {return true;});
	GPU_CheckEngineAdapter(unexpected);
	std::cout << "PASS explicit property/extension adapters, preparation rejection and matching engine checks" << std::endl;
}

class TestOperator : public Operator
{
public:
	TestOperator() {Init();}
	using Operator::FindPath;
	void SetTestTimestep(double value) {dT=value;}
};

class TestCylinder : public Operator_Cylinder
{
public:
	TestCylinder() {Init();}
	using Operator_Cylinder::GetRawDiscDelta;
	using Operator_Cylinder::FindPath;
};

class TestMultiGrid : public Operator_CylinderMultiGrid
{
public:
	explicit TestMultiGrid(const std::vector<double>& radii=std::vector<double>{12.0}) : Operator_CylinderMultiGrid(radii,0) {setNumThreads(1);Init();}
	using Operator_CylinderMultiGrid::SetupInterpolation;
	using Operator_CylinderMultiGrid::CalcECOperator;
	GPU_SetupInterpolationPoint Reference(unsigned int a,unsigned int n,bool current) const
	{
		GPU_SetupInterpolationPoint result={};
		result.positions[0]=current ? m_interpol_pos_i_2p[n][a] : m_interpol_pos_v_2p[n][a];
		result.positions[1]=current ? m_interpol_pos_i_2pp[n][a] : m_interpol_pos_v_2pp[n][a];
		result.weights[0]=current ? f4_interpol_i_2p[n][a].f[0] : f4_interpol_v_2p[n][a].f[0];
		result.weights[1]=current ? f4_interpol_i_2pp[n][a].f[0] : f4_interpol_v_2pp[n][a].f[0];return result;
	}
};

void CompareMultigridInterpolation(bool closed)
{
	ContinuousStructure structure;structure.SetCoordInputType(CYLINDRICAL);auto* grid=structure.GetGrid();grid->SetDeltaUnit(1e-3);
	for (int i=0;i<=16;++i) grid->AddDiscLine(0,2.0*i);
	for (int i=0;i<=24;++i) grid->AddDiscLine(1,-0.4+(closed ? 2*acos(-1.0) : 3.0)*pow(i/24.0,1.3));
	for (double z:{-2.0,0.0,1.0,4.0}) grid->AddDiscLine(2,z);
	for (int repeat=0;repeat<2;++repeat)
	{
		TestMultiGrid op;if (!op.SetGeometryCSX(&structure)) throw std::runtime_error("multigrid interpolation fixture failed");
		if (grid->GetQtyLines(0)!=17 || grid->GetQtyLines(1)!=25) throw std::runtime_error("multigrid changed the caller's mesh");
		op.SetupInterpolation();const auto* child=op.GetInnerOperator();
		const uint32_t count=op.GetNumberOfLines(1,true),child_count=child->GetNumberOfLines(1,true);
		std::vector<double> base(6ull*count),sub(6ull*child_count);
		for (uint32_t a=0;a<count;++a) {base[6*a]=op.GetDiscLine(1,a);base[6*a+1]=op.GetDiscLine(1,a,true);}
		for (uint32_t a=0;a<child_count;++a) {sub[6*a]=child->GetDiscLine(1,a);sub[6*a+1]=child->GetDiscLine(1,a,true);}
		for (uint32_t a=0;a<count;++a) for (uint32_t n=0;n<2;++n) for (bool current:{false,true})
		{
			const auto expected=op.Reference(a,n,current),actual=GPU_SetupInterpolateAlpha(base.data(),count,sub.data(),child_count,a,n,current,closed);
			for (int k=0;k<2;++k) if (actual.positions[k]!=expected.positions[k] || actual.weights[k]!=expected.weights[k]) throw std::runtime_error("multigrid interpolation differs from CPU float tables");
		}
	}
}

void CompareMultigridTimesteps(bool periodic,bool forced)
{
	ContinuousStructure structure;structure.SetCoordInputType(CYLINDRICAL);auto* grid=structure.GetGrid();grid->SetDeltaUnit(1e-3);
	for (int i=0;i<=32;++i) grid->AddDiscLine(0,2.0*i);
	for (int i=0;i<=48;++i) grid->AddDiscLine(1,-acos(-1.0)+2*acos(-1.0)*i/48);
	for (double z:{-2.0,0.0,1.0,4.0}) grid->AddDiscLine(2,z);
	Excitation excitation;if (periodic) excitation.SetupSinusoidal(1e9);else excitation.SetupGaussianPulse(3e9,2e9);
	TestMultiGrid op(std::vector<double>{12.0,28.0});
	if (!op.SetGeometryCSX(&structure)) throw std::runtime_error("nested timestep fixture failed");
	op.SetExcitationSignal(&excitation);op.SetTimestepFactor(0.8);if (forced) op.SetTimestep(1e-8);
	if (op.CalcECOperator()!=0) throw std::runtime_error("nested timestep construction failed");
	const Operator_Cylinder* child=&op;
	for (;;)
	{
		if (!child->GetTimestepValid() || child->GetTimestep()!=op.GetTimestep()) throw std::runtime_error("multigrid timesteps differ across levels");
		const auto* multigrid=dynamic_cast<const Operator_CylinderMultiGrid*>(child);if (!multigrid) break;
		child=multigrid->GetInnerOperator();
	}
	if (periodic)
	{
		const double samples=excitation.GetSignalPeriod()/op.GetTimestep();
		if (fabs(samples-round(samples))>1e-12*samples) throw std::runtime_error("multigrid timestep does not divide the signal period");
	}
}

#ifdef OPENEMS_CONTRACT_HIP
void CompareDeviceMultigridTables(bool closed)
{
	ContinuousStructure structure;structure.SetCoordInputType(CYLINDRICAL);auto* grid=structure.GetGrid();grid->SetDeltaUnit(1e-3);
	for (int i=0;i<=16;++i) grid->AddDiscLine(0,2.0*i);
	for (int i=0;i<=24;++i) grid->AddDiscLine(1,-.4+(closed ? 2*acos(-1.0) : 3.0)*pow(i/24.0,1.3));
	for (double z:{-2.0,0.0,1.0,4.0}) grid->AddDiscLine(2,z);
	Excitation excitation;excitation.SetupGaussianPulse(3e9,2e9);
	TestMultiGrid cpu;cpu.SetExcitationSignal(&excitation);
	if (!cpu.SetGeometryCSX(&structure) || cpu.CalcECOperator()!=0) throw std::runtime_error("CPU multigrid table snapshot failed");
	TestMultiGrid gpu;gpu.SetExcitationSignal(&excitation);gpu.SetGPUBackend("auto",nullptr,"gpu");
	if (!gpu.SetGeometryCSX(&structure) || gpu.CalcECOperator()!=0 || !gpu.GetConstruction()) throw std::runtime_error("GPU multigrid table snapshot skipped construction");
	GPU_MultiGridInterpolation actual;gpu.MaterializeMultiGrid(actual);
	if (actual.split_pos!=cpu.GetSplitPos()) throw std::runtime_error("device multigrid split differs");
	for (unsigned int a=0;a<cpu.GetNumberOfLines(1,true);++a) for (unsigned int n=0;n<2;++n) for (bool current:{false,true})
	{
		const auto expected=cpu.Reference(a,n,current);
		const auto& p=current ? actual.pos_i_2p[n] : actual.pos_v_2p[n];const auto& pp=current ? actual.pos_i_2pp[n] : actual.pos_v_2pp[n];
		const auto& w=current ? actual.w_i_2p[n] : actual.w_v_2p[n];const auto& wp=current ? actual.w_i_2pp[n] : actual.w_v_2pp[n];
		if (p.at(a)!=expected.positions[0] || pp.at(a)!=expected.positions[1] || w.at(a)!=expected.weights[0] || wp.at(a)!=expected.weights[1]) throw std::runtime_error("device multigrid table snapshot differs");
	}
}
#endif

void CompareCylindricalMetrics(bool closed,bool axis)
{
	ContinuousStructure structure;structure.SetCoordInputType(CYLINDRICAL);
	auto* grid=structure.GetGrid();grid->SetDeltaUnit(1e-3);
	for (double r:{0.0,0.3,0.9,2.0,4.0}) grid->AddDiscLine(0,r+(axis ? 0 : 1.5));
	for (double a:{-0.4,0.0,0.7,1.9,3.4}) grid->AddDiscLine(1,a);
	if (closed) grid->AddDiscLine(1,2*acos(-1.0)-0.4);
	for (double z:{-3.0,-0.5,0.0,0.8,2.0}) grid->AddDiscLine(2,z);
	TestCylinder op;if (!op.SetGeometryCSX(&structure)) throw std::runtime_error("cylindrical metric fixture failed");
	GPU_SetupMetrics metrics={};metrics.unit=op.GetGridDelta();metrics.cylindrical=1;metrics.closed_alpha=op.GetClosedAlpha();metrics.axis=op.GetR0Included();
	std::vector<double> packed;
	for (int n=0;n<3;++n)
	{
		metrics.mesh.dims[n]=op.GetNumberOfLines(n,true);metrics.mesh.offsets[n]=static_cast<uint32_t>(packed.size());
		for (unsigned int i=0;i<metrics.mesh.dims[n];++i)
		{
			packed.push_back(op.GetDiscLine(n,i));packed.push_back(op.GetDiscLine(n,i,true));
			packed.push_back(op.GetRawDiscDelta(n,i));packed.push_back(op.GetRawDiscDelta(n,int(i)-1));
			packed.push_back(op.GetDiscDelta(n,i)*metrics.unit);packed.push_back(op.GetDiscDelta(n,i,true)*metrics.unit);
		}
	}
	metrics.mesh.grid=packed.data();
	for (bool magnetic:{false,true}) for (unsigned int interpolation=0;interpolation<3;++interpolation)
	{
		std::vector<unsigned int> positions[3],reference_positions[3];unsigned int* lines[3];unsigned int counts[3];
		for (int n=0;n<3;++n)
		{
			counts[n]=metrics.mesh.dims[n];
			for (unsigned int p=0;p<counts[n];++p) positions[n].push_back(p);
			reference_positions[n]=positions[n];
			if (n==1 && closed) for (auto& p:reference_positions[n])
			{
				if (interpolation==1 && p==0) p=counts[1]-2;
				else if (interpolation==2 && p==counts[1]-2) p=0;
			}
			lines[n]=reference_positions[n].data();
		}
		std::unique_ptr<Reference_Field_Gather_FDTD> expected(Reference_Field_Gather_FDTD::New(&op,Engine_Interface_Base::InterpolationType(interpolation),nullptr,nullptr,magnetic,counts,lines));
		size_t q=0;
		for (unsigned int r=0;r<counts[0];++r) for (unsigned int a=0;a<counts[1];++a) for (unsigned int z=0;z<counts[2];++z) for (int n=0;n<3;++n)
		{
			int p[3]={int(r),int(a),int(z)};GPU_SetupGatherBuilder builder={metrics,n,magnetic};const auto actual=builder.Build(p,interpolation);
			const auto& reference=expected->entries[q++];
			if (actual.form!=reference.form || actual.rel!=reference.rel) throw std::runtime_error("sampling map interpolation differs");
			for (int k=0;k<4;++k) if (actual.idx[k]!=reference.idx[k] || actual.delta[k]!=reference.delta[k]) throw std::runtime_error("sampling map indices/metrics differ");
		}
	}
	const double points[][3]={{0.0,-0.6,-2},{0.3,-0.4,0},{2.0,0,0.8},{3.0,0.5,2},{4.0,3.3,-0.5},{2.0,5.7,0},{2.0,6.4,1}};
	for (const auto& start:points) for (const auto& stop:points)
	{
		for (bool full:{false,true}) for (bool dual:{false,true}) for (int method=0;method<3;++method)
		{
			unsigned int expectedStart[3]={},expectedStop[3]={},actualStart[3]={},actualStop[3]={};
			const int expected=op.SnapBox2Mesh(start,stop,expectedStart,expectedStop,dual,full,method);
			const int actual=GPU_SetupSnapBox(metrics.mesh,start,stop,actualStart,actualStop,dual,method,true,closed,full);
			if (actual!=expected) throw std::runtime_error("cylindrical snapped dimension differs");
			if (actual>=0) for (int n=0;n<3;++n) if (actualStart[n]!=expectedStart[n] || actualStop[n]!=expectedStop[n]) throw std::runtime_error("cylindrical snapped indices differ");
		}
		double a[3]={start[0],start[1],start[2]},b[3]={stop[0],stop[1],stop[2]};const auto expected=op.FindPath(a,b);
		Path actual;if (!GPU_SetupFindPath(metrics.mesh,start,stop,actual,true,closed)) {actual.directions.clear();actual.positions.clear();}
		if (actual.directions!=expected.dir) throw std::runtime_error("cylindrical curve directions differ");
		for (size_t q=0;q<actual.directions.size();++q) for (int n=0;n<3;++n) if (actual.positions[3*q+n]!=expected.posPath[n][q]) throw std::runtime_error("cylindrical curve positions differ");
	}
	auto same=[](double actual,double expected) {
		if (!std::isfinite(actual) || fabs(actual-expected)>1e-13*std::max(fabs(expected),1e-30)) throw std::runtime_error("cylindrical metric differs");
	};
	for (int r=-1;r<int(metrics.mesh.dims[0]);++r) for (int a=-1;a<int(metrics.mesh.dims[1]);++a) for (int z=-1;z<int(metrics.mesh.dims[2]);++z)
	{
		const int p[3]={r,a,z};
		for (int dual=0;dual<2;++dual) for (int n=0;n<3;++n)
		{
			same(metrics.NodeWidth(n,p,dual),op.GetNodeWidth(n,p,dual));same(metrics.NodeArea(n,p,dual),op.GetNodeArea(n,p,dual));
			if (r>=0 && a>=0 && z>=0)
			{
				unsigned int q[3]={unsigned(r),unsigned(a),unsigned(z)};double expected[3],actual[3];
				same(metrics.EdgeLength(n,p,dual),op.GetEdgeLength(n,q,dual));same(metrics.EdgeArea(n,p,dual),op.GetEdgeArea(n,q,dual));
				if (metrics.Yee(n,p,actual,dual)!=op.GetYeeCoords(n,q,expected,dual)) throw std::runtime_error("cylindrical Yee validity differs");
				for (int k=0;k<3;++k) same(actual[k],expected[k]);
			}
		}
		double actual[3],expected[3];const bool valid=metrics.CellCenter(p,actual);
		if (valid!=op.GetCellCenterMaterialAvgCoord(p,expected)) throw std::runtime_error("cylindrical cell-center validity differs");
		if (valid) for (int n=0;n<3;++n) same(actual[n],expected[n]);
	}
}

int openEMS_TestGPUSetupPath()
{
	try
	{
		ContinuousStructure structure;
		CSRectGrid* grid=structure.GetGrid();
		grid->SetDeltaUnit(1e-3);
		const double lines[]={-5,-3.5,-1,0,0.2,1.3,3,4,6};
		for (int n=0;n<3;++n) for (double line:lines) grid->AddDiscLine(n,line);
		std::unique_ptr<TestOperator> op(new TestOperator);
		if (!op->SetGeometryCSX(&structure)) throw std::runtime_error("path test mesh setup failed");
		GPU_SetupMeshView mesh={}; std::vector<double> packed;
		for (int n=0;n<3;++n)
		{
			mesh.dims[n]=op->GetNumberOfLines(n,true); mesh.offsets[n]=static_cast<uint32_t>(packed.size());
			for (unsigned int p=0;p<mesh.dims[n];++p)
			{
				packed.push_back(op->GetDiscLine(n,p)); packed.push_back(op->GetDiscLine(n,p,true));
				for (int k=0;k<4;++k) packed.push_back(0);
			}
		}
		mesh.grid=packed.data();
		const double points[][3]={{-6,-4,0},{-3.5,-1,0},{0,0,0},{0.1,0.2,0.3},{1.3,3,4},{6,6,6},{8,8,8},{-6,0,0},{8,0,0}};
		for (const auto& start:points) for (const auto& stop:points)
		{
			for (int dual=0;dual<2;++dual) for (int method=0;method<3;++method)
			{
				unsigned int expectedStart[3]={},expectedStop[3]={},actualStart[3]={},actualStop[3]={};
				const int expected=op->SnapBox2Mesh(start,stop,expectedStart,expectedStop,dual!=0,true,method);
				const int actual=GPU_SetupSnapBox(mesh,start,stop,actualStart,actualStop,dual!=0,method);
				if (actual!=expected) throw std::runtime_error("snapped box dimension differs");
				if (actual>=0) for (int n=0;n<3;++n)
					if (expectedStart[n]!=actualStart[n] || expectedStop[n]!=actualStop[n]) throw std::runtime_error("snapped box indices differ");
			}
			double a[3]={start[0],start[1],start[2]},b[3]={stop[0],stop[1],stop[2]};
			Grid_Path reference=op->FindPath(a,b);
			Path path;
			if (!GPU_SetupFindPath(mesh,start,stop,path)) {path.directions.clear();path.positions.clear();}
			if (path.directions!=reference.dir) throw std::runtime_error("curve path directions differ");
			for (size_t edge=0;edge<path.directions.size();++edge) for (int n=0;n<3;++n)
				if (path.positions[3*edge+n]!=reference.posPath[n][edge]) throw std::runtime_error("curve path positions differ");
		}
		CheckAdapters(*op);
		op->SetTestTimestep(1e-13);
		unsigned int start[3]={1,2,0},stop[3]={6,7,8};double spacing[3];
		for (int n=0;n<3;++n) spacing[n]=fabs(op->GetDiscLine(n,start[n])-op->GetDiscLine(n,stop[n]))*op->GetGridDelta()/(stop[n]-start[n]);
		const double directions[][3]={{1,0,0},{0,1,0},{0,0,-1},{0.6,0.8,0},{1/sqrt(3.0),1/sqrt(3.0),1/sqrt(3.0)}};
		for (const auto& direction:directions) for (float frequency:{1e7f,1e9f,1e10f})
		{
			double d[3]={direction[0],direction[1],direction[2]};
			const double expected=op->CalcNumericPhaseVelocity(start,stop,d,frequency);
			const double actual=GPU_SetupPhaseVelocity(spacing,d,frequency,op->GetTimestep(),C0);
			if (!std::isfinite(actual) || fabs(actual-expected)>1e-12*fabs(expected)) throw std::runtime_error("TFSF numerical phase velocity differs");
		}
		std::cout << "PASS Cartesian curve paths, snapping methods and TFSF numerical phase velocity" << std::endl;
		for (bool closed:{false,true}) for (bool axis:{false,true}) CompareCylindricalMetrics(closed,axis);
		std::cout << "PASS cylindrical metrics, annular areas, axis/seam Yee and averaging coordinates" << std::endl;
		CompareMultigridInterpolation(false);CompareMultigridInterpolation(true);
		std::cout << "PASS nonuniform multigrid interpolation and repeated setup without changing input mesh" << std::endl;
		for (bool periodic:{false,true}) for (bool forced:{false,true}) CompareMultigridTimesteps(periodic,forced);
		std::cout << "PASS shared nested timesteps, factors, periodic adjustment and invalid-child rebuilds" << std::endl;
#ifdef OPENEMS_CONTRACT_HIP
		CompareDeviceMultigridTables(false);CompareDeviceMultigridTables(true);
		std::cout << "PASS exact device multigrid split, interpolation indices and float weights" << std::endl;
#endif
		return 0;
	}
	catch (const std::exception& error) {std::cerr << error.what() << std::endl;return 1;}
}
#endif
