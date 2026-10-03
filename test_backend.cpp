/*
 *  Copyright (C) 2026 openEMS Project
 *
 *  Unit tests for pluggable EngineBackend and EngineVulkan
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <memory>
#include <cstdio>
#include <chrono>
#include <limits>

#include "openems.h"
#include "FDTD/engine_backend.h"
#include "FDTD/engine_cpu.h"
#include "FDTD/engine.h"
#include "FDTD/engine_interface_fdtd.h"
#include "FDTD/operator.h"
#include "FDTD/extensions/operator_ext_mur_abc.h"
#include "FDTD/extensions/operator_ext_upml.h"
#include "FDTD/extensions/operator_ext_steadystate.h"
#include "FDTD/extensions/engine_ext_steadystate.h"
#include "FDTD/extensions/operator_ext_tfsf.h"
#include "FDTD/extensions/operator_ext_lumpedRLC.h"
#include "FDTD/extensions/operator_ext_absorbing_bc.h"
#include "FDTD/extensions/operator_ext_dispersive.h"
#include "FDTD/extensions/operator_ext_lorentzmaterial.h"
#include "FDTD/extensions/operator_ext_conductingsheet.h"
#include "FDTD/extensions/operator_ext_cylinder.h"
#include "FDTD/operator_cylinder.h"
#include "FDTD/operator_cylindermultigrid.h"
#include "FDTD/vulkan/engine_vulkan.h"
#include "Common/processing.h"
#include "ContinuousStructure.h"
#include "CSProperties.h"
#include "CSPropExcitation.h"
#include "CSPropLumpedElement.h"
#include "CSPropAbsorbingBC.h"
#include "CSPrimBox.h"
#include "CSPropLorentzMaterial.h"
#include "CSPropDebyeMaterial.h"
#include "CSPropConductingSheet.h"
#include <cmath>

int tests_passed = 0;
int tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
	do { \
		if (!(cond)) { \
			std::cerr << "FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
			tests_failed++; \
			return false; \
		} \
	} while (0)

#define RUN_TEST(fn) \
	do { \
		std::cout << "[RUNNING] " << #fn << "..." << std::endl; \
		if (fn()) { \
			std::cout << "[PASSED]  " << #fn << std::endl; \
			tests_passed++; \
		} else { \
			std::cout << "[FAILED]  " << #fn << std::endl; \
		} \
	} while (0)

class TestEngineInterface : public Engine_Interface_Base
{
public:
	TestEngineInterface() : Engine_Interface_Base(nullptr), timestep(0) {}

	double* GetEField(const unsigned int*, double* out) const override { return out; }
	double* GetHField(const unsigned int*, double* out) const override { return out; }
	double* GetJField(const unsigned int*, double* out) const override { return out; }
	double* GetRotHField(const unsigned int*, double* out) const override { return out; }
	double* GetDField(const unsigned int*, double* out) const override { return out; }
	double* GetBField(const unsigned int*, double* out) const override { return out; }
	double CalcVoltageIntegral(const unsigned int*, const unsigned int*) const override { return 0.0; }
	double GetTime(bool = false) const override { return 0.0; }
	unsigned int GetNumberOfTimesteps() const override { return timestep; }
	double CalcFastEnergy() const override { return 0.0; }

	unsigned int timestep;
};

class TestProcessing : public Processing
{
public:
	explicit TestProcessing(Engine_Interface_Base* engine) : Processing(engine) {}
	std::string GetProcessingName() const override { return "test processing"; }
};

static ContinuousStructure* CreateSimpleGrid()
{
	ContinuousStructure* csx = new ContinuousStructure();
	CSRectGrid* grid = csx->GetGrid();
	grid->SetDeltaUnit(1e-3);

	for (int i = 0; i <= 10; ++i) grid->AddDiscLine(0, -50.0 + i * 10.0);
	for (int i = 0; i <= 10; ++i) grid->AddDiscLine(1, -50.0 + i * 10.0);
	for (int i = 0; i <= 4; ++i)  grid->AddDiscLine(2, -2.0 + i * 1.0);
	return csx;
}

bool Test_BackendInterface_NullOp()
{
	std::string reason;
	bool supported = EngineBackend::CheckModelSupport(nullptr, nullptr, reason);
	TEST_ASSERT(!supported, "Null operator should not be supported");
	TEST_ASSERT(!reason.empty(), "Failure reason should not be empty");
	return true;
}

bool Test_ProcessingTimestepPeekDoesNotConsume()
{
	TestEngineInterface* engine = new TestEngineInterface();
	TestProcessing processing(engine);
	processing.AddStep(5);
	engine->timestep = 5;

	TEST_ASSERT(processing.IsTimestep(), "Scheduled timestep should be due");
	TEST_ASSERT(processing.IsTimestep(), "Peeking must not consume a scheduled timestep");
	TEST_ASSERT(processing.CheckTimestep(), "Processing should consume the scheduled timestep");
	TEST_ASSERT(!processing.IsTimestep(), "Consumed timestep must no longer be due");
	return true;
}

bool Test_CapabilityScanner_StandardModel()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	std::string reason;
	std::unique_ptr<Operator> op(Operator::New());

	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(supported, "Standard grid model should be supported on GPU backend");
	TEST_ASSERT(reason.empty(), "Reason should be empty on success");

	delete csx;
	return true;
}

class DummyUnsupportedExtension : public Operator_Extension
{
public:
	DummyUnsupportedExtension(Operator* op) : Operator_Extension(op) {}
	virtual Operator_Extension* Clone(Operator* op) { return new DummyUnsupportedExtension(op); }
	virtual bool BuildExtension() { return true; }
	virtual Engine_Extension* CreateEngineExtention() { return nullptr; }
	virtual std::string GetExtensionName() const { return "Unsupported Custom Extension"; }
	bool IsCylinderCoordsSave(bool, bool) const override { return true; }
};

class DummyDispersiveExtension : public Operator_Ext_Dispersive
{
public:
	explicit DummyDispersiveExtension(Operator* op) : Operator_Ext_Dispersive(op) {}
	Operator_Extension* Clone(Operator* op) override { return new DummyDispersiveExtension(op); }
	std::string GetExtensionName() const override { return "Unsupported Dispersive Extension"; }
};

bool Test_CapabilityScanner_EngineExtensionFallback()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	std::unique_ptr<Operator> op(Operator::New());
	op->AddExtension(new DummyUnsupportedExtension(op.get()));

	std::string reason;
	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(!supported, "Engine extensions without a Vulkan implementation must be rejected");
	TEST_ASSERT(reason.find("Unsupported Custom Extension") != std::string::npos, "Reason should name the unsupported extension");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_UnknownDispersiveFallback()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	std::unique_ptr<Operator> op(Operator::New());
	op->AddExtension(new DummyDispersiveExtension(op.get()));

	std::string reason;
	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(!supported, "Unknown dispersive extensions must not be silently accepted");
	TEST_ASSERT(reason.find("Unsupported Dispersive Extension") != std::string::npos,
	            "Reason should name the unsupported dispersive extension");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_MurABC_Supported()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	std::unique_ptr<Operator> op(Operator::New());
	op->AddExtension(new Operator_Ext_Mur_ABC(op.get()));

	std::string reason;
	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(supported, "Mur ABC should be supported on Vulkan backend");
	TEST_ASSERT(reason.empty(), "Reason should be empty on success");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_SteadyState_Supported()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	std::unique_ptr<Operator> op(Operator::New());
	op->AddExtension(new Operator_Ext_SteadyState(op.get(), 1e-9));

	std::string reason;
	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(supported, "SteadyState should be supported on Vulkan backend");
	TEST_ASSERT(reason.empty(), "Reason should be empty on success");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_TFSF_Supported()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	std::unique_ptr<Operator> op(Operator::New());
	op->AddExtension(new Operator_Ext_TFSF(op.get()));

	std::string reason;
	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(supported, "TFSF should be supported on Vulkan backend");
	TEST_ASSERT(reason.empty(), "Reason should be empty on success");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_LumpedRLC_Supported()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	std::unique_ptr<Operator> op(Operator::New());
	op->AddExtension(new Operator_Ext_LumpedRLC(op.get()));

	std::string reason;
	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(supported, "Lumped RLC should be supported on Vulkan backend");
	TEST_ASSERT(reason.empty(), "Reason should be empty on success");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_AbsorbingBC_Supported()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	std::unique_ptr<Operator> op(Operator::New());
	op->AddExtension(new Operator_Ext_Absorbing_BC(op.get()));

	std::string reason;
	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(supported, "Local absorbing BC sheets should be supported on Vulkan backend");
	TEST_ASSERT(reason.empty(), "Reason should be empty on success");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_UPML_Supported()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	std::unique_ptr<Operator> op(Operator::New());
	TEST_ASSERT(op->SetGeometryCSX(csx), "Failed to initialize operator geometry");
	int bc[6] = {3, 3, 3, 3, 3, 3};
	unsigned int size[6] = {2, 2, 2, 2, 2, 2};
	Operator_Ext_UPML::Create_UPML(op.get(), bc, size, "");
	TEST_ASSERT(op->GetNumberOfExtentions() == 6, "Expected one UPML extension per boundary");

	std::string reason;
	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(supported, "UPML extensions should be supported on Vulkan backend");
	TEST_ASSERT(reason.empty(), "Reason should be empty on success");

	op.reset();
	delete csx;
	return true;
}

bool Test_CapabilityScanner_LorentzMaterial_Supported()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	CSPropLorentzMaterial* lorentz = new CSPropLorentzMaterial(csx->GetParameterSet());
	lorentz->SetDispersionOrder(1);
	lorentz->SetEpsPlasmaFreq(0, 1e10);
	csx->AddProperty(lorentz);

	std::string reason;
	std::unique_ptr<Operator> op(Operator::New());
	op->AddExtension(new Operator_Ext_LorentzMaterial(op.get()));

	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(supported, "Lorentz material should be supported on Vulkan backend");
	TEST_ASSERT(reason.empty(), "Reason should be empty on success");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_DebyeMaterial_Supported()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	CSPropDebyeMaterial* debye = new CSPropDebyeMaterial(csx->GetParameterSet());
	debye->SetDispersionOrder(1);
	debye->SetEpsDelta(0, 1.0);
	debye->SetEpsRelaxTime(0, 1e-9);
	csx->AddProperty(debye);

	std::string reason;
	std::unique_ptr<Operator> op(Operator::New());
	op->AddExtension(new Operator_Ext_LorentzMaterial(op.get()));

	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(supported, "Debye material should be supported on Vulkan backend");
	TEST_ASSERT(reason.empty(), "Reason should be empty on success");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_ConductingSheet_Supported()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	CSPropConductingSheet* sheet = new CSPropConductingSheet(csx->GetParameterSet());
	sheet->SetConductivity(5.8e7);
	sheet->SetThickness(35e-6);
	csx->AddProperty(sheet);

	std::string reason;
	std::unique_ptr<Operator> op(Operator::New());
	op->AddExtension(new Operator_Ext_ConductingSheet(op.get(), 1e9));

	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(supported, "Conducting sheet should be supported on Vulkan backend");
	TEST_ASSERT(reason.empty(), "Reason should be empty on success");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_Cylinder_Supported()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	std::string reason;
	std::unique_ptr<Operator_Cylinder> op(Operator_Cylinder::New());
	op->AddExtension(new Operator_Ext_Cylinder(op.get()));

	bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
	TEST_ASSERT(supported, "Cylindrical coordinates should be supported on Vulkan backend");
	TEST_ASSERT(reason.empty(), "Reason should be empty on success");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_CylinderMultiGrid_Supported()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	std::string reason;
	std::vector<double> mg = { 10.0, 20.0 };
	std::unique_ptr<Operator_CylinderMultiGrid> op(Operator_CylinderMultiGrid::New(mg));
	if (op)
	{
		bool supported = EngineBackend::CheckModelSupport(op.get(), csx, reason);
		TEST_ASSERT(supported, "Nested cylindrical multi-grid should be supported");
		TEST_ASSERT(reason.empty(), "Supported hierarchy should have no rejection reason");
		auto* child = dynamic_cast<Operator_CylinderMultiGrid*>(op->GetInnerOperator());
		TEST_ASSERT(child != nullptr, "Expected a nested child operator");
		child->GetInnerOperator()->AddExtension(new DummyUnsupportedExtension(child->GetInnerOperator()));
		TEST_ASSERT(!EngineBackend::CheckModelSupport(op.get(), csx, reason), "Unknown innermost extension must reject the hierarchy");
		TEST_ASSERT(reason.find("Unsupported Custom Extension") != std::string::npos, "Reason should identify the unknown child extension");
	}

	delete csx;
	return true;
}

bool Test_EngineVulkan_Lifecycle()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	EngineVulkan engine(nullptr);
	TEST_ASSERT(engine.GetBackendName().find("Vulkan") != std::string::npos, "Backend name should contain Vulkan");
	TEST_ASSERT(engine.GetNumberOfTimesteps() == 0, "Initial timesteps should be 0");

	bool init = engine.Initialize();
	TEST_ASSERT(init, "EngineVulkan initialization should succeed");

	engine.SetVolt(0, 5, 5, 2, 42.0f);
	engine.SetCurr(1, 3, 3, 1, 10.5f);

	TEST_ASSERT(engine.GetVolt(0, 5, 5, 2) == 42.0f, "Set/Get volt mismatch");
	TEST_ASSERT(engine.GetCurr(1, 3, 3, 1) == 10.5f, "Set/Get curr mismatch");

	bool iterated = engine.IterateTS(10);
	TEST_ASSERT(iterated, "IterateTS should succeed");
	TEST_ASSERT(engine.GetNumberOfTimesteps() == 10, "Timestep counter should be 10");

	bool syncedFields = engine.SyncFieldsToHost();
	TEST_ASSERT(syncedFields, "SyncFieldsToHost should succeed");

	bool syncedProbes = engine.SyncProbesToHost();
	TEST_ASSERT(syncedProbes, "SyncProbesToHost should succeed");

	engine.Reset();
	TEST_ASSERT(engine.GetNumberOfTimesteps() == 0, "Timestep counter should be reset to 0");
	return true;
#endif
}

bool Test_Vulkan_CheckedDimensionsAndIndices()
{
	class DimensionOperator : public Operator {
	public:
		explicit DimensionOperator(unsigned int size) : dimension(size) { Init(); }
		unsigned int GetNumberOfLines(int, bool = false) const override { return dimension; }
		std::string GetDirName(int) const override { return ""; }
		void SetMaterialStoreFlags(int, bool) override {}
		void SetBackgroundMaterial(double, double, double, double, double) override {}
	private:
		unsigned int dimension;
	};
	for (unsigned int size : {0u, 65536u, std::numeric_limits<unsigned int>::max()})
	{
		DimensionOperator op(size);
		EngineVulkan gpu(&op);
		TEST_ASSERT(!gpu.Initialize(), "Invalid dimensions must fail before GPU allocation");
		gpu.Reset();
	}
	EngineVulkan gpu(nullptr);
	gpu.SetVolt(0, 0, 0, 0, 2.0f);
	gpu.SetCurr(1, 0, 0, 0, 3.0f);
	gpu.SetVolt(0, 0, 0, 16, 9.0f); // must not alias the next y line
	gpu.SetCurr(0, 0, 16, 0, 9.0f); // must not alias the next x line
	TEST_ASSERT(gpu.GetVolt(0, 0, 0, 0) == 2.0f, "Pre-initialization voltage was lost");
	TEST_ASSERT(gpu.GetCurr(1, 0, 0, 0) == 3.0f, "Pre-initialization current was lost");
	TEST_ASSERT(gpu.GetVolt(0, 0, 1, 0) == 0.0f, "Invalid voltage index aliased another cell");
	TEST_ASSERT(gpu.GetCurr(0, 1, 0, 0) == 0.0f, "Invalid current index aliased another cell");
	TEST_ASSERT(gpu.GetVolt(3, 0, 0, 0) == 0.0f, "Invalid field component must return zero");
	return true;
}

bool Test_OpenEMS_CLIArgument_EngineVulkan()
{
	openEMS fdtd;
	std::vector<std::string> args = {"--engine=vulkan"};
	fdtd.SetLibraryArguments(args);

	std::vector<std::string> argsGpu = {"--engine=gpu"};
	fdtd.SetLibraryArguments(argsGpu);

	return true;
}

static bool RunSimulationWithEngine(const std::string& engineName)
{
	openEMS fdtd;
	std::vector<std::string> args = {"--engine=" + engineName};
	fdtd.SetLibraryArguments(args);
	fdtd.SetNumberOfTimeSteps(15);
	fdtd.SetEnableDumps(false);
	fdtd.SetGaussExcite(1e9, 500e6);

	ContinuousStructure* csx = CreateSimpleGrid();
	fdtd.SetCSX(csx);

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, ("SetupFDTD failed for engine: " + engineName).c_str());

	fdtd.RunFDTD();
	return true;
}

bool Test_EndToEnd_BasicEngine()
{
	return RunSimulationWithEngine("basic");
}

bool Test_EndToEnd_SSEEngine()
{
	return RunSimulationWithEngine("sse");
}

bool Test_EndToEnd_SSECompressedEngine()
{
	return RunSimulationWithEngine("sse-compressed");
}

bool Test_EndToEnd_MultithreadedEngine()
{
	return RunSimulationWithEngine("multithreaded");
}

bool Test_EndToEnd_VulkanEngine()
{
	return RunSimulationWithEngine("vulkan");
}

class TestFDTDAccess : public openEMS {
public:
	Operator* GetOp() { return FDTD_Op; }
	Engine* GetEng() { return FDTD_Eng; }
	EngineBackend* GetBackend() { return m_EngineBackend.get(); }
	Engine_Ext_SteadyState* GetSteadyStateDetector() { return Eng_Ext_SSD; }
};

static ContinuousStructure* CreateCustomGrid(int nx, int ny, int nz)
{
	ContinuousStructure* csx = new ContinuousStructure();
	CSRectGrid* grid = csx->GetGrid();
	grid->SetDeltaUnit(1e-3);

	for (int i = 0; i < nx; ++i) grid->AddDiscLine(0, -20.0 + i * (40.0 / (nx - 1)));
	for (int i = 0; i < ny; ++i) grid->AddDiscLine(1, -20.0 + i * (40.0 / (ny - 1)));
	for (int i = 0; i < nz; ++i) grid->AddDiscLine(2, -20.0 + i * (40.0 / (nz - 1)));
	return csx;
}

static bool RunVulkanSimulation(int nx, int ny, int nz, int timesteps = 15)
{
	openEMS fdtd;
	std::vector<std::string> args = {"--engine=vulkan"};
	fdtd.SetLibraryArguments(args);
	fdtd.SetNumberOfTimeSteps(timesteps);
	fdtd.SetEnableDumps(false);
	fdtd.SetGaussExcite(1e9, 500e6);

	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);
	fdtd.SetCSX(csx);

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed");

	fdtd.RunFDTD();
	return true;
}

bool Test_Vulkan_SubWarpGrid()
{
	// 7 x 9 x 11 grid: dimZ = 11 < 32 (sub-warp in fast dimension)
	return RunVulkanSimulation(7, 9, 11, 15);
}

bool Test_Vulkan_AsymmetricDimensions()
{
	// 15 x 5 x 47 grid: odd and prime non-multiple dimensions
	return RunVulkanSimulation(15, 5, 47, 15);
}

bool Test_Vulkan_ThinPlanarGrid()
{
	// 35 x 35 x 3 grid: extremely thin in Z (3 cells)
	return RunVulkanSimulation(35, 35, 3, 15);
}

bool Test_Vulkan_BoundaryConditions()
{
	openEMS fdtd;
	std::vector<std::string> args = {"--engine=vulkan"};
	fdtd.SetLibraryArguments(args);
	fdtd.SetNumberOfTimeSteps(15);
	fdtd.SetEnableDumps(false);
	fdtd.SetGaussExcite(1e9, 500e6);

	// Set mixed boundary conditions: PEC on x, PMC on y, Mur on z
	fdtd.Set_BC_Type(0, 0); // PEC
	fdtd.Set_BC_Type(1, 0); // PEC
	fdtd.Set_BC_Type(2, 1); // PMC
	fdtd.Set_BC_Type(3, 1); // PMC
	fdtd.Set_BC_Type(4, 2); // Mur ABC
	fdtd.Set_BC_Type(5, 2); // Mur ABC

	ContinuousStructure* csx = CreateCustomGrid(12, 12, 12);
	fdtd.SetCSX(csx);

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed with mixed BCs");

	fdtd.RunFDTD();
	return true;
}

bool Test_Vulkan_ZeroExcitationGeometry()
{
	openEMS fdtd;
	std::vector<std::string> args = {"--engine=vulkan"};
	fdtd.SetLibraryArguments(args);
	fdtd.SetNumberOfTimeSteps(10);
	fdtd.SetEnableDumps(false);
	fdtd.SetGaussExcite(1e9, 500e6);

	// csx has grid lines but zero excitation properties or boxes
	ContinuousStructure* csx = CreateCustomGrid(10, 10, 10);
	fdtd.SetCSX(csx);

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed without excitation");

	fdtd.RunFDTD();
	return true;
}

bool Test_Vulkan_ResetLifecycleMultiRun()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	EngineVulkan engine(nullptr);
	bool init1 = engine.Initialize();
	TEST_ASSERT(init1, "Initial Initialize() failed");

	engine.SetVolt(0, 2, 2, 2, 5.0f);
	bool it1 = engine.IterateTS(10);
	TEST_ASSERT(it1, "First IterateTS failed");
	TEST_ASSERT(engine.GetNumberOfTimesteps() == 10, "Timesteps should be 10");

	// Reset
	engine.Reset();
	TEST_ASSERT(engine.GetNumberOfTimesteps() == 0, "Timesteps should be 0 after reset");

	// Re-initialize and run again on same object
	bool init2 = engine.Initialize();
	TEST_ASSERT(init2, "Second Initialize() after reset failed");

	engine.SetVolt(0, 2, 2, 2, 10.0f);
	bool it2 = engine.IterateTS(15);
	TEST_ASSERT(it2, "Second IterateTS failed");
	TEST_ASSERT(engine.GetNumberOfTimesteps() == 15, "Timesteps should be 15");

	return true;
#endif
}

bool Test_Vulkan_NumericalEquivalence_Asymmetric()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	int nx = 13, ny = 11, nz = 23;
	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan"});
	fdtd.SetNumberOfTimeSteps(25);
	fdtd.SetGaussExcite(1e9, 500e6);
	fdtd.SetCSX(csx);
	fdtd.SetEnableDumps(false);

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed");

	Engine* cpuEng = fdtd.GetEng();
	EngineBackend* gpuEng = fdtd.GetBackend();
	TEST_ASSERT(cpuEng != nullptr, "CPU engine is null");
	TEST_ASSERT(gpuEng != nullptr, "GPU backend is null");
	TEST_ASSERT(dynamic_cast<EngineVulkan*>(gpuEng) != nullptr, "Vulkan initialization fell back to CPU");

	// Set initial impulse at center (Ez component)
	int cx = nx / 2, cy = ny / 2, cz = nz / 2;
	cpuEng->SetVolt(2, cx, cy, cz, 1.0f);
	gpuEng->SetVolt(2, cx, cy, cz, 1.0f);

	for (int step = 1; step <= 25; ++step)
	{
		cpuEng->IterateTS(1);
		gpuEng->IterateTS(1);
	}

	std::vector<float> cpuVolt(3 * nx * ny * nz);
	size_t cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
		cpuVolt[cpuIdx++] = cpuEng->GetVolt(n, x, y, z);

	gpuEng->SyncFieldsToHost();

	float maxDiff = 0.0f;
	cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
	{
		float vCpu = cpuVolt[cpuIdx++];
		float vGpu = gpuEng->GetVolt(n, x, y, z);
		TEST_ASSERT(std::isfinite(vGpu), "GPU field contains a non-finite value");
		float d = std::abs(vCpu - vGpu);
		if (d > maxDiff) maxDiff = d;
	}

	TEST_ASSERT(maxDiff < 1e-5f, ("GPU vs CPU field difference exceeded tolerance: " + std::to_string(maxDiff)).c_str());
	return true;
#endif
}

bool Test_Vulkan_UPML_Equivalence()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	int nx = 25, ny = 25, nz = 25;
	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan"});
	fdtd.SetNumberOfTimeSteps(25);
	fdtd.SetGaussExcite(1e9, 500e6);
	fdtd.SetCSX(csx);
	fdtd.SetEnableDumps(false);

	for (int n = 0; n < 6; ++n)
		fdtd.Set_BC_PML(n, 4);

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed with UPML");

	Engine* cpuEng = fdtd.GetEng();
	EngineBackend* gpuEng = fdtd.GetBackend();
	TEST_ASSERT(cpuEng != nullptr, "CPU engine is null");
	TEST_ASSERT(gpuEng != nullptr, "GPU backend is null");
	TEST_ASSERT(dynamic_cast<EngineVulkan*>(gpuEng) != nullptr, "Vulkan initialization fell back to CPU");

	// Initial impulse at center (Ez component)
	int cx = nx / 2, cy = ny / 2, cz = nz / 2;
	cpuEng->SetVolt(2, cx, cy, cz, 1.0f);
	gpuEng->SetVolt(2, cx, cy, cz, 1.0f);

	for (int step = 1; step <= 25; ++step)
	{
		cpuEng->IterateTS(1);
		gpuEng->IterateTS(1);
	}

	std::vector<float> cpuVolt(3 * nx * ny * nz);
	size_t cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
		cpuVolt[cpuIdx++] = cpuEng->GetVolt(n, x, y, z);

	gpuEng->SyncFieldsToHost();

	float maxDiff = 0.0f;
	cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
	{
		float vCpu = cpuVolt[cpuIdx++];
		float vGpu = gpuEng->GetVolt(n, x, y, z);
		TEST_ASSERT(std::isfinite(vGpu), "GPU field contains a non-finite value");
		float d = std::abs(vCpu - vGpu);
		if (d > maxDiff) maxDiff = d;
	}

	TEST_ASSERT(maxDiff < 1e-4f, ("GPU vs CPU field difference with UPML exceeded tolerance: " + std::to_string(maxDiff)).c_str());
	return true;
#endif
}

bool Test_Vulkan_MurABC_Equivalence()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	int nx = 21, ny = 21, nz = 21;
	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan"});
	fdtd.SetNumberOfTimeSteps(25);
	fdtd.SetGaussExcite(1e9, 500e6);
	fdtd.SetCSX(csx);
	fdtd.SetEnableDumps(false);

	// Set Mur ABC on all 6 boundaries
	for (int n = 0; n < 6; ++n)
		fdtd.Set_BC_Type(n, 2);

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed with Mur ABC");

	Engine* cpuEng = fdtd.GetEng();
	EngineBackend* gpuEng = fdtd.GetBackend();
	TEST_ASSERT(cpuEng != nullptr, "CPU engine is null");
	TEST_ASSERT(gpuEng != nullptr, "GPU backend is null");
	TEST_ASSERT(dynamic_cast<EngineVulkan*>(gpuEng) != nullptr, "Vulkan initialization fell back to CPU");

	// Initial impulse at center (Ez component)
	int cx = nx / 2, cy = ny / 2, cz = nz / 2;
	cpuEng->SetVolt(2, cx, cy, cz, 1.0f);
	gpuEng->SetVolt(2, cx, cy, cz, 1.0f);

	for (int step = 1; step <= 25; ++step)
	{
		cpuEng->IterateTS(1);
		gpuEng->IterateTS(1);
	}

	std::vector<float> cpuVolt(3 * nx * ny * nz);
	size_t cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
		cpuVolt[cpuIdx++] = cpuEng->GetVolt(n, x, y, z);

	gpuEng->SyncFieldsToHost();

	float maxDiff = 0.0f;
	cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
	{
		float vCpu = cpuVolt[cpuIdx++];
		float vGpu = gpuEng->GetVolt(n, x, y, z);
		TEST_ASSERT(std::isfinite(vGpu), "GPU field contains a non-finite value");
		float d = std::abs(vCpu - vGpu);
		if (d > maxDiff) maxDiff = d;
	}

	TEST_ASSERT(maxDiff < 1e-4f, ("GPU vs CPU field difference with Mur ABC exceeded tolerance: " + std::to_string(maxDiff)).c_str());
	return true;
#endif
}

bool Test_Vulkan_SteadyState_Execution()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	int nx = 15, ny = 15, nz = 15;
	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);
	CSPropExcitation* exc = new CSPropExcitation(csx->GetParameterSet());
	exc->SetExcitType(0);
	exc->SetExcitation(1.0, 2);
	CSPrimBox* excBox = new CSPrimBox(csx->GetParameterSet(), exc);
	excBox->SetCoord(0, -4.0);
	excBox->SetCoord(1,  4.0);
	excBox->SetCoord(2, -4.0);
	excBox->SetCoord(3,  4.0);
	excBox->SetCoord(4, -4.0);
	excBox->SetCoord(5,  4.0);
	csx->AddProperty(exc);

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan"});
	fdtd.SetNumberOfTimeSteps(160);
	fdtd.SetSinusExcite(10e9);
	fdtd.SetCSX(csx);
	fdtd.SetEnableDumps(false);

	// Set Mur ABC boundaries
	for (int n = 0; n < 6; ++n)
		fdtd.Set_BC_Type(n, 2);

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed with SinusExcite / SteadyState");

	EngineBackend* gpuEng = fdtd.GetBackend();
	TEST_ASSERT(gpuEng != nullptr, "GPU backend is null");
	TEST_ASSERT(dynamic_cast<EngineVulkan*>(gpuEng) != nullptr, "Vulkan initialization fell back to CPU");
	Engine_Ext_SteadyState* detector = fdtd.GetSteadyStateDetector();
	TEST_ASSERT(detector != nullptr, "Steady-state detector was not created");
	TEST_ASSERT(detector->GetTSPeriod() > 0, "Steady-state detector period is zero");

	fdtd.RunFDTD();
	TEST_ASSERT(gpuEng->GetNumberOfTimesteps() >= 2 * detector->GetTSPeriod(),
	            "Simulation did not execute enough timesteps to compare periods");
	TEST_ASSERT(std::isfinite(detector->GetLastDiff()), "Steady-state detector produced a non-finite result");
	TEST_ASSERT(detector->GetLastDiff() < 1.0, "Steady-state detector did not consume live Vulkan probe data");
	return true;
#endif
}

bool Test_Vulkan_TFSF_Equivalence()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	int nx = 21, ny = 21, nz = 21;
	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);

	CSPropExcitation* exc = new CSPropExcitation(csx->GetParameterSet());
	exc->SetExcitType(10); // plane wave
	exc->SetPropagationDir(1.0, 0); // x-direction
	exc->SetExcitation(1.0, 2); // Ez polarization
	CSPrimBox* box = new CSPrimBox(csx->GetParameterSet(), exc);
	box->SetCoord(0, -12.0);
	box->SetCoord(1,  12.0);
	box->SetCoord(2, -12.0);
	box->SetCoord(3,  12.0);
	box->SetCoord(4, -12.0);
	box->SetCoord(5,  12.0);
	csx->AddProperty(exc);

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan"});
	fdtd.SetNumberOfTimeSteps(25);
	fdtd.SetGaussExcite(1e9, 500e6);
	fdtd.SetCSX(csx);
	fdtd.SetEnableDumps(false);

	// Mur ABC on outer boundaries so the wave propagates cleanly
	for (int n = 0; n < 6; ++n)
		fdtd.Set_BC_Type(n, 2);

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed with TFSF plane wave");

	Engine* cpuEng = fdtd.GetEng();
	EngineBackend* gpuEng = fdtd.GetBackend();
	TEST_ASSERT(cpuEng != nullptr, "CPU engine is null");
	TEST_ASSERT(gpuEng != nullptr, "GPU backend is null");
	TEST_ASSERT(dynamic_cast<EngineVulkan*>(gpuEng) != nullptr, "Vulkan initialization fell back to CPU");

	for (int step = 1; step <= 25; ++step)
	{
		cpuEng->IterateTS(1);
		gpuEng->IterateTS(1);
	}

	std::vector<float> cpuVolt(3 * nx * ny * nz);
	size_t cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
		cpuVolt[cpuIdx++] = cpuEng->GetVolt(n, x, y, z);

	gpuEng->SyncFieldsToHost();

	float maxDiff = 0.0f;
	float maxVal = 0.0f;
	cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
	{
		float vCpu = cpuVolt[cpuIdx++];
		float vGpu = gpuEng->GetVolt(n, x, y, z);
		TEST_ASSERT(std::isfinite(vGpu), "GPU field contains a non-finite value");
		if (std::abs(vCpu) > maxVal) maxVal = std::abs(vCpu);
		float d = std::abs(vCpu - vGpu);
		if (d > maxDiff) maxDiff = d;
	}

	TEST_ASSERT(maxVal > 0.0f, "Incident plane wave produced zero field in CPU simulation");
	TEST_ASSERT(maxDiff < 1e-4f, ("GPU vs CPU field difference with TFSF exceeded tolerance: " + std::to_string(maxDiff)).c_str());
	return true;
#endif
}

bool Test_Vulkan_LumpedRLC_Equivalence()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	int nx = 21, ny = 21, nz = 21;
	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);

	CSPropLumpedElement* rlc = new CSPropLumpedElement(csx->GetParameterSet());
	rlc->SetDirection(2); // z
	rlc->SetLEtype(CSPropLumpedElement::PARALLEL);
	rlc->SetResistance(50.0);
	rlc->SetInductance(1e-9);
	rlc->SetCapacity(1e-12);
	rlc->SetCaps(false);

	CSPrimBox* box = new CSPrimBox(csx->GetParameterSet(), rlc);
	box->SetCoord(0, -0.5);
	box->SetCoord(1,  0.5);
	box->SetCoord(2, -0.5);
	box->SetCoord(3,  0.5);
	box->SetCoord(4, -0.5);
	box->SetCoord(5,  2.5);
	csx->AddProperty(rlc);

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan"});
	fdtd.SetNumberOfTimeSteps(25);
	fdtd.SetGaussExcite(1e9, 500e6);
	fdtd.SetCSX(csx);
	fdtd.SetEnableDumps(false);

	for (int n = 0; n < 6; ++n)
		fdtd.Set_BC_Type(n, 2);

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed with Lumped RLC");

	Engine* cpuEng = fdtd.GetEng();
	EngineBackend* gpuEng = fdtd.GetBackend();
	TEST_ASSERT(cpuEng != nullptr, "CPU engine is null");
	TEST_ASSERT(gpuEng != nullptr, "GPU backend is null");
	TEST_ASSERT(dynamic_cast<EngineVulkan*>(gpuEng) != nullptr, "Vulkan initialization fell back to CPU");

	int cx = nx / 2, cy = ny / 2, cz = nz / 2;
	cpuEng->SetVolt(2, cx, cy, cz, 1.0f);
	gpuEng->SetVolt(2, cx, cy, cz, 1.0f);

	for (int step = 1; step <= 25; ++step)
	{
		cpuEng->IterateTS(1);
		gpuEng->IterateTS(1);
	}

	std::vector<float> cpuVolt(3 * nx * ny * nz);
	size_t cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
		cpuVolt[cpuIdx++] = cpuEng->GetVolt(n, x, y, z);

	gpuEng->SyncFieldsToHost();

	float maxDiff = 0.0f;
	float maxVal = 0.0f;
	cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
	{
		float vCpu = cpuVolt[cpuIdx++];
		float vGpu = gpuEng->GetVolt(n, x, y, z);
		TEST_ASSERT(std::isfinite(vGpu), "GPU field contains a non-finite value");
		if (std::abs(vCpu) > maxVal) maxVal = std::abs(vCpu);
		float d = std::abs(vCpu - vGpu);
		if (d > maxDiff) maxDiff = d;
	}

	std::cout << "LumpedRLC maxDiff: " << maxDiff << ", maxVal: " << maxVal << std::endl;
	TEST_ASSERT(maxVal > 0.0f, "Simulation produced zero field");
	TEST_ASSERT(maxDiff < 1e-4f, ("GPU vs CPU field difference with Lumped RLC exceeded tolerance: " + std::to_string(maxDiff)).c_str());
	return true;
#endif
}

bool Test_Vulkan_AbsorbingBC_Equivalence()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	int nx = 21, ny = 21, nz = 21;
	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);

	CSPropAbsorbingBC* abc = new CSPropAbsorbingBC(csx->GetParameterSet());
	abc->SetAbsorbingBoundaryType(CSPropAbsorbingBC::MUR_1ST_SA);
	abc->SetNormalSignPositive(true);
	abc->SetPhaseVelocity(3e8);

	CSPrimBox* box = new CSPrimBox(csx->GetParameterSet(), abc);
	box->SetCoord(0, -20.0);
	box->SetCoord(1,  20.0);
	box->SetCoord(2, -20.0);
	box->SetCoord(3,  20.0);
	box->SetCoord(4,  10.0);
	box->SetCoord(5,  10.0);
	csx->AddProperty(abc);

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan"});
	fdtd.SetNumberOfTimeSteps(25);
	fdtd.SetGaussExcite(1e9, 500e6);
	fdtd.SetCSX(csx);
	fdtd.SetEnableDumps(false);

	for (int n = 0; n < 6; ++n)
		fdtd.Set_BC_Type(n, 2);

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed with Absorbing BC sheet");

	Engine* cpuEng = fdtd.GetEng();
	EngineBackend* gpuEng = fdtd.GetBackend();
	TEST_ASSERT(cpuEng != nullptr, "CPU engine is null");
	TEST_ASSERT(gpuEng != nullptr, "GPU backend is null");
	TEST_ASSERT(dynamic_cast<EngineVulkan*>(gpuEng) != nullptr, "Vulkan initialization fell back to CPU");

	int cx = nx / 2, cy = ny / 2, cz = nz / 2;
	cpuEng->SetVolt(2, cx, cy, cz, 1.0f);
	gpuEng->SetVolt(2, cx, cy, cz, 1.0f);

	for (int step = 1; step <= 25; ++step)
	{
		cpuEng->IterateTS(1);
		gpuEng->IterateTS(1);
	}

	std::vector<float> cpuVolt(3 * nx * ny * nz);
	size_t cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
		cpuVolt[cpuIdx++] = cpuEng->GetVolt(n, x, y, z);

	gpuEng->SyncFieldsToHost();

	float maxDiff = 0.0f;
	float maxVal = 0.0f;
	cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
	{
		float vCpu = cpuVolt[cpuIdx++];
		float vGpu = gpuEng->GetVolt(n, x, y, z);
		TEST_ASSERT(std::isfinite(vGpu), "GPU field contains a non-finite value");
		if (std::abs(vCpu) > maxVal) maxVal = std::abs(vCpu);
		float d = std::abs(vCpu - vGpu);
		if (d > maxDiff) maxDiff = d;
	}

	std::cout << "AbsorbingBC maxDiff: " << maxDiff << ", maxVal: " << maxVal << std::endl;
	TEST_ASSERT(maxVal > 0.0f, "Simulation produced zero field");
	TEST_ASSERT(maxDiff < 1e-4f, ("GPU vs CPU field difference with Absorbing BC sheet exceeded tolerance: " + std::to_string(maxDiff)).c_str());
	return true;
#endif
}

bool Test_Vulkan_LorentzMaterial_Equivalence()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	int nx = 21, ny = 21, nz = 21;
	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);

	CSPropLorentzMaterial* lorentz = new CSPropLorentzMaterial(csx->GetParameterSet());
	lorentz->SetEpsilon(2.0);
	lorentz->SetDispersionOrder(1);
	lorentz->SetEpsPlasmaFreq(0, 2e9);
	lorentz->SetEpsLorPoleFreq(0, 1e9);
	lorentz->SetEpsRelaxTime(0, 1e-9);

	CSPrimBox* box = new CSPrimBox(csx->GetParameterSet(), lorentz);
	box->SetCoord(0, -5.0);
	box->SetCoord(1,  5.0);
	box->SetCoord(2, -5.0);
	box->SetCoord(3,  5.0);
	box->SetCoord(4, -5.0);
	box->SetCoord(5,  5.0);
	csx->AddProperty(lorentz);

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan"});
	fdtd.SetNumberOfTimeSteps(25);
	fdtd.SetGaussExcite(1e9, 500e6);
	fdtd.SetCSX(csx);
	fdtd.SetEnableDumps(false);

	for (int n = 0; n < 6; ++n)
		fdtd.Set_BC_Type(n, 0); // PEC

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed with Lorentz material");

	Engine* cpuEng = fdtd.GetEng();
	EngineBackend* gpuEng = fdtd.GetBackend();
	TEST_ASSERT(cpuEng != nullptr, "CPU engine is null");
	TEST_ASSERT(gpuEng != nullptr, "GPU backend is null");
	TEST_ASSERT(dynamic_cast<EngineVulkan*>(gpuEng) != nullptr, "Vulkan initialization fell back to CPU");

	int cx = nx / 2, cy = ny / 2, cz = nz / 2;
	cpuEng->SetVolt(2, cx, cy, cz, 1.0f);
	gpuEng->SetVolt(2, cx, cy, cz, 1.0f);

	for (int step = 1; step <= 25; ++step)
	{
		cpuEng->IterateTS(1);
		gpuEng->IterateTS(1);
	}

	std::vector<float> cpuVolt(3 * nx * ny * nz);
	size_t cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
		cpuVolt[cpuIdx++] = cpuEng->GetVolt(n, x, y, z);

	gpuEng->SyncFieldsToHost();

	float maxDiff = 0.0f;
	float maxVal = 0.0f;
	cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
	{
		float vCpu = cpuVolt[cpuIdx++];
		float vGpu = gpuEng->GetVolt(n, x, y, z);
		TEST_ASSERT(std::isfinite(vGpu), "GPU field contains a non-finite value");
		if (std::abs(vCpu) > maxVal) maxVal = std::abs(vCpu);
		float d = std::abs(vCpu - vGpu);
		if (d > maxDiff) maxDiff = d;
	}

	std::cout << "LorentzMaterial maxDiff: " << maxDiff << ", maxVal: " << maxVal << std::endl;
	TEST_ASSERT(maxVal > 0.0f, "Simulation produced zero field");
	TEST_ASSERT(maxDiff < 1e-4f, ("GPU vs CPU field difference with Lorentz material exceeded tolerance: " + std::to_string(maxDiff)).c_str());
	return true;
#endif
}

bool Test_Vulkan_DebyeMaterial_Equivalence()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	int nx = 21, ny = 21, nz = 21;
	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);

	CSPropDebyeMaterial* debye = new CSPropDebyeMaterial(csx->GetParameterSet());
	debye->SetEpsilon(2.0);
	debye->SetDispersionOrder(1);
	debye->SetEpsDelta(0, 3.0);
	debye->SetEpsRelaxTime(0, 1e-9);

	CSPrimBox* box = new CSPrimBox(csx->GetParameterSet(), debye);
	box->SetCoord(0, -5.0);
	box->SetCoord(1,  5.0);
	box->SetCoord(2, -5.0);
	box->SetCoord(3,  5.0);
	box->SetCoord(4, -5.0);
	box->SetCoord(5,  5.0);
	csx->AddProperty(debye);

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan"});
	fdtd.SetNumberOfTimeSteps(25);
	fdtd.SetGaussExcite(1e9, 500e6);
	fdtd.SetCSX(csx);
	fdtd.SetEnableDumps(false);

	for (int n = 0; n < 6; ++n)
		fdtd.Set_BC_Type(n, 0); // PEC

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed with Debye material");

	Engine* cpuEng = fdtd.GetEng();
	EngineBackend* gpuEng = fdtd.GetBackend();
	TEST_ASSERT(cpuEng != nullptr, "CPU engine is null");
	TEST_ASSERT(gpuEng != nullptr, "GPU backend is null");
	TEST_ASSERT(dynamic_cast<EngineVulkan*>(gpuEng) != nullptr, "Vulkan initialization fell back to CPU");

	int cx = nx / 2, cy = ny / 2, cz = nz / 2;
	cpuEng->SetVolt(2, cx, cy, cz, 1.0f);
	gpuEng->SetVolt(2, cx, cy, cz, 1.0f);

	for (int step = 1; step <= 25; ++step)
	{
		cpuEng->IterateTS(1);
		gpuEng->IterateTS(1);
	}

	std::vector<float> cpuVolt(3 * nx * ny * nz);
	size_t cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
		cpuVolt[cpuIdx++] = cpuEng->GetVolt(n, x, y, z);

	gpuEng->SyncFieldsToHost();

	float maxDiff = 0.0f;
	float maxVal = 0.0f;
	cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
	{
		float vCpu = cpuVolt[cpuIdx++];
		float vGpu = gpuEng->GetVolt(n, x, y, z);
		TEST_ASSERT(std::isfinite(vGpu), "GPU field contains a non-finite value");
		if (std::abs(vCpu) > maxVal) maxVal = std::abs(vCpu);
		float d = std::abs(vCpu - vGpu);
		if (d > maxDiff) maxDiff = d;
	}

	std::cout << "DebyeMaterial maxDiff: " << maxDiff << ", maxVal: " << maxVal << std::endl;
	TEST_ASSERT(maxVal > 0.0f, "Simulation produced zero field");
	TEST_ASSERT(maxDiff < 1e-4f, ("GPU vs CPU field difference with Debye material exceeded tolerance: " + std::to_string(maxDiff)).c_str());
	return true;
#endif
}

bool Test_Vulkan_ConductingSheet_Equivalence()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	int nx = 21, ny = 21, nz = 21;
	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);

	CSPropConductingSheet* sheet = new CSPropConductingSheet(csx->GetParameterSet());
	sheet->SetConductivity(1e5);
	sheet->SetThickness(50e-6);

	CSPrimBox* box = new CSPrimBox(csx->GetParameterSet(), sheet);
	box->SetCoord(0, -10.0);
	box->SetCoord(1,  10.0);
	box->SetCoord(2, -10.0);
	box->SetCoord(3,  10.0);
	box->SetCoord(4,   0.0);
	box->SetCoord(5,   0.0);
	csx->AddProperty(sheet);

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan"});
	fdtd.SetNumberOfTimeSteps(25);
	fdtd.SetGaussExcite(1e9, 500e6);
	fdtd.SetCSX(csx);
	fdtd.SetEnableDumps(false);

	for (int n = 0; n < 6; ++n)
		fdtd.Set_BC_Type(n, 0); // PEC

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed with Conducting sheet");

	Engine* cpuEng = fdtd.GetEng();
	EngineBackend* gpuEng = fdtd.GetBackend();
	TEST_ASSERT(cpuEng != nullptr, "CPU engine is null");
	TEST_ASSERT(gpuEng != nullptr, "GPU backend is null");
	TEST_ASSERT(dynamic_cast<EngineVulkan*>(gpuEng) != nullptr, "Vulkan initialization fell back to CPU");

	int cx = nx / 2, cy = ny / 2, cz = nz / 2;
	cpuEng->SetVolt(2, cx, cy, cz, 1.0f);
	gpuEng->SetVolt(2, cx, cy, cz, 1.0f);

	for (int step = 1; step <= 25; ++step)
	{
		cpuEng->IterateTS(1);
		gpuEng->IterateTS(1);
	}

	std::vector<float> cpuVolt(3 * nx * ny * nz);
	size_t cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
		cpuVolt[cpuIdx++] = cpuEng->GetVolt(n, x, y, z);

	gpuEng->SyncFieldsToHost();

	float maxDiff = 0.0f;
	float maxVal = 0.0f;
	cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
	{
		float vCpu = cpuVolt[cpuIdx++];
		float vGpu = gpuEng->GetVolt(n, x, y, z);
		TEST_ASSERT(std::isfinite(vGpu), "GPU field contains a non-finite value");
		if (std::abs(vCpu) > maxVal) maxVal = std::abs(vCpu);
		float d = std::abs(vCpu - vGpu);
		if (d > maxDiff) maxDiff = d;
	}

	std::cout << "ConductingSheet maxDiff: " << maxDiff << ", maxVal: " << maxVal << std::endl;
	TEST_ASSERT(maxVal > 0.0f, "Simulation produced zero field");
	TEST_ASSERT(maxDiff < 1e-4f, ("GPU vs CPU field difference with Conducting sheet exceeded tolerance: " + std::to_string(maxDiff)).c_str());
	return true;
#endif
}

static ContinuousStructure* CreateCylindricalGrid(int nr, int nalpha, int nz)
{
	ContinuousStructure* csx = new ContinuousStructure();
	CSRectGrid* grid = csx->GetGrid();
	grid->SetDeltaUnit(1e-3);

	for (int i = 0; i < nr; ++i) grid->AddDiscLine(0, i * (20.0 / (nr - 1)));
	for (int i = 0; i < nalpha; ++i) grid->AddDiscLine(1, -M_PI + i * (2.0 * M_PI / (nalpha - 1)));
	for (int i = 0; i < nz; ++i) grid->AddDiscLine(2, i * (20.0 / (nz - 1)));
	return csx;
}

// Keep malformed-table testing internal, without exposing interpolation tables
// through the simulation API or deriving from non-exported CPU implementation.
struct VulkanMultigridTestAccess {
	static unsigned int& Index(Operator_CylinderMultiGrid& op) { return op.m_interpol_pos_v_2p[0][0]; }
	static float& Coefficient(Operator_CylinderMultiGrid& op) { return op.f4_interpol_i_2pp[1][0].f[0]; }
};

bool Test_Vulkan_Multigrid_InvalidMetadata()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan", "--numThreads=1"});
	fdtd.SetCylinderCoords(true);
	fdtd.SetupCylinderMultiGrid(std::vector<double>{10.0});
	fdtd.SetNumberOfTimeSteps(16);
	fdtd.SetGaussExcite(20e9, 10e9);
	fdtd.SetCSX(CreateCylindricalGrid(25, 17, 9));
	fdtd.SetEnableDumps(false);
	for (int n = 0; n < 6; ++n) fdtd.Set_BC_Type(n, 0);
	TEST_ASSERT(fdtd.SetupFDTD() == 0, "Metadata fixture setup failed");
	auto* op = dynamic_cast<Operator_CylinderMultiGrid*>(fdtd.GetOp());
	TEST_ASSERT(op != nullptr, "Metadata fixture multigrid operator missing");
	EngineVulkan gpu(op);
	const unsigned int originalIndex = VulkanMultigridTestAccess::Index(*op);
	VulkanMultigridTestAccess::Index(*op) = op->GetInnerOperator()->GetNumberOfLines(1, true);
	TEST_ASSERT(!gpu.Initialize(), "Out-of-range interpolation index must reject Vulkan");
	gpu.Reset(); // failed initialization and explicit reset must both be safe
	VulkanMultigridTestAccess::Index(*op) = originalIndex;
	const float originalCoefficient = VulkanMultigridTestAccess::Coefficient(*op);
	VulkanMultigridTestAccess::Coefficient(*op) = std::numeric_limits<float>::quiet_NaN();
	TEST_ASSERT(!gpu.Initialize(), "Non-finite interpolation coefficient must reject Vulkan");
	VulkanMultigridTestAccess::Coefficient(*op) = originalCoefficient;
	TEST_ASSERT(gpu.Initialize(), "Valid hierarchy must initialize after failed attempts");
	TEST_ASSERT(gpu.IterateTS(3) && gpu.SyncFieldsToHost(), "Reinitialized hierarchy did not execute");
	return true;
#endif
}

bool Test_Vulkan_Cylinder_Equivalence()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	int nr = 15, nalpha = 15, nz = 15;
	ContinuousStructure* csx = CreateCylindricalGrid(nr, nalpha, nz);

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan"});
	fdtd.SetCylinderCoords(true);
	fdtd.SetNumberOfTimeSteps(25);
	fdtd.SetGaussExcite(1e9, 500e6);
	fdtd.SetCSX(csx);
	fdtd.SetEnableDumps(false);

	for (int n = 0; n < 6; ++n)
		fdtd.Set_BC_Type(n, 0); // PEC

	int ec = fdtd.SetupFDTD();
	TEST_ASSERT(ec == 0, "SetupFDTD failed with Cylindrical coordinates");

	Engine* cpuEng = fdtd.GetEng();
	EngineBackend* gpuEng = fdtd.GetBackend();
	TEST_ASSERT(cpuEng != nullptr, "CPU engine is null");
	TEST_ASSERT(gpuEng != nullptr, "GPU backend is null");
	TEST_ASSERT(dynamic_cast<EngineVulkan*>(gpuEng) != nullptr, "Vulkan initialization fell back to CPU");

	unsigned int actualNr = fdtd.GetOp()->GetNumberOfLines(0, true);
	unsigned int actualNa = fdtd.GetOp()->GetNumberOfLines(1, true);
	unsigned int actualNz = fdtd.GetOp()->GetNumberOfLines(2, true);

	int cr = actualNr / 2, ca = actualNa / 2, cz = actualNz / 2;
	cpuEng->SetVolt(2, cr, ca, cz, 1.0f);
	gpuEng->SetVolt(2, cr, ca, cz, 1.0f);
	cpuEng->SetVolt(2, 0, 0, cz, 1.0f);
	gpuEng->SetVolt(2, 0, 0, cz, 1.0f);

	for (int step = 1; step <= 25; ++step)
	{
		cpuEng->IterateTS(1);
		gpuEng->IterateTS(1);
	}

	std::vector<float> cpuVolt(3 * actualNr * actualNa * actualNz);
	size_t cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (unsigned int x = 0; x < actualNr; ++x)
	for (unsigned int y = 0; y < actualNa; ++y)
	for (unsigned int z = 0; z < actualNz; ++z)
		cpuVolt[cpuIdx++] = cpuEng->GetVolt(n, x, y, z);

	gpuEng->SyncFieldsToHost();

	float maxDiff = 0.0f;
	float maxVal = 0.0f;
	cpuIdx = 0;
	for (int n = 0; n < 3; ++n)
	for (unsigned int x = 0; x < actualNr; ++x)
	for (unsigned int y = 0; y < actualNa; ++y)
	for (unsigned int z = 0; z < actualNz; ++z)
	{
		float vCpu = cpuVolt[cpuIdx++];
		float vGpu = gpuEng->GetVolt(n, x, y, z);
		TEST_ASSERT(std::isfinite(vGpu), "GPU field contains a non-finite value");
		if (std::abs(vCpu) > maxVal) maxVal = std::abs(vCpu);
		float d = std::abs(vCpu - vGpu);
		if (d > maxDiff) maxDiff = d;
	}

	std::cout << "Cylinder maxDiff: " << maxDiff << ", maxVal: " << maxVal << std::endl;
	TEST_ASSERT(maxVal > 0.0f, "Simulation produced zero field");
	TEST_ASSERT(maxDiff < 1e-4f, ("GPU vs CPU field difference with Cylindrical coordinates exceeded tolerance: " + std::to_string(maxDiff)).c_str());
	return true;
#endif
}

static bool RunMultigridEquivalence(const std::vector<double>& splits, bool openAlpha = false,
                                    bool boundaries = false, bool debye = false, bool reinitialize = false,
                                    bool benchmark = false)
{
#ifndef ENABLE_VULKAN
	return true;
#else
	const int angularLines = benchmark || splits.size() >= 4 ? 129 : 33;
	ContinuousStructure* csx = CreateCylindricalGrid(41, angularLines, 17);
	CSRectGrid* grid = csx->GetGrid();
	grid->ClearLines(0);
	for (int r = 0; r <= 40; ++r) grid->AddDiscLine(0, r);
	if (openAlpha)
	{
		grid->ClearLines(1);
		// Nonuniform angular spacing exercises the CPU interpolation coefficients.
		for (int a = 0; a < angularLines; ++a)
		{
			double t = static_cast<double>(a) / (angularLines - 1);
			grid->AddDiscLine(1, -1.2 + 2.4 * t * t);
		}
	}
	auto* excitation = new CSPropExcitation(csx->GetParameterSet());
	excitation->SetNumber(0);
	excitation->SetExcitType(0);
	excitation->SetExcitation(1.0, 2);
	auto* excBox = new CSPrimBox(csx->GetParameterSet(), excitation);
	const double bounds[6] = {0.0, 36.0, -0.4, 0.4, 8.0, 12.0};
	for (int i = 0; i < 6; ++i) excBox->SetCoord(i, bounds[i]);
	csx->AddProperty(excitation);
	if (debye)
	{
		auto* material = new CSPropDebyeMaterial(csx->GetParameterSet());
		material->SetEpsilon(2.0);
		material->SetDispersionOrder(1);
		material->SetEpsDelta(0, 3.0);
		material->SetEpsRelaxTime(0, 1e-10);
		auto* box = new CSPrimBox(csx->GetParameterSet(), material);
		const double region[6] = {1.0, 36.0, -0.8, 0.8, 5.0, 15.0};
		for (int i = 0; i < 6; ++i) box->SetCoord(i, region[i]);
		csx->AddProperty(material);
	}

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan", "--numThreads=1"});
	fdtd.SetCylinderCoords(true);
	fdtd.SetupCylinderMultiGrid(splits);
	fdtd.SetNumberOfTimeSteps(80);
	fdtd.SetGaussExcite(20e9, 10e9);
	fdtd.SetCSX(csx);
	fdtd.SetEnableDumps(false);
	for (int i = 0; i < 6; ++i) fdtd.Set_BC_Type(i, 0);
	if (boundaries)
	{
		fdtd.Set_BC_PML(1, 4); // outer radial UPML
		fdtd.Set_BC_PML(4, 3); // child axial UPML
		fdtd.Set_BC_Type(5, 2); // child axial Mur
	}
	TEST_ASSERT(fdtd.SetupFDTD() == 0, "Cylindrical multigrid setup failed");
	auto* gpu = dynamic_cast<EngineVulkan*>(fdtd.GetBackend());
	TEST_ASSERT(gpu != nullptr, "Multigrid Vulkan initialization fell back to CPU");
	std::vector<Operator*> levels;
	for (Operator* op = fdtd.GetOp(); op;)
	{
		levels.push_back(op);
		auto* mg = dynamic_cast<Operator_CylinderMultiGrid*>(op);
		op = mg ? mg->GetInnerOperator() : nullptr;
	}
	TEST_ASSERT(levels.size() == splits.size() + 1, "Missing multigrid level");
	std::vector<std::unique_ptr<TestProcessing>> levelDumps;
	for (size_t level = 0; level < levels.size(); ++level)
	{
		levelDumps.emplace_back(new TestProcessing(fdtd.NewEngineInterface(static_cast<int>(level))));
		levelDumps.back()->AddStep(27);
	}
	if (reinitialize)
	{
		gpu->Reset();
		TEST_ASSERT(gpu->Initialize(), "Multigrid reinitialization failed");
	}

	// Distinct batches check projection and continuation, including every child.
	const unsigned int batches[3] = {1, 7, 19};
	unsigned int completed = 0;
	for (unsigned int batch : batches)
	{
		TEST_ASSERT(fdtd.GetEng()->IterateTS(batch), "CPU multigrid iteration failed");
		std::vector<std::vector<float>> reference(levels.size());
		for (size_t level = 0; level < levels.size(); ++level)
		{
			Operator* op = levels[level];
			Engine* cpu = op->GetEngine();
			TEST_ASSERT(cpu != nullptr, "Child CPU engine missing");
			for (unsigned int n = 0; n < 3; ++n)
			for (unsigned int r = 0; r < op->GetNumberOfLines(0, true); ++r)
			for (unsigned int a = 0; a < op->GetNumberOfLines(1, true); ++a)
			for (unsigned int z = 0; z < op->GetNumberOfLines(2, true); ++z)
			{
				reference[level].push_back(cpu->GetVolt(n, r, a, z));
				reference[level].push_back(cpu->GetCurr(n, r, a, z));
			}
		}
		TEST_ASSERT(gpu->IterateTS(batch), "GPU multigrid iteration failed");
		completed += batch;
		for (Operator* op : levels)
			TEST_ASSERT(op->GetEngine()->GetNumberOfTimesteps() == completed, "Child timestep was not propagated");
		if (completed == 27)
			for (const auto& dump : levelDumps)
				TEST_ASSERT(dump->IsTimestep(), "MultiGridLevel processing did not reach its scheduled timestep");
		TEST_ASSERT(gpu->SyncFieldsToHost(), "Multigrid field synchronization failed");
		for (size_t level = 0; level < levels.size(); ++level)
		{
			Operator* op = levels[level];
			Engine* cpu = op->GetEngine();
			size_t index = 0;
			float maxError = 0.0f, peak = 0.0f;
			for (unsigned int n = 0; n < 3; ++n)
			for (unsigned int r = 0; r < op->GetNumberOfLines(0, true); ++r)
			for (unsigned int a = 0; a < op->GetNumberOfLines(1, true); ++a)
			for (unsigned int z = 0; z < op->GetNumberOfLines(2, true); ++z)
			{
				const float values[2] = {cpu->GetVolt(n, r, a, z), cpu->GetCurr(n, r, a, z)};
				for (float value : values)
				{
					TEST_ASSERT(std::isfinite(value), "Non-finite multigrid GPU field");
					float expected = reference[level][index++];
					peak = std::max(peak, std::abs(expected));
					maxError = std::max(maxError, std::abs(expected - value));
				}
			}
			std::cout << "Multigrid level " << level << " batch " << batch
			          << " maxError=" << maxError << " peak=" << peak << std::endl;
			TEST_ASSERT(completed == 1u || peak > 0.0f, "Multigrid level produced zero fields");
			TEST_ASSERT(maxError < 1e-4f, "Multigrid absolute field error exceeded 1e-4");
			TEST_ASSERT(maxError <= peak * 0.001f, "Multigrid relative field error exceeded 0.1%");
		}
	}
	if (benchmark)
	{
		const unsigned int steps = 1024;
		const auto start = std::chrono::steady_clock::now();
		TEST_ASSERT(gpu->IterateTS(steps) && gpu->SyncFieldsToHost(), "Multigrid benchmark execution failed");
		const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
		std::cout << "BENCHMARK levels=" << levels.size() << " steps=" << steps
		          << " seconds=" << seconds << " timesteps/s=" << steps / seconds
		          << " MCells/s=" << fdtd.GetOp()->GetNumberCells() * steps / seconds / 1e6 << std::endl;
	}
	if (reinitialize)
	{
		gpu->Reset();
		TEST_ASSERT(gpu->Initialize() && gpu->SyncFieldsToHost(), "Used hierarchy did not reset and reinitialize");
		for (Operator* op : levels)
		{
			Engine* cpu = op->GetEngine();
			TEST_ASSERT(cpu->GetNumberOfTimesteps() == 0u, "Child timestep did not reset");
			for (unsigned int n = 0; n < 3; ++n)
			for (unsigned int r = 0; r < op->GetNumberOfLines(0, true); ++r)
			for (unsigned int a = 0; a < op->GetNumberOfLines(1, true); ++a)
			for (unsigned int z = 0; z < op->GetNumberOfLines(2, true); ++z)
				TEST_ASSERT(cpu->GetVolt(n, r, a, z) == 0.0f && cpu->GetCurr(n, r, a, z) == 0.0f,
				            "Reinitialized child retained fields from the previous run");
		}
		TEST_ASSERT(gpu->IterateTS(8) && gpu->SyncFieldsToHost(), "Reinitialized hierarchy failed to advance");
	}
	return true;
#endif
}

bool Test_Vulkan_Multigrid_ClosedAlpha() { return RunMultigridEquivalence({24.0}); }
bool Test_Vulkan_Multigrid_Nested() { return RunMultigridEquivalence({12.0, 24.0}); }
bool Test_Vulkan_Multigrid_OpenAlpha() { return RunMultigridEquivalence({12.0, 24.0}, true); }
bool Test_Vulkan_Multigrid_Boundaries() { return RunMultigridEquivalence({12.0, 24.0}, false, true); }
bool Test_Vulkan_Multigrid_Debye() { return RunMultigridEquivalence({12.0, 24.0}, false, false, true); }
bool Test_Vulkan_Multigrid_Reset() { return RunMultigridEquivalence({12.0, 24.0}, false, false, false, true); }
bool Test_Vulkan_Multigrid_FiveLevels() { return RunMultigridEquivalence({8.0, 16.0, 24.0, 32.0}); }

bool Benchmark_Vulkan_Multigrid()
{
	return RunMultigridEquivalence({}, false, false, false, false, true) &&
	       RunMultigridEquivalence({24.0}, false, false, false, false, true) &&
	       RunMultigridEquivalence({8.0, 16.0, 24.0, 32.0}, false, false, false, false, true);
}

int main(int argc, char* argv[])
{
	if (argc > 1 && std::string(argv[1]) == "--multigrid-benchmark")
	{
		RUN_TEST(Benchmark_Vulkan_Multigrid);
		return tests_failed ? 1 : 0;
	}
	std::cout << "========================================" << std::endl;
	std::cout << " openEMS EngineBackend & Vulkan Test Suite" << std::endl;
	std::cout << "========================================" << std::endl;

	RUN_TEST(Test_BackendInterface_NullOp);
	RUN_TEST(Test_ProcessingTimestepPeekDoesNotConsume);
	RUN_TEST(Test_CapabilityScanner_StandardModel);
	RUN_TEST(Test_CapabilityScanner_EngineExtensionFallback);
	RUN_TEST(Test_CapabilityScanner_UnknownDispersiveFallback);
	RUN_TEST(Test_CapabilityScanner_UPML_Supported);
	RUN_TEST(Test_CapabilityScanner_MurABC_Supported);
	RUN_TEST(Test_CapabilityScanner_SteadyState_Supported);
	RUN_TEST(Test_CapabilityScanner_TFSF_Supported);
	RUN_TEST(Test_CapabilityScanner_LumpedRLC_Supported);
	RUN_TEST(Test_CapabilityScanner_AbsorbingBC_Supported);
	RUN_TEST(Test_CapabilityScanner_LorentzMaterial_Supported);
	RUN_TEST(Test_CapabilityScanner_DebyeMaterial_Supported);
	RUN_TEST(Test_CapabilityScanner_ConductingSheet_Supported);
	RUN_TEST(Test_CapabilityScanner_Cylinder_Supported);
	RUN_TEST(Test_CapabilityScanner_CylinderMultiGrid_Supported);
	RUN_TEST(Test_EngineVulkan_Lifecycle);
	RUN_TEST(Test_OpenEMS_CLIArgument_EngineVulkan);

	std::cout << "----------------------------------------" << std::endl;
	std::cout << " Baseline End-to-End Simulation Tests" << std::endl;
	std::cout << "----------------------------------------" << std::endl;
	RUN_TEST(Test_EndToEnd_BasicEngine);
	RUN_TEST(Test_EndToEnd_SSEEngine);
	RUN_TEST(Test_EndToEnd_SSECompressedEngine);
	RUN_TEST(Test_EndToEnd_MultithreadedEngine);
	RUN_TEST(Test_EndToEnd_VulkanEngine);

	std::cout << "----------------------------------------" << std::endl;
	std::cout << " Grid Sizes & Corner Cases Tests" << std::endl;
	std::cout << "----------------------------------------" << std::endl;
	RUN_TEST(Test_Vulkan_SubWarpGrid);
	RUN_TEST(Test_Vulkan_AsymmetricDimensions);
	RUN_TEST(Test_Vulkan_ThinPlanarGrid);
	RUN_TEST(Test_Vulkan_BoundaryConditions);
	RUN_TEST(Test_Vulkan_ZeroExcitationGeometry);
	RUN_TEST(Test_Vulkan_ResetLifecycleMultiRun);
	RUN_TEST(Test_Vulkan_NumericalEquivalence_Asymmetric);
	RUN_TEST(Test_Vulkan_UPML_Equivalence);
	RUN_TEST(Test_Vulkan_MurABC_Equivalence);
	RUN_TEST(Test_Vulkan_SteadyState_Execution);
	RUN_TEST(Test_Vulkan_TFSF_Equivalence);
	RUN_TEST(Test_Vulkan_LumpedRLC_Equivalence);
	RUN_TEST(Test_Vulkan_AbsorbingBC_Equivalence);
	RUN_TEST(Test_Vulkan_LorentzMaterial_Equivalence);
	RUN_TEST(Test_Vulkan_DebyeMaterial_Equivalence);
	RUN_TEST(Test_Vulkan_ConductingSheet_Equivalence);
	RUN_TEST(Test_Vulkan_Cylinder_Equivalence);
	RUN_TEST(Test_Vulkan_CheckedDimensionsAndIndices);
	RUN_TEST(Test_Vulkan_Multigrid_InvalidMetadata);
	RUN_TEST(Test_Vulkan_Multigrid_ClosedAlpha);
	RUN_TEST(Test_Vulkan_Multigrid_Nested);
	RUN_TEST(Test_Vulkan_Multigrid_OpenAlpha);
	RUN_TEST(Test_Vulkan_Multigrid_Boundaries);
	RUN_TEST(Test_Vulkan_Multigrid_Debye);
	RUN_TEST(Test_Vulkan_Multigrid_Reset);
	RUN_TEST(Test_Vulkan_Multigrid_FiveLevels);

	std::cout << "========================================" << std::endl;
	std::cout << "Tests completed: " << tests_passed << " passed, " << tests_failed << " failed." << std::endl;
	std::cout << "========================================" << std::endl;

	std::remove("et");
	std::remove("ht");

	return (tests_failed == 0) ? 0 : 1;
}
