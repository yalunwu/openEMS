/*
 *  Copyright (C) 2026 openEMS Project
 *
 *  Unit tests for pluggable EngineBackend and EngineVulkan
 */

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <memory>
#include <cstdio>
#include <chrono>
#include <limits>
#include <sstream>
#include <stdexcept>
#include "test_vulkan_performance.h"
#include "tools/denormal.h"

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
#include "FDTD/extensions/operator_ext_debyematerial.h"
#include "FDTD/extensions/operator_ext_conductingsheet.h"
#include "FDTD/extensions/operator_ext_cylinder.h"
#include "FDTD/operator_cylinder.h"
#include "FDTD/operator_cylindermultigrid.h"
#include "FDTD/vulkan/engine_vulkan.h"
#include "Common/processing.h"
#include "ContinuousStructure.h"
#include "CSProperties.h"
#include "CSPropExcitation.h"
#include "CSPropProbeBox.h"
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
	op->AddExtension(new Operator_Ext_DebyeMaterial(op.get()));

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
	Engine_Interface_Base* Interface() { return NewEngineInterface(); }
	ProcessingArray* GetProcessings() { return PA; }
	Engine_Ext_SteadyState* GetSteadyStateDetector() { return Eng_Ext_SSD; }
	void ReplaceBackend(std::unique_ptr<EngineBackend> backend) { m_EngineBackend = std::move(backend); }
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

bool RunDebyeMaterialEquivalence(unsigned int order)
{
#ifndef ENABLE_VULKAN
	return true;
#else
	int nx = 21, ny = 21, nz = 21;
	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);

	CSPropDebyeMaterial* debye = new CSPropDebyeMaterial(csx->GetParameterSet());
	debye->SetEpsilon(2.0);
	debye->SetDispersionOrder(order);
	for (unsigned int o = 0; o < order; ++o)
	{
		debye->SetEpsDelta(o, o == 0 ? 3.0 : 20.0);
		debye->SetEpsRelaxTime(o, 1e-9 / (o + 1));
	}

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

bool Test_Vulkan_DebyeMaterial_Equivalence() { return RunDebyeMaterialEquivalence(1); }
bool Test_Vulkan_DebyeMaterial_MultiPole() { return RunDebyeMaterialEquivalence(3); }

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
	fdtd.SetNumberOfTimeSteps(160);
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
	double unsupportedEnergy = 0;
	TEST_ASSERT(!gpu->SupportsFastEnergy() && !gpu->GetFastEnergy(unsupportedEnergy), "Multigrid SSE energy must retain the CPU calculation");
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
	const unsigned int batches[] = {1, 7, 19, 33, 65};
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

bool Test_Vulkan_BatchingAndProfiling()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	for (unsigned int size : {1u, 8u, 16u, 32u, 64u})
	{
		ContinuousStructure* csx = CreateCustomGrid(35, 23, 41);
		auto* excitation = new CSPropExcitation(csx->GetParameterSet());
		excitation->SetExcitType(0);
		excitation->SetExcitation(1.0, 2);
		excitation->SetDelay(5e-12);
		auto* box = new CSPrimBox(csx->GetParameterSet(), excitation);
		for (int i = 0; i < 6; ++i) box->SetCoord(i, i % 2 ? 2.0 : -2.0);
		csx->AddProperty(excitation);
		TestFDTDAccess fdtd;
		fdtd.SetLibraryArguments({"--engine=vulkan", "--numThreads=1", "--vulkan-profile",
		                          "--vulkan-batch-size=" + std::to_string(size)});
		fdtd.SetCSX(csx);
		fdtd.SetNumberOfTimeSteps(200);
		fdtd.SetGaussExcite(20e9, 10e9);
		fdtd.SetEnableDumps(false);
		fdtd.Set_BC_Type(0, 2);
		fdtd.Set_BC_PML(3, 4);
		TEST_ASSERT(fdtd.SetupFDTD() == 0, "Batching setup failed");
		auto* gpu = dynamic_cast<EngineVulkan*>(fdtd.GetBackend());
		TEST_ASSERT(gpu && gpu->GetBatchSize() == size && gpu->IsProfilingEnabled(), "Vulkan options were not applied");
		TEST_ASSERT(!gpu->SetBatchSize(0) && !gpu->SetBatchSize(65), "Invalid batch sizes must be rejected");
		TEST_ASSERT(gpu->ClearProfile(), "Profile reset failed");
		const unsigned int batches[] = {0, 1, 7, 33, 65};
		unsigned int completed = 0;
		for (unsigned int count : batches)
		{
			TEST_ASSERT(fdtd.GetEng()->IterateTS(count), "CPU reference failed");
			std::vector<float> reference;
			for (unsigned int n = 0; n < 3; ++n)
			for (unsigned int x = 0; x < 35; ++x)
			for (unsigned int y = 0; y < 23; ++y)
			for (unsigned int z = 0; z < 41; ++z) {
				reference.push_back(fdtd.GetEng()->GetVolt(n, x, y, z));
				reference.push_back(fdtd.GetEng()->GetCurr(n, x, y, z));
			}
			TEST_ASSERT(gpu->IterateTS(count) && gpu->Synchronize(), "Batched stepping failed");
			completed += count;
			const auto profile = gpu->GetProfile();
			TEST_ASSERT(profile.timesteps == count && profile.submissions == (count + size - 1u) / size, "Unexpected timestep submission count");
			TEST_ASSERT(profile.downloadedBytes == 0, "Stepping unexpectedly downloaded fields");
			TEST_ASSERT(std::isfinite(profile.recordSeconds) && std::isfinite(profile.waitSeconds), "Non-finite CPU profiling result");
			if (profile.gpuSamples[EngineVulkan::ProfileBatch]) {
				TEST_ASSERT(profile.gpuSamples[EngineVulkan::ProfileBatch] == profile.submissions, "Timestamp results were lost between submissions");
				TEST_ASSERT(std::isfinite(profile.gpuSeconds[EngineVulkan::ProfileBatch]) && profile.gpuSeconds[EngineVulkan::ProfileBatch] > 0, "Invalid GPU timing result");
			}
			TEST_ASSERT(gpu->SyncFieldsToHost(), "Field readback failed");
			size_t index = 0;
			float error = 0, peak = 0;
			for (unsigned int n = 0; n < 3; ++n)
			for (unsigned int x = 0; x < 35; ++x)
			for (unsigned int y = 0; y < 23; ++y)
			for (unsigned int z = 0; z < 41; ++z) {
				for (float value : {gpu->GetVolt(n, x, y, z), gpu->GetCurr(n, x, y, z)}) {
					TEST_ASSERT(std::isfinite(value), "Non-finite batched field");
					peak = std::max(peak, std::abs(reference[index]));
					error = std::max(error, std::abs(value - reference[index++]));
				}
			}
			std::cout << "Batch size=" << size << " count=" << count << " maxError=" << error << " peak=" << peak << std::endl;
			TEST_ASSERT(error < 1e-4f && error <= peak * 0.001f, "Batched fields differ from CPU by more than 1e-4 / 0.1%");
			if (count > 1u) TEST_ASSERT(peak > 0.0f, "Excited reference field is zero");
			TEST_ASSERT(gpu->GetNumberOfTimesteps() == completed, "Published timestep is incorrect");
			// Readback has its own submissions; start a fresh measurement interval.
			TEST_ASSERT(gpu->ClearProfile(), "Could not clear readback statistics");
		}
		const unsigned int ts = gpu->GetNumberOfTimesteps();
		TEST_ASSERT(!gpu->IterateTS(UINT32_MAX) && gpu->GetNumberOfTimesteps() == ts, "Overflow advanced the hierarchy");
		gpu->SetVolt(2, 17, 11, 20, 0.125f);
		fdtd.GetEng()->SetVolt(2, 17, 11, 20, 0.125f);
		TEST_ASSERT(fdtd.GetEng()->IterateTS(1), "Edited CPU iteration failed");
		const float expected = fdtd.GetEng()->GetVolt(2, 17, 11, 20);
		TEST_ASSERT(gpu->IterateTS(1) && gpu->SyncFieldsToHost(), "Edited GPU iteration failed");
		TEST_ASSERT(std::abs(gpu->GetVolt(2, 17, 11, 20) - expected) < 1e-4f, "Field edits were not uploaded before stepping");
		TEST_ASSERT(gpu->GetProfile().uploadedBytes == 2ull * 3u * 35u * 23u * 41u * sizeof(float), "Edited field upload count is incorrect");
		TEST_ASSERT(gpu->SetProfilingEnabled(false) && gpu->SetProfilingEnabled(true), "Profiling toggle failed");
		TEST_ASSERT(gpu->IterateTS(5), "Pending reset setup failed");
		gpu->Reset();
		TEST_ASSERT(gpu->Initialize() && gpu->GetNumberOfTimesteps() == 0, "Pending batch did not reset");
		TEST_ASSERT(gpu->GetVolt(2, 17, 11, 20) == 0, "Reset retained edited fields");
	}
	return true;
#endif
}

bool Test_Vulkan_EnergyReduction()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	for (bool fp64 : {true, false}) {
		TestFDTDAccess fdtd;
		fdtd.SetLibraryArguments({"--engine=basic", "--numThreads=1"});
		fdtd.SetCSX(CreateCustomGrid(35, 23, 41));
		fdtd.SetNumberOfTimeSteps(64);
		fdtd.SetGaussExcite(20e9, 10e9);
		fdtd.SetEnableDumps(false);
		TEST_ASSERT(fdtd.SetupFDTD() == 0, "Energy fixture setup failed");
		EngineVulkan gpu(fdtd.GetOp());
		TEST_ASSERT(gpu.SetEnergyFloat64Enabled(fp64) && gpu.Initialize(), "Energy backend setup failed");
		TEST_ASSERT(gpu.SupportsFastEnergy(), "Basic energy capability unavailable");
		TEST_ASSERT(gpu.SetProfilingEnabled(true), "Energy profiling failed");
		std::unique_ptr<Engine_Interface_Base> reference(fdtd.Interface());
		double energy = -1;
		TEST_ASSERT(gpu.GetFastEnergy(energy) && energy == 0, "Zero fields must have zero energy");
		for (unsigned int n = 0; n < 3; ++n)
		for (unsigned int x = 0; x < 35; ++x)
		for (unsigned int y = 0; y < 23; ++y)
		for (unsigned int z = 0; z < 41; ++z) {
			const unsigned int hash = (n * 7919u + x * 104729u + y * 8191u + z * 131u) * 2654435761u;
			const float v = std::ldexp(float(hash % 1023u) / 1023.0f, int(hash % 40u) - 20);
			const float h = std::ldexp(float((hash / 1023u) % 1023u) / 1023.0f, int((hash / 41u) % 40u) - 20);
			gpu.SetVolt(n, x, y, z, v);
			gpu.SetCurr(n, x, y, z, h);
			fdtd.GetEng()->SetVolt(n, x, y, z, v);
			fdtd.GetEng()->SetCurr(n, x, y, z, h);
		}
		TEST_ASSERT(gpu.ClearProfile() && gpu.GetFastEnergy(energy), "Edited energy reduction failed");
		const double expected = reference->CalcFastEnergy();
		const double relative = std::abs(energy - expected) / expected;
		std::cout << "Energy " << (fp64 ? "FP64" : "FP32") << " relative error=" << relative << std::endl;
		TEST_ASSERT(relative < (fp64 ? 1e-12 : 3e-7), "Energy reduction precision exceeded tolerance");
		const auto first = gpu.GetProfile();
		TEST_ASSERT(first.downloadedBytes == 0 && first.energyBytes > 0 && first.energyBytes < 35u * 23u * 41u,
		            "Energy check downloaded full fields");
		TEST_ASSERT(gpu.GetFastEnergy(energy) && gpu.GetProfile().submissions == first.submissions, "Energy cache submitted work twice");
		gpu.SetCurr(1, 10, 9, 8, 1e8f);
		fdtd.GetEng()->SetCurr(1, 10, 9, 8, 1e8f);
		TEST_ASSERT(gpu.GetFastEnergy(energy), "Current edit did not invalidate energy");
		TEST_ASSERT(std::abs(energy - reference->CalcFastEnergy()) / reference->CalcFastEnergy() < (fp64 ? 1e-12 : 3e-7), "Edited energy stale");
		if (!fp64) {
			for (unsigned int z = 0; z < 40; ++z) gpu.SetCurr(0, 1, 1, z, 1e19f);
			TEST_ASSERT(!gpu.GetFastEnergy(energy), "FP32 partial overflow must request CPU fallback");
			TEST_ASSERT(gpu.SyncFieldsToHost() && std::isfinite(reference->CalcFastEnergy()), "Partial overflow lost finite CPU energy");
		}
		for (float value : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
			gpu.SetVolt(0, 1, 1, 1, value);
			TEST_ASSERT(!gpu.GetFastEnergy(energy), "Non-finite energy must request CPU fallback");
			TEST_ASSERT(gpu.SyncFieldsToHost() && !std::isfinite(reference->CalcFastEnergy()), "CPU non-finite fallback changed behavior");
		}
		TEST_ASSERT(gpu.SetReadbackOptimizationsEnabled(false) && !gpu.GetFastEnergy(energy), "Reference mode still used GPU energy");
		gpu.Reset();
		TEST_ASSERT(gpu.Initialize() && gpu.SetReadbackOptimizationsEnabled(true) && gpu.GetFastEnergy(energy) && energy == 0, "Reset retained energy cache");
		TEST_ASSERT(gpu.ClearProfile() && gpu.IterateTS(1) && gpu.SyncFieldsToHost() && gpu.GetProfile().submissions == 3,
		            "Single-field staging fallback did not retain separate copies");
		{
#if BOOST_ARCH_X86
			struct RestoreDenormals { unsigned int previous = _mm_getcsr(); ~RestoreDenormals() { _mm_setcsr(previous); } } restore;
			_mm_setcsr(restore.previous & ~0x8040u);
#endif
			gpu.SetCurr(0, 1, 1, 1, 1e-20f);
			TEST_ASSERT(!gpu.GetFastEnergy(energy), "Subnormal-scale energy must retain CPU floating-point behavior");
			TEST_ASSERT(gpu.SyncFieldsToHost(), "Tiny energy fallback synchronization failed");
			TEST_ASSERT(std::isfinite(reference->CalcFastEnergy()), "Tiny CPU energy is non-finite");
#if BOOST_ARCH_X86
			TEST_ASSERT(reference->CalcFastEnergy() > 0, "Tiny CPU energy lost preserved subnormal products");
#endif
		}
	}
	TestFDTDAccess sse;
	sse.SetLibraryArguments({"--engine=sse", "--numThreads=1"});
	sse.SetCSX(CreateCustomGrid(7, 5, 9));
	sse.SetNumberOfTimeSteps(64);
	sse.SetGaussExcite(20e9, 10e9);
	sse.SetEnableDumps(false);
	TEST_ASSERT(sse.SetupFDTD() == 0, "SSE energy fallback fixture failed");
	EngineVulkan gpu(sse.GetOp());
	double energy = 0;
	TEST_ASSERT(gpu.Initialize() && !gpu.SupportsFastEnergy() && !gpu.GetFastEnergy(energy), "SSE energy must retain its CPU accumulation");
	gpu.SetCurr(2, 1, 1, 8, 0.5f); // SSE includes the final z vector, unlike basic energy.
	std::unique_ptr<Engine_Interface_Base> reference(sse.Interface());
	TEST_ASSERT(gpu.SyncFieldsToHost() && std::abs(reference->CalcFastEnergy() - MUE0 * 0.25) < 1e-20, "SSE fallback lost the final z lane");
	return true;
#endif
}

bool Test_Vulkan_EnergyDecay()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	for (bool fp64 : {true, false}) {
		TestFDTDAccess fdtd;
		fdtd.SetLibraryArguments({"--engine=basic", "--numThreads=1"});
		ContinuousStructure* csx = CreateCustomGrid(21, 17, 25);
		auto* material = new CSPropMaterial(csx->GetParameterSet());
		material->SetEpsilon(2.0);
		material->SetKappa(0.1);
		auto* slab = new CSPrimBox(csx->GetParameterSet(), material);
		const double bounds[] = {-20, 20, -20, 20, -20, 20};
		for (unsigned int i = 0; i < 6; ++i) slab->SetCoord(i, bounds[i]);
		csx->AddProperty(material);
		auto* source = new CSPropExcitation(csx->GetParameterSet());
		source->SetExcitType(0);
		source->SetExcitation(1.0, 2);
		auto* box = new CSPrimBox(csx->GetParameterSet(), source);
		const double sourceBounds[] = {-2, 2, -2, 2, -3, 3};
		for (unsigned int i = 0; i < 6; ++i) box->SetCoord(i, sourceBounds[i]);
		csx->AddProperty(source);
		fdtd.SetCSX(csx);
		fdtd.SetGaussExcite(20e9, 10e9);
		fdtd.SetEnableDumps(false);
		fdtd.SetNumberOfTimeSteps(1024);
		for (unsigned int i = 0; i < 6; ++i) fdtd.Set_BC_PML(i, 4);
		TEST_ASSERT(fdtd.SetupFDTD() == 0, "Decay energy fixture failed");
		EngineVulkan gpu(fdtd.GetOp());
		TEST_ASSERT(gpu.SetEnergyFloat64Enabled(fp64) && gpu.Initialize(), "Decay energy backend failed");
		std::unique_ptr<Engine_Interface_Base> reference(fdtd.Interface());
		double maxError = 0, peak = 0, last = 0;
		unsigned int fallbacks = 0;
		for (unsigned int count : {1u, 7u, 23u, 64u, 128u, 256u, 512u}) {
			TEST_ASSERT(fdtd.GetEng()->IterateTS(count), "CPU decay stepping failed");
			const double cpuEnergy = reference->CalcFastEnergy();
			double energy = 0;
			TEST_ASSERT(gpu.IterateTS(count), "GPU decay stepping failed");
			if (!gpu.GetFastEnergy(energy)) {
				TEST_ASSERT(gpu.SyncFieldsToHost(), "Decay energy fallback synchronization failed");
				energy = reference->CalcFastEnergy();
				const double tinyLimit = 3.0 * 21u * 17u * 25u * std::numeric_limits<float>::min() * (EPS0 + MUE0) / (fp64 ? 1e-12 : 3e-7);
				TEST_ASSERT(energy <= tinyLimit, "Ordinary decay energy unexpectedly requested CPU fallback");
				++fallbacks;
			}
			TEST_ASSERT(std::abs(energy - cpuEnergy) <= std::max(energy, cpuEnergy) * 0.001, "CPU/GPU decay energy differs by more than 0.1%");
			TEST_ASSERT(gpu.SyncFieldsToHost(), "Decay reference synchronization failed");
			const double expected = reference->CalcFastEnergy();
			const double error = expected > 0 ? std::abs(energy - expected) / expected : std::abs(energy);
			maxError = std::max(maxError, error);
			TEST_ASSERT(error < (fp64 ? 1e-12 : 3e-7), "Decay reduction precision exceeded tolerance");
			peak = std::max(peak, energy);
			last = energy;
		}
		std::cout << "Decay energy " << (fp64 ? "FP64" : "FP32") << " max relative error=" << maxError << " final/peak=" << last / peak << " zero/tiny fallbacks=" << fallbacks << std::endl;
		TEST_ASSERT(peak > 0 && last < peak * 1e-3, "Energy fixture did not excite and decay");
	}
	return true;
#endif
}

bool Test_Vulkan_ProbeReadbackCache()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=vulkan", "--numThreads=1"});
	ContinuousStructure* csx = CreateCustomGrid(17, 13, 19);
	auto* property = new CSPropProbeBox(csx->GetParameterSet());
	property->SetName("cache_probe");
	property->SetProbeType(0);
	auto* box = new CSPrimBox(csx->GetParameterSet(), property);
	const double bounds[] = {0, 0, 0, 0, -5, 5};
	for (unsigned int i = 0; i < 6; ++i) box->SetCoord(i, bounds[i]);
	csx->AddProperty(property);
	fdtd.SetCSX(csx);
	fdtd.SetGaussExcite(20e9, 10e9);
	fdtd.SetEnableDumps(false);
	fdtd.SetNumberOfTimeSteps(64);
	TEST_ASSERT(fdtd.SetupFDTD() == 0, "Probe cache fixture failed");
	auto* gpu = dynamic_cast<EngineVulkan*>(fdtd.GetBackend());
	TEST_ASSERT(gpu && gpu->SetProfilingEnabled(true), "Probe cache backend unavailable");
	ProcessingArray* processing = fdtd.GetProcessings();
	gpu->RegisterProbes(processing);
	gpu->SetVolt(2, 8, 6, 9, 0.25f);
	TEST_ASSERT(gpu->SyncProbesToHost() && fdtd.GetEng()->GetVolt(2, 8, 6, 9) == 0.25f, "On-demand gather missed field edit");
	const auto initial = gpu->GetProfile();
	TEST_ASSERT(initial.probeBytes > 0 && initial.downloadedBytes == 0, "Probe sync downloaded full fields");
	TEST_ASSERT(gpu->SyncProbesToHost() && gpu->GetProfile().submissions == initial.submissions, "Repeated probe sync submitted a gather");
	TEST_ASSERT(gpu->ClearProfile() && gpu->IterateTS(33) && gpu->SyncProbesToHost(), "Fused gather failed");
	const auto fused = gpu->GetProfile();
	TEST_ASSERT(fused.submissions == 2 && fused.probeBytes == initial.probeBytes, "Sample boundary needs a separate gather submission");
	TEST_ASSERT(gpu->SyncProbesToHost() && gpu->GetProfile().submissions == fused.submissions, "Fused probe result was not cached");
	TEST_ASSERT(gpu->SyncFieldsToHost(), "Full fields after partial sync failed");
	TEST_ASSERT(gpu->GetProfile().submissions == fused.submissions + 1 && gpu->GetProfile().downloadedBytes == 24ull * 17u * 13u * 19u,
	            "Partial sync marked full fields valid or copies were not batched");
	gpu->SetVolt(2, 8, 6, 9, 0.75f);
	TEST_ASSERT(gpu->SyncProbesToHost() && fdtd.GetEng()->GetVolt(2, 8, 6, 9) == 0.75f, "Probe cache survived a voltage edit");
	// Re-registration waits for pending stepping before replacing buffer descriptors.
	TEST_ASSERT(gpu->IterateTS(5), "Pending probe registration setup failed");
	for (unsigned int i = 0; i < 20; ++i) gpu->RegisterProbes(processing);
	TEST_ASSERT(gpu->ClearProfile() && gpu->SyncProbesToHost() && gpu->GetProfile().submissions == 1, "Probe registration did not invalidate cached values");
	gpu->RegisterProbes(nullptr);
	TEST_ASSERT(gpu->ClearProfile() && gpu->SyncProbesToHost() && gpu->GetProfile().submissions == 0, "Empty registry gathered probes");
	gpu->RegisterProbes(processing);
	TEST_ASSERT(gpu->SetReadbackOptimizationsEnabled(false) && gpu->ClearProfile() && gpu->IterateTS(1) && gpu->SyncProbesToHost(), "Reference gather failed");
	TEST_ASSERT(gpu->GetProfile().submissions == 2, "Reference mode did not retain separate probe gather");
	return true;
#endif
}

bool Test_Vulkan_EnergyStopping()
{
#ifndef ENABLE_VULKAN
	return true;
#else
	class EnergyTrace : public EngineVulkan {
	public:
		EnergyTrace(Operator* op, Engine_Interface_Base* engineInterface) : EngineVulkan(op), reference(engineInterface) {}
		bool GetFastEnergy(double& energy) override {
			if (!EngineVulkan::GetFastEnergy(energy)) {
				if (!SyncFieldsToHost()) return false;
				energy = reference->CalcFastEnergy();
			}
			samples.push_back({GetNumberOfTimesteps(), energy});
			return true;
		}
		std::vector<std::pair<unsigned int, double>> samples;
	private:
		std::unique_ptr<Engine_Interface_Base> reference;
	};
	auto setup = [](TestFDTDAccess& fdtd, bool sinus, double criterion, unsigned int mode) -> EnergyTrace* {
		ContinuousStructure* csx = CreateCustomGrid(17, 13, 19);
		auto* source = new CSPropExcitation(csx->GetParameterSet());
		source->SetExcitType(0);
		source->SetExcitation(1.0, 2);
		auto* box = new CSPrimBox(csx->GetParameterSet(), source);
		const double bounds[] = {-3, 3, -3, 3, -4, 4};
		for (unsigned int i = 0; i < 6; ++i) box->SetCoord(i, bounds[i]);
		csx->AddProperty(source);
		fdtd.SetCSX(csx);
		fdtd.SetLibraryArguments({"--engine=basic", "--numThreads=1", "--exact-endcriteria"});
		fdtd.SetNumberOfTimeSteps(512);
		fdtd.SetEndCriteria(criterion);
		fdtd.SetEnableDumps(false);
		if (sinus) fdtd.SetSinusExcite(10e9);
		else fdtd.SetGaussExcite(20e9, 10e9);
		for (unsigned int i = 0; i < 6; ++i) fdtd.Set_BC_Type(i, 2);
		if (fdtd.SetupFDTD() != 0) return nullptr;
		std::unique_ptr<EnergyTrace> gpu(new EnergyTrace(fdtd.GetOp(), fdtd.Interface()));
		if (!gpu->SetEnergyFloat64Enabled(mode != 2) || !gpu->SetReadbackOptimizationsEnabled(mode != 0) || !gpu->Initialize() || !gpu->SetProfilingEnabled(true)) return nullptr;
		gpu->RegisterProbes(fdtd.GetProcessings());
		EnergyTrace* result = gpu.get();
		fdtd.ReplaceBackend(std::move(gpu));
		return result;
	};
	TestFDTDAccess baseline;
	EnergyTrace* trace = setup(baseline, false, 0, false);
	TEST_ASSERT(trace, "Energy schedule fixture failed");
	baseline.RunFDTD();
	double peak = 0, threshold = 0;
	for (const auto& sample : trace->samples) {
		peak = std::max(peak, sample.second);
		const double ratio = peak > 0 ? sample.second / peak : 1;
		if (!threshold && sample.first > 80 && ratio < 0.1 && ratio > 0.001) threshold = ratio;
	}
	TEST_ASSERT(threshold > 0, "Stopping fixture did not cross a decay threshold");
	for (bool sinus : {false, true})
	for (double criterion : (sinus ? std::vector<double>{0.01} : std::vector<double>{threshold * (1 - 1e-6), threshold * (1 + 1e-6)})) {
		unsigned int stopped = 0;
		std::vector<std::pair<unsigned int, double>> reference;
		for (unsigned int optimized : {0u, 1u, 2u}) {
			TestFDTDAccess fdtd;
			EnergyTrace* gpu = setup(fdtd, sinus, criterion, optimized);
			TEST_ASSERT(gpu, "Stopping comparison setup failed");
			fdtd.RunFDTD();
			if (!optimized) { stopped = gpu->GetNumberOfTimesteps(); reference = gpu->samples; }
			else {
				TEST_ASSERT(gpu->GetNumberOfTimesteps() == stopped && gpu->samples.size() == reference.size(), "GPU energy changed stopping or sampling schedule");
				for (size_t i = 0; i < reference.size(); ++i) {
					TEST_ASSERT(reference[i].first == gpu->samples[i].first, "Energy check timestep changed");
					const double expected = reference[i].second;
					TEST_ASSERT(std::abs(expected - gpu->samples[i].second) <= std::max(expected * (optimized == 2 ? 3e-7 : 1e-12), 1e-30), "Energy trace exceeded reduction precision tolerance");
				}
				TEST_ASSERT(gpu->GetProfile().downloadedBytes == 24ull * 17u * 13u * 19u, "Energy-only stopping downloaded fields before final sync");
			}
			std::cout << "Stopping sinus=" << sinus << " optimized=" << optimized << " criterion=" << criterion << " timestep=" << gpu->GetNumberOfTimesteps() << std::endl;
		}
		TEST_ASSERT(stopped < 512, "Stopping fixture reached the fixed timestep limit");
	}
	return true;
#endif
}

bool Test_Vulkan_FailurePropagation()
{
	class FailingVulkan : public EngineVulkan {
	public:
		explicit FailingVulkan(bool iteration) : EngineVulkan(nullptr), failIteration(iteration) {}
		bool IterateTS(unsigned int) override { return !failIteration; }
		bool SyncFieldsToHost() override { return false; }
		bool SyncProbesToHost() override { return false; }
	private:
		bool failIteration;
	};
	for (bool iteration : {true, false}) {
		TestFDTDAccess fdtd;
		fdtd.SetLibraryArguments({"--engine=basic", "--numThreads=1"});
		fdtd.SetCSX(CreateSimpleGrid());
		fdtd.SetGaussExcite(1e9, 500e6);
		fdtd.SetNumberOfTimeSteps(1);
		fdtd.SetEnableDumps(false);
		TEST_ASSERT(fdtd.SetupFDTD() == 0, "Failure propagation setup failed");
		fdtd.ReplaceBackend(std::unique_ptr<EngineBackend>(new FailingVulkan(iteration)));
		bool failed = false;
		try { fdtd.RunFDTD(); }
		catch (const std::runtime_error&) { failed = true; }
		TEST_ASSERT(failed, "Backend failure must propagate from RunFDTD");
	}
	return true;
}

bool Benchmark_Vulkan_Multigrid()
{
	return RunMultigridEquivalence({}, false, false, false, false, true) &&
	       RunMultigridEquivalence({24.0}, false, false, false, false, true) &&
	       RunMultigridEquivalence({8.0, 16.0, 24.0, 32.0}, false, false, false, false, true);
}

int main(int argc, char* argv[])
{
	if (argc > 1 && std::string(argv[1]) == "--vulkan-benchmark")
		return RunVulkanPerformanceBenchmarks(argc, argv);
	if (argc > 1 && std::string(argv[1]) == "--batching-tests") {
		RUN_TEST(Test_Vulkan_BatchingAndProfiling);
		RUN_TEST(Test_Vulkan_FailurePropagation);
		return tests_failed ? 1 : 0;
	}
	if (argc > 1 && std::string(argv[1]) == "--readback-tests") {
		RUN_TEST(Test_Vulkan_EnergyReduction);
		RUN_TEST(Test_Vulkan_EnergyDecay);
		RUN_TEST(Test_Vulkan_ProbeReadbackCache);
		RUN_TEST(Test_Vulkan_EnergyStopping);
		return tests_failed ? 1 : 0;
	}
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
	RUN_TEST(Test_Vulkan_DebyeMaterial_MultiPole);
	RUN_TEST(Test_Vulkan_ConductingSheet_Equivalence);
	RUN_TEST(Test_Vulkan_Cylinder_Equivalence);
	RUN_TEST(Test_Vulkan_CheckedDimensionsAndIndices);
	RUN_TEST(Test_Vulkan_BatchingAndProfiling);
	RUN_TEST(Test_Vulkan_EnergyReduction);
	RUN_TEST(Test_Vulkan_EnergyDecay);
	RUN_TEST(Test_Vulkan_ProbeReadbackCache);
	RUN_TEST(Test_Vulkan_EnergyStopping);
	RUN_TEST(Test_Vulkan_FailurePropagation);
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
