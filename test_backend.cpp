/*
 *  Copyright (C) 2026 openEMS Project
 *
 *  Unit tests for pluggable EngineBackend and EngineWebGPU
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <string>

#include "openems.h"
#include "FDTD/engine_backend.h"
#include "FDTD/engine_cpu.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "FDTD/webgpu/engine_webgpu.h"
#include "ContinuousStructure.h"
#include "CSProperties.h"
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

bool Test_CapabilityScanner_StandardModel()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	std::string reason;
	Operator* dummyOp = reinterpret_cast<Operator*>(0x1234);

	bool supported = EngineBackend::CheckModelSupport(dummyOp, csx, reason);
	TEST_ASSERT(supported, "Standard grid model should be supported on GPU backend");
	TEST_ASSERT(reason.empty(), "Reason should be empty on success");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_LorentzMaterialFallback()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	CSPropLorentzMaterial* lorentz = new CSPropLorentzMaterial(csx->GetParameterSet());
	csx->AddProperty(lorentz);

	std::string reason;
	Operator* dummyOp = reinterpret_cast<Operator*>(0x1234);

	bool supported = EngineBackend::CheckModelSupport(dummyOp, csx, reason);
	TEST_ASSERT(!supported, "Lorentz material must be rejected by GPU capability scanner");
	TEST_ASSERT(reason.find("Lorentz") != std::string::npos, "Reason should mention Lorentz");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_DebyeMaterialFallback()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	CSPropDebyeMaterial* debye = new CSPropDebyeMaterial(csx->GetParameterSet());
	csx->AddProperty(debye);

	std::string reason;
	Operator* dummyOp = reinterpret_cast<Operator*>(0x1234);

	bool supported = EngineBackend::CheckModelSupport(dummyOp, csx, reason);
	TEST_ASSERT(!supported, "Debye material must be rejected by GPU capability scanner");
	TEST_ASSERT(reason.find("Debye") != std::string::npos, "Reason should mention Debye");

	delete csx;
	return true;
}

bool Test_CapabilityScanner_ConductingSheetFallback()
{
	ContinuousStructure* csx = CreateSimpleGrid();
	CSPropConductingSheet* sheet = new CSPropConductingSheet(csx->GetParameterSet());
	csx->AddProperty(sheet);

	std::string reason;
	Operator* dummyOp = reinterpret_cast<Operator*>(0x1234);

	bool supported = EngineBackend::CheckModelSupport(dummyOp, csx, reason);
	TEST_ASSERT(!supported, "Conducting sheet must be rejected by GPU capability scanner");
	TEST_ASSERT(reason.find("Conducting sheet") != std::string::npos, "Reason should mention Conducting sheets");

	delete csx;
	return true;
}

bool Test_EngineWebGPU_Lifecycle()
{
	EngineWebGPU engine(nullptr);
	TEST_ASSERT(engine.GetBackendName().find("WebGPU") != std::string::npos, "Backend name should contain WebGPU");
	TEST_ASSERT(engine.GetNumberOfTimesteps() == 0, "Initial timesteps should be 0");

	bool init = engine.Initialize();
	TEST_ASSERT(init, "EngineWebGPU initialization should succeed");

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
}

bool Test_OpenEMS_CLIArgument_EngineWebGPU()
{
	openEMS fdtd;
	std::vector<std::string> args = {"--engine=webgpu"};
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

bool Test_EndToEnd_WebGPUEngine()
{
	return RunSimulationWithEngine("webgpu");
}

class TestFDTDAccess : public openEMS {
public:
	Operator* GetOp() { return FDTD_Op; }
	Engine* GetEng() { return FDTD_Eng; }
	EngineBackend* GetBackend() { return m_EngineBackend.get(); }
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

static bool RunWebGPUSimulation(int nx, int ny, int nz, int timesteps = 15)
{
	openEMS fdtd;
	std::vector<std::string> args = {"--engine=webgpu"};
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

bool Test_WebGPU_SubWarpGrid()
{
	// 7 x 9 x 11 grid: dimZ = 11 < 32 (sub-warp in fast dimension)
	return RunWebGPUSimulation(7, 9, 11, 15);
}

bool Test_WebGPU_AsymmetricDimensions()
{
	// 15 x 5 x 47 grid: odd and prime non-multiple dimensions
	return RunWebGPUSimulation(15, 5, 47, 15);
}

bool Test_WebGPU_ThinPlanarGrid()
{
	// 35 x 35 x 3 grid: extremely thin in Z (3 cells)
	return RunWebGPUSimulation(35, 35, 3, 15);
}

bool Test_WebGPU_BoundaryConditions()
{
	openEMS fdtd;
	std::vector<std::string> args = {"--engine=webgpu"};
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

bool Test_WebGPU_ZeroExcitationGeometry()
{
	openEMS fdtd;
	std::vector<std::string> args = {"--engine=webgpu"};
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

bool Test_WebGPU_ResetLifecycleMultiRun()
{
	EngineWebGPU engine(nullptr);
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
}

bool Test_WebGPU_NumericalEquivalence_Asymmetric()
{
	int nx = 13, ny = 11, nz = 23;
	ContinuousStructure* csx = CreateCustomGrid(nx, ny, nz);

	TestFDTDAccess fdtd;
	fdtd.SetLibraryArguments({"--engine=webgpu"});
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

	// Set initial impulse at center (Ez component)
	int cx = nx / 2, cy = ny / 2, cz = nz / 2;
	cpuEng->SetVolt(2, cx, cy, cz, 1.0f);
	gpuEng->SetVolt(2, cx, cy, cz, 1.0f);

	for (int step = 1; step <= 25; ++step)
	{
		cpuEng->IterateTS(1);
		gpuEng->IterateTS(1);
	}

	gpuEng->SyncFieldsToHost();

	float maxDiff = 0.0f;
	for (int n = 0; n < 3; ++n)
	for (int x = 0; x < nx; ++x)
	for (int y = 0; y < ny; ++y)
	for (int z = 0; z < nz; ++z)
	{
		float vCpu = cpuEng->GetVolt(n, x, y, z);
		float vGpu = gpuEng->GetVolt(n, x, y, z);
		float d = std::abs(vCpu - vGpu);
		if (d > maxDiff) maxDiff = d;
	}

	TEST_ASSERT(maxDiff < 1e-5f, ("GPU vs CPU field difference exceeded tolerance: " + std::to_string(maxDiff)).c_str());
	return true;
}

int main(int argc, char* argv[])
{
	std::cout << "========================================" << std::endl;
	std::cout << " openEMS EngineBackend & WebGPU Test Suite" << std::endl;
	std::cout << "========================================" << std::endl;

	RUN_TEST(Test_BackendInterface_NullOp);
	RUN_TEST(Test_CapabilityScanner_StandardModel);
	RUN_TEST(Test_CapabilityScanner_LorentzMaterialFallback);
	RUN_TEST(Test_CapabilityScanner_DebyeMaterialFallback);
	RUN_TEST(Test_CapabilityScanner_ConductingSheetFallback);
	RUN_TEST(Test_EngineWebGPU_Lifecycle);
	RUN_TEST(Test_OpenEMS_CLIArgument_EngineWebGPU);

	std::cout << "----------------------------------------" << std::endl;
	std::cout << " Baseline End-to-End Simulation Tests" << std::endl;
	std::cout << "----------------------------------------" << std::endl;
	RUN_TEST(Test_EndToEnd_BasicEngine);
	RUN_TEST(Test_EndToEnd_SSEEngine);
	RUN_TEST(Test_EndToEnd_SSECompressedEngine);
	RUN_TEST(Test_EndToEnd_MultithreadedEngine);
	RUN_TEST(Test_EndToEnd_WebGPUEngine);

	std::cout << "----------------------------------------" << std::endl;
	std::cout << " Grid Sizes & Corner Cases Tests" << std::endl;
	std::cout << "----------------------------------------" << std::endl;
	RUN_TEST(Test_WebGPU_SubWarpGrid);
	RUN_TEST(Test_WebGPU_AsymmetricDimensions);
	RUN_TEST(Test_WebGPU_ThinPlanarGrid);
	RUN_TEST(Test_WebGPU_BoundaryConditions);
	RUN_TEST(Test_WebGPU_ZeroExcitationGeometry);
	RUN_TEST(Test_WebGPU_ResetLifecycleMultiRun);
	RUN_TEST(Test_WebGPU_NumericalEquivalence_Asymmetric);

	std::cout << "========================================" << std::endl;
	std::cout << "Tests completed: " << tests_passed << " passed, " << tests_failed << " failed." << std::endl;
	std::cout << "========================================" << std::endl;

	return (tests_failed == 0) ? 0 : 1;
}
