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
#include "FDTD/webgpu/engine_webgpu.h"
#include "ContinuousStructure.h"
#include "CSProperties.h"
#include "CSPropLorentzMaterial.h"
#include "CSPropDebyeMaterial.h"
#include "CSPropConductingSheet.h"

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

	std::cout << "========================================" << std::endl;
	std::cout << "Tests completed: " << tests_passed << " passed, " << tests_failed << " failed." << std::endl;
	std::cout << "========================================" << std::endl;

	return (tests_failed == 0) ? 0 : 1;
}
