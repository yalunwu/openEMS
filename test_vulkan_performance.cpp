#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif
#include "test_vulkan_performance.h"
#include "openems.h"
#include "FDTD/operator.h"
#include "FDTD/engine_interface_fdtd.h"
#include "FDTD/operator_cylindermultigrid.h"
#include "FDTD/vulkan/engine_vulkan.h"
#include "ContinuousStructure.h"
#include "CSPropExcitation.h"
#include "CSPropMaterial.h"
#include "CSPropDebyeMaterial.h"
#include "CSPropProbeBox.h"
#include "CSPropDumpBox.h"
#include "CSPrimBox.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
void Require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

class BenchmarkFDTD : public openEMS {
public:
	Operator* GetOp() { return FDTD_Op; }
	EngineVulkan* GPU() { return dynamic_cast<EngineVulkan*>(m_EngineBackend.get()); }
	double ReferenceEnergy() {
		std::unique_ptr<Engine_Interface_Base> engineInterface(NewEngineInterface());
		return engineInterface->CalcFastEnergy();
	}
};

struct Case {
	const char* name;
	int nx, ny, nz;
	bool cylinder, nonuniform, pml, debye;
	std::vector<double> splits;
	unsigned int processing; // 0: stepping, 1: probes, 2: dumps, 3: steady state, 4: energy
};

void Box(ContinuousStructure* csx, CSProperties* property, const double bounds[6])
{
	CSPrimBox* box = new CSPrimBox(csx->GetParameterSet(), property);
	for (int i = 0; i < 6; ++i) box->SetCoord(i, bounds[i]);
	csx->AddProperty(property);
}

void Setup(BenchmarkFDTD& fdtd, const Case& c, unsigned int steps, unsigned int batch, bool profile, const std::string& coefficients)
{
	ContinuousStructure* csx = new ContinuousStructure();
	CSRectGrid* grid = csx->GetGrid();
	grid->SetDeltaUnit(1e-3);
	const int dims[] = {c.nx, c.ny, c.nz};
	for (int d = 0; d < 3; ++d)
		for (int i = 0; i < dims[d]; ++i)
		{
			double t = static_cast<double>(i) / (dims[d] - 1);
			if (c.nonuniform) t = 0.5 * t + 0.5 * t * t;
			grid->AddDiscLine(d, c.cylinder && d == 1 ? 2 * std::acos(-1.0) * t : 40.0 * t);
		}
	CSPropExcitation* exc = new CSPropExcitation(csx->GetParameterSet());
	exc->SetExcitType(0);
	exc->SetExcitation(1.0, 2);
	const double cartSource[] = {18, 22, 18, 22, 18, 22};
	const double cylSource[] = {1, 36, 0.2, 0.4, 18, 22};
	Box(csx, exc, c.cylinder ? cylSource : cartSource);
	if (c.nonuniform || c.debye)
	{
		CSPropMaterial* material;
		if (c.debye)
		{
			auto* debye = new CSPropDebyeMaterial(csx->GetParameterSet());
			debye->SetDispersionOrder(2);
			for (unsigned int i = 0; i < 2; ++i) {
				debye->SetEpsDelta(i, 3.0 + i);
				debye->SetEpsRelaxTime(i, 1e-10 / (i + 1));
			}
			material = debye;
		}
		else material = new CSPropMaterial(csx->GetParameterSet());
		material->SetEpsilon(2.0);
		material->SetKappa(0.02);
		const double bounds[] = {10, 30, 10, 30, 10, 30};
		Box(csx, material, bounds);
	}
	if (c.processing == 1u)
	{
		auto* probe = new CSPropProbeBox(csx->GetParameterSet());
		probe->SetName("vulkan_benchmark_probe");
		probe->SetProbeType(0);
		const double bounds[] = {20, 20, 20, 20, 10, 30};
		Box(csx, probe, bounds);
	}
	if (c.processing == 2u)
	{
		auto* dump = new CSPropDumpBox(csx->GetParameterSet());
		dump->SetName("vulkan_benchmark_dump");
		dump->SetDumpType(0);
		dump->SetFileType(1); // HDF5
		const double bounds[] = {0, 40, 0, 40, 20, 20};
		Box(csx, dump, bounds);
	}
	fdtd.SetLibraryArguments({"--engine=vulkan", "--numThreads=1", "--exact-endcriteria",
	                          "--vulkan-batch-size=" + std::to_string(batch), "--vulkan-coefficients=" + coefficients});
	fdtd.SetNumberOfTimeSteps(steps);
	fdtd.SetEndCriteria(0.0);
	if (c.processing == 3u) fdtd.SetSinusExcite(10e9);
	else fdtd.SetGaussExcite(20e9, 10e9);
	fdtd.SetCSX(csx);
	fdtd.SetCylinderCoords(c.cylinder);
	if (!c.splits.empty()) fdtd.SetupCylinderMultiGrid(c.splits);
	fdtd.SetEnableDumps(c.processing == 2u);
	for (int i = 0; i < 6; ++i) {
		if (c.pml) fdtd.Set_BC_PML(i, 4);
		else fdtd.Set_BC_Type(i, 0);
	}
	Require(fdtd.SetupFDTD() == 0, "Benchmark setup failed");
	Require(fdtd.GPU() != nullptr, "Benchmark unexpectedly fell back to CPU");
	Require(fdtd.GPU()->SetProfilingEnabled(profile), "Could not configure profiling");
}

uint64_t Nodes(Operator* op, bool active)
{
	uint64_t result = 0;
	while (op) {
		auto* mg = dynamic_cast<Operator_CylinderMultiGrid*>(op);
		const uint64_t start = active && mg ? mg->GetSplitPos() - 1u : 0u;
		result += (op->GetNumberOfLines(0, true) - start) * uint64_t(op->GetNumberOfLines(1, true)) * op->GetNumberOfLines(2, true);
		op = mg ? mg->GetInnerOperator() : nullptr;
	}
	return result;
}

void Metadata(EngineVulkan& gpu)
{
	std::cout << "BENCHMARK_DEVICE " << gpu.GetDeviceDescription() << std::endl;
#ifdef GIT_VERSION
	std::cout << "BENCHMARK_REVISION " << GIT_VERSION << std::endl;
#endif
#ifdef _MSC_VER
	std::cout << "BENCHMARK_COMPILER MSVC " << _MSC_VER << std::endl;
#else
	std::cout << "BENCHMARK_COMPILER " << __VERSION__ << std::endl;
#endif
#ifdef NDEBUG
	std::cout << "BENCHMARK_BUILD Release" << std::endl;
#else
	std::cout << "BENCHMARK_BUILD Debug" << std::endl;
#endif
#ifdef _WIN32
	HMODULE module = GetModuleHandleA("libopenEMS.dll");
	if (!module) module = GetModuleHandleA("openEMS.dll");
	char path[32768] = {};
	if (module && GetModuleFileNameA(module, path, sizeof(path)))
		std::cout << "BENCHMARK_DLL " << path << std::endl;
#else
	std::ifstream maps("/proc/self/maps");
	std::string line;
	while (std::getline(maps, line))
		if (line.find("libopenEMS") != std::string::npos) {
			std::cout << "BENCHMARK_LIBRARY " << line.substr(line.find('/')) << std::endl;
			break;
		}
#endif
}

unsigned int Number(const std::string& text, unsigned int maximum)
{
	size_t used = 0;
	unsigned long value = std::stoul(text, &used);
	Require(used == text.size() && value > 0u && value <= maximum, "Invalid benchmark number");
	return static_cast<unsigned int>(value);
}
}

int RunVulkanPerformanceBenchmarks(int argc, char* argv[])
{
#ifndef ENABLE_VULKAN
	std::cerr << "Vulkan benchmarking requires ENABLE_VULKAN" << std::endl;
	return 1;
#else
	try {
		unsigned int steps = 256, repeats = 5;
		bool profile = false, referenceReadback = false, verifyEnergy = false;
		std::string selected, csvPath, coefficients = "dense";
		std::vector<unsigned int> batches = {1, 8, 16, 32, 64};
		for (int i = 2; i < argc; ++i) {
			std::string arg(argv[i]);
			if (arg == "--profile") profile = true;
			else if (arg == "--reference-readback") referenceReadback = true;
			else if (arg == "--verify-energy") verifyEnergy = true;
			else if (arg.find("--coefficients=") == 0) {
				coefficients = arg.substr(15);
				Require(coefficients == "dense" || coefficients == "palette" || coefficients == "analyze", "Invalid coefficient mode");
			}
			else if (arg.find("--case=") == 0) selected = arg.substr(7);
			else if (arg.find("--steps=") == 0) steps = Number(arg.substr(8), 1000000);
			else if (arg.find("--repeats=") == 0) repeats = Number(arg.substr(10), 100);
			else if (arg.find("--batch-size=") == 0) batches = {Number(arg.substr(13), 64)};
			else if (arg.find("--csv=") == 0) csvPath = arg.substr(6);
			else throw std::runtime_error("Unknown benchmark argument: " + arg);
		}
		const std::vector<Case> cases = {
			{"small", 25, 25, 25, false, false, false, false, {}, 0},
			{"large", 129, 129, 129, false, false, false, false, {}, 0},
			{"asymmetric", 97, 31, 65, false, false, false, false, {}, 0},
			{"thin", 129, 129, 3, false, false, false, false, {}, 0},
			{"lossy-nonuniform", 65, 49, 33, false, true, false, false, {}, 0},
			{"pml", 65, 65, 65, false, false, true, false, {}, 0},
			{"debye", 49, 49, 49, false, false, false, true, {}, 0},
			{"cylinder", 41, 129, 33, true, false, false, false, {}, 0},
			{"multigrid-1", 41, 129, 17, true, false, false, false, {}, 0},
			{"multigrid-2", 41, 129, 17, true, false, false, false, {24.0}, 0},
			{"multigrid-5", 41, 129, 17, true, false, false, false, {8.0, 16.0, 24.0, 32.0}, 0},
			{"probes", 33, 33, 33, false, false, false, false, {}, 1},
			{"dumps", 33, 33, 33, false, false, false, false, {}, 2},
			{"steady-state", 17, 17, 17, false, false, false, false, {}, 3},
			{"energy-large", 171, 171, 171, false, false, false, false, {}, 4}
		};
		std::ofstream csv;
		if (!csvPath.empty()) {
			csv.open(csvPath);
			Require(csv.good(), "Could not open benchmark CSV");
			csv << "case,batch,profile,repeat,steps,stored_nodes,active_voltage_nodes,operator_cells,initialization_s,iteration_s,synchronization_s,total_s,submissions,dispatches,uploaded_bytes,downloaded_bytes,probe_bytes,record_s,submit_s,wait_s,gpu_batch_s,energy_bytes,reference_readback,coefficient_mode,root_coefficient_bytes,root_allocated_bytes,root_dense_bytes,root_unique_tuples,root_palette\n";
			csv << std::setprecision(10);
		}
		bool found = false, metadata = false;
		for (const Case& c : cases) {
			if (!selected.empty() && selected != c.name) continue;
			found = true;
			for (unsigned int batch : batches) {
				std::unique_ptr<BenchmarkFDTD> fdtd;
				double initSeconds = 0;
				if (!c.processing) {
					fdtd.reset(new BenchmarkFDTD);
					const auto init = Clock::now();
					Setup(*fdtd, c, steps, batch, profile, coefficients);
					Require(fdtd->GPU()->SetReadbackOptimizationsEnabled(!referenceReadback), "Could not configure readback reference");
					initSeconds = Seconds(init);
					Require(fdtd->GPU()->IterateTS(64) && fdtd->GPU()->Synchronize(), "Warm-up failed");
				}
				std::vector<double> timings;
				for (unsigned int run = 0; run <= repeats; ++run) {
					if (c.processing) {
						fdtd.reset(new BenchmarkFDTD);
						const auto init = Clock::now();
						Setup(*fdtd, c, steps, batch, profile, coefficients);
						Require(fdtd->GPU()->SetReadbackOptimizationsEnabled(!referenceReadback), "Could not configure readback reference");
						initSeconds = Seconds(init);
					}
					EngineVulkan* gpu = fdtd->GPU();
					if (!metadata) { Metadata(*gpu); metadata = true; }
					Require(gpu->ClearProfile(), "Could not reset profiling");
					const auto begin = Clock::now();
					if (c.processing) fdtd->RunFDTD();
					else Require(gpu->IterateTS(steps), "Benchmark stepping failed");
					const double iteration = Seconds(begin);
					const auto sync = Clock::now();
					Require(gpu->Synchronize(), "Benchmark final synchronization failed");
					const double synchronization = Seconds(sync), total = Seconds(begin);
					if (run == 0u) continue; // warm-up result excluded
					const auto p = gpu->GetProfile();
					const auto coeff = gpu->GetCoefficientStatistics();
					if (profile) gpu->WriteProfile(std::cout);
					if (verifyEnergy && !referenceReadback) {
						double energy = 0;
						Require(gpu->GetFastEnergy(energy) && gpu->SyncFieldsToHost(), "Energy verification unavailable");
						const double expected = fdtd->ReferenceEnergy();
						const double error = expected > 0 ? std::abs(energy - expected) / expected : std::abs(energy);
						Require(std::isfinite(error) && error < 3e-7, "Large-domain energy precision exceeded tolerance");
						std::cout << "ENERGY_CHECK stored_nodes=" << Nodes(fdtd->GetOp(), false) << " relative_error=" << error << std::endl;
					}
					timings.push_back(total);
					if (csv) csv << c.name << ',' << batch << ',' << profile << ',' << run << ',' << steps << ','
					             << Nodes(fdtd->GetOp(), false) << ',' << Nodes(fdtd->GetOp(), true) << ','
					             << fdtd->GetOp()->GetNumberCells() << ',' << initSeconds << ',' << iteration << ','
					             << synchronization << ',' << total << ',' << p.submissions << ',' << p.dispatches << ','
					             << p.uploadedBytes << ',' << p.downloadedBytes << ',' << p.probeBytes << ','
					             << p.recordSeconds << ',' << p.submitSeconds << ',' << p.waitSeconds << ','
					             << p.gpuSeconds[EngineVulkan::ProfileBatch] << ',' << p.energyBytes << ',' << referenceReadback << ',' << coefficients << ',' << coeff.storageBytes << ',' << coeff.allocatedBytes << ',' << coeff.denseBytes << ',' << coeff.uniqueNodes << ',' << coeff.palette << '\n';
				}
				std::sort(timings.begin(), timings.end());
				const double median = (timings[(repeats - 1) / 2] + timings[repeats / 2]) / 2;
				std::cout << "BENCHMARK case=" << c.name << " batch=" << batch << " profile=" << profile
				          << " coefficients=" << coefficients << " reference_readback=" << referenceReadback
				          << " steps=" << steps << " repeats=" << repeats << " mode=" << (c.processing ? "solver-output" : "stepping")
				          << " stored_nodes=" << Nodes(fdtd->GetOp(), false) << " active_voltage_nodes=" << Nodes(fdtd->GetOp(), true)
				          << " operator_cells=" << fdtd->GetOp()->GetNumberCells() << " initialization_s=" << initSeconds
				          << " median_s=" << median << " min_s=" << timings.front() << " max_s=" << timings.back()
				          << " MCells/s=" << fdtd->GetOp()->GetNumberCells() * steps / median / 1e6 << std::endl;
			}
		}
		Require(found, "Unknown benchmark case");
		return 0;
	}
	catch (const std::exception& error) {
		std::cerr << "Vulkan benchmark: " << error.what() << std::endl;
		return 1;
	}
#endif
}
