/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "FDTD/hip/hip_internal.h"
#include "FDTD/hip/hip_setup_result.h"
#include "FDTD/excitation.h"
#include "fparser.hh"

void GPU_TestBuildWaveform(Excitation& excitation, unsigned int maxTS)
{
	std::unique_ptr<GPU_Backend_HIP> backend(GPU_Backend_HIP::New());
	if (!backend) throw std::runtime_error("required HIP backend unavailable");
	HIP_SetupResult result(backend->GetImpl()->ctx);
	HIP_BuildSetupWaveform(result,excitation,maxTS,false);
}

#define OPENEMS_DEVICE_WAVEFORM_TEST
#define openEMS_TestGPUSetupWaveform openEMS_TestGPUSetupWaveformDevice
#include "test_gpu_setup_waveform.cpp"
#undef openEMS_TestGPUSetupWaveform
