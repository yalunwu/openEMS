/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef HIP_SETUP_CANDIDATES_H
#define HIP_SETUP_CANDIDATES_H
#include "FDTD/gpu_setup_descriptors.h"
#include <hip/hip_runtime.h>

struct HIP_SetupCandidateInput
{
	uint32_t dims[3], grid_offsets[3], order_offsets[4];
	const double* grid;
	const GPU_SetupPrimitive* shapes;
	const uint32_t* order;
};

size_t HIP_SetupCandidateRows(const uint32_t dims[3]);
void HIP_CountSetupCandidates(HIP_SetupCandidateInput input, uint64_t* counts, hipStream_t stream);
void HIP_FillSetupCandidates(HIP_SetupCandidateInput input, const uint64_t* offsets, uint32_t* candidates, hipStream_t stream);
#endif
