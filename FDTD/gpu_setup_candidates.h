/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_CANDIDATES_H
#define GPU_SETUP_CANDIDATES_H
#include "gpu_setup_descriptors.h"
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_CANDIDATE __host__ __device__
#else
#define GPU_SETUP_CANDIDATE
#endif

GPU_SETUP_CANDIDATE inline bool GPU_SetupCandidate(const GPU_SetupPrimitive& primitive, const double bounds[6])
{
	// Unknown/transformed/mismatched-coordinate bounds are retained, as in
	// CSPrimitives::IsInsideBox. A conservative list must not drop a winner.
	if (!primitive.candidate_bounds) return true;
	for (int n=0;n<3;++n)
	{
		const double lo=fmin(primitive.bounds[2*n],primitive.bounds[2*n+1]);
		const double hi=fmax(primitive.bounds[2*n],primitive.bounds[2*n+1]);
		if (bounds[2*n+1]<lo || bounds[2*n]>hi) return false;
	}
	return true;
}

#undef GPU_SETUP_CANDIDATE
#endif
