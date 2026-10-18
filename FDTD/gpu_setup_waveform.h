/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_WAVEFORM_H
#define GPU_SETUP_WAVEFORM_H
#include "gpu_setup_expression.h"

struct GPU_SetupWaveform
{
	uint32_t type, length, nyquist;
	double dt, f0, fc, maximum_frequency, frequency_of_interest;
};

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_SIGNAL __host__ __device__
#else
#define GPU_SETUP_SIGNAL
#endif

GPU_SETUP_SIGNAL inline float GPU_SetupWaveformSample(const GPU_SetupWaveform& d, uint32_t index, bool current,
                                                       const GPU_SetupExpressionView&, const GPU_SetupExpression&)
{
	const double pi=3.1415926535897932384626433832795;
	if (d.type==2) return index==1 ? 1.0f : 0.0f;
	if (d.type==3) return 1.0f;
	if (!index && d.type!=10) return 0.0f;
	double time=index*d.dt;
	if (current) time+=0.5*d.dt;
	if (d.type==0)
	{
		const double position=2*pi*d.fc*time/3.0-3;
		return static_cast<float>(cos(2*pi*d.f0*(time-9.0/(2*pi*d.fc)))*exp(-position*position));
	}
	if (d.type==1) return static_cast<float>(sin(2*pi*d.f0*time));
	return 0; // Unsupported waveform types are rejected before allocation.
}

#undef GPU_SETUP_SIGNAL
#endif
