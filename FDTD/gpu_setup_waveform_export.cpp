/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "gpu_setup_waveform_export.h"
#include "gpu_operator_setup.h"
#include "excitation.h"
#include "tools/useful.h"
#include <algorithm>
#include <cmath>

GPU_SetupWaveform GPU_DescribeSetupWaveform(Excitation& excitation, unsigned int maxTS)
{
	GPU_SetupWaveform d={};
	d.type=excitation.GetExciteType(); d.dt=excitation.GetTimestep();
	if (d.type==Excitation::GaussianPulse || d.type==Excitation::Sinusoidal) d.f0=excitation.GetCenterFreq();
	if (d.type==Excitation::GaussianPulse) d.fc=excitation.GetCutOffFreq();
	if (!std::isfinite(d.dt) || d.dt<=0) throw std::invalid_argument("invalid waveform timestep");
	double length=0;
	switch (d.type)
	{
	case Excitation::GaussianPulse:
		if (!std::isfinite(d.f0) || !std::isfinite(d.fc) || d.fc<=0) throw std::invalid_argument("invalid Gaussian waveform frequencies");
		length=std::min(double(maxTS),ceil(2*9.0/(2*acos(-1.0)*d.fc)/d.dt));
		d.maximum_frequency=d.f0+d.fc; d.frequency_of_interest=d.f0; break;
	case Excitation::Sinusoidal:
		if (!std::isfinite(d.f0) || d.f0<=0) throw std::invalid_argument("invalid sinusoidal waveform frequency");
		length=round(2.0/d.f0/d.dt); d.maximum_frequency=d.frequency_of_interest=d.f0; break;
	case Excitation::DiracPulse: case Excitation::Step:
		length=2; d.nyquist=1; break;
	case Excitation::CustomExcite:
		throw GPU_SetupUnavailable("custom waveforms require CPU operator construction");
	default: throw GPU_SetupUnavailable("no device waveform adapter for excitation type");
	}
	if (!std::isfinite(length) || length<1 || length>UINT32_MAX-255u) throw GPU_SetupUnavailable("waveform sample index range exceeded");
	d.length=static_cast<uint32_t>(length);
	if (!d.nyquist)
	{
		if (!std::isfinite(d.maximum_frequency) || d.maximum_frequency<0) throw std::invalid_argument("invalid waveform maximum frequency");
		d.nyquist=CalcNyquistNum(d.maximum_frequency,d.dt);
	}
	if (!d.nyquist) throw std::invalid_argument("invalid waveform Nyquist rate");
	return d;
}
