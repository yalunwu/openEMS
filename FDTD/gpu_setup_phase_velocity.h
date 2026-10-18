/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_PHASE_VELOCITY_H
#define GPU_SETUP_PHASE_VELOCITY_H
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_PHASE __host__ __device__
#else
#define GPU_SETUP_PHASE
#endif

GPU_SETUP_PHASE inline double GPU_SetupPhaseVelocity(const double spacing[3], const double direction[3], float frequency, double dt, double c0)
{
	const double pi=3.1415926535897932384626433832795;
	for (int n=0;n<3;++n)
		if (fabs(direction[n])==1 && direction[(n+1)%3]==0 && direction[(n+2)%3]==0)
		{
			// Preserve the CPU's use of x spacing for each axial direction.
			const double k=asin(spacing[0]/c0/dt*sin(2*pi*frequency*dt/2))*2/spacing[0];
			return 2*pi*frequency/k;
		}
	const double k0=2*pi*frequency/c0;
	double k=k0,previous=0,phase=c0;
	const double rhs=pow(sin(2*pi*frequency*dt/2)/c0/dt,2);
	for (unsigned int iteration=0;fabs(previous-phase)>1e-6 && iteration<100;++iteration)
	{
		previous=phase;
		double value=0,derivative=0;
		for (int n=0;n<3;++n)
		{
			const double angle=direction[n]*k*spacing[n]/2;
			value+=pow(sin(angle)/spacing[n],2);
			derivative+=direction[n]*sin(angle)*cos(angle)/spacing[n];
		}
		value-=rhs;k-=value/derivative;
		if (k<k0) k=k0;
		phase=2*pi*frequency/k;
	}
	return phase;
}

#undef GPU_SETUP_PHASE
#endif
