/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_UPML_H
#define GPU_SETUP_UPML_H
#include <cmath>
#include <climits>
#include <string>

inline bool GPU_SetupDefaultGrading(const std::string& expression)
{
	std::string normalized;
	for (char c:expression) if (c!=' ' && c!='\t' && c!='\r' && c!='\n') normalized+=c;
	return normalized=="-log(1e-6)*log(2.5)/(2*dl*Z*(pow(2.5,W/dl)-1))*pow(2.5,D/dl)";
}
#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_UPML_EVAL __host__ __device__
#else
#define GPU_SETUP_UPML_EVAL
#endif
// Match the stock parser's positive-base power, including integral exponents.
GPU_SETUP_UPML_EVAL inline double GPU_SetupGradingPower(double exponent)
{
	if (exponent>=LONG_MIN && exponent<LONG_MAX && exponent==floor(exponent))
	{
		unsigned long power=static_cast<unsigned long>(fabs(exponent));
		double result=1,base=2.5;
		while (power)
		{
			if (power&1) {result*=base;--power;}
			else {base*=base;power/=2;}
		}
		return exponent<0 ? 1/result : result;
	}
	return exponent>=0 ? exp(log(2.5)*exponent) : exp(log(1/2.5)*(-exponent));
}
GPU_SETUP_UPML_EVAL inline double GPU_SetupDefaultGradingValue(double depth,double delta,double width,double impedance)
{
	return -log(1e-6)*log(2.5)/(2*delta*impedance*(GPU_SetupGradingPower(width/delta)-1))*GPU_SetupGradingPower(depth/delta);
}
#undef GPU_SETUP_UPML_EVAL
#endif
