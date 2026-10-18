/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "openems_global.h"
#if defined(OPENEMS_NATIVE_SETUP_TESTS) || defined(OPENEMS_DEVICE_WAVEFORM_TEST)
#include "FDTD/excitation.h"
#include "FDTD/gpu_setup_waveform_export.h"
#include "FDTD/gpu_operator_setup.h"
#include "fparser.hh"
#include "tools/useful.h"
#include <climits>
#include <limits>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef OPENEMS_DEVICE_WAVEFORM_TEST
void GPU_TestBuildWaveform(Excitation&, unsigned int);
#endif

extern "C" OPENEMS_EXPORT int openEMS_TestGPUSetupWaveform()
{
	try
	{
		if (CalcNyquistNum(0,1e-12)!=UINT_MAX || CalcNyquistNum(1e-300,1e-300)!=UINT_MAX ||
		    CalcNyquistNum(1e300,1e300)!=0 || CalcNyquistNum(2e9,1e-12)!=250)
			throw std::runtime_error("Nyquist interval bounds differ");
		for (double value:{-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
		{
			bool rejected=false;try {CalcNyquistNum(value,1e-12);} catch (const std::invalid_argument&) {rejected=true;}
			if (!rejected) throw std::runtime_error("invalid Nyquist frequency accepted");
			rejected=false;try {CalcNyquistNum(1,value);} catch (const std::invalid_argument&) {rejected=true;}
			if (!rejected) throw std::runtime_error("invalid Nyquist timestep accepted");
		}
		for (int type=0;type<4;++type)
		{
			Excitation expected,actual;
			for (Excitation* signal:{&expected,&actual})
			{
				if (type==0) signal->SetupGaussianPulse(2e9,1e9);
				if (type==1) signal->SetupSinusoidal(2e9);
				if (type==2) signal->SetupDiracPulse(3e9);
				if (type==3) signal->SetupStepExcite(3e9);
				if (type==4) signal->SetupCustomExcite("if(t<2e-12,1/2,sin(2*pi*2e9*t))",2e9,3e9);
				signal->Reset(1.7e-12);
			}
			const unsigned int maxTS=900;
			if (!expected.buildExcitationSignal(maxTS)) throw std::runtime_error("CPU waveform fixture failed");
#ifdef OPENEMS_DEVICE_WAVEFORM_TEST
			GPU_TestBuildWaveform(actual,maxTS);
			if (actual.GetLength()!=expected.GetLength() || actual.GetNyquistNum()!=expected.GetNyquistNum() ||
			    actual.GetMaxExcitationTimestep()!=expected.GetMaxExcitationTimestep()) throw std::runtime_error("device waveform metadata differs");
			const float* voltage=actual.GetVoltageSignal(); const float* current=actual.GetCurrentSignal();
#else
			const GPU_SetupWaveform d=GPU_DescribeSetupWaveform(actual,maxTS);
			if (d.length!=expected.GetLength() || d.nyquist!=expected.GetNyquistNum()) throw std::runtime_error("waveform descriptor metadata differs");
			GPU_SetupExpression custom={};
			std::vector<float> v(d.length),i(d.length);
			for (unsigned int n=0;n<d.length;++n)
			{
				int error=0;
				GPU_SetupExpressionView expressions={};
				v[n]=GPU_SetupWaveformSample(d,n,false,expressions,custom); if (error) throw std::runtime_error("voltage waveform expression failed");
				i[n]=GPU_SetupWaveformSample(d,n,true,expressions,custom); if (error) throw std::runtime_error("current waveform expression failed");
			}
			const float* voltage=v.data(); const float* current=i.data();
#endif
			for (unsigned int n=0;n<expected.GetLength();++n)
			{
				if (fabs(voltage[n]-expected.GetVoltageSignal()[n])>3e-7 || fabs(current[n]-expected.GetCurrentSignal()[n])>3e-7)
					throw std::runtime_error("waveform samples differ from CPU constructor");
			}
		}
		Excitation custom;custom.SetupCustomExcite("sin(t)",2e9,3e9);custom.Reset(1.7e-12);
		bool rejected=false;try {GPU_DescribeSetupWaveform(custom,900);} catch (const GPU_SetupUnavailable&) {rejected=true;}
		if (!rejected) throw std::runtime_error("custom waveform was accepted for GPU construction");
		std::cout << "PASS Gaussian, sinusoidal, Dirac and step waveforms; custom waveform rejection" << std::endl;
		return 0;
	}
	catch (const std::exception& error) {std::cerr << error.what() << std::endl;return 1;}
}
#else
extern "C" OPENEMS_EXPORT int openEMS_TestGPUSetupWaveform();
int main() {return openEMS_TestGPUSetupWaveform();}
#endif
