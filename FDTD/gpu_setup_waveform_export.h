/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_WAVEFORM_EXPORT_H
#define GPU_SETUP_WAVEFORM_EXPORT_H
#include "gpu_setup_waveform.h"
class Excitation;
GPU_SetupWaveform GPU_DescribeSetupWaveform(Excitation& excitation, unsigned int maxTS);
#endif
