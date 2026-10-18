/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SETUP_PROCESSING_HIP_H
#define SETUP_PROCESSING_HIP_H
#include "hip_internal.h"
#include "FDTD/gpu_setup_metrics.h"
struct HIP_SamplingMesh : Engine_SamplingMesh
{
	explicit HIP_SamplingMesh(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	~HIP_SamplingMesh();
	unsigned int Lines(int axis) const override {return lines[axis];}
	void Read(unsigned int* positions[3],double* coordinates[3]) const override;
	std::shared_ptr<HIP_Context> context;
	GPU_SetupMetrics metrics={};
	uint32_t lines[3]={},offsets[3]={};
	double* grid=nullptr;
	uint32_t* positions=nullptr;
	double* coordinates=nullptr;
	bool diagnostics=false;
};
#endif
