/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SETUP_GEOMETRY_INDEX_HIP_H
#define SETUP_GEOMETRY_INDEX_HIP_H
#include "FDTD/gpu_setup_descriptors.h"
#include <hip/hip_runtime.h>
#include <vector>

void HIP_BuildSetupBounds(GPU_SetupPrimitive* shapes,uint32_t count,const double* geometry,hipStream_t stream);
void HIP_BuildSetupGeometryIndex(const std::vector<GPU_SetupPrimitive>& shapes, const GPU_SetupPrimitive* device_shapes,const double* geometry,
                                 GPU_SetupBVHNode* tree, uint64_t* keys, hipStream_t stream);
#endif
