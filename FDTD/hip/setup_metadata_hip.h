/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef HIP_SETUP_METADATA_H
#define HIP_SETUP_METADATA_H
#include "FDTD/gpu_setup_path.h"
#include <hip/hip_runtime.h>
void HIP_BuildSetupMetadata(unsigned int mode,GPU_SetupMeshView mesh,const double* input,double* output,double dt,hipStream_t stream,bool cylindrical=false,bool closed=false);
#endif
