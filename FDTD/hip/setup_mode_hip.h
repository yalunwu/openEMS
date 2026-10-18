/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SETUP_MODE_HIP_H
#define SETUP_MODE_HIP_H
#include "hip_internal.h"
#include "Common/engine_setup_tables.h"
std::shared_ptr<Engine_SetupTable> HIP_BuildModeTable(GPU_Backend_HIP::Impl* backend,const Operator* op,const Engine_ModeTableInput& input);
#endif
