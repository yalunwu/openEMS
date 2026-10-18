/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_MODE_EXPORT_H
#define GPU_SETUP_MODE_EXPORT_H
#include "gpu_operator_setup.h"
#include "Common/engine_setup_tables.h"
struct GPU_SetupModeInput
{
	GPU_SetupModel data;
	uint32_t expressions[2]={},file=0;
};
GPU_SetupModeInput GPU_ExportModeInput(const Engine_ModeTableInput& input);
#endif
