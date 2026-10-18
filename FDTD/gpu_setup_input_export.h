/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_INPUT_EXPORT_H
#define GPU_SETUP_INPUT_EXPORT_H
#include <cstdint>
class CSPropDiscMaterial;
class CSPropExcitation;
class CSPropMaterial;
struct GPU_SetupModel;
struct GPU_SetupProperty;
uint32_t GPU_ExportSetupDiscrete(CSPropDiscMaterial& material, GPU_SetupModel& model);
uint32_t GPU_ExportSetupFileWeight(CSPropExcitation& source, GPU_SetupModel& model);
void GPU_ExportSetupDispersion(CSPropMaterial& material, GPU_SetupModel& model, GPU_SetupProperty& property);
#endif
