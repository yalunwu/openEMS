/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SETUP_CELL_MATERIAL_HIP_H
#define SETUP_CELL_MATERIAL_HIP_H
#include "hip_internal.h"
#include "FDTD/gpu_setup_metrics.h"

struct HIP_CellQueryDevice
{
	GPU_SetupMetrics metrics;
	const GPU_SetupPrimitive* shapes;
	const GPU_SetupProperty* properties;
	const GPU_SetupBVHNode* geometry_index;
	const double* geometry;
	const uint32_t* candidates;
	const GPU_SetupExpression* programs;
	const GPU_SetupInstruction* instructions;
	const double* constants;
	uint32_t stack_size;
	const GPU_SetupDiscreteMaterial* discrete;
	const float* material_data;
	const uint8_t* material_indices;
};

// Immutable inputs and device acceleration are retained only for models whose
// processing requests cell materials. No CPU geometry tree or cell sweep is
// needed to prepare SAR density, conductivity and cell volumes.
struct HIP_CellQuery
{
	explicit HIP_CellQuery(const std::shared_ptr<HIP_Context>& ctx) : context(ctx),device{} {}
	~HIP_CellQuery();
	std::shared_ptr<HIP_Context> context;
	HIP_CellQueryDevice device;
	std::vector<void*> allocations;
};

std::shared_ptr<Engine_CellMaterialTable> HIP_BuildCellMaterials(GPU_Backend_HIP::Impl* backend,const unsigned int lines[3],unsigned int* const positions[3],const std::shared_ptr<Engine_SamplingMesh>& mesh);
#endif
