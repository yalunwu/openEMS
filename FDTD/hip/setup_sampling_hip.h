/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SETUP_SAMPLING_HIP_H
#define SETUP_SAMPLING_HIP_H
#include "hip_internal.h"
struct HIP_FieldGather : GPU_FieldGather
{
	explicit HIP_FieldGather(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	~HIP_FieldGather();
	std::shared_ptr<HIP_Context> context;
	GPU_GatherEntry* entries=nullptr;
	float* values=nullptr;
	GPU_Backend_HIP::Impl* backend=nullptr; // Processing is destroyed before its engine.
	std::shared_ptr<Engine_SamplingMesh> mesh;
	bool magnetic=false;
	size_t count=0;
	bool diagnostics=false;
	size_t Count() const override {return count;}
	void Read(std::vector<Engine_GatherEntry>& output) const override;
	bool Evaluate(float* output,size_t size) const override;
};
#endif
