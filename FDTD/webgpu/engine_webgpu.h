/*
 *  Copyright (C) 2026 openEMS Project
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#ifndef ENGINE_WEBGPU_H
#define ENGINE_WEBGPU_H

#include "../engine_backend.h"
#include "webgpu_common.h"
#include <vector>
#include <memory>

class Operator;
class ContinuousStructure;

//! High-performance GPU acceleration engine powered by WebGPU (Metal, Vulkan, DX12).
class EngineWebGPU : public EngineBackend
{
public:
	explicit EngineWebGPU(const Operator* op);
	virtual ~EngineWebGPU();

	bool Initialize() override;
	void Reset() override;
	bool IterateTS(unsigned int iterTS) override;
	unsigned int GetNumberOfTimesteps() const override;
	void NextInterval(float curr_speed) override;

	FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
	FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
	void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val) override;
	void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val) override;

	bool SyncFieldsToHost() override;
	bool SyncProbesToHost() override;

	std::string GetBackendName() const override;

	//! Inspect the simulation setup to ensure all features are supported on GPU
	static bool CheckModelSupport(const Operator* op, const ContinuousStructure* csx, std::string& unsupportedReason);

	//! Query detected device capabilities
	OpenEMS_WebGPU::DeviceCaps GetDeviceCaps() const { return m_caps; }

private:
	const Operator* m_op;
	unsigned int m_numTS;
	OpenEMS_WebGPU::GridDimensions m_grid;
	OpenEMS_WebGPU::DeviceCaps m_caps;

	// WebGPU objects
	WGPUInstance m_instance;
	WGPUAdapter m_adapter;
	WGPUDevice m_device;
	WGPUQueue m_queue;

	// Device storage buffers
	WGPUBuffer m_bufVolt;
	WGPUBuffer m_bufCurr;
	WGPUBuffer m_bufVv;
	WGPUBuffer m_bufVi;
	WGPUBuffer m_bufIi;
	WGPUBuffer m_bufIv;
	WGPUBuffer m_bufGridUniforms;
	WGPUBuffer m_bufProbeHistory;

	// Compute pipelines
	WGPUComputePipeline m_pipelineVolt;
	WGPUComputePipeline m_pipelineCurr;
	WGPUComputePipeline m_pipelineProbe;

	// Bind groups
	WGPUBindGroup m_bindGroupFields;
	WGPUBindGroup m_bindGroupProbes;

	// Host-side mirrors (synced on demand)
	mutable std::vector<float> m_hostVolt;
	mutable std::vector<float> m_hostCurr;
	mutable bool m_hostFieldsValid;

	bool InitDevice();
	bool AllocateBuffers();
	bool CompileShaders();
	void DispatchWorkgroups(WGPUComputePipeline pipeline, uint32_t gx, uint32_t gy, uint32_t gz);
	size_t GetLinearIndex(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const;
};

#endif // ENGINE_WEBGPU_H
