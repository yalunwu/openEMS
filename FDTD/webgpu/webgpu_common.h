/*
 *  Copyright (C) 2026 openEMS Project
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#ifndef WEBGPU_COMMON_H
#define WEBGPU_COMMON_H

#include <cstdint>
#include <cstddef>
#include <string>

#ifdef ENABLE_WEBGPU
#include <webgpu/webgpu.h>
#else
// Lightweight fallback types when WebGPU is disabled at compile-time
typedef struct WGPUInstanceImpl* WGPUInstance;
typedef struct WGPUAdapterImpl* WGPUAdapter;
typedef struct WGPUDeviceImpl* WGPUDevice;
typedef struct WGPUQueueImpl* WGPUQueue;
typedef struct WGPUBufferImpl* WGPUBuffer;
typedef struct WGPUComputePipelineImpl* WGPUComputePipeline;
typedef struct WGPUBindGroupImpl* WGPUBindGroup;
typedef struct WGPUBindGroupLayoutImpl* WGPUBindGroupLayout;
typedef struct WGPUShaderModuleImpl* WGPUShaderModule;
typedef struct WGPUPipelineLayoutImpl* WGPUPipelineLayout;
typedef struct WGPUCommandEncoderImpl* WGPUCommandEncoder;
typedef struct WGPUCommandBufferImpl* WGPUCommandBuffer;
#endif

namespace OpenEMS_WebGPU
{
	struct GridDimensions
	{
		uint32_t dimX = 0;
		uint32_t dimY = 0;
		uint32_t dimZ = 0;
		uint32_t numCells = 0;
	};

	struct DeviceCaps
	{
		std::string adapterName = "Unknown";
		std::string backendType = "None"; // "Metal", "Vulkan", "D3D12"
		bool isUnifiedMemory = false;      // True for Apple Silicon
		uint64_t maxStorageBufferSize = 0;
	};
}

#endif // WEBGPU_COMMON_H
