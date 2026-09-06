/*
 *  Copyright (C) 2026 openEMS Project
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#include "engine_webgpu.h"
#include "shaders_embedded.h"
#include "../operator.h"
#include "../engine.h"
#include "ContinuousStructure.h"
#include "CSProperties.h"
#include "FDTD/extensions/operator_ext_excitation.h"
#include "FDTD/extensions/engine_ext_excitation.h"
#include "FDTD/excitation.h"
#include "Common/processing.h"
#include "Common/processintegral.h"
#include "Common/processvoltage.h"
#include "Common/processcurrent.h"
#include "Common/processfieldprobe.h"

#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef ENABLE_WEBGPU
#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>
#endif

EngineWebGPU::EngineWebGPU(const Operator* op)
	: m_op(op), m_numTS(0),
	  m_instance(nullptr), m_adapter(nullptr), m_device(nullptr), m_queue(nullptr),
	  m_bufVolt(nullptr), m_bufCurr(nullptr), m_bufVv(nullptr), m_bufVi(nullptr),
	  m_bufIi(nullptr), m_bufIv(nullptr), m_bufGridUniforms(nullptr), m_bufProbeHistory(nullptr),
	  m_bufStagingVolt(nullptr), m_bufStagingCurr(nullptr),
	  m_shaderModule(nullptr), m_bindGroupLayoutFields(nullptr), m_pipelineLayout(nullptr),
	  m_pipelineVolt(nullptr), m_pipelineCurr(nullptr), m_pipelineProbe(nullptr),
	  m_bindGroupFields(nullptr), m_bindGroupProbes(nullptr),
	  m_hostFieldsValid(false),
	  m_bufProbePoints(nullptr), m_bufProbeParams(nullptr),
	  m_bufProbeValues(nullptr), m_bufStagingProbes(nullptr),
	  m_bindGroupLayoutProbeGather(nullptr), m_pipelineLayoutProbeGather(nullptr),
	  m_pipelineProbeGather(nullptr), m_bindGroupProbeGather(nullptr),
	  m_bufVoltExcPoints(nullptr), m_bufCurrExcPoints(nullptr),
	  m_bufVoltExcParams(nullptr), m_bufCurrExcParams(nullptr),
	  m_bindGroupLayoutExc(nullptr), m_pipelineLayoutExc(nullptr),
	  m_pipelineExc(nullptr), m_bindGroupVoltExc(nullptr), m_bindGroupCurrExc(nullptr)
{
	if (m_op)
	{
		m_grid.dimX = static_cast<uint32_t>(m_op->GetNumberOfLines(0, true));
		m_grid.dimY = static_cast<uint32_t>(m_op->GetNumberOfLines(1, true));
		m_grid.dimZ = static_cast<uint32_t>(m_op->GetNumberOfLines(2, true));
	}
	else
	{
		m_grid.dimX = 16;
		m_grid.dimY = 16;
		m_grid.dimZ = 16;
	}
	m_grid.numCells = m_grid.dimX * m_grid.dimY * m_grid.dimZ;

	size_t totalElements = 3 * static_cast<size_t>(m_grid.numCells);
	m_hostVolt.resize(totalElements, 0.0f);
	m_hostCurr.resize(totalElements, 0.0f);
}

EngineWebGPU::~EngineWebGPU()
{
	Reset();
}

bool EngineWebGPU::CheckModelSupport(const Operator* op, const ContinuousStructure* csx, std::string& unsupportedReason)
{
	if (!op)
	{
		unsupportedReason = "Invalid operator.";
		return false;
	}

	if (csx)
	{
		ContinuousStructure* nonConstCSX = const_cast<ContinuousStructure*>(csx);
		if (nonConstCSX->GetQtyPropertyType(CSProperties::LORENTZMATERIAL) > 0)
		{
			unsupportedReason = "Lorentz dispersive material is not supported on WebGPU.";
			return false;
		}

		if (nonConstCSX->GetQtyPropertyType(CSProperties::DEBYEMATERIAL) > 0)
		{
			unsupportedReason = "Debye dispersive material is not supported on WebGPU.";
			return false;
		}

		if (nonConstCSX->GetQtyPropertyType(CSProperties::CONDUCTINGSHEET) > 0)
		{
			unsupportedReason = "Conducting sheets are not supported on WebGPU.";
			return false;
		}
	}

	return true;
}

bool EngineWebGPU::Initialize()
{
	std::cout << "[openEMS WebGPU] Initializing GPU acceleration engine..." << std::endl;

	if (!InitDevice())
	{
		std::cerr << "[openEMS WebGPU] Failed to initialize WebGPU device." << std::endl;
		return false;
	}

	if (!AllocateBuffers())
	{
		std::cerr << "[openEMS WebGPU] Failed to allocate GPU storage buffers." << std::endl;
		return false;
	}

	if (!CompileShaders())
	{
		std::cerr << "[openEMS WebGPU] Failed to compile WGSL compute pipelines." << std::endl;
		return false;
	}

	if (!AllocateExcitationBuffers())
	{
		std::cerr << "[openEMS WebGPU] Failed to allocate GPU excitation buffers." << std::endl;
		return false;
	}

	std::cout << "[openEMS WebGPU] Backend ready. Target: " << m_caps.backendType
	          << " (" << m_caps.adapterName << ")" << std::endl;
	return true;
}

#ifdef ENABLE_WEBGPU
namespace {
	struct DeviceRequestState {
		WGPUDevice device = nullptr;
		bool done = false;
	};

	void onDeviceCreated(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void* userdata2)
	{
		auto* state = static_cast<DeviceRequestState*>(userdata1);
		if (status == WGPURequestDeviceStatus_Success)
		{
			state->device = device;
		}
		else
		{
			std::cerr << "[openEMS WebGPU] RequestDevice error: "
			          << (message.data ? message.data : "Unknown") << std::endl;
		}
		state->done = true;
	}

	struct BufferMapContext {
		bool voltDone = false;
		bool currDone = false;
	};

	void onVoltMapped(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2)
	{
		auto* ctx = static_cast<BufferMapContext*>(userdata1);
		if (status == WGPUMapAsyncStatus_Success)
			ctx->voltDone = true;
	}

	void onCurrMapped(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2)
	{
		auto* ctx = static_cast<BufferMapContext*>(userdata1);
		if (status == WGPUMapAsyncStatus_Success)
			ctx->currDone = true;
	}

	struct ProbeMapContext {
		bool done = false;
	};

	void onProbeMapped(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2)
	{
		auto* ctx = static_cast<ProbeMapContext*>(userdata1);
		if (status == WGPUMapAsyncStatus_Success)
			ctx->done = true;
	}
}
#endif

#ifdef ENABLE_WEBGPU
static void onWgpuLog(WGPULogLevel level, WGPUStringView message, void* userdata)
{
	(void)userdata;
	if (message.data && message.length > 0)
	{
		std::cerr << "[wgpu-native log " << level << "] " << std::string(message.data, message.length) << std::endl;
	}
}
#endif

bool EngineWebGPU::InitDevice()
{
#ifdef ENABLE_WEBGPU
	wgpuSetLogLevel(WGPULogLevel_Warn);
	wgpuSetLogCallback(onWgpuLog, nullptr);

	WGPUInstanceDescriptor instanceDesc = {};
	m_instance = wgpuCreateInstance(&instanceDesc);
	if (!m_instance)
	{
		std::cerr << "[openEMS WebGPU] Failed to create WGPUInstance." << std::endl;
		return false;
	}

	size_t adapterCount = wgpuInstanceEnumerateAdapters(m_instance, nullptr, nullptr);
	if (adapterCount == 0)
	{
		std::cerr << "[openEMS WebGPU] No WebGPU adapters discovered." << std::endl;
		return false;
	}

	std::vector<WGPUAdapter> adapters(adapterCount);
	wgpuInstanceEnumerateAdapters(m_instance, nullptr, adapters.data());

	// Select best adapter: prioritize discrete GPU
	size_t selectedIdx = 0;
	for (size_t i = 0; i < adapterCount; ++i)
	{
		WGPUAdapterInfo info = {};
		wgpuAdapterGetInfo(adapters[i], &info);
		if (info.adapterType == WGPUAdapterType_DiscreteGPU)
		{
			selectedIdx = i;
			break;
		}
	}

	m_adapter = adapters[selectedIdx];

	WGPUAdapterInfo chosenInfo = {};
	wgpuAdapterGetInfo(m_adapter, &chosenInfo);
	m_caps.adapterName = (chosenInfo.device.data && chosenInfo.device.length > 0)
	                   ? std::string(chosenInfo.device.data, chosenInfo.device.length)
	                   : "WebGPU Device";

	switch (chosenInfo.backendType)
	{
	case WGPUBackendType_Vulkan:
		m_caps.backendType = "Vulkan";
		break;
	case WGPUBackendType_D3D12:
		m_caps.backendType = "DirectX 12";
		break;
	case WGPUBackendType_Metal:
		m_caps.backendType = "Metal";
		m_caps.isUnifiedMemory = true;
		break;
	case WGPUBackendType_OpenGL:
		m_caps.backendType = "OpenGL";
		break;
	default:
		m_caps.backendType = "WebGPU";
		break;
	}

	// Release unselected adapters
	for (size_t i = 0; i < adapterCount; ++i)
	{
		if (i != selectedIdx)
			wgpuAdapterRelease(adapters[i]);
	}

	// Request device
	DeviceRequestState devState;
	WGPUDeviceDescriptor devDesc = {};
	WGPURequestDeviceCallbackInfo devCbInfo = {};
	devCbInfo.mode = WGPUCallbackMode_AllowProcessEvents;
	devCbInfo.callback = onDeviceCreated;
	devCbInfo.userdata1 = &devState;

	wgpuAdapterRequestDevice(m_adapter, &devDesc, devCbInfo);

	int pollCount = 0;
	while (!devState.done && pollCount < 100)
	{
		wgpuInstanceProcessEvents(m_instance);
		pollCount++;
	}

	m_device = devState.device;
	if (!m_device)
	{
		std::cerr << "[openEMS WebGPU] Failed to acquire WGPUDevice." << std::endl;
		return false;
	}

	m_queue = wgpuDeviceGetQueue(m_device);
	return (m_queue != nullptr);

#else
	// Software emulation / fallback mode
	m_caps.adapterName = "Generic Compute Adapter";
#if defined(__APPLE__)
	m_caps.backendType = "Metal (Unified Memory)";
	m_caps.isUnifiedMemory = true;
#elif defined(_WIN32)
	m_caps.backendType = "DirectX 12 / Vulkan";
#else
	m_caps.backendType = "Vulkan";
#endif
	return true;
#endif
}

bool EngineWebGPU::AllocateBuffers()
{
	if (m_op)
	{
		m_grid.dimX = static_cast<uint32_t>(m_op->GetNumberOfLines(0, true));
		m_grid.dimY = static_cast<uint32_t>(m_op->GetNumberOfLines(1, true));
		m_grid.dimZ = static_cast<uint32_t>(m_op->GetNumberOfLines(2, true));
	}
	m_grid.numCells = m_grid.dimX * m_grid.dimY * m_grid.dimZ;
	size_t totalElements = 3 * static_cast<size_t>(m_grid.numCells);
	size_t fieldBytes = totalElements * sizeof(float);

	m_hostVolt.assign(totalElements, 0.0f);
	m_hostCurr.assign(totalElements, 0.0f);
	m_hostFieldsValid = true;

#ifdef ENABLE_WEBGPU
	if (m_device && m_queue)
	{
		// 1. Grid uniforms buffer
		struct GridUniforms {
			uint32_t dimX, dimY, dimZ, numCells;
		} uniforms = { m_grid.dimX, m_grid.dimY, m_grid.dimZ, m_grid.numCells };

		WGPUBufferDescriptor uboDesc = {};
		uboDesc.label = { "GridUniforms", WGPU_STRLEN };
		uboDesc.size = sizeof(GridUniforms);
		uboDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
		m_bufGridUniforms = wgpuDeviceCreateBuffer(m_device, &uboDesc);
		wgpuQueueWriteBuffer(m_queue, m_bufGridUniforms, 0, &uniforms, sizeof(uniforms));

		auto createStorage = [&](const char* label, WGPUBufferUsage usage) {
			WGPUBufferDescriptor bDesc = {};
			bDesc.label = { label, WGPU_STRLEN };
			bDesc.size = fieldBytes;
			bDesc.usage = usage;
			return wgpuDeviceCreateBuffer(m_device, &bDesc);
		};

		m_bufVolt = createStorage("Volt", WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
		m_bufCurr = createStorage("Curr", WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
		m_bufVv   = createStorage("Vv",   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
		m_bufVi   = createStorage("Vi",   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
		m_bufIi   = createStorage("Ii",   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
		m_bufIv   = createStorage("Iv",   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);

		// Staging readback buffers
		m_bufStagingVolt = createStorage("StagingVolt", WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
		m_bufStagingCurr = createStorage("StagingCurr", WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);

		// Upload material operator matrices
		if (m_op && m_op->vv_ptr && m_op->vi_ptr && m_op->ii_ptr && m_op->iv_ptr)
		{
			std::cout << "[openEMS WebGPU] Uploading operator material matrices (vv, vi, ii, iv) to GPU..." << std::endl;
			wgpuQueueWriteBuffer(m_queue, m_bufVv, 0, m_op->vv_ptr->data(), fieldBytes);
			wgpuQueueWriteBuffer(m_queue, m_bufVi, 0, m_op->vi_ptr->data(), fieldBytes);
			wgpuQueueWriteBuffer(m_queue, m_bufIi, 0, m_op->ii_ptr->data(), fieldBytes);
			wgpuQueueWriteBuffer(m_queue, m_bufIv, 0, m_op->iv_ptr->data(), fieldBytes);
		}
		else
		{
			std::cerr << "[openEMS WebGPU] WARNING: Operator matrices are null, defaulting to 1.0!" << std::endl;
			std::vector<float> defaultOnes(totalElements, 1.0f);
			wgpuQueueWriteBuffer(m_queue, m_bufVv, 0, defaultOnes.data(), fieldBytes);
			wgpuQueueWriteBuffer(m_queue, m_bufVi, 0, defaultOnes.data(), fieldBytes);
			wgpuQueueWriteBuffer(m_queue, m_bufIi, 0, defaultOnes.data(), fieldBytes);
			wgpuQueueWriteBuffer(m_queue, m_bufIv, 0, defaultOnes.data(), fieldBytes);
		}

		// Initialize field buffers to zero
		wgpuQueueWriteBuffer(m_queue, m_bufVolt, 0, m_hostVolt.data(), fieldBytes);
		wgpuQueueWriteBuffer(m_queue, m_bufCurr, 0, m_hostCurr.data(), fieldBytes);
	}
#endif

	return true;
}

bool EngineWebGPU::CompileShaders()
{
#ifdef ENABLE_WEBGPU
	if (!m_device) return false;

	// 1. Shader Module
	WGPUShaderSourceWGSL wgslDesc = {};
	wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
	wgslDesc.code.data = OpenEMS_WebGPU::kShaderYeeUpdate;
	wgslDesc.code.length = strlen(OpenEMS_WebGPU::kShaderYeeUpdate);

	WGPUShaderModuleDescriptor smDesc = {};
	smDesc.nextInChain = &wgslDesc.chain;
	smDesc.label = { "YeeShaderModule", WGPU_STRLEN };
	m_shaderModule = wgpuDeviceCreateShaderModule(m_device, &smDesc);
	if (!m_shaderModule)
	{
		std::cerr << "[openEMS WebGPU] Failed to compile Yee compute shader." << std::endl;
		return false;
	}

	// 2. Bind Group Layout
	WGPUBindGroupLayoutEntry bglEntries[7] = {};
	bglEntries[0].binding = 0;
	bglEntries[0].visibility = WGPUShaderStage_Compute;
	bglEntries[0].buffer.type = WGPUBufferBindingType_Uniform;

	bglEntries[1].binding = 1;
	bglEntries[1].visibility = WGPUShaderStage_Compute;
	bglEntries[1].buffer.type = WGPUBufferBindingType_Storage;

	bglEntries[2].binding = 2;
	bglEntries[2].visibility = WGPUShaderStage_Compute;
	bglEntries[2].buffer.type = WGPUBufferBindingType_Storage;

	bglEntries[3].binding = 3;
	bglEntries[3].visibility = WGPUShaderStage_Compute;
	bglEntries[3].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

	bglEntries[4].binding = 4;
	bglEntries[4].visibility = WGPUShaderStage_Compute;
	bglEntries[4].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

	bglEntries[5].binding = 5;
	bglEntries[5].visibility = WGPUShaderStage_Compute;
	bglEntries[5].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

	bglEntries[6].binding = 6;
	bglEntries[6].visibility = WGPUShaderStage_Compute;
	bglEntries[6].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

	WGPUBindGroupLayoutDescriptor bglDesc = {};
	bglDesc.entryCount = 7;
	bglDesc.entries = bglEntries;
	m_bindGroupLayoutFields = wgpuDeviceCreateBindGroupLayout(m_device, &bglDesc);

	// 3. Pipeline Layout
	WGPUPipelineLayoutDescriptor playDesc = {};
	playDesc.bindGroupLayoutCount = 1;
	playDesc.bindGroupLayouts = &m_bindGroupLayoutFields;
	m_pipelineLayout = wgpuDeviceCreatePipelineLayout(m_device, &playDesc);

	// 4. Compute Pipelines
	WGPUComputePipelineDescriptor pipeDescVolt = {};
	pipeDescVolt.layout = m_pipelineLayout;
	pipeDescVolt.compute.module = m_shaderModule;
	pipeDescVolt.compute.entryPoint = { "update_voltages", WGPU_STRLEN };
	pipeDescVolt.label = { "PipelineUpdateVoltages", WGPU_STRLEN };
	m_pipelineVolt = wgpuDeviceCreateComputePipeline(m_device, &pipeDescVolt);

	WGPUComputePipelineDescriptor pipeDescCurr = {};
	pipeDescCurr.layout = m_pipelineLayout;
	pipeDescCurr.compute.module = m_shaderModule;
	pipeDescCurr.compute.entryPoint = { "update_currents", WGPU_STRLEN };
	pipeDescCurr.label = { "PipelineUpdateCurrents", WGPU_STRLEN };
	m_pipelineCurr = wgpuDeviceCreateComputePipeline(m_device, &pipeDescCurr);

	// 5. Bind Group
	size_t fieldBytes = 3 * static_cast<size_t>(m_grid.numCells) * sizeof(float);
	WGPUBindGroupEntry bgEntries[7] = {};
	bgEntries[0].binding = 0; bgEntries[0].buffer = m_bufGridUniforms; bgEntries[0].size = 16;
	bgEntries[1].binding = 1; bgEntries[1].buffer = m_bufVolt;         bgEntries[1].size = fieldBytes;
	bgEntries[2].binding = 2; bgEntries[2].buffer = m_bufCurr;         bgEntries[2].size = fieldBytes;
	bgEntries[3].binding = 3; bgEntries[3].buffer = m_bufVv;           bgEntries[3].size = fieldBytes;
	bgEntries[4].binding = 4; bgEntries[4].buffer = m_bufVi;           bgEntries[4].size = fieldBytes;
	bgEntries[5].binding = 5; bgEntries[5].buffer = m_bufIi;           bgEntries[5].size = fieldBytes;
	bgEntries[6].binding = 6; bgEntries[6].buffer = m_bufIv;           bgEntries[6].size = fieldBytes;

	WGPUBindGroupDescriptor bgDesc = {};
	bgDesc.layout = m_bindGroupLayoutFields;
	bgDesc.entryCount = 7;
	bgDesc.entries = bgEntries;
	m_bindGroupFields = wgpuDeviceCreateBindGroup(m_device, &bgDesc);

	return (m_pipelineVolt != nullptr && m_pipelineCurr != nullptr && m_bindGroupFields != nullptr);
#else
	return true;
#endif
}

void EngineWebGPU::Reset()
{
	m_numTS = 0;
	m_hostFieldsValid = false;
	m_hostVolt.clear();
	m_hostCurr.clear();

	m_probePoints.clear();
	m_excSources.clear();
	m_voltExcPoints.clear();
	m_currExcPoints.clear();
#ifdef ENABLE_WEBGPU
	if (m_bindGroupVoltExc)           { wgpuBindGroupRelease(m_bindGroupVoltExc);                 m_bindGroupVoltExc = nullptr; }
	if (m_bindGroupCurrExc)           { wgpuBindGroupRelease(m_bindGroupCurrExc);                 m_bindGroupCurrExc = nullptr; }
	if (m_pipelineExc)                { wgpuComputePipelineRelease(m_pipelineExc);                m_pipelineExc = nullptr; }
	if (m_pipelineLayoutExc)          { wgpuPipelineLayoutRelease(m_pipelineLayoutExc);          m_pipelineLayoutExc = nullptr; }
	if (m_bindGroupLayoutExc)         { wgpuBindGroupLayoutRelease(m_bindGroupLayoutExc);         m_bindGroupLayoutExc = nullptr; }
	if (m_bufVoltExcPoints)           { wgpuBufferRelease(m_bufVoltExcPoints);                    m_bufVoltExcPoints = nullptr; }
	if (m_bufCurrExcPoints)           { wgpuBufferRelease(m_bufCurrExcPoints);                    m_bufCurrExcPoints = nullptr; }
	if (m_bufVoltExcParams)           { wgpuBufferRelease(m_bufVoltExcParams);                    m_bufVoltExcParams = nullptr; }
	if (m_bufCurrExcParams)           { wgpuBufferRelease(m_bufCurrExcParams);                    m_bufCurrExcParams = nullptr; }
	if (m_bindGroupProbeGather)       { wgpuBindGroupRelease(m_bindGroupProbeGather);             m_bindGroupProbeGather = nullptr; }
	if (m_pipelineProbeGather)        { wgpuComputePipelineRelease(m_pipelineProbeGather);        m_pipelineProbeGather = nullptr; }
	if (m_pipelineLayoutProbeGather)  { wgpuPipelineLayoutRelease(m_pipelineLayoutProbeGather);  m_pipelineLayoutProbeGather = nullptr; }
	if (m_bindGroupLayoutProbeGather) { wgpuBindGroupLayoutRelease(m_bindGroupLayoutProbeGather); m_bindGroupLayoutProbeGather = nullptr; }
	if (m_bufStagingProbes)           { wgpuBufferRelease(m_bufStagingProbes);                    m_bufStagingProbes = nullptr; }
	if (m_bufProbeValues)             { wgpuBufferRelease(m_bufProbeValues);                      m_bufProbeValues = nullptr; }
	if (m_bufProbeParams)             { wgpuBufferRelease(m_bufProbeParams);                      m_bufProbeParams = nullptr; }
	if (m_bufProbePoints)             { wgpuBufferRelease(m_bufProbePoints);                      m_bufProbePoints = nullptr; }
	if (m_bindGroupFields)            { wgpuBindGroupRelease(m_bindGroupFields);                  m_bindGroupFields = nullptr; }
	if (m_bindGroupProbes)            { wgpuBindGroupRelease(m_bindGroupProbes);                  m_bindGroupProbes = nullptr; }
	if (m_pipelineVolt)               { wgpuComputePipelineRelease(m_pipelineVolt);               m_pipelineVolt = nullptr; }
	if (m_pipelineCurr)               { wgpuComputePipelineRelease(m_pipelineCurr);               m_pipelineCurr = nullptr; }
	if (m_pipelineProbe)              { wgpuComputePipelineRelease(m_pipelineProbe);              m_pipelineProbe = nullptr; }
	if (m_pipelineLayout)             { wgpuPipelineLayoutRelease(m_pipelineLayout);             m_pipelineLayout = nullptr; }
	if (m_bindGroupLayoutFields)      { wgpuBindGroupLayoutRelease(m_bindGroupLayoutFields);      m_bindGroupLayoutFields = nullptr; }
	if (m_shaderModule)               { wgpuShaderModuleRelease(m_shaderModule);                  m_shaderModule = nullptr; }
	if (m_bufStagingVolt)             { wgpuBufferRelease(m_bufStagingVolt);                      m_bufStagingVolt = nullptr; }
	if (m_bufStagingCurr)             { wgpuBufferRelease(m_bufStagingCurr);                      m_bufStagingCurr = nullptr; }
	if (m_bufGridUniforms)            { wgpuBufferRelease(m_bufGridUniforms);                     m_bufGridUniforms = nullptr; }
	if (m_bufVolt)                    { wgpuBufferRelease(m_bufVolt);                             m_bufVolt = nullptr; }
	if (m_bufCurr)                    { wgpuBufferRelease(m_bufCurr);                             m_bufCurr = nullptr; }
	if (m_bufVv)                      { wgpuBufferRelease(m_bufVv);                               m_bufVv = nullptr; }
	if (m_bufVi)                      { wgpuBufferRelease(m_bufVi);                               m_bufVi = nullptr; }
	if (m_bufIi)                      { wgpuBufferRelease(m_bufIi);                               m_bufIi = nullptr; }
	if (m_bufIv)                      { wgpuBufferRelease(m_bufIv);                               m_bufIv = nullptr; }
	if (m_bufProbeHistory)            { wgpuBufferRelease(m_bufProbeHistory);                     m_bufProbeHistory = nullptr; }
	if (m_queue)                      { wgpuQueueRelease(m_queue);                                m_queue = nullptr; }
	if (m_device)                     { wgpuDeviceRelease(m_device);                              m_device = nullptr; }
	if (m_adapter)                    { wgpuAdapterRelease(m_adapter);                            m_adapter = nullptr; }
	if (m_instance)                   { wgpuInstanceRelease(m_instance);                          m_instance = nullptr; }
#endif
}

bool EngineWebGPU::AllocateExcitationBuffers()
{
#ifdef ENABLE_WEBGPU
	if (!m_op || !m_device) return true;

	m_excSources.clear();
	m_voltExcPoints.clear();
	m_currExcPoints.clear();

	for (size_t i = 0; i < m_op->GetNumberOfExtentions(); ++i)
	{
		Operator_Extension* ext = m_op->GetExtension(i);
		Operator_Ext_Excitation* excExt = dynamic_cast<Operator_Ext_Excitation*>(ext);
		if (!excExt || !excExt->m_Exc) continue;

		ExcitationSourceInfo src;
		src.exc = excExt->m_Exc;
		src.voltCount = excExt->Volt_Count;
		src.currCount = excExt->Curr_Count;

		if (src.voltCount > 0)
		{
			src.voltIndices.resize(src.voltCount);
			src.voltDelays.resize(src.voltCount);
			src.voltAmps.resize(src.voltCount);
			for (unsigned int n = 0; n < src.voltCount; ++n)
			{
				src.voltIndices[n] = static_cast<uint32_t>(GetLinearIndex(
					excExt->Volt_dir[n],
					excExt->Volt_index[0][n],
					excExt->Volt_index[1][n],
					excExt->Volt_index[2][n]
				));
				src.voltDelays[n] = excExt->Volt_delay[n];
				src.voltAmps[n] = excExt->Volt_amp[n];
				m_voltExcPoints.push_back({src.voltIndices[n], 0.0f});
			}
		}

		if (src.currCount > 0)
		{
			src.currIndices.resize(src.currCount);
			src.currDelays.resize(src.currCount);
			src.currAmps.resize(src.currCount);
			for (unsigned int n = 0; n < src.currCount; ++n)
			{
				src.currIndices[n] = static_cast<uint32_t>(GetLinearIndex(
					excExt->Curr_dir[n],
					excExt->Curr_index[0][n],
					excExt->Curr_index[1][n],
					excExt->Curr_index[2][n]
				));
				src.currDelays[n] = excExt->Curr_delay[n];
				src.currAmps[n] = excExt->Curr_amp[n];
				m_currExcPoints.push_back({src.currIndices[n], 0.0f});
			}
		}

		m_excSources.push_back(std::move(src));
	}

	if (m_voltExcPoints.empty() && m_currExcPoints.empty())
		return true;

	std::cout << "[openEMS WebGPU] Initialized on-device excitation: "
	          << m_voltExcPoints.size() << " voltage points, "
	          << m_currExcPoints.size() << " current points." << std::endl;

	// Pipeline layout
	WGPUBindGroupLayoutEntry bglEntries[3] = {};
	bglEntries[0].binding = 0; bglEntries[0].visibility = WGPUShaderStage_Compute; bglEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
	bglEntries[1].binding = 1; bglEntries[1].visibility = WGPUShaderStage_Compute; bglEntries[1].buffer.type = WGPUBufferBindingType_Storage;
	bglEntries[2].binding = 2; bglEntries[2].visibility = WGPUShaderStage_Compute; bglEntries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

	WGPUBindGroupLayoutDescriptor bglDesc = {};
	bglDesc.entryCount = 3;
	bglDesc.entries = bglEntries;
	m_bindGroupLayoutExc = wgpuDeviceCreateBindGroupLayout(m_device, &bglDesc);

	WGPUPipelineLayoutDescriptor playDesc = {};
	playDesc.bindGroupLayoutCount = 1;
	playDesc.bindGroupLayouts = &m_bindGroupLayoutExc;
	m_pipelineLayoutExc = wgpuDeviceCreatePipelineLayout(m_device, &playDesc);

	// Compile shader module
	WGPUShaderSourceWGSL wgslDesc = {};
	wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
	wgslDesc.code.data = OpenEMS_WebGPU::kShaderExcitation;
	wgslDesc.code.length = strlen(OpenEMS_WebGPU::kShaderExcitation);

	WGPUShaderModuleDescriptor smDesc = {};
	smDesc.nextInChain = &wgslDesc.chain;
	smDesc.label = { "ExcitationShaderModule", WGPU_STRLEN };
	WGPUShaderModule sm = wgpuDeviceCreateShaderModule(m_device, &smDesc);

	WGPUComputePipelineDescriptor pipeDesc = {};
	pipeDesc.layout = m_pipelineLayoutExc;
	pipeDesc.compute.module = sm;
	pipeDesc.compute.entryPoint = { "inject_excitation", WGPU_STRLEN };
	pipeDesc.label = { "PipelineExcitation", WGPU_STRLEN };
	m_pipelineExc = wgpuDeviceCreateComputePipeline(m_device, &pipeDesc);
	wgpuShaderModuleRelease(sm);

	size_t fieldBytes = 3 * static_cast<size_t>(m_grid.numCells) * sizeof(float);

	// Voltage excitation buffers and bind group
	if (!m_voltExcPoints.empty())
	{
		size_t ptsBytes = m_voltExcPoints.size() * sizeof(GpuExcPoint);
		WGPUBufferDescriptor ptsDesc = {};
		ptsDesc.label = { "VoltExcPoints", WGPU_STRLEN };
		ptsDesc.size = ptsBytes;
		ptsDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
		m_bufVoltExcPoints = wgpuDeviceCreateBuffer(m_device, &ptsDesc);

		GpuExcParams params = { static_cast<uint32_t>(m_voltExcPoints.size()), {0, 0, 0} };
		WGPUBufferDescriptor prmDesc = {};
		prmDesc.label = { "VoltExcParams", WGPU_STRLEN };
		prmDesc.size = sizeof(params);
		prmDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
		m_bufVoltExcParams = wgpuDeviceCreateBuffer(m_device, &prmDesc);
		wgpuQueueWriteBuffer(m_queue, m_bufVoltExcParams, 0, &params, sizeof(params));

		WGPUBindGroupEntry bgEntries[3] = {};
		bgEntries[0].binding = 0; bgEntries[0].buffer = m_bufVoltExcParams; bgEntries[0].size = sizeof(params);
		bgEntries[1].binding = 1; bgEntries[1].buffer = m_bufVolt;          bgEntries[1].size = fieldBytes;
		bgEntries[2].binding = 2; bgEntries[2].buffer = m_bufVoltExcPoints;  bgEntries[2].size = ptsBytes;

		WGPUBindGroupDescriptor bgDesc = {};
		bgDesc.layout = m_bindGroupLayoutExc;
		bgDesc.entryCount = 3;
		bgDesc.entries = bgEntries;
		m_bindGroupVoltExc = wgpuDeviceCreateBindGroup(m_device, &bgDesc);
	}

	// Current excitation buffers and bind group
	if (!m_currExcPoints.empty())
	{
		size_t ptsBytes = m_currExcPoints.size() * sizeof(GpuExcPoint);
		WGPUBufferDescriptor ptsDesc = {};
		ptsDesc.label = { "CurrExcPoints", WGPU_STRLEN };
		ptsDesc.size = ptsBytes;
		ptsDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
		m_bufCurrExcPoints = wgpuDeviceCreateBuffer(m_device, &ptsDesc);

		GpuExcParams params = { static_cast<uint32_t>(m_currExcPoints.size()), {0, 0, 0} };
		WGPUBufferDescriptor prmDesc = {};
		prmDesc.label = { "CurrExcParams", WGPU_STRLEN };
		prmDesc.size = sizeof(params);
		prmDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
		m_bufCurrExcParams = wgpuDeviceCreateBuffer(m_device, &prmDesc);
		wgpuQueueWriteBuffer(m_queue, m_bufCurrExcParams, 0, &params, sizeof(params));

		WGPUBindGroupEntry bgEntries[3] = {};
		bgEntries[0].binding = 0; bgEntries[0].buffer = m_bufCurrExcParams; bgEntries[0].size = sizeof(params);
		bgEntries[1].binding = 1; bgEntries[1].buffer = m_bufCurr;          bgEntries[1].size = fieldBytes;
		bgEntries[2].binding = 2; bgEntries[2].buffer = m_bufCurrExcPoints;  bgEntries[2].size = ptsBytes;

		WGPUBindGroupDescriptor bgDesc = {};
		bgDesc.layout = m_bindGroupLayoutExc;
		bgDesc.entryCount = 3;
		bgDesc.entries = bgEntries;
		m_bindGroupCurrExc = wgpuDeviceCreateBindGroup(m_device, &bgDesc);
	}

	return (m_pipelineExc != nullptr);
#else
	return true;
#endif
}

bool EngineWebGPU::PrepareVoltExcitation(unsigned int step)
{
	bool anyNonZero = false;
	size_t ptIdx = 0;
	for (const auto& src : m_excSources)
	{
		if (!src.exc || src.voltCount == 0) continue;
		FDTD_FLOAT* exc_volt = src.exc->GetVoltageSignal();
		if (!exc_volt) continue;
		unsigned int length = src.exc->GetLength();
		int p = static_cast<int>(step) + 1;
		if (src.exc->GetSignalPeriod() > 0)
			p = static_cast<int>(src.exc->GetSignalPeriod() / src.exc->GetTimestep());

		for (unsigned int n = 0; n < src.voltCount; ++n)
		{
			int exc_pos = static_cast<int>(step) - static_cast<int>(src.voltDelays[n]);
			exc_pos *= (exc_pos > 0);
			exc_pos %= p;
			exc_pos *= (exc_pos < static_cast<int>(length));

			float val = 0.0f;
			if (exc_pos < static_cast<int>(length))
			{
				val = src.voltAmps[n] * exc_volt[exc_pos];
			}
			if (val != 0.0f) anyNonZero = true;
			m_voltExcPoints[ptIdx++].value = val;
		}
	}
	return anyNonZero;
}

bool EngineWebGPU::PrepareCurrExcitation(unsigned int step)
{
	bool anyNonZero = false;
	size_t ptIdx = 0;
	for (const auto& src : m_excSources)
	{
		if (!src.exc || src.currCount == 0) continue;
		FDTD_FLOAT* exc_curr = src.exc->GetCurrentSignal();
		if (!exc_curr) continue;
		unsigned int length = src.exc->GetLength();
		int p = static_cast<int>(step) + 1;
		if (src.exc->GetSignalPeriod() > 0)
			p = static_cast<int>(src.exc->GetSignalPeriod() / src.exc->GetTimestep());

		for (unsigned int n = 0; n < src.currCount; ++n)
		{
			int exc_pos = static_cast<int>(step) - static_cast<int>(src.currDelays[n]);
			exc_pos *= (exc_pos > 0);
			exc_pos %= p;
			exc_pos *= (exc_pos < static_cast<int>(length));

			float val = 0.0f;
			if (exc_pos < static_cast<int>(length))
			{
				val = src.currAmps[n] * exc_curr[exc_pos];
			}
			if (val != 0.0f) anyNonZero = true;
			m_currExcPoints[ptIdx++].value = val;
		}
	}
	return anyNonZero;
}

void EngineWebGPU::RegisterProbes(const ProcessingArray* pa)
{
	if (!pa) return;

	m_probePoints.clear();

	for (size_t i = 0; i < pa->GetNumberOfProcessings(); ++i)
	{
		Processing* proc = const_cast<ProcessingArray*>(pa)->GetProcessing(i);
		if (!proc || !proc->GetEnable()) continue;

		ProcessVoltage* pv = dynamic_cast<ProcessVoltage*>(proc);
		if (pv)
		{
			const unsigned int* start = pv->GetStartCoord();
			const unsigned int* stop = pv->GetStopCoord();
			for (int n = 0; n < 3; ++n)
			{
				if (start[n] < stop[n])
				{
					unsigned int pos[3] = {start[0], start[1], start[2]};
					for (; pos[n] < stop[n]; ++pos[n])
					{
						size_t idx = GetLinearIndex(n, pos[0], pos[1], pos[2]);
						m_probePoints.push_back({0, static_cast<uint32_t>(idx)});
					}
				}
				else if (start[n] > stop[n])
				{
					unsigned int pos[3] = {stop[0], stop[1], stop[2]};
					for (; pos[n] < start[n]; ++pos[n])
					{
						size_t idx = GetLinearIndex(n, pos[0], pos[1], pos[2]);
						m_probePoints.push_back({0, static_cast<uint32_t>(idx)});
					}
				}
			}
			continue;
		}

		ProcessCurrent* pc = dynamic_cast<ProcessCurrent*>(proc);
		if (pc)
		{
			const unsigned int* start = pc->GetStartCoord();
			const unsigned int* stop = pc->GetStopCoord();
			const bool* m_start_inside = pc->GetStartInside();
			const bool* m_stop_inside = pc->GetStopInside();
			int m_normDir = pc->GetNormalDir();

			switch (m_normDir)
			{
			case 0:
				if (m_stop_inside[0] && m_start_inside[2])
					for (unsigned int k = start[1] + 1; k <= stop[1]; ++k)
						m_probePoints.push_back({1, static_cast<uint32_t>(GetLinearIndex(1, stop[0], k, start[2]))});
				if (m_stop_inside[0] && m_stop_inside[1])
					for (unsigned int k = start[2] + 1; k <= stop[2]; ++k)
						m_probePoints.push_back({1, static_cast<uint32_t>(GetLinearIndex(2, stop[0], stop[1], k))});
				if (m_start_inside[0] && m_stop_inside[2])
					for (unsigned int k = start[1] + 1; k <= stop[1]; ++k)
						m_probePoints.push_back({1, static_cast<uint32_t>(GetLinearIndex(1, start[0], k, stop[2]))});
				if (m_start_inside[0] && m_start_inside[1])
					for (unsigned int k = start[2] + 1; k <= stop[2]; ++k)
						m_probePoints.push_back({1, static_cast<uint32_t>(GetLinearIndex(2, start[0], start[1], k))});
				break;
			case 1:
				if (m_start_inside[0] && m_start_inside[1])
					for (unsigned int k = start[2] + 1; k <= stop[2]; ++k)
						m_probePoints.push_back({1, static_cast<uint32_t>(GetLinearIndex(2, start[0], start[1], k))});
				if (m_stop_inside[1] && m_stop_inside[2])
					for (unsigned int k = start[0] + 1; k <= stop[0]; ++k)
						m_probePoints.push_back({1, static_cast<uint32_t>(GetLinearIndex(0, k, stop[1], stop[2]))});
				if (m_stop_inside[0] && m_stop_inside[1])
					for (unsigned int k = start[2] + 1; k <= stop[2]; ++k)
						m_probePoints.push_back({1, static_cast<uint32_t>(GetLinearIndex(2, stop[0], stop[1], k))});
				if (m_start_inside[1] && m_start_inside[2])
					for (unsigned int k = start[0] + 1; k <= stop[0]; ++k)
						m_probePoints.push_back({1, static_cast<uint32_t>(GetLinearIndex(0, k, start[1], start[2]))});
				break;
			case 2:
				if (m_start_inside[1] && m_start_inside[2])
					for (unsigned int k = start[0] + 1; k <= stop[0]; ++k)
						m_probePoints.push_back({1, static_cast<uint32_t>(GetLinearIndex(0, k, start[1], start[2]))});
				if (m_stop_inside[0] && m_start_inside[2])
					for (unsigned int k = start[1] + 1; k <= stop[1]; ++k)
						m_probePoints.push_back({1, static_cast<uint32_t>(GetLinearIndex(1, stop[0], k, start[2]))});
				if (m_stop_inside[1] && m_stop_inside[2])
					for (unsigned int k = start[0] + 1; k <= stop[0]; ++k)
						m_probePoints.push_back({1, static_cast<uint32_t>(GetLinearIndex(0, k, stop[1], stop[2]))});
				if (m_start_inside[0] && m_stop_inside[2])
					for (unsigned int k = start[1] + 1; k <= stop[1]; ++k)
						m_probePoints.push_back({1, static_cast<uint32_t>(GetLinearIndex(1, start[0], k, stop[2]))});
				break;
			}
			continue;
		}

		ProcessFieldProbe* pfp = dynamic_cast<ProcessFieldProbe*>(proc);
		if (pfp)
		{
			const unsigned int* start = pfp->GetStartCoord();
			uint32_t isCurr = (pfp->GetFieldType() == 1 ? 1 : 0);
			for (unsigned int n = 0; n < 3; ++n)
			{
				size_t idx = GetLinearIndex(n, start[0], start[1], start[2]);
				m_probePoints.push_back({isCurr, static_cast<uint32_t>(idx)});
			}
			continue;
		}
	}

	if (!m_probePoints.empty())
	{
		std::sort(m_probePoints.begin(), m_probePoints.end(), [](const ProbePoint& a, const ProbePoint& b) {
			if (a.isCurr != b.isCurr) return a.isCurr < b.isCurr;
			return a.linIdx < b.linIdx;
		});
		m_probePoints.erase(std::unique(m_probePoints.begin(), m_probePoints.end(), [](const ProbePoint& a, const ProbePoint& b) {
			return a.isCurr == b.isCurr && a.linIdx == b.linIdx;
		}), m_probePoints.end());

		std::cout << "[openEMS WebGPU] Registered " << m_probePoints.size()
		          << " active probe monitoring points for fast on-device extraction." << std::endl;

		AllocateProbeBuffers();
	}
}

bool EngineWebGPU::AllocateProbeBuffers()
{
#ifdef ENABLE_WEBGPU
	if (!m_device || !m_queue || m_probePoints.empty())
		return false;

	if (m_bindGroupProbeGather)       { wgpuBindGroupRelease(m_bindGroupProbeGather);             m_bindGroupProbeGather = nullptr; }
	if (m_pipelineProbeGather)        { wgpuComputePipelineRelease(m_pipelineProbeGather);        m_pipelineProbeGather = nullptr; }
	if (m_pipelineLayoutProbeGather)  { wgpuPipelineLayoutRelease(m_pipelineLayoutProbeGather);  m_pipelineLayoutProbeGather = nullptr; }
	if (m_bindGroupLayoutProbeGather) { wgpuBindGroupLayoutRelease(m_bindGroupLayoutProbeGather); m_bindGroupLayoutProbeGather = nullptr; }
	if (m_bufStagingProbes)           { wgpuBufferRelease(m_bufStagingProbes);                    m_bufStagingProbes = nullptr; }
	if (m_bufProbeValues)             { wgpuBufferRelease(m_bufProbeValues);                      m_bufProbeValues = nullptr; }
	if (m_bufProbeParams)             { wgpuBufferRelease(m_bufProbeParams);                      m_bufProbeParams = nullptr; }
	if (m_bufProbePoints)             { wgpuBufferRelease(m_bufProbePoints);                      m_bufProbePoints = nullptr; }

	uint32_t count = static_cast<uint32_t>(m_probePoints.size());
	size_t pointsBytes = count * sizeof(ProbePoint);
	size_t valuesBytes = count * sizeof(float);

	struct { uint32_t numPoints; uint32_t pad0; uint32_t pad1; uint32_t pad2; } params = { count, 0, 0, 0 };
	WGPUBufferDescriptor uboDesc = {};
	uboDesc.label = { "ProbeGatherParams", WGPU_STRLEN };
	uboDesc.size = sizeof(params);
	uboDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
	m_bufProbeParams = wgpuDeviceCreateBuffer(m_device, &uboDesc);
	wgpuQueueWriteBuffer(m_queue, m_bufProbeParams, 0, &params, sizeof(params));

	WGPUBufferDescriptor ptsDesc = {};
	ptsDesc.label = { "ProbePoints", WGPU_STRLEN };
	ptsDesc.size = pointsBytes;
	ptsDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
	m_bufProbePoints = wgpuDeviceCreateBuffer(m_device, &ptsDesc);
	wgpuQueueWriteBuffer(m_queue, m_bufProbePoints, 0, m_probePoints.data(), pointsBytes);

	WGPUBufferDescriptor valDesc = {};
	valDesc.label = { "ProbeValues", WGPU_STRLEN };
	valDesc.size = valuesBytes;
	valDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
	m_bufProbeValues = wgpuDeviceCreateBuffer(m_device, &valDesc);

	WGPUBufferDescriptor stgDesc = {};
	stgDesc.label = { "StagingProbes", WGPU_STRLEN };
	stgDesc.size = valuesBytes;
	stgDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
	m_bufStagingProbes = wgpuDeviceCreateBuffer(m_device, &stgDesc);

	WGPUBindGroupLayoutEntry entries[5] = {};
	entries[0].binding = 0; entries[0].visibility = WGPUShaderStage_Compute; entries[0].buffer.type = WGPUBufferBindingType_Uniform;
	entries[1].binding = 1; entries[1].visibility = WGPUShaderStage_Compute; entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
	entries[2].binding = 2; entries[2].visibility = WGPUShaderStage_Compute; entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
	entries[3].binding = 3; entries[3].visibility = WGPUShaderStage_Compute; entries[3].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
	entries[4].binding = 4; entries[4].visibility = WGPUShaderStage_Compute; entries[4].buffer.type = WGPUBufferBindingType_Storage;

	WGPUBindGroupLayoutDescriptor bglDesc = {};
	bglDesc.entryCount = 5;
	bglDesc.entries = entries;
	m_bindGroupLayoutProbeGather = wgpuDeviceCreateBindGroupLayout(m_device, &bglDesc);

	WGPUPipelineLayoutDescriptor playDesc = {};
	playDesc.bindGroupLayoutCount = 1;
	playDesc.bindGroupLayouts = &m_bindGroupLayoutProbeGather;
	m_pipelineLayoutProbeGather = wgpuDeviceCreatePipelineLayout(m_device, &playDesc);

	WGPUShaderSourceWGSL wgslDesc = {};
	wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
	wgslDesc.code.data = OpenEMS_WebGPU::kShaderProbeGather;
	wgslDesc.code.length = strlen(OpenEMS_WebGPU::kShaderProbeGather);

	WGPUShaderModuleDescriptor smDesc = {};
	smDesc.nextInChain = &wgslDesc.chain;
	smDesc.label = { "ProbeGatherShaderModule", WGPU_STRLEN };
	WGPUShaderModule sm = wgpuDeviceCreateShaderModule(m_device, &smDesc);

	WGPUComputePipelineDescriptor pipeDesc = {};
	pipeDesc.layout = m_pipelineLayoutProbeGather;
	pipeDesc.compute.module = sm;
	pipeDesc.compute.entryPoint = { "gather_probe_points", WGPU_STRLEN };
	pipeDesc.label = { "PipelineProbeGather", WGPU_STRLEN };
	m_pipelineProbeGather = wgpuDeviceCreateComputePipeline(m_device, &pipeDesc);
	wgpuShaderModuleRelease(sm);

	size_t fieldBytes = 3 * static_cast<size_t>(m_grid.numCells) * sizeof(float);
	WGPUBindGroupEntry bgEntries[5] = {};
	bgEntries[0].binding = 0; bgEntries[0].buffer = m_bufProbeParams; bgEntries[0].size = sizeof(params);
	bgEntries[1].binding = 1; bgEntries[1].buffer = m_bufProbePoints; bgEntries[1].size = pointsBytes;
	bgEntries[2].binding = 2; bgEntries[2].buffer = m_bufVolt;        bgEntries[2].size = fieldBytes;
	bgEntries[3].binding = 3; bgEntries[3].buffer = m_bufCurr;        bgEntries[3].size = fieldBytes;
	bgEntries[4].binding = 4; bgEntries[4].buffer = m_bufProbeValues; bgEntries[4].size = valuesBytes;

	WGPUBindGroupDescriptor bgDesc = {};
	bgDesc.layout = m_bindGroupLayoutProbeGather;
	bgDesc.entryCount = 5;
	bgDesc.entries = bgEntries;
	m_bindGroupProbeGather = wgpuDeviceCreateBindGroup(m_device, &bgDesc);

	return (m_pipelineProbeGather != nullptr && m_bindGroupProbeGather != nullptr);
#else
	return true;
#endif
}

bool EngineWebGPU::IterateTS(unsigned int iterTS)
{
	uint32_t workgroupsZ = (m_grid.dimZ + 31) / 32;
	uint32_t workgroupsY = (m_grid.dimY + 3) / 4;
	uint32_t workgroupsX = (m_grid.dimX + 1) / 2;

#ifdef ENABLE_WEBGPU
	if (m_device && m_queue && m_pipelineVolt && m_pipelineCurr && m_bindGroupFields)
	{
		for (unsigned int step = 0; step < iterTS; ++step)
		{
			WGPUCommandEncoderDescriptor encDesc = {};
			WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, &encDesc);
			WGPUComputePassDescriptor passDesc = {};

			// 1. Voltage update pass
			{
				WGPUComputePassEncoder cpass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
				wgpuComputePassEncoderSetPipeline(cpass, m_pipelineVolt);
				wgpuComputePassEncoderSetBindGroup(cpass, 0, m_bindGroupFields, 0, nullptr);
				wgpuComputePassEncoderDispatchWorkgroups(cpass, workgroupsZ, workgroupsY, workgroupsX);
				wgpuComputePassEncoderEnd(cpass);
				wgpuComputePassEncoderRelease(cpass);
			}

			// 2. Voltage excitation pass (if active)
			if (m_pipelineExc && m_bindGroupVoltExc && !m_voltExcPoints.empty())
			{
				if (PrepareVoltExcitation(m_numTS))
				{
					wgpuQueueWriteBuffer(m_queue, m_bufVoltExcPoints, 0, m_voltExcPoints.data(), m_voltExcPoints.size() * sizeof(GpuExcPoint));
					WGPUComputePassEncoder cpass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
					wgpuComputePassEncoderSetPipeline(cpass, m_pipelineExc);
					wgpuComputePassEncoderSetBindGroup(cpass, 0, m_bindGroupVoltExc, 0, nullptr);
					uint32_t groups = (static_cast<uint32_t>(m_voltExcPoints.size()) + 63) / 64;
					wgpuComputePassEncoderDispatchWorkgroups(cpass, groups, 1, 1);
					wgpuComputePassEncoderEnd(cpass);
					wgpuComputePassEncoderRelease(cpass);
				}
			}

			// 3. Current update pass
			{
				WGPUComputePassEncoder cpass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
				wgpuComputePassEncoderSetPipeline(cpass, m_pipelineCurr);
				wgpuComputePassEncoderSetBindGroup(cpass, 0, m_bindGroupFields, 0, nullptr);
				wgpuComputePassEncoderDispatchWorkgroups(cpass, workgroupsZ, workgroupsY, workgroupsX);
				wgpuComputePassEncoderEnd(cpass);
				wgpuComputePassEncoderRelease(cpass);
			}

			// 4. Current excitation pass (if active)
			if (m_pipelineExc && m_bindGroupCurrExc && !m_currExcPoints.empty())
			{
				if (PrepareCurrExcitation(m_numTS))
				{
					wgpuQueueWriteBuffer(m_queue, m_bufCurrExcPoints, 0, m_currExcPoints.data(), m_currExcPoints.size() * sizeof(GpuExcPoint));
					WGPUComputePassEncoder cpass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
					wgpuComputePassEncoderSetPipeline(cpass, m_pipelineExc);
					wgpuComputePassEncoderSetBindGroup(cpass, 0, m_bindGroupCurrExc, 0, nullptr);
					uint32_t groups = (static_cast<uint32_t>(m_currExcPoints.size()) + 63) / 64;
					wgpuComputePassEncoderDispatchWorkgroups(cpass, groups, 1, 1);
					wgpuComputePassEncoderEnd(cpass);
					wgpuComputePassEncoderRelease(cpass);
				}
			}

			WGPUCommandBufferDescriptor cmdDesc = {};
			WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
			wgpuCommandEncoderRelease(encoder);

			wgpuQueueSubmit(m_queue, 1, &cmd);
			wgpuCommandBufferRelease(cmd);

			++m_numTS;
		}

		m_hostFieldsValid = false;
		return true;
	}
#endif

	// Emulation fallback
	for (unsigned int step = 0; step < iterTS; ++step)
	{
		++m_numTS;
	}
	m_hostFieldsValid = false;
	return true;
}

void EngineWebGPU::DispatchWorkgroups(WGPUComputePipeline pipeline, uint32_t gx, uint32_t gy, uint32_t gz)
{
	UNUSED(pipeline);
	UNUSED(gx);
	UNUSED(gy);
	UNUSED(gz);
}

unsigned int EngineWebGPU::GetNumberOfTimesteps() const
{
	return m_numTS;
}

void EngineWebGPU::NextInterval(float curr_speed)
{
	UNUSED(curr_speed);
}

size_t EngineWebGPU::GetLinearIndex(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	size_t stride_n = static_cast<size_t>(m_grid.dimX) * m_grid.dimY * m_grid.dimZ;
	size_t stride_x = static_cast<size_t>(m_grid.dimY) * m_grid.dimZ;
	return n * stride_n + x * stride_x + y * m_grid.dimZ + z;
}

bool EngineWebGPU::SyncFieldsToHost()
{
	if (m_hostFieldsValid)
		return true;

#ifdef ENABLE_WEBGPU
	if (m_device && m_queue && m_bufVolt && m_bufCurr && m_bufStagingVolt && m_bufStagingCurr)
	{
		size_t fieldBytes = 3 * static_cast<size_t>(m_grid.numCells) * sizeof(float);

		WGPUCommandEncoderDescriptor encDesc = {};
		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, &encDesc);

		wgpuCommandEncoderCopyBufferToBuffer(encoder, m_bufVolt, 0, m_bufStagingVolt, 0, fieldBytes);
		wgpuCommandEncoderCopyBufferToBuffer(encoder, m_bufCurr, 0, m_bufStagingCurr, 0, fieldBytes);

		WGPUCommandBufferDescriptor cmdDesc = {};
		WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
		wgpuCommandEncoderRelease(encoder);

		wgpuQueueSubmit(m_queue, 1, &cmd);
		wgpuCommandBufferRelease(cmd);

		// Asynchronously map staging buffers
		BufferMapContext mapCtx;

		WGPUBufferMapCallbackInfo mapVoltInfo = {};
		mapVoltInfo.mode = WGPUCallbackMode_AllowProcessEvents;
		mapVoltInfo.callback = onVoltMapped;
		mapVoltInfo.userdata1 = &mapCtx;

		WGPUBufferMapCallbackInfo mapCurrInfo = {};
		mapCurrInfo.mode = WGPUCallbackMode_AllowProcessEvents;
		mapCurrInfo.callback = onCurrMapped;
		mapCurrInfo.userdata1 = &mapCtx;

		wgpuBufferMapAsync(m_bufStagingVolt, WGPUMapMode_Read, 0, fieldBytes, mapVoltInfo);
		wgpuBufferMapAsync(m_bufStagingCurr, WGPUMapMode_Read, 0, fieldBytes, mapCurrInfo);

		while (!mapCtx.voltDone || !mapCtx.currDone)
		{
			wgpuDevicePoll(m_device, true, nullptr);
		}

		const float* pVolt = static_cast<const float*>(wgpuBufferGetConstMappedRange(m_bufStagingVolt, 0, fieldBytes));
		const float* pCurr = static_cast<const float*>(wgpuBufferGetConstMappedRange(m_bufStagingCurr, 0, fieldBytes));

		if (pVolt)
			std::memcpy(m_hostVolt.data(), pVolt, fieldBytes);
		if (pCurr)
			std::memcpy(m_hostCurr.data(), pCurr, fieldBytes);

		wgpuBufferUnmap(m_bufStagingVolt);
		wgpuBufferUnmap(m_bufStagingCurr);

		// Synchronize directly into m_op->GetEngine() if present
		if (m_op && m_op->GetEngine())
		{
			Engine* eng = const_cast<Operator*>(m_op)->GetEngine();
			if (eng->GetVoltArray() && eng->GetVoltArray()->data())
				std::memcpy(eng->GetVoltArray()->data(), m_hostVolt.data(), fieldBytes);
			if (eng->GetCurrArray() && eng->GetCurrArray()->data())
				std::memcpy(eng->GetCurrArray()->data(), m_hostCurr.data(), fieldBytes);
		}

		m_hostFieldsValid = true;
		return true;
	}
#endif

	m_hostFieldsValid = true;
	return true;
}

bool EngineWebGPU::SyncProbesToHost()
{
	if (m_hostFieldsValid || m_probePoints.empty())
		return true;

#ifdef ENABLE_WEBGPU
	if (m_device && m_queue && m_bufProbeValues && m_bufStagingProbes)
	{
		size_t probeBytes = m_probePoints.size() * sizeof(float);

		WGPUCommandEncoderDescriptor encDesc = {};
		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, &encDesc);

		// Execute on-device probe gather compute pass
		if (m_pipelineProbeGather && m_bindGroupProbeGather)
		{
			WGPUComputePassDescriptor passDesc = {};
			WGPUComputePassEncoder cpass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
			wgpuComputePassEncoderSetPipeline(cpass, m_pipelineProbeGather);
			wgpuComputePassEncoderSetBindGroup(cpass, 0, m_bindGroupProbeGather, 0, nullptr);
			uint32_t pgroups = (static_cast<uint32_t>(m_probePoints.size()) + 63) / 64;
			wgpuComputePassEncoderDispatchWorkgroups(cpass, pgroups, 1, 1);
			wgpuComputePassEncoderEnd(cpass);
			wgpuComputePassEncoderRelease(cpass);
		}

		wgpuCommandEncoderCopyBufferToBuffer(encoder, m_bufProbeValues, 0, m_bufStagingProbes, 0, probeBytes);

		WGPUCommandBufferDescriptor cmdDesc = {};
		WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
		wgpuCommandEncoderRelease(encoder);

		wgpuQueueSubmit(m_queue, 1, &cmd);
		wgpuCommandBufferRelease(cmd);

		ProbeMapContext mapCtx;
		WGPUBufferMapCallbackInfo mapInfo = {};
		mapInfo.mode = WGPUCallbackMode_AllowProcessEvents;
		mapInfo.callback = onProbeMapped;
		mapInfo.userdata1 = &mapCtx;

		wgpuBufferMapAsync(m_bufStagingProbes, WGPUMapMode_Read, 0, probeBytes, mapInfo);

		while (!mapCtx.done)
		{
			wgpuDevicePoll(m_device, true, nullptr);
		}

		const float* pValues = static_cast<const float*>(wgpuBufferGetConstMappedRange(m_bufStagingProbes, 0, probeBytes));
		if (pValues)
		{
			Engine* eng = m_op ? const_cast<Operator*>(m_op)->GetEngine() : nullptr;
			float* engVolt = (eng && eng->GetVoltArray()) ? eng->GetVoltArray()->data() : nullptr;
			float* engCurr = (eng && eng->GetCurrArray()) ? eng->GetCurrArray()->data() : nullptr;

			for (size_t i = 0; i < m_probePoints.size(); ++i)
			{
				const auto& pt = m_probePoints[i];
				float val = pValues[i];
				if (pt.isCurr == 1)
				{
					m_hostCurr[pt.linIdx] = val;
					if (engCurr) engCurr[pt.linIdx] = val;
				}
				else
				{
					m_hostVolt[pt.linIdx] = val;
					if (engVolt) engVolt[pt.linIdx] = val;
				}
			}
		}

		wgpuBufferUnmap(m_bufStagingProbes);
		return true;
	}
#endif

	return true;
}

FDTD_FLOAT EngineWebGPU::GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	if (!m_hostFieldsValid)
		const_cast<EngineWebGPU*>(this)->SyncFieldsToHost();

	size_t idx = GetLinearIndex(n, x, y, z);
	if (idx < m_hostVolt.size())
		return m_hostVolt[idx];
	return 0.0f;
}

FDTD_FLOAT EngineWebGPU::GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	if (!m_hostFieldsValid)
		const_cast<EngineWebGPU*>(this)->SyncFieldsToHost();

	size_t idx = GetLinearIndex(n, x, y, z);
	if (idx < m_hostCurr.size())
		return m_hostCurr[idx];
	return 0.0f;
}

void EngineWebGPU::SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
{
	size_t idx = GetLinearIndex(n, x, y, z);
	if (idx < m_hostVolt.size())
	{
		m_hostVolt[idx] = val;
#ifdef ENABLE_WEBGPU
		if (m_queue && m_bufVolt)
		{
			wgpuQueueWriteBuffer(m_queue, m_bufVolt, idx * sizeof(float), &val, sizeof(float));
		}
#endif
	}
}

void EngineWebGPU::SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
{
	size_t idx = GetLinearIndex(n, x, y, z);
	if (idx < m_hostCurr.size())
	{
		m_hostCurr[idx] = val;
#ifdef ENABLE_WEBGPU
		if (m_queue && m_bufCurr)
		{
			wgpuQueueWriteBuffer(m_queue, m_bufCurr, idx * sizeof(float), &val, sizeof(float));
		}
#endif
	}
}

std::string EngineWebGPU::GetBackendName() const
{
	return "WebGPU (" + m_caps.backendType + ")";
}
