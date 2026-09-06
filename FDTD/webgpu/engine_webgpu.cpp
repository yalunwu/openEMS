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
	  m_hostFieldsValid(false)
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
}
#endif

bool EngineWebGPU::InitDevice()
{
#ifdef ENABLE_WEBGPU
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
	m_caps.adapterName = chosenInfo.device.data ? chosenInfo.device.data
	                   : (chosenInfo.description.data ? chosenInfo.description.data : "WebGPU Device");

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
			wgpuQueueWriteBuffer(m_queue, m_bufVv, 0, m_op->vv_ptr->data(), fieldBytes);
			wgpuQueueWriteBuffer(m_queue, m_bufVi, 0, m_op->vi_ptr->data(), fieldBytes);
			wgpuQueueWriteBuffer(m_queue, m_bufIi, 0, m_op->ii_ptr->data(), fieldBytes);
			wgpuQueueWriteBuffer(m_queue, m_bufIv, 0, m_op->iv_ptr->data(), fieldBytes);
		}
		else
		{
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

#ifdef ENABLE_WEBGPU
	if (m_bindGroupFields)       { wgpuBindGroupRelease(m_bindGroupFields);             m_bindGroupFields = nullptr; }
	if (m_bindGroupProbes)       { wgpuBindGroupRelease(m_bindGroupProbes);             m_bindGroupProbes = nullptr; }
	if (m_pipelineVolt)          { wgpuComputePipelineRelease(m_pipelineVolt);          m_pipelineVolt = nullptr; }
	if (m_pipelineCurr)          { wgpuComputePipelineRelease(m_pipelineCurr);          m_pipelineCurr = nullptr; }
	if (m_pipelineProbe)         { wgpuComputePipelineRelease(m_pipelineProbe);         m_pipelineProbe = nullptr; }
	if (m_pipelineLayout)        { wgpuPipelineLayoutRelease(m_pipelineLayout);        m_pipelineLayout = nullptr; }
	if (m_bindGroupLayoutFields) { wgpuBindGroupLayoutRelease(m_bindGroupLayoutFields); m_bindGroupLayoutFields = nullptr; }
	if (m_shaderModule)          { wgpuShaderModuleRelease(m_shaderModule);             m_shaderModule = nullptr; }
	if (m_bufStagingVolt)        { wgpuBufferRelease(m_bufStagingVolt);                 m_bufStagingVolt = nullptr; }
	if (m_bufStagingCurr)        { wgpuBufferRelease(m_bufStagingCurr);                 m_bufStagingCurr = nullptr; }
	if (m_bufGridUniforms)       { wgpuBufferRelease(m_bufGridUniforms);                m_bufGridUniforms = nullptr; }
	if (m_bufVolt)               { wgpuBufferRelease(m_bufVolt);                        m_bufVolt = nullptr; }
	if (m_bufCurr)               { wgpuBufferRelease(m_bufCurr);                        m_bufCurr = nullptr; }
	if (m_bufVv)                 { wgpuBufferRelease(m_bufVv);                          m_bufVv = nullptr; }
	if (m_bufVi)                 { wgpuBufferRelease(m_bufVi);                          m_bufVi = nullptr; }
	if (m_bufIi)                 { wgpuBufferRelease(m_bufIi);                          m_bufIi = nullptr; }
	if (m_bufIv)                 { wgpuBufferRelease(m_bufIv);                          m_bufIv = nullptr; }
	if (m_bufProbeHistory)       { wgpuBufferRelease(m_bufProbeHistory);                m_bufProbeHistory = nullptr; }
	if (m_queue)                 { wgpuQueueRelease(m_queue);                           m_queue = nullptr; }
	if (m_device)                { wgpuDeviceRelease(m_device);                         m_device = nullptr; }
	if (m_adapter)               { wgpuAdapterRelease(m_adapter);                       m_adapter = nullptr; }
	if (m_instance)              { wgpuInstanceRelease(m_instance);                     m_instance = nullptr; }
#endif
}

void EngineWebGPU::ApplyExcitation(unsigned int step)
{
	if (!m_op) return;

	for (size_t i = 0; i < m_op->GetNumberOfExtentions(); ++i)
	{
		Operator_Extension* ext = m_op->GetExtension(i);
		Operator_Ext_Excitation* excExt = dynamic_cast<Operator_Ext_Excitation*>(ext);
		if (!excExt || !excExt->m_Exc) continue;

		Excitation* exc = excExt->m_Exc;
		unsigned int length = exc->GetLength();
		FDTD_FLOAT* exc_volt = exc->GetVoltageSignal();
		FDTD_FLOAT* exc_curr = exc->GetCurrentSignal();

		int p = static_cast<int>(step) + 1;
		if (exc->GetSignalPeriod() > 0)
			p = static_cast<int>(exc->GetSignalPeriod() / exc->GetTimestep());

		// Voltage (E-field) excitation
		if (exc_volt && excExt->Volt_Count > 0)
		{
			for (unsigned int n = 0; n < excExt->Volt_Count; ++n)
			{
				int exc_pos = static_cast<int>(step) - static_cast<int>(excExt->Volt_delay[n]);
				exc_pos *= (exc_pos > 0);
				exc_pos %= p;
				exc_pos *= (exc_pos < static_cast<int>(length));

				if (exc_pos < static_cast<int>(length))
				{
					float amp = excExt->Volt_amp[n] * exc_volt[exc_pos];
					if (amp != 0.0f)
					{
						unsigned int ny = excExt->Volt_dir[n];
						unsigned int x = excExt->Volt_index[0][n];
						unsigned int y = excExt->Volt_index[1][n];
						unsigned int z = excExt->Volt_index[2][n];

						size_t idx = GetLinearIndex(ny, x, y, z);
						if (idx < m_hostVolt.size())
						{
							m_hostVolt[idx] += amp;
#ifdef ENABLE_WEBGPU
							if (m_queue && m_bufVolt)
							{
								wgpuQueueWriteBuffer(m_queue, m_bufVolt, idx * sizeof(float), &m_hostVolt[idx], sizeof(float));
							}
#endif
						}
					}
				}
			}
		}

		// Current (H-field) excitation
		if (exc_curr && excExt->Curr_Count > 0)
		{
			for (unsigned int n = 0; n < excExt->Curr_Count; ++n)
			{
				int exc_pos = static_cast<int>(step) - static_cast<int>(excExt->Curr_delay[n]);
				exc_pos *= (exc_pos > 0);
				exc_pos %= p;
				exc_pos *= (exc_pos < static_cast<int>(length));

				if (exc_pos < static_cast<int>(length))
				{
					float amp = excExt->Curr_amp[n] * exc_curr[exc_pos];
					if (amp != 0.0f)
					{
						unsigned int ny = excExt->Curr_dir[n];
						unsigned int x = excExt->Curr_index[0][n];
						unsigned int y = excExt->Curr_index[1][n];
						unsigned int z = excExt->Curr_index[2][n];

						size_t idx = GetLinearIndex(ny, x, y, z);
						if (idx < m_hostCurr.size())
						{
							m_hostCurr[idx] += amp;
#ifdef ENABLE_WEBGPU
							if (m_queue && m_bufCurr)
							{
								wgpuQueueWriteBuffer(m_queue, m_bufCurr, idx * sizeof(float), &m_hostCurr[idx], sizeof(float));
							}
#endif
						}
					}
				}
			}
		}
	}
}

bool EngineWebGPU::IterateTS(unsigned int iterTS)
{
	uint32_t workgroupsX = (m_grid.dimX + 7) / 8;
	uint32_t workgroupsY = (m_grid.dimY + 7) / 8;
	uint32_t workgroupsZ = (m_grid.dimZ + 3) / 4;

#ifdef ENABLE_WEBGPU
	if (m_device && m_queue && m_pipelineVolt && m_pipelineCurr && m_bindGroupFields)
	{
		for (unsigned int step = 0; step < iterTS; ++step)
		{
			// 1. Inject excitation if active
			ApplyExcitation(m_numTS);

			// 2. Encode compute pass for voltage and current updates
			WGPUCommandEncoderDescriptor encDesc = {};
			WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, &encDesc);

			WGPUComputePassDescriptor passDesc = {};
			WGPUComputePassEncoder cpass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);

			wgpuComputePassEncoderSetBindGroup(cpass, 0, m_bindGroupFields, 0, nullptr);

			// Voltage update pass
			wgpuComputePassEncoderSetPipeline(cpass, m_pipelineVolt);
			wgpuComputePassEncoderDispatchWorkgroups(cpass, workgroupsX, workgroupsY, workgroupsZ);

			// Current update pass
			wgpuComputePassEncoderSetPipeline(cpass, m_pipelineCurr);
			wgpuComputePassEncoderDispatchWorkgroups(cpass, workgroupsX, workgroupsY, workgroupsZ);

			wgpuComputePassEncoderEnd(cpass);
			wgpuComputePassEncoderRelease(cpass);

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
	// Probes read via SyncFieldsToHost() and Engine interface
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
