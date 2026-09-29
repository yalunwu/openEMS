#include "engine_vulkan.h"
#include "shaders_glsl.h"

#include <iostream>
#include <algorithm>
#include <cstring>
#include <chrono>

#include "FDTD/operator.h"
#include "FDTD/engine.h"
#include "FDTD/excitation.h"
#include "FDTD/extensions/operator_ext_excitation.h"
#include "FDTD/extensions/operator_ext_upml.h"
#include "FDTD/extensions/operator_ext_mur_abc.h"
#include "FDTD/extensions/operator_ext_steadystate.h"
#include "FDTD/extensions/operator_ext_tfsf.h"
#include "FDTD/extensions/operator_ext_lumpedRLC.h"
#include "FDTD/extensions/operator_ext_absorbing_bc.h"
#include "FDTD/extensions/operator_ext_lorentzmaterial.h"
#include "FDTD/extensions/operator_ext_conductingsheet.h"
#include "FDTD/extensions/operator_ext_cylinder.h"
#include "Common/processvoltage.h"
#include "Common/processcurrent.h"
#include "Common/processfieldprobe.h"
#include "Common/processing.h"
#include "CSXCAD/ContinuousStructure.h"

#ifdef ENABLE_VULKAN
#include <shaderc/shaderc.hpp>

static std::vector<uint32_t> CompileGLSLToSpirv(const std::string& source, shaderc_shader_kind kind, const char* name)
{
	shaderc::Compiler compiler;
	shaderc::CompileOptions options;
	options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
	options.SetOptimizationLevel(shaderc_optimization_level_performance);

	auto result = compiler.CompileGlslToSpv(source, kind, name, options);
	if (result.GetCompilationStatus() != shaderc_compilation_status_success)
	{
		std::cerr << "[openEMS Vulkan] Shader compile error (" << name << "): "
		          << result.GetErrorMessage() << std::endl;
		return {};
	}
	return {result.begin(), result.end()};
}

#endif // ENABLE_VULKAN

EngineVulkan::EngineVulkan(const Operator* op)
	: m_op(op), m_pa(nullptr), m_numTS(0), m_hostFieldsValid(true), m_hostFieldsDirty(true)
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

EngineVulkan::~EngineVulkan()
{
	Reset();
}

bool EngineVulkan::CheckModelSupport(const Operator* op, const ContinuousStructure* csx, std::string& unsupportedReason)
{
	return EngineBackend::CheckModelSupport(op, csx, unsupportedReason);
}

bool EngineVulkan::Initialize()
{
	std::cout << "[openEMS Vulkan] Initializing Vulkan 1.2 compute engine..." << std::endl;

#ifdef ENABLE_VULKAN
	if (!InitVulkan())
	{
		std::cerr << "[openEMS Vulkan] Failed to initialize Vulkan device and queues." << std::endl;
		return false;
	}

	if (!AllocateBuffers())
	{
		std::cerr << "[openEMS Vulkan] Failed to allocate Vulkan device buffers." << std::endl;
		return false;
	}

	if (!CreatePipelines())
	{
		std::cerr << "[openEMS Vulkan] Failed to compile and create compute pipelines." << std::endl;
		return false;
	}

	if (!AllocateExcitationBuffers())
	{
		std::cerr << "[openEMS Vulkan] Failed to allocate excitation buffers." << std::endl;
		return false;
	}

	if (!AllocateUpmlBuffers())
	{
		std::cerr << "[openEMS Vulkan] Failed to allocate UPML buffers." << std::endl;
		return false;
	}

	if (!AllocateMurBuffers())
	{
		std::cerr << "[openEMS Vulkan] Failed to allocate Mur ABC buffers." << std::endl;
		return false;
	}

	if (!AllocateTfsfBuffers())
	{
		std::cerr << "[openEMS Vulkan] Failed to allocate TFSF buffers." << std::endl;
		return false;
	}

	if (!AllocateRlcBuffers())
	{
		std::cerr << "[openEMS Vulkan] Failed to allocate Lumped RLC buffers." << std::endl;
		return false;
	}

	if (!AllocateAbsorbingBCBuffers())
	{
		std::cerr << "[openEMS Vulkan] Failed to allocate Absorbing BC buffers." << std::endl;
		return false;
	}

	if (!AllocateDispersiveBuffers())
	{
		std::cerr << "[openEMS Vulkan] Failed to allocate Dispersive Media buffers." << std::endl;
		return false;
	}

	if (!AllocateCylinderBuffers())
	{
		std::cerr << "[openEMS Vulkan] Failed to allocate Cylindrical Coordinates buffers." << std::endl;
		return false;
	}

	std::cout << "[openEMS Vulkan] Backend ready. Device: " << m_deviceName << std::endl;
	return true;
#else
	std::cerr << "[openEMS Vulkan] Engine built without ENABLE_VULKAN support!" << std::endl;
	return false;
#endif
}

#ifdef ENABLE_VULKAN

uint32_t EngineVulkan::FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
	{
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}
	return UINT32_MAX;
}

bool EngineVulkan::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VulkanBuffer& outBuf)
{
	outBuf.size = size;

	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = usage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &outBuf.buffer) != VK_SUCCESS)
	{
		return false;
	}

	VkMemoryRequirements memReqs;
	vkGetBufferMemoryRequirements(m_device, outBuf.buffer, &memReqs);

	uint32_t memTypeIndex = FindMemoryType(memReqs.memoryTypeBits, properties);
	if (memTypeIndex == UINT32_MAX)
	{
		vkDestroyBuffer(m_device, outBuf.buffer, nullptr);
		outBuf.buffer = VK_NULL_HANDLE;
		return false;
	}

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReqs.size;
	allocInfo.memoryTypeIndex = memTypeIndex;

	if (vkAllocateMemory(m_device, &allocInfo, nullptr, &outBuf.memory) != VK_SUCCESS)
	{
		vkDestroyBuffer(m_device, outBuf.buffer, nullptr);
		outBuf.buffer = VK_NULL_HANDLE;
		return false;
	}

	if (vkBindBufferMemory(m_device, outBuf.buffer, outBuf.memory, 0) != VK_SUCCESS)
	{
		vkFreeMemory(m_device, outBuf.memory, nullptr);
		vkDestroyBuffer(m_device, outBuf.buffer, nullptr);
		outBuf.buffer = VK_NULL_HANDLE;
		outBuf.memory = VK_NULL_HANDLE;
		return false;
	}

	if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
	{
		vkMapMemory(m_device, outBuf.memory, 0, size, 0, &outBuf.mapped);
	}

	return true;
}

void EngineVulkan::DestroyBuffer(VulkanBuffer& buf)
{
	if (m_device != VK_NULL_HANDLE)
	{
		if (buf.mapped)
		{
			vkUnmapMemory(m_device, buf.memory);
			buf.mapped = nullptr;
		}
		if (buf.buffer != VK_NULL_HANDLE)
		{
			vkDestroyBuffer(m_device, buf.buffer, nullptr);
			buf.buffer = VK_NULL_HANDLE;
		}
		if (buf.memory != VK_NULL_HANDLE)
		{
			vkFreeMemory(m_device, buf.memory, nullptr);
			buf.memory = VK_NULL_HANDLE;
		}
	}
	buf.size = 0;
}

VkShaderModule EngineVulkan::CreateShaderModule(const std::vector<uint32_t>& spirv)
{
	if (spirv.empty()) return VK_NULL_HANDLE;

	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = spirv.size() * sizeof(uint32_t);
	createInfo.pCode = spirv.data();

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(m_device, &createInfo, nullptr, &module) != VK_SUCCESS)
	{
		return VK_NULL_HANDLE;
	}
	return module;
}

bool EngineVulkan::InitVulkan()
{
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "openEMS";
	appInfo.applicationVersion = VK_MAKE_VERSION(0, 37, 0);
	appInfo.pEngineName = "openEMS Vulkan Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_2;

	VkInstanceCreateInfo instInfo = {};
	instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instInfo.pApplicationInfo = &appInfo;

	if (vkCreateInstance(&instInfo, nullptr, &m_instance) != VK_SUCCESS)
	{
		return false;
	}

	uint32_t devCount = 0;
	vkEnumeratePhysicalDevices(m_instance, &devCount, nullptr);
	if (devCount == 0) return false;

	std::vector<VkPhysicalDevice> devices(devCount);
	vkEnumeratePhysicalDevices(m_instance, &devCount, devices.data());

	// Select discrete GPU if available, else first compatible device
	m_physicalDevice = devices[0];
	for (const auto& d : devices)
	{
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(d, &props);
		if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		{
			m_physicalDevice = d;
			m_deviceName = props.deviceName;
			break;
		}
	}
	if (m_deviceName == "Vulkan GPU")
	{
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
		m_deviceName = props.deviceName;
	}

	// Find compute queue family
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueProps(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, queueProps.data());

	bool foundCompute = false;
	for (uint32_t i = 0; i < queueFamilyCount; ++i)
	{
		if (queueProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
		{
			m_computeQueueFamily = i;
			foundCompute = true;
			break;
		}
	}
	if (!foundCompute) return false;

	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo = {};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = m_computeQueueFamily;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = &queuePriority;

	VkDeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

	if (vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device) != VK_SUCCESS)
	{
		return false;
	}

	vkGetDeviceQueue(m_device, m_computeQueueFamily, 0, &m_computeQueue);

	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = m_computeQueueFamily;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_cmdPool) != VK_SUCCESS)
	{
		return false;
	}

	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = m_cmdPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	if (vkAllocateCommandBuffers(m_device, &allocInfo, &m_cmdBuffer) != VK_SUCCESS)
	{
		return false;
	}

	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	if (vkCreateFence(m_device, &fenceInfo, nullptr, &m_fence) != VK_SUCCESS)
	{
		return false;
	}

	return true;
}

bool EngineVulkan::AllocateBuffers()
{
	size_t fieldBytes = 3 * static_cast<size_t>(m_grid.numCells) * sizeof(float);

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	if (!CreateBuffer(fieldBytes, storageUsage, devLocal, m_bufVv)) return false;
	if (!CreateBuffer(fieldBytes, storageUsage, devLocal, m_bufVi)) return false;
	if (!CreateBuffer(fieldBytes, storageUsage, devLocal, m_bufIi)) return false;
	if (!CreateBuffer(fieldBytes, storageUsage, devLocal, m_bufIv)) return false;
	if (!CreateBuffer(fieldBytes, storageUsage, devLocal, m_bufVolt)) return false;
	if (!CreateBuffer(fieldBytes, storageUsage, devLocal, m_bufCurr)) return false;

	// Staging buffer in host memory for transfers
	if (!CreateBuffer(fieldBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_bufFieldStaging))
	{
		return false;
	}

	// Upload initial material coefficient matrices if operator is present
	if (m_op)
	{
		std::cout << "[openEMS Vulkan] Uploading operator material matrices (vv, vi, ii, iv) to GPU..." << std::endl;
		std::vector<float> hostCoeff(3 * m_grid.numCells, 0.0f);

		auto uploadMatrix = [&](VulkanBuffer& targetBuf, int matrix) {
			size_t idx = 0;
			for (unsigned int n = 0; n < 3; ++n)
			{
				for (unsigned int x = 0; x < m_grid.dimX; ++x)
				{
					for (unsigned int y = 0; y < m_grid.dimY; ++y)
					{
						for (unsigned int z = 0; z < m_grid.dimZ; ++z)
						{
							switch (matrix)
							{
							case 0: hostCoeff[idx++] = m_op->GetVV(n, x, y, z); break;
							case 1: hostCoeff[idx++] = m_op->GetVI(n, x, y, z); break;
							case 2: hostCoeff[idx++] = m_op->GetII(n, x, y, z); break;
							default: hostCoeff[idx++] = m_op->GetIV(n, x, y, z); break;
							}
						}
					}
				}
			}

			std::memcpy(m_bufFieldStaging.mapped, hostCoeff.data(), fieldBytes);

			// Copy staging to device local
			vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
			vkResetFences(m_device, 1, &m_fence);

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);

			VkBufferCopy copyRegion = { 0, 0, fieldBytes };
			vkCmdCopyBuffer(m_cmdBuffer, m_bufFieldStaging.buffer, targetBuf.buffer, 1, &copyRegion);

			vkEndCommandBuffer(m_cmdBuffer);

			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &m_cmdBuffer;
			vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
			vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		};

		uploadMatrix(m_bufVv, 0);
		uploadMatrix(m_bufVi, 1);
		uploadMatrix(m_bufIi, 2);
		uploadMatrix(m_bufIv, 3);
	}

	return SyncFieldsToDevice();
}

bool EngineVulkan::SyncFieldsToDevice()
{
	if (!m_hostFieldsDirty)
		return true;
	if (!m_device || !m_bufFieldStaging.mapped)
		return false;

	size_t fieldBytes = 3 * static_cast<size_t>(m_grid.numCells) * sizeof(float);
	auto uploadBuffer = [&](const std::vector<float>& source, VulkanBuffer& target) {
		std::memcpy(m_bufFieldStaging.mapped, source.data(), fieldBytes);
		if (vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
			return false;
		if (vkResetFences(m_device, 1, &m_fence) != VK_SUCCESS)
			return false;

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		if (vkBeginCommandBuffer(m_cmdBuffer, &beginInfo) != VK_SUCCESS)
			return false;

		VkBufferCopy copyRegion = { 0, 0, fieldBytes };
		vkCmdCopyBuffer(m_cmdBuffer, m_bufFieldStaging.buffer, target.buffer, 1, &copyRegion);
		if (vkEndCommandBuffer(m_cmdBuffer) != VK_SUCCESS)
			return false;

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_cmdBuffer;
		if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence) != VK_SUCCESS)
			return false;
		return vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
	};

	if (!uploadBuffer(m_hostVolt, m_bufVolt) || !uploadBuffer(m_hostCurr, m_bufCurr))
		return false;

	m_hostFieldsDirty = false;
	m_hostFieldsValid = true;
	return true;
}

bool EngineVulkan::CreatePipelines()
{
	bool hasUpml = false;
	bool hasMur = false;
	bool hasTfsf = false;
	bool hasRlc = false;
	bool hasAbc = false;
	bool hasDispersive = false;
	bool hasCylinder = false;
	if (m_op)
	{
		for (size_t i = 0; i < m_op->GetNumberOfExtentions(); ++i)
		{
			if (dynamic_cast<Operator_Ext_UPML*>(m_op->GetExtension(i)))
				hasUpml = true;
			if (dynamic_cast<Operator_Ext_Mur_ABC*>(m_op->GetExtension(i)))
				hasMur = true;
			if (dynamic_cast<Operator_Ext_TFSF*>(m_op->GetExtension(i)))
				hasTfsf = true;
			if (dynamic_cast<Operator_Ext_LumpedRLC*>(m_op->GetExtension(i)))
				hasRlc = true;
			if (dynamic_cast<Operator_Ext_Absorbing_BC*>(m_op->GetExtension(i)))
				hasAbc = true;
			if (dynamic_cast<Operator_Ext_LorentzMaterial*>(m_op->GetExtension(i)))
				hasDispersive = true;
			if (dynamic_cast<Operator_Ext_Cylinder*>(m_op->GetExtension(i)))
				hasCylinder = true;
		}
	}

	// Compile GLSL compute shaders to SPIR-V bytecode at runtime
	auto spvVolt = CompileGLSLToSpirv(VulkanShaders::kShaderVoltageUpdate, shaderc_glsl_compute_shader, "yee_voltage.comp");
	auto spvCurr = CompileGLSLToSpirv(VulkanShaders::kShaderCurrentUpdate, shaderc_glsl_compute_shader, "yee_current.comp");
	auto spvExc  = CompileGLSLToSpirv(VulkanShaders::kShaderExcitation, shaderc_glsl_compute_shader, "excitation.comp");
	auto spvPrb  = CompileGLSLToSpirv(VulkanShaders::kShaderProbeGather, shaderc_glsl_compute_shader, "probe_gather.comp");
	std::vector<uint32_t> spvUpmlPre;
	std::vector<uint32_t> spvUpmlPost;
	if (hasUpml)
	{
		spvUpmlPre = CompileGLSLToSpirv(VulkanShaders::kShaderUpmlPre, shaderc_glsl_compute_shader, "upml_pre.comp");
		spvUpmlPost = CompileGLSLToSpirv(VulkanShaders::kShaderUpmlPost, shaderc_glsl_compute_shader, "upml_post.comp");
	}
	std::vector<uint32_t> spvMurPre;
	std::vector<uint32_t> spvMurPost;
	std::vector<uint32_t> spvMurApply;
	if (hasMur)
	{
		spvMurPre = CompileGLSLToSpirv(VulkanShaders::kShaderMurPre, shaderc_glsl_compute_shader, "mur_pre.comp");
		spvMurPost = CompileGLSLToSpirv(VulkanShaders::kShaderMurPost, shaderc_glsl_compute_shader, "mur_post.comp");
		spvMurApply = CompileGLSLToSpirv(VulkanShaders::kShaderMurApply, shaderc_glsl_compute_shader, "mur_apply.comp");
	}
	std::vector<uint32_t> spvTfsf;
	if (hasTfsf)
	{
		spvTfsf = CompileGLSLToSpirv(VulkanShaders::kShaderTFSF, shaderc_glsl_compute_shader, "tfsf.comp");
	}
	std::vector<uint32_t> spvRlc;
	if (hasRlc)
	{
		spvRlc = CompileGLSLToSpirv(VulkanShaders::kShaderRLC, shaderc_glsl_compute_shader, "rlc.comp");
	}
	std::vector<uint32_t> spvAbcVolt;
	std::vector<uint32_t> spvAbcCurr;
	if (hasAbc)
	{
		spvAbcVolt = CompileGLSLToSpirv(VulkanShaders::kShaderAbcVolt, shaderc_glsl_compute_shader, "abc_volt.comp");
		spvAbcCurr = CompileGLSLToSpirv(VulkanShaders::kShaderAbcCurr, shaderc_glsl_compute_shader, "abc_curr.comp");
	}
	std::vector<uint32_t> spvDisp;
	if (hasDispersive)
	{
		spvDisp = CompileGLSLToSpirv(VulkanShaders::kShaderDispersive, shaderc_glsl_compute_shader, "dispersive.comp");
	}
	std::vector<uint32_t> spvCylinder;
	if (hasCylinder)
	{
		spvCylinder = CompileGLSLToSpirv(VulkanShaders::kShaderCylinder, shaderc_glsl_compute_shader, "cylinder.comp");
	}

	if (spvVolt.empty() || spvCurr.empty() || spvExc.empty() || spvPrb.empty() ||
	    (hasUpml && (spvUpmlPre.empty() || spvUpmlPost.empty())) ||
	    (hasMur && (spvMurPre.empty() || spvMurPost.empty() || spvMurApply.empty())) ||
	    (hasTfsf && spvTfsf.empty()) ||
	    (hasRlc && spvRlc.empty()) ||
	    (hasAbc && (spvAbcVolt.empty() || spvAbcCurr.empty())) ||
	    (hasDispersive && spvDisp.empty()) ||
	    (hasCylinder && spvCylinder.empty()))
	{
		return false;
	}

	VkShaderModule modVolt = CreateShaderModule(spvVolt);
	VkShaderModule modCurr = CreateShaderModule(spvCurr);
	VkShaderModule modExc  = CreateShaderModule(spvExc);
	VkShaderModule modPrb  = CreateShaderModule(spvPrb);
	VkShaderModule modUpmlPre = VK_NULL_HANDLE;
	VkShaderModule modUpmlPost = VK_NULL_HANDLE;
	if (hasUpml)
	{
		modUpmlPre = CreateShaderModule(spvUpmlPre);
		modUpmlPost = CreateShaderModule(spvUpmlPost);
	}
	VkShaderModule modMurPre = VK_NULL_HANDLE;
	VkShaderModule modMurPost = VK_NULL_HANDLE;
	VkShaderModule modMurApply = VK_NULL_HANDLE;
	if (hasMur)
	{
		modMurPre = CreateShaderModule(spvMurPre);
		modMurPost = CreateShaderModule(spvMurPost);
		modMurApply = CreateShaderModule(spvMurApply);
	}
	VkShaderModule modTfsf = VK_NULL_HANDLE;
	if (hasTfsf)
	{
		modTfsf = CreateShaderModule(spvTfsf);
	}
	VkShaderModule modRlc = VK_NULL_HANDLE;
	if (hasRlc)
	{
		modRlc = CreateShaderModule(spvRlc);
	}
	VkShaderModule modAbcVolt = VK_NULL_HANDLE;
	VkShaderModule modAbcCurr = VK_NULL_HANDLE;
	if (hasAbc)
	{
		modAbcVolt = CreateShaderModule(spvAbcVolt);
		modAbcCurr = CreateShaderModule(spvAbcCurr);
	}
	VkShaderModule modDisp = VK_NULL_HANDLE;
	if (hasDispersive)
	{
		modDisp = CreateShaderModule(spvDisp);
	}
	VkShaderModule modCyl = VK_NULL_HANDLE;
	if (hasCylinder)
	{
		modCyl = CreateShaderModule(spvCylinder);
	}

	// Create descriptor pool
	uint32_t maxStorage = 32u;
	uint32_t maxSets = 16u;
	if (hasUpml) { maxStorage += 32u; maxSets += 16u; }
	if (hasMur)  { maxStorage += 32u; maxSets += 16u; }
	if (hasTfsf) { maxStorage += 32u; maxSets += 16u; }
	if (hasRlc)  { maxStorage += 32u; maxSets += 16u; }
	if (hasAbc)  { maxStorage += 32u; maxSets += 16u; }
	if (hasDispersive) { maxStorage += 32u; maxSets += 16u; }
	if (hasCylinder)   { maxStorage += 32u; maxSets += 16u; }
	std::vector<VkDescriptorPoolSize> poolSizes = {
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxStorage }
	};
	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = maxSets;
	if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descPool) != VK_SUCCESS)
	{
		return false;
	}

	// 1. Fields Pipeline Layout & Descriptor Set (6 storage buffers)
	std::vector<VkDescriptorSetLayoutBinding> fieldsBindings(6);
	for (uint32_t i = 0; i < 6; ++i)
	{
		fieldsBindings[i].binding = i;
		fieldsBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		fieldsBindings[i].descriptorCount = 1;
		fieldsBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 6;
	layoutInfo.pBindings = fieldsBindings.data();
	if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayoutFields) != VK_SUCCESS)
	{
		return false;
	}

	VkPushConstantRange pcRange = {};
	pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pcRange.offset = 0;
	pcRange.size = 4 * sizeof(uint32_t); // dimX, dimY, dimZ, numCells

	VkPipelineLayoutCreateInfo pipeLayoutInfo = {};
	pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeLayoutInfo.setLayoutCount = 1;
	pipeLayoutInfo.pSetLayouts = &m_descLayoutFields;
	pipeLayoutInfo.pushConstantRangeCount = 1;
	pipeLayoutInfo.pPushConstantRanges = &pcRange;
	if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_pipelineLayoutFields) != VK_SUCCESS)
	{
		return false;
	}

	// Create compute pipelines for Voltage and Current updates
	auto createComputePipe = [&](VkShaderModule mod, VkPipelineLayout playout, VkPipeline& outPipe) {
		VkComputePipelineCreateInfo cpInfo = {};
		cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		cpInfo.stage.module = mod;
		cpInfo.stage.pName = "main";
		cpInfo.layout = playout;
		return vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &outPipe) == VK_SUCCESS;
	};

	if (!createComputePipe(modVolt, m_pipelineLayoutFields, m_pipelineVolt)) return false;
	if (!createComputePipe(modCurr, m_pipelineLayoutFields, m_pipelineCurr)) return false;

	// Allocate and update descriptor set for Fields
	VkDescriptorSetAllocateInfo dsAlloc = {};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = m_descPool;
	dsAlloc.descriptorSetCount = 1;
	dsAlloc.pSetLayouts = &m_descLayoutFields;
	if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetFields) != VK_SUCCESS)
	{
		return false;
	}

	std::vector<VkDescriptorBufferInfo> bufInfos(6);
	VulkanBuffer* bufs[6] = { &m_bufVv, &m_bufVi, &m_bufIi, &m_bufIv, &m_bufVolt, &m_bufCurr };
	std::vector<VkWriteDescriptorSet> writes(6);
	for (uint32_t i = 0; i < 6; ++i)
	{
		bufInfos[i].buffer = bufs[i]->buffer;
		bufInfos[i].offset = 0;
		bufInfos[i].range = bufs[i]->size;

		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = m_descSetFields;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo = &bufInfos[i];
	}
	vkUpdateDescriptorSets(m_device, 6, writes.data(), 0, nullptr);

	// 2. Excitation Pipeline Layout
	std::vector<VkDescriptorSetLayoutBinding> excBindings(2);
	for (uint32_t i = 0; i < 2; ++i)
	{
		excBindings[i].binding = i;
		excBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		excBindings[i].descriptorCount = 1;
		excBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	layoutInfo.bindingCount = 2;
	layoutInfo.pBindings = excBindings.data();
	if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayoutExc) != VK_SUCCESS)
	{
		return false;
	}

	VkPushConstantRange excPcRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) };
	pipeLayoutInfo.setLayoutCount = 1;
	pipeLayoutInfo.pSetLayouts = &m_descLayoutExc;
	pipeLayoutInfo.pushConstantRangeCount = 1;
	pipeLayoutInfo.pPushConstantRanges = &excPcRange;
	if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_pipelineLayoutExc) != VK_SUCCESS)
	{
		return false;
	}
	if (!createComputePipe(modExc, m_pipelineLayoutExc, m_pipelineExc)) return false;

	// 3. Probe Gather Pipeline Layout
	std::vector<VkDescriptorSetLayoutBinding> prbBindings(4);
	for (uint32_t i = 0; i < 4; ++i)
	{
		prbBindings[i].binding = i;
		prbBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		prbBindings[i].descriptorCount = 1;
		prbBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	layoutInfo.bindingCount = 4;
	layoutInfo.pBindings = prbBindings.data();
	if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayoutProbe) != VK_SUCCESS)
	{
		return false;
	}

	pipeLayoutInfo.pSetLayouts = &m_descLayoutProbe;
	pipeLayoutInfo.pPushConstantRanges = &excPcRange;
	if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_pipelineLayoutProbe) != VK_SUCCESS)
	{
		return false;
	}
	if (!createComputePipe(modPrb, m_pipelineLayoutProbe, m_pipelineProbe)) return false;

	// 4. UPML Pipeline Layout & Compute Pipelines
	if (hasUpml)
	{
		std::vector<VkDescriptorSetLayoutBinding> upmlBindings(4);
		for (uint32_t i = 0; i < 4; ++i)
		{
			upmlBindings[i].binding = i;
			upmlBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			upmlBindings[i].descriptorCount = 1;
			upmlBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		layoutInfo.bindingCount = 4;
		layoutInfo.pBindings = upmlBindings.data();
		if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayoutUpml) != VK_SUCCESS)
		{
			return false;
		}

		VkPushConstantRange upmlPcRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) };
		pipeLayoutInfo.pSetLayouts = &m_descLayoutUpml;
		pipeLayoutInfo.pPushConstantRanges = &upmlPcRange;
		if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_pipelineLayoutUpml) != VK_SUCCESS)
		{
			return false;
		}
		if (!createComputePipe(modUpmlPre, m_pipelineLayoutUpml, m_pipelineUpmlPre)) return false;
		if (!createComputePipe(modUpmlPost, m_pipelineLayoutUpml, m_pipelineUpmlPost)) return false;
	}

	// 5. Mur ABC Pipeline Layout & Compute Pipelines
	if (hasMur)
	{
		std::vector<VkDescriptorSetLayoutBinding> murBindings(3);
		for (uint32_t i = 0; i < 3; ++i)
		{
			murBindings[i].binding = i;
			murBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			murBindings[i].descriptorCount = 1;
			murBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		layoutInfo.bindingCount = 3;
		layoutInfo.pBindings = murBindings.data();
		if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayoutMur) != VK_SUCCESS)
		{
			return false;
		}

		VkPushConstantRange murPcRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * sizeof(uint32_t) };
		pipeLayoutInfo.pSetLayouts = &m_descLayoutMur;
		pipeLayoutInfo.pPushConstantRanges = &murPcRange;
		if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_pipelineLayoutMur) != VK_SUCCESS)
		{
			return false;
		}
		if (!createComputePipe(modMurPre, m_pipelineLayoutMur, m_pipelineMurPre)) return false;
		if (!createComputePipe(modMurPost, m_pipelineLayoutMur, m_pipelineMurPost)) return false;
		if (!createComputePipe(modMurApply, m_pipelineLayoutMur, m_pipelineMurApply)) return false;
	}

	// 6. TFSF Pipeline Layout & Compute Pipeline
	if (hasTfsf)
	{
		std::vector<VkDescriptorSetLayoutBinding> tfsfBindings(3);
		for (uint32_t i = 0; i < 3; ++i)
		{
			tfsfBindings[i].binding = i;
			tfsfBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			tfsfBindings[i].descriptorCount = 1;
			tfsfBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		layoutInfo.bindingCount = 3;
		layoutInfo.pBindings = tfsfBindings.data();
		if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayoutTfsf) != VK_SUCCESS)
		{
			return false;
		}

		VkPushConstantRange tfsfPcRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 5 * sizeof(uint32_t) };
		pipeLayoutInfo.pSetLayouts = &m_descLayoutTfsf;
		pipeLayoutInfo.pPushConstantRanges = &tfsfPcRange;
		if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_pipelineLayoutTfsf) != VK_SUCCESS)
		{
			return false;
		}
		if (!createComputePipe(modTfsf, m_pipelineLayoutTfsf, m_pipelineTfsf)) return false;
	}

	// 7. Lumped RLC Pipeline Layout & Compute Pipeline
	if (hasRlc)
	{
		std::vector<VkDescriptorSetLayoutBinding> rlcBindings(3);
		for (uint32_t i = 0; i < 3; ++i)
		{
			rlcBindings[i].binding = i;
			rlcBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			rlcBindings[i].descriptorCount = 1;
			rlcBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		layoutInfo.bindingCount = 3;
		layoutInfo.pBindings = rlcBindings.data();
		if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayoutRlc) != VK_SUCCESS)
		{
			return false;
		}

		VkPushConstantRange rlcPcRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(uint32_t) };
		pipeLayoutInfo.pSetLayouts = &m_descLayoutRlc;
		pipeLayoutInfo.pPushConstantRanges = &rlcPcRange;
		if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_pipelineLayoutRlc) != VK_SUCCESS)
		{
			return false;
		}
		if (!createComputePipe(modRlc, m_pipelineLayoutRlc, m_pipelineRlc)) return false;
	}

	// 8. Absorbing BC Pipeline Layout & Compute Pipelines
	if (hasAbc)
	{
		std::vector<VkDescriptorSetLayoutBinding> abcBindings(3);
		for (uint32_t i = 0; i < 3; ++i)
		{
			abcBindings[i].binding = i;
			abcBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			abcBindings[i].descriptorCount = 1;
			abcBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		layoutInfo.bindingCount = 3;
		layoutInfo.pBindings = abcBindings.data();
		if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayoutAbc) != VK_SUCCESS)
		{
			return false;
		}

		VkPushConstantRange abcPcRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(uint32_t) };
		pipeLayoutInfo.pSetLayouts = &m_descLayoutAbc;
		pipeLayoutInfo.pPushConstantRanges = &abcPcRange;
		if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_pipelineLayoutAbc) != VK_SUCCESS)
		{
			return false;
		}
		if (modAbcVolt != VK_NULL_HANDLE && !createComputePipe(modAbcVolt, m_pipelineLayoutAbc, m_pipelineAbcVolt)) return false;
		if (modAbcCurr != VK_NULL_HANDLE && !createComputePipe(modAbcCurr, m_pipelineLayoutAbc, m_pipelineAbcCurr)) return false;
	}

	// 9. Dispersive Media Pipeline Layout & Compute Pipeline
	if (hasDispersive)
	{
		std::vector<VkDescriptorSetLayoutBinding> dispBindings(3);
		for (uint32_t i = 0; i < 3; ++i)
		{
			dispBindings[i].binding = i;
			dispBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			dispBindings[i].descriptorCount = 1;
			dispBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		layoutInfo.bindingCount = 3;
		layoutInfo.pBindings = dispBindings.data();
		if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayoutDisp) != VK_SUCCESS)
		{
			return false;
		}

		VkPushConstantRange dispPcRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(uint32_t) };
		pipeLayoutInfo.pSetLayouts = &m_descLayoutDisp;
		pipeLayoutInfo.pPushConstantRanges = &dispPcRange;
		if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_pipelineLayoutDisp) != VK_SUCCESS)
		{
			return false;
		}
		if (modDisp != VK_NULL_HANDLE && !createComputePipe(modDisp, m_pipelineLayoutDisp, m_pipelineDisp)) return false;
	}

	// 10. Cylindrical Coordinates Pipeline Layout & Compute Pipeline
	if (hasCylinder)
	{
		std::vector<VkDescriptorSetLayoutBinding> cylBindings(3);
		for (uint32_t i = 0; i < 3; ++i)
		{
			cylBindings[i].binding = i;
			cylBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			cylBindings[i].descriptorCount = 1;
			cylBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		layoutInfo.bindingCount = 3;
		layoutInfo.pBindings = cylBindings.data();
		if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayoutCyl) != VK_SUCCESS)
		{
			return false;
		}

		VkPushConstantRange cylPcRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 7 * sizeof(uint32_t) };
		pipeLayoutInfo.pSetLayouts = &m_descLayoutCyl;
		pipeLayoutInfo.pPushConstantRanges = &cylPcRange;
		if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_pipelineLayoutCyl) != VK_SUCCESS)
		{
			return false;
		}
		if (modCyl != VK_NULL_HANDLE && !createComputePipe(modCyl, m_pipelineLayoutCyl, m_pipelineCyl)) return false;
	}

	// Release shader modules after pipeline creation
	vkDestroyShaderModule(m_device, modVolt, nullptr);
	vkDestroyShaderModule(m_device, modCurr, nullptr);
	vkDestroyShaderModule(m_device, modExc, nullptr);
	vkDestroyShaderModule(m_device, modPrb, nullptr);
	if (modUpmlPre != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, modUpmlPre, nullptr);
	if (modUpmlPost != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, modUpmlPost, nullptr);
	if (modMurPre != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, modMurPre, nullptr);
	if (modMurPost != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, modMurPost, nullptr);
	if (modMurApply != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, modMurApply, nullptr);
	if (modTfsf != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, modTfsf, nullptr);
	if (modRlc != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, modRlc, nullptr);
	if (modAbcVolt != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, modAbcVolt, nullptr);
	if (modAbcCurr != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, modAbcCurr, nullptr);
	if (modDisp != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, modDisp, nullptr);
	if (modCyl != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, modCyl, nullptr);

	return true;
}

bool EngineVulkan::AllocateExcitationBuffers()
{
	m_excSources.clear();
	m_voltExcPoints.clear();
	m_currExcPoints.clear();

	if (!m_op) return true;

	for (size_t i = 0; i < m_op->GetNumberOfExtentions(); ++i)
	{
		Operator_Extension* ext = m_op->GetExtension(i);
		Operator_Ext_Excitation* excExt = dynamic_cast<Operator_Ext_Excitation*>(ext);
		if (!excExt || !excExt->m_Exc) continue;

		ExcSourceData sdata;
		sdata.exc = excExt->m_Exc;
		sdata.voltCount = excExt->Volt_Count;
		sdata.currCount = excExt->Curr_Count;

		if (sdata.voltCount > 0)
		{
			sdata.voltIndices.resize(sdata.voltCount);
			sdata.voltAmps.resize(sdata.voltCount);
			sdata.voltDelays.resize(sdata.voltCount);

			for (unsigned int n = 0; n < sdata.voltCount; ++n)
			{
				uint32_t linIdx = static_cast<uint32_t>(excExt->Volt_dir[n]) * m_grid.numCells +
				                  excExt->Volt_index[0][n] * (m_grid.dimY * m_grid.dimZ) +
				                  excExt->Volt_index[1][n] * m_grid.dimZ +
				                  excExt->Volt_index[2][n];
				sdata.voltIndices[n] = linIdx;
				sdata.voltAmps[n]    = static_cast<float>(excExt->Volt_amp[n]);
				sdata.voltDelays[n]  = static_cast<float>(excExt->Volt_delay[n]);

				m_voltExcPoints.push_back({ linIdx, 0.0f });
			}
		}

		if (sdata.currCount > 0)
		{
			sdata.currIndices.resize(sdata.currCount);
			sdata.currAmps.resize(sdata.currCount);
			sdata.currDelays.resize(sdata.currCount);

			for (unsigned int n = 0; n < sdata.currCount; ++n)
			{
				uint32_t linIdx = static_cast<uint32_t>(excExt->Curr_dir[n]) * m_grid.numCells +
				                  excExt->Curr_index[0][n] * (m_grid.dimY * m_grid.dimZ) +
				                  excExt->Curr_index[1][n] * m_grid.dimZ +
				                  excExt->Curr_index[2][n];
				sdata.currIndices[n] = linIdx;
				sdata.currAmps[n]    = static_cast<float>(excExt->Curr_amp[n]);
				sdata.currDelays[n]  = static_cast<float>(excExt->Curr_delay[n]);

				m_currExcPoints.push_back({ linIdx, 0.0f });
			}
		}

		m_excSources.push_back(std::move(sdata));
	}

	std::cout << "[openEMS Vulkan] Initialized on-device excitation: "
	          << m_voltExcPoints.size() << " voltage points, "
	          << m_currExcPoints.size() << " current points." << std::endl;

	// Allocate host-visible mapped buffers for fast excitation updates
	if (!m_voltExcPoints.empty())
	{
		size_t ptsBytes = m_voltExcPoints.size() * sizeof(GpuExcPoint);
		if (!CreateBuffer(ptsBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_bufVoltExcPoints))
		{
			return false;
		}

		VkDescriptorSetAllocateInfo dsAlloc = {};
		dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		dsAlloc.descriptorPool = m_descPool;
		dsAlloc.descriptorSetCount = 1;
		dsAlloc.pSetLayouts = &m_descLayoutExc;
		if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetVoltExc) != VK_SUCCESS)
		{
			return false;
		}

		VkDescriptorBufferInfo bInfos[2] = {
			{ m_bufVoltExcPoints.buffer, 0, ptsBytes },
			{ m_bufVolt.buffer, 0, m_bufVolt.size }
		};
		VkWriteDescriptorSet writes[2] = {};
		writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[0].dstSet = m_descSetVoltExc;
		writes[0].dstBinding = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[0].pBufferInfo = &bInfos[0];

		writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[1].dstSet = m_descSetVoltExc;
		writes[1].dstBinding = 1;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[1].pBufferInfo = &bInfos[1];

		vkUpdateDescriptorSets(m_device, 2, writes, 0, nullptr);
	}

	if (!m_currExcPoints.empty())
	{
		size_t ptsBytes = m_currExcPoints.size() * sizeof(GpuExcPoint);
		if (!CreateBuffer(ptsBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_bufCurrExcPoints))
		{
			return false;
		}

		VkDescriptorSetAllocateInfo dsAlloc = {};
		dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		dsAlloc.descriptorPool = m_descPool;
		dsAlloc.descriptorSetCount = 1;
		dsAlloc.pSetLayouts = &m_descLayoutExc;
		if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetCurrExc) != VK_SUCCESS)
		{
			return false;
		}

		VkDescriptorBufferInfo bInfos[2] = {
			{ m_bufCurrExcPoints.buffer, 0, ptsBytes },
			{ m_bufCurr.buffer, 0, m_bufCurr.size }
		};
		VkWriteDescriptorSet writes[2] = {};
		writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[0].dstSet = m_descSetCurrExc;
		writes[0].dstBinding = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[0].pBufferInfo = &bInfos[0];

		writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[1].dstSet = m_descSetCurrExc;
		writes[1].dstBinding = 1;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[1].pBufferInfo = &bInfos[1];

		vkUpdateDescriptorSets(m_device, 2, writes, 0, nullptr);
	}

	return true;
}

bool EngineVulkan::AllocateProbeBuffers()
{
	if (m_probePoints.empty()) return true;

	size_t count = m_probePoints.size();
	size_t pointsBytes = count * sizeof(ProbePoint);
	size_t valuesBytes = count * sizeof(float);

	// Create and populate probe points buffer
	if (!CreateBuffer(pointsBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_bufProbePoints))
	{
		return false;
	}
	std::memcpy(m_bufProbePoints.mapped, m_probePoints.data(), pointsBytes);

	// Create host-visible probe values buffer (directly mapped for zero-copy readback!)
	if (!CreateBuffer(valuesBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_bufProbeValues))
	{
		return false;
	}

	VkDescriptorSetAllocateInfo dsAlloc = {};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = m_descPool;
	dsAlloc.descriptorSetCount = 1;
	dsAlloc.pSetLayouts = &m_descLayoutProbe;
	if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetProbe) != VK_SUCCESS)
	{
		return false;
	}

	VkDescriptorBufferInfo bInfos[4] = {
		{ m_bufProbePoints.buffer, 0, pointsBytes },
		{ m_bufVolt.buffer, 0, m_bufVolt.size },
		{ m_bufCurr.buffer, 0, m_bufCurr.size },
		{ m_bufProbeValues.buffer, 0, valuesBytes }
	};
	VkWriteDescriptorSet writes[4] = {};
	for (uint32_t i = 0; i < 4; ++i)
	{
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = m_descSetProbe;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo = &bInfos[i];
	}
	vkUpdateDescriptorSets(m_device, 4, writes, 0, nullptr);

	return true;
}

bool EngineVulkan::AllocateUpmlBuffers()
{
	m_numUpmlCells = 0;
	if (!m_op) return true;

	std::vector<Operator_Ext_UPML*> upmlExts;
	for (size_t i = 0; i < m_op->GetNumberOfExtentions(); ++i)
	{
		Operator_Ext_UPML* upml = dynamic_cast<Operator_Ext_UPML*>(m_op->GetExtension(i));
		if (upml)
			upmlExts.push_back(upml);
	}

	if (upmlExts.empty())
		return true;

	std::vector<uint32_t> hostIndices;
	std::vector<float> hostVoltCoeffs; // 4 floats per cell: vv, vvfo, vvfn, 0
	std::vector<float> hostCurrCoeffs; // 4 floats per cell: ii, iifo, iifn, 0

	for (Operator_Ext_UPML* upml : upmlExts)
	{
		for (unsigned int comp = 0; comp < 3; ++comp)
		{
			for (unsigned int loc_x = 0; loc_x < upml->m_numLines[0]; ++loc_x)
			{
				unsigned int posX = loc_x + upml->m_StartPos[0];
				for (unsigned int loc_y = 0; loc_y < upml->m_numLines[1]; ++loc_y)
				{
					unsigned int posY = loc_y + upml->m_StartPos[1];
					for (unsigned int loc_z = 0; loc_z < upml->m_numLines[2]; ++loc_z)
					{
						unsigned int posZ = loc_z + upml->m_StartPos[2];
						uint32_t linFieldIdx = static_cast<uint32_t>(GetLinearIndex(comp, posX, posY, posZ));

						hostIndices.push_back(linFieldIdx);

						hostVoltCoeffs.push_back(static_cast<float>(upml->vv(comp, loc_x, loc_y, loc_z)));
						hostVoltCoeffs.push_back(static_cast<float>(upml->vvfo(comp, loc_x, loc_y, loc_z)));
						hostVoltCoeffs.push_back(static_cast<float>(upml->vvfn(comp, loc_x, loc_y, loc_z)));
						hostVoltCoeffs.push_back(0.0f);

						hostCurrCoeffs.push_back(static_cast<float>(upml->ii(comp, loc_x, loc_y, loc_z)));
						hostCurrCoeffs.push_back(static_cast<float>(upml->iifo(comp, loc_x, loc_y, loc_z)));
						hostCurrCoeffs.push_back(static_cast<float>(upml->iifn(comp, loc_x, loc_y, loc_z)));
						hostCurrCoeffs.push_back(0.0f);
					}
				}
			}
		}
	}

	m_numUpmlCells = static_cast<uint32_t>(hostIndices.size());
	if (m_numUpmlCells == 0)
		return true;

	std::cout << "[openEMS Vulkan] Initialized on-device UPML: "
	          << m_numUpmlCells << " cell components across "
	          << upmlExts.size() << " boundary layer(s)." << std::endl;

	VkDeviceSize idxBytes   = m_numUpmlCells * sizeof(uint32_t);
	VkDeviceSize coeffBytes = m_numUpmlCells * 4 * sizeof(float);
	VkDeviceSize fluxBytes  = m_numUpmlCells * sizeof(float);

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	if (!CreateBuffer(idxBytes, storageUsage, devLocal, m_bufUpmlIndices)) return false;
	if (!CreateBuffer(coeffBytes, storageUsage, devLocal, m_bufUpmlVoltCoeffs)) return false;
	if (!CreateBuffer(coeffBytes, storageUsage, devLocal, m_bufUpmlCurrCoeffs)) return false;
	if (!CreateBuffer(fluxBytes, storageUsage, devLocal, m_bufUpmlVoltFlux)) return false;
	if (!CreateBuffer(fluxBytes, storageUsage, devLocal, m_bufUpmlCurrFlux)) return false;

	// Staging buffers to upload indices and coefficients
	VulkanBuffer stageIndices, stageVoltCoeffs, stageCurrCoeffs;
	VkBufferUsageFlags stageUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkMemoryPropertyFlags stageProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	if (!CreateBuffer(idxBytes, stageUsage, stageProps, stageIndices)) return false;
	if (!CreateBuffer(coeffBytes, stageUsage, stageProps, stageVoltCoeffs)) return false;
	if (!CreateBuffer(coeffBytes, stageUsage, stageProps, stageCurrCoeffs)) return false;

	std::memcpy(stageIndices.mapped, hostIndices.data(), idxBytes);
	std::memcpy(stageVoltCoeffs.mapped, hostVoltCoeffs.data(), coeffBytes);
	std::memcpy(stageCurrCoeffs.mapped, hostCurrCoeffs.data(), coeffBytes);

	vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
	vkResetFences(m_device, 1, &m_fence);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);

	VkBufferCopy copyIdx = { 0, 0, idxBytes };
	vkCmdCopyBuffer(m_cmdBuffer, stageIndices.buffer, m_bufUpmlIndices.buffer, 1, &copyIdx);

	VkBufferCopy copyCoeff = { 0, 0, coeffBytes };
	vkCmdCopyBuffer(m_cmdBuffer, stageVoltCoeffs.buffer, m_bufUpmlVoltCoeffs.buffer, 1, &copyCoeff);
	vkCmdCopyBuffer(m_cmdBuffer, stageCurrCoeffs.buffer, m_bufUpmlCurrCoeffs.buffer, 1, &copyCoeff);

	// Zero flux buffers
	vkCmdFillBuffer(m_cmdBuffer, m_bufUpmlVoltFlux.buffer, 0, VK_WHOLE_SIZE, 0);
	vkCmdFillBuffer(m_cmdBuffer, m_bufUpmlCurrFlux.buffer, 0, VK_WHOLE_SIZE, 0);

	vkEndCommandBuffer(m_cmdBuffer);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &m_cmdBuffer;
	vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
	vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

	DestroyBuffer(stageIndices);
	DestroyBuffer(stageVoltCoeffs);
	DestroyBuffer(stageCurrCoeffs);

	// Allocate and configure descriptor sets for UPML
	VkDescriptorSetAllocateInfo dsAlloc = {};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = m_descPool;
	dsAlloc.descriptorSetCount = 1;
	dsAlloc.pSetLayouts = &m_descLayoutUpml;

	if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetUpmlVolt) != VK_SUCCESS) return false;
	if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetUpmlCurr) != VK_SUCCESS) return false;

	// Update m_descSetUpmlVolt
	VkDescriptorBufferInfo voltBufInfos[4] = {
		{ m_bufUpmlIndices.buffer, 0, idxBytes },
		{ m_bufUpmlVoltCoeffs.buffer, 0, coeffBytes },
		{ m_bufUpmlVoltFlux.buffer, 0, fluxBytes },
		{ m_bufVolt.buffer, 0, m_bufVolt.size }
	};
	VkWriteDescriptorSet voltWrites[4] = {};
	for (uint32_t i = 0; i < 4; ++i)
	{
		voltWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		voltWrites[i].dstSet = m_descSetUpmlVolt;
		voltWrites[i].dstBinding = i;
		voltWrites[i].descriptorCount = 1;
		voltWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		voltWrites[i].pBufferInfo = &voltBufInfos[i];
	}
	vkUpdateDescriptorSets(m_device, 4, voltWrites, 0, nullptr);

	// Update m_descSetUpmlCurr
	VkDescriptorBufferInfo currBufInfos[4] = {
		{ m_bufUpmlIndices.buffer, 0, idxBytes },
		{ m_bufUpmlCurrCoeffs.buffer, 0, coeffBytes },
		{ m_bufUpmlCurrFlux.buffer, 0, fluxBytes },
		{ m_bufCurr.buffer, 0, m_bufCurr.size }
	};
	VkWriteDescriptorSet currWrites[4] = {};
	for (uint32_t i = 0; i < 4; ++i)
	{
		currWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		currWrites[i].dstSet = m_descSetUpmlCurr;
		currWrites[i].dstBinding = i;
		currWrites[i].descriptorCount = 1;
		currWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		currWrites[i].pBufferInfo = &currBufInfos[i];
	}
	vkUpdateDescriptorSets(m_device, 4, currWrites, 0, nullptr);

	return true;
}

bool EngineVulkan::AllocateMurBuffers()
{
#ifdef ENABLE_VULKAN
	m_totalMurPoints = 0;
	m_murFaces.clear();
	if (!m_op) return true;

	std::vector<GpuMurPoint> hostPoints;

	for (size_t extIdx = 0; extIdx < m_op->GetNumberOfExtentions(); ++extIdx)
	{
		Operator_Ext_Mur_ABC* mur = dynamic_cast<Operator_Ext_Mur_ABC*>(m_op->GetExtension(extIdx));
		if (!mur) continue;

		uint32_t start_TS = 0;
		int maxDelay = -1;
		Operator_Ext_Excitation* excExt = m_op->GetExcitationExtension();
		if (excExt)
		{
			for (unsigned int n = 0; n < excExt->GetVoltCount(); ++n)
			{
				if (((excExt->Volt_dir[n] == mur->m_nyP) || (excExt->Volt_dir[n] == mur->m_nyPP)) &&
				    (excExt->Volt_index[mur->m_ny][n] == mur->m_LineNr))
				{
					if (static_cast<int>(excExt->Volt_delay[n]) > maxDelay)
						maxDelay = static_cast<int>(excExt->Volt_delay[n]);
				}
			}
			if (maxDelay >= 0 && m_op->GetExcitationSignal())
			{
				start_TS = static_cast<uint32_t>(maxDelay + m_op->GetExcitationSignal()->GetLength() + 10);
			}
		}

		MurFace face;
		face.offset = static_cast<uint32_t>(hostPoints.size());
		face.start_TS = start_TS;

		unsigned int pos[3] = {0, 0, 0};
		unsigned int pos_shift[3] = {0, 0, 0};
		pos[mur->m_ny] = mur->m_LineNr;
		pos_shift[mur->m_ny] = mur->m_LineNr_Shift;

		for (unsigned int i = 0; i < mur->m_numLines[0]; ++i)
		{
			pos[mur->m_nyP] = i;
			pos_shift[mur->m_nyP] = i;
			for (unsigned int j = 0; j < mur->m_numLines[1]; ++j)
			{
				pos[mur->m_nyPP] = j;
				pos_shift[mur->m_nyPP] = j;

				// Component nyP
				GpuMurPoint ptP;
				ptP.pos_idx = static_cast<uint32_t>(GetLinearIndex(mur->m_nyP, pos[0], pos[1], pos[2]));
				ptP.shift_idx = static_cast<uint32_t>(GetLinearIndex(mur->m_nyP, pos_shift[0], pos_shift[1], pos_shift[2]));
				ptP.coeff = static_cast<float>(mur->m_Mur_Coeff_nyP(i, j));
				ptP.start_TS = start_TS;
				hostPoints.push_back(ptP);

				// Component nyPP
				GpuMurPoint ptPP;
				ptPP.pos_idx = static_cast<uint32_t>(GetLinearIndex(mur->m_nyPP, pos[0], pos[1], pos[2]));
				ptPP.shift_idx = static_cast<uint32_t>(GetLinearIndex(mur->m_nyPP, pos_shift[0], pos_shift[1], pos_shift[2]));
				ptPP.coeff = static_cast<float>(mur->m_Mur_Coeff_nyPP(i, j));
				ptPP.start_TS = start_TS;
				hostPoints.push_back(ptPP);
			}
		}

		face.count = static_cast<uint32_t>(hostPoints.size()) - face.offset;
		m_murFaces.push_back(face);
	}

	m_totalMurPoints = static_cast<uint32_t>(hostPoints.size());
	if (m_totalMurPoints == 0) return true;

	std::cout << "[openEMS Vulkan] Initialized on-device Mur ABC: "
	          << m_totalMurPoints << " boundary points across "
	          << m_murFaces.size() << " face(s)." << std::endl;

	VkDeviceSize paramsBytes = m_totalMurPoints * sizeof(GpuMurPoint);
	VkDeviceSize storeBytes  = m_totalMurPoints * sizeof(float);

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	if (!CreateBuffer(paramsBytes, storageUsage, devLocal, m_bufMurParams)) return false;
	if (!CreateBuffer(storeBytes, storageUsage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, devLocal, m_bufMurStore)) return false;

	// Staging buffer to upload params
	VulkanBuffer stageParams;
	if (!CreateBuffer(paramsBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stageParams))
		return false;

	void* pData = nullptr;
	vkMapMemory(m_device, stageParams.memory, 0, paramsBytes, 0, &pData);
	memcpy(pData, hostPoints.data(), paramsBytes);
	vkUnmapMemory(m_device, stageParams.memory);

	vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
	vkResetFences(m_device, 1, &m_fence);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);

	VkBufferCopy copyRegion = { 0, 0, paramsBytes };
	vkCmdCopyBuffer(m_cmdBuffer, stageParams.buffer, m_bufMurParams.buffer, 1, &copyRegion);

	// Zero store buffer
	vkCmdFillBuffer(m_cmdBuffer, m_bufMurStore.buffer, 0, VK_WHOLE_SIZE, 0);

	vkEndCommandBuffer(m_cmdBuffer);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &m_cmdBuffer;
	vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
	vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

	DestroyBuffer(stageParams);

	// Allocate and configure descriptor set for Mur ABC
	VkDescriptorSetAllocateInfo dsAlloc = {};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = m_descPool;
	dsAlloc.descriptorSetCount = 1;
	dsAlloc.pSetLayouts = &m_descLayoutMur;

	if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetMur) != VK_SUCCESS) return false;

	VkDescriptorBufferInfo murBufInfos[3] = {
		{ m_bufMurParams.buffer, 0, paramsBytes },
		{ m_bufVolt.buffer, 0, m_bufVolt.size },
		{ m_bufMurStore.buffer, 0, storeBytes }
	};
	VkWriteDescriptorSet murWrites[3] = {};
	for (uint32_t i = 0; i < 3; ++i)
	{
		murWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		murWrites[i].dstSet = m_descSetMur;
		murWrites[i].dstBinding = i;
		murWrites[i].descriptorCount = 1;
		murWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		murWrites[i].pBufferInfo = &murBufInfos[i];
	}
	vkUpdateDescriptorSets(m_device, 3, murWrites, 0, nullptr);

	return true;
#else
	return true;
#endif
}

bool EngineVulkan::AllocateTfsfBuffers()
{
#ifdef ENABLE_VULKAN
	m_tfsfVoltFaces.clear();
	m_tfsfVoltPoints.clear();
	m_tfsfCurrFaces.clear();
	m_tfsfCurrPoints.clear();
	m_tfsfSigLength = 0;
	m_tfsfPeriod = 0;

	if (!m_op) return true;

	Operator_Ext_TFSF* tfsf = nullptr;
	for (size_t i = 0; i < m_op->GetNumberOfExtentions(); ++i)
	{
		tfsf = dynamic_cast<Operator_Ext_TFSF*>(m_op->GetExtension(i));
		if (tfsf) break;
	}
	if (!tfsf || !tfsf->m_Exc) return true;

	m_tfsfSigLength = tfsf->m_Exc->GetLength();
	if (tfsf->m_Exc->GetTimestep() > 0.0)
	{
		m_tfsfPeriod = static_cast<int32_t>(tfsf->m_Exc->GetSignalPeriod() / tfsf->m_Exc->GetTimestep());
	}

	// 1. Build Voltage points & faces
	for (int n = 0; n < 3; ++n)
	{
		int nP = (n + 1) % 3;
		int nPP = (n + 2) % 3;

		for (int l = 0; l < 2; ++l)
		{
			if (!tfsf->m_ActiveDir[n][l]) continue;

			TfsfFace face;
			face.offset = static_cast<uint32_t>(m_tfsfVoltPoints.size());

			for (int c = 0; c < 2; ++c)
			{
				unsigned int dir = (c == 0) ? nP : nPP;
				unsigned int ui_pos = 0;

				for (unsigned int i = 0; i < tfsf->m_numLines[nP]; ++i)
				{
					for (unsigned int j = 0; j < tfsf->m_numLines[nPP]; ++j)
					{
						float amp = static_cast<float>(tfsf->m_VoltAmp[n][l][c][ui_pos]);
						if (amp != 0.0f)
						{
							unsigned int pos[3];
							pos[nP] = tfsf->m_Start[nP] + i;
							pos[nPP] = tfsf->m_Start[nPP] + j;
							pos[n] = (l == 0) ? tfsf->m_Start[n] : tfsf->m_Stop[n];

							float delta = static_cast<float>(tfsf->m_VoltDelayDelta[n][l][c][ui_pos]);
							uint32_t delay = tfsf->m_VoltDelay[n][l][c][ui_pos];
							size_t linIdx = GetLinearIndex(dir, pos[0], pos[1], pos[2]);

							GpuTfsfPoint pt;
							pt.pos_idx = static_cast<uint32_t>(linIdx);
							pt.delay = delay;
							pt.w0 = (1.0f - delta) * amp;
							pt.w1 = delta * amp;
							m_tfsfVoltPoints.push_back(pt);
						}
						++ui_pos;
					}
				}
			}

			face.count = static_cast<uint32_t>(m_tfsfVoltPoints.size()) - face.offset;
			if (face.count > 0)
			{
				m_tfsfVoltFaces.push_back(face);
			}
		}
	}

	// 2. Build Current points & faces
	for (int n = 0; n < 3; ++n)
	{
		int nP = (n + 1) % 3;
		int nPP = (n + 2) % 3;

		for (int l = 0; l < 2; ++l)
		{
			if (!tfsf->m_ActiveDir[n][l]) continue;

			TfsfFace face;
			face.offset = static_cast<uint32_t>(m_tfsfCurrPoints.size());

			for (int c = 0; c < 2; ++c)
			{
				unsigned int dir = (c == 0) ? nP : nPP;
				unsigned int ui_pos = 0;

				for (unsigned int i = 0; i < tfsf->m_numLines[nP]; ++i)
				{
					for (unsigned int j = 0; j < tfsf->m_numLines[nPP]; ++j)
					{
						float amp = static_cast<float>(tfsf->m_CurrAmp[n][l][c][ui_pos]);
						if (amp != 0.0f)
						{
							unsigned int pos[3];
							pos[nP] = tfsf->m_Start[nP] + i;
							pos[nPP] = tfsf->m_Start[nPP] + j;
							pos[n] = (l == 0) ? (tfsf->m_Start[n] - 1) : tfsf->m_Stop[n];

							float delta = static_cast<float>(tfsf->m_CurrDelayDelta[n][l][c][ui_pos]);
							uint32_t delay = tfsf->m_CurrDelay[n][l][c][ui_pos];
							size_t linIdx = GetLinearIndex(dir, pos[0], pos[1], pos[2]);

							GpuTfsfPoint pt;
							pt.pos_idx = static_cast<uint32_t>(linIdx);
							pt.delay = delay;
							pt.w0 = (1.0f - delta) * amp;
							pt.w1 = delta * amp;
							m_tfsfCurrPoints.push_back(pt);
						}
						++ui_pos;
					}
				}
			}

			face.count = static_cast<uint32_t>(m_tfsfCurrPoints.size()) - face.offset;
			if (face.count > 0)
			{
				m_tfsfCurrFaces.push_back(face);
			}
		}
	}

	if (m_tfsfVoltPoints.empty() && m_tfsfCurrPoints.empty())
		return true;

	std::cout << "[openEMS Vulkan] Initialized on-device TFSF: "
	          << m_tfsfVoltPoints.size() << " voltage points across " << m_tfsfVoltFaces.size() << " face(s), "
	          << m_tfsfCurrPoints.size() << " current points across " << m_tfsfCurrFaces.size() << " face(s)." << std::endl;

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	// Upload volt params
	if (!m_tfsfVoltPoints.empty())
	{
		VkDeviceSize bytes = m_tfsfVoltPoints.size() * sizeof(GpuTfsfPoint);
		if (!CreateBuffer(bytes, storageUsage, devLocal, m_bufTfsfVoltParams)) return false;

		VulkanBuffer stage;
		if (!CreateBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;
		void* pData = nullptr;
		vkMapMemory(m_device, stage.memory, 0, bytes, 0, &pData);
		memcpy(pData, m_tfsfVoltPoints.data(), bytes);
		vkUnmapMemory(m_device, stage.memory);

		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, bytes };
		vkCmdCopyBuffer(m_cmdBuffer, stage.buffer, m_bufTfsfVoltParams.buffer, 1, &copyRegion);
		vkEndCommandBuffer(m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);
	}

	// Upload curr params
	if (!m_tfsfCurrPoints.empty())
	{
		VkDeviceSize bytes = m_tfsfCurrPoints.size() * sizeof(GpuTfsfPoint);
		if (!CreateBuffer(bytes, storageUsage, devLocal, m_bufTfsfCurrParams)) return false;

		VulkanBuffer stage;
		if (!CreateBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;
		void* pData = nullptr;
		vkMapMemory(m_device, stage.memory, 0, bytes, 0, &pData);
		memcpy(pData, m_tfsfCurrPoints.data(), bytes);
		vkUnmapMemory(m_device, stage.memory);

		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, bytes };
		vkCmdCopyBuffer(m_cmdBuffer, stage.buffer, m_bufTfsfCurrParams.buffer, 1, &copyRegion);
		vkEndCommandBuffer(m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);
	}

	// Upload signals
	if (m_tfsfSigLength > 0)
	{
		VkDeviceSize sigBytes = m_tfsfSigLength * sizeof(float);
		if (!CreateBuffer(sigBytes, storageUsage, devLocal, m_bufTfsfCurrSignal)) return false;
		if (!CreateBuffer(sigBytes, storageUsage, devLocal, m_bufTfsfVoltSignal)) return false;

		std::vector<float> currSigHost(m_tfsfSigLength);
		std::vector<float> voltSigHost(m_tfsfSigLength);
		FDTD_FLOAT* rawCurrSig = tfsf->m_Exc->GetCurrentSignal();
		FDTD_FLOAT* rawVoltSig = tfsf->m_Exc->GetVoltageSignal();
		for (uint32_t i = 0; i < m_tfsfSigLength; ++i)
		{
			currSigHost[i] = rawCurrSig ? static_cast<float>(rawCurrSig[i]) : 0.0f;
			voltSigHost[i] = rawVoltSig ? static_cast<float>(rawVoltSig[i]) : 0.0f;
		}

		VulkanBuffer stage;
		if (!CreateBuffer(sigBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;

		// Upload currSigHost to m_bufTfsfCurrSignal
		void* pData = nullptr;
		vkMapMemory(m_device, stage.memory, 0, sigBytes, 0, &pData);
		memcpy(pData, currSigHost.data(), sigBytes);
		vkUnmapMemory(m_device, stage.memory);

		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_fence);
		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, sigBytes };
		vkCmdCopyBuffer(m_cmdBuffer, stage.buffer, m_bufTfsfCurrSignal.buffer, 1, &copyRegion);
		vkEndCommandBuffer(m_cmdBuffer);
		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

		// Upload voltSigHost to m_bufTfsfVoltSignal
		vkMapMemory(m_device, stage.memory, 0, sigBytes, 0, &pData);
		memcpy(pData, voltSigHost.data(), sigBytes);
		vkUnmapMemory(m_device, stage.memory);

		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_fence);
		vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);
		vkCmdCopyBuffer(m_cmdBuffer, stage.buffer, m_bufTfsfVoltSignal.buffer, 1, &copyRegion);
		vkEndCommandBuffer(m_cmdBuffer);
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);
	}

	// Allocate and update descriptor sets
	if (m_descLayoutTfsf != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE)
	{
		VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		dsAlloc.descriptorPool = m_descPool;
		dsAlloc.descriptorSetCount = 1;
		dsAlloc.pSetLayouts = &m_descLayoutTfsf;

		if (!m_tfsfVoltPoints.empty())
		{
			if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetTfsfVolt) != VK_SUCCESS)
				return false;

			VkDescriptorBufferInfo b0 = { m_bufVolt.buffer, 0, m_bufVolt.size };
			VkDescriptorBufferInfo b1 = { m_bufTfsfVoltParams.buffer, 0, m_bufTfsfVoltParams.size };
			VkDescriptorBufferInfo b2 = { m_bufTfsfCurrSignal.buffer, 0, m_bufTfsfCurrSignal.size };

			VkWriteDescriptorSet writes[3] = {};
			for (int w = 0; w < 3; ++w)
			{
				writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[w].dstSet = m_descSetTfsfVolt;
				writes[w].dstBinding = w;
				writes[w].descriptorCount = 1;
				writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			}
			writes[0].pBufferInfo = &b0;
			writes[1].pBufferInfo = &b1;
			writes[2].pBufferInfo = &b2;
			vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);
		}

		if (!m_tfsfCurrPoints.empty())
		{
			if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetTfsfCurr) != VK_SUCCESS)
				return false;

			VkDescriptorBufferInfo b0 = { m_bufCurr.buffer, 0, m_bufCurr.size };
			VkDescriptorBufferInfo b1 = { m_bufTfsfCurrParams.buffer, 0, m_bufTfsfCurrParams.size };
			VkDescriptorBufferInfo b2 = { m_bufTfsfVoltSignal.buffer, 0, m_bufTfsfVoltSignal.size };

			VkWriteDescriptorSet writes[3] = {};
			for (int w = 0; w < 3; ++w)
			{
				writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[w].dstSet = m_descSetTfsfCurr;
				writes[w].dstBinding = w;
				writes[w].descriptorCount = 1;
				writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			}
			writes[0].pBufferInfo = &b0;
			writes[1].pBufferInfo = &b1;
			writes[2].pBufferInfo = &b2;
			vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);
		}
	}

	return true;
#else
	return true;
#endif
}

bool EngineVulkan::AllocateRlcBuffers()
{
#ifdef ENABLE_VULKAN
	if (!m_op) return true;

	Operator_Ext_LumpedRLC* rlc = nullptr;
	for (size_t i = 0; i < m_op->GetNumberOfExtentions(); ++i)
	{
		rlc = dynamic_cast<Operator_Ext_LumpedRLC*>(m_op->GetExtension(i));
		if (rlc) break;
	}
	if (!rlc || rlc->RLC_count == 0) return true;

	m_rlcCount = rlc->RLC_count;
	std::vector<GpuRlcParam> rlcParams(m_rlcCount);
	std::vector<GpuRlcState> rlcState(m_rlcCount);
	std::memset(rlcState.data(), 0, m_rlcCount * sizeof(GpuRlcState));

	for (uint32_t i = 0; i < m_rlcCount; ++i)
	{
		uint32_t dir = static_cast<uint32_t>(rlc->v_RLC_dir[i]);
		uint32_t x = rlc->v_RLC_pos[0][i];
		uint32_t y = rlc->v_RLC_pos[1][i];
		uint32_t z = rlc->v_RLC_pos[2][i];

		rlcParams[i].field_index = static_cast<uint32_t>(GetLinearIndex(dir, x, y, z));
		rlcParams[i].ilv_i2v = (rlc->v_RLC_ilv && rlc->v_RLC_i2v) ? static_cast<float>(rlc->v_RLC_ilv[i] * rlc->v_RLC_i2v[i]) : 0.0f;
		rlcParams[i].vvd = static_cast<float>(rlc->v_RLC_vvd[i]);
		rlcParams[i].vv2 = static_cast<float>(rlc->v_RLC_vv2[i]);
		rlcParams[i].vj1 = static_cast<float>(rlc->v_RLC_vj1[i]);
		rlcParams[i].vj2 = static_cast<float>(rlc->v_RLC_vj2[i]);
		rlcParams[i].ib0 = static_cast<float>(rlc->v_RLC_ib0[i]);
		rlcParams[i].b1_ib0 = static_cast<float>(rlc->v_RLC_b1[i] * rlc->v_RLC_ib0[i]);
		rlcParams[i].b2_ib0 = static_cast<float>(rlc->v_RLC_b2[i] * rlc->v_RLC_ib0[i]);
		rlcParams[i].pad0 = 1.0f;
		rlcParams[i].pad1 = 0.0f;
		rlcParams[i].pad2 = 0.0f;
	}

	// Engine_Ext_LumpedRLC reads every source voltage before writing any
	// result, then writes elements in array order. Mark only the last element
	// at a duplicated node for the shader's separate write pass.
	for (uint32_t i = 0; i < m_rlcCount; ++i)
	{
		for (uint32_t j = i + 1; j < m_rlcCount; ++j)
		{
			if (rlcParams[i].field_index == rlcParams[j].field_index)
			{
				rlcParams[i].pad0 = 0.0f;
				break;
			}
		}
	}

	std::cout << "[openEMS Vulkan] Initialized on-device Lumped RLC: " << m_rlcCount << " element(s)." << std::endl;

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	VkDeviceSize paramsBytes = m_rlcCount * sizeof(GpuRlcParam);
	VkDeviceSize stateBytes = m_rlcCount * sizeof(GpuRlcState);

	if (!CreateBuffer(paramsBytes, storageUsage, devLocal, m_bufRlcParams)) return false;
	if (!CreateBuffer(stateBytes, storageUsage, devLocal, m_bufRlcState)) return false;

	// Upload initial params
	{
		VulkanBuffer stage;
		if (!CreateBuffer(paramsBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;
		void* pData = nullptr;
		vkMapMemory(m_device, stage.memory, 0, paramsBytes, 0, &pData);
		memcpy(pData, rlcParams.data(), paramsBytes);
		vkUnmapMemory(m_device, stage.memory);

		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, paramsBytes };
		vkCmdCopyBuffer(m_cmdBuffer, stage.buffer, m_bufRlcParams.buffer, 1, &copyRegion);
		vkEndCommandBuffer(m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		DestroyBuffer(stage);
	}

	// Upload initial state (zeros)
	{
		VulkanBuffer stage;
		if (!CreateBuffer(stateBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;
		void* pData = nullptr;
		vkMapMemory(m_device, stage.memory, 0, stateBytes, 0, &pData);
		memcpy(pData, rlcState.data(), stateBytes);
		vkUnmapMemory(m_device, stage.memory);

		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, stateBytes };
		vkCmdCopyBuffer(m_cmdBuffer, stage.buffer, m_bufRlcState.buffer, 1, &copyRegion);
		vkEndCommandBuffer(m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		DestroyBuffer(stage);
	}

	// Allocate and update descriptor set
	VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	allocInfo.descriptorPool = m_descPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &m_descLayoutRlc;
	if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_descSetRlc) != VK_SUCCESS)
		return false;

	VkDescriptorBufferInfo bInfoVolt = { m_bufVolt.buffer, 0, m_bufVolt.size };
	VkDescriptorBufferInfo bInfoParams = { m_bufRlcParams.buffer, 0, m_bufRlcParams.size };
	VkDescriptorBufferInfo bInfoState = { m_bufRlcState.buffer, 0, m_bufRlcState.size };

	VkWriteDescriptorSet writes[3] = {};
	writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].dstSet = m_descSetRlc;
	writes[0].dstBinding = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[0].pBufferInfo = &bInfoVolt;

	writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet = m_descSetRlc;
	writes[1].dstBinding = 1;
	writes[1].descriptorCount = 1;
	writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[1].pBufferInfo = &bInfoParams;

	writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[2].dstSet = m_descSetRlc;
	writes[2].dstBinding = 2;
	writes[2].descriptorCount = 1;
	writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[2].pBufferInfo = &bInfoState;

	vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);

	return true;
#else
	return true;
#endif
}

bool EngineVulkan::AllocateAbsorbingBCBuffers()
{
#ifdef ENABLE_VULKAN
	if (!m_op) return true;

	std::vector<Operator_Ext_Absorbing_BC*> abcExts;
	for (size_t i = 0; i < m_op->GetNumberOfExtentions(); ++i)
	{
		Operator_Ext_Absorbing_BC* abc = dynamic_cast<Operator_Ext_Absorbing_BC*>(m_op->GetExtension(i));
		if (abc)
			abcExts.push_back(abc);
	}
	if (abcExts.empty()) return true;

	std::vector<GpuAbcVoltPoint> hostVoltPoints;
	std::vector<GpuAbcCurrPoint> hostCurrPoints;
	m_abcVoltSheets.clear();
	m_abcCurrSheets.clear();

	for (Operator_Ext_Absorbing_BC* abc : abcExts)
	{
		TfsfFace voltSheet = { static_cast<uint32_t>(hostVoltPoints.size()), 0 };
		TfsfFace currSheet = { static_cast<uint32_t>(hostCurrPoints.size()), 0 };
		int ny = abc->m_ny;
		int nyP = abc->m_nyP;
		int nyPP = abc->m_nyPP;
		bool normalSignPositive = abc->m_normalSignPositive;

		unsigned int pos[3] = {0, 0, 0};
		unsigned int pos_shift[3] = {0, 0, 0};

		unsigned int posStart[3];
		for (int d = 0; d < 3; ++d) posStart[d] = abc->m_sheetX0[d];

		unsigned int pos_ny0_shift_V = posStart[ny] + (normalSignPositive ? 1 : -1);

		pos[ny] = posStart[ny];
		pos_shift[ny] = pos_ny0_shift_V;

		for (unsigned int i = 0; i < abc->m_numLines[0]; ++i)
		{
			pos_shift[nyP] = pos[nyP] = posStart[nyP] + i;
			for (unsigned int j = 0; j < abc->m_numLines[1]; ++j)
			{
				pos_shift[nyPP] = pos[nyPP] = posStart[nyPP] + j;

				// Direction nyP
				GpuAbcVoltPoint ptP;
				ptP.pos_idx = static_cast<uint32_t>(GetLinearIndex(nyP, pos[0], pos[1], pos[2]));
				ptP.shift_idx = static_cast<uint32_t>(GetLinearIndex(nyP, pos_shift[0], pos_shift[1], pos_shift[2]));
				ptP.k1 = static_cast<float>(abc->m_K1_nyP(i, j));
				ptP.pad = 0;
				hostVoltPoints.push_back(ptP);

				// Direction nyPP
				GpuAbcVoltPoint ptPP;
				ptPP.pos_idx = static_cast<uint32_t>(GetLinearIndex(nyPP, pos[0], pos[1], pos[2]));
				ptPP.shift_idx = static_cast<uint32_t>(GetLinearIndex(nyPP, pos_shift[0], pos_shift[1], pos_shift[2]));
				ptPP.k1 = static_cast<float>(abc->m_K1_nyPP(i, j));
				ptPP.pad = 0;
				hostVoltPoints.push_back(ptPP);
			}
		}

		if (abc->m_ABCtype == Operator_Ext_Absorbing_BC::MUR_1ST_SA)
		{
			unsigned int pos_ny0_I = posStart[ny] + (normalSignPositive ? 0 : -1);
			unsigned int pos_ny0_shift_I = posStart[ny] + (normalSignPositive ? 1 : -2);

			pos[ny] = pos_ny0_I;
			pos_shift[ny] = pos_ny0_shift_I;

			if (abc->m_numLines[0] > 1 && abc->m_numLines[1] > 1)
			{
				for (unsigned int i = 0; i < (abc->m_numLines[0] - 1); ++i)
				{
					pos_shift[nyP] = pos[nyP] = posStart[nyP] + i;
					for (unsigned int j = 0; j < (abc->m_numLines[1] - 1); ++j)
					{
						pos_shift[nyPP] = pos[nyPP] = posStart[nyPP] + j;

						// Direction nyP
						GpuAbcCurrPoint ptP;
						ptP.pos_idx = static_cast<uint32_t>(GetLinearIndex(nyP, pos[0], pos[1], pos[2]));
						ptP.shift_idx = static_cast<uint32_t>(GetLinearIndex(nyP, pos_shift[0], pos_shift[1], pos_shift[2]));
						ptP.k1 = static_cast<float>(abc->m_K1_nyP(i, j));
						ptP.k2 = static_cast<float>(abc->m_K2_nyP(i, j));
						hostCurrPoints.push_back(ptP);

						// Direction nyPP
						GpuAbcCurrPoint ptPP;
						ptPP.pos_idx = static_cast<uint32_t>(GetLinearIndex(nyPP, pos[0], pos[1], pos[2]));
						ptPP.shift_idx = static_cast<uint32_t>(GetLinearIndex(nyPP, pos_shift[0], pos_shift[1], pos_shift[2]));
						ptPP.k1 = static_cast<float>(abc->m_K1_nyPP(i, j));
						ptPP.k2 = static_cast<float>(abc->m_K2_nyPP(i, j));
						hostCurrPoints.push_back(ptPP);
					}
				}
			}
		}

		voltSheet.count = static_cast<uint32_t>(hostVoltPoints.size()) - voltSheet.offset;
		currSheet.count = static_cast<uint32_t>(hostCurrPoints.size()) - currSheet.offset;
		if (voltSheet.count > 0) m_abcVoltSheets.push_back(voltSheet);
		if (currSheet.count > 0) m_abcCurrSheets.push_back(currSheet);
	}

	m_abcVoltCount = static_cast<uint32_t>(hostVoltPoints.size());
	m_abcCurrCount = static_cast<uint32_t>(hostCurrPoints.size());

	std::cout << "[openEMS Vulkan] Initialized on-device Absorbing BC: "
	          << m_abcVoltCount << " voltage points, "
	          << m_abcCurrCount << " current points across "
	          << abcExts.size() << " sheet(s)." << std::endl;

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	if (m_abcVoltCount > 0)
	{
		VkDeviceSize paramBytes = m_abcVoltCount * sizeof(GpuAbcVoltPoint);
		VkDeviceSize storeBytes = m_abcVoltCount * sizeof(float);

		if (!CreateBuffer(paramBytes, storageUsage, devLocal, m_bufAbcVoltParams)) return false;
		if (!CreateBuffer(storeBytes, storageUsage, devLocal, m_bufAbcVoltStore)) return false;

		VulkanBuffer stage;
		if (!CreateBuffer(paramBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;
		std::memcpy(stage.mapped, hostVoltPoints.data(), paramBytes);

		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, paramBytes };
		vkCmdCopyBuffer(m_cmdBuffer, stage.buffer, m_bufAbcVoltParams.buffer, 1, &copyRegion);
		vkCmdFillBuffer(m_cmdBuffer, m_bufAbcVoltStore.buffer, 0, storeBytes, 0);
		vkEndCommandBuffer(m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);

		if (m_descLayoutAbc != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE)
		{
			VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			dsAlloc.descriptorPool = m_descPool;
			dsAlloc.descriptorSetCount = 1;
			dsAlloc.pSetLayouts = &m_descLayoutAbc;
			if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetAbcVolt) != VK_SUCCESS)
				return false;

			VkDescriptorBufferInfo b0 = { m_bufAbcVoltParams.buffer, 0, m_bufAbcVoltParams.size };
			VkDescriptorBufferInfo b1 = { m_bufVolt.buffer, 0, m_bufVolt.size };
			VkDescriptorBufferInfo b2 = { m_bufAbcVoltStore.buffer, 0, m_bufAbcVoltStore.size };

			VkWriteDescriptorSet writes[3] = {};
			for (int w = 0; w < 3; ++w)
			{
				writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[w].dstSet = m_descSetAbcVolt;
				writes[w].dstBinding = w;
				writes[w].descriptorCount = 1;
				writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			}
			writes[0].pBufferInfo = &b0;
			writes[1].pBufferInfo = &b1;
			writes[2].pBufferInfo = &b2;
			vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);
		}
	}

	if (m_abcCurrCount > 0)
	{
		VkDeviceSize paramBytes = m_abcCurrCount * sizeof(GpuAbcCurrPoint);
		VkDeviceSize storeBytes = m_abcCurrCount * sizeof(float);

		if (!CreateBuffer(paramBytes, storageUsage, devLocal, m_bufAbcCurrParams)) return false;
		if (!CreateBuffer(storeBytes, storageUsage, devLocal, m_bufAbcCurrStore)) return false;

		VulkanBuffer stage;
		if (!CreateBuffer(paramBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;
		std::memcpy(stage.mapped, hostCurrPoints.data(), paramBytes);

		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, paramBytes };
		vkCmdCopyBuffer(m_cmdBuffer, stage.buffer, m_bufAbcCurrParams.buffer, 1, &copyRegion);
		vkCmdFillBuffer(m_cmdBuffer, m_bufAbcCurrStore.buffer, 0, storeBytes, 0);
		vkEndCommandBuffer(m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);

		if (m_descLayoutAbc != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE)
		{
			VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			dsAlloc.descriptorPool = m_descPool;
			dsAlloc.descriptorSetCount = 1;
			dsAlloc.pSetLayouts = &m_descLayoutAbc;
			if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetAbcCurr) != VK_SUCCESS)
				return false;

			VkDescriptorBufferInfo b0 = { m_bufAbcCurrParams.buffer, 0, m_bufAbcCurrParams.size };
			VkDescriptorBufferInfo b1 = { m_bufCurr.buffer, 0, m_bufCurr.size };
			VkDescriptorBufferInfo b2 = { m_bufAbcCurrStore.buffer, 0, m_bufAbcCurrStore.size };

			VkWriteDescriptorSet writes[3] = {};
			for (int w = 0; w < 3; ++w)
			{
				writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[w].dstSet = m_descSetAbcCurr;
				writes[w].dstBinding = w;
				writes[w].descriptorCount = 1;
				writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			}
			writes[0].pBufferInfo = &b0;
			writes[1].pBufferInfo = &b1;
			writes[2].pBufferInfo = &b2;
			vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);
		}
	}

	return true;
#else
	return true;
#endif
}

bool EngineVulkan::AllocateDispersiveBuffers()
{
#ifdef ENABLE_VULKAN
	if (!m_op) return true;

	std::vector<Operator_Ext_LorentzMaterial*> lorExts;
	for (size_t i = 0; i < m_op->GetNumberOfExtentions(); ++i)
	{
		Operator_Ext_LorentzMaterial* lor = dynamic_cast<Operator_Ext_LorentzMaterial*>(m_op->GetExtension(i));
		if (lor)
			lorExts.push_back(lor);
	}
	if (lorExts.empty()) return true;

	std::vector<GpuDispersivePoint> hostVoltPoints;
	std::vector<GpuDispersivePoint> hostCurrPoints;
	m_dispVoltPasses.clear();
	m_dispCurrPasses.clear();

	for (Operator_Ext_LorentzMaterial* lor : lorExts)
	{
		for (int o = 0; o < lor->m_Order; ++o)
		{
			// Voltage points for order o
			if (lor->m_volt_ADE_On && lor->m_volt_ADE_On[o] && o < static_cast<int>(lor->m_LM_Count.size()))
			{
				TfsfFace pass = { static_cast<uint32_t>(hostVoltPoints.size()), 0 };
				for (unsigned int j = 0; j < lor->m_LM_Count[o]; ++j)
				{
					unsigned int x = lor->m_LM_pos[o][0][j];
					unsigned int y = lor->m_LM_pos[o][1][j];
					unsigned int z = lor->m_LM_pos[o][2][j];
					for (int n = 0; n < 3; ++n)
					{
						float v_int = static_cast<float>(lor->v_int_ADE[o][n][j]);
						float v_ext = static_cast<float>(lor->v_ext_ADE[o][n][j]);
						float v_lor = (lor->m_volt_Lor_ADE_On && lor->m_volt_Lor_ADE_On[o] && lor->v_Lor_ADE[o])
						              ? static_cast<float>(lor->v_Lor_ADE[o][n][j]) : 0.0f;

						if (v_ext != 0.0f || v_lor != 0.0f || v_int != 1.0f)
						{
							GpuDispersivePoint pt;
							pt.pos_idx = static_cast<uint32_t>(GetLinearIndex(n, x, y, z));
							pt.int_coeff = v_int;
							pt.ext_coeff = v_ext;
							pt.lor_coeff = v_lor;
							hostVoltPoints.push_back(pt);
						}
					}
				}
				pass.count = static_cast<uint32_t>(hostVoltPoints.size()) - pass.offset;
				if (pass.count > 0) m_dispVoltPasses.push_back(pass);
			}

			// Current points for order o
			if (lor->m_curr_ADE_On && lor->m_curr_ADE_On[o] && o < static_cast<int>(lor->m_LM_Count.size()))
			{
				TfsfFace pass = { static_cast<uint32_t>(hostCurrPoints.size()), 0 };
				for (unsigned int j = 0; j < lor->m_LM_Count[o]; ++j)
				{
					unsigned int x = lor->m_LM_pos[o][0][j];
					unsigned int y = lor->m_LM_pos[o][1][j];
					unsigned int z = lor->m_LM_pos[o][2][j];
					for (int n = 0; n < 3; ++n)
					{
						float i_int = static_cast<float>(lor->i_int_ADE[o][n][j]);
						float i_ext = static_cast<float>(lor->i_ext_ADE[o][n][j]);
						float i_lor = (lor->m_curr_Lor_ADE_On && lor->m_curr_Lor_ADE_On[o] && lor->i_Lor_ADE[o])
						              ? static_cast<float>(lor->i_Lor_ADE[o][n][j]) : 0.0f;

						if (i_ext != 0.0f || i_lor != 0.0f || i_int != 1.0f)
						{
							GpuDispersivePoint pt;
							pt.pos_idx = static_cast<uint32_t>(GetLinearIndex(n, x, y, z));
							pt.int_coeff = i_int;
							pt.ext_coeff = i_ext;
							pt.lor_coeff = i_lor;
							hostCurrPoints.push_back(pt);
						}
					}
				}
				pass.count = static_cast<uint32_t>(hostCurrPoints.size()) - pass.offset;
				if (pass.count > 0) m_dispCurrPasses.push_back(pass);
			}
		}
	}

	m_dispVoltCount = static_cast<uint32_t>(hostVoltPoints.size());
	m_dispCurrCount = static_cast<uint32_t>(hostCurrPoints.size());

	if (m_dispVoltCount == 0 && m_dispCurrCount == 0) return true;

	std::cout << "[openEMS Vulkan] Initialized on-device Dispersive Media: "
	          << m_dispVoltCount << " voltage components across " << m_dispVoltPasses.size() << " pass(es), "
	          << m_dispCurrCount << " current components across " << m_dispCurrPasses.size() << " pass(es)." << std::endl;

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	if (m_dispVoltCount > 0)
	{
		VkDeviceSize paramBytes = m_dispVoltCount * sizeof(GpuDispersivePoint);
		VkDeviceSize stateBytes = m_dispVoltCount * sizeof(GpuDispersiveState);

		if (!CreateBuffer(paramBytes, storageUsage, devLocal, m_bufDispVoltParams)) return false;
		if (!CreateBuffer(stateBytes, storageUsage, devLocal, m_bufDispVoltState)) return false;

		VulkanBuffer stage;
		if (!CreateBuffer(paramBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;

		void* pData = nullptr;
		vkMapMemory(m_device, stage.memory, 0, paramBytes, 0, &pData);
		memcpy(pData, hostVoltPoints.data(), paramBytes);
		vkUnmapMemory(m_device, stage.memory);

		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, paramBytes };
		vkCmdCopyBuffer(m_cmdBuffer, stage.buffer, m_bufDispVoltParams.buffer, 1, &copyRegion);
		vkCmdFillBuffer(m_cmdBuffer, m_bufDispVoltState.buffer, 0, stateBytes, 0);
		vkEndCommandBuffer(m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);

		if (m_descLayoutDisp != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE)
		{
			VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			dsAlloc.descriptorPool = m_descPool;
			dsAlloc.descriptorSetCount = 1;
			dsAlloc.pSetLayouts = &m_descLayoutDisp;
			if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetDispVolt) != VK_SUCCESS)
				return false;

			VkDescriptorBufferInfo b0 = { m_bufDispVoltParams.buffer, 0, m_bufDispVoltParams.size };
			VkDescriptorBufferInfo b1 = { m_bufVolt.buffer, 0, m_bufVolt.size };
			VkDescriptorBufferInfo b2 = { m_bufDispVoltState.buffer, 0, m_bufDispVoltState.size };

			VkWriteDescriptorSet writes[3] = {};
			for (int w = 0; w < 3; ++w)
			{
				writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[w].dstSet = m_descSetDispVolt;
				writes[w].dstBinding = w;
				writes[w].descriptorCount = 1;
				writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			}
			writes[0].pBufferInfo = &b0;
			writes[1].pBufferInfo = &b1;
			writes[2].pBufferInfo = &b2;
			vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);
		}
	}

	if (m_dispCurrCount > 0)
	{
		VkDeviceSize paramBytes = m_dispCurrCount * sizeof(GpuDispersivePoint);
		VkDeviceSize stateBytes = m_dispCurrCount * sizeof(GpuDispersiveState);

		if (!CreateBuffer(paramBytes, storageUsage, devLocal, m_bufDispCurrParams)) return false;
		if (!CreateBuffer(stateBytes, storageUsage, devLocal, m_bufDispCurrState)) return false;

		VulkanBuffer stage;
		if (!CreateBuffer(paramBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;

		void* pData = nullptr;
		vkMapMemory(m_device, stage.memory, 0, paramBytes, 0, &pData);
		memcpy(pData, hostCurrPoints.data(), paramBytes);
		vkUnmapMemory(m_device, stage.memory);

		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, paramBytes };
		vkCmdCopyBuffer(m_cmdBuffer, stage.buffer, m_bufDispCurrParams.buffer, 1, &copyRegion);
		vkCmdFillBuffer(m_cmdBuffer, m_bufDispCurrState.buffer, 0, stateBytes, 0);
		vkEndCommandBuffer(m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
		vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);

		if (m_descLayoutDisp != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE)
		{
			VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			dsAlloc.descriptorPool = m_descPool;
			dsAlloc.descriptorSetCount = 1;
			dsAlloc.pSetLayouts = &m_descLayoutDisp;
			if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetDispCurr) != VK_SUCCESS)
				return false;

			VkDescriptorBufferInfo b0 = { m_bufDispCurrParams.buffer, 0, m_bufDispCurrParams.size };
			VkDescriptorBufferInfo b1 = { m_bufCurr.buffer, 0, m_bufCurr.size };
			VkDescriptorBufferInfo b2 = { m_bufDispCurrState.buffer, 0, m_bufDispCurrState.size };

			VkWriteDescriptorSet writes[3] = {};
			for (int w = 0; w < 3; ++w)
			{
				writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[w].dstSet = m_descSetDispCurr;
				writes[w].dstBinding = w;
				writes[w].descriptorCount = 1;
				writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			}
			writes[0].pBufferInfo = &b0;
			writes[1].pBufferInfo = &b1;
			writes[2].pBufferInfo = &b2;
			vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);
		}
	}

	return true;
#else
	return true;
#endif
}

bool EngineVulkan::AllocateCylinderBuffers()
{
#ifdef ENABLE_VULKAN
	if (!m_op) return true;

	Operator_Ext_Cylinder* cyl = nullptr;
	for (size_t i = 0; i < m_op->GetNumberOfExtentions(); ++i)
	{
		Operator_Ext_Cylinder* c = dynamic_cast<Operator_Ext_Cylinder*>(m_op->GetExtension(i));
		if (c)
		{
			cyl = c;
			break;
		}
	}

	if (!cyl)
	{
		m_hasCylinder = false;
		return true;
	}

	m_hasCylinder = true;
	m_cylClosedAlpha = cyl->CC_closedAlpha;
	m_cylR0Included = cyl->CC_R0_included;
	m_cylLastALine = (m_grid.dimY >= 2) ? (m_grid.dimY - 2) : 0;

	size_t numZ = m_grid.dimZ > 0 ? m_grid.dimZ : 1;
	std::vector<float> r0Data(numZ * 2, 0.0f);
	if (m_cylR0Included && cyl->vv_R0 && cyl->vi_R0)
	{
		for (size_t z = 0; z < m_grid.dimZ; ++z)
		{
			r0Data[2 * z]     = static_cast<float>(cyl->vv_R0[z]);
			r0Data[2 * z + 1] = static_cast<float>(cyl->vi_R0[z]);
		}
	}

	VkDeviceSize r0Bytes = r0Data.size() * sizeof(float);
	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	if (!CreateBuffer(r0Bytes, storageUsage, devLocal, m_bufCylR0)) return false;

	VulkanBuffer stage;
	if (!CreateBuffer(r0Bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
		return false;

	void* pData = nullptr;
	vkMapMemory(m_device, stage.memory, 0, r0Bytes, 0, &pData);
	memcpy(pData, r0Data.data(), r0Bytes);
	vkUnmapMemory(m_device, stage.memory);

	vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
	vkResetFences(m_device, 1, &m_fence);

	VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);

	VkBufferCopy copyRegion = { 0, 0, r0Bytes };
	vkCmdCopyBuffer(m_cmdBuffer, stage.buffer, m_bufCylR0.buffer, 1, &copyRegion);

	VkMemoryBarrier memBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
	memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

	vkEndCommandBuffer(m_cmdBuffer);

	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &m_cmdBuffer;
	vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
	vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
	DestroyBuffer(stage);

	if (m_descLayoutCyl != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE)
	{
		VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		dsAlloc.descriptorPool = m_descPool;
		dsAlloc.descriptorSetCount = 1;
		dsAlloc.pSetLayouts = &m_descLayoutCyl;
		if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_descSetCyl) != VK_SUCCESS)
			return false;

		VkDescriptorBufferInfo b0 = { m_bufVolt.buffer, 0, m_bufVolt.size };
		VkDescriptorBufferInfo b1 = { m_bufCurr.buffer, 0, m_bufCurr.size };
		VkDescriptorBufferInfo b2 = { m_bufCylR0.buffer, 0, m_bufCylR0.size };

		VkWriteDescriptorSet writes[3] = {};
		for (int w = 0; w < 3; ++w)
		{
			writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writes[w].dstSet = m_descSetCyl;
			writes[w].dstBinding = w;
			writes[w].descriptorCount = 1;
			writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		}
		writes[0].pBufferInfo = &b0;
		writes[1].pBufferInfo = &b1;
		writes[2].pBufferInfo = &b2;
		vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);
	}

	std::cout << "[openEMS Vulkan] Initialized cylindrical coordinates extension (closedAlpha="
	          << m_cylClosedAlpha << ", R0Included=" << m_cylR0Included << ")" << std::endl;
	return true;
#else
	return true;
#endif
}

bool EngineVulkan::PrepareVoltExcitation(unsigned int step)
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
	if (anyNonZero && m_bufVoltExcPoints.mapped)
	{
		std::memcpy(m_bufVoltExcPoints.mapped, m_voltExcPoints.data(), m_voltExcPoints.size() * sizeof(GpuExcPoint));
	}
	return anyNonZero;
}

bool EngineVulkan::PrepareCurrExcitation(unsigned int step)
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
	if (anyNonZero && m_bufCurrExcPoints.mapped)
	{
		std::memcpy(m_bufCurrExcPoints.mapped, m_currExcPoints.data(), m_currExcPoints.size() * sizeof(GpuExcPoint));
	}
	return anyNonZero;
}

#endif // ENABLE_VULKAN

void EngineVulkan::RegisterProbes(const ProcessingArray* pa)
{
	m_pa = pa;
	m_probePoints.clear();
	if (!m_pa) return;

	for (size_t i = 0; i < m_pa->GetNumberOfProcessings(); ++i)
	{
		Processing* proc = const_cast<ProcessingArray*>(m_pa)->GetProcessing(i);
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
						m_probePoints.push_back({static_cast<uint32_t>(idx), 0u});
					}
				}
				else if (start[n] > stop[n])
				{
					unsigned int pos[3] = {stop[0], stop[1], stop[2]};
					for (; pos[n] < start[n]; ++pos[n])
					{
						size_t idx = GetLinearIndex(n, pos[0], pos[1], pos[2]);
						m_probePoints.push_back({static_cast<uint32_t>(idx), 0u});
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
						m_probePoints.push_back({static_cast<uint32_t>(GetLinearIndex(1, stop[0], k, start[2])), 1u});
				if (m_stop_inside[0] && m_stop_inside[1])
					for (unsigned int k = start[2] + 1; k <= stop[2]; ++k)
						m_probePoints.push_back({static_cast<uint32_t>(GetLinearIndex(2, stop[0], stop[1], k)), 1u});
				if (m_start_inside[0] && m_stop_inside[2])
					for (unsigned int k = start[1] + 1; k <= stop[1]; ++k)
						m_probePoints.push_back({static_cast<uint32_t>(GetLinearIndex(1, start[0], k, stop[2])), 1u});
				if (m_start_inside[0] && m_start_inside[1])
					for (unsigned int k = start[2] + 1; k <= stop[2]; ++k)
						m_probePoints.push_back({static_cast<uint32_t>(GetLinearIndex(2, start[0], start[1], k)), 1u});
				break;
			case 1:
				if (m_start_inside[0] && m_start_inside[1])
					for (unsigned int k = start[2] + 1; k <= stop[2]; ++k)
						m_probePoints.push_back({static_cast<uint32_t>(GetLinearIndex(2, start[0], start[1], k)), 1u});
				if (m_stop_inside[1] && m_stop_inside[2])
					for (unsigned int k = start[0] + 1; k <= stop[0]; ++k)
						m_probePoints.push_back({static_cast<uint32_t>(GetLinearIndex(0, k, stop[1], stop[2])), 1u});
				if (m_stop_inside[0] && m_stop_inside[1])
					for (unsigned int k = start[2] + 1; k <= stop[2]; ++k)
						m_probePoints.push_back({static_cast<uint32_t>(GetLinearIndex(2, stop[0], stop[1], k)), 1u});
				if (m_start_inside[1] && m_start_inside[2])
					for (unsigned int k = start[0] + 1; k <= stop[0]; ++k)
						m_probePoints.push_back({static_cast<uint32_t>(GetLinearIndex(0, k, start[1], start[2])), 1u});
				break;
			case 2:
				if (m_start_inside[1] && m_start_inside[2])
					for (unsigned int k = start[0] + 1; k <= stop[0]; ++k)
						m_probePoints.push_back({static_cast<uint32_t>(GetLinearIndex(0, k, start[1], start[2])), 1u});
				if (m_stop_inside[0] && m_start_inside[2])
					for (unsigned int k = start[1] + 1; k <= stop[1]; ++k)
						m_probePoints.push_back({static_cast<uint32_t>(GetLinearIndex(1, stop[0], k, start[2])), 1u});
				if (m_stop_inside[1] && m_stop_inside[2])
					for (unsigned int k = start[0] + 1; k <= stop[0]; ++k)
						m_probePoints.push_back({static_cast<uint32_t>(GetLinearIndex(0, k, stop[1], stop[2])), 1u});
				if (m_start_inside[0] && m_stop_inside[2])
					for (unsigned int k = start[1] + 1; k <= stop[1]; ++k)
						m_probePoints.push_back({static_cast<uint32_t>(GetLinearIndex(1, start[0], k, stop[2])), 1u});
				break;
			}
			continue;
		}

		ProcessFieldProbe* pfp = dynamic_cast<ProcessFieldProbe*>(proc);
		if (pfp)
		{
			const unsigned int* start = pfp->GetStartCoord();
			uint32_t isCurr = (pfp->GetFieldType() == 1 ? 1u : 0u);
			for (unsigned int n = 0; n < 3; ++n)
			{
				size_t idx = GetLinearIndex(n, start[0], start[1], start[2]);
				m_probePoints.push_back({static_cast<uint32_t>(idx), isCurr});
			}
			continue;
		}
	}

	if (m_op)
	{
		for (size_t i = 0; i < m_op->GetNumberOfExtentions(); ++i)
		{
			Operator_Extension* ext = m_op->GetExtension(i);
			Operator_Ext_SteadyState* ss = dynamic_cast<Operator_Ext_SteadyState*>(ext);
			if (ss)
			{
				for (size_t n = 0; n < ss->GetNumberOfEProbes(); ++n)
				{
					unsigned int pos[3];
					ss->GetEProbePos(n, pos);
					unsigned int dir = ss->GetEProbeDir(n);
					size_t idx = GetLinearIndex(dir, pos[0], pos[1], pos[2]);
					m_probePoints.push_back({static_cast<uint32_t>(idx), 0u});
				}
				for (size_t n = 0; n < ss->GetNumberOfHProbes(); ++n)
				{
					unsigned int pos[3];
					ss->GetHProbePos(n, pos);
					unsigned int dir = ss->GetHProbeDir(n);
					size_t idx = GetLinearIndex(dir, pos[0], pos[1], pos[2]);
					m_probePoints.push_back({static_cast<uint32_t>(idx), 1u});
				}
			}
		}
	}

	// Sort and deduplicate probe points
	std::sort(m_probePoints.begin(), m_probePoints.end(), [](const ProbePoint& a, const ProbePoint& b) {
		if (a.field_type != b.field_type) return a.field_type < b.field_type;
		return a.linear_index < b.linear_index;
	});
	m_probePoints.erase(std::unique(m_probePoints.begin(), m_probePoints.end(), [](const ProbePoint& a, const ProbePoint& b) {
		return a.field_type == b.field_type && a.linear_index == b.linear_index;
	}), m_probePoints.end());

	if (!m_probePoints.empty())
	{
		std::cout << "[openEMS Vulkan] Registered " << m_probePoints.size()
		          << " active probe monitoring points for fast on-device extraction." << std::endl;
#ifdef ENABLE_VULKAN
		AllocateProbeBuffers();
#endif
	}
}

bool EngineVulkan::IterateTS(unsigned int iterTS)
{
#ifdef ENABLE_VULKAN
	if (m_device && m_computeQueue && m_pipelineVolt && m_pipelineCurr)
	{
		if (!SyncFieldsToDevice())
			return false;

		uint32_t wgZ = (m_grid.dimZ + 31) / 32;
		uint32_t wgY = (m_grid.dimY + 3) / 4;
		uint32_t wgX = (m_grid.dimX + 1) / 2;

		struct YeePushConstants {
			uint32_t dimX;
			uint32_t dimY;
			uint32_t dimZ;
			uint32_t numCells;
		} pc = { m_grid.dimX, m_grid.dimY, m_grid.dimZ, m_grid.numCells };

		VkMemoryBarrier memBarrier = {};
		memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

		for (unsigned int step = 0; step < iterTS; ++step)
		{
			vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
			vkResetFences(m_device, 1, &m_fence);

			bool hasVoltExc = PrepareVoltExcitation(m_numTS);
			bool hasCurrExc = PrepareCurrExcitation(m_numTS);

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);

			// 1. UPML Pre-Voltage Pass
			if (m_numUpmlCells > 0 && m_pipelineUpmlPre && m_descSetUpmlVolt)
			{
				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineUpmlPre);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutUpml, 0, 1, &m_descSetUpmlVolt, 0, nullptr);
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutUpml, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_numUpmlCells), &m_numUpmlCells);
				vkCmdDispatch(m_cmdBuffer, (m_numUpmlCells + 255) / 256, 1, 1);

				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
			}

			// Mur ABC Pre-Voltage Pass
			if (m_totalMurPoints > 0 && m_pipelineMurPre && m_descSetMur)
			{
				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineMurPre);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutMur, 0, 1, &m_descSetMur, 0, nullptr);
				struct {
					uint32_t offset;
					uint32_t count;
					uint32_t current_TS;
					uint32_t pad;
				} murPC = { 0, m_totalMurPoints, m_numTS, 0 };
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutMur, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(murPC), &murPC);
				vkCmdDispatch(m_cmdBuffer, (m_totalMurPoints + 255) / 256, 1, 1);

				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
			}

			// Absorbing BC Pre-Voltage Pass
			if (m_abcVoltCount > 0 && m_pipelineAbcVolt && m_descSetAbcVolt)
			{
				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineAbcVolt);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutAbc, 0, 1, &m_descSetAbcVolt, 0, nullptr);
				struct { uint32_t offset; uint32_t count; uint32_t mode; } abcPC = { 0, m_abcVoltCount, 0 };
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutAbc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(abcPC), &abcPC);
				vkCmdDispatch(m_cmdBuffer, (m_abcVoltCount + 255) / 256, 1, 1);

				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
			}

			// Lumped RLC Pre-Voltage Pass
			if (m_rlcCount > 0 && m_pipelineRlc && m_descSetRlc)
			{
				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineRlc);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutRlc, 0, 1, &m_descSetRlc, 0, nullptr);
				struct {
					uint32_t count;
					uint32_t mode;
				} rlcPC = { m_rlcCount, 0 };
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutRlc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rlcPC), &rlcPC);
				vkCmdDispatch(m_cmdBuffer, (m_rlcCount + 63) / 64, 1, 1);

				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
			}

			// Dispersive Media Pre-Voltage Pass
			if (m_dispVoltCount > 0 && m_pipelineDisp && m_descSetDispVolt)
			{
				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineDisp);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutDisp, 0, 1, &m_descSetDispVolt, 0, nullptr);
				for (const auto& pass : m_dispVoltPasses)
				{
					struct { uint32_t offset; uint32_t count; uint32_t mode; } dispPC = { pass.offset, pass.count, 0 };
					vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutDisp, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dispPC), &dispPC);
					vkCmdDispatch(m_cmdBuffer, (pass.count + 255) / 256, 1, 1);
				}

				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
			}

			// 2. Voltage update
			vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineVolt);
			vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutFields, 0, 1, &m_descSetFields, 0, nullptr);
			vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutFields, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			vkCmdDispatch(m_cmdBuffer, wgZ, wgY, wgX);

			// 3. UPML Post-Voltage Pass
			if (m_numUpmlCells > 0 && m_pipelineUpmlPost && m_descSetUpmlVolt)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineUpmlPost);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutUpml, 0, 1, &m_descSetUpmlVolt, 0, nullptr);
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutUpml, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_numUpmlCells), &m_numUpmlCells);
				vkCmdDispatch(m_cmdBuffer, (m_numUpmlCells + 255) / 256, 1, 1);
			}

			// Cylindrical Coordinates Post-Voltage Pass
			if (m_hasCylinder && m_cylClosedAlpha && m_pipelineCyl && m_descSetCyl)
			{
				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineCyl);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutCyl, 0, 1, &m_descSetCyl, 0, nullptr);

				struct CylPC {
					uint32_t dimX;
					uint32_t dimY;
					uint32_t dimZ;
					uint32_t numCells;
					uint32_t last_A_Line;
					uint32_t hasR0;
					uint32_t mode;
				};

				if (m_cylR0Included)
				{
					vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

					CylPC r0PC = { m_grid.dimX, m_grid.dimY, m_grid.dimZ, m_grid.numCells, m_cylLastALine, 1u, 0u };
					vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutCyl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(r0PC), &r0PC);
					vkCmdDispatch(m_cmdBuffer, (m_grid.dimZ + 255) / 256, 1, 1);
				}

				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				CylPC wrapPC = { m_grid.dimX, m_grid.dimY, m_grid.dimZ, m_grid.numCells, m_cylLastALine, m_cylR0Included ? 1u : 0u, 1u };
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutCyl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(wrapPC), &wrapPC);
				vkCmdDispatch(m_cmdBuffer, (m_grid.dimX + 15) / 16, (m_grid.dimZ + 15) / 16, 1);
			}

			// TFSF has a higher priority than the boundary and lumped-element
			// extensions, so its entire post-update phase must run first.
			if (!m_tfsfVoltFaces.empty() && m_pipelineTfsf && m_descSetTfsfVolt)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineTfsf);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutTfsf, 0, 1, &m_descSetTfsfVolt, 0, nullptr);

				for (const auto& face : m_tfsfVoltFaces)
				{
					struct {
						uint32_t offset;
						uint32_t count;
						uint32_t current_TS;
						uint32_t sigLength;
						int32_t p;
					} tfsfPC = { face.offset, face.count, m_numTS, m_tfsfSigLength, m_tfsfPeriod };
					vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutTfsf, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tfsfPC), &tfsfPC);
					vkCmdDispatch(m_cmdBuffer, (face.count + 255) / 256, 1, 1);

					if (m_tfsfVoltFaces.size() > 1)
						vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
				}
			}

			// Equal-priority extensions are reversed by Engine's stable-sort plus
			// reverse sequence. Local sheets therefore precede Mur boundaries.
			if (m_abcVoltCount > 0 && m_pipelineAbcVolt && m_descSetAbcVolt)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineAbcVolt);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutAbc, 0, 1, &m_descSetAbcVolt, 0, nullptr);
				struct { uint32_t offset; uint32_t count; uint32_t mode; } abcPC = { 0, m_abcVoltCount, 1 };
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutAbc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(abcPC), &abcPC);
				vkCmdDispatch(m_cmdBuffer, (m_abcVoltCount + 255) / 256, 1, 1);
			}

			// Mur ABC Post-Voltage Pass (updates storeData += coeff * voltData[shift])
			if (m_totalMurPoints > 0 && m_pipelineMurPost && m_descSetMur)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineMurPost);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutMur, 0, 1, &m_descSetMur, 0, nullptr);
				struct {
					uint32_t offset;
					uint32_t count;
					uint32_t current_TS;
					uint32_t pad;
				} murPC = { 0, m_totalMurPoints, m_numTS, 0 };
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutMur, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(murPC), &murPC);
				vkCmdDispatch(m_cmdBuffer, (m_totalMurPoints + 255) / 256, 1, 1);
			}

			// Apply phases begin only after every extension's post-update phase
			// has completed, matching Engine::IterateTS.
			// Absorbing BC Apply Pass
			if (m_abcVoltCount > 0 && m_pipelineAbcVolt && m_descSetAbcVolt)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineAbcVolt);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutAbc, 0, 1, &m_descSetAbcVolt, 0, nullptr);
				for (auto it = m_abcVoltSheets.rbegin(); it != m_abcVoltSheets.rend(); ++it)
				{
					const TfsfFace& sheet = *it;
					struct { uint32_t offset; uint32_t count; uint32_t mode; } abcPC = { sheet.offset, sheet.count, 2 };
					vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutAbc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(abcPC), &abcPC);
					vkCmdDispatch(m_cmdBuffer, (sheet.count + 255) / 256, 1, 1);
					if (m_abcVoltSheets.size() > 1)
						vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
				}
			}

			// Lumped RLC Apply2Voltages. Calculation and field writes are split
			// so every element sees the same pre-RLC Yee voltage.
			if (m_rlcCount > 0 && m_pipelineRlc && m_descSetRlc)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineRlc);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutRlc, 0, 1, &m_descSetRlc, 0, nullptr);
				struct {
					uint32_t count;
					uint32_t mode;
				} rlcPC = { m_rlcCount, 1 };
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutRlc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rlcPC), &rlcPC);
				vkCmdDispatch(m_cmdBuffer, (m_rlcCount + 63) / 64, 1, 1);

				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
				rlcPC.mode = 2;
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutRlc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rlcPC), &rlcPC);
				vkCmdDispatch(m_cmdBuffer, (m_rlcCount + 63) / 64, 1, 1);
			}

			// Dispersive Media Apply-Voltage Pass
			if (m_dispVoltCount > 0 && m_pipelineDisp && m_descSetDispVolt)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineDisp);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutDisp, 0, 1, &m_descSetDispVolt, 0, nullptr);
				for (const auto& pass : m_dispVoltPasses)
				{
					struct { uint32_t offset; uint32_t count; uint32_t mode; } dispPC = { pass.offset, pass.count, 1 };
					vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutDisp, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dispPC), &dispPC);
					vkCmdDispatch(m_cmdBuffer, (pass.count + 255) / 256, 1, 1);
					if (m_dispVoltPasses.size() > 1)
						vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
				}
			}

			// Mur boundary extensions are the earliest default-priority
			// extensions added by openEMS, and therefore apply last among that
			// priority group after Engine reverses its sorted extension list.
			if (m_totalMurPoints > 0 && m_pipelineMurApply && m_descSetMur)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineMurApply);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutMur, 0, 1, &m_descSetMur, 0, nullptr);

				for (auto it = m_murFaces.rbegin(); it != m_murFaces.rend(); ++it)
				{
					const MurFace& face = *it;
					if (m_numTS < face.start_TS)
						continue;
					struct {
						uint32_t offset;
						uint32_t count;
						uint32_t current_TS;
						uint32_t pad;
					} murPC = { face.offset, face.count, m_numTS, 0 };
					vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutMur, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(murPC), &murPC);
					vkCmdDispatch(m_cmdBuffer, (face.count + 255) / 256, 1, 1);

					if (m_murFaces.size() > 1)
						vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
				}
			}

			// Excitation has the lowest extension priority and therefore applies
			// after boundary and lumped-element corrections.
			if (hasVoltExc && m_pipelineExc && m_descSetVoltExc)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				uint32_t count = static_cast<uint32_t>(m_voltExcPoints.size());
				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineExc);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutExc, 0, 1, &m_descSetVoltExc, 0, nullptr);
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutExc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count), &count);
				vkCmdDispatch(m_cmdBuffer, (count + 63) / 64, 1, 1);
			}

			// Barrier between voltage and current
			vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

			// 5. UPML Pre-Current Pass
			if (m_numUpmlCells > 0 && m_pipelineUpmlPre && m_descSetUpmlCurr)
			{
				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineUpmlPre);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutUpml, 0, 1, &m_descSetUpmlCurr, 0, nullptr);
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutUpml, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_numUpmlCells), &m_numUpmlCells);
				vkCmdDispatch(m_cmdBuffer, (m_numUpmlCells + 255) / 256, 1, 1);

				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
			}

			// Absorbing BC Pre-Current Pass
			if (m_abcCurrCount > 0 && m_pipelineAbcCurr && m_descSetAbcCurr)
			{
				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineAbcCurr);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutAbc, 0, 1, &m_descSetAbcCurr, 0, nullptr);
				struct { uint32_t offset; uint32_t count; uint32_t mode; } abcPC = { 0, m_abcCurrCount, 0 };
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutAbc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(abcPC), &abcPC);
				vkCmdDispatch(m_cmdBuffer, (m_abcCurrCount + 255) / 256, 1, 1);

				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
			}

			// Dispersive Media Pre-Current Pass
			if (m_dispCurrCount > 0 && m_pipelineDisp && m_descSetDispCurr)
			{
				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineDisp);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutDisp, 0, 1, &m_descSetDispCurr, 0, nullptr);
				for (const auto& pass : m_dispCurrPasses)
				{
					struct { uint32_t offset; uint32_t count; uint32_t mode; } dispPC = { pass.offset, pass.count, 0 };
					vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutDisp, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dispPC), &dispPC);
					vkCmdDispatch(m_cmdBuffer, (pass.count + 255) / 256, 1, 1);
				}

				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
			}

			// 6. Current update
			vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineCurr);
			vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutFields, 0, 1, &m_descSetFields, 0, nullptr);
			vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutFields, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			vkCmdDispatch(m_cmdBuffer, wgZ, wgY, wgX);

			// 7. UPML Post-Current Pass
			if (m_numUpmlCells > 0 && m_pipelineUpmlPost && m_descSetUpmlCurr)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineUpmlPost);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutUpml, 0, 1, &m_descSetUpmlCurr, 0, nullptr);
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutUpml, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_numUpmlCells), &m_numUpmlCells);
				vkCmdDispatch(m_cmdBuffer, (m_numUpmlCells + 255) / 256, 1, 1);
			}

			// Cylindrical Coordinates Post-Current Pass
			if (m_hasCylinder && m_cylClosedAlpha && m_pipelineCyl && m_descSetCyl)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineCyl);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutCyl, 0, 1, &m_descSetCyl, 0, nullptr);

				struct CylPC {
					uint32_t dimX;
					uint32_t dimY;
					uint32_t dimZ;
					uint32_t numCells;
					uint32_t last_A_Line;
					uint32_t hasR0;
					uint32_t mode;
				};

				CylPC currPC = { m_grid.dimX, m_grid.dimY, m_grid.dimZ, m_grid.numCells, m_cylLastALine, 0u, 2u };
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutCyl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(currPC), &currPC);
				vkCmdDispatch(m_cmdBuffer, (m_grid.dimX + 15) / 16, (m_grid.dimZ + 15) / 16, 1);
			}

			// TFSF post-current updates precede default-priority boundary sheets.
			if (!m_tfsfCurrFaces.empty() && m_pipelineTfsf && m_descSetTfsfCurr)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineTfsf);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutTfsf, 0, 1, &m_descSetTfsfCurr, 0, nullptr);

				for (const auto& face : m_tfsfCurrFaces)
				{
					struct {
						uint32_t offset;
						uint32_t count;
						uint32_t current_TS;
						uint32_t sigLength;
						int32_t p;
					} tfsfPC = { face.offset, face.count, m_numTS, m_tfsfSigLength, m_tfsfPeriod };
					vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutTfsf, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tfsfPC), &tfsfPC);
					vkCmdDispatch(m_cmdBuffer, (face.count + 255) / 256, 1, 1);

					if (m_tfsfCurrFaces.size() > 1)
						vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
				}
			}

			// Absorbing BC Post-Current Pass
			if (m_abcCurrCount > 0 && m_pipelineAbcCurr && m_descSetAbcCurr)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineAbcCurr);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutAbc, 0, 1, &m_descSetAbcCurr, 0, nullptr);
				struct { uint32_t offset; uint32_t count; uint32_t mode; } abcPC = { 0, m_abcCurrCount, 1 };
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutAbc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(abcPC), &abcPC);
				vkCmdDispatch(m_cmdBuffer, (m_abcCurrCount + 255) / 256, 1, 1);
			}

			// Absorbing BC Apply Pass
			if (m_abcCurrCount > 0 && m_pipelineAbcCurr && m_descSetAbcCurr)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineAbcCurr);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutAbc, 0, 1, &m_descSetAbcCurr, 0, nullptr);
				for (auto it = m_abcCurrSheets.rbegin(); it != m_abcCurrSheets.rend(); ++it)
				{
					const TfsfFace& sheet = *it;
					struct { uint32_t offset; uint32_t count; uint32_t mode; } abcPC = { sheet.offset, sheet.count, 2 };
					vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutAbc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(abcPC), &abcPC);
					vkCmdDispatch(m_cmdBuffer, (sheet.count + 255) / 256, 1, 1);
					if (m_abcCurrSheets.size() > 1)
						vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
				}
			}

			// Dispersive Media Apply-Current Pass
			if (m_dispCurrCount > 0 && m_pipelineDisp && m_descSetDispCurr)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineDisp);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutDisp, 0, 1, &m_descSetDispCurr, 0, nullptr);
				for (const auto& pass : m_dispCurrPasses)
				{
					struct { uint32_t offset; uint32_t count; uint32_t mode; } dispPC = { pass.offset, pass.count, 1 };
					vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutDisp, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dispPC), &dispPC);
					vkCmdDispatch(m_cmdBuffer, (pass.count + 255) / 256, 1, 1);
					if (m_dispCurrPasses.size() > 1)
						vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);
				}
			}

			// 8. Current excitation pass
			if (hasCurrExc && m_pipelineExc && m_descSetCurrExc)
			{
				vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

				uint32_t count = static_cast<uint32_t>(m_currExcPoints.size());
				vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineExc);
				vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutExc, 0, 1, &m_descSetCurrExc, 0, nullptr);
				vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutExc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count), &count);
				vkCmdDispatch(m_cmdBuffer, (count + 63) / 64, 1, 1);
			}

			// Barrier between current and next step
			vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

			vkEndCommandBuffer(m_cmdBuffer);

			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &m_cmdBuffer;
			vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);

			++m_numTS;
		}

		m_hostFieldsValid = false;
		return true;
	}
#endif

	for (unsigned int step = 0; step < iterTS; ++step)
	{
		++m_numTS;
	}
	m_hostFieldsValid = false;
	return true;
}

bool EngineVulkan::SyncProbesToHost()
{
#ifdef ENABLE_VULKAN
	if (m_probePoints.empty() || !m_bufProbeValues.mapped) return true;

	uint32_t count = static_cast<uint32_t>(m_probePoints.size());

	vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
	vkResetFences(m_device, 1, &m_fence);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);

	vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineProbe);
	vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutProbe, 0, 1, &m_descSetProbe, 0, nullptr);
	vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutProbe, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count), &count);
	vkCmdDispatch(m_cmdBuffer, (count + 63) / 64, 1, 1);

	// Barrier to host read
	VkMemoryBarrier hostBarrier = {};
	hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
	vkCmdPipelineBarrier(m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
	                     0, 1, &hostBarrier, 0, nullptr, 0, nullptr);

	vkEndCommandBuffer(m_cmdBuffer);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &m_cmdBuffer;
	vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
	vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

	// Direct zero-copy read from host-visible mapped memory!
	const float* values = static_cast<const float*>(m_bufProbeValues.mapped);
	Engine* cpuEng = (m_op ? m_op->GetEngine() : nullptr);
	FDTD_FLOAT* engVolt = (cpuEng && cpuEng->GetVoltArray()) ? cpuEng->GetVoltArray()->data() : nullptr;
	FDTD_FLOAT* engCurr = (cpuEng && cpuEng->GetCurrArray()) ? cpuEng->GetCurrArray()->data() : nullptr;

	for (size_t i = 0; i < m_probePoints.size(); ++i)
	{
		uint32_t linIdx = m_probePoints[i].linear_index;
		uint32_t fType  = m_probePoints[i].field_type;
		float val = values[i];

		if (fType == 1u)
		{
			if (linIdx < m_hostCurr.size()) m_hostCurr[linIdx] = val;
			if (engCurr && linIdx < cpuEng->GetCurrArray()->size())
			{
				engCurr[linIdx] = val;
			}
			else if (cpuEng)
			{
				uint32_t ny = linIdx / m_grid.numCells;
				uint32_t rem = linIdx % m_grid.numCells;
				uint32_t slice = m_grid.dimY * m_grid.dimZ;
				uint32_t x = rem / slice;
				uint32_t y = (rem % slice) / m_grid.dimZ;
				uint32_t z = rem % m_grid.dimZ;
				cpuEng->SetCurr(ny, x, y, z, val);
			}
		}
		else
		{
			if (linIdx < m_hostVolt.size()) m_hostVolt[linIdx] = val;
			if (engVolt && linIdx < cpuEng->GetVoltArray()->size())
			{
				engVolt[linIdx] = val;
			}
			else if (cpuEng)
			{
				uint32_t ny = linIdx / m_grid.numCells;
				uint32_t rem = linIdx % m_grid.numCells;
				uint32_t slice = m_grid.dimY * m_grid.dimZ;
				uint32_t x = rem / slice;
				uint32_t y = (rem % slice) / m_grid.dimZ;
				uint32_t z = rem % m_grid.dimZ;
				cpuEng->SetVolt(ny, x, y, z, val);
			}
		}
	}
	return true;
#else
	return true;
#endif
}

bool EngineVulkan::SyncFieldsToHost()
{
	if (m_hostFieldsValid) return true;

#ifdef ENABLE_VULKAN
	if (m_device && m_bufFieldStaging.mapped)
	{
		size_t fieldBytes = 3 * static_cast<size_t>(m_grid.numCells) * sizeof(float);

		auto downloadBuffer = [&](VulkanBuffer& srcBuf, std::vector<float>& targetHost) {
			vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
			vkResetFences(m_device, 1, &m_fence);

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(m_cmdBuffer, &beginInfo);

			VkBufferCopy copyRegion = { 0, 0, fieldBytes };
			vkCmdCopyBuffer(m_cmdBuffer, srcBuf.buffer, m_bufFieldStaging.buffer, 1, &copyRegion);

			vkEndCommandBuffer(m_cmdBuffer);

			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &m_cmdBuffer;
			vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_fence);
			vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

			std::memcpy(targetHost.data(), m_bufFieldStaging.mapped, fieldBytes);
		};

		downloadBuffer(m_bufVolt, m_hostVolt);
		downloadBuffer(m_bufCurr, m_hostCurr);
		m_hostFieldsDirty = false;

		Engine* cpuEng = (m_op ? m_op->GetEngine() : nullptr);
		if (cpuEng)
		{
			size_t idx = 0;
			for (unsigned int n = 0; n < 3; ++n)
			{
				for (unsigned int x = 0; x < m_grid.dimX; ++x)
				{
					for (unsigned int y = 0; y < m_grid.dimY; ++y)
					{
						for (unsigned int z = 0; z < m_grid.dimZ; ++z)
						{
							cpuEng->SetVolt(n, x, y, z, m_hostVolt[idx]);
							cpuEng->SetCurr(n, x, y, z, m_hostCurr[idx]);
							++idx;
						}
					}
				}
			}
		}

		m_hostFieldsValid = true;
		return true;
	}
#endif

	m_hostFieldsValid = true;
	return true;
}

unsigned int EngineVulkan::GetNumberOfTimesteps() const
{
	return m_numTS;
}

FDTD_FLOAT EngineVulkan::GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	if (!m_hostFieldsValid)
	{
		const_cast<EngineVulkan*>(this)->SyncFieldsToHost();
	}
	size_t idx = n * static_cast<size_t>(m_grid.numCells) +
	             x * (m_grid.dimY * m_grid.dimZ) +
	             y * m_grid.dimZ +
	             z;
	return (idx < m_hostVolt.size()) ? m_hostVolt[idx] : 0.0f;
}

FDTD_FLOAT EngineVulkan::GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	if (!m_hostFieldsValid)
	{
		const_cast<EngineVulkan*>(this)->SyncFieldsToHost();
	}
	size_t idx = n * static_cast<size_t>(m_grid.numCells) +
	             x * (m_grid.dimY * m_grid.dimZ) +
	             y * m_grid.dimZ +
	             z;
	return (idx < m_hostCurr.size()) ? m_hostCurr[idx] : 0.0f;
}

void EngineVulkan::SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
{
	if (!m_hostFieldsValid && !SyncFieldsToHost())
		return;
	size_t idx = n * static_cast<size_t>(m_grid.numCells) +
	             x * (m_grid.dimY * m_grid.dimZ) +
	             y * m_grid.dimZ +
	             z;
	if (idx < m_hostVolt.size())
	{
		m_hostVolt[idx] = val;
		m_hostFieldsDirty = true;
	}
}

void EngineVulkan::SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
{
	if (!m_hostFieldsValid && !SyncFieldsToHost())
		return;
	size_t idx = n * static_cast<size_t>(m_grid.numCells) +
	             x * (m_grid.dimY * m_grid.dimZ) +
	             y * m_grid.dimZ +
	             z;
	if (idx < m_hostCurr.size())
	{
		m_hostCurr[idx] = val;
		m_hostFieldsDirty = true;
	}
}

void EngineVulkan::Reset()
{
#ifdef ENABLE_VULKAN
	if (m_device != VK_NULL_HANDLE)
	{
		vkDeviceWaitIdle(m_device);

		DestroyBuffer(m_bufVv);
		DestroyBuffer(m_bufVi);
		DestroyBuffer(m_bufIi);
		DestroyBuffer(m_bufIv);
		DestroyBuffer(m_bufVolt);
		DestroyBuffer(m_bufCurr);
		DestroyBuffer(m_bufFieldStaging);

		DestroyBuffer(m_bufVoltExcPoints);
		DestroyBuffer(m_bufCurrExcPoints);
		DestroyBuffer(m_bufProbePoints);
		DestroyBuffer(m_bufProbeValues);

		DestroyBuffer(m_bufUpmlIndices);
		DestroyBuffer(m_bufUpmlVoltCoeffs);
		DestroyBuffer(m_bufUpmlCurrCoeffs);
		DestroyBuffer(m_bufUpmlVoltFlux);
		DestroyBuffer(m_bufUpmlCurrFlux);

		DestroyBuffer(m_bufMurParams);
		DestroyBuffer(m_bufMurStore);

		DestroyBuffer(m_bufTfsfVoltParams);
		DestroyBuffer(m_bufTfsfCurrParams);
		DestroyBuffer(m_bufTfsfCurrSignal);
		DestroyBuffer(m_bufTfsfVoltSignal);

		DestroyBuffer(m_bufRlcParams);
		DestroyBuffer(m_bufRlcState);

		DestroyBuffer(m_bufAbcVoltParams);
		DestroyBuffer(m_bufAbcVoltStore);
		DestroyBuffer(m_bufAbcCurrParams);
		DestroyBuffer(m_bufAbcCurrStore);

		DestroyBuffer(m_bufDispVoltParams);
		DestroyBuffer(m_bufDispVoltState);
		DestroyBuffer(m_bufDispCurrParams);
		DestroyBuffer(m_bufDispCurrState);

		DestroyBuffer(m_bufCylR0);

		if (m_pipelineVolt != VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipelineVolt, nullptr); m_pipelineVolt = VK_NULL_HANDLE; }
		if (m_pipelineCurr != VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipelineCurr, nullptr); m_pipelineCurr = VK_NULL_HANDLE; }
		if (m_pipelineExc  != VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipelineExc, nullptr);  m_pipelineExc  = VK_NULL_HANDLE; }
		if (m_pipelineProbe!= VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipelineProbe, nullptr);m_pipelineProbe= VK_NULL_HANDLE; }
		if (m_pipelineUpmlPre != VK_NULL_HANDLE)  { vkDestroyPipeline(m_device, m_pipelineUpmlPre, nullptr);  m_pipelineUpmlPre  = VK_NULL_HANDLE; }
		if (m_pipelineUpmlPost != VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipelineUpmlPost, nullptr); m_pipelineUpmlPost = VK_NULL_HANDLE; }
		if (m_pipelineMurPre != VK_NULL_HANDLE)   { vkDestroyPipeline(m_device, m_pipelineMurPre, nullptr);   m_pipelineMurPre   = VK_NULL_HANDLE; }
		if (m_pipelineMurPost != VK_NULL_HANDLE)  { vkDestroyPipeline(m_device, m_pipelineMurPost, nullptr);  m_pipelineMurPost  = VK_NULL_HANDLE; }
		if (m_pipelineMurApply != VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipelineMurApply, nullptr); m_pipelineMurApply = VK_NULL_HANDLE; }
		if (m_pipelineTfsf != VK_NULL_HANDLE)     { vkDestroyPipeline(m_device, m_pipelineTfsf, nullptr);     m_pipelineTfsf     = VK_NULL_HANDLE; }
		if (m_pipelineRlc != VK_NULL_HANDLE)      { vkDestroyPipeline(m_device, m_pipelineRlc, nullptr);      m_pipelineRlc      = VK_NULL_HANDLE; }
		if (m_pipelineAbcVolt != VK_NULL_HANDLE)  { vkDestroyPipeline(m_device, m_pipelineAbcVolt, nullptr);  m_pipelineAbcVolt  = VK_NULL_HANDLE; }
		if (m_pipelineAbcCurr != VK_NULL_HANDLE)  { vkDestroyPipeline(m_device, m_pipelineAbcCurr, nullptr);  m_pipelineAbcCurr  = VK_NULL_HANDLE; }
		if (m_pipelineDisp != VK_NULL_HANDLE)     { vkDestroyPipeline(m_device, m_pipelineDisp, nullptr);     m_pipelineDisp     = VK_NULL_HANDLE; }
		if (m_pipelineCyl != VK_NULL_HANDLE)      { vkDestroyPipeline(m_device, m_pipelineCyl, nullptr);      m_pipelineCyl      = VK_NULL_HANDLE; }

		if (m_pipelineLayoutFields != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutFields, nullptr); m_pipelineLayoutFields = VK_NULL_HANDLE; }
		if (m_pipelineLayoutExc    != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutExc, nullptr);    m_pipelineLayoutExc    = VK_NULL_HANDLE; }
		if (m_pipelineLayoutProbe  != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutProbe, nullptr);  m_pipelineLayoutProbe  = VK_NULL_HANDLE; }
		if (m_pipelineLayoutUpml   != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutUpml, nullptr);   m_pipelineLayoutUpml   = VK_NULL_HANDLE; }
		if (m_pipelineLayoutMur    != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutMur, nullptr);    m_pipelineLayoutMur    = VK_NULL_HANDLE; }
		if (m_pipelineLayoutTfsf   != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutTfsf, nullptr);   m_pipelineLayoutTfsf   = VK_NULL_HANDLE; }
		if (m_pipelineLayoutRlc    != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutRlc, nullptr);    m_pipelineLayoutRlc    = VK_NULL_HANDLE; }
		if (m_pipelineLayoutAbc    != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutAbc, nullptr);    m_pipelineLayoutAbc    = VK_NULL_HANDLE; }
		if (m_pipelineLayoutDisp   != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutDisp, nullptr);   m_pipelineLayoutDisp   = VK_NULL_HANDLE; }
		if (m_pipelineLayoutCyl    != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutCyl, nullptr);    m_pipelineLayoutCyl    = VK_NULL_HANDLE; }

		if (m_descLayoutFields != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutFields, nullptr); m_descLayoutFields = VK_NULL_HANDLE; }
		if (m_descLayoutExc    != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutExc, nullptr);    m_descLayoutExc    = VK_NULL_HANDLE; }
		if (m_descLayoutProbe  != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutProbe, nullptr);  m_descLayoutProbe  = VK_NULL_HANDLE; }
		if (m_descLayoutUpml   != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutUpml, nullptr);   m_descLayoutUpml   = VK_NULL_HANDLE; }
		if (m_descLayoutMur    != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutMur, nullptr);    m_descLayoutMur    = VK_NULL_HANDLE; }
		if (m_descLayoutTfsf   != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutTfsf, nullptr);   m_descLayoutTfsf   = VK_NULL_HANDLE; }
		if (m_descLayoutRlc    != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutRlc, nullptr);    m_descLayoutRlc    = VK_NULL_HANDLE; }
		if (m_descLayoutAbc    != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutAbc, nullptr);    m_descLayoutAbc    = VK_NULL_HANDLE; }
		if (m_descLayoutDisp   != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutDisp, nullptr);   m_descLayoutDisp   = VK_NULL_HANDLE; }
		if (m_descLayoutCyl    != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutCyl, nullptr);    m_descLayoutCyl    = VK_NULL_HANDLE; }

		if (m_descPool != VK_NULL_HANDLE) { vkDestroyDescriptorPool(m_device, m_descPool, nullptr); m_descPool = VK_NULL_HANDLE; }
		if (m_fence != VK_NULL_HANDLE)    { vkDestroyFence(m_device, m_fence, nullptr);              m_fence = VK_NULL_HANDLE; }
		if (m_cmdPool != VK_NULL_HANDLE)  { vkDestroyCommandPool(m_device, m_cmdPool, nullptr);      m_cmdPool = VK_NULL_HANDLE; }

		vkDestroyDevice(m_device, nullptr);
		m_device = VK_NULL_HANDLE;
	}

	if (m_instance != VK_NULL_HANDLE)
	{
		vkDestroyInstance(m_instance, nullptr);
		m_instance = VK_NULL_HANDLE;
	}
#endif

	m_numTS = 0;
	m_numUpmlCells = 0;
	m_totalMurPoints = 0;
	m_murFaces.clear();
	m_tfsfVoltFaces.clear();
	m_tfsfVoltPoints.clear();
	m_tfsfCurrFaces.clear();
	m_tfsfCurrPoints.clear();
	m_tfsfSigLength = 0;
	m_tfsfPeriod = 0;
	m_rlcCount = 0;
	m_abcVoltCount = 0;
	m_abcCurrCount = 0;
	m_abcVoltSheets.clear();
	m_abcCurrSheets.clear();
	m_dispVoltCount = 0;
	m_dispCurrCount = 0;
	m_dispVoltPasses.clear();
	m_dispCurrPasses.clear();
	m_descSetCyl = VK_NULL_HANDLE;
	m_hasCylinder = false;
	m_cylClosedAlpha = false;
	m_cylR0Included = false;
	m_cylLastALine = 0;
	m_hostFieldsValid = true;
	m_hostFieldsDirty = true;
	std::fill(m_hostVolt.begin(), m_hostVolt.end(), 0.0f);
	std::fill(m_hostCurr.begin(), m_hostCurr.end(), 0.0f);
	m_excSources.clear();
	m_voltExcPoints.clear();
	m_currExcPoints.clear();
	m_probePoints.clear();
}

std::string EngineVulkan::GetBackendName() const
{
	return "Vulkan (" + m_deviceName + ")";
}
