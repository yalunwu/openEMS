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
	// Compile GLSL compute shaders to SPIR-V bytecode at runtime
	auto spvVolt = CompileGLSLToSpirv(VulkanShaders::kShaderVoltageUpdate, shaderc_glsl_compute_shader, "yee_voltage.comp");
	auto spvCurr = CompileGLSLToSpirv(VulkanShaders::kShaderCurrentUpdate, shaderc_glsl_compute_shader, "yee_current.comp");
	auto spvExc  = CompileGLSLToSpirv(VulkanShaders::kShaderExcitation, shaderc_glsl_compute_shader, "excitation.comp");
	auto spvPrb  = CompileGLSLToSpirv(VulkanShaders::kShaderProbeGather, shaderc_glsl_compute_shader, "probe_gather.comp");

	if (spvVolt.empty() || spvCurr.empty() || spvExc.empty() || spvPrb.empty())
	{
		return false;
	}

	VkShaderModule modVolt = CreateShaderModule(spvVolt);
	VkShaderModule modCurr = CreateShaderModule(spvCurr);
	VkShaderModule modExc  = CreateShaderModule(spvExc);
	VkShaderModule modPrb  = CreateShaderModule(spvPrb);

	// Create descriptor pool
	std::vector<VkDescriptorPoolSize> poolSizes = {
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32 }
	};
	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = 16;
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

	// Release shader modules after pipeline creation
	vkDestroyShaderModule(m_device, modVolt, nullptr);
	vkDestroyShaderModule(m_device, modCurr, nullptr);
	vkDestroyShaderModule(m_device, modExc, nullptr);
	vkDestroyShaderModule(m_device, modPrb, nullptr);

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

			// 1. Voltage update
			vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineVolt);
			vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutFields, 0, 1, &m_descSetFields, 0, nullptr);
			vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutFields, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			vkCmdDispatch(m_cmdBuffer, wgZ, wgY, wgX);

			// 2. Voltage excitation pass
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

			// 3. Current update
			vkCmdBindPipeline(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineCurr);
			vkCmdBindDescriptorSets(m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutFields, 0, 1, &m_descSetFields, 0, nullptr);
			vkCmdPushConstants(m_cmdBuffer, m_pipelineLayoutFields, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			vkCmdDispatch(m_cmdBuffer, wgZ, wgY, wgX);

			// 4. Current excitation pass
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

		if (m_pipelineVolt != VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipelineVolt, nullptr); m_pipelineVolt = VK_NULL_HANDLE; }
		if (m_pipelineCurr != VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipelineCurr, nullptr); m_pipelineCurr = VK_NULL_HANDLE; }
		if (m_pipelineExc  != VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipelineExc, nullptr);  m_pipelineExc  = VK_NULL_HANDLE; }
		if (m_pipelineProbe!= VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipelineProbe, nullptr);m_pipelineProbe= VK_NULL_HANDLE; }

		if (m_pipelineLayoutFields != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutFields, nullptr); m_pipelineLayoutFields = VK_NULL_HANDLE; }
		if (m_pipelineLayoutExc    != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutExc, nullptr);    m_pipelineLayoutExc    = VK_NULL_HANDLE; }
		if (m_pipelineLayoutProbe  != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_pipelineLayoutProbe, nullptr);  m_pipelineLayoutProbe  = VK_NULL_HANDLE; }

		if (m_descLayoutFields != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutFields, nullptr); m_descLayoutFields = VK_NULL_HANDLE; }
		if (m_descLayoutExc    != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutExc, nullptr);    m_descLayoutExc    = VK_NULL_HANDLE; }
		if (m_descLayoutProbe  != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_descLayoutProbe, nullptr);  m_descLayoutProbe  = VK_NULL_HANDLE; }

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
