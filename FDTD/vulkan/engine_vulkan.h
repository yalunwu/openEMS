#ifndef ENGINE_VULKAN_H
#define ENGINE_VULKAN_H

#include "FDTD/engine_backend.h"
#include <vector>
#include <string>
#include <memory>

#ifdef ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

class Excitation;

class OPENEMS_EXPORT EngineVulkan : public EngineBackend
{
public:
	explicit EngineVulkan(const Operator* op);
	~EngineVulkan() override;

	bool Initialize() override;
	void Reset() override;
	bool IterateTS(unsigned int iterTS) override;
	unsigned int GetNumberOfTimesteps() const override;

	FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
	FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
	void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val) override;
	void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val) override;

	bool SyncFieldsToHost() override;
	bool SyncProbesToHost() override;
	void RegisterProbes(const ProcessingArray* pa) override;
	std::string GetBackendName() const override;

	static bool CheckModelSupport(const Operator* op, const ContinuousStructure* csx, std::string& unsupportedReason);

private:
	struct GridDimensions {
		uint32_t dimX = 0;
		uint32_t dimY = 0;
		uint32_t dimZ = 0;
		uint32_t numCells = 0;
	};

	inline size_t GetLinearIndex(unsigned int ny, unsigned int x, unsigned int y, unsigned int z) const
	{
		return static_cast<size_t>(ny) * m_grid.numCells +
		       static_cast<size_t>(x) * (m_grid.dimY * m_grid.dimZ) +
		       static_cast<size_t>(y) * m_grid.dimZ +
		       static_cast<size_t>(z);
	}

	struct GpuExcPoint {
		uint32_t index;
		float value;
	};

	struct ExcSourceData {
		const Excitation* exc = nullptr;
		std::vector<uint32_t> voltIndices;
		std::vector<float> voltAmps;
		std::vector<float> voltDelays;
		uint32_t voltCount = 0;

		std::vector<uint32_t> currIndices;
		std::vector<float> currAmps;
		std::vector<float> currDelays;
		uint32_t currCount = 0;
	};

	struct ProbePoint {
		uint32_t linear_index;
		uint32_t field_type; // 0 = volt, 1 = curr
	};

	const Operator* m_op = nullptr;
	const ProcessingArray* m_pa = nullptr;
	GridDimensions m_grid;
	unsigned int m_numTS = 0;
	bool m_hostFieldsValid = true;

	std::vector<float> m_hostVolt;
	std::vector<float> m_hostCurr;
	std::string m_deviceName = "Vulkan GPU";

	std::vector<ExcSourceData> m_excSources;
	std::vector<GpuExcPoint> m_voltExcPoints;
	std::vector<GpuExcPoint> m_currExcPoints;
	std::vector<ProbePoint> m_probePoints;

#ifdef ENABLE_VULKAN
	struct VulkanBuffer {
		VkBuffer buffer = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		VkDeviceSize size = 0;
		void* mapped = nullptr;
	};

	VkInstance m_instance = VK_NULL_HANDLE;
	VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
	VkDevice m_device = VK_NULL_HANDLE;
	VkQueue m_computeQueue = VK_NULL_HANDLE;
	uint32_t m_computeQueueFamily = 0;

	VkCommandPool m_cmdPool = VK_NULL_HANDLE;
	VkCommandBuffer m_cmdBuffer = VK_NULL_HANDLE;
	VkFence m_fence = VK_NULL_HANDLE;

	// Material and Field Buffers
	VulkanBuffer m_bufVv;
	VulkanBuffer m_bufVi;
	VulkanBuffer m_bufIi;
	VulkanBuffer m_bufIv;
	VulkanBuffer m_bufVolt;
	VulkanBuffer m_bufCurr;
	VulkanBuffer m_bufFieldStaging;

	// Excitation Buffers
	VulkanBuffer m_bufVoltExcPoints;
	VulkanBuffer m_bufCurrExcPoints;

	// Probe Gather Buffers
	VulkanBuffer m_bufProbePoints;
	VulkanBuffer m_bufProbeValues; // Host visible

	// Descriptors and Pipelines
	VkDescriptorPool m_descPool = VK_NULL_HANDLE;

	VkDescriptorSetLayout m_descLayoutFields = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutFields = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetFields = VK_NULL_HANDLE;
	VkPipeline m_pipelineVolt = VK_NULL_HANDLE;
	VkPipeline m_pipelineCurr = VK_NULL_HANDLE;

	VkDescriptorSetLayout m_descLayoutExc = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutExc = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetVoltExc = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetCurrExc = VK_NULL_HANDLE;
	VkPipeline m_pipelineExc = VK_NULL_HANDLE;

	VkDescriptorSetLayout m_descLayoutProbe = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutProbe = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetProbe = VK_NULL_HANDLE;
	VkPipeline m_pipelineProbe = VK_NULL_HANDLE;

	bool InitVulkan();
	bool AllocateBuffers();
	bool AllocateExcitationBuffers();
	bool AllocateProbeBuffers();
	bool CreatePipelines();

	bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VulkanBuffer& outBuf);
	void DestroyBuffer(VulkanBuffer& buf);
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	VkShaderModule CreateShaderModule(const std::vector<uint32_t>& spirv);

	bool PrepareVoltExcitation(unsigned int step);
	bool PrepareCurrExcitation(unsigned int step);
#endif
};

#endif // ENGINE_VULKAN_H
