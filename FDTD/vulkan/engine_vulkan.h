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
class Operator_Ext_Absorbing_BC;

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
	bool m_hostFieldsDirty = true;

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

	// UPML Buffers
	uint32_t m_numUpmlCells = 0;
	VulkanBuffer m_bufUpmlIndices;
	VulkanBuffer m_bufUpmlVoltCoeffs;
	VulkanBuffer m_bufUpmlCurrCoeffs;
	VulkanBuffer m_bufUpmlVoltFlux;
	VulkanBuffer m_bufUpmlCurrFlux;

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

	VkDescriptorSetLayout m_descLayoutUpml = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutUpml = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetUpmlVolt = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetUpmlCurr = VK_NULL_HANDLE;
	VkPipeline m_pipelineUpmlPre = VK_NULL_HANDLE;
	VkPipeline m_pipelineUpmlPost = VK_NULL_HANDLE;

	// Mur ABC Buffers & Pipelines
	struct MurFace {
		uint32_t offset;
		uint32_t count;
		uint32_t start_TS;
	};

	struct GpuMurPoint {
		uint32_t pos_idx;
		uint32_t shift_idx;
		float coeff;
		uint32_t start_TS;
	};

	uint32_t m_totalMurPoints = 0;
	std::vector<MurFace> m_murFaces;
	VulkanBuffer m_bufMurParams;
	VulkanBuffer m_bufMurStore;

	VkDescriptorSetLayout m_descLayoutMur = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutMur = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetMur = VK_NULL_HANDLE;
	VkPipeline m_pipelineMurPre = VK_NULL_HANDLE;
	VkPipeline m_pipelineMurPost = VK_NULL_HANDLE;
	VkPipeline m_pipelineMurApply = VK_NULL_HANDLE;

	// TFSF Buffers & Pipelines
	struct TfsfFace {
		uint32_t offset;
		uint32_t count;
	};

	struct GpuTfsfPoint {
		uint32_t pos_idx;
		uint32_t delay;
		float w0;
		float w1;
	};

	uint32_t m_tfsfSigLength = 0;
	int32_t m_tfsfPeriod = 0;

	std::vector<TfsfFace> m_tfsfVoltFaces;
	std::vector<GpuTfsfPoint> m_tfsfVoltPoints;
	std::vector<TfsfFace> m_tfsfCurrFaces;
	std::vector<GpuTfsfPoint> m_tfsfCurrPoints;

	VulkanBuffer m_bufTfsfVoltParams;
	VulkanBuffer m_bufTfsfCurrParams;
	VulkanBuffer m_bufTfsfCurrSignal;
	VulkanBuffer m_bufTfsfVoltSignal;

	VkDescriptorSetLayout m_descLayoutTfsf = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutTfsf = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetTfsfVolt = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetTfsfCurr = VK_NULL_HANDLE;
	VkPipeline m_pipelineTfsf = VK_NULL_HANDLE;

	// Lumped RLC Buffers & Pipeline
	struct GpuRlcParam {
		uint32_t field_index;
		float ilv_i2v;
		float vvd;
		float vv2;
		float vj1;
		float vj2;
		float ib0;
		float b1_ib0;
		float b2_ib0;
		float pad0;
		float pad1;
		float pad2;
	};

	struct GpuRlcState {
		float v_Il;
		float vd0;
		float vd1;
		float vd2;
		float j0;
		float j1;
		float j2;
		float pad;
	};

	uint32_t m_rlcCount = 0;
	VulkanBuffer m_bufRlcParams;
	VulkanBuffer m_bufRlcState;

	VkDescriptorSetLayout m_descLayoutRlc = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutRlc = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetRlc = VK_NULL_HANDLE;
	VkPipeline m_pipelineRlc = VK_NULL_HANDLE;

	// Local Absorbing Boundary Sheets
	struct GpuAbcVoltPoint {
		uint32_t pos_idx;
		uint32_t shift_idx;
		float k1;
		uint32_t pad;
	};

	struct GpuAbcCurrPoint {
		uint32_t pos_idx;
		uint32_t shift_idx;
		float k1;
		float k2;
	};

	uint32_t m_abcVoltCount = 0;
	uint32_t m_abcCurrCount = 0;
	std::vector<TfsfFace> m_abcVoltSheets;
	std::vector<TfsfFace> m_abcCurrSheets;
	VulkanBuffer m_bufAbcVoltParams;
	VulkanBuffer m_bufAbcVoltStore;
	VulkanBuffer m_bufAbcCurrParams;
	VulkanBuffer m_bufAbcCurrStore;

	VkDescriptorSetLayout m_descLayoutAbc = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutAbc = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetAbcVolt = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetAbcCurr = VK_NULL_HANDLE;
	VkPipeline m_pipelineAbcVolt = VK_NULL_HANDLE;
	VkPipeline m_pipelineAbcCurr = VK_NULL_HANDLE;

	// Dispersive Materials (Lorentz / Debye / Drude / Conducting Sheets)
	struct alignas(16) GpuDispersivePoint {
		uint32_t pos_idx;
		float int_coeff;
		float ext_coeff;
		float lor_coeff;
	};

	struct GpuDispersiveState {
		float ade_val;
		float lor_val;
	};

	uint32_t m_dispVoltCount = 0;
	uint32_t m_dispCurrCount = 0;
	std::vector<TfsfFace> m_dispVoltPasses;
	std::vector<TfsfFace> m_dispCurrPasses;
	VulkanBuffer m_bufDispVoltParams;
	VulkanBuffer m_bufDispVoltState;
	VulkanBuffer m_bufDispCurrParams;
	VulkanBuffer m_bufDispCurrState;

	VkDescriptorSetLayout m_descLayoutDisp = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutDisp = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetDispVolt = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetDispCurr = VK_NULL_HANDLE;
	VkPipeline m_pipelineDisp = VK_NULL_HANDLE;

	// Cylindrical Coordinates Extension
	bool m_hasCylinder = false;
	bool m_cylClosedAlpha = false;
	bool m_cylR0Included = false;
	uint32_t m_cylLastALine = 0;
	VulkanBuffer m_bufCylR0;
	VkDescriptorSetLayout m_descLayoutCyl = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutCyl = VK_NULL_HANDLE;
	VkDescriptorSet m_descSetCyl = VK_NULL_HANDLE;
	VkPipeline m_pipelineCyl = VK_NULL_HANDLE;

	bool InitVulkan();
	bool AllocateBuffers();
	bool AllocateExcitationBuffers();
	bool AllocateProbeBuffers();
	bool AllocateUpmlBuffers();
	bool AllocateMurBuffers();
	bool AllocateTfsfBuffers();
	bool AllocateRlcBuffers();
	bool AllocateAbsorbingBCBuffers();
	bool AllocateDispersiveBuffers();
	bool AllocateCylinderBuffers();
	bool CreatePipelines();
	bool SyncFieldsToDevice();

	bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VulkanBuffer& outBuf);
	void DestroyBuffer(VulkanBuffer& buf);
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	VkShaderModule CreateShaderModule(const std::vector<uint32_t>& spirv);

	bool PrepareVoltExcitation(unsigned int step);
	bool PrepareCurrExcitation(unsigned int step);
#endif
};

#endif // ENGINE_VULKAN_H
