#ifndef ENGINE_VULKAN_H
#define ENGINE_VULKAN_H

#include "FDTD/engine_backend.h"
#include <vector>
#include <string>
#include <memory>
#include <array>
#include <cstdint>
#include <iosfwd>
#include <utility>

#ifdef ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

class Excitation;
class Operator_Ext_Absorbing_BC;
class Operator_CylinderMultiGrid;

class OPENEMS_EXPORT EngineVulkan : public EngineBackend
{
public:
	explicit EngineVulkan(const Operator* op);
	~EngineVulkan() override;

	bool Initialize() override;
	void Reset() override;
	bool IterateTS(unsigned int iterTS) override;
	// Record future probe frames, then advance CPU processing with IterateTS().
	// Full-field and energy access require replay to reach the batch end.
	bool BeginProbeHistory(unsigned int steps);
	unsigned int GetPendingProbeHistorySteps() const;
	unsigned int GetNumberOfTimesteps() const override;

	FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
	FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
	void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val) override;
	void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val) override;

	bool SyncFieldsToHost() override;
	bool SyncProbesToHost() override;
	bool GetFastEnergy(double& energy) override;
	bool SupportsFastEnergy() const;
	bool SetReadbackOptimizationsEnabled(bool enabled); // benchmark/reference switch
	bool SetEnergyFloat64Enabled(bool enabled); // before Initialize(); exercise FP32 fallback
	void RegisterProbes(const ProcessingArray* pa) override;
	std::string GetBackendName() const override;

	static bool CheckModelSupport(const Operator* op, const ContinuousStructure* csx, std::string& unsupportedReason);

	enum GpuProfileCategory { ProfileBatch, ProfileVoltage, ProfileCurrent,
	                          ProfileExtension, ProfileMultigrid, ProfileProbe,
	                          ProfileReadback, ProfileEnergy, ProfileCategoryCount };
	struct ProfileStatistics {
		uint64_t submissions = 0, dispatches = 0, timesteps = 0, sampledSteps = 0;
		uint64_t uploadedBytes = 0, downloadedBytes = 0, probeBytes = 0;
		uint64_t energyBytes = 0;
		double recordSeconds = 0, submitSeconds = 0, waitSeconds = 0;
		double readbackSeconds = 0, mirrorSeconds = 0;
		std::array<double, ProfileCategoryCount> gpuSeconds = {};
		std::array<uint64_t, ProfileCategoryCount> gpuSamples = {};
	};
	bool SetBatchSize(unsigned int size); // 1..64; does not change processing intervals
	unsigned int GetBatchSize() const { return m_batchSize; }
	bool SetProfilingEnabled(bool enabled);
	bool IsProfilingEnabled() const { return m_profileEnabled; }
	bool Synchronize(); // wait for stepping without downloading fields
	bool ClearProfile();
	ProfileStatistics GetProfile(); // collects only available timestamp results
	void WriteProfile(std::ostream& stream);
	std::string GetDeviceDescription() const;
	bool SetCoefficientMode(const std::string& mode); // dense, palette, analyze; before Initialize()
	struct CoefficientStatistics {
		uint64_t nodes = 0, uniqueNodes = 0, denseBytes = 0, storageBytes = 0;
		uint64_t allocatedBytes = 0;
		uint64_t uniqueComponents = 0, componentBytes = 0;
		bool palette = false, complete = false;
	};
	CoefficientStatistics GetCoefficientStatistics(unsigned int level = 0) const {
		if (level == 0) return m_level->m_coefficients;
		return m_level->m_innerGrid ? m_level->m_innerGrid->GetCoefficientStatistics(level - 1) : CoefficientStatistics();
	}

private:
	friend bool Test_Vulkan_OptionalResources();
	friend bool Test_Vulkan_ProbeAllocationFailure();
	friend bool Test_Vulkan_ExtensionIndexValidation();
	friend bool Test_Vulkan_ProbeHistory();

	struct GridDimensions {
		uint32_t dimX = 0;
		uint32_t dimY = 0;
		uint32_t dimZ = 0;
		uint32_t numCells = 0;
	};

	inline size_t GetLinearIndex(unsigned int ny, unsigned int x, unsigned int y, unsigned int z) const
	{
		if (ny >= 3u || x >= m_level->m_grid.dimX || y >= m_level->m_grid.dimY || z >= m_level->m_grid.dimZ)
			return static_cast<size_t>(-1);
		return static_cast<size_t>(ny) * m_level->m_grid.numCells +
		       static_cast<size_t>(x) * m_level->m_grid.dimY * m_level->m_grid.dimZ +
		       static_cast<size_t>(y) * m_level->m_grid.dimZ +
		       static_cast<size_t>(z);
	}

	struct GpuExcPoint {
		uint32_t index;
		float amplitude;
		uint32_t delay;
		uint32_t signalOffset;
		uint32_t signalLength;
		uint32_t period;
	};

	struct ProbePoint {
		uint32_t linear_index;
		uint32_t field_type; // 0 = volt, 1 = curr
	};
	std::string m_deviceName = "Vulkan GPU";
	unsigned int m_batchSize = 32;
	bool m_profileEnabled = false;
	bool m_readbackOptimizations = true;
	bool m_enableEnergyFloat64 = true;
	std::string m_coefficientMode = "dense";
	unsigned int m_probeHistoryStart = 0, m_probeHistoryCount = 0, m_probeHistoryCursor = 0;
	ProfileStatistics m_profile;
	bool IterateTSImpl(unsigned int iterTS, bool history);
#ifdef ENABLE_VULKAN
	struct ProfileSpan { uint32_t first; GpuProfileCategory category; };
	VkQueryPool m_queryPool = VK_NULL_HANDLE;
	std::array<ProfileSpan, 256> m_profileSpans;
	uint32_t m_profileSpanCount = 0, m_queryCount = 0, m_timestampBits = 0;
	float m_timestampPeriod = 0;
	bool m_queriesPending = false, m_sampleDispatches = false, m_runtimeFailed = false;
	EngineVulkan& ProfileOwner();
	bool InitProfiling();
	void CollectProfile();
	void BeginProfileCommands(VkCommandBuffer cmd);
	uint32_t BeginGpuSpan(VkCommandBuffer cmd, GpuProfileCategory category);
	void EndGpuSpan(VkCommandBuffer cmd, uint32_t first);
	void Dispatch(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z,
	              GpuProfileCategory category = ProfileExtension);
	void CopyBuffer(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst,
	                const VkBufferCopy& region, bool download = false);
	bool WaitForFence();
	bool SubmitCommandBuffer();
	struct VulkanBuffer {
		VulkanBuffer() = default;
		VulkanBuffer(const VulkanBuffer&) = delete;
		VulkanBuffer& operator=(const VulkanBuffer&) = delete;
		~VulkanBuffer() { Release(); }
		void Swap(VulkanBuffer& other)
		{
			std::swap(device, other.device);
			std::swap(buffer, other.buffer);
			std::swap(memory, other.memory);
			std::swap(size, other.size);
			std::swap(memoryProperties, other.memoryProperties);
			std::swap(mapped, other.mapped);
		}
		void Release()
		{
			if (device)
			{
				if (mapped) vkUnmapMemory(device, memory);
				if (buffer) vkDestroyBuffer(device, buffer, nullptr);
				if (memory) vkFreeMemory(device, memory, nullptr);
			}
			device = VK_NULL_HANDLE;
			buffer = VK_NULL_HANDLE;
			memory = VK_NULL_HANDLE;
			mapped = nullptr;
			size = 0;
			memoryProperties = 0;
		}
		VkDevice device = VK_NULL_HANDLE;
		VkBuffer buffer = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		VkDeviceSize size = 0;
		VkMemoryPropertyFlags memoryProperties = 0;
		void* mapped = nullptr;
	};

	VkInstance m_instance = VK_NULL_HANDLE;
	VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
	VkDevice m_device = VK_NULL_HANDLE;
	VkQueue m_computeQueue = VK_NULL_HANDLE;
	uint32_t m_computeQueueFamily = 0;
	bool m_ownsVulkanDevice = true;
	EngineVulkan* m_deviceOwner = nullptr;
	// Shared descriptors and pipelines; level storage is owned by LevelState.
	VkDescriptorPool m_descPool = VK_NULL_HANDLE;

	VkDescriptorSetLayout m_descLayoutFields = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutFields = VK_NULL_HANDLE;
	VkPipeline m_pipelineVolt = VK_NULL_HANDLE;
	VkPipeline m_pipelineCurr = VK_NULL_HANDLE;
	VkPipeline m_pipelineVoltPalette = VK_NULL_HANDLE;
	VkPipeline m_pipelineCurrPalette = VK_NULL_HANDLE;
	bool CreatePalettePipelines();

	VkDescriptorSetLayout m_descLayoutExc = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutExc = VK_NULL_HANDLE;
	VkPipeline m_pipelineExc = VK_NULL_HANDLE;

	VkDescriptorSetLayout m_descLayoutProbe = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutProbe = VK_NULL_HANDLE;
	VkPipeline m_pipelineProbe = VK_NULL_HANDLE;
	bool m_energyFloat64 = false;
	bool AllocateEnergyResources();
	void DestroyEnergyResources();
	void RecordProbeGather(VkCommandBuffer cmd, unsigned int slot = 0);

	VkDescriptorSetLayout m_descLayoutUpml = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutUpml = VK_NULL_HANDLE;
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
	VkDescriptorSetLayout m_descLayoutMur = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutMur = VK_NULL_HANDLE;
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
	VkDescriptorSetLayout m_descLayoutTfsf = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutTfsf = VK_NULL_HANDLE;
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
	VkDescriptorSetLayout m_descLayoutRlc = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutRlc = VK_NULL_HANDLE;
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
	VkDescriptorSetLayout m_descLayoutAbc = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutAbc = VK_NULL_HANDLE;
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
	VkDescriptorSetLayout m_descLayoutDisp = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutDisp = VK_NULL_HANDLE;
	VkPipeline m_pipelineDisp = VK_NULL_HANDLE;

	struct GpuDebyePoint {
		uint32_t pos_idx;
		uint32_t pole_offset;
		uint32_t pole_count;
		float solve_coeff;
	};
	struct GpuDebyePole {
		float relax_coeff;
		float drive_coeff;
	};
	VkDescriptorSetLayout m_descLayoutDebye = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutDebye = VK_NULL_HANDLE;
	VkPipeline m_pipelineDebye = VK_NULL_HANDLE;

	// Cylindrical Coordinates Extension
	VkDescriptorSetLayout m_descLayoutCyl = VK_NULL_HANDLE;
	VkPipelineLayout m_pipelineLayoutCyl = VK_NULL_HANDLE;
	VkPipeline m_pipelineCyl = VK_NULL_HANDLE;

	// Cylindrical multigrid interface owned by the parent level.
	struct alignas(16) GpuMultigridInterpolation {
		uint32_t posP[4];
		uint32_t posPP[4];
		float coeffP[4];
		float coeffPP[4];
	};

	bool InitVulkan();
	bool InitSharedVulkan(EngineVulkan& parent);
	bool InitCommandResources();
	bool InitializeLevel();
	bool AllocateBuffers();
	bool AllocateFieldStagingBuffer();
	bool AllocateExcitationBuffers();
	bool AllocateProbeBuffers(const std::vector<ProbePoint>& points, unsigned int capacity = 1);
	uint32_t GetCheckedLinearIndex(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const;
	void SetHierarchyHostTimestep(unsigned int ts);
	bool AllocateUpmlBuffers();
	bool AllocateMurBuffers();
	bool AllocateTfsfBuffers();
	bool AllocateRlcBuffers();
	bool AllocateAbsorbingBCBuffers();
	bool AllocateDispersiveBuffers();
	bool AllocateDebyeBuffers();
	void RecordDebyePhase(VkCommandBuffer cmd, uint32_t mode);
	bool AllocateCylinderBuffers();
	bool AllocateMultigridBuffers();
	bool CreatePipelines();
	bool AllocateFieldDescriptors();
	bool SyncFieldsToDevice();
	bool SyncHierarchyToDevice();

	bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VulkanBuffer& outBuf);
	bool UploadStorageBuffer(const void* data, VkDeviceSize size, VulkanBuffer& outBuf, bool* allocationFailed = nullptr);
	void DestroyBuffer(VulkanBuffer& buf);
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	VkShaderModule CreateShaderModule(const std::vector<uint32_t>& spirv);

	void RecordVoltagePhase(VkCommandBuffer cmd, bool hasVoltExc, unsigned int timestep);
	void RecordCurrentPhase(VkCommandBuffer cmd, bool hasCurrExc, unsigned int timestep);
	void RecordVoltageHierarchy(VkCommandBuffer cmd, unsigned int timestep);
	void RecordCurrentHierarchy(VkCommandBuffer cmd, unsigned int timestep);
	void RecordProjectionHierarchy(VkCommandBuffer cmd);
	void RecordMultigridTransfer(VkCommandBuffer cmd, uint32_t mode, uint32_t radialCount);
	void MarkHierarchyHostInvalid();
	void SetHierarchyTimestep(unsigned int ts);
	bool SyncLevelFieldsToHost();
#endif
	// Independent fields, operators, extensions, descriptors, and mirrors per grid.
	struct LevelState {
		const Operator* m_op = nullptr;
		const Operator_CylinderMultiGrid* m_multigridOp = nullptr;
		const ProcessingArray* m_pa = nullptr;
		GridDimensions m_grid;
		CoefficientStatistics m_coefficients;
		uint32_t m_activeXStart = 0;
		bool m_dimensionsValid = true;
		unsigned int m_numTS = 0;
		bool m_hostFieldsValid = true;
		bool m_hostFieldsDirty = true;
		bool m_probesValid = false, m_energyValid = false;
		double m_cachedEnergy = 0;
		std::vector<float> m_hostVolt;
		std::vector<float> m_hostCurr;
		std::vector<GpuExcPoint> m_voltExcPoints;
		std::vector<GpuExcPoint> m_currExcPoints;
		std::vector<ProbePoint> m_probePoints;
		unsigned int m_probeCapacity = 1;
		std::unique_ptr<EngineVulkan> m_innerGrid;
#ifdef ENABLE_VULKAN
		VkCommandPool m_cmdPool = VK_NULL_HANDLE;
		VkCommandBuffer m_cmdBuffer = VK_NULL_HANDLE;
		VkFence m_fence = VK_NULL_HANDLE;
		VulkanBuffer m_bufVv;
		VulkanBuffer m_bufVi;
		VulkanBuffer m_bufIi;
		VulkanBuffer m_bufIv;
		VulkanBuffer m_bufVolt;
		VulkanBuffer m_bufCurr;
		VulkanBuffer m_bufFieldStaging;
		VulkanBuffer m_bufVoltExcPoints;
		VulkanBuffer m_bufCurrExcPoints;
		VulkanBuffer m_bufExcSignals;
		VulkanBuffer m_bufProbePoints;
		VulkanBuffer m_bufProbeValues; // Host visible
		uint32_t m_numUpmlCells = 0;
		VulkanBuffer m_bufUpmlIndices;
		VulkanBuffer m_bufUpmlVoltCoeffs;
		VulkanBuffer m_bufUpmlCurrCoeffs;
		VulkanBuffer m_bufUpmlVoltFlux;
		VulkanBuffer m_bufUpmlCurrFlux;
		VkDescriptorSet m_descSetFields = VK_NULL_HANDLE;
		VkDescriptorSet m_descSetVoltExc = VK_NULL_HANDLE;
		VkDescriptorSet m_descSetCurrExc = VK_NULL_HANDLE;
		VkDescriptorSet m_descSetProbe = VK_NULL_HANDLE;
		VulkanBuffer m_bufEnergy;
		uint32_t m_energyGroups = 0;
		bool m_energyAvailable = false;
		VkDescriptorSetLayout m_descLayoutEnergy = VK_NULL_HANDLE;
		VkPipelineLayout m_pipelineLayoutEnergy = VK_NULL_HANDLE;
		VkPipeline m_pipelineEnergy = VK_NULL_HANDLE;
		VkDescriptorSet m_descSetEnergy = VK_NULL_HANDLE;
		VkDescriptorSet m_descSetUpmlVolt = VK_NULL_HANDLE;
		VkDescriptorSet m_descSetUpmlCurr = VK_NULL_HANDLE;
		uint32_t m_totalMurPoints = 0;
		std::vector<MurFace> m_murFaces;
		VulkanBuffer m_bufMurParams;
		VulkanBuffer m_bufMurStore;
		VkDescriptorSet m_descSetMur = VK_NULL_HANDLE;
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
		VkDescriptorSet m_descSetTfsfVolt = VK_NULL_HANDLE;
		VkDescriptorSet m_descSetTfsfCurr = VK_NULL_HANDLE;
		uint32_t m_rlcCount = 0;
		VulkanBuffer m_bufRlcParams;
		VulkanBuffer m_bufRlcState;
		VkDescriptorSet m_descSetRlc = VK_NULL_HANDLE;
		uint32_t m_abcVoltCount = 0;
		uint32_t m_abcCurrCount = 0;
		std::vector<TfsfFace> m_abcVoltSheets;
		std::vector<TfsfFace> m_abcCurrSheets;
		VulkanBuffer m_bufAbcVoltParams;
		VulkanBuffer m_bufAbcVoltStore;
		VulkanBuffer m_bufAbcCurrParams;
		VulkanBuffer m_bufAbcCurrStore;
		VkDescriptorSet m_descSetAbcVolt = VK_NULL_HANDLE;
		VkDescriptorSet m_descSetAbcCurr = VK_NULL_HANDLE;
		uint32_t m_dispVoltCount = 0;
		uint32_t m_dispCurrCount = 0;
		std::vector<TfsfFace> m_dispVoltPasses;
		std::vector<TfsfFace> m_dispCurrPasses;
		VulkanBuffer m_bufDispVoltParams;
		VulkanBuffer m_bufDispVoltState;
		VulkanBuffer m_bufDispCurrParams;
		VulkanBuffer m_bufDispCurrState;
		VkDescriptorSet m_descSetDispVolt = VK_NULL_HANDLE;
		VkDescriptorSet m_descSetDispCurr = VK_NULL_HANDLE;
		uint32_t m_debyeCount = 0;
		VulkanBuffer m_bufDebyeParams;
		VulkanBuffer m_bufDebyePoles;
		VulkanBuffer m_bufDebyeState;
		VkDescriptorSet m_descSetDebye = VK_NULL_HANDLE;
		bool m_hasCylinder = false;
		bool m_cylClosedAlpha = false;
		bool m_cylR0Included = false;
		uint32_t m_cylLastALine = 0;
		VulkanBuffer m_bufCylR0;
		VkDescriptorSet m_descSetCyl = VK_NULL_HANDLE;
		VulkanBuffer m_bufMultigridInterpolation;
		VkDescriptorSetLayout m_descLayoutMultigrid = VK_NULL_HANDLE;
		VkPipelineLayout m_pipelineLayoutMultigrid = VK_NULL_HANDLE;
		VkDescriptorSet m_descSetMultigrid = VK_NULL_HANDLE;
		VkPipeline m_pipelineMultigrid = VK_NULL_HANDLE;
#endif
	};
	std::unique_ptr<LevelState> m_level;

};

#endif // ENGINE_VULKAN_H
