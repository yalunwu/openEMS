#include "engine_vulkan.h"
#include "shaders_glsl.h"
#include "coefficient_palette.h"

#include <iostream>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <limits>
#include <cmath>
#include <stdexcept>

#include "FDTD/operator.h"
#include "FDTD/operator_cylindermultigrid.h"
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
#include "FDTD/extensions/operator_ext_debyematerial.h"
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

namespace {
class ProfileTimer {
public:
	explicit ProfileTimer(double* seconds) : m_seconds(seconds)
	{
		if (m_seconds) m_start = std::chrono::steady_clock::now();
	}
	~ProfileTimer()
	{
		if (m_seconds)
			*m_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - m_start).count();
	}
private:
	double* m_seconds;
	std::chrono::steady_clock::time_point m_start;
};
}

bool EngineVulkan::SetBatchSize(unsigned int size)
{
	if (size == 0u || size > 64u || GetPendingProbeHistorySteps()) return false;
	m_batchSize = size;
	return true;
}

bool EngineVulkan::SetCoefficientMode(const std::string& mode)
{
	if (mode != "dense" && mode != "palette" && mode != "analyze") return false;
#ifdef ENABLE_VULKAN
	if (m_device) return false;
#endif
	m_coefficientMode = mode;
	return true;
}

bool EngineVulkan::SetReadbackOptimizationsEnabled(bool enabled)
{
	if (GetPendingProbeHistorySteps()) return false;
	m_probeHistoryCount = m_probeHistoryCursor = 0;
#ifdef ENABLE_VULKAN
	if (m_device && !Synchronize()) return false;
	const bool changed = m_readbackOptimizations != enabled;
#endif
	m_readbackOptimizations = enabled;
	m_level->m_probesValid = m_level->m_energyValid = false;
#ifdef ENABLE_VULKAN
	if (m_device && changed)
	{
		if (!AllocateFieldStagingBuffer()) { ProfileOwner().m_runtimeFailed = true; return false; }
		if (!AllocateProbeBuffers(m_level->m_probePoints)) { ProfileOwner().m_runtimeFailed = true; return false; }
	}
#endif
	if (m_level->m_innerGrid && !m_level->m_innerGrid->SetReadbackOptimizationsEnabled(enabled)) return false;
	return true;
}

bool EngineVulkan::SetEnergyFloat64Enabled(bool enabled)
{
#ifdef ENABLE_VULKAN
	if (m_device) return false;
#endif
	m_enableEnergyFloat64 = enabled;
	return true;
}

bool EngineVulkan::SetBarrierCoalescingEnabled(bool enabled)
{
	if (GetPendingProbeHistorySteps()) return false;
#ifdef ENABLE_VULKAN
	EngineVulkan& owner = ProfileOwner();
	if (&owner != this) return owner.SetBarrierCoalescingEnabled(enabled);
	if (m_device && !Synchronize()) return false;
	m_computeBarrierEmitted = false;
#endif
	m_coalesceBarriers = enabled;
	return true;
}

bool EngineVulkan::SetDispersivePreFusionEnabled(bool enabled)
{
	if (GetPendingProbeHistorySteps()) return false;
#ifdef ENABLE_VULKAN
	EngineVulkan& owner = ProfileOwner();
	if (&owner != this) return owner.SetDispersivePreFusionEnabled(enabled);
	if (m_device && !Synchronize()) return false;
#endif
	m_fuseDispersivePre = enabled;
	return true;
}

bool EngineVulkan::SupportsFastEnergy() const
{
#ifdef ENABLE_VULKAN
	// SSE CalcFastEnergy includes padded z lanes and ordered FP32 accumulation.
	// Retain that implementation, including projected root fields in multigrid.
	const Engine* engine = m_level->m_op ? m_level->m_op->GetEngine() : nullptr;
	return m_readbackOptimizations && m_device && engine && engine->GetType() == Engine::BASIC && m_level->m_energyAvailable &&
	       (!m_level->m_energyGroups || (m_level->m_pipelineEnergy && m_level->m_descSetEnergy && m_level->m_bufEnergy.mapped));
#else
	return false;
#endif
}

bool EngineVulkan::GetFastEnergy(double& energy)
{
#ifdef ENABLE_VULKAN
	if (GetPendingProbeHistorySteps() || !SupportsFastEnergy() || ProfileOwner().m_runtimeFailed) return false;
	if (m_level->m_energyValid) { energy = m_level->m_cachedEnergy; return true; }
	if (!m_level->m_energyGroups) { energy = 0; return true; }
	ProfileTimer readbackTimer(ProfileOwner().m_profileEnabled ? &ProfileOwner().m_profile.readbackSeconds : nullptr);
	if (!SyncHierarchyToDevice() || !WaitForFence()) return false;
	VkCommandBufferBeginInfo begin = {};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (vkBeginCommandBuffer(m_level->m_cmdBuffer, &begin) != VK_SUCCESS) return false;
	BeginProfileCommands(m_level->m_cmdBuffer);
	const uint32_t span = BeginGpuSpan(m_level->m_cmdBuffer, ProfileEnergy);
	VkMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	MemoryBarrier(m_level->m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
	              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, barrier);
	const auto& g = m_level->m_grid;
	uint32_t pc[] = {g.dimX, g.dimY, g.dimZ, g.numCells, g.dimX - 1u, g.dimY - 1u, g.dimZ - 1u, m_level->m_energyGroups};
	vkCmdBindPipeline(m_level->m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_level->m_pipelineEnergy);
	vkCmdBindDescriptorSets(m_level->m_cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_level->m_pipelineLayoutEnergy,
	                        0, 1, &m_level->m_descSetEnergy, 0, nullptr);
	vkCmdPushConstants(m_level->m_cmdBuffer, m_level->m_pipelineLayoutEnergy, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
	VkPhysicalDeviceProperties props = {};
	vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
	const uint32_t groupsX = std::min(m_level->m_energyGroups, props.limits.maxComputeWorkGroupCount[0]);
	Dispatch(m_level->m_cmdBuffer, groupsX, (m_level->m_energyGroups + groupsX - 1u) / groupsX, 1, ProfileEnergy);
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
	MemoryBarrier(m_level->m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	              VK_PIPELINE_STAGE_HOST_BIT, barrier);
	EndGpuSpan(m_level->m_cmdBuffer, span);
	if (vkEndCommandBuffer(m_level->m_cmdBuffer) != VK_SUCCESS || !SubmitCommandBuffer()) return false;
	EngineVulkan& owner = ProfileOwner();
	owner.m_queriesPending = owner.m_queryCount != 0u;
	if (!WaitForFence()) return false;
	if (owner.m_profileEnabled) owner.m_profile.energyBytes += m_level->m_bufEnergy.size;
	double electric = 0, magnetic = 0;
	bool nonzeroFields = false;
	const auto* d = static_cast<const double*>(m_level->m_bufEnergy.mapped);
	const auto* f = static_cast<const float*>(m_level->m_bufEnergy.mapped);
	for (uint32_t i = 0; i < m_level->m_energyGroups; ++i) {
		electric += m_energyFloat64 ? d[3u * i] : static_cast<double>(f[3u * i]);
		magnetic += m_energyFloat64 ? d[3u * i + 1u] : static_cast<double>(f[3u * i + 1u]);
		nonzeroFields |= (m_energyFloat64 ? d[3u * i + 2u] : static_cast<double>(f[3u * i + 2u])) != 0;
	}
	energy = EPS0 * electric + MUE0 * magnetic;
	// Preserve the CPU's non-finite/overflow behavior through its existing path.
	if (!std::isfinite(energy)) return false;
	// Devices may flush subnormal float products. Their total omitted contribution
	// is bounded by the number of products times FLT_MIN and the energy weights.
	const double denormalBound = 3.0 * m_level->m_grid.numCells * std::numeric_limits<float>::min() * (EPS0 + MUE0);
	if (energy > 0 && energy < denormalBound / (m_energyFloat64 ? 1e-12 : 3e-7)) return false;
	// A GPU-side bit check distinguishes exact zero from underflow without a field download.
	if (energy == 0 && nonzeroFields) return false;
	m_level->m_cachedEnergy = energy;
	m_level->m_energyValid = true;
	return true;
#else
	UNUSED(energy);
	return false;
#endif
}

bool EngineVulkan::SetProfilingEnabled(bool enabled)
{
#ifdef ENABLE_VULKAN
	EngineVulkan& owner = ProfileOwner();
	if (&owner != this) return owner.SetProfilingEnabled(enabled);
	if (enabled == m_profileEnabled) return true;
	if (m_device && !Synchronize()) return false;
	if (m_queryPool) vkDestroyQueryPool(m_device, m_queryPool, nullptr);
	m_queryPool = VK_NULL_HANDLE;
	m_queriesPending = false;
	m_queryCount = m_profileSpanCount = 0;
#endif
	m_profileEnabled = enabled;
	m_profile = ProfileStatistics();
#ifdef ENABLE_VULKAN
	for (const auto& dump : m_fdDumps) {
		m_profile.fdAccumulatorBytes += dump->bytes;
		m_profile.fdMappingBytes += dump->mappingBytes;
	}
#endif
#ifdef ENABLE_VULKAN
	if (m_device && enabled) return InitProfiling();
#endif
	return true;
}

bool EngineVulkan::Synchronize()
{
#ifdef ENABLE_VULKAN
	return m_device && WaitForFence();
#else
	return false;
#endif
}

bool EngineVulkan::ClearProfile()
{
#ifdef ENABLE_VULKAN
	EngineVulkan& owner = ProfileOwner();
	if (&owner != this) return owner.ClearProfile();
	if (m_device && !Synchronize()) return false;
#endif
	m_profile = ProfileStatistics();
#ifdef ENABLE_VULKAN
	for (const auto& dump : m_fdDumps) {
		m_profile.fdAccumulatorBytes += dump->bytes;
		m_profile.fdMappingBytes += dump->mappingBytes;
	}
#endif
	return true;
}

EngineVulkan::ProfileStatistics EngineVulkan::GetProfile()
{
#ifdef ENABLE_VULKAN
	EngineVulkan& owner = ProfileOwner();
	owner.CollectProfile();
	return owner.m_profile;
#else
	return m_profile;
#endif
}

void EngineVulkan::WriteProfile(std::ostream& stream)
{
	const ProfileStatistics p = GetProfile();
	stream << "VULKAN_PROFILE timesteps=" << p.timesteps << " submissions=" << p.submissions
	       << " dispatches=" << p.dispatches << " sampled_steps=" << p.sampledSteps
	       << " barriers=" << p.barriers << " coalesced_barriers=" << p.coalescedBarriers
	       << " uploaded_bytes=" << p.uploadedBytes << " downloaded_bytes=" << p.downloadedBytes
	       << " probe_bytes=" << p.probeBytes << " record_ms=" << p.recordSeconds * 1000
	       << " submit_ms=" << p.submitSeconds * 1000 << " wait_ms=" << p.waitSeconds * 1000
	       << " readback_ms=" << p.readbackSeconds * 1000 << " mirror_ms=" << p.mirrorSeconds * 1000;
	stream << " energy_bytes=" << p.energyBytes;
	stream << " fd_samples=" << p.fdSamples << " fd_download_bytes=" << p.fdDownloadBytes
	       << " fd_phase_bytes=" << p.fdPhaseBytes << " fd_accumulator_bytes=" << p.fdAccumulatorBytes
	       << " fd_mapping_bytes=" << p.fdMappingBytes
	       << " fd_phase_ms=" << p.fdPhaseSeconds*1000
	       << " fd_download_ms=" << p.fdDownloadSeconds*1000;
	const char* names[] = {"batch", "voltage", "current", "extension", "multigrid", "probe", "readback", "energy", "barrier", "fd"};
	for (unsigned int i = 0; i < ProfileCategoryCount; ++i)
		stream << " gpu_" << names[i] << "_ms=" << p.gpuSeconds[i] * 1000
		       << " gpu_" << names[i] << "_samples=" << p.gpuSamples[i]
		       << " " << names[i] << "_dispatches=" << p.categoryDispatches[i];
	stream << std::endl;
}

std::string EngineVulkan::GetDeviceDescription() const
{
#ifdef ENABLE_VULKAN
	if (m_physicalDevice)
	{
		VkPhysicalDeviceProperties props = {};
		vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
		return m_deviceName + " vendor=" + std::to_string(props.vendorID) +
		       " device=" + std::to_string(props.deviceID) + " driver_raw=" + std::to_string(props.driverVersion) +
		       " api=" + std::to_string(props.apiVersion) + " staging_host_cached=" +
		       std::to_string((m_level->m_bufFieldStaging.memoryProperties & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0);
	}
#endif
	return m_deviceName;
}

#ifdef ENABLE_VULKAN
EngineVulkan& EngineVulkan::ProfileOwner()
{
	return m_deviceOwner ? *m_deviceOwner : *this;
}

bool EngineVulkan::InitProfiling()
{
	if (!m_profileEnabled || !m_ownsVulkanDevice) return true;
	VkPhysicalDeviceProperties props = {};
	vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
	uint32_t count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &count, nullptr);
	std::vector<VkQueueFamilyProperties> queues(count);
	vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &count, queues.data());
	m_timestampBits = queues[m_computeQueueFamily].timestampValidBits;
	m_timestampPeriod = props.limits.timestampPeriod;
	if (!m_timestampBits || m_timestampPeriod <= 0) return true;
	VkQueryPoolCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	info.queryType = VK_QUERY_TYPE_TIMESTAMP;
	info.queryCount = static_cast<uint32_t>(m_profileSpans.size()) * 2u;
	if (vkCreateQueryPool(m_device, &info, nullptr, &m_queryPool) != VK_SUCCESS)
	{
		std::cerr << "[openEMS Vulkan] GPU timestamps unavailable; CPU profiling remains enabled." << std::endl;
		m_queryPool = VK_NULL_HANDLE;
	}
	return true;
}

void EngineVulkan::CollectProfile()
{
	if (!m_queriesPending || !m_queryPool) return;
	struct Result { uint64_t ticks, available; };
	std::array<Result, 512> results = {};
	VkResult status = vkGetQueryPoolResults(m_device, m_queryPool, 0, m_queryCount,
	                                      m_queryCount * sizeof(Result), results.data(), sizeof(Result),
	                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
	if (status != VK_SUCCESS) return;
	const uint64_t mask = m_timestampBits == 64u ? UINT64_MAX : (uint64_t(1) << m_timestampBits) - 1u;
	for (uint32_t i = 0; i < m_profileSpanCount; ++i)
	{
		const ProfileSpan& span = m_profileSpans[i];
		if (!results[span.first].available || !results[span.first + 1u].available) return;
	}
	for (uint32_t i = 0; i < m_profileSpanCount; ++i)
	{
		const ProfileSpan& span = m_profileSpans[i];
		const uint64_t ticks = (results[span.first + 1u].ticks - results[span.first].ticks) & mask;
		m_profile.gpuSeconds[span.category] += ticks * static_cast<double>(m_timestampPeriod) * 1e-9;
		++m_profile.gpuSamples[span.category];
	}
	m_queriesPending = false;
}

uint32_t EngineVulkan::BeginGpuSpan(VkCommandBuffer cmd, GpuProfileCategory category)
{
	EngineVulkan& owner = ProfileOwner();
	if (!owner.m_queryPool || owner.m_profileSpanCount == owner.m_profileSpans.size()) return UINT32_MAX;
	const uint32_t first = owner.m_queryCount;
	owner.m_queryCount += 2u;
	owner.m_profileSpans[owner.m_profileSpanCount++] = {first, category};
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, owner.m_queryPool, first);
	return first;
}

void EngineVulkan::BeginProfileCommands(VkCommandBuffer cmd)
{
	EngineVulkan& owner = ProfileOwner();
	owner.m_queryCount = owner.m_profileSpanCount = 0;
	owner.m_queriesPending = false;
	owner.m_computeBarrierEmitted = false;
	if (owner.m_queryPool)
		vkCmdResetQueryPool(cmd, owner.m_queryPool, 0, static_cast<uint32_t>(owner.m_profileSpans.size()) * 2u);
}

void EngineVulkan::EndGpuSpan(VkCommandBuffer cmd, uint32_t first)
{
	if (first != UINT32_MAX)
		vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ProfileOwner().m_queryPool, first + 1u);
}

void EngineVulkan::Dispatch(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z, GpuProfileCategory category)
{
	EngineVulkan& owner = ProfileOwner();
	owner.m_computeBarrierEmitted = false;
	if (!owner.m_profileEnabled) { vkCmdDispatch(cmd, x, y, z); return; }
	++owner.m_profile.dispatches;
	++owner.m_profile.categoryDispatches[category];
	const uint32_t span = owner.m_sampleDispatches ? BeginGpuSpan(cmd, category) : UINT32_MAX;
	vkCmdDispatch(cmd, x, y, z);
	EndGpuSpan(cmd, span);
}

void EngineVulkan::MemoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                 const VkMemoryBarrier& barrier)
{
	EngineVulkan& owner = ProfileOwner();
	const bool compute = src == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT && dst == src &&
	                     barrier.srcAccessMask == VK_ACCESS_SHADER_WRITE_BIT &&
	                     barrier.dstAccessMask == (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
	// Identical barriers with only bindings/push constants between them have the
	// same memory scope. Dispatch, copy and command-buffer begin invalidate this.
	if (compute && owner.m_computeBarrierEmitted && owner.m_coalesceBarriers) {
		if (owner.m_profileEnabled) ++owner.m_profile.coalescedBarriers;
		return;
	}
	owner.m_computeBarrierEmitted = compute;
	if (owner.m_profileEnabled) ++owner.m_profile.barriers;
	const uint32_t span = owner.m_sampleDispatches ? BeginGpuSpan(cmd, ProfileBarrier) : UINT32_MAX;
	vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &barrier, 0, nullptr, 0, nullptr);
	EndGpuSpan(cmd, span);
}

void EngineVulkan::CopyBuffer(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst,
                             const VkBufferCopy& region, bool download)
{
	EngineVulkan& owner = ProfileOwner();
	owner.m_computeBarrierEmitted = false;
	if (owner.m_profileEnabled)
	{
		if (download) owner.m_profile.downloadedBytes += region.size;
		else owner.m_profile.uploadedBytes += region.size;
	}
	vkCmdCopyBuffer(cmd, src, dst, 1, &region);
}

bool EngineVulkan::WaitForFence()
{
	EngineVulkan& owner = ProfileOwner();
	if (owner.m_runtimeFailed) return false;
	ProfileTimer timer(owner.m_profileEnabled ? &owner.m_profile.waitSeconds : nullptr);
	if (vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
	{
		owner.m_runtimeFailed = true;
		return false;
	}
	owner.CollectProfile();
	return true;
}

bool EngineVulkan::SubmitCommandBuffer()
{
	EngineVulkan& owner = ProfileOwner();
	if (owner.m_runtimeFailed) return false;
	ProfileTimer timer(owner.m_profileEnabled ? &owner.m_profile.submitSeconds : nullptr);
	VkSubmitInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	info.commandBufferCount = 1;
	info.pCommandBuffers = &m_level->m_cmdBuffer;
	if (vkResetFences(m_device, 1, &m_level->m_fence) != VK_SUCCESS ||
	    vkQueueSubmit(m_computeQueue, 1, &info, m_level->m_fence) != VK_SUCCESS)
	{
		owner.m_runtimeFailed = true;
		return false;
	}
	if (owner.m_profileEnabled) ++owner.m_profile.submissions;
	return true;
}
#endif

EngineVulkan::EngineVulkan(const Operator* op)
	: m_level(new LevelState)
{
	m_level->m_op = op;
	m_level->m_multigridOp = dynamic_cast<const Operator_CylinderMultiGrid*>(m_level->m_op);
	if (m_level->m_op)
	{
		m_level->m_grid.dimX = static_cast<uint32_t>(m_level->m_op->GetNumberOfLines(0, true));
		m_level->m_grid.dimY = static_cast<uint32_t>(m_level->m_op->GetNumberOfLines(1, true));
		m_level->m_grid.dimZ = static_cast<uint32_t>(m_level->m_op->GetNumberOfLines(2, true));
		if (m_level->m_multigridOp)
			m_level->m_activeXStart = m_level->m_multigridOp->GetSplitPos() - 1u;
	}
	else
	{
		m_level->m_grid.dimX = 16;
		m_level->m_grid.dimY = 16;
		m_level->m_grid.dimZ = 16;
	}
	uint64_t cells = 1u;
	for (uint32_t dimension : {m_level->m_grid.dimX, m_level->m_grid.dimY, m_level->m_grid.dimZ})
	{
		if (dimension == 0u || cells > (UINT32_MAX / 3u) / dimension)
		{
			m_level->m_dimensionsValid = false;
			break;
		}
		cells *= dimension;
	}
	m_level->m_dimensionsValid = m_level->m_dimensionsValid &&
	                           m_level->m_activeXStart < m_level->m_grid.dimX;
	m_level->m_grid.numCells = m_level->m_dimensionsValid ? static_cast<uint32_t>(cells) : 0u;

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
	try
	{
		if (m_device) Reset();
		if (!m_level->m_dimensionsValid || !InitVulkan() || !InitializeLevel())
		{
			std::cerr << "[openEMS Vulkan] Failed to initialize the GPU grid hierarchy." << std::endl;
			Reset();
			return false;
		}
	}
	catch (const std::exception& error)
	{
		std::cerr << "[openEMS Vulkan] Hierarchy initialization failed: " << error.what() << std::endl;
		Reset();
		return false;
	}

	SetHierarchyTimestep(0u);
	// A reused CPU hierarchy may still contain fields from its previous run.
	MarkHierarchyHostInvalid();
	if (!SyncFieldsToHost())
	{
		Reset();
		return false;
	}
	std::cout << "[openEMS Vulkan] Backend ready. Device: " << m_deviceName << std::endl;
	m_profile = ProfileStatistics();
	return true;
#else
	std::cerr << "[openEMS Vulkan] Engine built without ENABLE_VULKAN support!" << std::endl;
	return false;
#endif
}

#ifdef ENABLE_VULKAN

bool EngineVulkan::InitializeLevel()
{
	if (!m_level->m_dimensionsValid)
	{
		std::cerr << "[openEMS Vulkan] Invalid or overflowing grid dimensions." << std::endl;
		return false;
	}

	VkPhysicalDeviceProperties props = {};
	vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
	const VkDeviceSize fieldBytes = 3ull * m_level->m_grid.numCells * sizeof(float);
	if (props.limits.maxComputeWorkGroupInvocations < 256u ||
	    props.limits.maxComputeWorkGroupSize[0] < 256u ||
	    props.limits.maxComputeWorkGroupSize[1] < 16u ||
	    props.limits.maxComputeWorkGroupSize[2] < 2u ||
	    props.limits.maxPerStageDescriptorStorageBuffers < 6u ||
	    props.limits.maxDescriptorSetStorageBuffers < 6u ||
	    props.limits.maxPushConstantsSize < 11u * sizeof(uint32_t) ||
	    fieldBytes > props.limits.maxStorageBufferRange ||
	    (m_level->m_grid.dimZ + 31u) / 32u > props.limits.maxComputeWorkGroupCount[0] ||
	    (m_level->m_grid.dimY + 3u) / 4u > props.limits.maxComputeWorkGroupCount[1] ||
	    (m_level->m_grid.dimX + 1u) / 2u > props.limits.maxComputeWorkGroupCount[2])
	{
		std::cerr << "[openEMS Vulkan] Grid exceeds Vulkan storage-buffer or dispatch limits." << std::endl;
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
	if (m_level->m_coefficients.palette && !ProfileOwner().CreatePalettePipelines()) return false;
	// Optional capability; unsupported precision/resources retain CPU energy.
	if (!AllocateEnergyResources())
		std::cerr << "[openEMS Vulkan] Fast energy unavailable; retaining CPU energy checks." << std::endl;

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

	if (!AllocateDebyeBuffers()) return false;

	if (!AllocateCylinderBuffers())
	{
		std::cerr << "[openEMS Vulkan] Failed to allocate Cylindrical Coordinates buffers." << std::endl;
		return false;
	}
	const uint64_t maxGroups = props.limits.maxComputeWorkGroupCount[0];
	m_level->m_dispersivePreMaxGroups = props.limits.maxComputeWorkGroupCount[0];
	auto validDispatch = [maxGroups](uint64_t count, uint32_t groupSize) {
		return (count + groupSize - 1u) / groupSize <= maxGroups;
	};
	if (!validDispatch(m_level->m_numUpmlCells, 256u) ||
	    !validDispatch(m_level->m_totalMurPoints, 256u) ||
	    !validDispatch(m_level->m_abcVoltCount, 256u) ||
	    !validDispatch(m_level->m_abcCurrCount, 256u) ||
	    !validDispatch(m_level->m_debyeCount, 256u) ||
	    !validDispatch(m_level->m_rlcCount, 64u) ||
	    !validDispatch(m_level->m_voltExcPoints.size(), 64u) ||
	    !validDispatch(m_level->m_currExcPoints.size(), 64u) ||
	    (m_level->m_hasCylinder && (m_level->m_grid.dimZ + 15u) / 16u > props.limits.maxComputeWorkGroupCount[1]))
		return false;
	for (const auto* passes : {&m_level->m_dispVoltPasses, &m_level->m_dispCurrPasses,
	                           &m_level->m_tfsfVoltFaces, &m_level->m_tfsfCurrFaces})
		for (const auto& pass : *passes)
			if (!validDispatch(pass.count, 256u)) return false;

	if (m_level->m_multigridOp)
	{
		const Operator* innerOp = m_level->m_multigridOp->GetInnerOperator();
		if (!innerOp)
			return false;
		m_level->m_innerGrid.reset(new EngineVulkan(innerOp));
		if (!m_level->m_innerGrid->InitSharedVulkan(*this) || !m_level->m_innerGrid->InitializeLevel() || !AllocateMultigridBuffers())
			return false;
	}

	return true;
}

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
	VkPhysicalDeviceProperties limits = {};
	vkGetPhysicalDeviceProperties(m_physicalDevice, &limits);
	if (size == 0 || ((usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && size > limits.limits.maxStorageBufferRange))
		return false;
	outBuf.size = size;
	outBuf.device = m_device;

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

	uint32_t memTypeIndex = UINT32_MAX;
	if (m_readbackOptimizations && (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
		memTypeIndex = FindMemoryType(memReqs.memoryTypeBits, properties | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
	if (memTypeIndex == UINT32_MAX) memTypeIndex = FindMemoryType(memReqs.memoryTypeBits, properties);
	if (memTypeIndex == UINT32_MAX)
	{
		vkDestroyBuffer(m_device, outBuf.buffer, nullptr);
		outBuf.buffer = VK_NULL_HANDLE;
		return false;
	}
	VkPhysicalDeviceMemoryProperties memoryProperties = {};
	vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);
	outBuf.memoryProperties = memoryProperties.memoryTypes[memTypeIndex].propertyFlags;

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
		if (vkMapMemory(m_device, outBuf.memory, 0, size, 0, &outBuf.mapped) != VK_SUCCESS)
		{
			DestroyBuffer(outBuf);
			return false;
		}
	}

	outBuf.allocationSize = memReqs.size;
	outBuf.memoryHeap = memoryProperties.memoryTypes[memTypeIndex].heapIndex;
	return true;
}

bool EngineVulkan::UploadStorageBuffer(const void* data, VkDeviceSize size, VulkanBuffer& outBuf, bool* allocationFailed)
{
	if (allocationFailed) *allocationFailed = true;
	if (!data || !size) return false;
	if (!CreateBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuf))
		return false;
	VulkanBuffer staging;
	if (!CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging))
		return false;
	std::memcpy(staging.mapped, data, static_cast<size_t>(size));
	if (allocationFailed) *allocationFailed = false;
	bool ok = vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
	if (ok) ok = vkResetFences(m_device, 1, &m_level->m_fence) == VK_SUCCESS;
	VkCommandBufferBeginInfo begin = {};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (ok) ok = vkBeginCommandBuffer(m_level->m_cmdBuffer, &begin) == VK_SUCCESS;
	if (ok)
	{
		VkBufferCopy copy = {0, 0, size};
		vkCmdCopyBuffer(m_level->m_cmdBuffer, staging.buffer, outBuf.buffer, 1, &copy);
		VkMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(m_level->m_cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
		ok = vkEndCommandBuffer(m_level->m_cmdBuffer) == VK_SUCCESS;
	}
	if (ok)
	{
		VkSubmitInfo submit = {};
		submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &m_level->m_cmdBuffer;
		ok = vkQueueSubmit(m_computeQueue, 1, &submit, m_level->m_fence) == VK_SUCCESS;
		if (ok) ok = vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
	}
	DestroyBuffer(staging);
	return ok;
}

void EngineVulkan::DestroyBuffer(VulkanBuffer& buf)
{
	buf.Release();
}

bool EngineVulkan::AllocateFieldStagingBuffer()
{
	const VkDeviceSize fieldBytes = 3ull * m_level->m_grid.numCells * sizeof(float);
	const VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	const VkMemoryPropertyFlags memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	VulkanBuffer replacement;
	if (CreateBuffer((m_readbackOptimizations ? 2u : 1u) * fieldBytes, usage, memory, replacement))
	{
		replacement.Swap(m_level->m_bufFieldStaging);
		return true;
	}
	if (!m_readbackOptimizations) return false;
	replacement.Release();
	// Re-enabling must keep a working single-field buffer if the larger allocation fails.
	if (m_level->m_bufFieldStaging.mapped && m_level->m_bufFieldStaging.size >= fieldBytes) return true;
	if (!CreateBuffer(fieldBytes, usage, memory, replacement)) return false;
	replacement.Swap(m_level->m_bufFieldStaging);
	return true;
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
	VkPhysicalDeviceFeatures available = {}, enabled = {};
	vkGetPhysicalDeviceFeatures(m_physicalDevice, &available);
	enabled.shaderFloat64 = m_enableEnergyFloat64 ? available.shaderFloat64 : VK_FALSE;
	m_energyFloat64 = enabled.shaderFloat64 != VK_FALSE;
	deviceCreateInfo.pEnabledFeatures = &enabled;
	uint32_t extensionCount = 0;
	m_memoryBudget = false;
	vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extensionCount, nullptr);
	std::vector<VkExtensionProperties> extensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extensionCount, extensions.data());
	for (const auto& extension : extensions)
		if (std::strcmp(extension.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) m_memoryBudget = true;
	const char* budgetExtension = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
	if (m_memoryBudget) {
		deviceCreateInfo.enabledExtensionCount = 1;
		deviceCreateInfo.ppEnabledExtensionNames = &budgetExtension;
	}

	if (vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device) != VK_SUCCESS)
	{
		return false;
	}

	vkGetDeviceQueue(m_device, m_computeQueueFamily, 0, &m_computeQueue);

	return InitCommandResources();
}

bool EngineVulkan::InitSharedVulkan(EngineVulkan& parent)
{
	m_coefficientMode = parent.m_coefficientMode;
	m_instance = parent.m_instance;
	m_physicalDevice = parent.m_physicalDevice;
	m_device = parent.m_device;
	m_computeQueue = parent.m_computeQueue;
	m_computeQueueFamily = parent.m_computeQueueFamily;
	m_deviceName = parent.m_deviceName;
	m_energyFloat64 = parent.m_energyFloat64;
	m_ownsVulkanDevice = false;
	m_deviceOwner = parent.m_deviceOwner ? parent.m_deviceOwner : &parent;
	return InitCommandResources();
}

bool EngineVulkan::InitCommandResources()
{

	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = m_computeQueueFamily;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_level->m_cmdPool) != VK_SUCCESS)
	{
		return false;
	}

	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = m_level->m_cmdPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	if (vkAllocateCommandBuffers(m_device, &allocInfo, &m_level->m_cmdBuffer) != VK_SUCCESS)
	{
		return false;
	}

	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	if (vkCreateFence(m_device, &fenceInfo, nullptr, &m_level->m_fence) != VK_SUCCESS)
	{
		return false;
	}
	if (!InitProfiling()) return false;

	return true;
}

bool EngineVulkan::AllocateBuffers()
{
	auto& stats = m_level->m_coefficients;
	stats = CoefficientStatistics();
	stats.nodes = m_level->m_grid.numCells;
	stats.denseBytes = stats.storageBytes = 48ull * stats.nodes;
	VulkanCoefficients::Palette palette;
	if (m_coefficientMode != "dense" && m_level->m_op) {
		// Cap dictionary growth and limit the GPU palette payload to 3 MiB.
		try {
			palette = VulkanCoefficients::Build(*m_level->m_op, m_coefficientMode == "analyze" ? UINT32_MAX : 65536u);
		} catch (const std::bad_alloc&) {
			if (m_coefficientMode == "analyze") throw;
			palette.complete = false;
			std::cerr << "[openEMS Vulkan] Palette host allocation failed; retaining dense coefficients." << std::endl;
		}
		stats.uniqueNodes = palette.tuples.size();
		stats.complete = palette.complete;
		if (m_coefficientMode == "analyze") {
			stats.uniqueComponents = VulkanCoefficients::CountComponents(*m_level->m_op);
			stats.componentBytes = 12ull * stats.nodes + 16ull * stats.uniqueComponents;
		}
		VkPhysicalDeviceProperties props = {};
		vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
		stats.palette = m_coefficientMode == "palette" && palette.complete &&
		                palette.Bytes() < stats.denseBytes &&
		                palette.tuples.size() * 48ull <= props.limits.maxStorageBufferRange;
		if (stats.palette) stats.storageBytes = palette.Bytes();
	}
	const size_t totalElements = 3u * static_cast<size_t>(m_level->m_grid.numCells);
	m_level->m_hostVolt.resize(totalElements, 0.0f);
	m_level->m_hostCurr.resize(totalElements, 0.0f);
	size_t fieldBytes = 3 * static_cast<size_t>(m_level->m_grid.numCells) * sizeof(float);

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	if (stats.palette) {
		bool allocationFailed = false;
		if (!UploadStorageBuffer(palette.tuples.data(), palette.tuples.size() * 48ull, m_level->m_bufVv, &allocationFailed) ||
		    !UploadStorageBuffer(palette.indices.data(), palette.indices.size() * 4ull, m_level->m_bufVi, &allocationFailed)) {
			if (!allocationFailed) return false; // Do not retry failed device work.
			m_level->m_bufVv.Release();
			m_level->m_bufVi.Release();
			stats.palette = false;
			stats.storageBytes = stats.denseBytes;
			std::cerr << "[openEMS Vulkan] Palette allocation failed; retaining dense coefficients." << std::endl;
		}
	}
	if (!stats.palette) {
		if (!CreateBuffer(fieldBytes, storageUsage, devLocal, m_level->m_bufVv)) return false;
		if (!CreateBuffer(fieldBytes, storageUsage, devLocal, m_level->m_bufVi)) return false;
		if (!CreateBuffer(fieldBytes, storageUsage, devLocal, m_level->m_bufIi)) return false;
		if (!CreateBuffer(fieldBytes, storageUsage, devLocal, m_level->m_bufIv)) return false;
	}
	for (const VulkanBuffer* buffer : {&m_level->m_bufVv, &m_level->m_bufVi, &m_level->m_bufIi, &m_level->m_bufIv}) {
		if (!buffer->buffer) continue;
		VkMemoryRequirements requirements = {};
		vkGetBufferMemoryRequirements(m_device, buffer->buffer, &requirements);
		stats.allocatedBytes += requirements.size;
	}
	if (m_coefficientMode != "dense" && m_level->m_op) {
		std::cout << "VULKAN_COEFFICIENTS nodes=" << stats.nodes
		          << " node_unique=" << stats.uniqueNodes << " node_bytes=" << (48ull * stats.uniqueNodes + 4ull * stats.nodes)
		          << " component_unique=" << stats.uniqueComponents << " component_bytes=" << stats.componentBytes
		          << " dense_bytes=" << stats.denseBytes << " storage_bytes=" << stats.storageBytes
		          << " allocated_bytes=" << stats.allocatedBytes
		          << " complete=" << stats.complete << " selected=" << (stats.palette ? "palette" : "dense") << std::endl;
	}
	palette = VulkanCoefficients::Palette();
	if (!CreateBuffer(fieldBytes, storageUsage, devLocal, m_level->m_bufVolt)) return false;
	if (!CreateBuffer(fieldBytes, storageUsage, devLocal, m_level->m_bufCurr)) return false;

	if (!AllocateFieldStagingBuffer()) return false;

	// Upload initial material coefficient matrices if operator is present
	if (m_level->m_op && !stats.palette)
	{
		std::cout << "[openEMS Vulkan] Uploading operator material matrices (vv, vi, ii, iv) to GPU..." << std::endl;
		std::vector<float> hostCoeff(3 * m_level->m_grid.numCells, 0.0f);

		auto uploadMatrix = [&](VulkanBuffer& targetBuf, int matrix) {
			size_t idx = 0;
			for (unsigned int n = 0; n < 3; ++n)
			{
				for (unsigned int x = 0; x < m_level->m_grid.dimX; ++x)
				{
					for (unsigned int y = 0; y < m_level->m_grid.dimY; ++y)
					{
						for (unsigned int z = 0; z < m_level->m_grid.dimZ; ++z)
						{
							switch (matrix)
							{
							case 0: hostCoeff[idx++] = m_level->m_op->GetVV(n, x, y, z); break;
							case 1: hostCoeff[idx++] = m_level->m_op->GetVI(n, x, y, z); break;
							case 2: hostCoeff[idx++] = m_level->m_op->GetII(n, x, y, z); break;
							default: hostCoeff[idx++] = m_level->m_op->GetIV(n, x, y, z); break;
							}
						}
					}
				}
			}

			std::memcpy(m_level->m_bufFieldStaging.mapped, hostCoeff.data(), fieldBytes);

			// Copy staging to device local
			if (vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
			    vkResetFences(m_device, 1, &m_level->m_fence) != VK_SUCCESS)
				return false;

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			if (vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo) != VK_SUCCESS)
				return false;

			VkBufferCopy copyRegion = { 0, 0, fieldBytes };
			vkCmdCopyBuffer(m_level->m_cmdBuffer, m_level->m_bufFieldStaging.buffer, targetBuf.buffer, 1, &copyRegion);
			VkMemoryBarrier barrier = {};
			barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			vkCmdPipelineBarrier(m_level->m_cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
			                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

			if (vkEndCommandBuffer(m_level->m_cmdBuffer) != VK_SUCCESS)
				return false;

			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
			if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence) != VK_SUCCESS)
				return false;
			return vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
		};

		if (!uploadMatrix(m_level->m_bufVv, 0) || !uploadMatrix(m_level->m_bufVi, 1) ||
		    !uploadMatrix(m_level->m_bufIi, 2) || !uploadMatrix(m_level->m_bufIv, 3))
			return false;
	}

	return SyncFieldsToDevice();
}

bool EngineVulkan::SyncFieldsToDevice()
{
	if (!m_level->m_hostFieldsDirty)
		return true;
	if (!m_device || !m_level->m_bufFieldStaging.mapped)
		return false;

	size_t fieldBytes = 3 * static_cast<size_t>(m_level->m_grid.numCells) * sizeof(float);
	auto uploadBuffer = [&](const std::vector<float>& source, VulkanBuffer& target) {
		std::memcpy(m_level->m_bufFieldStaging.mapped, source.data(), fieldBytes);
		if (!WaitForFence())
			return false;

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		if (vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo) != VK_SUCCESS)
			return false;

		VkBufferCopy copyRegion = { 0, 0, fieldBytes };
		CopyBuffer(m_level->m_cmdBuffer, m_level->m_bufFieldStaging.buffer, target.buffer, copyRegion);
		VkMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		MemoryBarrier(m_level->m_cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, barrier);
		if (vkEndCommandBuffer(m_level->m_cmdBuffer) != VK_SUCCESS)
			return false;

		return SubmitCommandBuffer() && WaitForFence();
	};

	if (!uploadBuffer(m_level->m_hostVolt, m_level->m_bufVolt) || !uploadBuffer(m_level->m_hostCurr, m_level->m_bufCurr))
		return false;

	m_level->m_hostFieldsDirty = false;
	m_level->m_hostFieldsValid = true;
	return true;
}

bool EngineVulkan::CreatePipelines()
{
	if (!m_ownsVulkanDevice)
	{
		// All levels use the root's pipelines, layouts, and hierarchy-sized pool.
		const EngineVulkan& owner = *m_deviceOwner;
		m_descLayoutFields = owner.m_descLayoutFields;
		m_pipelineLayoutFields = owner.m_pipelineLayoutFields;
		m_pipelineVolt = owner.m_pipelineVolt;
		m_pipelineCurr = owner.m_pipelineCurr;
		m_descLayoutExc = owner.m_descLayoutExc;
		m_pipelineLayoutExc = owner.m_pipelineLayoutExc;
		m_pipelineExc = owner.m_pipelineExc;
		m_descLayoutProbe = owner.m_descLayoutProbe;
		m_pipelineLayoutProbe = owner.m_pipelineLayoutProbe;
		m_pipelineProbe = owner.m_pipelineProbe;
		m_descLayoutUpml = owner.m_descLayoutUpml;
		m_pipelineLayoutUpml = owner.m_pipelineLayoutUpml;
		m_pipelineUpmlPre = owner.m_pipelineUpmlPre;
		m_pipelineUpmlPost = owner.m_pipelineUpmlPost;
		m_descLayoutMur = owner.m_descLayoutMur;
		m_pipelineLayoutMur = owner.m_pipelineLayoutMur;
		m_pipelineMurPre = owner.m_pipelineMurPre;
		m_pipelineMurPost = owner.m_pipelineMurPost;
		m_pipelineMurApply = owner.m_pipelineMurApply;
		m_descLayoutTfsf = owner.m_descLayoutTfsf;
		m_pipelineLayoutTfsf = owner.m_pipelineLayoutTfsf;
		m_pipelineTfsf = owner.m_pipelineTfsf;
		m_descLayoutRlc = owner.m_descLayoutRlc;
		m_pipelineLayoutRlc = owner.m_pipelineLayoutRlc;
		m_pipelineRlc = owner.m_pipelineRlc;
		m_descLayoutAbc = owner.m_descLayoutAbc;
		m_pipelineLayoutAbc = owner.m_pipelineLayoutAbc;
		m_pipelineAbcVolt = owner.m_pipelineAbcVolt;
		m_pipelineAbcCurr = owner.m_pipelineAbcCurr;
		m_descLayoutDisp = owner.m_descLayoutDisp;
		m_pipelineLayoutDisp = owner.m_pipelineLayoutDisp;
		m_pipelineDisp = owner.m_pipelineDisp;
		m_descLayoutDebye = owner.m_descLayoutDebye;
		m_pipelineLayoutDebye = owner.m_pipelineLayoutDebye;
		m_pipelineDebye = owner.m_pipelineDebye;
		m_descLayoutCyl = owner.m_descLayoutCyl;
		m_pipelineLayoutCyl = owner.m_pipelineLayoutCyl;
		m_pipelineCyl = owner.m_pipelineCyl;
		m_descPool = owner.m_descPool;
		return AllocateFieldDescriptors();
	}
	uint32_t levelCount = 0u;

	bool hasUpml = false;
	bool hasMur = false;
	bool hasTfsf = false;
	bool hasRlc = false;
	bool hasAbc = false;
	bool hasDispersive = false;
	bool hasDebye = false;
	bool hasCylinder = false;
	bool hasMultigrid = m_level->m_multigridOp != nullptr;
	for (const Operator* levelOp = m_level->m_op; levelOp;)
	{
		++levelCount;
		for (size_t i = 0; i < levelOp->GetNumberOfExtentions(); ++i)
		{
			if (dynamic_cast<Operator_Ext_UPML*>(levelOp->GetExtension(i)))
				hasUpml = true;
			if (dynamic_cast<Operator_Ext_Mur_ABC*>(levelOp->GetExtension(i)))
				hasMur = true;
			if (dynamic_cast<Operator_Ext_TFSF*>(levelOp->GetExtension(i)))
				hasTfsf = true;
			if (dynamic_cast<Operator_Ext_LumpedRLC*>(levelOp->GetExtension(i)))
				hasRlc = true;
			if (dynamic_cast<Operator_Ext_Absorbing_BC*>(levelOp->GetExtension(i)))
				hasAbc = true;
			if (dynamic_cast<Operator_Ext_LorentzMaterial*>(levelOp->GetExtension(i)))
				hasDispersive = true;
			if (dynamic_cast<Operator_Ext_DebyeMaterial*>(levelOp->GetExtension(i)))
				hasDebye = true;
			if (dynamic_cast<Operator_Ext_Cylinder*>(levelOp->GetExtension(i)))
				hasCylinder = true;
		}
		const auto* mg = dynamic_cast<const Operator_CylinderMultiGrid*>(levelOp);
		levelOp = mg ? mg->GetInnerOperator() : nullptr;
	}
	if (levelCount > CYLIDINDERMULTIGRID_LIMIT + 1u) return false;
	levelCount = std::max(levelCount, 1u);

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
	std::vector<uint32_t> spvDebye;
	if (hasDebye)
		spvDebye = CompileGLSLToSpirv(VulkanShaders::kShaderDebye, shaderc_glsl_compute_shader, "debye.comp");
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
	    (hasDispersive && spvDisp.empty()) || (hasDebye && spvDebye.empty()) ||
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
	VkShaderModule modDebye = hasDebye ? CreateShaderModule(spvDebye) : VK_NULL_HANDLE;
	VkShaderModule modCyl = VK_NULL_HANDLE;
	if (hasCylinder)
	{
		modCyl = CreateShaderModule(spvCylinder);
	}
	// Release shader modules on success and on every partial pipeline failure.
	struct ModuleCleanup {
		VkDevice device;
		VkShaderModule modules[16];
		~ModuleCleanup()
		{
			for (VkShaderModule module : modules)
				if (module) vkDestroyShaderModule(device, module, nullptr);
		}
	} moduleCleanup = {m_device, {modVolt, modCurr, modExc, modPrb, modUpmlPre, modUpmlPost,
	                            modMurPre, modMurPost, modMurApply, modTfsf, modRlc,
	                            modAbcVolt, modAbcCurr, modDisp, modCyl, modDebye}};
	if (!modVolt || !modCurr || !modExc || !modPrb ||
	    (hasUpml && (!modUpmlPre || !modUpmlPost)) ||
	    (hasMur && (!modMurPre || !modMurPost || !modMurApply)) ||
	    (hasTfsf && !modTfsf) || (hasRlc && !modRlc) ||
	    (hasAbc && (!modAbcVolt || !modAbcCurr)) ||
	    (hasDispersive && !modDisp) || (hasDebye && !modDebye) || (hasCylinder && !modCyl))
		return false;

	// Create descriptor pool
	uint32_t maxStorage = 32u;
	uint32_t maxSets = 16u;
	if (hasUpml) { maxStorage += 32u; maxSets += 16u; }
	if (hasMur)  { maxStorage += 32u; maxSets += 16u; }
	if (hasTfsf) { maxStorage += 32u; maxSets += 16u; }
	if (hasRlc)  { maxStorage += 32u; maxSets += 16u; }
	if (hasAbc)  { maxStorage += 32u; maxSets += 16u; }
	if (hasDispersive) { maxStorage += 32u; maxSets += 16u; }
	if (hasDebye)      { maxStorage += 4u; maxSets += 1u; }
	if (hasCylinder)   { maxStorage += 32u; maxSets += 16u; }
	if (hasMultigrid)  { maxStorage += 16u; maxSets += 4u; }
	std::vector<VkDescriptorPoolSize> poolSizes = {
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxStorage * levelCount }
	};
	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = maxSets * levelCount;
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
	pcRange.size = 6 * sizeof(uint32_t); // dimensions plus active radial start/count

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

	if (!AllocateFieldDescriptors()) return false;

	// 2. Excitation Pipeline Layout
	std::vector<VkDescriptorSetLayoutBinding> excBindings(3);
	for (uint32_t i = 0; i < 3; ++i)
	{
		excBindings[i].binding = i;
		excBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		excBindings[i].descriptorCount = 1;
		excBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	layoutInfo.bindingCount = 3;
	layoutInfo.pBindings = excBindings.data();
	if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayoutExc) != VK_SUCCESS)
	{
		return false;
	}

	VkPushConstantRange excPcRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 2u * sizeof(uint32_t) };
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
	excPcRange.size = 2u * sizeof(uint32_t);
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

	if (hasDebye)
	{
		VkDescriptorSetLayoutBinding bindings[4] = {};
		for (uint32_t i = 0; i < 4; ++i)
		{
			bindings[i].binding = i;
			bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		layoutInfo.bindingCount = 4;
		layoutInfo.pBindings = bindings;
		if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayoutDebye) != VK_SUCCESS)
			return false;
		VkPushConstantRange range = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(uint32_t) };
		pipeLayoutInfo.pSetLayouts = &m_descLayoutDebye;
		pipeLayoutInfo.pPushConstantRanges = &range;
		if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_pipelineLayoutDebye) != VK_SUCCESS)
			return false;
		if (!createComputePipe(modDebye, m_pipelineLayoutDebye, m_pipelineDebye)) return false;
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

	return true;
}

bool EngineVulkan::CreatePalettePipelines()
{
	if (m_pipelineVoltPalette && m_pipelineCurrPalette) return true;
	const char* sources[] = {VulkanShaders::kShaderVoltageUpdate, VulkanShaders::kShaderCurrentUpdate};
	VkPipeline* pipelines[] = {&m_pipelineVoltPalette, &m_pipelineCurrPalette};
	for (unsigned int i = 0; i < 2; ++i) {
		if (*pipelines[i]) continue;
		std::string source(sources[i]);
		source.insert(source.find('\n') + 1, "#define COEFFICIENT_PALETTE\n");
		const auto spirv = CompileGLSLToSpirv(source, shaderc_glsl_compute_shader, "yee_palette.comp");
		if (spirv.empty()) return false;
		const VkShaderModule module = CreateShaderModule(spirv);
		if (!module) return false;
		VkComputePipelineCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		info.stage.module = module;
		info.stage.pName = "main";
		info.layout = m_pipelineLayoutFields;
		const VkResult result = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &info, nullptr, pipelines[i]);
		vkDestroyShaderModule(m_device, module, nullptr);
		if (result != VK_SUCCESS) return false;
	}
	return true;
}

bool EngineVulkan::AllocateFieldDescriptors()
{
	// Allocate and update descriptor set for Fields
	VkDescriptorSetAllocateInfo dsAlloc = {};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = m_descPool;
	dsAlloc.descriptorSetCount = 1;
	dsAlloc.pSetLayouts = &m_descLayoutFields;
	if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_level->m_descSetFields) != VK_SUCCESS)
	{
		return false;
	}

	std::vector<VkDescriptorBufferInfo> bufInfos(6);
	VulkanBuffer* bufs[6] = { &m_level->m_bufVv, &m_level->m_bufVi, &m_level->m_bufIi, &m_level->m_bufIv, &m_level->m_bufVolt, &m_level->m_bufCurr };
	// Palette shaders only read bindings 0/1; keep the shared six-binding layout.
	if (m_level->m_coefficients.palette) bufs[2] = bufs[3] = &m_level->m_bufVv;
	std::vector<VkWriteDescriptorSet> writes(6);
	for (uint32_t i = 0; i < 6; ++i)
	{
		bufInfos[i].buffer = bufs[i]->buffer;
		bufInfos[i].offset = 0;
		bufInfos[i].range = bufs[i]->size;

		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = m_level->m_descSetFields;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo = &bufInfos[i];
	}
	vkUpdateDescriptorSets(m_device, 6, writes.data(), 0, nullptr);

	return true;
}

bool EngineVulkan::AllocateExcitationBuffers()
{
	m_level->m_voltExcPoints.clear();
	m_level->m_currExcPoints.clear();
	if (!m_level->m_op) return true;

	std::vector<float> signals;
	for (size_t i = 0; i < m_level->m_op->GetNumberOfExtentions(); ++i)
	{
		auto* excExt = dynamic_cast<Operator_Ext_Excitation*>(m_level->m_op->GetExtension(i));
		if (!excExt || !excExt->m_Exc) continue;
		const Excitation* exc = excExt->m_Exc;
		const uint32_t length = exc->GetLength();
		const double periodSteps = exc->GetSignalPeriod() / exc->GetTimestep();
		if (!length || !std::isfinite(periodSteps) || periodSteps > INT32_MAX ||
		    (exc->GetSignalPeriod() > 0 && periodSteps < 1.0))
			return false;
		const uint32_t period = exc->GetSignalPeriod() > 0 ? static_cast<uint32_t>(periodSteps) : 0u;

		auto append = [&](bool voltage, std::vector<GpuExcPoint>& points) {
			const uint32_t count = voltage ? excExt->Volt_Count : excExt->Curr_Count;
			const FDTD_FLOAT* signal = voltage ? exc->GetVoltageSignal() : exc->GetCurrentSignal();
			if (!count || !signal) return true;
			if (signals.size() > UINT32_MAX - length) return false;
			const uint32_t offset = static_cast<uint32_t>(signals.size());
			signals.insert(signals.end(), signal, signal + length);
			for (uint32_t n = 0; n < count; ++n)
			{
				const unsigned int direction = voltage ? excExt->Volt_dir[n] : excExt->Curr_dir[n];
				const unsigned int x = voltage ? excExt->Volt_index[0][n] : excExt->Curr_index[0][n];
				const unsigned int y = voltage ? excExt->Volt_index[1][n] : excExt->Curr_index[1][n];
				const unsigned int z = voltage ? excExt->Volt_index[2][n] : excExt->Curr_index[2][n];
				const double delay = voltage ? excExt->Volt_delay[n] : excExt->Curr_delay[n];
				if (direction >= 3u || x >= m_level->m_grid.dimX || y >= m_level->m_grid.dimY || z >= m_level->m_grid.dimZ ||
				    !std::isfinite(delay) || delay < 0 || delay > INT32_MAX)
					return false;
				const float amplitude = voltage ? excExt->Volt_amp[n] : excExt->Curr_amp[n];
				points.push_back({GetCheckedLinearIndex(direction, x, y, z),
				                  amplitude, static_cast<uint32_t>(delay), offset, length, period});
			}
			return true;
		};
		if (!append(true, m_level->m_voltExcPoints) || !append(false, m_level->m_currExcPoints))
			return false;
	}

	if (signals.empty()) return true;
	if (!UploadStorageBuffer(signals.data(), signals.size() * sizeof(float), m_level->m_bufExcSignals))
		return false;

	auto allocate = [&](const std::vector<GpuExcPoint>& points, VulkanBuffer& pointBuffer,
	                    VulkanBuffer& fields, VkDescriptorSet& set) {
		if (points.empty()) return true;
		if (!UploadStorageBuffer(points.data(), points.size() * sizeof(GpuExcPoint), pointBuffer))
			return false;
		VkDescriptorSetAllocateInfo alloc = {};
		alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		alloc.descriptorPool = m_descPool;
		alloc.descriptorSetCount = 1;
		alloc.pSetLayouts = &m_descLayoutExc;
		if (vkAllocateDescriptorSets(m_device, &alloc, &set) != VK_SUCCESS)
			return false;
		VkDescriptorBufferInfo infos[3] = {
			{pointBuffer.buffer, 0, pointBuffer.size}, {fields.buffer, 0, fields.size},
			{m_level->m_bufExcSignals.buffer, 0, m_level->m_bufExcSignals.size}
		};
		VkWriteDescriptorSet writes[3] = {};
		for (uint32_t i = 0; i < 3u; ++i)
		{
			writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writes[i].dstSet = set;
			writes[i].dstBinding = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writes[i].pBufferInfo = &infos[i];
		}
		vkUpdateDescriptorSets(m_device, 3u, writes, 0u, nullptr);
		return true;
	};
	std::cout << "[openEMS Vulkan] Initialized on-device excitation: "
	          << m_level->m_voltExcPoints.size() << " voltage points, "
	          << m_level->m_currExcPoints.size() << " current points." << std::endl;
	return allocate(m_level->m_voltExcPoints, m_level->m_bufVoltExcPoints, m_level->m_bufVolt, m_level->m_descSetVoltExc) &&
	       allocate(m_level->m_currExcPoints, m_level->m_bufCurrExcPoints, m_level->m_bufCurr, m_level->m_descSetCurrExc);
}

bool EngineVulkan::AllocateProbeBuffers(const std::vector<ProbePoint>& points, unsigned int capacity)
{
	if (!capacity || capacity > 64u || points.size() > UINT32_MAX / capacity ||
	    points.size() > std::numeric_limits<size_t>::max() / sizeof(ProbePoint)) return false;
	if (points.empty())
	{
		DestroyBuffer(m_level->m_bufProbePoints);
		DestroyBuffer(m_level->m_bufProbeValues);
		m_level->m_probeCapacity = 1;
		// Retain the allocated descriptor set for reuse when probes are added again.
		return true;
	}

	VulkanBuffer pointsBuffer, valuesBuffer;
	size_t count = points.size();
	size_t pointsBytes = count * sizeof(ProbePoint);
	VkDeviceSize valuesBytes = VkDeviceSize(count) * capacity * sizeof(float);
	VkPhysicalDeviceProperties props = {};
	vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
	if (pointsBytes > props.limits.maxStorageBufferRange || valuesBytes > props.limits.maxStorageBufferRange) return false;

	// Create and populate probe points buffer
	if (!CreateBuffer(pointsBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, pointsBuffer))
	{
		return false;
	}
	std::memcpy(pointsBuffer.mapped, points.data(), pointsBytes);

	// Create host-visible probe values buffer (directly mapped for zero-copy readback!)
	if (!CreateBuffer(valuesBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, valuesBuffer))
	{
		return false;
	}

	VkDescriptorSetAllocateInfo dsAlloc = {};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = m_descPool;
	dsAlloc.descriptorSetCount = 1;
	dsAlloc.pSetLayouts = &m_descLayoutProbe;
	VkDescriptorSet descriptor = m_level->m_descSetProbe;
	if (!descriptor && vkAllocateDescriptorSets(m_device, &dsAlloc, &descriptor) != VK_SUCCESS)
	{
		return false;
	}

	VkDescriptorBufferInfo bInfos[4] = {
		{ pointsBuffer.buffer, 0, pointsBytes },
		{ m_level->m_bufVolt.buffer, 0, m_level->m_bufVolt.size },
		{ m_level->m_bufCurr.buffer, 0, m_level->m_bufCurr.size },
		{ valuesBuffer.buffer, 0, valuesBytes }
	};
	VkWriteDescriptorSet writes[4] = {};
	for (uint32_t i = 0; i < 4; ++i)
	{
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = descriptor;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo = &bInfos[i];
	}
	vkUpdateDescriptorSets(m_device, 4, writes, 0, nullptr);

	// Commit only after all allocations succeed; local buffers release the old storage.
	m_level->m_descSetProbe = descriptor;
	pointsBuffer.Swap(m_level->m_bufProbePoints);
	valuesBuffer.Swap(m_level->m_bufProbeValues);
	m_level->m_probeCapacity = capacity;
	return true;
}

bool EngineVulkan::AllocateEnergyResources()
{
	const Engine* engine = m_level->m_op ? m_level->m_op->GetEngine() : nullptr;
	if (!engine || engine->GetType() != Engine::BASIC) return true;
	struct AllocationGuard {
		EngineVulkan* engine;
		bool committed;
		~AllocationGuard() { if (!committed) engine->DestroyEnergyResources(); }
	} guard = {this, false};
	const auto& g = m_level->m_grid;
	const uint64_t cells = uint64_t(g.dimX - 1u) * (g.dimY - 1u) * (g.dimZ - 1u);
	m_level->m_energyGroups = static_cast<uint32_t>((cells + 255u) / 256u);
	if (!m_level->m_energyGroups) {
		m_level->m_energyAvailable = guard.committed = true;
		return true;
	}
	VkPhysicalDeviceProperties props = {};
	vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
	const uint32_t gx = std::min(m_level->m_energyGroups, props.limits.maxComputeWorkGroupCount[0]);
	if ((m_level->m_energyGroups + gx - 1u) / gx > props.limits.maxComputeWorkGroupCount[1]) return false;
	const VkDeviceSize bytes = uint64_t(m_level->m_energyGroups) * 3u * (m_energyFloat64 ? sizeof(double) : sizeof(float));
	if (!CreateBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_level->m_bufEnergy)) return false;
	std::string source(VulkanShaders::kShaderFastEnergy);
	if (m_energyFloat64) source.insert(source.find('\n') + 1u, "#define USE_FP64\n");
	const auto spirv = CompileGLSLToSpirv(source, shaderc_glsl_compute_shader, "fast_energy.comp");
	VkShaderModule module = CreateShaderModule(spirv);
	if (!module) return false;
	struct ModuleGuard { VkDevice device; VkShaderModule module; ~ModuleGuard() { vkDestroyShaderModule(device, module, nullptr); } } moduleGuard = {m_device, module};
	VkDescriptorSetLayoutBinding bindings[3] = {};
	for (uint32_t i = 0; i < 3u; ++i) {
		bindings[i].binding = i;
		bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	VkDescriptorSetLayoutCreateInfo layout = {};
	layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layout.bindingCount = 3;
	layout.pBindings = bindings;
	if (vkCreateDescriptorSetLayout(m_device, &layout, nullptr, &m_level->m_descLayoutEnergy) != VK_SUCCESS) return false;
	VkPushConstantRange range = {};
	range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	range.size = 8u * sizeof(uint32_t);
	VkPipelineLayoutCreateInfo pipelineLayout = {};
	pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayout.setLayoutCount = 1;
	pipelineLayout.pSetLayouts = &m_level->m_descLayoutEnergy;
	pipelineLayout.pushConstantRangeCount = 1;
	pipelineLayout.pPushConstantRanges = &range;
	if (vkCreatePipelineLayout(m_device, &pipelineLayout, nullptr, &m_level->m_pipelineLayoutEnergy) != VK_SUCCESS) return false;
	VkComputePipelineCreateInfo pipeline = {};
	pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	pipeline.stage.module = module;
	pipeline.stage.pName = "main";
	pipeline.layout = m_level->m_pipelineLayoutEnergy;
	if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &m_level->m_pipelineEnergy) != VK_SUCCESS) return false;
	VkDescriptorSetAllocateInfo allocate = {};
	allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocate.descriptorPool = m_descPool;
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts = &m_level->m_descLayoutEnergy;
	if (vkAllocateDescriptorSets(m_device, &allocate, &m_level->m_descSetEnergy) != VK_SUCCESS) return false;
	VulkanBuffer* buffers[] = {&m_level->m_bufVolt, &m_level->m_bufCurr, &m_level->m_bufEnergy};
	VkDescriptorBufferInfo info[3] = {};
	VkWriteDescriptorSet writes[3] = {};
	for (uint32_t i = 0; i < 3u; ++i) {
		info[i].buffer = buffers[i]->buffer;
		info[i].range = buffers[i]->size;
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = m_level->m_descSetEnergy;
		writes[i].dstBinding = i;
		writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].descriptorCount = 1;
		writes[i].pBufferInfo = &info[i];
	}
	vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);
	m_level->m_energyAvailable = guard.committed = true;
	std::cout << "[openEMS Vulkan] Fast energy uses " << (m_energyFloat64 ? "FP64" : "FP32") << " partial sums and CPU double accumulation." << std::endl;
	return true;
}

void EngineVulkan::DestroyEnergyResources()
{
	DestroyBuffer(m_level->m_bufEnergy);
	if (m_level->m_pipelineEnergy) vkDestroyPipeline(m_device, m_level->m_pipelineEnergy, nullptr);
	if (m_level->m_pipelineLayoutEnergy) vkDestroyPipelineLayout(m_device, m_level->m_pipelineLayoutEnergy, nullptr);
	if (m_level->m_descLayoutEnergy) vkDestroyDescriptorSetLayout(m_device, m_level->m_descLayoutEnergy, nullptr);
	m_level->m_pipelineEnergy = VK_NULL_HANDLE;
	m_level->m_pipelineLayoutEnergy = VK_NULL_HANDLE;
	m_level->m_descLayoutEnergy = VK_NULL_HANDLE;
	// Sets are owned by the descriptor pool; allocation is the last fallible step.
	m_level->m_descSetEnergy = VK_NULL_HANDLE;
	m_level->m_energyGroups = 0;
	m_level->m_energyAvailable = m_level->m_energyValid = false;
}

void EngineVulkan::RecordProbeGather(VkCommandBuffer cmd, unsigned int slot)
{
	const uint32_t count = static_cast<uint32_t>(m_level->m_probePoints.size());
	const uint32_t span = BeginGpuSpan(cmd, ProfileProbe);
	VkMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
	              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, barrier);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineProbe);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutProbe, 0, 1, &m_level->m_descSetProbe, 0, nullptr);
	const uint32_t pc[] = {count, slot * count};
	vkCmdPushConstants(cmd, m_pipelineLayoutProbe, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
	Dispatch(cmd, (count + 63u) / 64u, 1, 1, ProfileProbe);
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
	// History gathering reads the fields between timesteps. Besides publishing
	// probe values to the host, order those reads before the next field writes.
	MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	              VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, barrier);
	EndGpuSpan(cmd, span);
	EngineVulkan& owner = ProfileOwner();
	if (owner.m_profileEnabled) owner.m_profile.probeBytes += count * sizeof(float);
}

uint32_t EngineVulkan::GetCheckedLinearIndex(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	const size_t index = GetLinearIndex(n, x, y, z);
	if (index == static_cast<size_t>(-1) || index > UINT32_MAX)
		throw std::out_of_range("[openEMS Vulkan] Extension index is outside the field grid");
	return static_cast<uint32_t>(index);
}

bool EngineVulkan::AllocateUpmlBuffers()
{
	m_level->m_numUpmlCells = 0;
	if (!m_level->m_op) return true;

	std::vector<Operator_Ext_UPML*> upmlExts;
	for (size_t i = 0; i < m_level->m_op->GetNumberOfExtentions(); ++i)
	{
		Operator_Ext_UPML* upml = dynamic_cast<Operator_Ext_UPML*>(m_level->m_op->GetExtension(i));
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
		const unsigned int dims[] = {m_level->m_grid.dimX, m_level->m_grid.dimY, m_level->m_grid.dimZ};
		for (unsigned int n = 0; n < 3; ++n)
			if (upml->m_StartPos[n] >= dims[n] || !upml->m_numLines[n] || upml->m_numLines[n] > dims[n] - upml->m_StartPos[n])
				throw std::out_of_range("[openEMS Vulkan] UPML range is outside the field grid");
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
						uint32_t linFieldIdx = GetCheckedLinearIndex(comp, posX, posY, posZ);

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

	m_level->m_numUpmlCells = static_cast<uint32_t>(hostIndices.size());
	if (m_level->m_numUpmlCells == 0)
		return true;

	std::cout << "[openEMS Vulkan] Initialized on-device UPML: "
	          << m_level->m_numUpmlCells << " cell components across "
	          << upmlExts.size() << " boundary layer(s)." << std::endl;

	VkDeviceSize idxBytes   = m_level->m_numUpmlCells * sizeof(uint32_t);
	VkDeviceSize coeffBytes = m_level->m_numUpmlCells * 4 * sizeof(float);
	VkDeviceSize fluxBytes  = m_level->m_numUpmlCells * sizeof(float);

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	if (!CreateBuffer(idxBytes, storageUsage, devLocal, m_level->m_bufUpmlIndices)) return false;
	if (!CreateBuffer(coeffBytes, storageUsage, devLocal, m_level->m_bufUpmlVoltCoeffs)) return false;
	if (!CreateBuffer(coeffBytes, storageUsage, devLocal, m_level->m_bufUpmlCurrCoeffs)) return false;
	if (!CreateBuffer(fluxBytes, storageUsage, devLocal, m_level->m_bufUpmlVoltFlux)) return false;
	if (!CreateBuffer(fluxBytes, storageUsage, devLocal, m_level->m_bufUpmlCurrFlux)) return false;

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

	vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
	vkResetFences(m_device, 1, &m_level->m_fence);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);

	VkBufferCopy copyIdx = { 0, 0, idxBytes };
	vkCmdCopyBuffer(m_level->m_cmdBuffer, stageIndices.buffer, m_level->m_bufUpmlIndices.buffer, 1, &copyIdx);

	VkBufferCopy copyCoeff = { 0, 0, coeffBytes };
	vkCmdCopyBuffer(m_level->m_cmdBuffer, stageVoltCoeffs.buffer, m_level->m_bufUpmlVoltCoeffs.buffer, 1, &copyCoeff);
	vkCmdCopyBuffer(m_level->m_cmdBuffer, stageCurrCoeffs.buffer, m_level->m_bufUpmlCurrCoeffs.buffer, 1, &copyCoeff);

	// Zero flux buffers
	vkCmdFillBuffer(m_level->m_cmdBuffer, m_level->m_bufUpmlVoltFlux.buffer, 0, VK_WHOLE_SIZE, 0);
	vkCmdFillBuffer(m_level->m_cmdBuffer, m_level->m_bufUpmlCurrFlux.buffer, 0, VK_WHOLE_SIZE, 0);

	vkEndCommandBuffer(m_level->m_cmdBuffer);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
	vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
	vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);

	DestroyBuffer(stageIndices);
	DestroyBuffer(stageVoltCoeffs);
	DestroyBuffer(stageCurrCoeffs);

	// Allocate and configure descriptor sets for UPML
	VkDescriptorSetAllocateInfo dsAlloc = {};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = m_descPool;
	dsAlloc.descriptorSetCount = 1;
	dsAlloc.pSetLayouts = &m_descLayoutUpml;

	if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_level->m_descSetUpmlVolt) != VK_SUCCESS) return false;
	if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_level->m_descSetUpmlCurr) != VK_SUCCESS) return false;

	// Update m_level->m_descSetUpmlVolt
	VkDescriptorBufferInfo voltBufInfos[4] = {
		{ m_level->m_bufUpmlIndices.buffer, 0, idxBytes },
		{ m_level->m_bufUpmlVoltCoeffs.buffer, 0, coeffBytes },
		{ m_level->m_bufUpmlVoltFlux.buffer, 0, fluxBytes },
		{ m_level->m_bufVolt.buffer, 0, m_level->m_bufVolt.size }
	};
	VkWriteDescriptorSet voltWrites[4] = {};
	for (uint32_t i = 0; i < 4; ++i)
	{
		voltWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		voltWrites[i].dstSet = m_level->m_descSetUpmlVolt;
		voltWrites[i].dstBinding = i;
		voltWrites[i].descriptorCount = 1;
		voltWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		voltWrites[i].pBufferInfo = &voltBufInfos[i];
	}
	vkUpdateDescriptorSets(m_device, 4, voltWrites, 0, nullptr);

	// Update m_level->m_descSetUpmlCurr
	VkDescriptorBufferInfo currBufInfos[4] = {
		{ m_level->m_bufUpmlIndices.buffer, 0, idxBytes },
		{ m_level->m_bufUpmlCurrCoeffs.buffer, 0, coeffBytes },
		{ m_level->m_bufUpmlCurrFlux.buffer, 0, fluxBytes },
		{ m_level->m_bufCurr.buffer, 0, m_level->m_bufCurr.size }
	};
	VkWriteDescriptorSet currWrites[4] = {};
	for (uint32_t i = 0; i < 4; ++i)
	{
		currWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		currWrites[i].dstSet = m_level->m_descSetUpmlCurr;
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
	m_level->m_totalMurPoints = 0;
	m_level->m_murFaces.clear();
	if (!m_level->m_op) return true;

	std::vector<GpuMurPoint> hostPoints;

	for (size_t extIdx = 0; extIdx < m_level->m_op->GetNumberOfExtentions(); ++extIdx)
	{
		Operator_Ext_Mur_ABC* mur = dynamic_cast<Operator_Ext_Mur_ABC*>(m_level->m_op->GetExtension(extIdx));
		if (!mur) continue;
		if (mur->m_ny < 0 || mur->m_ny > 2 || mur->m_nyP < 0 || mur->m_nyP > 2 || mur->m_nyPP < 0 || mur->m_nyPP > 2)
			throw std::out_of_range("[openEMS Vulkan] Invalid Mur direction");

		uint32_t start_TS = 0;
		int maxDelay = -1;
		Operator_Ext_Excitation* excExt = m_level->m_op->GetExcitationExtension();
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
			if (maxDelay >= 0 && m_level->m_op->GetExcitationSignal())
			{
				start_TS = static_cast<uint32_t>(maxDelay + m_level->m_op->GetExcitationSignal()->GetLength() + 10);
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
				ptP.pos_idx = GetCheckedLinearIndex(mur->m_nyP, pos[0], pos[1], pos[2]);
				ptP.shift_idx = GetCheckedLinearIndex(mur->m_nyP, pos_shift[0], pos_shift[1], pos_shift[2]);
				ptP.coeff = static_cast<float>(mur->m_Mur_Coeff_nyP(i, j));
				ptP.start_TS = start_TS;
				hostPoints.push_back(ptP);

				// Component nyPP
				GpuMurPoint ptPP;
				ptPP.pos_idx = GetCheckedLinearIndex(mur->m_nyPP, pos[0], pos[1], pos[2]);
				ptPP.shift_idx = GetCheckedLinearIndex(mur->m_nyPP, pos_shift[0], pos_shift[1], pos_shift[2]);
				ptPP.coeff = static_cast<float>(mur->m_Mur_Coeff_nyPP(i, j));
				ptPP.start_TS = start_TS;
				hostPoints.push_back(ptPP);
			}
		}

		face.count = static_cast<uint32_t>(hostPoints.size()) - face.offset;
		m_level->m_murFaces.push_back(face);
	}

	m_level->m_totalMurPoints = static_cast<uint32_t>(hostPoints.size());
	if (m_level->m_totalMurPoints == 0) return true;

	std::cout << "[openEMS Vulkan] Initialized on-device Mur ABC: "
	          << m_level->m_totalMurPoints << " boundary points across "
	          << m_level->m_murFaces.size() << " face(s)." << std::endl;

	VkDeviceSize paramsBytes = m_level->m_totalMurPoints * sizeof(GpuMurPoint);
	VkDeviceSize storeBytes  = m_level->m_totalMurPoints * sizeof(float);

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	if (!CreateBuffer(paramsBytes, storageUsage, devLocal, m_level->m_bufMurParams)) return false;
	if (!CreateBuffer(storeBytes, storageUsage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, devLocal, m_level->m_bufMurStore)) return false;

	// Staging buffer to upload params
	VulkanBuffer stageParams;
	if (!CreateBuffer(paramsBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stageParams))
		return false;

	memcpy(stageParams.mapped, hostPoints.data(), paramsBytes);

	vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
	vkResetFences(m_device, 1, &m_level->m_fence);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);

	VkBufferCopy copyRegion = { 0, 0, paramsBytes };
	vkCmdCopyBuffer(m_level->m_cmdBuffer, stageParams.buffer, m_level->m_bufMurParams.buffer, 1, &copyRegion);

	// Zero store buffer
	vkCmdFillBuffer(m_level->m_cmdBuffer, m_level->m_bufMurStore.buffer, 0, VK_WHOLE_SIZE, 0);

	vkEndCommandBuffer(m_level->m_cmdBuffer);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
	vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
	vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);

	DestroyBuffer(stageParams);

	// Allocate and configure descriptor set for Mur ABC
	VkDescriptorSetAllocateInfo dsAlloc = {};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = m_descPool;
	dsAlloc.descriptorSetCount = 1;
	dsAlloc.pSetLayouts = &m_descLayoutMur;

	if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_level->m_descSetMur) != VK_SUCCESS) return false;

	VkDescriptorBufferInfo murBufInfos[3] = {
		{ m_level->m_bufMurParams.buffer, 0, paramsBytes },
		{ m_level->m_bufVolt.buffer, 0, m_level->m_bufVolt.size },
		{ m_level->m_bufMurStore.buffer, 0, storeBytes }
	};
	VkWriteDescriptorSet murWrites[3] = {};
	for (uint32_t i = 0; i < 3; ++i)
	{
		murWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		murWrites[i].dstSet = m_level->m_descSetMur;
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
	m_level->m_tfsfVoltFaces.clear();
	m_level->m_tfsfVoltPoints.clear();
	m_level->m_tfsfCurrFaces.clear();
	m_level->m_tfsfCurrPoints.clear();
	m_level->m_tfsfSigLength = 0;
	m_level->m_tfsfPeriod = 0;

	if (!m_level->m_op) return true;

	Operator_Ext_TFSF* tfsf = nullptr;
	for (size_t i = 0; i < m_level->m_op->GetNumberOfExtentions(); ++i)
	{
		tfsf = dynamic_cast<Operator_Ext_TFSF*>(m_level->m_op->GetExtension(i));
		if (tfsf) break;
	}
	if (!tfsf || !tfsf->m_Exc) return true;

	m_level->m_tfsfSigLength = tfsf->m_Exc->GetLength();
	if (tfsf->m_Exc->GetTimestep() > 0.0)
	{
		m_level->m_tfsfPeriod = static_cast<int32_t>(tfsf->m_Exc->GetSignalPeriod() / tfsf->m_Exc->GetTimestep());
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
			face.offset = static_cast<uint32_t>(m_level->m_tfsfVoltPoints.size());

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
							const uint32_t linIdx = GetCheckedLinearIndex(dir, pos[0], pos[1], pos[2]);

							GpuTfsfPoint pt;
							pt.pos_idx = linIdx;
							pt.delay = delay;
							pt.w0 = (1.0f - delta) * amp;
							pt.w1 = delta * amp;
							m_level->m_tfsfVoltPoints.push_back(pt);
						}
						++ui_pos;
					}
				}
			}

			face.count = static_cast<uint32_t>(m_level->m_tfsfVoltPoints.size()) - face.offset;
			if (face.count > 0)
			{
				m_level->m_tfsfVoltFaces.push_back(face);
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
			face.offset = static_cast<uint32_t>(m_level->m_tfsfCurrPoints.size());

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
							const uint32_t linIdx = GetCheckedLinearIndex(dir, pos[0], pos[1], pos[2]);

							GpuTfsfPoint pt;
							pt.pos_idx = linIdx;
							pt.delay = delay;
							pt.w0 = (1.0f - delta) * amp;
							pt.w1 = delta * amp;
							m_level->m_tfsfCurrPoints.push_back(pt);
						}
						++ui_pos;
					}
				}
			}

			face.count = static_cast<uint32_t>(m_level->m_tfsfCurrPoints.size()) - face.offset;
			if (face.count > 0)
			{
				m_level->m_tfsfCurrFaces.push_back(face);
			}
		}
	}

	if (m_level->m_tfsfVoltPoints.empty() && m_level->m_tfsfCurrPoints.empty())
		return true;

	std::cout << "[openEMS Vulkan] Initialized on-device TFSF: "
	          << m_level->m_tfsfVoltPoints.size() << " voltage points across " << m_level->m_tfsfVoltFaces.size() << " face(s), "
	          << m_level->m_tfsfCurrPoints.size() << " current points across " << m_level->m_tfsfCurrFaces.size() << " face(s)." << std::endl;

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	// Upload volt params
	if (!m_level->m_tfsfVoltPoints.empty())
	{
		VkDeviceSize bytes = m_level->m_tfsfVoltPoints.size() * sizeof(GpuTfsfPoint);
		if (!CreateBuffer(bytes, storageUsage, devLocal, m_level->m_bufTfsfVoltParams)) return false;

		VulkanBuffer stage;
		if (!CreateBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;
		memcpy(stage.mapped, m_level->m_tfsfVoltPoints.data(), bytes);

		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_level->m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, bytes };
		vkCmdCopyBuffer(m_level->m_cmdBuffer, stage.buffer, m_level->m_bufTfsfVoltParams.buffer, 1, &copyRegion);
		vkEndCommandBuffer(m_level->m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);
	}

	// Upload curr params
	if (!m_level->m_tfsfCurrPoints.empty())
	{
		VkDeviceSize bytes = m_level->m_tfsfCurrPoints.size() * sizeof(GpuTfsfPoint);
		if (!CreateBuffer(bytes, storageUsage, devLocal, m_level->m_bufTfsfCurrParams)) return false;

		VulkanBuffer stage;
		if (!CreateBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;
		memcpy(stage.mapped, m_level->m_tfsfCurrPoints.data(), bytes);

		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_level->m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, bytes };
		vkCmdCopyBuffer(m_level->m_cmdBuffer, stage.buffer, m_level->m_bufTfsfCurrParams.buffer, 1, &copyRegion);
		vkEndCommandBuffer(m_level->m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);
	}

	// Upload signals
	if (m_level->m_tfsfSigLength > 0)
	{
		VkDeviceSize sigBytes = m_level->m_tfsfSigLength * sizeof(float);
		if (!CreateBuffer(sigBytes, storageUsage, devLocal, m_level->m_bufTfsfCurrSignal)) return false;
		if (!CreateBuffer(sigBytes, storageUsage, devLocal, m_level->m_bufTfsfVoltSignal)) return false;

		std::vector<float> currSigHost(m_level->m_tfsfSigLength);
		std::vector<float> voltSigHost(m_level->m_tfsfSigLength);
		FDTD_FLOAT* rawCurrSig = tfsf->m_Exc->GetCurrentSignal();
		FDTD_FLOAT* rawVoltSig = tfsf->m_Exc->GetVoltageSignal();
		for (uint32_t i = 0; i < m_level->m_tfsfSigLength; ++i)
		{
			currSigHost[i] = rawCurrSig ? static_cast<float>(rawCurrSig[i]) : 0.0f;
			voltSigHost[i] = rawVoltSig ? static_cast<float>(rawVoltSig[i]) : 0.0f;
		}

		VulkanBuffer stage;
		if (!CreateBuffer(sigBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;

		// Upload currSigHost to m_level->m_bufTfsfCurrSignal
		memcpy(stage.mapped, currSigHost.data(), sigBytes);

		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_level->m_fence);
		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, sigBytes };
		vkCmdCopyBuffer(m_level->m_cmdBuffer, stage.buffer, m_level->m_bufTfsfCurrSignal.buffer, 1, &copyRegion);
		vkEndCommandBuffer(m_level->m_cmdBuffer);
		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);

		// Upload voltSigHost to m_level->m_bufTfsfVoltSignal
		memcpy(stage.mapped, voltSigHost.data(), sigBytes);

		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_level->m_fence);
		vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);
		vkCmdCopyBuffer(m_level->m_cmdBuffer, stage.buffer, m_level->m_bufTfsfVoltSignal.buffer, 1, &copyRegion);
		vkEndCommandBuffer(m_level->m_cmdBuffer);
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);
	}

	// Allocate and update descriptor sets
	if (m_descLayoutTfsf != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE)
	{
		VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		dsAlloc.descriptorPool = m_descPool;
		dsAlloc.descriptorSetCount = 1;
		dsAlloc.pSetLayouts = &m_descLayoutTfsf;

		if (!m_level->m_tfsfVoltPoints.empty())
		{
			if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_level->m_descSetTfsfVolt) != VK_SUCCESS)
				return false;

			VkDescriptorBufferInfo b0 = { m_level->m_bufVolt.buffer, 0, m_level->m_bufVolt.size };
			VkDescriptorBufferInfo b1 = { m_level->m_bufTfsfVoltParams.buffer, 0, m_level->m_bufTfsfVoltParams.size };
			VkDescriptorBufferInfo b2 = { m_level->m_bufTfsfCurrSignal.buffer, 0, m_level->m_bufTfsfCurrSignal.size };

			VkWriteDescriptorSet writes[3] = {};
			for (int w = 0; w < 3; ++w)
			{
				writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[w].dstSet = m_level->m_descSetTfsfVolt;
				writes[w].dstBinding = w;
				writes[w].descriptorCount = 1;
				writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			}
			writes[0].pBufferInfo = &b0;
			writes[1].pBufferInfo = &b1;
			writes[2].pBufferInfo = &b2;
			vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);
		}

		if (!m_level->m_tfsfCurrPoints.empty())
		{
			if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_level->m_descSetTfsfCurr) != VK_SUCCESS)
				return false;

			VkDescriptorBufferInfo b0 = { m_level->m_bufCurr.buffer, 0, m_level->m_bufCurr.size };
			VkDescriptorBufferInfo b1 = { m_level->m_bufTfsfCurrParams.buffer, 0, m_level->m_bufTfsfCurrParams.size };
			VkDescriptorBufferInfo b2 = { m_level->m_bufTfsfVoltSignal.buffer, 0, m_level->m_bufTfsfVoltSignal.size };

			VkWriteDescriptorSet writes[3] = {};
			for (int w = 0; w < 3; ++w)
			{
				writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[w].dstSet = m_level->m_descSetTfsfCurr;
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
	if (!m_level->m_op) return true;

	Operator_Ext_LumpedRLC* rlc = nullptr;
	for (size_t i = 0; i < m_level->m_op->GetNumberOfExtentions(); ++i)
	{
		rlc = dynamic_cast<Operator_Ext_LumpedRLC*>(m_level->m_op->GetExtension(i));
		if (rlc) break;
	}
	if (!rlc || rlc->RLC_count == 0) return true;

	m_level->m_rlcCount = rlc->RLC_count;
	std::vector<GpuRlcParam> rlcParams(m_level->m_rlcCount);
	std::vector<GpuRlcState> rlcState(m_level->m_rlcCount);
	std::memset(rlcState.data(), 0, m_level->m_rlcCount * sizeof(GpuRlcState));

	for (uint32_t i = 0; i < m_level->m_rlcCount; ++i)
	{
		uint32_t dir = static_cast<uint32_t>(rlc->v_RLC_dir[i]);
		uint32_t x = rlc->v_RLC_pos[0][i];
		uint32_t y = rlc->v_RLC_pos[1][i];
		uint32_t z = rlc->v_RLC_pos[2][i];

		rlcParams[i].field_index = GetCheckedLinearIndex(dir, x, y, z);
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
	for (uint32_t i = 0; i < m_level->m_rlcCount; ++i)
	{
		for (uint32_t j = i + 1; j < m_level->m_rlcCount; ++j)
		{
			if (rlcParams[i].field_index == rlcParams[j].field_index)
			{
				rlcParams[i].pad0 = 0.0f;
				break;
			}
		}
	}

	std::cout << "[openEMS Vulkan] Initialized on-device Lumped RLC: " << m_level->m_rlcCount << " element(s)." << std::endl;

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	VkDeviceSize paramsBytes = m_level->m_rlcCount * sizeof(GpuRlcParam);
	VkDeviceSize stateBytes = m_level->m_rlcCount * sizeof(GpuRlcState);

	if (!CreateBuffer(paramsBytes, storageUsage, devLocal, m_level->m_bufRlcParams)) return false;
	if (!CreateBuffer(stateBytes, storageUsage, devLocal, m_level->m_bufRlcState)) return false;

	// Upload initial params
	{
		VulkanBuffer stage;
		if (!CreateBuffer(paramsBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;
		memcpy(stage.mapped, rlcParams.data(), paramsBytes);

		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_level->m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, paramsBytes };
		vkCmdCopyBuffer(m_level->m_cmdBuffer, stage.buffer, m_level->m_bufRlcParams.buffer, 1, &copyRegion);
		vkEndCommandBuffer(m_level->m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
		DestroyBuffer(stage);
	}

	// Upload initial state (zeros)
	{
		VulkanBuffer stage;
		if (!CreateBuffer(stateBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;
		memcpy(stage.mapped, rlcState.data(), stateBytes);

		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_level->m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, stateBytes };
		vkCmdCopyBuffer(m_level->m_cmdBuffer, stage.buffer, m_level->m_bufRlcState.buffer, 1, &copyRegion);
		vkEndCommandBuffer(m_level->m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
		DestroyBuffer(stage);
	}

	// Allocate and update descriptor set
	VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	allocInfo.descriptorPool = m_descPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &m_descLayoutRlc;
	if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_level->m_descSetRlc) != VK_SUCCESS)
		return false;

	VkDescriptorBufferInfo bInfoVolt = { m_level->m_bufVolt.buffer, 0, m_level->m_bufVolt.size };
	VkDescriptorBufferInfo bInfoParams = { m_level->m_bufRlcParams.buffer, 0, m_level->m_bufRlcParams.size };
	VkDescriptorBufferInfo bInfoState = { m_level->m_bufRlcState.buffer, 0, m_level->m_bufRlcState.size };

	VkWriteDescriptorSet writes[3] = {};
	writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].dstSet = m_level->m_descSetRlc;
	writes[0].dstBinding = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[0].pBufferInfo = &bInfoVolt;

	writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet = m_level->m_descSetRlc;
	writes[1].dstBinding = 1;
	writes[1].descriptorCount = 1;
	writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[1].pBufferInfo = &bInfoParams;

	writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[2].dstSet = m_level->m_descSetRlc;
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
	if (!m_level->m_op) return true;

	std::vector<Operator_Ext_Absorbing_BC*> abcExts;
	for (size_t i = 0; i < m_level->m_op->GetNumberOfExtentions(); ++i)
	{
		Operator_Ext_Absorbing_BC* abc = dynamic_cast<Operator_Ext_Absorbing_BC*>(m_level->m_op->GetExtension(i));
		if (abc)
			abcExts.push_back(abc);
	}
	if (abcExts.empty()) return true;

	std::vector<GpuAbcVoltPoint> hostVoltPoints;
	std::vector<GpuAbcCurrPoint> hostCurrPoints;
	m_level->m_abcVoltSheets.clear();
	m_level->m_abcCurrSheets.clear();

	for (Operator_Ext_Absorbing_BC* abc : abcExts)
	{
		TfsfFace voltSheet = { static_cast<uint32_t>(hostVoltPoints.size()), 0 };
		TfsfFace currSheet = { static_cast<uint32_t>(hostCurrPoints.size()), 0 };
		int ny = abc->m_ny;
		int nyP = abc->m_nyP;
		int nyPP = abc->m_nyPP;
		if (ny < 0 || ny > 2 || nyP < 0 || nyP > 2 || nyPP < 0 || nyPP > 2)
			throw std::out_of_range("[openEMS Vulkan] Invalid absorbing boundary direction");
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
				ptP.pos_idx = GetCheckedLinearIndex(nyP, pos[0], pos[1], pos[2]);
				ptP.shift_idx = GetCheckedLinearIndex(nyP, pos_shift[0], pos_shift[1], pos_shift[2]);
				ptP.k1 = static_cast<float>(abc->m_K1_nyP(i, j));
				ptP.pad = 0;
				hostVoltPoints.push_back(ptP);

				// Direction nyPP
				GpuAbcVoltPoint ptPP;
				ptPP.pos_idx = GetCheckedLinearIndex(nyPP, pos[0], pos[1], pos[2]);
				ptPP.shift_idx = GetCheckedLinearIndex(nyPP, pos_shift[0], pos_shift[1], pos_shift[2]);
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
						ptP.pos_idx = GetCheckedLinearIndex(nyP, pos[0], pos[1], pos[2]);
						ptP.shift_idx = GetCheckedLinearIndex(nyP, pos_shift[0], pos_shift[1], pos_shift[2]);
						ptP.k1 = static_cast<float>(abc->m_K1_nyP(i, j));
						ptP.k2 = static_cast<float>(abc->m_K2_nyP(i, j));
						hostCurrPoints.push_back(ptP);

						// Direction nyPP
						GpuAbcCurrPoint ptPP;
						ptPP.pos_idx = GetCheckedLinearIndex(nyPP, pos[0], pos[1], pos[2]);
						ptPP.shift_idx = GetCheckedLinearIndex(nyPP, pos_shift[0], pos_shift[1], pos_shift[2]);
						ptPP.k1 = static_cast<float>(abc->m_K1_nyPP(i, j));
						ptPP.k2 = static_cast<float>(abc->m_K2_nyPP(i, j));
						hostCurrPoints.push_back(ptPP);
					}
				}
			}
		}

		voltSheet.count = static_cast<uint32_t>(hostVoltPoints.size()) - voltSheet.offset;
		currSheet.count = static_cast<uint32_t>(hostCurrPoints.size()) - currSheet.offset;
		if (voltSheet.count > 0) m_level->m_abcVoltSheets.push_back(voltSheet);
		if (currSheet.count > 0) m_level->m_abcCurrSheets.push_back(currSheet);
	}

	m_level->m_abcVoltCount = static_cast<uint32_t>(hostVoltPoints.size());
	m_level->m_abcCurrCount = static_cast<uint32_t>(hostCurrPoints.size());

	std::cout << "[openEMS Vulkan] Initialized on-device Absorbing BC: "
	          << m_level->m_abcVoltCount << " voltage points, "
	          << m_level->m_abcCurrCount << " current points across "
	          << abcExts.size() << " sheet(s)." << std::endl;

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	if (m_level->m_abcVoltCount > 0)
	{
		VkDeviceSize paramBytes = m_level->m_abcVoltCount * sizeof(GpuAbcVoltPoint);
		VkDeviceSize storeBytes = m_level->m_abcVoltCount * sizeof(float);

		if (!CreateBuffer(paramBytes, storageUsage, devLocal, m_level->m_bufAbcVoltParams)) return false;
		if (!CreateBuffer(storeBytes, storageUsage, devLocal, m_level->m_bufAbcVoltStore)) return false;

		VulkanBuffer stage;
		if (!CreateBuffer(paramBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;
		std::memcpy(stage.mapped, hostVoltPoints.data(), paramBytes);

		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_level->m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, paramBytes };
		vkCmdCopyBuffer(m_level->m_cmdBuffer, stage.buffer, m_level->m_bufAbcVoltParams.buffer, 1, &copyRegion);
		vkCmdFillBuffer(m_level->m_cmdBuffer, m_level->m_bufAbcVoltStore.buffer, 0, storeBytes, 0);
		vkEndCommandBuffer(m_level->m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);

		if (m_descLayoutAbc != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE)
		{
			VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			dsAlloc.descriptorPool = m_descPool;
			dsAlloc.descriptorSetCount = 1;
			dsAlloc.pSetLayouts = &m_descLayoutAbc;
			if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_level->m_descSetAbcVolt) != VK_SUCCESS)
				return false;

			VkDescriptorBufferInfo b0 = { m_level->m_bufAbcVoltParams.buffer, 0, m_level->m_bufAbcVoltParams.size };
			VkDescriptorBufferInfo b1 = { m_level->m_bufVolt.buffer, 0, m_level->m_bufVolt.size };
			VkDescriptorBufferInfo b2 = { m_level->m_bufAbcVoltStore.buffer, 0, m_level->m_bufAbcVoltStore.size };

			VkWriteDescriptorSet writes[3] = {};
			for (int w = 0; w < 3; ++w)
			{
				writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[w].dstSet = m_level->m_descSetAbcVolt;
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

	if (m_level->m_abcCurrCount > 0)
	{
		VkDeviceSize paramBytes = m_level->m_abcCurrCount * sizeof(GpuAbcCurrPoint);
		VkDeviceSize storeBytes = m_level->m_abcCurrCount * sizeof(float);

		if (!CreateBuffer(paramBytes, storageUsage, devLocal, m_level->m_bufAbcCurrParams)) return false;
		if (!CreateBuffer(storeBytes, storageUsage, devLocal, m_level->m_bufAbcCurrStore)) return false;

		VulkanBuffer stage;
		if (!CreateBuffer(paramBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;
		std::memcpy(stage.mapped, hostCurrPoints.data(), paramBytes);

		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_level->m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, paramBytes };
		vkCmdCopyBuffer(m_level->m_cmdBuffer, stage.buffer, m_level->m_bufAbcCurrParams.buffer, 1, &copyRegion);
		vkCmdFillBuffer(m_level->m_cmdBuffer, m_level->m_bufAbcCurrStore.buffer, 0, storeBytes, 0);
		vkEndCommandBuffer(m_level->m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);

		if (m_descLayoutAbc != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE)
		{
			VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			dsAlloc.descriptorPool = m_descPool;
			dsAlloc.descriptorSetCount = 1;
			dsAlloc.pSetLayouts = &m_descLayoutAbc;
			if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_level->m_descSetAbcCurr) != VK_SUCCESS)
				return false;

			VkDescriptorBufferInfo b0 = { m_level->m_bufAbcCurrParams.buffer, 0, m_level->m_bufAbcCurrParams.size };
			VkDescriptorBufferInfo b1 = { m_level->m_bufCurr.buffer, 0, m_level->m_bufCurr.size };
			VkDescriptorBufferInfo b2 = { m_level->m_bufAbcCurrStore.buffer, 0, m_level->m_bufAbcCurrStore.size };

			VkWriteDescriptorSet writes[3] = {};
			for (int w = 0; w < 3; ++w)
			{
				writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[w].dstSet = m_level->m_descSetAbcCurr;
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
	if (!m_level->m_op) return true;

	std::vector<Operator_Ext_LorentzMaterial*> lorExts;
	for (size_t i = 0; i < m_level->m_op->GetNumberOfExtentions(); ++i)
	{
		Operator_Ext_LorentzMaterial* lor = dynamic_cast<Operator_Ext_LorentzMaterial*>(m_level->m_op->GetExtension(i));
		if (lor)
			lorExts.push_back(lor);
	}
	if (lorExts.empty()) return true;

	std::vector<GpuDispersivePoint> hostVoltPoints;
	std::vector<GpuDispersivePoint> hostCurrPoints;
	m_level->m_dispVoltPasses.clear();
	m_level->m_dispCurrPasses.clear();

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
							pt.pos_idx = GetCheckedLinearIndex(n, x, y, z);
							pt.int_coeff = v_int;
							pt.ext_coeff = v_ext;
							pt.lor_coeff = v_lor;
							hostVoltPoints.push_back(pt);
						}
					}
				}
				pass.count = static_cast<uint32_t>(hostVoltPoints.size()) - pass.offset;
				if (pass.count > 0) m_level->m_dispVoltPasses.push_back(pass);
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
							pt.pos_idx = GetCheckedLinearIndex(n, x, y, z);
							pt.int_coeff = i_int;
							pt.ext_coeff = i_ext;
							pt.lor_coeff = i_lor;
							hostCurrPoints.push_back(pt);
						}
					}
				}
				pass.count = static_cast<uint32_t>(hostCurrPoints.size()) - pass.offset;
				if (pass.count > 0) m_level->m_dispCurrPasses.push_back(pass);
			}
		}
	}

	m_level->m_dispVoltCount = static_cast<uint32_t>(hostVoltPoints.size());
	m_level->m_dispCurrCount = static_cast<uint32_t>(hostCurrPoints.size());

	if (m_level->m_dispVoltCount == 0 && m_level->m_dispCurrCount == 0) return true;

	std::cout << "[openEMS Vulkan] Initialized on-device Dispersive Media: "
	          << m_level->m_dispVoltCount << " voltage components across " << m_level->m_dispVoltPasses.size() << " pass(es), "
	          << m_level->m_dispCurrCount << " current components across " << m_level->m_dispCurrPasses.size() << " pass(es)." << std::endl;

	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	if (m_level->m_dispVoltCount > 0)
	{
		VkDeviceSize paramBytes = m_level->m_dispVoltCount * sizeof(GpuDispersivePoint);
		VkDeviceSize stateBytes = m_level->m_dispVoltCount * sizeof(GpuDispersiveState);

		if (!CreateBuffer(paramBytes, storageUsage, devLocal, m_level->m_bufDispVoltParams)) return false;
		if (!CreateBuffer(stateBytes, storageUsage, devLocal, m_level->m_bufDispVoltState)) return false;

		VulkanBuffer stage;
		if (!CreateBuffer(paramBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;

		memcpy(stage.mapped, hostVoltPoints.data(), paramBytes);

		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_level->m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, paramBytes };
		vkCmdCopyBuffer(m_level->m_cmdBuffer, stage.buffer, m_level->m_bufDispVoltParams.buffer, 1, &copyRegion);
		vkCmdFillBuffer(m_level->m_cmdBuffer, m_level->m_bufDispVoltState.buffer, 0, stateBytes, 0);
		vkEndCommandBuffer(m_level->m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);

		if (m_descLayoutDisp != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE)
		{
			VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			dsAlloc.descriptorPool = m_descPool;
			dsAlloc.descriptorSetCount = 1;
			dsAlloc.pSetLayouts = &m_descLayoutDisp;
			if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_level->m_descSetDispVolt) != VK_SUCCESS)
				return false;

			VkDescriptorBufferInfo b0 = { m_level->m_bufDispVoltParams.buffer, 0, m_level->m_bufDispVoltParams.size };
			VkDescriptorBufferInfo b1 = { m_level->m_bufVolt.buffer, 0, m_level->m_bufVolt.size };
			VkDescriptorBufferInfo b2 = { m_level->m_bufDispVoltState.buffer, 0, m_level->m_bufDispVoltState.size };

			VkWriteDescriptorSet writes[3] = {};
			for (int w = 0; w < 3; ++w)
			{
				writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[w].dstSet = m_level->m_descSetDispVolt;
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

	if (m_level->m_dispCurrCount > 0)
	{
		VkDeviceSize paramBytes = m_level->m_dispCurrCount * sizeof(GpuDispersivePoint);
		VkDeviceSize stateBytes = m_level->m_dispCurrCount * sizeof(GpuDispersiveState);

		if (!CreateBuffer(paramBytes, storageUsage, devLocal, m_level->m_bufDispCurrParams)) return false;
		if (!CreateBuffer(stateBytes, storageUsage, devLocal, m_level->m_bufDispCurrState)) return false;

		VulkanBuffer stage;
		if (!CreateBuffer(paramBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
			return false;

		memcpy(stage.mapped, hostCurrPoints.data(), paramBytes);

		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
		vkResetFences(m_device, 1, &m_level->m_fence);

		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);
		VkBufferCopy copyRegion = { 0, 0, paramBytes };
		vkCmdCopyBuffer(m_level->m_cmdBuffer, stage.buffer, m_level->m_bufDispCurrParams.buffer, 1, &copyRegion);
		vkCmdFillBuffer(m_level->m_cmdBuffer, m_level->m_bufDispCurrState.buffer, 0, stateBytes, 0);
		vkEndCommandBuffer(m_level->m_cmdBuffer);

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
		vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
		vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);

		DestroyBuffer(stage);

		if (m_descLayoutDisp != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE)
		{
			VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
			dsAlloc.descriptorPool = m_descPool;
			dsAlloc.descriptorSetCount = 1;
			dsAlloc.pSetLayouts = &m_descLayoutDisp;
			if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_level->m_descSetDispCurr) != VK_SUCCESS)
				return false;

			VkDescriptorBufferInfo b0 = { m_level->m_bufDispCurrParams.buffer, 0, m_level->m_bufDispCurrParams.size };
			VkDescriptorBufferInfo b1 = { m_level->m_bufCurr.buffer, 0, m_level->m_bufCurr.size };
			VkDescriptorBufferInfo b2 = { m_level->m_bufDispCurrState.buffer, 0, m_level->m_bufDispCurrState.size };

			VkWriteDescriptorSet writes[3] = {};
			for (int w = 0; w < 3; ++w)
			{
				writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writes[w].dstSet = m_level->m_descSetDispCurr;
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

bool EngineVulkan::AllocateDebyeBuffers()
{
#ifdef ENABLE_VULKAN
	std::vector<GpuDebyePoint> points;
	std::vector<GpuDebyePole> poles;
	for (size_t i = 0; m_level->m_op && i < m_level->m_op->GetNumberOfExtentions(); ++i)
	{
		auto* debye = dynamic_cast<Operator_Ext_DebyeMaterial*>(m_level->m_op->GetExtension(i));
		if (!debye || debye->m_PoleCount <= 0 || debye->m_LM_Count.empty()) continue;
		for (unsigned int j = 0; j < debye->m_LM_Count[0]; ++j)
			for (unsigned int n = 0; n < 3; ++n)
			{
				if (points.size() >= UINT32_MAX || poles.size() > UINT32_MAX - static_cast<uint32_t>(debye->m_PoleCount))
					return false;
				GpuDebyePoint point = {
					GetCheckedLinearIndex(n, debye->m_LM_pos[0][0][j],
					    debye->m_LM_pos[0][1][j], debye->m_LM_pos[0][2][j]),
					static_cast<uint32_t>(poles.size()), static_cast<uint32_t>(debye->m_PoleCount),
					debye->v_solve_ADE[n][j]
				};
				points.push_back(point);
				for (int o = 0; o < debye->m_PoleCount; ++o)
					poles.push_back({debye->v_relax_ADE[o][n][j], debye->v_drive_ADE[o][n][j]});
			}
	}
	if (points.empty()) return true;
	if (points.size() + poles.size() > UINT32_MAX) return false;
	m_level->m_debyeCount = static_cast<uint32_t>(points.size());
	std::vector<float> state(points.size() + poles.size(), 0.0f);
	if (!UploadStorageBuffer(points.data(), points.size() * sizeof(GpuDebyePoint), m_level->m_bufDebyeParams) ||
	    !UploadStorageBuffer(poles.data(), poles.size() * sizeof(GpuDebyePole), m_level->m_bufDebyePoles) ||
	    !UploadStorageBuffer(state.data(), state.size() * sizeof(float), m_level->m_bufDebyeState))
		return false;
	VkDescriptorSetAllocateInfo alloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	alloc.descriptorPool = m_descPool;
	alloc.descriptorSetCount = 1;
	alloc.pSetLayouts = &m_descLayoutDebye;
	if (vkAllocateDescriptorSets(m_device, &alloc, &m_level->m_descSetDebye) != VK_SUCCESS) return false;
	VulkanBuffer* buffers[4] = {&m_level->m_bufDebyeParams, &m_level->m_bufVolt,
	                           &m_level->m_bufDebyePoles, &m_level->m_bufDebyeState};
	VkDescriptorBufferInfo infos[4] = {};
	VkWriteDescriptorSet writes[4] = {};
	for (uint32_t i = 0; i < 4; ++i)
	{
		infos[i] = {buffers[i]->buffer, 0, buffers[i]->size};
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = m_level->m_descSetDebye;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo = &infos[i];
	}
	vkUpdateDescriptorSets(m_device, 4, writes, 0, nullptr);
	std::cout << "[openEMS Vulkan] Initialized on-device Debye: " << points.size()
	          << " voltage components, " << poles.size() << " pole states." << std::endl;
#endif
	return true;
}

bool EngineVulkan::AllocateCylinderBuffers()
{
#ifdef ENABLE_VULKAN
	if (!m_level->m_op) return true;

	Operator_Ext_Cylinder* cyl = nullptr;
	for (size_t i = 0; i < m_level->m_op->GetNumberOfExtentions(); ++i)
	{
		Operator_Ext_Cylinder* c = dynamic_cast<Operator_Ext_Cylinder*>(m_level->m_op->GetExtension(i));
		if (c)
		{
			cyl = c;
			break;
		}
	}

	if (!cyl)
	{
		m_level->m_hasCylinder = false;
		return true;
	}

	m_level->m_hasCylinder = true;
	m_level->m_cylClosedAlpha = cyl->CC_closedAlpha;
	m_level->m_cylR0Included = cyl->CC_R0_included;
	m_level->m_cylLastALine = (m_level->m_grid.dimY >= 2) ? (m_level->m_grid.dimY - 2) : 0;

	size_t numZ = m_level->m_grid.dimZ > 0 ? m_level->m_grid.dimZ : 1;
	std::vector<float> r0Data(numZ * 2, 0.0f);
	if (m_level->m_cylR0Included && cyl->vv_R0 && cyl->vi_R0)
	{
		for (size_t z = 0; z < m_level->m_grid.dimZ; ++z)
		{
			r0Data[2 * z]     = static_cast<float>(cyl->vv_R0[z]);
			r0Data[2 * z + 1] = static_cast<float>(cyl->vi_R0[z]);
		}
	}

	VkDeviceSize r0Bytes = r0Data.size() * sizeof(float);
	VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags devLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	if (!CreateBuffer(r0Bytes, storageUsage, devLocal, m_level->m_bufCylR0)) return false;

	VulkanBuffer stage;
	if (!CreateBuffer(r0Bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage))
		return false;

	memcpy(stage.mapped, r0Data.data(), r0Bytes);

	vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
	vkResetFences(m_device, 1, &m_level->m_fence);

	VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo);

	VkBufferCopy copyRegion = { 0, 0, r0Bytes };
	vkCmdCopyBuffer(m_level->m_cmdBuffer, stage.buffer, m_level->m_bufCylR0.buffer, 1, &copyRegion);

	VkMemoryBarrier memBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
	memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(m_level->m_cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     0, 1, &memBarrier, 0, nullptr, 0, nullptr);

	vkEndCommandBuffer(m_level->m_cmdBuffer);

	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &m_level->m_cmdBuffer;
	vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_level->m_fence);
	vkWaitForFences(m_device, 1, &m_level->m_fence, VK_TRUE, UINT64_MAX);
	DestroyBuffer(stage);

	if (m_descLayoutCyl != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE)
	{
		VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		dsAlloc.descriptorPool = m_descPool;
		dsAlloc.descriptorSetCount = 1;
		dsAlloc.pSetLayouts = &m_descLayoutCyl;
		if (vkAllocateDescriptorSets(m_device, &dsAlloc, &m_level->m_descSetCyl) != VK_SUCCESS)
			return false;

		VkDescriptorBufferInfo b0 = { m_level->m_bufVolt.buffer, 0, m_level->m_bufVolt.size };
		VkDescriptorBufferInfo b1 = { m_level->m_bufCurr.buffer, 0, m_level->m_bufCurr.size };
		VkDescriptorBufferInfo b2 = { m_level->m_bufCylR0.buffer, 0, m_level->m_bufCylR0.size };

		VkWriteDescriptorSet writes[3] = {};
		for (int w = 0; w < 3; ++w)
		{
			writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writes[w].dstSet = m_level->m_descSetCyl;
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
	          << m_level->m_cylClosedAlpha << ", R0Included=" << m_level->m_cylR0Included << ")" << std::endl;
	return true;
#else
	return true;
#endif
}

bool EngineVulkan::AllocateMultigridBuffers()
{
	if (!m_level->m_multigridOp || !m_level->m_innerGrid)
		return true;

	const GridDimensions& child = m_level->m_innerGrid->m_level->m_grid;
	const uint32_t split = m_level->m_multigridOp->GetSplitPos();
	VkPhysicalDeviceProperties limits = {};
	vkGetPhysicalDeviceProperties(m_physicalDevice, &limits);
	if (m_level->m_grid.dimX < 8u || split < 4u || split > m_level->m_grid.dimX - 4u || split != child.dimX ||
	    child.dimZ != m_level->m_grid.dimZ || m_level->m_grid.dimY < 3u ||
	    child.dimY != m_level->m_grid.dimY / 2u + 1u ||
	    split - 1u > limits.limits.maxComputeWorkGroupCount[2])
	{
		std::cerr << "[openEMS Vulkan] Invalid cylindrical multigrid dimensions or split metadata." << std::endl;
		return false;
	}

	std::vector<GpuMultigridInterpolation> table(m_level->m_grid.dimY);
	for (uint32_t a = 0; a < m_level->m_grid.dimY; ++a)
	{
		for (uint32_t kind = 0; kind < 2u; ++kind)
		{
			const uint32_t vslot = kind;
			const uint32_t islot = 2u + kind;
			if (!m_level->m_multigridOp->m_interpol_pos_v_2p[kind] ||
			    !m_level->m_multigridOp->m_interpol_pos_v_2pp[kind] ||
			    !m_level->m_multigridOp->f4_interpol_v_2p[kind] ||
			    !m_level->m_multigridOp->f4_interpol_v_2pp[kind] ||
			    !m_level->m_multigridOp->m_interpol_pos_i_2p[kind] ||
			    !m_level->m_multigridOp->m_interpol_pos_i_2pp[kind] ||
			    !m_level->m_multigridOp->f4_interpol_i_2p[kind] ||
			    !m_level->m_multigridOp->f4_interpol_i_2pp[kind])
			{
				std::cerr << "[openEMS Vulkan] Missing cylindrical multigrid interpolation table." << std::endl;
				return false;
			}

			table[a].posP[vslot] = m_level->m_multigridOp->m_interpol_pos_v_2p[kind][a];
			table[a].posPP[vslot] = m_level->m_multigridOp->m_interpol_pos_v_2pp[kind][a];
			table[a].coeffP[vslot] = m_level->m_multigridOp->f4_interpol_v_2p[kind][a].f[0];
			table[a].coeffPP[vslot] = m_level->m_multigridOp->f4_interpol_v_2pp[kind][a].f[0];

			table[a].posP[islot] = m_level->m_multigridOp->m_interpol_pos_i_2p[kind][a];
			table[a].posPP[islot] = m_level->m_multigridOp->m_interpol_pos_i_2pp[kind][a];
			table[a].coeffP[islot] = m_level->m_multigridOp->f4_interpol_i_2p[kind][a].f[0];
			table[a].coeffPP[islot] = m_level->m_multigridOp->f4_interpol_i_2pp[kind][a].f[0];

			if (table[a].posP[vslot] >= child.dimY || table[a].posPP[vslot] >= child.dimY ||
			    table[a].posP[islot] >= child.dimY || table[a].posPP[islot] >= child.dimY)
			{
				std::cerr << "[openEMS Vulkan] Cylindrical multigrid interpolation index is outside the child grid." << std::endl;
				return false;
			}
			for (uint32_t slot : {vslot, islot})
			{
				if (!std::isfinite(table[a].coeffP[slot]) || !std::isfinite(table[a].coeffPP[slot]))
					return false;
			}
		}
	}

	const VkDeviceSize tableBytes = table.size() * sizeof(GpuMultigridInterpolation);
	if (!UploadStorageBuffer(table.data(), tableBytes, m_level->m_bufMultigridInterpolation))
		return false;

	auto spirv = CompileGLSLToSpirv(VulkanShaders::kShaderCylinderMultigrid,
	                                shaderc_glsl_compute_shader, "cylinder_multigrid.comp");
	VkShaderModule module = CreateShaderModule(spirv);
	if (module == VK_NULL_HANDLE)
		return false;

	std::vector<VkDescriptorSetLayoutBinding> bindings(5);
	for (uint32_t i = 0; i < bindings.size(); ++i)
	{
		bindings[i].binding = i;
		bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();
	if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_level->m_descLayoutMultigrid) != VK_SUCCESS)
	{
		vkDestroyShaderModule(m_device, module, nullptr);
		return false;
	}

	VkPushConstantRange pcRange = {};
	pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pcRange.size = 11u * sizeof(uint32_t);
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &m_level->m_descLayoutMultigrid;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pcRange;
	if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_level->m_pipelineLayoutMultigrid) != VK_SUCCESS)
	{
		vkDestroyShaderModule(m_device, module, nullptr);
		return false;
	}

	VkComputePipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	pipelineInfo.stage.module = module;
	pipelineInfo.stage.pName = "main";
	pipelineInfo.layout = m_level->m_pipelineLayoutMultigrid;
	const VkResult pipelineResult = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo,
	                                                        nullptr, &m_level->m_pipelineMultigrid);
	vkDestroyShaderModule(m_device, module, nullptr);
	if (pipelineResult != VK_SUCCESS)
		return false;

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = m_descPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &m_level->m_descLayoutMultigrid;
	if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_level->m_descSetMultigrid) != VK_SUCCESS)
		return false;

	VulkanBuffer* buffers[5] = {
		&m_level->m_bufVolt, &m_level->m_bufCurr, &m_level->m_innerGrid->m_level->m_bufVolt, &m_level->m_innerGrid->m_level->m_bufCurr,
		&m_level->m_bufMultigridInterpolation
	};
	VkDescriptorBufferInfo infos[5] = {};
	VkWriteDescriptorSet writes[5] = {};
	for (uint32_t i = 0; i < 5u; ++i)
	{
		infos[i].buffer = buffers[i]->buffer;
		infos[i].range = buffers[i]->size;
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = m_level->m_descSetMultigrid;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo = &infos[i];
	}
	vkUpdateDescriptorSets(m_device, 5u, writes, 0u, nullptr);
	return true;
}


#endif // ENABLE_VULKAN

void EngineVulkan::RegisterProbes(const ProcessingArray* pa)
{
	if (GetPendingProbeHistorySteps())
		throw std::logic_error("[openEMS Vulkan] Cannot register probes during history replay");
#ifdef ENABLE_VULKAN
	if (m_device && !ProfileOwner().Synchronize()) return;
#endif
	std::vector<ProbePoint> points;
	auto appendPoint = [&](unsigned int component, unsigned int x, unsigned int y, unsigned int z, uint32_t fieldType) {
		const size_t index = GetLinearIndex(component, x, y, z);
		if (index == static_cast<size_t>(-1) || index > UINT32_MAX)
			throw std::out_of_range("[openEMS Vulkan] Probe index is outside the field grid");
		points.push_back({static_cast<uint32_t>(index), fieldType});
	};
	auto validatePosition = [&](const unsigned int* position) {
		if (GetLinearIndex(0, position[0], position[1], position[2]) == static_cast<size_t>(-1))
			throw std::out_of_range("[openEMS Vulkan] Probe coordinate is outside the field grid");
	};

	for (size_t i = 0; pa && i < pa->GetNumberOfProcessings(); ++i)
	{
		Processing* proc = const_cast<ProcessingArray*>(pa)->GetProcessing(i);
		if (!proc || !proc->GetEnable()) continue;

		ProcessVoltage* pv = dynamic_cast<ProcessVoltage*>(proc);
		if (pv)
		{
			const unsigned int* start = pv->GetStartCoord();
			const unsigned int* stop = pv->GetStopCoord();
			validatePosition(start);
			validatePosition(stop);
			for (int n = 0; n < 3; ++n)
			{
				if (start[n] < stop[n])
				{
					unsigned int pos[3] = {start[0], start[1], start[2]};
					for (; pos[n] < stop[n]; ++pos[n])
					{
						appendPoint(n, pos[0], pos[1], pos[2], 0u);
					}
				}
				else if (start[n] > stop[n])
				{
					unsigned int pos[3] = {stop[0], stop[1], stop[2]};
					for (; pos[n] < start[n]; ++pos[n])
					{
						appendPoint(n, pos[0], pos[1], pos[2], 0u);
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
			validatePosition(start);
			validatePosition(stop);
			const bool* m_start_inside = pc->GetStartInside();
			const bool* m_stop_inside = pc->GetStopInside();
			int m_normDir = pc->GetNormalDir();
			if (m_normDir < 0 || m_normDir >= 3)
				throw std::out_of_range("[openEMS Vulkan] Invalid current probe normal direction");

			switch (m_normDir)
			{
			case 0:
				if (m_stop_inside[0] && m_start_inside[2])
					for (unsigned int k = start[1] + 1; k <= stop[1]; ++k)
						appendPoint(1, stop[0], k, start[2], 1u);
				if (m_stop_inside[0] && m_stop_inside[1])
					for (unsigned int k = start[2] + 1; k <= stop[2]; ++k)
						appendPoint(2, stop[0], stop[1], k, 1u);
				if (m_start_inside[0] && m_stop_inside[2])
					for (unsigned int k = start[1] + 1; k <= stop[1]; ++k)
						appendPoint(1, start[0], k, stop[2], 1u);
				if (m_start_inside[0] && m_start_inside[1])
					for (unsigned int k = start[2] + 1; k <= stop[2]; ++k)
						appendPoint(2, start[0], start[1], k, 1u);
				break;
			case 1:
				if (m_start_inside[0] && m_start_inside[1])
					for (unsigned int k = start[2] + 1; k <= stop[2]; ++k)
						appendPoint(2, start[0], start[1], k, 1u);
				if (m_stop_inside[1] && m_stop_inside[2])
					for (unsigned int k = start[0] + 1; k <= stop[0]; ++k)
						appendPoint(0, k, stop[1], stop[2], 1u);
				if (m_stop_inside[0] && m_stop_inside[1])
					for (unsigned int k = start[2] + 1; k <= stop[2]; ++k)
						appendPoint(2, stop[0], stop[1], k, 1u);
				if (m_start_inside[1] && m_start_inside[2])
					for (unsigned int k = start[0] + 1; k <= stop[0]; ++k)
						appendPoint(0, k, start[1], start[2], 1u);
				break;
			case 2:
				if (m_start_inside[1] && m_start_inside[2])
					for (unsigned int k = start[0] + 1; k <= stop[0]; ++k)
						appendPoint(0, k, start[1], start[2], 1u);
				if (m_stop_inside[0] && m_start_inside[2])
					for (unsigned int k = start[1] + 1; k <= stop[1]; ++k)
						appendPoint(1, stop[0], k, start[2], 1u);
				if (m_stop_inside[1] && m_stop_inside[2])
					for (unsigned int k = start[0] + 1; k <= stop[0]; ++k)
						appendPoint(0, k, stop[1], stop[2], 1u);
				if (m_start_inside[0] && m_stop_inside[2])
					for (unsigned int k = start[1] + 1; k <= stop[1]; ++k)
						appendPoint(1, start[0], k, stop[2], 1u);
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
				appendPoint(n, start[0], start[1], start[2], isCurr);
			}
			continue;
		}
	}

	if (pa && m_level->m_op)
	{
		for (size_t i = 0; i < m_level->m_op->GetNumberOfExtentions(); ++i)
		{
			Operator_Extension* ext = m_level->m_op->GetExtension(i);
			Operator_Ext_SteadyState* ss = dynamic_cast<Operator_Ext_SteadyState*>(ext);
			if (ss)
			{
				for (size_t n = 0; n < ss->GetNumberOfEProbes(); ++n)
				{
					unsigned int pos[3];
					ss->GetEProbePos(n, pos);
					unsigned int dir = ss->GetEProbeDir(n);
					appendPoint(dir, pos[0], pos[1], pos[2], 0u);
				}
				for (size_t n = 0; n < ss->GetNumberOfHProbes(); ++n)
				{
					unsigned int pos[3];
					ss->GetHProbePos(n, pos);
					unsigned int dir = ss->GetHProbeDir(n);
					appendPoint(dir, pos[0], pos[1], pos[2], 1u);
				}
			}
		}
	}

	// Sort and deduplicate probe points
	std::sort(points.begin(), points.end(), [](const ProbePoint& a, const ProbePoint& b) {
		if (a.field_type != b.field_type) return a.field_type < b.field_type;
		return a.linear_index < b.linear_index;
	});
	points.erase(std::unique(points.begin(), points.end(), [](const ProbePoint& a, const ProbePoint& b) {
		return a.field_type == b.field_type && a.linear_index == b.linear_index;
	}), points.end());
#ifdef ENABLE_VULKAN
	if (m_device && !AllocateProbeBuffers(points))
		throw std::runtime_error("[openEMS Vulkan] Probe buffer allocation failed");
#endif
	m_level->m_probePoints.swap(points);
	m_probeHistoryCount = m_probeHistoryCursor = 0;
	m_level->m_pa = pa;
	m_level->m_probesValid = false;

	if (!m_level->m_probePoints.empty())
	{
		std::cout << "[openEMS Vulkan] Registered " << m_level->m_probePoints.size()
		          << " active probe monitoring points for fast on-device extraction." << std::endl;
	}
}

#ifdef ENABLE_VULKAN
void EngineVulkan::RecordDispersivePre(VkCommandBuffer cmd, VkDescriptorSet descriptors,
                                      uint32_t count, const std::vector<TfsfFace>& passes)
{
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineDisp);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutDisp, 0, 1, &descriptors, 0, nullptr);
	const uint64_t groups = (uint64_t(count) + 255u) / 256u;
	if (ProfileOwner().m_fuseDispersivePre && groups <= m_level->m_dispersivePreMaxGroups) {
		// Pre mode only reads the field and writes disjoint per-point state.
		// Overlapping field positions across poles do not require separate passes.
		const uint32_t pc[] = {0u, count, 0u};
		vkCmdPushConstants(cmd, m_pipelineLayoutDisp, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
		Dispatch(cmd, static_cast<uint32_t>(groups), 1, 1);
	} else {
		for (const auto& pass : passes) {
			const uint32_t pc[] = {pass.offset, pass.count, 0u};
			vkCmdPushConstants(cmd, m_pipelineLayoutDisp, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
			Dispatch(cmd, (pass.count + 255u) / 256u, 1, 1);
		}
	}
}

void EngineVulkan::RecordDebyePhase(VkCommandBuffer cmd, uint32_t mode)
{
	if (!m_level->m_debyeCount) return;
	VkMemoryBarrier barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, barrier);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineDebye);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutDebye,
	                       0, 1, &m_level->m_descSetDebye, 0, nullptr);
	uint32_t pc[2] = {m_level->m_debyeCount, mode};
	vkCmdPushConstants(cmd, m_pipelineLayoutDebye, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
	Dispatch(cmd, (m_level->m_debyeCount + 255u) / 256u, 1, 1);
	MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, barrier);
}

void EngineVulkan::RecordVoltagePhase(VkCommandBuffer cmd, bool hasVoltExc, unsigned int timestep)
{
	uint32_t wgZ = (m_level->m_grid.dimZ + 31u) / 32u;
	uint32_t wgY = (m_level->m_grid.dimY + 3u) / 4u;

	struct YeePushConstants {
		uint32_t dimX;
		uint32_t dimY;
		uint32_t dimZ;
		uint32_t numCells;
		uint32_t startX;
		uint32_t countX;
	} pc = { m_level->m_grid.dimX, m_level->m_grid.dimY, m_level->m_grid.dimZ, m_level->m_grid.numCells,
	         m_level->m_activeXStart, m_level->m_grid.dimX - m_level->m_activeXStart };

	VkMemoryBarrier memBarrier = {};
	memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

	// Mur ABC Pre-Voltage Pass
	if (m_level->m_totalMurPoints > 0 && m_pipelineMurPre && m_level->m_descSetMur)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineMurPre);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutMur, 0, 1, &m_level->m_descSetMur, 0, nullptr);
		struct {
			uint32_t offset;
			uint32_t count;
			uint32_t current_TS;
			uint32_t pad;
		} murPC = { 0, m_level->m_totalMurPoints, timestep, 0 };
		vkCmdPushConstants(cmd, m_pipelineLayoutMur, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(murPC), &murPC);
		Dispatch(cmd, (m_level->m_totalMurPoints + 255) / 256, 1, 1);

		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
	}

	// Absorbing BC Pre-Voltage Pass
	if (m_level->m_abcVoltCount > 0 && m_pipelineAbcVolt && m_level->m_descSetAbcVolt)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineAbcVolt);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutAbc, 0, 1, &m_level->m_descSetAbcVolt, 0, nullptr);
		struct { uint32_t offset; uint32_t count; uint32_t mode; } abcPC = { 0, m_level->m_abcVoltCount, 0 };
		vkCmdPushConstants(cmd, m_pipelineLayoutAbc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(abcPC), &abcPC);
		Dispatch(cmd, (m_level->m_abcVoltCount + 255) / 256, 1, 1);

		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
	}

	// Lumped RLC Pre-Voltage Pass
	if (m_level->m_rlcCount > 0 && m_pipelineRlc && m_level->m_descSetRlc)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineRlc);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutRlc, 0, 1, &m_level->m_descSetRlc, 0, nullptr);
		struct {
			uint32_t count;
			uint32_t mode;
		} rlcPC = { m_level->m_rlcCount, 0 };
		vkCmdPushConstants(cmd, m_pipelineLayoutRlc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rlcPC), &rlcPC);
		Dispatch(cmd, (m_level->m_rlcCount + 63) / 64, 1, 1);

		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
	}

	RecordDebyePhase(cmd, 0u);

	// Dispersive Media Pre-Voltage Pass
	if (m_level->m_dispVoltCount > 0 && m_pipelineDisp && m_level->m_descSetDispVolt)
	{
		RecordDispersivePre(cmd, m_level->m_descSetDispVolt, m_level->m_dispVoltCount, m_level->m_dispVoltPasses);

		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
	}

	// UPML pre-updates run after lower-priority extensions, as on the CPU.
	// UPML Pre-Voltage Pass
	if (m_level->m_numUpmlCells > 0 && m_pipelineUpmlPre && m_level->m_descSetUpmlVolt)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineUpmlPre);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutUpml, 0, 1, &m_level->m_descSetUpmlVolt, 0, nullptr);
		vkCmdPushConstants(cmd, m_pipelineLayoutUpml, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_level->m_numUpmlCells), &m_level->m_numUpmlCells);
		Dispatch(cmd, (m_level->m_numUpmlCells + 255) / 256, 1, 1);

		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
	}

	// 2. Voltage update
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
	                  m_level->m_coefficients.palette ? ProfileOwner().m_pipelineVoltPalette : m_pipelineVolt);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutFields, 0, 1, &m_level->m_descSetFields, 0, nullptr);
	vkCmdPushConstants(cmd, m_pipelineLayoutFields, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
	Dispatch(cmd, wgZ, wgY, (pc.countX + 1u) / 2u, ProfileVoltage);

	// 3. UPML Post-Voltage Pass
	if (m_level->m_numUpmlCells > 0 && m_pipelineUpmlPost && m_level->m_descSetUpmlVolt)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineUpmlPost);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutUpml, 0, 1, &m_level->m_descSetUpmlVolt, 0, nullptr);
		vkCmdPushConstants(cmd, m_pipelineLayoutUpml, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_level->m_numUpmlCells), &m_level->m_numUpmlCells);
		Dispatch(cmd, (m_level->m_numUpmlCells + 255) / 256, 1, 1);
	}

	// Cylindrical Coordinates Post-Voltage Pass
	if (m_level->m_hasCylinder && m_level->m_cylClosedAlpha && m_pipelineCyl && m_level->m_descSetCyl)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineCyl);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutCyl, 0, 1, &m_level->m_descSetCyl, 0, nullptr);

		struct CylPC {
			uint32_t dimX;
			uint32_t dimY;
			uint32_t dimZ;
			uint32_t numCells;
			uint32_t last_A_Line;
			uint32_t hasR0;
			uint32_t mode;
		};

		if (m_level->m_cylR0Included)
		{
			MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

			CylPC r0PC = { m_level->m_grid.dimX, m_level->m_grid.dimY, m_level->m_grid.dimZ, m_level->m_grid.numCells, m_level->m_cylLastALine, 1u, 0u };
			vkCmdPushConstants(cmd, m_pipelineLayoutCyl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(r0PC), &r0PC);
			Dispatch(cmd, (m_level->m_grid.dimZ + 255) / 256, 1, 1);
		}

		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		CylPC wrapPC = { m_level->m_grid.dimX, m_level->m_grid.dimY, m_level->m_grid.dimZ, m_level->m_grid.numCells, m_level->m_cylLastALine, m_level->m_cylR0Included ? 1u : 0u, 1u };
		vkCmdPushConstants(cmd, m_pipelineLayoutCyl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(wrapPC), &wrapPC);
		Dispatch(cmd, (m_level->m_grid.dimX + 15) / 16, (m_level->m_grid.dimZ + 15) / 16, 1);
	}

	// TFSF has a higher priority than the boundary and lumped-element
	// extensions, so its entire post-update phase must run first.
	if (!m_level->m_tfsfVoltFaces.empty() && m_pipelineTfsf && m_level->m_descSetTfsfVolt)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineTfsf);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutTfsf, 0, 1, &m_level->m_descSetTfsfVolt, 0, nullptr);

		for (const auto& face : m_level->m_tfsfVoltFaces)
		{
			struct {
				uint32_t offset;
				uint32_t count;
				uint32_t current_TS;
				uint32_t sigLength;
				int32_t p;
			} tfsfPC = { face.offset, face.count, timestep, m_level->m_tfsfSigLength, m_level->m_tfsfPeriod };
			vkCmdPushConstants(cmd, m_pipelineLayoutTfsf, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tfsfPC), &tfsfPC);
			Dispatch(cmd, (face.count + 255) / 256, 1, 1);

			if (m_level->m_tfsfVoltFaces.size() > 1)
				MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
		}
	}

	RecordDebyePhase(cmd, 1u);

	// Equal-priority extensions are reversed by Engine's stable-sort plus
	// reverse sequence. Local sheets therefore precede Mur boundaries.
	if (m_level->m_abcVoltCount > 0 && m_pipelineAbcVolt && m_level->m_descSetAbcVolt)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineAbcVolt);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutAbc, 0, 1, &m_level->m_descSetAbcVolt, 0, nullptr);
		struct { uint32_t offset; uint32_t count; uint32_t mode; } abcPC = { 0, m_level->m_abcVoltCount, 1 };
		vkCmdPushConstants(cmd, m_pipelineLayoutAbc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(abcPC), &abcPC);
		Dispatch(cmd, (m_level->m_abcVoltCount + 255) / 256, 1, 1);
	}

	// Mur ABC Post-Voltage Pass (updates storeData += coeff * voltData[shift])
	if (m_level->m_totalMurPoints > 0 && m_pipelineMurPost && m_level->m_descSetMur)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineMurPost);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutMur, 0, 1, &m_level->m_descSetMur, 0, nullptr);
		struct {
			uint32_t offset;
			uint32_t count;
			uint32_t current_TS;
			uint32_t pad;
		} murPC = { 0, m_level->m_totalMurPoints, timestep, 0 };
		vkCmdPushConstants(cmd, m_pipelineLayoutMur, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(murPC), &murPC);
		Dispatch(cmd, (m_level->m_totalMurPoints + 255) / 256, 1, 1);
	}

	// Apply phases begin only after every extension's post-update phase
	// has completed, matching Engine::IterateTS.
	// Absorbing BC Apply Pass
	if (m_level->m_abcVoltCount > 0 && m_pipelineAbcVolt && m_level->m_descSetAbcVolt)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineAbcVolt);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutAbc, 0, 1, &m_level->m_descSetAbcVolt, 0, nullptr);
		for (auto it = m_level->m_abcVoltSheets.rbegin(); it != m_level->m_abcVoltSheets.rend(); ++it)
		{
			const TfsfFace& sheet = *it;
			struct { uint32_t offset; uint32_t count; uint32_t mode; } abcPC = { sheet.offset, sheet.count, 2 };
			vkCmdPushConstants(cmd, m_pipelineLayoutAbc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(abcPC), &abcPC);
			Dispatch(cmd, (sheet.count + 255) / 256, 1, 1);
			if (m_level->m_abcVoltSheets.size() > 1)
				MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
		}
	}

	// Lumped RLC Apply2Voltages. Calculation and field writes are split
	// so every element sees the same pre-RLC Yee voltage.
	if (m_level->m_rlcCount > 0 && m_pipelineRlc && m_level->m_descSetRlc)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineRlc);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutRlc, 0, 1, &m_level->m_descSetRlc, 0, nullptr);
		struct {
			uint32_t count;
			uint32_t mode;
		} rlcPC = { m_level->m_rlcCount, 1 };
		vkCmdPushConstants(cmd, m_pipelineLayoutRlc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rlcPC), &rlcPC);
		Dispatch(cmd, (m_level->m_rlcCount + 63) / 64, 1, 1);

		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
		rlcPC.mode = 2;
		vkCmdPushConstants(cmd, m_pipelineLayoutRlc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rlcPC), &rlcPC);
		Dispatch(cmd, (m_level->m_rlcCount + 63) / 64, 1, 1);
	}

	RecordDebyePhase(cmd, 2u);

	// Dispersive Media Apply-Voltage Pass
	if (m_level->m_dispVoltCount > 0 && m_pipelineDisp && m_level->m_descSetDispVolt)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineDisp);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutDisp, 0, 1, &m_level->m_descSetDispVolt, 0, nullptr);
		for (const auto& pass : m_level->m_dispVoltPasses)
		{
			struct { uint32_t offset; uint32_t count; uint32_t mode; } dispPC = { pass.offset, pass.count, 1 };
			vkCmdPushConstants(cmd, m_pipelineLayoutDisp, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dispPC), &dispPC);
			Dispatch(cmd, (pass.count + 255) / 256, 1, 1);
			if (m_level->m_dispVoltPasses.size() > 1)
				MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
		}
	}

	// Mur boundary extensions are the earliest default-priority
	// extensions added by openEMS, and therefore apply last among that
	// priority group after Engine reverses its sorted extension list.
	if (m_level->m_totalMurPoints > 0 && m_pipelineMurApply && m_level->m_descSetMur)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineMurApply);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutMur, 0, 1, &m_level->m_descSetMur, 0, nullptr);

		for (auto it = m_level->m_murFaces.rbegin(); it != m_level->m_murFaces.rend(); ++it)
		{
			const MurFace& face = *it;
			if (timestep < face.start_TS)
				continue;
			struct {
				uint32_t offset;
				uint32_t count;
				uint32_t current_TS;
				uint32_t pad;
			} murPC = { face.offset, face.count, timestep, 0 };
			vkCmdPushConstants(cmd, m_pipelineLayoutMur, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(murPC), &murPC);
			Dispatch(cmd, (face.count + 255) / 256, 1, 1);

			if (m_level->m_murFaces.size() > 1)
				MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
		}
	}

	// Excitation has the lowest extension priority and therefore applies
	// after boundary and lumped-element corrections.
	if (hasVoltExc && m_pipelineExc && m_level->m_descSetVoltExc)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		uint32_t excPC[2] = {static_cast<uint32_t>(m_level->m_voltExcPoints.size()), timestep};
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineExc);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutExc, 0, 1, &m_level->m_descSetVoltExc, 0, nullptr);
		vkCmdPushConstants(cmd, m_pipelineLayoutExc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(excPC), excPC);
		Dispatch(cmd, (excPC[0] + 63) / 64, 1, 1);
	}

	// Barrier between voltage and current
	MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
}

void EngineVulkan::RecordCurrentPhase(VkCommandBuffer cmd, bool hasCurrExc, unsigned int timestep)
{
	uint32_t wgZ = (m_level->m_grid.dimZ + 31u) / 32u;
	uint32_t wgY = (m_level->m_grid.dimY + 3u) / 4u;

	struct YeePushConstants {
		uint32_t dimX;
		uint32_t dimY;
		uint32_t dimZ;
		uint32_t numCells;
		uint32_t startX;
		uint32_t countX;
	} pc = { m_level->m_grid.dimX, m_level->m_grid.dimY, m_level->m_grid.dimZ, m_level->m_grid.numCells,
	         m_level->m_activeXStart, m_level->m_grid.dimX - 1u - m_level->m_activeXStart };

	VkMemoryBarrier memBarrier = {};
	memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

	// Absorbing BC Pre-Current Pass
	if (m_level->m_abcCurrCount > 0 && m_pipelineAbcCurr && m_level->m_descSetAbcCurr)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineAbcCurr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutAbc, 0, 1, &m_level->m_descSetAbcCurr, 0, nullptr);
		struct { uint32_t offset; uint32_t count; uint32_t mode; } abcPC = { 0, m_level->m_abcCurrCount, 0 };
		vkCmdPushConstants(cmd, m_pipelineLayoutAbc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(abcPC), &abcPC);
		Dispatch(cmd, (m_level->m_abcCurrCount + 255) / 256, 1, 1);

		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
	}

	// Dispersive Media Pre-Current Pass
	if (m_level->m_dispCurrCount > 0 && m_pipelineDisp && m_level->m_descSetDispCurr)
	{
		RecordDispersivePre(cmd, m_level->m_descSetDispCurr, m_level->m_dispCurrCount, m_level->m_dispCurrPasses);

		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
	}

	// UPML pre-updates run after lower-priority extensions, as on the CPU.
	// UPML Pre-Current Pass
	if (m_level->m_numUpmlCells > 0 && m_pipelineUpmlPre && m_level->m_descSetUpmlCurr)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineUpmlPre);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutUpml, 0, 1, &m_level->m_descSetUpmlCurr, 0, nullptr);
		vkCmdPushConstants(cmd, m_pipelineLayoutUpml, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_level->m_numUpmlCells), &m_level->m_numUpmlCells);
		Dispatch(cmd, (m_level->m_numUpmlCells + 255) / 256, 1, 1);

		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
	}

	// 6. Current update
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
	                  m_level->m_coefficients.palette ? ProfileOwner().m_pipelineCurrPalette : m_pipelineCurr);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutFields, 0, 1, &m_level->m_descSetFields, 0, nullptr);
	vkCmdPushConstants(cmd, m_pipelineLayoutFields, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
	Dispatch(cmd, wgZ, wgY, (pc.countX + 1u) / 2u, ProfileCurrent);

	// 7. UPML Post-Current Pass
	if (m_level->m_numUpmlCells > 0 && m_pipelineUpmlPost && m_level->m_descSetUpmlCurr)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineUpmlPost);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutUpml, 0, 1, &m_level->m_descSetUpmlCurr, 0, nullptr);
		vkCmdPushConstants(cmd, m_pipelineLayoutUpml, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_level->m_numUpmlCells), &m_level->m_numUpmlCells);
		Dispatch(cmd, (m_level->m_numUpmlCells + 255) / 256, 1, 1);
	}

	// Cylindrical Coordinates Post-Current Pass
	if (m_level->m_hasCylinder && m_level->m_cylClosedAlpha && m_pipelineCyl && m_level->m_descSetCyl)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineCyl);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutCyl, 0, 1, &m_level->m_descSetCyl, 0, nullptr);

		struct CylPC {
			uint32_t dimX;
			uint32_t dimY;
			uint32_t dimZ;
			uint32_t numCells;
			uint32_t last_A_Line;
			uint32_t hasR0;
			uint32_t mode;
		};

		CylPC currPC = { m_level->m_grid.dimX, m_level->m_grid.dimY, m_level->m_grid.dimZ, m_level->m_grid.numCells, m_level->m_cylLastALine, 0u, 2u };
		vkCmdPushConstants(cmd, m_pipelineLayoutCyl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(currPC), &currPC);
		Dispatch(cmd, (m_level->m_grid.dimX + 15) / 16, (m_level->m_grid.dimZ + 15) / 16, 1);
	}

	// TFSF post-current updates precede default-priority boundary sheets.
	if (!m_level->m_tfsfCurrFaces.empty() && m_pipelineTfsf && m_level->m_descSetTfsfCurr)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineTfsf);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutTfsf, 0, 1, &m_level->m_descSetTfsfCurr, 0, nullptr);

		for (const auto& face : m_level->m_tfsfCurrFaces)
		{
			struct {
				uint32_t offset;
				uint32_t count;
				uint32_t current_TS;
				uint32_t sigLength;
				int32_t p;
			} tfsfPC = { face.offset, face.count, timestep, m_level->m_tfsfSigLength, m_level->m_tfsfPeriod };
			vkCmdPushConstants(cmd, m_pipelineLayoutTfsf, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tfsfPC), &tfsfPC);
			Dispatch(cmd, (face.count + 255) / 256, 1, 1);

			if (m_level->m_tfsfCurrFaces.size() > 1)
				MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
		}
	}

	// Absorbing BC Post-Current Pass
	if (m_level->m_abcCurrCount > 0 && m_pipelineAbcCurr && m_level->m_descSetAbcCurr)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineAbcCurr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutAbc, 0, 1, &m_level->m_descSetAbcCurr, 0, nullptr);
		struct { uint32_t offset; uint32_t count; uint32_t mode; } abcPC = { 0, m_level->m_abcCurrCount, 1 };
		vkCmdPushConstants(cmd, m_pipelineLayoutAbc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(abcPC), &abcPC);
		Dispatch(cmd, (m_level->m_abcCurrCount + 255) / 256, 1, 1);
	}

	// Absorbing BC Apply Pass
	if (m_level->m_abcCurrCount > 0 && m_pipelineAbcCurr && m_level->m_descSetAbcCurr)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineAbcCurr);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutAbc, 0, 1, &m_level->m_descSetAbcCurr, 0, nullptr);
		for (auto it = m_level->m_abcCurrSheets.rbegin(); it != m_level->m_abcCurrSheets.rend(); ++it)
		{
			const TfsfFace& sheet = *it;
			struct { uint32_t offset; uint32_t count; uint32_t mode; } abcPC = { sheet.offset, sheet.count, 2 };
			vkCmdPushConstants(cmd, m_pipelineLayoutAbc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(abcPC), &abcPC);
			Dispatch(cmd, (sheet.count + 255) / 256, 1, 1);
			if (m_level->m_abcCurrSheets.size() > 1)
				MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
		}
	}

	// Dispersive Media Apply-Current Pass
	if (m_level->m_dispCurrCount > 0 && m_pipelineDisp && m_level->m_descSetDispCurr)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineDisp);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutDisp, 0, 1, &m_level->m_descSetDispCurr, 0, nullptr);
		for (const auto& pass : m_level->m_dispCurrPasses)
		{
			struct { uint32_t offset; uint32_t count; uint32_t mode; } dispPC = { pass.offset, pass.count, 1 };
			vkCmdPushConstants(cmd, m_pipelineLayoutDisp, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dispPC), &dispPC);
			Dispatch(cmd, (pass.count + 255) / 256, 1, 1);
			if (m_level->m_dispCurrPasses.size() > 1)
				MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
		}
	}

	// 8. Current excitation pass
	if (hasCurrExc && m_pipelineExc && m_level->m_descSetCurrExc)
	{
		MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);

		uint32_t excPC[2] = {static_cast<uint32_t>(m_level->m_currExcPoints.size()), timestep};
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineExc);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayoutExc, 0, 1, &m_level->m_descSetCurrExc, 0, nullptr);
		vkCmdPushConstants(cmd, m_pipelineLayoutExc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(excPC), excPC);
		Dispatch(cmd, (excPC[0] + 63) / 64, 1, 1);
	}

	// Make this phase visible to the child, interface transfer, and next step.
	MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, memBarrier);
}

void EngineVulkan::RecordMultigridTransfer(VkCommandBuffer cmd, uint32_t mode, uint32_t radialCount)
{
	if (!m_level->m_innerGrid || !m_level->m_pipelineMultigrid || !m_level->m_descSetMultigrid || radialCount == 0u)
		return;

	struct MultigridPushConstants {
		uint32_t parentDimX;
		uint32_t parentDimY;
		uint32_t parentDimZ;
		uint32_t parentCells;
		uint32_t childDimX;
		uint32_t childDimY;
		uint32_t childDimZ;
		uint32_t childCells;
		uint32_t radialStart;
		uint32_t radialCount;
		uint32_t mode;
	} pc = {
		m_level->m_grid.dimX, m_level->m_grid.dimY, m_level->m_grid.dimZ, m_level->m_grid.numCells,
		m_level->m_innerGrid->m_level->m_grid.dimX, m_level->m_innerGrid->m_level->m_grid.dimY, m_level->m_innerGrid->m_level->m_grid.dimZ,
		m_level->m_innerGrid->m_level->m_grid.numCells, 0u, radialCount, mode
	};

	if (mode == 0u)
	{
		pc.radialStart = m_level->m_multigridOp->GetSplitPos() - 1u;
		pc.radialCount = 1u;
	}
	else if (mode == 1u)
	{
		pc.radialStart = m_level->m_multigridOp->GetSplitPos() - 2u;
		pc.radialCount = 1u;
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_level->m_pipelineMultigrid);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_level->m_pipelineLayoutMultigrid,
	                        0, 1, &m_level->m_descSetMultigrid, 0, nullptr);
	vkCmdPushConstants(cmd, m_level->m_pipelineLayoutMultigrid, VK_SHADER_STAGE_COMPUTE_BIT,
	                   0, sizeof(pc), &pc);
	if (mode == 0u)
	{
		const uint32_t childAlphaCount = m_level->m_grid.dimY / 2u;
		Dispatch(cmd, (m_level->m_innerGrid->m_level->m_grid.dimZ + 31u) / 32u,
		              (childAlphaCount + 3u) / 4u, 1u, ProfileMultigrid);
	}
	else
	{
		Dispatch(cmd, (m_level->m_grid.dimZ + 31u) / 32u,
		              (m_level->m_grid.dimY + 3u) / 4u, pc.radialCount, ProfileMultigrid);
	}

	VkMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	MemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, barrier);
}

void EngineVulkan::RecordVoltageHierarchy(VkCommandBuffer cmd, unsigned int timestep)
{
	const bool hasVoltExc = !m_level->m_voltExcPoints.empty();
	RecordVoltagePhase(cmd, hasVoltExc, timestep);
	if (m_level->m_innerGrid)
	{
		m_level->m_innerGrid->RecordVoltageHierarchy(cmd, timestep);
		RecordMultigridTransfer(cmd, 0u, 1u);
	}
}

void EngineVulkan::RecordCurrentHierarchy(VkCommandBuffer cmd, unsigned int timestep)
{
	const bool hasCurrExc = !m_level->m_currExcPoints.empty();
	RecordCurrentPhase(cmd, hasCurrExc, timestep);
	if (m_level->m_innerGrid)
	{
		m_level->m_innerGrid->RecordCurrentHierarchy(cmd, timestep);
		RecordMultigridTransfer(cmd, 1u, 1u);
	}
}

void EngineVulkan::RecordProjectionHierarchy(VkCommandBuffer cmd)
{
	if (!m_level->m_innerGrid)
		return;
	m_level->m_innerGrid->RecordProjectionHierarchy(cmd);
	const uint32_t split = m_level->m_multigridOp->GetSplitPos();
	RecordMultigridTransfer(cmd, 2u, split - 1u);
	RecordMultigridTransfer(cmd, 3u, split - 2u);
}

bool EngineVulkan::SyncHierarchyToDevice()
{
	if (!SyncFieldsToDevice())
		return false;
	return !m_level->m_innerGrid || m_level->m_innerGrid->SyncHierarchyToDevice();
}

void EngineVulkan::MarkHierarchyHostInvalid()
{
	m_level->m_hostFieldsValid = false;
	m_level->m_probesValid = m_level->m_energyValid = false;
	if (m_level->m_innerGrid)
		m_level->m_innerGrid->MarkHierarchyHostInvalid();
}

void EngineVulkan::SetHierarchyTimestep(unsigned int ts)
{
	m_level->m_numTS = ts;
	Engine* cpuEngine = m_level->m_op ? m_level->m_op->GetEngine() : nullptr;
	if (cpuEngine)
		cpuEngine->SetNumberOfTimesteps(ts);
	if (m_level->m_innerGrid)
		m_level->m_innerGrid->SetHierarchyTimestep(ts);
}

void EngineVulkan::SetHierarchyHostTimestep(unsigned int ts)
{
	Engine* cpuEngine = m_level->m_op ? m_level->m_op->GetEngine() : nullptr;
	if (cpuEngine) cpuEngine->SetNumberOfTimesteps(ts);
	if (m_level->m_innerGrid) m_level->m_innerGrid->SetHierarchyHostTimestep(ts);
}
#endif

bool EngineVulkan::IterateTS(unsigned int iterTS)
{
#ifdef ENABLE_VULKAN
	if (ProfileOwner().m_runtimeFailed) return false;
#endif
	if (iterTS == 0u) return IterateTSImpl(0, false);
	if (GetPendingProbeHistorySteps())
	{
		if (iterTS > GetPendingProbeHistorySteps()) return false;
		m_probeHistoryCursor += iterTS;
#ifdef ENABLE_VULKAN
		SetHierarchyHostTimestep(m_probeHistoryStart + m_probeHistoryCursor);
#endif
		return true;
	}
	m_probeHistoryCount = m_probeHistoryCursor = 0;
	return IterateTSImpl(iterTS, false);
}

unsigned int EngineVulkan::GetPendingProbeHistorySteps() const
{
	return m_probeHistoryCount - m_probeHistoryCursor;
}

bool EngineVulkan::BeginProbeHistory(unsigned int steps)
{
#ifdef ENABLE_VULKAN
	if (!m_device || !m_readbackOptimizations || m_level->m_probePoints.empty() ||
	    GetPendingProbeHistorySteps() || steps < 2 || steps > m_batchSize ||
	    steps > std::numeric_limits<unsigned int>::max() - m_level->m_numTS) return false;
	if (!Synchronize()) return false;
	if (steps > m_level->m_probeCapacity && !AllocateProbeBuffers(m_level->m_probePoints, steps)) return false;
	const unsigned int start = m_level->m_numTS;
	if (!IterateTSImpl(steps, true)) return false;
	m_probeHistoryStart = start;
	m_probeHistoryCount = steps;
	m_probeHistoryCursor = 0;
	SetHierarchyHostTimestep(start);
	return true;
#else
	(void)steps;
	return false;
#endif
}

bool EngineVulkan::IterateTSImpl(unsigned int iterTS, bool history)
{
	if (iterTS > std::numeric_limits<unsigned int>::max() - m_level->m_numTS)
		return false;
#ifdef ENABLE_VULKAN
	if (!m_device || ProfileOwner().m_runtimeFailed) return false;
	if (iterTS == 0u) return true;
	if (!SyncHierarchyToDevice()) return false;
	for (unsigned int done = 0; done < iterTS;)
	{
		if (!WaitForFence()) return false;
		const unsigned int count = std::min(m_batchSize, iterTS - done);
		const unsigned int firstTS = m_level->m_numTS;
		PrepareFDPhases(firstTS + 1u, count);
		VkCommandBufferBeginInfo begin = {};
		begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		if (vkBeginCommandBuffer(m_level->m_cmdBuffer, &begin) != VK_SUCCESS)
		{
			ProfileOwner().m_runtimeFailed = true;
			return false;
		}
		{
			ProfileTimer timer(m_profileEnabled ? &m_profile.recordSeconds : nullptr);
			BeginProfileCommands(m_level->m_cmdBuffer);
			const uint32_t batchSpan = BeginGpuSpan(m_level->m_cmdBuffer, ProfileBatch);
			// Cover initial uploads and previous submissions without a host field copy.
			VkMemoryBarrier barrier = {};
			barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			MemoryBarrier(m_level->m_cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, barrier);
			bool fdTimingSampled = false;
			for (unsigned int step = 0; step < count; ++step)
			{
				m_sampleDispatches = m_profileEnabled && step == 0u;
				if (m_sampleDispatches) ++m_profile.sampledSteps;
				RecordVoltageHierarchy(m_level->m_cmdBuffer, firstTS + step);
				RecordCurrentHierarchy(m_level->m_cmdBuffer, firstTS + step);
				if (!m_fdDumps.empty())
					fdTimingSampled = RecordFieldDumps(m_level->m_cmdBuffer, firstTS + step + 1u, step, !fdTimingSampled) || fdTimingSampled;
				if (history)
				{
					RecordProjectionHierarchy(m_level->m_cmdBuffer);
					RecordProbeGather(m_level->m_cmdBuffer, step);
				}
			}
			m_sampleDispatches = m_profileEnabled;
			if (!history && done + count == iterTS) RecordProjectionHierarchy(m_level->m_cmdBuffer);
			m_sampleDispatches = false;
			if (!history && m_readbackOptimizations && done + count == iterTS && !m_level->m_probePoints.empty())
				RecordProbeGather(m_level->m_cmdBuffer);
			EndGpuSpan(m_level->m_cmdBuffer, batchSpan);
			if (vkEndCommandBuffer(m_level->m_cmdBuffer) != VK_SUCCESS)
			{
				m_runtimeFailed = true;
				return false;
			}
		}
		if (!SubmitCommandBuffer()) return false;
		m_queriesPending = m_queryCount != 0u;
		if (m_profileEnabled) m_profile.timesteps += count;
		SetHierarchyTimestep(firstTS + count);
		MarkHierarchyHostInvalid();
		m_level->m_probesValid = m_readbackOptimizations && done + count == iterTS && !m_level->m_probePoints.empty();
		done += count;
	}
	return true;
#else
	(void)history;
	if (iterTS == 0u) return true;
	m_level->m_numTS += iterTS;
	m_level->m_hostFieldsValid = false;
	return true;
#endif
}

bool EngineVulkan::SyncProbesToHost()
{
#ifdef ENABLE_VULKAN
	if (GetPendingProbeHistorySteps() && m_probeHistoryCursor == 0) return false;
	if (!m_device || ProfileOwner().m_runtimeFailed) return false;
	if (m_level->m_probePoints.empty()) return true;
	if (!m_level->m_bufProbeValues.mapped || !SyncHierarchyToDevice()) return false;
	if (!m_readbackOptimizations || !m_level->m_probesValid)
	{
		if (!WaitForFence()) return false;
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		if (vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo) != VK_SUCCESS) return false;
		BeginProfileCommands(m_level->m_cmdBuffer);
		RecordProbeGather(m_level->m_cmdBuffer);
		if (vkEndCommandBuffer(m_level->m_cmdBuffer) != VK_SUCCESS || !SubmitCommandBuffer()) return false;
		ProfileOwner().m_queriesPending = ProfileOwner().m_queryCount != 0u;
		m_level->m_probesValid = true;
	}
	if (!WaitForFence()) return false;
	EngineVulkan& owner = ProfileOwner();
	ProfileTimer mirrorTimer(owner.m_profileEnabled ? &owner.m_profile.mirrorSeconds : nullptr);

	// Direct zero-copy read from host-visible mapped memory!
	const float* values = static_cast<const float*>(m_level->m_bufProbeValues.mapped);
	if (m_probeHistoryCount && m_probeHistoryCursor)
		values += size_t(m_probeHistoryCursor - 1u) * m_level->m_probePoints.size();
	Engine* cpuEng = (m_level->m_op ? m_level->m_op->GetEngine() : nullptr);
	FDTD_FLOAT* engVolt = (cpuEng && cpuEng->GetVoltArray()) ? cpuEng->GetVoltArray()->data() : nullptr;
	FDTD_FLOAT* engCurr = (cpuEng && cpuEng->GetCurrArray()) ? cpuEng->GetCurrArray()->data() : nullptr;

	for (size_t i = 0; i < m_level->m_probePoints.size(); ++i)
	{
		uint32_t linIdx = m_level->m_probePoints[i].linear_index;
		uint32_t fType  = m_level->m_probePoints[i].field_type;
		float val = values[i];
		if (linIdx >= m_level->m_hostVolt.size() || linIdx >= m_level->m_hostCurr.size() || fType > 1u)
		{
			owner.m_runtimeFailed = true;
			return false;
		}

		if (fType == 1u)
		{
			if (linIdx < m_level->m_hostCurr.size()) m_level->m_hostCurr[linIdx] = val;
			if (engCurr && linIdx < cpuEng->GetCurrArray()->size())
			{
				engCurr[linIdx] = val;
			}
			else if (cpuEng)
			{
				uint32_t ny = linIdx / m_level->m_grid.numCells;
				uint32_t rem = linIdx % m_level->m_grid.numCells;
				uint32_t slice = m_level->m_grid.dimY * m_level->m_grid.dimZ;
				uint32_t x = rem / slice;
				uint32_t y = (rem % slice) / m_level->m_grid.dimZ;
				uint32_t z = rem % m_level->m_grid.dimZ;
				cpuEng->SetCurr(ny, x, y, z, val);
			}
		}
		else
		{
			if (linIdx < m_level->m_hostVolt.size()) m_level->m_hostVolt[linIdx] = val;
			if (engVolt && linIdx < cpuEng->GetVoltArray()->size())
			{
				engVolt[linIdx] = val;
			}
			else if (cpuEng)
			{
				uint32_t ny = linIdx / m_level->m_grid.numCells;
				uint32_t rem = linIdx % m_level->m_grid.numCells;
				uint32_t slice = m_level->m_grid.dimY * m_level->m_grid.dimZ;
				uint32_t x = rem / slice;
				uint32_t y = (rem % slice) / m_level->m_grid.dimZ;
				uint32_t z = rem % m_level->m_grid.dimZ;
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
	if (GetPendingProbeHistorySteps()) return false;
#ifdef ENABLE_VULKAN
	if (!SyncLevelFieldsToHost())
		return false;
	return !m_level->m_innerGrid || m_level->m_innerGrid->SyncFieldsToHost();
#else
	m_level->m_hostFieldsValid = true;
	return true;
#endif
}

#ifdef ENABLE_VULKAN
bool EngineVulkan::SyncLevelFieldsToHost()
{
	EngineVulkan& owner = ProfileOwner();
	if (owner.m_runtimeFailed) return false;
	if (m_level->m_hostFieldsValid) return true;

	if (m_device && m_level->m_bufFieldStaging.mapped)
	{
		size_t fieldBytes = 3 * static_cast<size_t>(m_level->m_grid.numCells) * sizeof(float);

		auto downloadBuffers = [&](unsigned int first, unsigned int count) {
			if (!WaitForFence())
				return false;

			ProfileTimer readbackTimer(owner.m_profileEnabled ? &owner.m_profile.readbackSeconds : nullptr);
			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			if (vkBeginCommandBuffer(m_level->m_cmdBuffer, &beginInfo) != VK_SUCCESS)
				return false;

			BeginProfileCommands(m_level->m_cmdBuffer);
			const uint32_t readbackSpan = BeginGpuSpan(m_level->m_cmdBuffer, ProfileReadback);
			VkMemoryBarrier barrier = {};
			barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
			barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			MemoryBarrier(m_level->m_cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
			              VK_PIPELINE_STAGE_TRANSFER_BIT, barrier);

			VulkanBuffer* sources[] = {&m_level->m_bufVolt, &m_level->m_bufCurr};
			for (unsigned int i = 0; i < count; ++i)
			{
				VkBufferCopy copyRegion = { 0, i * fieldBytes, fieldBytes };
				CopyBuffer(m_level->m_cmdBuffer, sources[first + i]->buffer,
				           m_level->m_bufFieldStaging.buffer, copyRegion, true);
			}
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
			MemoryBarrier(m_level->m_cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
			              VK_PIPELINE_STAGE_HOST_BIT, barrier);

			EndGpuSpan(m_level->m_cmdBuffer, readbackSpan);
			if (vkEndCommandBuffer(m_level->m_cmdBuffer) != VK_SUCCESS)
				return false;

			if (!SubmitCommandBuffer()) return false;
			owner.m_queriesPending = owner.m_queryCount != 0u;
			if (!WaitForFence()) return false;

			std::vector<float>* targets[] = {&m_level->m_hostVolt, &m_level->m_hostCurr};
			for (unsigned int i = 0; i < count; ++i)
				std::memcpy(targets[first + i]->data(),
				            static_cast<const char*>(m_level->m_bufFieldStaging.mapped) + i * fieldBytes, fieldBytes);
			return true;
		};

		if ((m_readbackOptimizations && m_level->m_bufFieldStaging.size >= 2 * fieldBytes) ? !downloadBuffers(0, 2) :
		    (!downloadBuffers(0, 1) || !downloadBuffers(1, 1)))
			return false;
		m_level->m_hostFieldsDirty = false;

		ProfileTimer mirrorTimer(owner.m_profileEnabled ? &owner.m_profile.mirrorSeconds : nullptr);
		Engine* cpuEng = (m_level->m_op ? m_level->m_op->GetEngine() : nullptr);
		if (cpuEng)
		{
			// Basic arrays use the same component-major layout as the GPU fields.
			if (m_readbackOptimizations && cpuEng->GetType() == Engine::BASIC &&
			    cpuEng->GetVoltArray() && cpuEng->GetCurrArray() &&
			    cpuEng->GetVoltArray()->size() == m_level->m_hostVolt.size() &&
			    cpuEng->GetCurrArray()->size() == m_level->m_hostCurr.size())
			{
				std::memcpy(cpuEng->GetVoltArray()->data(), m_level->m_hostVolt.data(), fieldBytes);
				std::memcpy(cpuEng->GetCurrArray()->data(), m_level->m_hostCurr.data(), fieldBytes);
				m_level->m_hostFieldsValid = true;
				return true;
			}
			size_t idx = 0;
			for (unsigned int n = 0; n < 3; ++n)
			{
				for (unsigned int x = 0; x < m_level->m_grid.dimX; ++x)
				{
					for (unsigned int y = 0; y < m_level->m_grid.dimY; ++y)
					{
						for (unsigned int z = 0; z < m_level->m_grid.dimZ; ++z)
						{
							cpuEng->SetVolt(n, x, y, z, m_level->m_hostVolt[idx]);
							cpuEng->SetCurr(n, x, y, z, m_level->m_hostCurr[idx]);
							++idx;
						}
					}
				}
			}
		}

		m_level->m_hostFieldsValid = true;
		return true;
	}

	return false;
}
#endif

unsigned int EngineVulkan::GetNumberOfTimesteps() const
{
	return GetPendingProbeHistorySteps() ? m_probeHistoryStart + m_probeHistoryCursor : m_level->m_numTS;
}

FDTD_FLOAT EngineVulkan::GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	if (!m_level->m_hostFieldsValid && !const_cast<EngineVulkan*>(this)->SyncFieldsToHost())
		return 0.0f;
	size_t idx = GetLinearIndex(n, x, y, z);
	return (idx < m_level->m_hostVolt.size()) ? m_level->m_hostVolt[idx] : 0.0f;
}

FDTD_FLOAT EngineVulkan::GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	if (!m_level->m_hostFieldsValid && !const_cast<EngineVulkan*>(this)->SyncFieldsToHost())
		return 0.0f;
	size_t idx = GetLinearIndex(n, x, y, z);
	return (idx < m_level->m_hostCurr.size()) ? m_level->m_hostCurr[idx] : 0.0f;
}

void EngineVulkan::SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
{
	if (!m_level->m_hostFieldsValid && !SyncFieldsToHost())
		return;
	size_t idx = GetLinearIndex(n, x, y, z);
	if (!m_level->m_dimensionsValid || idx >= 3u * static_cast<size_t>(m_level->m_grid.numCells))
		return;
	if (m_level->m_hostVolt.empty())
		m_level->m_hostVolt.resize(3u * static_cast<size_t>(m_level->m_grid.numCells), 0.0f);
	if (idx < m_level->m_hostVolt.size())
	{
		m_level->m_hostVolt[idx] = val;
		m_probeHistoryCount = m_probeHistoryCursor = 0;
		if (m_level->m_op && m_level->m_op->GetEngine()) m_level->m_op->GetEngine()->SetVolt(n, x, y, z, val);
		m_level->m_probesValid = m_level->m_energyValid = false;
		m_level->m_hostFieldsDirty = true;
	}
}

void EngineVulkan::SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
{
	if (!m_level->m_hostFieldsValid && !SyncFieldsToHost())
		return;
	size_t idx = GetLinearIndex(n, x, y, z);
	if (!m_level->m_dimensionsValid || idx >= 3u * static_cast<size_t>(m_level->m_grid.numCells))
		return;
	if (m_level->m_hostCurr.empty())
		m_level->m_hostCurr.resize(3u * static_cast<size_t>(m_level->m_grid.numCells), 0.0f);
	if (idx < m_level->m_hostCurr.size())
	{
		m_level->m_hostCurr[idx] = val;
		m_probeHistoryCount = m_probeHistoryCursor = 0;
		if (m_level->m_op && m_level->m_op->GetEngine()) m_level->m_op->GetEngine()->SetCurr(n, x, y, z, val);
		m_level->m_probesValid = m_level->m_energyValid = false;
		m_level->m_hostFieldsDirty = true;
	}
}

void EngineVulkan::Reset()
{
#ifdef ENABLE_VULKAN
	if (m_device != VK_NULL_HANDLE && m_ownsVulkanDevice)
		vkDeviceWaitIdle(m_device);

	// Children share the device and must release their level resources first.
	m_level->m_innerGrid.reset();

	if (m_device != VK_NULL_HANDLE)
	{
		DestroyFieldDumps(true);
		DestroyBuffer(m_level->m_bufVv);
		DestroyBuffer(m_level->m_bufVi);
		DestroyBuffer(m_level->m_bufIi);
		DestroyBuffer(m_level->m_bufIv);
		DestroyBuffer(m_level->m_bufVolt);
		DestroyBuffer(m_level->m_bufCurr);
		DestroyBuffer(m_level->m_bufFieldStaging);

		DestroyBuffer(m_level->m_bufVoltExcPoints);
		DestroyBuffer(m_level->m_bufCurrExcPoints);
		DestroyBuffer(m_level->m_bufExcSignals);
		DestroyBuffer(m_level->m_bufProbePoints);
		DestroyBuffer(m_level->m_bufProbeValues);
		DestroyEnergyResources();

		DestroyBuffer(m_level->m_bufUpmlIndices);
		DestroyBuffer(m_level->m_bufUpmlVoltCoeffs);
		DestroyBuffer(m_level->m_bufUpmlCurrCoeffs);
		DestroyBuffer(m_level->m_bufUpmlVoltFlux);
		DestroyBuffer(m_level->m_bufUpmlCurrFlux);

		DestroyBuffer(m_level->m_bufMurParams);
		DestroyBuffer(m_level->m_bufMurStore);

		DestroyBuffer(m_level->m_bufTfsfVoltParams);
		DestroyBuffer(m_level->m_bufTfsfCurrParams);
		DestroyBuffer(m_level->m_bufTfsfCurrSignal);
		DestroyBuffer(m_level->m_bufTfsfVoltSignal);

		DestroyBuffer(m_level->m_bufRlcParams);
		DestroyBuffer(m_level->m_bufRlcState);

		DestroyBuffer(m_level->m_bufAbcVoltParams);
		DestroyBuffer(m_level->m_bufAbcVoltStore);
		DestroyBuffer(m_level->m_bufAbcCurrParams);
		DestroyBuffer(m_level->m_bufAbcCurrStore);

		DestroyBuffer(m_level->m_bufDebyeParams);
		DestroyBuffer(m_level->m_bufDebyePoles);
		DestroyBuffer(m_level->m_bufDebyeState);

		DestroyBuffer(m_level->m_bufDispVoltParams);
		DestroyBuffer(m_level->m_bufDispVoltState);
		DestroyBuffer(m_level->m_bufDispCurrParams);
		DestroyBuffer(m_level->m_bufDispCurrState);

		DestroyBuffer(m_level->m_bufCylR0);
		DestroyBuffer(m_level->m_bufMultigridInterpolation);

		if (m_pipelineVoltPalette != VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipelineVoltPalette, nullptr); m_pipelineVoltPalette = VK_NULL_HANDLE; }
		if (m_pipelineCurrPalette != VK_NULL_HANDLE) { vkDestroyPipeline(m_device, m_pipelineCurrPalette, nullptr); m_pipelineCurrPalette = VK_NULL_HANDLE; }
		if (m_pipelineVolt != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineVolt, nullptr); m_pipelineVolt = VK_NULL_HANDLE; }
		if (m_pipelineCurr != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineCurr, nullptr); m_pipelineCurr = VK_NULL_HANDLE; }
		if (m_pipelineExc  != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineExc, nullptr);  m_pipelineExc  = VK_NULL_HANDLE; }
		if (m_pipelineProbe!= VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineProbe, nullptr);m_pipelineProbe= VK_NULL_HANDLE; }
		if (m_pipelineUpmlPre != VK_NULL_HANDLE)  { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineUpmlPre, nullptr);  m_pipelineUpmlPre  = VK_NULL_HANDLE; }
		if (m_pipelineUpmlPost != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineUpmlPost, nullptr); m_pipelineUpmlPost = VK_NULL_HANDLE; }
		if (m_pipelineMurPre != VK_NULL_HANDLE)   { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineMurPre, nullptr);   m_pipelineMurPre   = VK_NULL_HANDLE; }
		if (m_pipelineMurPost != VK_NULL_HANDLE)  { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineMurPost, nullptr);  m_pipelineMurPost  = VK_NULL_HANDLE; }
		if (m_pipelineMurApply != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineMurApply, nullptr); m_pipelineMurApply = VK_NULL_HANDLE; }
		if (m_pipelineTfsf != VK_NULL_HANDLE)     { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineTfsf, nullptr);     m_pipelineTfsf     = VK_NULL_HANDLE; }
		if (m_pipelineRlc != VK_NULL_HANDLE)      { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineRlc, nullptr);      m_pipelineRlc      = VK_NULL_HANDLE; }
		if (m_pipelineAbcVolt != VK_NULL_HANDLE)  { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineAbcVolt, nullptr);  m_pipelineAbcVolt  = VK_NULL_HANDLE; }
		if (m_pipelineAbcCurr != VK_NULL_HANDLE)  { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineAbcCurr, nullptr);  m_pipelineAbcCurr  = VK_NULL_HANDLE; }
		if (m_pipelineDisp != VK_NULL_HANDLE)     { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineDisp, nullptr);     m_pipelineDisp     = VK_NULL_HANDLE; }
		if (m_pipelineDebye != VK_NULL_HANDLE)     { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineDebye, nullptr);     m_pipelineDebye     = VK_NULL_HANDLE; }
		if (m_pipelineCyl != VK_NULL_HANDLE)      { if (m_ownsVulkanDevice) vkDestroyPipeline(m_device, m_pipelineCyl, nullptr);      m_pipelineCyl      = VK_NULL_HANDLE; }
		if (m_level->m_pipelineMultigrid != VK_NULL_HANDLE){ vkDestroyPipeline(m_device, m_level->m_pipelineMultigrid, nullptr);m_level->m_pipelineMultigrid= VK_NULL_HANDLE; }

		if (m_pipelineLayoutFields != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipelineLayout(m_device, m_pipelineLayoutFields, nullptr); m_pipelineLayoutFields = VK_NULL_HANDLE; }
		if (m_pipelineLayoutExc    != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipelineLayout(m_device, m_pipelineLayoutExc, nullptr);    m_pipelineLayoutExc    = VK_NULL_HANDLE; }
		if (m_pipelineLayoutProbe  != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipelineLayout(m_device, m_pipelineLayoutProbe, nullptr);  m_pipelineLayoutProbe  = VK_NULL_HANDLE; }
		if (m_pipelineLayoutUpml   != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipelineLayout(m_device, m_pipelineLayoutUpml, nullptr);   m_pipelineLayoutUpml   = VK_NULL_HANDLE; }
		if (m_pipelineLayoutMur    != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipelineLayout(m_device, m_pipelineLayoutMur, nullptr);    m_pipelineLayoutMur    = VK_NULL_HANDLE; }
		if (m_pipelineLayoutTfsf   != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipelineLayout(m_device, m_pipelineLayoutTfsf, nullptr);   m_pipelineLayoutTfsf   = VK_NULL_HANDLE; }
		if (m_pipelineLayoutRlc    != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipelineLayout(m_device, m_pipelineLayoutRlc, nullptr);    m_pipelineLayoutRlc    = VK_NULL_HANDLE; }
		if (m_pipelineLayoutAbc    != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipelineLayout(m_device, m_pipelineLayoutAbc, nullptr);    m_pipelineLayoutAbc    = VK_NULL_HANDLE; }
		if (m_pipelineLayoutDisp   != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipelineLayout(m_device, m_pipelineLayoutDisp, nullptr);   m_pipelineLayoutDisp   = VK_NULL_HANDLE; }
		if (m_pipelineLayoutDebye   != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipelineLayout(m_device, m_pipelineLayoutDebye, nullptr);   m_pipelineLayoutDebye   = VK_NULL_HANDLE; }
		if (m_pipelineLayoutCyl    != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyPipelineLayout(m_device, m_pipelineLayoutCyl, nullptr);    m_pipelineLayoutCyl    = VK_NULL_HANDLE; }
		if (m_level->m_pipelineLayoutMultigrid != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device, m_level->m_pipelineLayoutMultigrid, nullptr); m_level->m_pipelineLayoutMultigrid = VK_NULL_HANDLE; }

		if (m_descLayoutFields != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyDescriptorSetLayout(m_device, m_descLayoutFields, nullptr); m_descLayoutFields = VK_NULL_HANDLE; }
		if (m_descLayoutExc    != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyDescriptorSetLayout(m_device, m_descLayoutExc, nullptr);    m_descLayoutExc    = VK_NULL_HANDLE; }
		if (m_descLayoutProbe  != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyDescriptorSetLayout(m_device, m_descLayoutProbe, nullptr);  m_descLayoutProbe  = VK_NULL_HANDLE; }
		if (m_descLayoutUpml   != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyDescriptorSetLayout(m_device, m_descLayoutUpml, nullptr);   m_descLayoutUpml   = VK_NULL_HANDLE; }
		if (m_descLayoutMur    != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyDescriptorSetLayout(m_device, m_descLayoutMur, nullptr);    m_descLayoutMur    = VK_NULL_HANDLE; }
		if (m_descLayoutTfsf   != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyDescriptorSetLayout(m_device, m_descLayoutTfsf, nullptr);   m_descLayoutTfsf   = VK_NULL_HANDLE; }
		if (m_descLayoutRlc    != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyDescriptorSetLayout(m_device, m_descLayoutRlc, nullptr);    m_descLayoutRlc    = VK_NULL_HANDLE; }
		if (m_descLayoutAbc    != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyDescriptorSetLayout(m_device, m_descLayoutAbc, nullptr);    m_descLayoutAbc    = VK_NULL_HANDLE; }
		if (m_descLayoutDisp   != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyDescriptorSetLayout(m_device, m_descLayoutDisp, nullptr);   m_descLayoutDisp   = VK_NULL_HANDLE; }
		if (m_descLayoutDebye   != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyDescriptorSetLayout(m_device, m_descLayoutDebye, nullptr);   m_descLayoutDebye   = VK_NULL_HANDLE; }
		if (m_descLayoutCyl    != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyDescriptorSetLayout(m_device, m_descLayoutCyl, nullptr);    m_descLayoutCyl    = VK_NULL_HANDLE; }
		if (m_level->m_descLayoutMultigrid != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_level->m_descLayoutMultigrid, nullptr); m_level->m_descLayoutMultigrid = VK_NULL_HANDLE; }

		if (m_descPool != VK_NULL_HANDLE) { if (m_ownsVulkanDevice) vkDestroyDescriptorPool(m_device, m_descPool, nullptr); m_descPool = VK_NULL_HANDLE; }
		if (m_queryPool) { vkDestroyQueryPool(m_device, m_queryPool, nullptr); m_queryPool = VK_NULL_HANDLE; }
		if (m_level->m_fence != VK_NULL_HANDLE)    { vkDestroyFence(m_device, m_level->m_fence, nullptr);              m_level->m_fence = VK_NULL_HANDLE; }
		if (m_level->m_cmdPool != VK_NULL_HANDLE)  { vkDestroyCommandPool(m_device, m_level->m_cmdPool, nullptr);      m_level->m_cmdPool = VK_NULL_HANDLE; }

		if (m_ownsVulkanDevice)
			vkDestroyDevice(m_device, nullptr);
		m_device = VK_NULL_HANDLE;
	}

	if (m_instance != VK_NULL_HANDLE)
	{
		if (m_ownsVulkanDevice)
			vkDestroyInstance(m_instance, nullptr);
		m_instance = VK_NULL_HANDLE;
	}
#endif

	m_level->m_numTS = 0;
	m_probeHistoryStart = m_probeHistoryCount = m_probeHistoryCursor = 0;
	m_level->m_probeCapacity = 1;
	m_level->m_probesValid = m_level->m_energyValid = false;
	m_profile = ProfileStatistics();
#ifdef ENABLE_VULKAN
	m_runtimeFailed = m_queriesPending = m_sampleDispatches = false;
	m_queryCount = m_profileSpanCount = 0;
	m_physicalDevice = VK_NULL_HANDLE;
	m_level->m_numUpmlCells = 0;
	m_level->m_totalMurPoints = 0;
	m_level->m_murFaces.clear();
	m_level->m_tfsfVoltFaces.clear();
	m_level->m_tfsfVoltPoints.clear();
	m_level->m_tfsfCurrFaces.clear();
	m_level->m_tfsfCurrPoints.clear();
	m_level->m_tfsfSigLength = 0;
	m_level->m_tfsfPeriod = 0;
	m_level->m_rlcCount = 0;
	m_level->m_abcVoltCount = 0;
	m_level->m_abcCurrCount = 0;
	m_level->m_abcVoltSheets.clear();
	m_level->m_abcCurrSheets.clear();
	m_level->m_debyeCount = 0;
	m_level->m_descSetDebye = VK_NULL_HANDLE;
	m_level->m_descSetProbe = m_level->m_descSetEnergy = VK_NULL_HANDLE;
	m_level->m_energyGroups = 0;
	m_level->m_dispVoltCount = 0;
	m_level->m_dispCurrCount = 0;
	m_level->m_dispVoltPasses.clear();
	m_level->m_dispCurrPasses.clear();
	m_level->m_descSetCyl = VK_NULL_HANDLE;
	m_level->m_descSetMultigrid = VK_NULL_HANDLE;
	m_level->m_hasCylinder = false;
	m_level->m_cylClosedAlpha = false;
	m_level->m_cylR0Included = false;
	m_level->m_cylLastALine = 0;
#endif
	m_level->m_hostFieldsValid = true;
	m_level->m_hostFieldsDirty = true;
	std::fill(m_level->m_hostVolt.begin(), m_level->m_hostVolt.end(), 0.0f);
	m_level->m_coefficients = CoefficientStatistics();
	std::fill(m_level->m_hostCurr.begin(), m_level->m_hostCurr.end(), 0.0f);
	m_level->m_voltExcPoints.clear();
	m_level->m_currExcPoints.clear();
	m_level->m_probePoints.clear();
}

std::string EngineVulkan::GetBackendName() const
{
	return "Vulkan (" + m_deviceName + ")";
}
