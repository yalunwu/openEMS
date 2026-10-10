/* Copyright (C) 2026 openEMS Project. SPDX-License-Identifier: GPL-3.0-or-later */
#include "engine_vulkan.h"
#include "Common/processfields_fd.h"
#include "Common/processfields_td.h"
#include "FDTD/engine_interface_fdtd.h"
#include "FDTD/operator.h"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <limits>
#include <stdexcept>

#ifdef ENABLE_VULKAN
#include <shaderc/shaderc.hpp>

namespace {
struct FDTimer {
	explicit FDTimer(double* value) : value(value), start(value ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point()) {}
	~FDTimer() { if(value) *value+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); }
	double* value;
	std::chrono::steady_clock::time_point start;
};
const char* fdShader = R"(#version 450
layout(local_size_x=256) in;
struct Axis { uint position; float a; float b; uint padding; };
layout(std430,binding=0) readonly buffer Field { float fieldData[]; };
layout(std430,binding=1) readonly buffer Mapping { Axis axes[]; };
#ifdef TIME_DOMAIN
layout(std430,binding=2) writeonly buffer Values { float values[]; };
#else
layout(std430,binding=2) buffer Sums { vec2 sums[]; };
layout(std430,binding=3) readonly buffer Phases { vec2 phases[]; };
#endif
layout(push_constant) uniform Parameters {
    uint nx; uint ny; uint nz; uint gx; uint gy; uint gz;
    uint offset; uint count; uint phase; uint mode; uint magnetic; uint cells;
} pc;
void main() {
    uint i=gl_GlobalInvocationID.x;
    if(i>=pc.count) return;
    uint global=pc.offset+i;
    uint points=pc.nx*pc.ny*pc.nz;
    uint n=global/points, point=global%points;
    uvec3 outputPos=uvec3(point/(pc.ny*pc.nz),(point/pc.nz)%pc.ny,point%pc.nz);
    Axis a[3]=Axis[3](axes[outputPos.x],axes[pc.nx+outputPos.y],axes[pc.nx+pc.ny+outputPos.z]);
    uvec3 pos=uvec3(a[0].position,a[1].position,a[2].position);
    uint index=n*pc.cells+(pos.x*pc.gy+pos.y)*pc.gz+pos.z;
    uvec3 stride=uvec3(pc.gy*pc.gz,pc.gz,1);
    uint np=(n+1)%3, npp=(n+2)%3;
    precise float value=0;
    if(pc.mode==0) {
        value=fieldData[index]*a[n].a;
    } else if((pc.mode==1 && pc.magnetic==0) || (pc.mode==2 && pc.magnetic==1)) {
        value=fieldData[index]*a[n].a;
        if(a[n].b!=0) {
            uint neighbor=pc.magnetic==0 ? index-stride[n] : index+stride[n];
            value=value+fieldData[neighbor]*a[n].b;
        }
    } else {
        uvec3 grid=uvec3(pc.gx,pc.gy,pc.gz);
        bool outside=any(equal(pos,grid-uvec3(1)));
        if(pc.magnetic==1) outside=outside || pos[np]==0 || pos[npp]==0;
        if(!outside) {
            uint first=pc.magnetic==0 ? index+stride[np] : index-stride[np];
            uint second=pc.magnetic==0 ? first+stride[npp] : first-stride[npp];
            uint third=pc.magnetic==0 ? index+stride[npp] : index-stride[npp];
            value=fieldData[index]*a[n].a;
            value=value+fieldData[first]*a[n].a;
            value=value+fieldData[second]*a[n].a;
            value=value+fieldData[third]*a[n].a;
        }
    }
#ifdef TIME_DOMAIN
    values[i]=value;
#else
    precise vec2 contribution=value*phases[pc.phase];
    precise vec2 result=sums[i]+contribution;
    sums[i]=result;
#endif
}
)";
}

bool EngineVulkan::CreateFDPipeline(bool* allocationFailed)
{
	return CreateFieldPipeline(false, allocationFailed);
}

bool EngineVulkan::CreateFieldPipeline(bool timeDomain, bool* allocationFailed)
{
	*allocationFailed=false;
	auto& target=timeDomain ? m_pipelineTD : m_pipelineFD;
	auto& descriptorLayout=timeDomain ? m_descLayoutTD : m_descLayoutFD;
	auto& targetLayout=timeDomain ? m_pipelineLayoutTD : m_pipelineLayoutFD;
	if (target) return true;
	auto failed=[&](VkResult result) {
		*allocationFailed=result==VK_ERROR_OUT_OF_HOST_MEMORY || result==VK_ERROR_OUT_OF_DEVICE_MEMORY;
		return result!=VK_SUCCESS;
	};
	shaderc::Compiler compiler;
	shaderc::CompileOptions options;
	options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
	options.SetOptimizationLevel(shaderc_optimization_level_performance);
	if (timeDomain) options.AddMacroDefinition("TIME_DOMAIN");
	const auto compiled = compiler.CompileGlslToSpv(fdShader, shaderc_glsl_compute_shader, "field_fd.comp", options);
	if (compiled.GetCompilationStatus() != shaderc_compilation_status_success) {
		std::cerr << compiled.GetErrorMessage();
		return false;
	}
	VkShaderModuleCreateInfo shader={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	shader.codeSize=(compiled.end()-compiled.begin())*sizeof(uint32_t); shader.pCode=compiled.begin();
	VkShaderModule module=VK_NULL_HANDLE;
	if (failed(vkCreateShaderModule(m_device,&shader,nullptr,&module))) return false;
	struct ModuleGuard { VkDevice device; VkShaderModule module; ~ModuleGuard() {vkDestroyShaderModule(device,module,nullptr);} } guard = {m_device,module};
	VkDescriptorSetLayoutBinding bindings[4] = {};
	for (uint32_t i=0; i<4; ++i) {
		bindings[i].binding=i; bindings[i].descriptorCount=1;
		bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
	}
	VkDescriptorSetLayoutCreateInfo layout = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	layout.bindingCount=timeDomain ? 3 : 4; layout.pBindings=bindings;
	if (failed(vkCreateDescriptorSetLayout(m_device,&layout,nullptr,&descriptorLayout))) return false;
	VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT,0,48};
	VkPipelineLayoutCreateInfo pipelineLayout = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	pipelineLayout.setLayoutCount=1; pipelineLayout.pSetLayouts=&descriptorLayout;
	pipelineLayout.pushConstantRangeCount=1; pipelineLayout.pPushConstantRanges=&range;
	if (failed(vkCreatePipelineLayout(m_device,&pipelineLayout,nullptr,&targetLayout))) return false;
	VkComputePipelineCreateInfo pipeline = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	pipeline.layout=targetLayout;
	pipeline.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipeline.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; pipeline.stage.module=module; pipeline.stage.pName="main";
	return !failed(vkCreateComputePipelines(m_device,VK_NULL_HANDLE,1,&pipeline,nullptr,&target));
}

void EngineVulkan::DestroyFieldDumps(bool reset)
{
	for (auto& dump : m_fdDumps) {
		if (auto active=dump->active.lock()) {
			*active=false;
			if (reset) dump->processing->Reset();
		}
	}
	m_fdDumps.clear();
	m_profile.fdAccumulatorBytes=0;
	m_profile.fdMappingBytes=0;
	m_fdStaging.Release();
	if (m_pipelineFD) vkDestroyPipeline(m_device,m_pipelineFD,nullptr);
	if (m_pipelineLayoutFD) vkDestroyPipelineLayout(m_device,m_pipelineLayoutFD,nullptr);
	if (m_descLayoutFD) vkDestroyDescriptorSetLayout(m_device,m_descLayoutFD,nullptr);
	m_pipelineFD=VK_NULL_HANDLE; m_pipelineLayoutFD=VK_NULL_HANDLE; m_descLayoutFD=VK_NULL_HANDLE;
}

bool EngineVulkan::SupportsFDDump(const ProcessFieldsFD* processing) const
{
	return typeid(*processing)==typeid(ProcessFieldsFD) && SupportsFieldDump(processing);
}

bool EngineVulkan::SupportsFieldDump(const ProcessFields* processing) const
{
	if (!m_level->m_op || typeid(*processing->m_Eng_Interface)!=typeid(Engine_Interface_FDTD) ||
	    typeid(*m_level->m_op)!=typeid(Operator) || !m_level->m_op->GetEngine() ||
	    m_level->m_op->GetEngine()->GetType()!=Engine::BASIC || processing->Op!=m_level->m_op ||
	    (processing->m_DumpType!=ProcessFields::E_FIELD_DUMP && processing->m_DumpType!=ProcessFields::H_FIELD_DUMP)) return false;
	if (processing->m_Eng_Interface->GetInterpolationType()<Engine_Interface_Base::NO_INTERPOLATION ||
	    processing->m_Eng_Interface->GetInterpolationType()>Engine_Interface_Base::CELL_INTERPOLATE) return false;
	return true;
}

bool EngineVulkan::BuildFieldMapping(const ProcessFields* processing, uint32_t* dims, uint32_t& mode,
                                    uint32_t& magnetic, std::vector<FieldAxis>& mapping) const
{
	magnetic=processing->m_DumpType==ProcessFields::H_FIELD_DUMP;
	mode=processing->m_Eng_Interface->GetInterpolationType();
	uint64_t components=3;
	for (unsigned int n=0; n<3; ++n) {
		dims[n]=processing->numLines[n];
		if (!dims[n] || components>UINT32_MAX/dims[n]) return false;
		components*=dims[n];
		for (unsigned int i=0; i<dims[n]; ++i) {
			unsigned int pos[3]={0,0,0}; pos[n]=processing->posLines[n][i];
			if (pos[n]>=m_level->m_op->GetNumberOfLines(n)) return false;
			const double delta=m_level->m_op->GetEdgeLength(n,pos,magnetic!=0);
			double a=delta ? 1.0/delta : 0, b=0;
			if (mode==1 && !magnetic) {
				if (pos[n]==m_level->m_op->GetNumberOfLines(n,true)-1) {
					--pos[n]; const double lower=m_level->m_op->GetEdgeLength(n,pos); ++pos[n];
					a=0; b=lower ? 1.0/lower : 0;
				} else if (delta && pos[n]>0) {
					--pos[n]; const double lower=m_level->m_op->GetEdgeLength(n,pos); ++pos[n];
					const double fraction=delta/(delta+lower);
					a*=(1-fraction); b=lower ? fraction/lower : 0;
				}
			} else if (mode==2 && magnetic) {
				if (pos[n]>=m_level->m_op->GetNumberOfLines(n,true)-1) a=0;
				else {
					++pos[n]; const double upper=m_level->m_op->GetEdgeLength(n,pos,true); --pos[n];
					const double fraction=delta/(delta+upper);
					a*=(1-fraction); b=upper ? fraction/upper : 0;
				}
			} else if (mode!=0) a*=0.25;
			if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(static_cast<float>(a)) || !std::isfinite(static_cast<float>(b))) return false;
			mapping.push_back({pos[n],static_cast<float>(a),static_cast<float>(b),0});
		}
	}
	return true;
}

bool EngineVulkan::CheckFDFallbackMemory(const FDDump* pending, uint64_t* available) const
{
	VkPhysicalDeviceMemoryProperties memory={};
	vkGetPhysicalDeviceMemoryProperties(m_physicalDevice,&memory);
	std::array<uint64_t,VK_MAX_MEMORY_HEAPS> usage={};
	auto account=[&](const VulkanBuffer& buffer) {
		if (buffer.memory && buffer.memoryHeap<memory.memoryHeapCount)
			usage[buffer.memoryHeap]+=buffer.allocationSize;
	};
	auto accountDump=[&](const FDDump& dump) {
		account(dump.mapping); account(dump.phases);
		for (const auto& chunk : dump.chunks) account(chunk->sums);
	};
	account(m_fdStaging);
	account(m_tdValues); account(m_tdStaging);
	for (const auto& dump : m_tdDumps) account(dump->mapping);
	for (const auto& dump : m_fdDumps) accountDump(*dump);
	if (pending) accountDump(*pending);
	bool fits=true;
	uint64_t committed=0;
	if (available) *available=0;
	for (uint32_t h=0; h<memory.memoryHeapCount; ++h)
		if (memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
			committed+=usage[h];
			const uint64_t limit=(memory.memoryHeaps[h].size/4)*3/4;
			if (usage[h]>limit) fits=false;
			if (available && usage[h]<limit) *available=std::max(*available,limit-usage[h]);
		}
	if (committed>m_fdMemoryLimit) fits=false;
	if (available) *available=std::min(*available,m_fdMemoryLimit>committed ? m_fdMemoryLimit-committed : 0);
	return fits;
}

bool EngineVulkan::AllocateFDDump(ProcessFieldsFD* processing, std::unique_ptr<FDDump>& dump, std::string& reason)
{
	reason="unsupported field mapping";
	if (!SupportsFDDump(processing)) return false;
	dump.reset(new FDDump);
	dump->device=m_device; dump->processing=processing;
	dump->active=processing->m_deviceAccumulation;
	std::vector<FieldAxis> mapping;
	if (!BuildFieldMapping(processing,dump->dims,dump->mode,dump->magnetic,mapping)) return false;
	const uint64_t components=3ull*dump->dims[0]*dump->dims[1]*dump->dims[2];
	const uint64_t frequencies=processing->m_FD_Samples.size();
	if (!frequencies || frequencies>UINT32_MAX || components>UINT64_MAX/(8*frequencies) ||
	    processing->m_FD_Fields.size()!=frequencies) return false;
	dump->bytes=components*8*frequencies;
	dump->mappingBytes=mapping.size()*sizeof(FieldAxis);
	const uint64_t phaseBytes=frequencies*64*8;
	VkPhysicalDeviceProperties properties={}; vkGetPhysicalDeviceProperties(m_physicalDevice,&properties);
	const uint64_t limit=std::min<uint64_t>(m_fdChunkBytes,properties.limits.maxStorageBufferRange);
	const uint64_t perChunk=std::min<uint64_t>(limit/8,uint64_t(properties.limits.maxComputeWorkGroupCount[0])*256);
	if (!perChunk || phaseBytes>properties.limits.maxStorageBufferRange || dump->mappingBytes>properties.limits.maxStorageBufferRange) return false;
	const uint64_t chunkCount=((components+perChunk-1)/perChunk)*frequencies;
	if (chunkCount>UINT32_MAX/4 || chunkCount+8>properties.limits.maxMemoryAllocationCount) return false;
	VkPhysicalDeviceMemoryBudgetPropertiesEXT budget={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
	VkPhysicalDeviceMemoryProperties2 memory={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
	if (m_memoryBudget) memory.pNext=&budget;
	vkGetPhysicalDeviceMemoryProperties2(m_physicalDevice,&memory);
	uint64_t available=0;
	for (uint32_t h=0; h<memory.memoryProperties.memoryHeapCount; ++h)
		if (memory.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
			const uint64_t capacity=m_memoryBudget ? (budget.heapBudget[h]>budget.heapUsage[h] ? budget.heapBudget[h]-budget.heapUsage[h] : 0) : memory.memoryProperties.memoryHeaps[h].size/4;
			available=std::max(available,capacity);
		}
	const uint64_t estimate=dump->bytes+dump->mappingBytes+phaseBytes+(m_fdStaging.buffer ? 0 : 16ull*1024*1024)+chunkCount*properties.limits.bufferImageGranularity;
	reason="insufficient memory budget or optional allocation";
	const bool fallbackFits=m_memoryBudget || CheckFDFallbackMemory(nullptr,&available);
	std::cout << "VULKAN_FD name=" << processing->GetName() << " file=" << processing->m_filename << " estimated_bytes=" << estimate
	          << " available_bytes=" << available << " memory_budget=" << m_memoryBudget << std::endl;
	if (m_memoryBudget) {
		// Driver usage already includes prior allocations; do not subtract them again.
		if (estimate>std::min(m_fdMemoryLimit,available*3/4)) return false;
	} else {
		// Bound the device-local sums first, then verify every actual allocation,
		// including host-visible buffers if their selected heap is device-local.
		if (!fallbackFits || dump->bytes+dump->mappingBytes>available) return false;
	}
	bool allocationFailed=false;
	if (!UploadStorageBuffer(mapping.data(),dump->mappingBytes,dump->mapping,&allocationFailed)) {
		if (!allocationFailed) throw std::runtime_error("Vulkan FD mapping upload failed");
		return false;
	}
	if (!CreateBuffer(phaseBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,dump->phases)) return false;
	if (!m_memoryBudget && !CheckFDFallbackMemory(dump.get())) return false;
	VkDescriptorPoolSize poolSize={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,static_cast<uint32_t>(chunkCount*4)};
	VkDescriptorPoolCreateInfo pool={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	pool.maxSets=static_cast<uint32_t>(chunkCount); pool.poolSizeCount=1; pool.pPoolSizes=&poolSize;
	if (vkCreateDescriptorPool(m_device,&pool,nullptr,&dump->pool)!=VK_SUCCESS) return false;
	for (uint32_t f=0; f<frequencies; ++f)
	for (uint64_t offset=0; offset<components; offset+=perChunk) {
		std::unique_ptr<FDChunk> chunk(new FDChunk);
		chunk->offset=static_cast<uint32_t>(offset); chunk->count=static_cast<uint32_t>(std::min(perChunk,components-offset)); chunk->frequency=f;
		if (!CreateBuffer(uint64_t(chunk->count)*8,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,chunk->sums)) return false;
		VkDescriptorSetAllocateInfo allocate={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		allocate.descriptorPool=dump->pool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&m_descLayoutFD;
		if (vkAllocateDescriptorSets(m_device,&allocate,&chunk->descriptors)!=VK_SUCCESS) return false;
		VulkanBuffer* buffers[]={dump->magnetic ? &m_level->m_bufCurr : &m_level->m_bufVolt,&dump->mapping,&chunk->sums,&dump->phases};
		VkDescriptorBufferInfo infos[4]={}; VkWriteDescriptorSet writes[4]={};
		for (uint32_t b=0; b<4; ++b) {
			infos[b]={buffers[b]->buffer,0,buffers[b]->size};
			writes[b].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[b].dstSet=chunk->descriptors;
			writes[b].dstBinding=b; writes[b].descriptorCount=1; writes[b].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[b].pBufferInfo=&infos[b];
		}
		vkUpdateDescriptorSets(m_device,4,writes,0,nullptr);
		dump->chunks.push_back(std::move(chunk));
		if (!m_memoryBudget && !CheckFDFallbackMemory(dump.get())) return false;
	}
	for (const auto& chunk : dump->chunks) dump->allocatedBytes+=chunk->sums.allocationSize;
	return true;
}

void EngineVulkan::PrepareFDPhases(unsigned int firstTS, unsigned int count)
{
	if (m_fdDumps.empty()) return;
	FDTimer timer(m_profileEnabled ? &m_profile.fdPhaseSeconds : nullptr);
	for (auto& dump : m_fdDumps) {
		const auto active=dump->active.lock();
		if (!active || !*active) continue;
		auto* phases=static_cast<std::complex<float>*>(dump->phases.mapped);
		auto* processing=dump->processing;
		if (!processing->GetEnable()) continue;
		if (processing->m_FD_Samples.size()!=dump->phases.size/(64*8) ||
		    processing->m_FD_Fields.size()!=processing->m_FD_Samples.size() ||
		    processing->m_DumpType!=(dump->magnetic ? ProcessFields::H_FIELD_DUMP : ProcessFields::E_FIELD_DUMP) ||
		    processing->m_Eng_Interface->GetInterpolationType()!=dump->mode)
			throw std::runtime_error("Vulkan FD mapping or frequencies changed; reset and reinitialize before another run");
		for (unsigned int slot=0; slot<count; ++slot) {
			const unsigned int ts=firstTS+slot;
			if (ts<processing->startTS || ts>processing->stopTS || !processing->m_FD_Interval || ts%processing->m_FD_Interval) continue;
			const double time=(double(ts)+double(processing->m_dualTime)*0.5)*m_level->m_op->GetTimestep();
			for (size_t f=0; f<processing->m_FD_Samples.size(); ++f) {
				std::complex<float> phase=std::exp(std::complex<float>(-2.0*I_UNIT*PI*processing->m_FD_Samples[f]*time));
				phase*=2; phase*=m_level->m_op->GetTimestep()*processing->m_FD_Interval;
				phases[slot*processing->m_FD_Samples.size()+f]=phase;
			}
			if (m_profileEnabled) m_profile.fdPhaseBytes+=processing->m_FD_Samples.size()*8;
		}
	}
}

bool EngineVulkan::RecordFieldDumps(VkCommandBuffer cmd, unsigned int timestep, unsigned int slot, bool sampleTimings)
{
	const bool previousSampling=m_sampleDispatches;
	m_sampleDispatches=m_profileEnabled && sampleTimings;
	bool sampled=false;
	for (auto& dump : m_fdDumps) {
		const auto active=dump->active.lock();
		if (!active || !*active) continue;
		auto* processing=dump->processing;
		if (!processing->GetEnable() || timestep<processing->startTS || timestep>processing->stopTS ||
		    !processing->m_FD_Interval || timestep%processing->m_FD_Interval) continue;
		if (!sampled) {
			VkMemoryBarrier barrier={VK_STRUCTURE_TYPE_MEMORY_BARRIER};
			barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
			MemoryBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,barrier);
			sampled=true;
		}
		vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,m_pipelineFD);
		for (auto& chunk : dump->chunks) {
			const auto& g=m_level->m_grid;
			const uint32_t parameters[]={dump->dims[0],dump->dims[1],dump->dims[2],g.dimX,g.dimY,g.dimZ,
				chunk->offset,chunk->count,static_cast<uint32_t>(slot*processing->m_FD_Samples.size()+chunk->frequency),dump->mode,dump->magnetic,g.numCells};
			vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,m_pipelineLayoutFD,0,1,&chunk->descriptors,0,nullptr);
			vkCmdPushConstants(cmd,m_pipelineLayoutFD,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(parameters),parameters);
			Dispatch(cmd,(chunk->count+255u)/256u,1,1,ProfileFD);
		}
		++processing->m_FD_SampleCount;
		if (m_profileEnabled) ++m_profile.fdSamples;
	}
	if (sampled) {
		// Publish sums and finish field reads before the next timestep overwrites them.
		VkMemoryBarrier barrier={VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		barrier.srcAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
		MemoryBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,barrier);
	}
	m_sampleDispatches=previousSampling;
	return sampled;
}
#endif

bool EngineVulkan::RegisterTimeDomainDumps(ProcessingArray* pa, const std::string& mode)
{
	if (mode!="auto" && mode!="cpu" && mode!="gpu") return false;
#ifdef ENABLE_VULKAN
	if (GetPendingProbeHistorySteps() || !Synchronize()) return false;
	DestroyTimeDomainDumps();
	if (mode=="cpu" || !pa) return true;
	try {
	VkPhysicalDeviceProperties properties={};
	vkGetPhysicalDeviceProperties(m_physicalDevice,&properties);
	uint64_t largest=0;
	for (size_t i=0; i<pa->GetNumberOfProcessings(); ++i) {
		auto* processing=dynamic_cast<ProcessFieldsTD*>(pa->GetProcessing(i));
		if (!processing || !processing->GetEnable() || typeid(*processing)!=typeid(ProcessFieldsTD) || !SupportsFieldDump(processing)) continue;
		uint64_t count=3;
		for (unsigned int n=0; n<3; ++n) {
			if (!processing->numLines[n] || count>UINT32_MAX/processing->numLines[n]) { count=0; break; }
			count*=processing->numLines[n];
		}
		largest=std::max(largest,count*4);
	}
	const uint64_t chunkBytes=std::min({largest,m_tdChunkBytes,uint64_t(properties.limits.maxStorageBufferRange),
	                                 uint64_t(properties.limits.maxComputeWorkGroupCount[0])*256*4})/4*4;
	for (size_t i=0; i<pa->GetNumberOfProcessings(); ++i) {
		auto* processing=dynamic_cast<ProcessFieldsTD*>(pa->GetProcessing(i));
		if (!processing || !processing->GetEnable()) continue;
		auto fallback=[&](const std::string& reason) {
			std::cout << "VULKAN_TD name=" << processing->GetName() << " file=" << processing->m_filename
			          << " mode=cpu reason=" << reason << std::endl;
		};
		if (typeid(*processing)!=typeid(ProcessFieldsTD) || !SupportsFieldDump(processing)) {
			fallback("unsupported field mapping"); continue;
		}
		bool allocationFailed=false;
		if (!CreateFieldPipeline(true,&allocationFailed)) {
			if (!allocationFailed) return false;
			DestroyTimeDomainDumps();
			fallback("optional pipeline allocation"); return true;
		}
		std::unique_ptr<TDDump> dump(new TDDump);
		dump->device=m_device; dump->processing=processing; dump->reader=processing->m_deviceGather;
		std::vector<FieldAxis> mapping;
		if (!BuildFieldMapping(processing,dump->dims,dump->mode,dump->magnetic,mapping) || !chunkBytes) {
			fallback("unsupported output size or mapping"); continue;
		}
		dump->count=3u*dump->dims[0]*dump->dims[1]*dump->dims[2];
		const uint64_t estimate=mapping.size()*sizeof(FieldAxis)+(m_tdValues.buffer ? 0 : chunkBytes*2);
		VkPhysicalDeviceMemoryBudgetPropertiesEXT budget={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
		VkPhysicalDeviceMemoryProperties2 memory={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
		if (m_memoryBudget) memory.pNext=&budget;
		vkGetPhysicalDeviceMemoryProperties2(m_physicalDevice,&memory);
		uint64_t available=0;
		for (uint32_t h=0; h<memory.memoryProperties.memoryHeapCount; ++h)
			if (memory.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
				const uint64_t capacity=m_memoryBudget ? (budget.heapBudget[h]>budget.heapUsage[h] ? budget.heapBudget[h]-budget.heapUsage[h] : 0) : memory.memoryProperties.memoryHeaps[h].size/4;
				available=std::max(available,capacity*3/4);
			}
		if (!m_memoryBudget && !CheckFDFallbackMemory(nullptr,&available)) available=0;
		if (estimate>std::min(available,m_tdMemoryLimit)) {
			fallback("insufficient memory budget"); continue;
		}
		if (!UploadStorageBuffer(mapping.data(),mapping.size()*sizeof(FieldAxis),dump->mapping,&allocationFailed)) {
			if (!allocationFailed) return false;
			fallback("optional mapping allocation"); continue;
		}
		if (!m_tdValues.buffer &&
		    (!CreateBuffer(chunkBytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,m_tdValues) ||
		     !CreateBuffer(chunkBytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,m_tdStaging))) {
			m_tdValues.Release(); m_tdStaging.Release();
			fallback("optional readback allocation"); continue;
		}
		VkDescriptorPoolSize poolSize={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3};
		VkDescriptorPoolCreateInfo pool={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		pool.maxSets=1; pool.poolSizeCount=1; pool.pPoolSizes=&poolSize;
		VkResult result=vkCreateDescriptorPool(m_device,&pool,nullptr,&dump->pool);
		VkDescriptorSetAllocateInfo allocation={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		allocation.descriptorPool=dump->pool; allocation.descriptorSetCount=1; allocation.pSetLayouts=&m_descLayoutTD;
		if (result==VK_SUCCESS) result=vkAllocateDescriptorSets(m_device,&allocation,&dump->descriptors);
		if (result!=VK_SUCCESS) {
			if (result!=VK_ERROR_OUT_OF_DEVICE_MEMORY && result!=VK_ERROR_OUT_OF_HOST_MEMORY) return false;
			fallback("optional descriptor allocation"); continue;
		}
		const auto& field=dump->magnetic ? m_level->m_bufCurr : m_level->m_bufVolt;
		VkDescriptorBufferInfo buffers[]={{field.buffer,0,field.size},{dump->mapping.buffer,0,dump->mapping.size},{m_tdValues.buffer,0,m_tdValues.size}};
		VkWriteDescriptorSet writes[3]={};
		for (uint32_t binding=0; binding<3; ++binding) {
			writes[binding].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[binding].dstSet=dump->descriptors;
			writes[binding].dstBinding=binding; writes[binding].descriptorCount=1;
			writes[binding].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[binding].pBufferInfo=&buffers[binding];
		}
		vkUpdateDescriptorSets(m_device,3,writes,0,nullptr);
		TDDump* registered=dump.get();
		m_tdDumps.push_back(std::move(dump));
		if (!m_memoryBudget && !CheckFDFallbackMemory()) {
			m_tdDumps.pop_back(); fallback("insufficient cumulative memory budget"); continue;
		}
		*processing->m_deviceGather=[this,registered](ArrayLib::ArrayNIJK<float>& output) {return ReadTimeDomainDump(*registered,output);};
		std::cout << "VULKAN_TD name=" << processing->GetName() << " file=" << processing->m_filename
		          << " mode=gpu output_bytes=" << uint64_t(registered->count)*4 << " staging_bytes=" << m_tdStaging.size << std::endl;
	}
	if (m_tdDumps.empty()) DestroyTimeDomainDumps();
	} catch (const std::bad_alloc&) {
		DestroyTimeDomainDumps();
		std::cout << "VULKAN_TD mode=cpu reason=optional host allocation" << std::endl;
	}
#else
	(void)pa;
#endif
	return true;
}

#ifdef ENABLE_VULKAN
void EngineVulkan::DestroyTimeDomainDumps()
{
	for (auto& dump : m_tdDumps)
		if (auto reader=dump->reader.lock()) *reader={};
	m_tdDumps.clear();
	m_tdValues.Release(); m_tdStaging.Release();
	if (m_pipelineTD) vkDestroyPipeline(m_device,m_pipelineTD,nullptr);
	if (m_pipelineLayoutTD) vkDestroyPipelineLayout(m_device,m_pipelineLayoutTD,nullptr);
	if (m_descLayoutTD) vkDestroyDescriptorSetLayout(m_device,m_descLayoutTD,nullptr);
	m_pipelineTD=VK_NULL_HANDLE; m_pipelineLayoutTD=VK_NULL_HANDLE; m_descLayoutTD=VK_NULL_HANDLE;
}

bool EngineVulkan::ReadTimeDomainDump(TDDump& dump, ArrayLib::ArrayNIJK<float>& field)
{
	if (GetPendingProbeHistorySteps() || !Synchronize() || !SyncHierarchyToDevice() || !WaitForFence()) return false;
	if (field.size()!=dump.count || dump.processing->m_Eng_Interface->GetInterpolationType()!=dump.mode ||
	    dump.processing->m_DumpType!=(dump.magnetic ? ProcessFields::H_FIELD_DUMP : ProcessFields::E_FIELD_DUMP))
		throw std::runtime_error("Vulkan TD mapping changed; reset and reinitialize before another run");
	const auto& grid=m_level->m_grid;
	for (uint64_t offset=0; offset<dump.count; offset+=m_tdValues.size/4) {
		const uint32_t count=static_cast<uint32_t>(std::min<uint64_t>(dump.count-offset,m_tdValues.size/4));
		VkCommandBufferBeginInfo begin={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		if (vkBeginCommandBuffer(m_level->m_cmdBuffer,&begin)!=VK_SUCCESS) return false;
		BeginProfileCommands(m_level->m_cmdBuffer);
		VkMemoryBarrier barrier={VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
		MemoryBarrier(m_level->m_cmdBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,barrier);
		const uint32_t parameters[]={dump.dims[0],dump.dims[1],dump.dims[2],grid.dimX,grid.dimY,grid.dimZ,
			static_cast<uint32_t>(offset),count,0,dump.mode,dump.magnetic,grid.numCells};
		vkCmdBindPipeline(m_level->m_cmdBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,m_pipelineTD);
		vkCmdBindDescriptorSets(m_level->m_cmdBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,m_pipelineLayoutTD,0,1,&dump.descriptors,0,nullptr);
		vkCmdPushConstants(m_level->m_cmdBuffer,m_pipelineLayoutTD,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(parameters),parameters);
		const bool previous=m_sampleDispatches; m_sampleDispatches=m_profileEnabled;
		Dispatch(m_level->m_cmdBuffer,(count+255u)/256u,1,1,ProfileTD);
		m_sampleDispatches=previous;
		barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
		MemoryBarrier(m_level->m_cmdBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,barrier);
		VkBufferCopy copy={0,0,uint64_t(count)*4};
		vkCmdCopyBuffer(m_level->m_cmdBuffer,m_tdValues.buffer,m_tdStaging.buffer,1,&copy);
		barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
		MemoryBarrier(m_level->m_cmdBuffer,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,barrier);
		if (vkEndCommandBuffer(m_level->m_cmdBuffer)!=VK_SUCCESS || !SubmitCommandBuffer() || !WaitForFence()) return false;
		std::memcpy(field.data()+offset,m_tdStaging.mapped,static_cast<size_t>(copy.size));
		if (m_profileEnabled) m_profile.tdDownloadBytes+=copy.size;
	}
	if (m_profileEnabled) ++m_profile.tdSamples;
	return true;
}
#endif

bool EngineVulkan::RegisterFieldDumps(ProcessingArray* pa, bool enabled)
{
#ifdef ENABLE_VULKAN
	if (GetPendingProbeHistorySteps() || !Synchronize()) return false;
	for (const auto& dump : m_fdDumps) {
		const auto active=dump->active.lock();
		if (active && *active && dump->processing->m_FD_SampleCount) return false;
	}
	// Validate every candidate before releasing or allocating any registration.
	if (enabled && pa)
		for (size_t i=0; i<pa->GetNumberOfProcessings(); ++i) {
			auto* processing=dynamic_cast<ProcessFieldsFD*>(pa->GetProcessing(i));
			if (processing && processing->GetEnable() && SupportsFDDump(processing) &&
			    processing->m_FD_SampleCount) return false;
		}
	DestroyFieldDumps();
	if (!enabled || !pa) return true;
	for (size_t i=0; i<pa->GetNumberOfProcessings(); ++i) {
		auto* processing=dynamic_cast<ProcessFieldsFD*>(pa->GetProcessing(i));
		if (!processing || !processing->GetEnable()) continue;
		if (!SupportsFDDump(processing)) {
			std::cout << "VULKAN_FD name=" << processing->GetName() << " file=" << processing->m_filename
			          << " mode=cpu reason=unsupported field mapping" << std::endl;
			continue;
		}
		bool pipelineAllocationFailed=false, pipelineReady=false;
		try { pipelineReady=CreateFDPipeline(&pipelineAllocationFailed); }
		catch (const std::bad_alloc&) { pipelineAllocationFailed=true; }
		if (!pipelineReady) {
			DestroyFieldDumps();
			if (!pipelineAllocationFailed) return false;
			for (size_t remaining=0; remaining<pa->GetNumberOfProcessings(); ++remaining) {
				auto* fallback=dynamic_cast<ProcessFieldsFD*>(pa->GetProcessing(remaining));
				if (fallback && fallback->GetEnable())
					std::cout << "VULKAN_FD name=" << fallback->GetName() << " file=" << fallback->m_filename
					          << " mode=cpu reason=optional pipeline allocation" << std::endl;
			}
			return true;
		}
		std::unique_ptr<FDDump> dump; std::string reason;
		bool allocated=false;
		try { allocated=AllocateFDDump(processing,dump,reason); }
		catch (const std::bad_alloc&) { reason="optional host allocation"; }
		if (!allocated) {
			std::cout << "VULKAN_FD name=" << processing->GetName() << " file=" << processing->m_filename << " mode=cpu reason=" << reason << std::endl;
			continue;
		}
		const bool newStaging=!m_fdStaging.buffer;
		if (newStaging && !CreateBuffer(16ull*1024*1024,VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,m_fdStaging)) {
			std::cout << "VULKAN_FD name=" << processing->GetName() << " file=" << processing->m_filename << " mode=cpu reason=optional readback allocation" << std::endl;
			m_fdStaging.Release();
			continue;
		}
		if (!m_memoryBudget && !CheckFDFallbackMemory(dump.get())) {
			if (newStaging) m_fdStaging.Release();
			std::cout << "VULKAN_FD name=" << processing->GetName() << " file=" << processing->m_filename
			          << " mode=cpu reason=insufficient cumulative memory budget" << std::endl;
			continue;
		}
		*processing->m_deviceAccumulation=true;
		std::cout << "VULKAN_FD name=" << processing->GetName() << " file=" << processing->m_filename << " mode=gpu interval=" << processing->m_FD_Interval
		          << " accumulator_bytes=" << dump->bytes << " allocated_bytes=" << dump->allocatedBytes
		          << " mapping_bytes=" << dump->mappingBytes << " chunks=" << dump->chunks.size()
		          << " traffic_bytes_per_timestep=" << 2.0*dump->bytes/processing->m_FD_Interval << std::endl;
		m_profile.fdAccumulatorBytes+=dump->bytes;
		m_profile.fdMappingBytes+=dump->mappingBytes;
		m_fdDumps.push_back(std::move(dump));
	}
	if (m_fdDumps.empty()) return true;
	if (!SyncHierarchyToDevice() || !WaitForFence()) return false;
	PrepareFDPhases(m_level->m_numTS,1);
	VkCommandBufferBeginInfo begin={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (vkBeginCommandBuffer(m_level->m_cmdBuffer,&begin)!=VK_SUCCESS) return false;
	BeginProfileCommands(m_level->m_cmdBuffer);
	for (auto& dump : m_fdDumps)
		for (auto& chunk : dump->chunks) vkCmdFillBuffer(m_level->m_cmdBuffer,chunk->sums.buffer,0,chunk->sums.size,0);
	RecordFieldDumps(m_level->m_cmdBuffer,m_level->m_numTS,0,true);
	if (vkEndCommandBuffer(m_level->m_cmdBuffer)!=VK_SUCCESS) return false;
	m_queriesPending=m_queryCount!=0;
	return SubmitCommandBuffer() && WaitForFence();
#else
	(void)pa; (void)enabled; return true;
#endif
}

bool EngineVulkan::FinalizeFieldDumps()
{
#ifdef ENABLE_VULKAN
	if (m_fdDumps.empty()) return true;
	FDTimer timer(m_profileEnabled ? &m_profile.fdDownloadSeconds : nullptr);
	if (GetPendingProbeHistorySteps() || !Synchronize()) return false;
	static_assert(sizeof(std::complex<float>)==8,"Complex FP32 output layout must contain two floats");
	for (auto& dump : m_fdDumps) {
		const auto active=dump->active.lock();
		if (!active || !*active) continue;
		for (auto& chunk : dump->chunks)
		for (uint64_t offset=0; offset<chunk->sums.size; offset+=m_fdStaging.size) {
			const uint64_t bytes=std::min<uint64_t>(m_fdStaging.size,chunk->sums.size-offset);
			VkCommandBufferBeginInfo begin={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
			begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			if (vkBeginCommandBuffer(m_level->m_cmdBuffer,&begin)!=VK_SUCCESS) return false;
			BeginProfileCommands(m_level->m_cmdBuffer);
			VkMemoryBarrier barrier={VK_STRUCTURE_TYPE_MEMORY_BARRIER};
			barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
			MemoryBarrier(m_level->m_cmdBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,barrier);
			VkBufferCopy copy={offset,0,bytes};
			vkCmdCopyBuffer(m_level->m_cmdBuffer,chunk->sums.buffer,m_fdStaging.buffer,1,&copy);
			barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
			MemoryBarrier(m_level->m_cmdBuffer,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,barrier);
			if (vkEndCommandBuffer(m_level->m_cmdBuffer)!=VK_SUCCESS || !SubmitCommandBuffer() || !WaitForFence()) return false;
			std::memcpy(dump->processing->m_FD_Fields[chunk->frequency]->data()+chunk->offset+offset/8,m_fdStaging.mapped,static_cast<size_t>(bytes));
			if (m_profileEnabled) m_profile.fdDownloadBytes+=bytes;
		}
	}
#endif
	return true;
}
