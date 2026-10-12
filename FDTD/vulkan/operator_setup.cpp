/*
 * Copyright (C) 2026 openEMS Project
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "engine_vulkan.h"
#include "FDTD/operator.h"
#include "FDTD/operator_setup.h"
#include "FDTD/extensions/operator_extension.h"
#include "FDTD/extensions/operator_ext_upml.h"
#include "FDTD/extensions/operator_ext_excitation.h"
#include "FDTD/extensions/operator_ext_tfsf.h"
#include "FDTD/extensions/operator_ext_lumpedRLC.h"
#include "FDTD/extensions/operator_ext_steadystate.h"
#include "FDTD/extensions/operator_ext_mur_abc.h"
#include "FDTD/extensions/operator_ext_absorbing_bc.h"
#include "FDTD/extensions/operator_ext_lorentzmaterial.h"
#include "FDTD/extensions/operator_ext_debyematerial.h"
#include "FDTD/extensions/operator_ext_conductingsheet.h"
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <map>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>
#include <typeinfo>

#ifdef ENABLE_VULKAN
#include "setup_glsl.h"
#include <shaderc/shaderc.hpp>
#include "CSPrimBox.h"
#include "CSPrimSphere.h"
#include "CSPrimCylinder.h"
#include "CSPropMaterial.h"
#include "CSPropExcitation.h"
#include "CSPropLumpedElement.h"
#include "CSPropLorentzMaterial.h"
#include "CSPropDebyeMaterial.h"
#include "CSPropConductingSheet.h"
#include "CSTransform.h"

class VulkanOperatorSetup : public OperatorSetup
{
	typedef EngineVulkan::VulkanBuffer Buffer;
	EngineVulkan& engine;
	Operator& op;
	std::array<Buffer,22> buffers;
	Buffer staging;
	VkDescriptorSetLayout layout = VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;
	VkDescriptorPool pool = VK_NULL_HANDLE;
	VkDescriptorSet descriptor = VK_NULL_HANDLE;
	uint32_t cells, primCount, stores = 0;
	uint32_t upmlOffset = 0, upmlCount = 0;
	struct Params {
		uint32_t mode, base;
		double dt;
		uint32_t average, method, stores, reserved;
	};
	std::vector<CSPrimitives*> primitives;
	std::map<CSPrimitives*,uint32_t> indices;
	static void Require(bool success, const char* operation) {
		if (!success) throw std::runtime_error(std::string("GPU operator setup: ")+operation);
	}
	static bool ConstantWeight(const std::string& term) {
		if (term.empty()) return true; // ParameterScalar stores numeric values without a term.
		char* end = nullptr;
		const double value = std::strtod(term.c_str(), &end);
		while (end && (*end==' ' || *end=='\t')) ++end;
		return end != term.c_str() && end && *end=='\0' && std::isfinite(value);
	}
	void Create(Buffer& buffer, size_t size, bool host = false) {
		Require(engine.m_setupAllocationFailAfter!=0,"simulated construction allocation failure");
		if (engine.m_setupAllocationFailAfter!=UINT32_MAX) --engine.m_setupAllocationFailAfter;
		Require(engine.CreateBuffer(std::max(size,size_t(4)),
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			buffer), "buffer allocation failed");
	}
	void Upload(unsigned int index, const void* data, size_t size) {
		Require(engine.UploadStorageBuffer(data, size, buffers[index], nullptr, VK_BUFFER_USAGE_TRANSFER_SRC_BIT), "metadata upload failed");
	}
	void Begin() {
		Require(vkWaitForFences(engine.m_device,1,&engine.m_level->m_fence,VK_TRUE,UINT64_MAX)==VK_SUCCESS,"fence wait failed");
		Require(vkResetFences(engine.m_device,1,&engine.m_level->m_fence)==VK_SUCCESS,"fence reset failed");
		Require(vkResetCommandBuffer(engine.m_level->m_cmdBuffer,0)==VK_SUCCESS,"command reset failed");
		VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		Require(vkBeginCommandBuffer(engine.m_level->m_cmdBuffer,&begin)==VK_SUCCESS,"command begin failed");
	}
	void Finish() {
		Require(vkEndCommandBuffer(engine.m_level->m_cmdBuffer)==VK_SUCCESS,"command end failed");
		VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submit.commandBufferCount=1; submit.pCommandBuffers=&engine.m_level->m_cmdBuffer;
		Require(vkQueueSubmit(engine.m_computeQueue,1,&submit,engine.m_level->m_fence)==VK_SUCCESS,"submission failed");
		Require(vkWaitForFences(engine.m_device,1,&engine.m_level->m_fence,VK_TRUE,UINT64_MAX)==VK_SUCCESS,"GPU construction failed");
	}
	void Download(unsigned int index, void* destination, size_t size, size_t offset=0) {
		Require(offset<=buffers[index].size && size<=buffers[index].size-offset,"download range exceeded");
		for (size_t copied=0; copied<size;) {
			const size_t chunk=std::min(size-copied,size_t(staging.size));
			Begin();
			VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
			barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
			vkCmdPipelineBarrier(engine.m_level->m_cmdBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
			VkBufferCopy copy = {offset+copied,0,chunk};
			vkCmdCopyBuffer(engine.m_level->m_cmdBuffer,buffers[index].buffer,staging.buffer,1,&copy);
			barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
			vkCmdPipelineBarrier(engine.m_level->m_cmdBuffer,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,
				0,1,&barrier,0,nullptr,0,nullptr);
			Finish();
			std::memcpy(static_cast<char*>(destination)+copied,staging.mapped,chunk);
			copied+=chunk;
		}
	}
	void Run(uint32_t mode, uint32_t count=0) {
		Begin();
		VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(engine.m_level->m_cmdBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
		vkCmdBindPipeline(engine.m_level->m_cmdBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
		vkCmdBindDescriptorSets(engine.m_level->m_cmdBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,pipelineLayout,0,1,&descriptor,0,nullptr);
		VkPhysicalDeviceProperties properties = {};
		vkGetPhysicalDeviceProperties(engine.m_physicalDevice,&properties);
		const uint32_t maxGroups=properties.limits.maxComputeWorkGroupCount[0];
		const uint64_t invocations=(mode==11||mode==16||mode==26||mode==27||mode==31)?count:
			(mode==10||mode==30||mode==32||mode==33)?2ull*count:(mode==15||mode==18)?4ull*count:3ull*(mode==4&&count?count:cells);
		Require(invocations<=UINT32_MAX-255ull,"dispatch index range exceeded");
		const uint32_t groups=(mode==6||mode==8)?1:uint32_t((invocations+255)/256);
		for (uint32_t group=0; group<groups;) {
			Params params = {mode,group*256,op.dT,uint32_t(op.m_MatAverageMethod),uint32_t(op.m_TimeStepVar),mode==4?upmlOffset:stores,count};
			vkCmdPushConstants(engine.m_level->m_cmdBuffer,pipelineLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(params),&params);
			const uint32_t count=std::min(maxGroups,groups-group);
			vkCmdDispatch(engine.m_level->m_cmdBuffer,count,1,1); group+=count;
		}
		barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT;
		vkCmdPipelineBarrier(engine.m_level->m_cmdBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
		Finish();
	}
public:
	explicit VulkanOperatorSetup(EngineVulkan& e, Operator& o) : engine(e), op(o), cells(0), primCount(0) {}
	~VulkanOperatorSetup() {
		if (pipeline) vkDestroyPipeline(engine.m_device,pipeline,nullptr);
		if (pipelineLayout) vkDestroyPipelineLayout(engine.m_device,pipelineLayout,nullptr);
		if (pool) vkDestroyDescriptorPool(engine.m_device,pool,nullptr);
		if (layout) vkDestroyDescriptorSetLayout(engine.m_device,layout,nullptr);
	}
	void Prepare() {
		Require(typeid(op)==typeid(Operator),"only the basic Cartesian operator is supported");
		std::vector<VulkanPml::Region> pmlRegions;
		for (size_t n=0;n<op.GetNumberOfExtentions();++n) {
			Operator_Extension* extension=op.GetExtension(n);
			Require(typeid(*extension)==typeid(Operator_Ext_UPML)||typeid(*extension)==typeid(Operator_Ext_Excitation)||
				typeid(*extension)==typeid(Operator_Ext_TFSF)||typeid(*extension)==typeid(Operator_Ext_LumpedRLC)||
				typeid(*extension)==typeid(Operator_Ext_SteadyState)||typeid(*extension)==typeid(Operator_Ext_Mur_ABC)||
				typeid(*extension)==typeid(Operator_Ext_Absorbing_BC)||typeid(*extension)==typeid(Operator_Ext_LorentzMaterial)||
				typeid(*extension)==typeid(Operator_Ext_DebyeMaterial)||typeid(*extension)==typeid(Operator_Ext_ConductingSheet),
				(std::string("construction kernel not yet available for ")+extension->GetExtensionName()).c_str());
			if (auto* upml=dynamic_cast<Operator_Ext_UPML*>(extension)) {
				const uint64_t count=3ull*upml->m_numLines[0]*upml->m_numLines[1]*upml->m_numLines[2];
				Require(count<=UINT32_MAX-upmlCount,"UPML construction index range exceeded"); upmlCount+=uint32_t(count);
				for (uint32_t component = 0; component < 3u; ++component) {
					VulkanPml::Region region = {};
					region.component = component;
					for (unsigned int d = 0; d < 3u; ++d) {
						region.origin[d] = upml->m_StartPos[d];
						region.extent[d] = upml->m_numLines[d];
					}
					pmlRegions.push_back(region);
				}
			}
		}
		VulkanPml::Validate(pmlRegions, op.numLines);
		uint64_t size=1;
		for (int n=0;n<3;++n) { Require(op.numLines[n]>=2,"each axis needs two mesh lines"); size*=op.numLines[n]; }
		Require(size<=(UINT32_MAX-255ull)/3,"construction index range exceeded"); cells=uint32_t(size);
		VkPhysicalDeviceProperties properties = {};
		vkGetPhysicalDeviceProperties(engine.m_physicalDevice,&properties);
		Require(engine.m_energyFloat64,"shaderFloat64 is required for GPU setup");
		Require(properties.limits.maxPerStageDescriptorStorageBuffers>=22 && properties.limits.maxDescriptorSetStorageBuffers>=22,
			"22 storage bindings are required for GPU setup");
		Require(properties.limits.maxComputeWorkGroupInvocations>=256 && properties.limits.maxComputeWorkGroupSize[0]>=256,
			"256 construction invocations are required");
		Require(3ull*cells*sizeof(float)<=properties.limits.maxStorageBufferRange,"construction buffer range exceeded");
		Require(uint64_t(upmlCount)*4*sizeof(float)<=properties.limits.maxStorageBufferRange,"aggregate UPML buffer range exceeded");
		VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
		VkPhysicalDeviceMemoryProperties2 memory = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
		memory.pNext=engine.m_memoryBudget?&budget:nullptr;
		vkGetPhysicalDeviceMemoryProperties2(engine.m_physicalDevice,&memory);
		uint64_t available=0;
		for (unsigned int heap=0;heap<memory.memoryProperties.memoryHeapCount;++heap)
			if (memory.memoryProperties.memoryHeaps[heap].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
				const uint64_t limit=engine.m_memoryBudget?budget.heapBudget[heap]:memory.memoryProperties.memoryHeaps[heap].size*3/4;
				const uint64_t used=engine.m_memoryBudget?budget.heapUsage[heap]:0;
				available=std::max(available,limit>used?limit-used:0);
			}
		// Include dense EC/core, material/scan scratch, the largest output and
		// compatibility staging, plus room for runtime fields and driver use.
		const uint64_t estimate=uint64_t(cells)*216+uint64_t(upmlCount)*36+64ull*1024*1024;
		Require(estimate<=std::min(available,engine.m_setupMemoryLimit),"insufficient device memory budget for operator construction");
		std::vector<double> grid;
		unsigned int offsets[3];
		for (int n=0;n<3;++n) {
			offsets[n]=unsigned(grid.size());
			for (unsigned int p=0;p<op.numLines[n];++p) {
				grid.push_back(op.GetDiscLine(n,p)); grid.push_back(op.GetDiscLine(n,p,true));
				grid.push_back(op.GetRawDiscDelta(n,p)); grid.push_back(op.GetRawDiscDelta(n,int(p)-1));
				grid.push_back(op.GetDiscDelta(n,p)*op.gridDelta); grid.push_back(op.GetDiscDelta(n,p,true)*op.gridDelta);
			}
		}
		primitives=op.CSX->GetAllPrimitives(false,CSProperties::PropertyType(CSProperties::MATERIAL|CSProperties::METAL|CSProperties::EXCITATION));
		Require(primitives.size()<UINT32_MAX/48-1,"primitive index range exceeded");
		primCount=uint32_t(primitives.size()+1);
		std::vector<double> geometry(size_t(primCount)*48,0.0);
		for (int n=0;n<3;++n) {
			geometry[32+n*4]=op.GetBackgroundEpsR(); geometry[33+n*4]=op.GetBackgroundKappa();
			geometry[34+n*4]=op.GetBackgroundMueR(); geometry[35+n*4]=op.GetBackgroundSigma();
		}
		for (size_t i=0;i<primitives.size();++i) {
			CSPrimitives* prim=primitives[i]; indices[prim]=uint32_t(i+1);
			double* data=&geometry[(i+1)*48];
			data[4]=data[9]=data[14]=1.0;
			Require(prim->GetCoordinateSystem()!=CYLINDRICAL,"cylindrical primitive coordinates require CPU setup");
			if (prim->HasTransform()) {
				CSTransform inverse(prim->GetTransform());
				double* matrix=inverse.GetMatrix();
				Require(matrix[12]==0 && matrix[13]==0 && matrix[14]==0 && matrix[15]==1,"non-affine transform");
				double determinant=matrix[0]*(matrix[5]*matrix[10]-matrix[6]*matrix[9])-
					matrix[1]*(matrix[4]*matrix[10]-matrix[6]*matrix[8])+matrix[2]*(matrix[4]*matrix[9]-matrix[5]*matrix[8]);
				Require(std::isfinite(determinant) && determinant!=0,"singular primitive transform");
				inverse.Invert(); std::copy(inverse.GetMatrix(),inverse.GetMatrix()+12,data+4);
			}
			const double *start=nullptr,*stop=nullptr;
			if (prim->GetType()==CSPrimitives::BOX) {
				CSPrimBox* box=static_cast<CSPrimBox*>(prim); data[0]=1;
				start=box->GetStartCoord()->GetCartesianCoords(); stop=box->GetStopCoord()->GetCartesianCoords();
			} else if (prim->GetType()==CSPrimitives::SPHERE) {
				CSPrimSphere* sphere=static_cast<CSPrimSphere*>(prim); data[0]=2;
				start=stop=sphere->GetCenter()->GetCartesianCoords(); data[22]=sphere->GetRadius();
			} else if (prim->GetType()==CSPrimitives::CYLINDER) {
				CSPrimCylinder* cylinder=static_cast<CSPrimCylinder*>(prim); data[0]=3;
				start=cylinder->GetAxisStartCoord()->GetCartesianCoords(); stop=cylinder->GetAxisStopCoord()->GetCartesianCoords();
				data[22]=cylinder->GetRadius(); cylinder->GetBoundBox(data+23);
				Require(start[0]!=stop[0]||start[1]!=stop[1]||start[2]!=stop[2],"zero-length cylinder");
			} else throw std::runtime_error("GPU operator setup: unsupported primitive "+prim->GetTypeName());
			std::copy(start,start+3,data+16); std::copy(stop,stop+3,data+19);
			data[1]=prim->GetProperty()->GetType()==CSProperties::METAL?1:0;
			CSPropMaterial* material=dynamic_cast<CSPropMaterial*>(prim->GetProperty());
			if (material) for (int n=0;n<3;++n) {
				Require(ConstantWeight(material->GetEpsilonWeightFunction(n)) && ConstantWeight(material->GetKappaWeightFunction(n)) &&
					ConstantWeight(material->GetMueWeightFunction(n)) && ConstantWeight(material->GetSigmaWeightFunction(n)),
					"spatial material weights require CPU setup");
				double coords[3]={0,0,0};
				data[32+n*4]=material->GetEpsilonWeighted(n,coords); data[33+n*4]=material->GetKappaWeighted(n,coords);
				data[34+n*4]=material->GetMueWeighted(n,coords); data[35+n*4]=material->GetSigmaWeighted(n,coords);
			}
			CSPropExcitation* source=dynamic_cast<CSPropExcitation*>(prim->GetProperty());
			if (source) {
				Require(source->GetWeightFile().empty(),"file-weighted sources require CPU setup");
				Require(std::isfinite(source->GetDelay())&&source->GetDelay()>=0,"invalid source delay");
				data[2]=source->GetEnabled()?1:0; data[3]=source->GetExcitType(); data[30]=source->GetDelay();
				for (int n=0;n<3;++n) {
					Require(ConstantWeight(source->GetWeightFunction(n)),"spatial source weights require CPU setup");
					double coords[3]={0,0,0}; data[32+n*4]=source->GetWeightedExcitation(n,coords);
					data[33+n*4]=source->GetActiveDir(n)?1:0;
				}
			}
			if (auto* sheet=dynamic_cast<CSPropConductingSheet*>(prim->GetProperty())) {
				data[29]=1; data[31]=prim->GetDimension(); prim->GetBoundBox(data+23);
				data[32]=sheet->GetConductivity(); data[33]=sheet->GetThickness();
			}
		}
		for (double value: geometry) Require(std::isfinite(value),"non-finite primitive/material descriptor");
		// Export the reference's exact priority order after its XY bounding-box cull.
		const size_t rows=size_t(op.numLines[0])*op.numLines[1];
		const size_t sourceRows=size_t(op.numLines[1])*op.numLines[2];
		std::vector<uint32_t> candidates(2*(rows+1)+sourceRows+1);
		for (int which=0;which<2;++which) {
			for (unsigned int x=0;x<op.numLines[0];++x) for (unsigned int y=0;y<op.numLines[1];++y) {
				candidates[which*(rows+1)+size_t(x)*op.numLines[1]+y]=uint32_t(candidates.size());
				auto list=op.GetPrimitivesBoundBox(x,y,-1,which==0?CSProperties::MATERIAL:
					CSProperties::PropertyType(CSProperties::MATERIAL|CSProperties::METAL));
				for (CSPrimitives* prim:list) candidates.push_back(indices.at(prim));
			}
			candidates[which*(rows+1)+rows]=uint32_t(candidates.size());
		}
		for (unsigned int z=0;z<op.numLines[2];++z) for (unsigned int y=0;y<op.numLines[1];++y) {
			candidates[2*(rows+1)+size_t(z)*op.numLines[1]+y]=uint32_t(candidates.size());
			auto list=op.GetPrimitivesBoundBox(-1,y,z,CSProperties::EXCITATION);
			for (CSPrimitives* prim:list) candidates.push_back(indices.at(prim));
		}
		candidates[2*(rows+1)+sourceRows]=uint32_t(candidates.size());
		Require(candidates.size()<=UINT32_MAX,"candidate index range exceeded");
		Upload(0,grid.data(),grid.size()*sizeof(double)); Upload(1,geometry.data(),geometry.size()*sizeof(double));
		Upload(2,candidates.data(),candidates.size()*sizeof(uint32_t));
		std::vector<uint32_t> used(primCount+4,0); Upload(3,used.data(),used.size()*sizeof(uint32_t));
		for (unsigned int i=4;i<12;++i) Create(buffers[i],3ull*cells*sizeof(float));
		for (unsigned int i=0;i<4;++i) {
			if (op.m_StoreMaterial[i]) stores|=1u<<i;
			Create(buffers[12+i],op.m_StoreMaterial[i]?3ull*cells*sizeof(float):sizeof(float));
		}
		Create(buffers[16],((3ull*cells+255)/256)*2*sizeof(double));
		Create(buffers[17],sizeof(double)); Create(buffers[18],sizeof(float));
		Create(buffers[19],size_t(upmlCount)*sizeof(uint32_t));
		Create(buffers[20],size_t(upmlCount)*4*sizeof(float)); Create(buffers[21],size_t(upmlCount)*4*sizeof(float));
		Create(staging,std::max(3ull*cells*sizeof(float),uint64_t((primCount+4)*sizeof(uint32_t))),true);
		std::ostringstream source;
		source << "#version 450\n#extension GL_ARB_gpu_shader_fp64 : require\n";
		source << "const uvec3 DIMS=uvec3("<<op.numLines[0]<<","<<op.numLines[1]<<","<<op.numLines[2]<<");\n";
		source << "const uvec3 OFFSETS=uvec3("<<offsets[0]<<","<<offsets[1]<<","<<offsets[2]<<");\n";
		source << "const uint CELLS="<<cells<<"; const uint PEC_INDEX="<<primCount<<"; const uint ERROR_INDEX="<<primCount+3<<";\n";
		source << "const int BC[6]=int[6](";
		for (int n=0;n<6;++n) source<<(n?",":"")<<op.m_BC[n];
		source << ");\n";
		source << "const uint BC_SIZE[6]=uint[6](";
		for (int n=0;n<6;++n) source<<(n?",":"")<<op.GetBCSize(n);
		source << ");\n";
		for (const char* part:kVulkanSetupShaderParts) source << part;
		shaderc::Compiler compiler; shaderc::CompileOptions options;
		options.SetTargetEnvironment(shaderc_target_env_vulkan,shaderc_env_version_vulkan_1_2);
		options.SetOptimizationLevel(shaderc_optimization_level_performance);
		auto result=compiler.CompileGlslToSpv(source.str(),shaderc_glsl_compute_shader,"operator_setup.comp",options);
		if (result.GetCompilationStatus()!=shaderc_compilation_status_success) throw std::runtime_error(result.GetErrorMessage());
		std::vector<uint32_t> spirv(result.begin(),result.end());
		std::array<VkDescriptorSetLayoutBinding,22> bindings = {};
		for (uint32_t n=0;n<bindings.size();++n) { bindings[n].binding=n; bindings[n].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			bindings[n].descriptorCount=1; bindings[n].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
		VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		layoutInfo.bindingCount=uint32_t(bindings.size()); layoutInfo.pBindings=bindings.data();
		Require(vkCreateDescriptorSetLayout(engine.m_device,&layoutInfo,nullptr,&layout)==VK_SUCCESS,"descriptor layout failed");
		VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(Params)};
		VkPipelineLayoutCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		pipelineInfo.setLayoutCount=1; pipelineInfo.pSetLayouts=&layout; pipelineInfo.pushConstantRangeCount=1; pipelineInfo.pPushConstantRanges=&push;
		Require(vkCreatePipelineLayout(engine.m_device,&pipelineInfo,nullptr,&pipelineLayout)==VK_SUCCESS,"pipeline layout failed");
		VkShaderModuleCreateInfo moduleInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
		moduleInfo.codeSize=spirv.size()*sizeof(uint32_t); moduleInfo.pCode=spirv.data();
		VkShaderModule module=VK_NULL_HANDLE;
		Require(vkCreateShaderModule(engine.m_device,&moduleInfo,nullptr,&module)==VK_SUCCESS,"shader module failed");
		VkComputePipelineCreateInfo compute = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
		compute.layout=pipelineLayout; compute.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		compute.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; compute.stage.module=module; compute.stage.pName="main";
		VkResult status=vkCreateComputePipelines(engine.m_device,VK_NULL_HANDLE,1,&compute,nullptr,&pipeline);
		vkDestroyShaderModule(engine.m_device,module,nullptr); Require(status==VK_SUCCESS,"compute pipeline failed");
		VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,22};
		VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		poolInfo.maxSets=1; poolInfo.poolSizeCount=1; poolInfo.pPoolSizes=&poolSize;
		Require(vkCreateDescriptorPool(engine.m_device,&poolInfo,nullptr,&pool)==VK_SUCCESS,"descriptor pool failed");
		VkDescriptorSetAllocateInfo allocate = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		allocate.descriptorPool=pool; allocate.descriptorSetCount=1; allocate.pSetLayouts=&layout;
		Require(vkAllocateDescriptorSets(engine.m_device,&allocate,&descriptor)==VK_SUCCESS,"descriptor set failed");
		UpdateDescriptors();
	}
	void UpdateDescriptors() {
		std::array<VkDescriptorBufferInfo,22> infos = {};
		std::array<VkWriteDescriptorSet,22> writes = {};
		for (uint32_t n=0;n<infos.size();++n) {
			infos[n]={buffers[n].buffer,0,buffers[n].size}; writes[n].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writes[n].dstSet=descriptor; writes[n].dstBinding=n; writes[n].descriptorCount=1;
			writes[n].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[n].pBufferInfo=&infos[n];
		}
		vkUpdateDescriptorSets(engine.m_device,uint32_t(writes.size()),writes.data(),0,nullptr);
	}
	void CalcEC(Operator&) override {
		Run(0);
		std::vector<uint32_t> used(primCount+4); Download(3,used.data(),used.size()*sizeof(uint32_t));
		Require(!used.back(),"effective material parameter is NaN/Inf");
		ArrayLib::ArrayNIJK<float>* materials[4]={op.m_epsR_ptr,op.m_kappa_ptr,op.m_mueR_ptr,op.m_sigma_ptr};
		for (unsigned int kind=0;kind<4;++kind) if (materials[kind])
			Download(12+kind,materials[kind]->data(),3ull*cells*sizeof(float));
	}
	void CalcTimestep(Operator&) override {
		Run(1);
		std::vector<double> results(((3ull*cells+255)/256)*2);
		Download(16,results.data(),results.size()*sizeof(double));
		op.dT=1e200; uint32_t winner=UINT32_MAX;
		for (size_t n=0;n<results.size();n+=2) if (results[n]<op.dT || (results[n]==op.dT && results[n+1]<winner)) {
			op.dT=results[n]; winner=uint32_t(results[n+1]);
		}
		Require(op.dT>0 && op.dT<1e200,"no positive stable timestep");
		op.m_Used_TS_Name=op.m_TimeStepVar==3?"Rennings_2":"Rennings_1";
		op.MainOp->SetReflection2Cell();
	}
	void CalcCoefficients(Operator&) override {
		Run(2);
	}
	void CalcPEC(Operator&) override {
		Run(3);
	}
	void SynchronizeOperator(Operator&) override {
		ArrayLib::ArrayNIJK<float>* matrices[4]={op.vv_ptr,op.vi_ptr,op.ii_ptr,op.iv_ptr};
		for (unsigned int n=0;n<4;++n) Download(8+n,matrices[n]->data(),3ull*cells*sizeof(float));
		std::vector<uint32_t> used(primCount+4); Download(3,used.data(),used.size()*sizeof(uint32_t));
		for (unsigned int n=0;n<3;++n) op.m_Nr_PEC[n]=used[primCount+n];
		for (size_t n=0;n<primitives.size();++n) if (used[n+1]) primitives[n]->SetPrimitiveUsed(true);
	}
	bool BuildExtension(Operator&, Operator_Extension& extension) override {
		if (auto* source=dynamic_cast<Operator_Ext_Excitation*>(&extension)) return BuildExcitation(*source);
		if (auto* mur=dynamic_cast<Operator_Ext_Mur_ABC*>(&extension)) return BuildMur(*mur);
		if (auto* abc=dynamic_cast<Operator_Ext_Absorbing_BC*>(&extension)) return BuildAbsorber(*abc);
		if (auto* sheet=dynamic_cast<Operator_Ext_ConductingSheet*>(&extension)) return BuildSheet(*sheet);
		if (auto* debye=dynamic_cast<Operator_Ext_DebyeMaterial*>(&extension)) return BuildDebye(*debye);
		if (auto* rlc=dynamic_cast<Operator_Ext_LumpedRLC*>(&extension)) return BuildRLC(*rlc);
		if (auto* lorentz=dynamic_cast<Operator_Ext_LorentzMaterial*>(&extension)) return BuildLorentz(*lorentz);
		Operator_Ext_UPML* upml=dynamic_cast<Operator_Ext_UPML*>(&extension);
		if (!upml) return extension.BuildExtension();
		// Only O(Nx+Ny+Nz) grading values are evaluated by the host parser.
		const size_t stride=op.numLines[0]+op.numLines[1]+op.numLines[2];
		std::vector<double> tables(6+6*stride);
		for (int n=0;n<3;++n) { tables[n]=upml->m_StartPos[n]; tables[3+n]=upml->m_numLines[n]; }
		for (unsigned int component=0;component<3;++component) {
			size_t offset=0;
			for (unsigned int axis=0;axis<3;++axis) {
				for (unsigned int index=0;index<op.numLines[axis];++index) {
					unsigned int p[3]={0,0,0}; p[axis]=index;
					double kv[3]={0,0,0},ki[3]={0,0,0};
					upml->CalcGradingKappa(component,p,Z0,kv,ki);
					tables[6+component*2*stride+offset+index]=kv[axis];
					tables[6+(component*2+1)*stride+offset+index]=ki[axis];
				}
				offset+=op.numLines[axis];
			}
		}
		const uint32_t count=upml->m_numLines[0]*upml->m_numLines[1]*upml->m_numLines[2];
		buffers[17].Release(); Upload(17,tables.data(),tables.size()*sizeof(double));
		buffers[18].Release(); Create(buffers[18],18ull*count*sizeof(float));
		UpdateDescriptors(); Run(4,count);
		upmlOffset+=3*count;
		ArrayLib::ArrayNIJK<float>* outputs[6]={&upml->vv,&upml->vvfo,&upml->vvfn,&upml->ii,&upml->iifo,&upml->iifn};
		for (unsigned int n=0;n<6;++n) {
			outputs[n]->Init("GPU UPML",upml->m_numLines);
			Download(18,outputs[n]->data(),3ull*count*sizeof(float),n*3ull*count*sizeof(float));
		}
		return true;
	}
	void SetAuxiliary(const std::vector<double>& values) {
		for (double value:values) Require(std::isfinite(value),"non-finite extension descriptor");
		buffers[17].Release(); Upload(17,values.data(),values.size()*sizeof(double)); UpdateDescriptors();
	}
	void AppendPrepared(Buffer& destination, size_t size) {
		if (!size) return;
		Require(size<=buffers[18].size && destination.size<=UINT32_MAX-size,"prepared extension range exceeded");
		Buffer combined; Create(combined,size_t(destination.size)+size);
		Begin();
		VkMemoryBarrier barrier={VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
		vkCmdPipelineBarrier(engine.m_level->m_cmdBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
		if (destination.size) { VkBufferCopy copy={0,0,destination.size}; vkCmdCopyBuffer(engine.m_level->m_cmdBuffer,destination.buffer,combined.buffer,1,&copy); }
		VkBufferCopy copy={0,destination.size,size}; vkCmdCopyBuffer(engine.m_level->m_cmdBuffer,buffers[18].buffer,combined.buffer,1,&copy);
		Finish(); destination.Swap(combined); engine.m_setupPreparedBuffers.insert(&destination);
	}
	void PackDispersive(std::vector<double> metadata, bool magnetic) {
		metadata[3]=magnetic?1:0; SetAuxiliary(metadata); Run(21); Run(6);
		double total=0; Download(16,&total,sizeof(total),((3ull*cells+255)/256)*sizeof(double));
		Require(total>=0 && total<=3ull*cells,"invalid ADE point count");
		const size_t size=size_t(total)*sizeof(EngineVulkan::GpuDispersivePoint);
		buffers[18].Release(); Create(buffers[18],std::max(size,size_t(4))); UpdateDescriptors(); Run(22);
		AppendPrepared(magnetic?engine.m_level->m_bufDispCurrParams:engine.m_level->m_bufDispVoltParams,size);
	}
	void AllocateScan() {
		for (unsigned int n=12;n<14;++n) { buffers[n].Release(); Create(buffers[n],3ull*cells*sizeof(float)); }
		UpdateDescriptors();
	}
	std::vector<uint32_t> CompactNodes(std::vector<double>& metadata) {
		SetAuxiliary(metadata);
		Run(12); Run(6);
		double total=0; Download(16,&total,sizeof(total),((3ull*cells+255)/256)*sizeof(double));
		Require(total>=0 && total<=cells,"invalid sparse construction count");
		const uint32_t count=uint32_t(total);
		buffers[18].Release(); Create(buffers[18],std::max(3ull*count*sizeof(float),4ull)); UpdateDescriptors();
		Run(14,count);
		std::vector<uint32_t> nodes(count);
		if (count) Download(18,nodes.data(),nodes.size()*sizeof(uint32_t));
		return nodes;
	}
	void CopySparsePositions(Operator_Ext_Dispersive& material, unsigned int order, const std::vector<uint32_t>& nodes) {
		material.m_LM_Count[order]=uint32_t(nodes.size());
		material.m_LM_pos[order]=new unsigned int*[3]();
		for (unsigned int n=0;n<3;++n) {
			material.m_LM_pos[order][n]=new unsigned int[nodes.size()];
			for (size_t q=0;q<nodes.size();++q) {
				const uint32_t p[3]={nodes[q]/(op.numLines[1]*op.numLines[2]),(nodes[q]/op.numLines[2])%op.numLines[1],nodes[q]%op.numLines[2]};
				material.m_LM_pos[order][n][q]=p[n];
			}
		}
	}
	void CopySparseCoefficients(std::vector<double>& metadata, uint32_t count, float**& destination) {
		SetAuxiliary(metadata); Run(13,count);
		destination=new float*[3]();
		for (unsigned int n=0;n<3;++n) {
			destination[n]=new float[count];
			if (count) Download(18,destination[n],size_t(count)*sizeof(float),size_t(n)*count*sizeof(float));
		}
	}
	bool BuildLorentz(Operator_Ext_LorentzMaterial& material) {
		Require(material.m_LM_pos==nullptr,"rebuilding a constructed Lorentz extension requires a fresh operator");
		int order=0;
		for (CSProperties* property:op.CSX->GetPropertyByType(CSProperties::LORENTZMATERIAL))
			order=std::max(order,static_cast<CSPropLorentzMaterial*>(property)->GetDispersionOrder());
		if (!order) return false;
		std::vector<double> metadata(4+size_t(primCount)*order*18,0.0); metadata[1]=order;
		double coord[3]={0,0,0};
		for (size_t p=0;p<primitives.size();++p) if (auto* mat=dynamic_cast<CSPropLorentzMaterial*>(primitives[p]->GetProperty()))
			for (int pole=0;pole<mat->GetDispersionOrder();++pole) for (int n=0;n<3;++n) {
				Require(ConstantWeight(mat->GetEpsPlasmaFreqWeightFunction(pole,n)) && ConstantWeight(mat->GetEpsRelaxTimeWeightFunction(pole,n)) &&
					ConstantWeight(mat->GetEpsLorPoleFreqWeightFunction(pole,n)) && ConstantWeight(mat->GetMuePlasmaFreqWeightFunction(pole,n)) &&
					ConstantWeight(mat->GetMueRelaxTimeWeightFunction(pole,n)) && ConstantWeight(mat->GetMueLorPoleFreqWeightFunction(pole,n)),
					"spatial dispersion weights require CPU setup");
				double* data=&metadata[4+((p+1)*order+pole)*18+n*6];
				data[0]=mat->GetEpsPlasmaFreqWeighted(pole,n,coord); data[1]=mat->GetEpsRelaxTimeWeighted(pole,n,coord);
				data[2]=mat->GetEpsLorPoleFreqWeighted(pole,n,coord); data[3]=mat->GetMuePlasmaFreqWeighted(pole,n,coord);
				data[4]=mat->GetMueRelaxTimeWeighted(pole,n,coord); data[5]=mat->GetMueLorPoleFreqWeighted(pole,n,coord);
			}
		material.m_Order=order; material.m_LM_Count.assign(order,0); material.m_LM_pos=new unsigned int**[order]();
		material.m_volt_ADE_On=new bool[order](); material.m_curr_ADE_On=new bool[order]();
		material.m_volt_Lor_ADE_On=new bool[order](); material.m_curr_Lor_ADE_On=new bool[order]();
		float**** outputs[6]={&material.v_int_ADE,&material.v_ext_ADE,&material.v_Lor_ADE,&material.i_int_ADE,&material.i_ext_ADE,&material.i_Lor_ADE};
		for (auto output:outputs) *output=new float**[order]();
		AllocateScan();
		for (int pole=0;pole<order;++pole) {
			metadata[2]=pole;
			uint32_t flags=0; Begin();
			vkCmdFillBuffer(engine.m_level->m_cmdBuffer,buffers[3].buffer,size_t(primCount+3)*sizeof(uint32_t),sizeof(uint32_t),0);
			Finish();
			auto nodes=CompactNodes(metadata); CopySparsePositions(material,pole,nodes);
			Download(3,&flags,sizeof(flags),size_t(primCount+3)*sizeof(uint32_t)); flags>>=4;
			material.m_volt_ADE_On[pole]=(flags&1)!=0; material.m_curr_ADE_On[pole]=(flags&2)!=0;
			material.m_volt_Lor_ADE_On[pole]=(flags&4)!=0; material.m_curr_Lor_ADE_On[pole]=(flags&8)!=0;
			const bool enabled[6]={(flags&1)!=0,(flags&1)!=0,(flags&4)!=0,(flags&2)!=0,(flags&2)!=0,(flags&8)!=0};
			for (unsigned int kind=0;kind<6;++kind) if (enabled[kind]) {
				metadata[3]=kind; CopySparseCoefficients(metadata,uint32_t(nodes.size()),(*outputs[kind])[pole]);
			}
			if (flags&1) PackDispersive(metadata,false);
			if (flags&2) PackDispersive(metadata,true);
		}
		return true;
	}
	bool BuildDebye(Operator_Ext_DebyeMaterial& material) {
		Require(material.m_LM_pos==nullptr,"rebuilding a constructed Debye extension requires a fresh operator");
		int poles=0;
		for (CSProperties* property:op.CSX->GetPropertyByType(CSProperties::DEBYEMATERIAL))
			poles=std::max(poles,static_cast<CSPropDebyeMaterial*>(property)->GetDispersionOrder());
		if (!poles) return false;
		std::vector<double> metadata(4+size_t(primCount)*poles*6,0.0); metadata[0]=1; metadata[1]=poles;
		double coord[3]={0,0,0};
		for (size_t p=0;p<primitives.size();++p) if (auto* mat=dynamic_cast<CSPropDebyeMaterial*>(primitives[p]->GetProperty()))
			for (int pole=0;pole<mat->GetDispersionOrder();++pole) for (int n=0;n<3;++n) {
				Require(ConstantWeight(mat->GetEpsDeltaWeightFunction(pole,n))&&ConstantWeight(mat->GetEpsRelaxTimeWeightFunction(pole,n)),
					"spatial dispersion weights require CPU setup");
				metadata[4+((p+1)*poles+pole)*6+n*2]=mat->GetEpsDeltaWeighted(pole,n,coord);
				metadata[5+((p+1)*poles+pole)*6+n*2]=mat->GetEpsRelaxTimeWeighted(pole,n,coord);
			}
		AllocateScan(); auto nodes=CompactNodes(metadata);
		if (nodes.empty()) return false;
		material.m_PoleCount=poles; material.m_Order=1; material.m_LM_Count.assign(1,0);
		material.m_LM_pos=new unsigned int**[1](); CopySparsePositions(material,0,nodes);
		material.m_volt_ADE_On=new bool[1]{true}; material.m_curr_ADE_On=new bool[1]{false};
		material.v_relax_ADE=new float**[poles](); material.v_drive_ADE=new float**[poles]();
		metadata[3]=2; CopySparseCoefficients(metadata,uint32_t(nodes.size()),material.v_solve_ADE);
		for (int pole=0;pole<poles;++pole) {
			metadata[2]=pole; metadata[3]=0; CopySparseCoefficients(metadata,uint32_t(nodes.size()),material.v_relax_ADE[pole]);
			metadata[3]=1; CopySparseCoefficients(metadata,uint32_t(nodes.size()),material.v_drive_ADE[pole]);
		}
		uint32_t flags=0; Download(3,&flags,sizeof(flags),size_t(primCount+3)*sizeof(uint32_t));
		if (flags&2) std::cerr << "GPU Debye setup: relaxation time needs at least 3 timesteps; skipping this pole." << std::endl;
		metadata[3]=double(engine.m_level->m_bufDebyePoles.size/sizeof(EngineVulkan::GpuDebyePole)); SetAuxiliary(metadata);
		const size_t pointBytes=3*nodes.size()*sizeof(EngineVulkan::GpuDebyePoint),poleBytes=3*nodes.size()*poles*sizeof(EngineVulkan::GpuDebyePole);
		buffers[18].Release(); Create(buffers[18],pointBytes); UpdateDescriptors(); Run(24);
		AppendPrepared(engine.m_level->m_bufDebyeParams,pointBytes);
		buffers[18].Release(); Create(buffers[18],poleBytes); UpdateDescriptors(); Run(25);
		AppendPrepared(engine.m_level->m_bufDebyePoles,poleBytes);
		return true;
	}
	bool BuildAbsorber(Operator_Ext_Absorbing_BC& abc) {
		Require(abc.m_ny>=0 && abc.m_ny<3 && abc.m_phaseVelocity>0,"invalid local absorber metadata");
		abc.m_nyP=(abc.m_ny+1)%3; abc.m_nyPP=(abc.m_ny+2)%3;
		abc.m_numLines[0]=abc.m_sheetX1[abc.m_nyP]-abc.m_sheetX0[abc.m_nyP]+1;
		abc.m_numLines[1]=abc.m_sheetX1[abc.m_nyPP]-abc.m_sheetX0[abc.m_nyPP]+1;
		abc.m_numCells=abc.m_numLines[0]*abc.m_numLines[1];
		unsigned int p[3]={0,0,0}; p[abc.m_ny]=abc.m_sheetX0[abc.m_ny];
		SetAuxiliary({abc.m_phaseVelocity,std::abs(op.GetEdgeLength(abc.m_ny,p)),double(abc.m_ny),
			double(abc.m_sheetX0[0]),double(abc.m_sheetX0[1]),double(abc.m_sheetX0[2]),
			double(abc.m_numLines[0]),double(abc.m_numLines[1]),abc.m_normalSignPositive?1.0:0.0});
		buffers[18].Release(); Create(buffers[18],4ull*abc.m_numCells*sizeof(float)); UpdateDescriptors(); Run(15,abc.m_numCells);
		ArrayLib::ArrayIJ<float>* outputs[4]={&abc.m_K1_nyP,&abc.m_K1_nyPP,&abc.m_K2_nyP,&abc.m_K2_nyPP};
		const unsigned int arrays=abc.m_ABCtype==Operator_Ext_Absorbing_BC::MUR_1ST_SA?4:2;
		for (unsigned int kind=0;kind<arrays;++kind) {
			outputs[kind]->Init("GPU local absorber",abc.m_numLines);
			Download(18,outputs[kind]->data(),size_t(abc.m_numCells)*sizeof(float),size_t(kind)*abc.m_numCells*sizeof(float));
		}
		buffers[15].Release(); buffers[15].Swap(buffers[18]);
		Create(buffers[18],2ull*abc.m_numCells*sizeof(EngineVulkan::GpuAbcVoltPoint)); UpdateDescriptors(); Run(32,abc.m_numCells);
		AppendPrepared(engine.m_level->m_bufAbcVoltParams,2ull*abc.m_numCells*sizeof(EngineVulkan::GpuAbcVoltPoint));
		if (arrays==4 && abc.m_numLines[0]>1 && abc.m_numLines[1]>1) {
			const uint32_t count=(abc.m_numLines[0]-1)*(abc.m_numLines[1]-1);
			buffers[18].Release(); Create(buffers[18],2ull*count*sizeof(EngineVulkan::GpuAbcCurrPoint)); UpdateDescriptors(); Run(33,count);
			AppendPrepared(engine.m_level->m_bufAbcCurrParams,2ull*count*sizeof(EngineVulkan::GpuAbcCurrPoint));
		}
		return true;
	}
	bool BuildSheet(Operator_Ext_ConductingSheet& sheet) {
		// These fixed fitting tables are also used by the reference constructor.
#include "FDTD/extensions/cond_sheet_parameter.h"
		Require(sheet.m_LM_pos==nullptr,"rebuilding a constructed sheet extension requires a fresh operator");
		std::vector<double> metadata(2+numOptPara*6); metadata[0]=sheet.m_f_max; metadata[1]=numOptPara;
		for (unsigned int q=0;q<numOptPara;++q) {
			metadata[2+q*6]=omega_stop[q]; metadata[3+q*6]=g[q]; metadata[4+q*6]=l1[q];
			metadata[5+q*6]=l2[q]; metadata[6+q*6]=r1[q]; metadata[7+q*6]=r2[q];
		}
		AllocateScan(); buffers[14].Release(); Create(buffers[14],3ull*cells*sizeof(float)); SetAuxiliary(metadata);
		Run(19); Run(6);
		double total=0; Download(16,&total,sizeof(total),((3ull*cells+255)/256)*sizeof(double));
		Require(total>=0 && total<=cells,"invalid conducting sheet cell count");
		const uint32_t count=uint32_t(total); if (!count) return false;
		buffers[18].Release(); Create(buffers[18],12ull*count*sizeof(float)); UpdateDescriptors(); Run(14,count);
		std::vector<uint32_t> nodes(count); Download(18,nodes.data(),nodes.size()*sizeof(uint32_t));
		Run(20,count);
		sheet.m_Order=2; sheet.m_LM_Count.assign(2,0); sheet.m_LM_pos=new unsigned int**[2]();
		for (int order=0;order<2;++order) CopySparsePositions(sheet,order,nodes);
		sheet.m_volt_ADE_On=new bool[2]{true,true}; sheet.m_curr_ADE_On=new bool[2]{false,false};
		sheet.m_volt_Lor_ADE_On=new bool[2]{false,false}; sheet.m_curr_Lor_ADE_On=new bool[2]{false,false};
		sheet.v_int_ADE=new float**[2](); sheet.v_ext_ADE=new float**[2]();
		for (int order=0;order<2;++order) {
			sheet.v_int_ADE[order]=new float*[3](); sheet.v_ext_ADE[order]=new float*[3]();
			for (int n=0;n<3;++n) {
				sheet.v_int_ADE[order][n]=new float[count]; sheet.v_ext_ADE[order][n]=new float[count];
				Download(18,sheet.v_int_ADE[order][n],size_t(count)*sizeof(float),size_t(order*6+n)*count*sizeof(float));
				Download(18,sheet.v_ext_ADE[order][n],size_t(count)*sizeof(float),size_t(order*6+3+n)*count*sizeof(float));
			}
		}
		buffers[15].Release(); buffers[15].Swap(buffers[18]); Create(buffers[18],sizeof(float)); UpdateDescriptors(); Run(23);
		std::vector<double> packing={2.0,double(count),0.0,0.0}; PackDispersive(packing,false);
		packing[2]=1; PackDispersive(packing,false);
		return true;
	}
	bool BuildRLC(Operator_Ext_LumpedRLC& rlc) {
		Require(rlc.RLC_count==0,"rebuilding a constructed RLC extension requires a fresh operator");
		std::vector<uint32_t> nodes; std::vector<int> directions; std::vector<float> values[9];
		for (CSProperties* property:op.CSX->GetPropertyByType(CSProperties::LUMPED_ELEMENT)) {
			auto* load=static_cast<CSPropLumpedElement*>(property);
			if (!rlc.IsLElumpedRLC(load)) continue;
			const int n=load->GetDirection(),a=(n+1)%3,b=(n+2)%3;
			if (n<0||n>2) { std::cerr << "GPU RLC setup: invalid direction; skipping load." << std::endl; continue; }
			double R=load->GetResistance(),L=load->GetInductance(),C=load->GetCapacity();
			for (double* value:{&R,&L,&C}) if (*value<0) { *value=0; std::cerr << "GPU RLC setup: negative component value clamped to zero." << std::endl; }
			for (CSPrimitives* primitive:load->GetAllPrimitives()) {
				auto* box=dynamic_cast<CSPrimBox*>(primitive); if (!box) continue;
				unsigned int start[3],stop[3];
				if (op.SnapBox2Mesh(box->GetStartCoord()->GetCoords(op.m_MeshType),box->GetStopCoord()->GetCoords(op.m_MeshType),start,stop,false,true)<=0 || start[n]==stop[n]) continue;
				const uint32_t series=stop[n]-start[n],parallel=(stop[a]-start[a]+1)*(stop[b]-start[b]+1),count=series*parallel;
				const double dL=std::isnan(L)?0:L*parallel/series,dR=std::isnan(R)?0:R*parallel/series;
				const double dG=std::isnan(R)||R==0?0:(1/R)*series/parallel,dC=std::isnan(C)?0:C*series/parallel;
				double ib0=0,b1=0,b2=0;
				if (load->GetLEtype()==CSPropLumpedElement::SERIES) {
					if (dC==0) { ib0=op.dT/(2*dL+op.dT*dR); b1=-4*dL/op.dT; b2=(2*dL-op.dT*dR)/op.dT; }
					else { ib0=2*op.dT*dC/(4*dL*dC+2*op.dT*dR*dC+op.dT*op.dT); b1=(op.dT*op.dT-4*dL*dC)/(op.dT*dC); b2=(4*dL*dC-2*op.dT*dR*dC+op.dT*op.dT)/(2*op.dT*dC); }
				}
				std::vector<double> metadata(18);
				for (int axis=0;axis<3;++axis) { metadata[axis]=start[axis]; metadata[3+axis]=stop[axis]; }
				metadata[6]=n; metadata[7]=load->GetLEtype()==CSPropLumpedElement::SERIES?1:0;
				metadata[8]=dL; metadata[9]=dR; metadata[10]=dG; metadata[11]=dC; metadata[12]=R>=0?1:0;
				metadata[13]=L>0?1:0; metadata[14]=op.GetExcitationSignal()->GetCenterFreq()+op.GetExcitationSignal()->GetCutOffFreq();
				metadata[15]=ib0; metadata[16]=b1; metadata[17]=b2; SetAuxiliary(metadata);
				buffers[18].Release(); Create(buffers[18],10ull*count*sizeof(float)); UpdateDescriptors(); Run(16,count);
				std::vector<float> result(10ull*count); Download(18,result.data(),result.size()*sizeof(float));
				for (uint32_t q=0;q<count;++q) {
					uint32_t node; std::memcpy(&node,&result[q*10],sizeof(node)); nodes.push_back(node); directions.push_back(n);
					for (int kind=0;kind<9;++kind) values[kind].push_back(result[q*10+kind+1]);
				}
				buffers[15].Release(); buffers[15].Swap(buffers[18]); Create(buffers[18],uint64_t(count)*sizeof(EngineVulkan::GpuRlcParam));
				UpdateDescriptors(); Run(26,count); AppendPrepared(engine.m_level->m_bufRlcParams,uint64_t(count)*sizeof(EngineVulkan::GpuRlcParam));
				if (load->GetCaps()) {
					std::vector<double> caps(12,0.0);
					for (int axis=0;axis<3;++axis) { caps[axis]=start[axis]; caps[3+axis]=stop[axis]; }
					caps[6]=n; caps[11]=1; SetAuxiliary(caps); Run(17);
				}
				primitive->SetPrimitiveUsed(true);
			}
		}
		rlc.RLC_count=uint32_t(nodes.size()); if (nodes.empty()) return true;
		rlc.v_RLC_dir=new int[nodes.size()]; std::copy(directions.begin(),directions.end(),rlc.v_RLC_dir);
		rlc.v_RLC_pos=new unsigned int*[3]();
		for (int n=0;n<3;++n) {
			rlc.v_RLC_pos[n]=new unsigned int[nodes.size()];
			for (size_t q=0;q<nodes.size();++q) rlc.v_RLC_pos[n][q]=n==0?nodes[q]/(op.numLines[1]*op.numLines[2]):n==1?(nodes[q]/op.numLines[2])%op.numLines[1]:nodes[q]%op.numLines[2];
		}
		float** targets[9]={&rlc.v_RLC_ilv,&rlc.v_RLC_i2v,&rlc.v_RLC_vv2,&rlc.v_RLC_vj1,&rlc.v_RLC_vj2,&rlc.v_RLC_vvd,&rlc.v_RLC_ib0,&rlc.v_RLC_b1,&rlc.v_RLC_b2};
		for (int kind=0;kind<9;++kind) { *targets[kind]=new float[nodes.size()]; std::copy(values[kind].begin(),values[kind].end(),*targets[kind]); }
		return true;
	}
	bool BuildTFSF(Operator_Ext_TFSF& source, const double origin[3]) override {
		source.m_maxDelay=0;
		for (int axis=0;axis<3;++axis) for (int side=0;side<2;++side) {
			if (!source.m_ActiveDir[axis][side]) continue;
			const int a=(axis+1)%3,b=(axis+2)%3;
			const uint32_t count=source.m_numLines[a]*source.m_numLines[b];
			std::vector<double> metadata(23); metadata[0]=axis; metadata[1]=side;
			for (int n=0;n<3;++n) {
				metadata[2+n]=source.m_Start[n]; metadata[5+n]=source.m_Stop[n]; metadata[8+n]=origin[n];
				metadata[11+n]=source.m_PropDir[n]; metadata[14+n]=source.m_E_Amp[n]; metadata[17+n]=source.m_H_Amp[n];
			}
			metadata[20]=source.m_PhVel; metadata[21]=op.GetGridDelta(); SetAuxiliary(metadata);
			buffers[18].Release(); Create(buffers[18],12ull*count*sizeof(float)); UpdateDescriptors(); Run(18,count);
			std::vector<float> points(12ull*count); Download(18,points.data(),points.size()*sizeof(float));
			for (unsigned int part=0;part<4;++part) {
				const int component=part%2;
				unsigned int*& delays=part<2?source.m_CurrDelay[axis][side][component]:source.m_VoltDelay[axis][side][component];
				float*& deltas=part<2?source.m_CurrDelayDelta[axis][side][component]:source.m_VoltDelayDelta[axis][side][component];
				float*& amplitudes=part<2?source.m_CurrAmp[axis][side][component]:source.m_VoltAmp[axis][side][component];
				delays=new unsigned int[count]; deltas=new float[count]; amplitudes=new float[count];
				for (uint32_t q=0;q<count;++q) {
					const float* point=&points[(part*count+q)*3]; std::memcpy(&delays[q],point,sizeof(uint32_t));
					deltas[q]=point[1]; amplitudes[q]=point[2]; source.m_maxDelay=std::max(source.m_maxDelay,delays[q]);
				}
			}
			buffers[15].Release(); buffers[15].Swap(buffers[18]); Create(buffers[18],sizeof(float));
			for (unsigned int voltage=0;voltage<2;++voltage) {
				metadata[22]=voltage; SetAuxiliary(metadata); Run(28,count); Run(6);
				double total=0; Download(16,&total,sizeof(total),((3ull*cells+255)/256)*sizeof(double));
				Require(total>=0 && total<=2ull*count,"invalid TFSF point count");
				const size_t bytes=size_t(total)*sizeof(EngineVulkan::GpuTfsfPoint);
				buffers[18].Release(); Create(buffers[18],std::max(bytes,size_t(4))); UpdateDescriptors(); Run(29,count);
				AppendPrepared(voltage?engine.m_level->m_bufTfsfVoltParams:engine.m_level->m_bufTfsfCurrParams,bytes);
			}
		}
		++source.m_maxDelay; return true;
	}
	bool BuildMur(Operator_Ext_Mur_ABC& mur) {
		if (mur.m_ny<0) return false;
		unsigned int p[3]={0,0,0}; p[mur.m_ny]=mur.m_LineNr;
		const double delta=std::abs(op.GetEdgeLength(mur.m_ny,p));
		std::vector<double> metadata(6+mur.m_numLines[0]);
		metadata[0]=mur.m_ny; metadata[1]=mur.m_LineNr; metadata[2]=mur.m_v_phase;
		metadata[3]=op.GetDiscLine(mur.m_ny,mur.m_LineNr)+(mur.m_LineNr==0?delta:-delta)/2/op.GetGridDelta();
		metadata[4]=delta;
		for (unsigned int row=0;row<mur.m_numLines[0];++row) {
			metadata[5+row]=double(metadata.size());
			int bounds[3]={-1,-1,-1}; bounds[mur.m_ny]=mur.m_LineNr; bounds[mur.m_nyP]=row;
			for (CSPrimitives* prim:op.GetPrimitivesBoundBox(bounds[0],bounds[1],bounds[2],CSProperties::MATERIAL))
				metadata.push_back(indices.at(prim));
		}
		metadata[5+mur.m_numLines[0]]=double(metadata.size());
		buffers[17].Release(); Upload(17,metadata.data(),metadata.size()*sizeof(double));
		const uint32_t count=mur.m_numLines[0]*mur.m_numLines[1];
		buffers[18].Release(); Create(buffers[18],2ull*count*sizeof(float)); UpdateDescriptors(); Run(10,count);
		Download(18,mur.m_Mur_Coeff_nyP.data(),size_t(count)*sizeof(float));
		Download(18,mur.m_Mur_Coeff_nyPP.data(),size_t(count)*sizeof(float),size_t(count)*sizeof(float));
		buffers[15].Release(); buffers[15].Swap(buffers[18]); Create(buffers[18],2ull*count*sizeof(EngineVulkan::GpuMurPoint));
		UpdateDescriptors(); Run(30,count); AppendPrepared(engine.m_level->m_bufMurParams,2ull*count*sizeof(EngineVulkan::GpuMurPoint));
		return true;
	}
	bool BuildExcitation(Operator_Ext_Excitation& source) {
		source.Reset(); source.m_Exc=op.GetExcitationSignal();
		if (!source.m_Exc || op.dT==0) return false;
		if (op.CSX->GetPropertyByType(CSProperties::EXCITATION).empty()) {
			std::cerr << "Operator::CalcFieldExcitation: Warning, no excitation properties found" << std::endl;
			return false;
		}
		for (CSProperties* property:op.CSX->GetPropertyByType(CSProperties::EXCITATION))
			Require(static_cast<CSPropExcitation*>(property)->GetDelay()/op.dT<=INT32_MAX,"source delay exceeds runtime index range");
		for (unsigned int n=0;n<2;++n) {
			buffers[12+n].Release(); Create(buffers[12+n],3ull*cells*sizeof(float));
		}
		for (unsigned int magnetic=0;magnetic<2;++magnetic) {
			UpdateDescriptors(); Run(5,magnetic); Run(6);
			double pointCount=0;
			Download(16,&pointCount,sizeof(pointCount),((3ull*cells+255)/256)*sizeof(double));
			const uint32_t count=uint32_t(pointCount);
			buffers[18].Release(); Create(buffers[18],std::max(24ull*count,4ull));
			UpdateDescriptors(); Run(7,magnetic);
			std::vector<EngineVulkan::GpuExcPoint> points(count);
			if (count) Download(18,points.data(),points.size()*sizeof(EngineVulkan::GpuExcPoint));
			std::vector<unsigned int> positions[3],delays,directions;
			std::vector<float> amplitudes;
			for (const auto& point:points) {
				const uint32_t node=point.index%cells;
				directions.push_back(point.index/cells); delays.push_back(point.delay); amplitudes.push_back(point.amplitude);
				positions[0].push_back(node/(op.numLines[1]*op.numLines[2]));
				positions[1].push_back((node/op.numLines[2])%op.numLines[1]); positions[2].push_back(node%op.numLines[2]);
			}
			if (magnetic) source.setupCurrentExcitation(positions,amplitudes,delays,directions);
			else source.setupVoltageExcitation(positions,amplitudes,delays,directions);
			if (count) buffers[18].Swap(magnetic?engine.m_level->m_bufCurrExcPoints:engine.m_level->m_bufVoltExcPoints);
			buffers[18].Release(); Create(buffers[18],sizeof(float));
		}
		return true;
	}
	double LumpedGeometry(int direction, const unsigned int start[3], const unsigned int stop[3]) override {
		double data[7];
		for (int n=0;n<3;++n) { data[n]=start[n]; data[3+n]=stop[n]; } data[6]=direction;
		buffers[17].Release(); Upload(17,data,sizeof(data)); UpdateDescriptors(); Run(8);
		double result=0; Download(16,&result,sizeof(result)); return result;
	}
	void ApplyLumpedRC(int direction, const unsigned int start[3], const unsigned int stop[3],
		double capacitance, double resistance, double epsilon, double kappa, bool caps) override {
		double data[12];
		for (int n=0;n<3;++n) { data[n]=start[n]; data[3+n]=stop[n]; }
		data[6]=direction; data[7]=capacitance; data[8]=resistance; data[9]=epsilon; data[10]=kappa; data[11]=caps?1:0;
		buffers[17].Release(); Upload(17,data,sizeof(data)); UpdateDescriptors(); Run(9);
	}
	void Finalize() {
		if (engine.m_level->m_bufRlcParams.size) {
			buffers[18].Release(); buffers[18].Swap(engine.m_level->m_bufRlcParams); UpdateDescriptors();
			Run(27,uint32_t(buffers[18].size/sizeof(EngineVulkan::GpuRlcParam)));
			buffers[18].Swap(engine.m_level->m_bufRlcParams);
		}
		std::vector<uint32_t> used(primCount+4); Download(3,used.data(),used.size()*sizeof(uint32_t));
		for (size_t n=0;n<primitives.size();++n) if (used[n+1]) primitives[n]->SetPrimitiveUsed(true);
		buffers[8].Swap(engine.m_level->m_bufVv); buffers[9].Swap(engine.m_level->m_bufVi);
		buffers[10].Swap(engine.m_level->m_bufIi); buffers[11].Swap(engine.m_level->m_bufIv);
		engine.m_setupCoefficientsReady=true;
		buffers[19].Swap(engine.m_level->m_bufUpmlIndices); buffers[20].Swap(engine.m_level->m_bufUpmlVoltCoeffs);
		buffers[21].Swap(engine.m_level->m_bufUpmlCurrCoeffs);
		engine.m_level->m_numUpmlCells=upmlCount; engine.m_setupUpmlReady=true;
		Begin();
		VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(engine.m_level->m_cmdBuffer,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0,1,&barrier,0,nullptr,0,nullptr); Finish();
		// Retain only the small shader context for waveform metadata completion.
		staging.Release();
		for (auto& buffer:buffers) { buffer.Release(); Create(buffer,sizeof(double)); }
		UpdateDescriptors();
	}
	void CompleteExcitation(bool magnetic, unsigned int count, unsigned int offset, unsigned int length, unsigned int period) override {
		Buffer& points=magnetic?engine.m_level->m_bufCurrExcPoints:engine.m_level->m_bufVoltExcPoints;
		Require(points.size==uint64_t(count)*sizeof(EngineVulkan::GpuExcPoint),"prepared source point count changed");
		double metadata[3]={double(offset),double(length),double(period)};
		buffers[17].Release(); Upload(17,metadata,sizeof(metadata));
		buffers[18].Swap(points); UpdateDescriptors();
		try { Run(11,count); }
		catch (...) { buffers[18].Swap(points); throw; }
		buffers[18].Swap(points); UpdateDescriptors();
	}
	void CompleteMur(unsigned int offset, unsigned int count, unsigned int startTimestep) override {
		if (!count) return;
		SetAuxiliary({double(offset),double(startTimestep)});
		buffers[18].Swap(engine.m_level->m_bufMurParams); UpdateDescriptors();
		try { Run(31,count); }
		catch (...) { buffers[18].Swap(engine.m_level->m_bufMurParams); throw; }
		buffers[18].Swap(engine.m_level->m_bufMurParams); UpdateDescriptors();
	}
};
#endif

bool EngineVulkan::BuildOperator(Operator& op, int debugFlags, std::string& reason)
{
#ifdef ENABLE_VULKAN
	if (!op.GetGeometryCSX()) { reason="operator geometry is not initialized"; return false; }
	if (m_device) Reset();
	const double requestedTimestep=op.GetTimestep();
	const auto originalPrimitives=op.GetGeometryCSX()->GetAllPrimitives();
	std::vector<bool> originalUsed;
	for (CSPrimitives* primitive:originalPrimitives) originalUsed.push_back(primitive->GetPrimitiveUsed());
	try {
		if (!m_level->m_dimensionsValid || !InitVulkan()) throw std::runtime_error("Vulkan device initialization failed");
		{
			std::unique_ptr<VulkanOperatorSetup> setup(new VulkanOperatorSetup(*this,op));
			setup->Prepare();
			op.SetSetupBackend(setup.get());
			try { op.CalcECOperator(Operator::DebugFlags(debugFlags)); }
			catch (...) { op.SetSetupBackend(nullptr); throw; }
			op.SetSetupBackend(nullptr);
			setup->Finalize();
			m_operatorSetup=std::move(setup);
		}
		m_setupDeviceReady=true;
		reason.clear();
		return true;
	} catch (const std::exception& error) {
		for (size_t n=0;n<originalPrimitives.size();++n) originalPrimitives[n]->SetPrimitiveUsed(originalUsed[n]);
		reason=error.what(); op.SetTimestep(requestedTimestep); Reset(); return false;
	}
#else
	(void)op; (void)debugFlags;
	reason="built without ENABLE_VULKAN"; return false;
#endif
}
