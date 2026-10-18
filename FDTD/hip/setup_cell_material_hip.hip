/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "setup_cell_material_hip.h"
#include "setup_processing_hip.h"
#include "FDTD/gpu_setup_geometry.h"
#include "FDTD/gpu_setup_lookup.h"
#include "FDTD/gpu_setup_memory.h"
#include "tools/setup_timer.h"

HIP_CellQuery::~HIP_CellQuery() {for (void* buffer:allocations) hipFree(buffer);}

namespace {
struct CellMaterials : Engine_CellMaterialTable
{
	std::shared_ptr<HIP_Context> context;
	double* values=nullptr;size_t count=0;bool anisotropic=false,diagnostics=false;
	explicit CellMaterials(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	~CellMaterials() {if (values) hipFree(values);}
	size_t Count() const override {return count;}
	bool AnyAnisotropic() const override {return anisotropic;}
	void Read(double* output,size_t length) const override
	{
		if (length!=count || !output) throw std::out_of_range("requested cell material table size");
		HIP_Check(hipMemcpyAsync(output,values,count*sizeof(double),hipMemcpyDeviceToHost,context->stream),"requested cell material readback");
		HIP_Check(hipStreamSynchronize(context->stream),"requested cell material completion");
		if (diagnostics) std::cout << "GPU requested cell material readback: bytes=" << count*sizeof(double) << std::endl;
	}
};
struct CellInputs
{
	HIP_CellQueryDevice query;
	const uint32_t* positions;
	double* stack;
	uint32_t count,lines[3],offsets[3];
};
__device__ void CellWeightCoordinates(const GPU_SetupProperty& property,const double* coord,double* variables)
{
	for (int n=0;n<3;++n) variables[n]=coord[n];
	if (property.coordinate_system==1)
	{variables[0]=coord[0]*cos(coord[1]);variables[1]=coord[0]*sin(coord[1]);}
	for (int n=0;n<3;++n) variables[n]-=property.weight_origin[n];
	variables[3]=sqrt(variables[0]*variables[0]+variables[1]*variables[1]);
	variables[4]=sqrt(variables[0]*variables[0]+variables[1]*variables[1]+variables[2]*variables[2]);
	variables[5]=atan2(variables[1],variables[0]);variables[6]=asin(1.0)-atan(variables[2]/variables[3]);
	if (property.coordinate_system==1)
	{
		variables[3]=coord[0];variables[4]=sqrt(coord[0]*coord[0]+coord[2]*coord[2]);
		variables[5]=coord[1];variables[6]=asin(1.0)-atan(coord[2]/coord[0]);
	}
}
__device__ double CellMaterialValue(const HIP_CellQueryDevice& query,uint32_t primitive,uint32_t channel,const double* coord,
                                  const GPU_SetupExpressionView& expressions,int& error)
{
	const auto& property=query.properties[primitive];
	if (property.discrete)
	{
		const auto& database=query.discrete[property.discrete];
		const int index=GPU_SetupDatabasePosition(database,query.material_data,query.material_indices,coord);
		if (index>=0 && database.channels[channel]!=UINT32_MAX) return query.material_data[database.channels[channel]+index];
	}
	const double base=channel==4 ? property.density : property.values[0][channel];
	const uint32_t program=channel==4 ? property.density_weight : property.weights[0][channel];
	if (!program) return base;
	double variables[7];CellWeightCoordinates(property,coord,variables);
	return base*GPU_SetupEvaluate(expressions.programs[program],expressions.instructions,expressions.constants,variables,expressions.stack,expressions.capacity,error);
}
__global__ void SampleCellMaterials(CellInputs input,double* values,uint32_t* flags)
{
	const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;if (id>=input.count) return;
	uint32_t q=id;int p[3];
	for (int n=2;n>=0;--n) {p[n]=int(input.positions[input.offsets[n]+q%input.lines[n]]);q/=input.lines[n];}
	const auto& query=input.query;const auto& metrics=query.metrics;
	double volume=metrics.cylindrical ? metrics.EdgeArea(2,p)*metrics.EdgeLength(2,p) :
		metrics.EdgeLength(0,p)*metrics.EdgeLength(1,p)*metrics.EdgeLength(2,p);
	double density=0,conductivity=0,coord[3];int error=0;
	GPU_SetupExpressionView expressions={query.programs,query.instructions,query.constants,input.stack+size_t(id)*query.stack_size,query.stack_size,&error};
	if (!metrics.CellCenter(p,coord)) error=1;
	else
	{
		const uint32_t row=uint32_t(p[0])*metrics.mesh.dims[1]+uint32_t(p[1]);
		for (uint32_t n=query.candidates[row];n<query.candidates[row+1];++n)
		{
			const uint32_t primitive=query.candidates[n];
			if (!GPU_SetupContains(query.shapes[primitive],query.geometry,coord,query.geometry_index,&expressions)) continue;
			if (!query.properties[primitive].isotropic) atomicOr(flags,2u);
			density=CellMaterialValue(query,primitive,4,coord,expressions,error);
			conductivity=CellMaterialValue(query,primitive,1,coord,expressions,error);break;
		}
	}
	if (error || !isfinite(volume) || !isfinite(density) || !isfinite(conductivity)) atomicOr(flags,1u);
	values[id]=volume;values[size_t(input.count)+id]=density;values[2ull*input.count+id]=conductivity;
}
}

std::shared_ptr<Engine_CellMaterialTable> HIP_BuildCellMaterials(GPU_Backend_HIP::Impl* backend,const unsigned int lines[3],unsigned int* const positions[3],const std::shared_ptr<Engine_SamplingMesh>& prepared_mesh)
{
	if (!backend->setup_result) return {};
	if (!backend->setup_result->cell_query) throw std::logic_error("cell material inputs were not retained during construction");
	SetupTimer timer("GPU processing construction");const auto& query=*backend->setup_result->cell_query;
	const auto mesh=std::dynamic_pointer_cast<HIP_SamplingMesh>(prepared_mesh);
	if (prepared_mesh && (!mesh || mesh->context!=backend->ctx)) throw std::logic_error("cell material mesh context mismatch");
	CellInputs input={};input.query=query.device;std::vector<uint32_t> raw;size_t count=1;
	for (int n=0;n<3;++n)
	{
		input.lines[n]=lines[n];input.offsets[n]=uint32_t(raw.size());count=GPU_SetupMultiply(count,lines[n]);
		if (mesh)
		{
			if (lines[n]!=mesh->lines[n]) throw std::logic_error("cell material mesh dimensions differ");
			input.offsets[n]=mesh->offsets[n];continue;
		}
		for (uint32_t p=0;p<lines[n];++p)
		{
			if (positions[n][p]>=query.device.metrics.mesh.dims[n]) throw std::out_of_range("cell material sampling position");
			raw.push_back(positions[n][p]);
		}
	}
	if (!count || count>(UINT32_MAX-255u)/3) throw GPU_SetupUnavailable("cell material sampling index range exceeded");
	input.count=uint32_t(count);
	const size_t output_bytes=GPU_SetupMultiply(3*count,sizeof(double));
	const size_t stack_bytes=GPU_SetupMultiply((std::max)(size_t(1),GPU_SetupMultiply(count,query.device.stack_size)),sizeof(double));
	const size_t scratch_bytes=GPU_SetupAdd(GPU_SetupAdd(stack_bytes,GPU_SetupMultiply(raw.size(),sizeof(uint32_t))),sizeof(uint32_t));
	const size_t needed=GPU_SetupAdd(output_bytes,scratch_bytes);
	size_t free=0,total=0;HIP_Check(hipMemGetInfo(&free,&total),"cell material memory query");
	const size_t cap=GPU_SetupMemoryLimit(),live=backend->ctx->setup_start_free>free ? backend->ctx->setup_start_free-free : 0;
	if (needed>free || live>cap || needed>cap-live) throw GPU_SetupUnavailable("insufficient device memory for cell materials");
	auto result=std::make_shared<CellMaterials>(backend->ctx);result->count=3*count;result->diagnostics=backend->diagnostics;
	uint32_t* device_positions=nullptr;uint32_t* flags=nullptr;double* stack=nullptr;
	struct Cleanup {uint32_t*& positions;uint32_t*& flags;double*& stack;~Cleanup() {if (positions) hipFree(positions);if (flags) hipFree(flags);if (stack) hipFree(stack);}} cleanup={device_positions,flags,stack};
	auto allocate=[backend](void** output,size_t bytes) {
		const auto status=HIP_AllocateDevice(backend->Stream(),output,(std::max)(bytes,size_t(1)));
		if (status==hipErrorOutOfMemory) {hipGetLastError();throw GPU_SetupUnavailable("insufficient device memory for cell materials");}
		HIP_Check(status,"cell material allocation");
	};
	allocate(reinterpret_cast<void**>(&result->values),output_bytes);
	if (!mesh) allocate(reinterpret_cast<void**>(&device_positions),raw.size()*sizeof(uint32_t));
	allocate(reinterpret_cast<void**>(&flags),sizeof(uint32_t));allocate(reinterpret_cast<void**>(&stack),stack_bytes);
	if (!mesh) HIP_Check(hipMemcpyAsync(device_positions,raw.data(),raw.size()*sizeof(uint32_t),hipMemcpyHostToDevice,backend->Stream()),"cell material positions upload");
	HIP_Check(hipMemsetAsync(flags,0,sizeof(uint32_t),backend->Stream()),"cell material flags initialization");
	input.positions=mesh ? mesh->positions : device_positions;input.stack=stack;
	SampleCellMaterials<<<(input.count+255)/256,256,0,backend->Stream()>>>(input,result->values,flags);HIP_Check(hipGetLastError(),"cell material sampling");
	uint32_t status=0;HIP_Check(hipMemcpyAsync(&status,flags,sizeof(status),hipMemcpyDeviceToHost,backend->Stream()),"cell material status readback");backend->Flush();
	if (status&1u) throw std::runtime_error("HIP cell material sampling: invalid numerical result");
	result->anisotropic=(status&2u)!=0;
	std::cout << "HIP cell material sampling active" << std::endl;
	if (backend->diagnostics) std::cout << "GPU cell material preparation: H2D=" << raw.size()*sizeof(uint32_t) << " D2H=" << sizeof(status) << " device_bytes=" << output_bytes << " peak=" << needed << std::endl;
	timer.Report("density, conductivity and cell volumes");return result;
}

std::shared_ptr<Engine_CellMaterialTable> GPU_Backend_HIP::BuildCellMaterials(const unsigned int lines[3],unsigned int* const positions[3],const std::shared_ptr<Engine_SamplingMesh>& mesh)
{
	return HIP_BuildCellMaterials(d,lines,positions,mesh);
}
