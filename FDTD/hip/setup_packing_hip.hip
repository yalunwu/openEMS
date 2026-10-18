/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_internal.h"
#include "hip_setup_result.h"
#include "FDTD/gpu_setup_memory.h"
#include "tools/setup_timer.h"
#include <limits>
#include <cstdlib>
#include <iostream>

HIP_SetupResult::HIP_SetupResult(const std::shared_ptr<HIP_Context>& ctx) :
	context(ctx), dims{0,0,0}, mode(0), sets(0), coefficients{NULL,NULL,NULL,NULL}, palette(NULL), index(NULL),
	waveform_source(NULL), waveform_length(0), waveform{NULL,NULL} {}

HIP_SetupResult::~HIP_SetupResult()
{
	context->setup_future_bytes-=pending_runtime_bytes;
	hipStreamSynchronize(context->stream);
	for (float* buffer:coefficients) if (buffer) hipFree(buffer);
	for (void* buffer:coefficient_snapshots) if (buffer) hipFree(buffer);
	for (float* buffer:materials) if (buffer) hipFree(buffer);
	if (palette) hipFree(palette);
	if (index) hipFree(index);
	if (requested_readback_bytes && std::getenv("OPENEMS_GPU_DIAGNOSTICS") && std::atoi(std::getenv("OPENEMS_GPU_DIAGNOSTICS")))
		std::cout << "GPU requested host operator: d2h_bytes=" << requested_readback_bytes << std::endl;
}

namespace
{
struct Coefficients
{
	const float *values[6];
	unsigned int cells,width;
	__device__ unsigned int Bits(unsigned int node, unsigned int k) const
	{ return __float_as_uint(values[k/3][(k%3)*cells+node]); }
	__device__ bool Equal(unsigned int a, unsigned int b) const
	{
		for (unsigned int k=0;k<width;++k) if (Bits(a,k)!=Bits(b,k)) return false;
		return true;
	}
};

unsigned int HashSlots(size_t cells)
{
	size_t capacity=1;
	while (capacity<cells+cells/2) capacity*=2;
	if (capacity>std::numeric_limits<unsigned int>::max()) throw GPU_SetupUnavailable("coefficient hash index range exceeded");
	return static_cast<unsigned int>(capacity);
}

size_t MaxSets(size_t cells,unsigned int width)
{
	return std::min(cells,std::max(size_t(65536),(2*width*cells-4*cells)/(4*width)));
}

__global__ void FindRepresentatives(Coefficients coeff, unsigned int* slots, unsigned int capacity, unsigned int* representatives)
{
	const unsigned int node=blockIdx.x*blockDim.x+threadIdx.x;
	if (node>=coeff.cells) return;
	uint64_t hash=1469598103934665603ULL;
	for (unsigned int k=0;k<coeff.width;++k) hash=(hash^coeff.Bits(node,k))*1099511628211ULL;
	unsigned int slot=static_cast<unsigned int>(hash)&(capacity-1);
	for (unsigned int probe=0;probe<capacity;++probe)
	{
		const unsigned int previous=atomicCAS(slots+slot,0u,node+1);
		if (!previous || coeff.Equal(node,previous-1))
		{ representatives[node]=previous?previous-1:node; return; }
		slot=(slot+1)&(capacity-1);
	}
}

__global__ void NumberSets(unsigned int cells, const unsigned int* representatives, unsigned int* ids, unsigned int* count)
{
	const unsigned int node=blockIdx.x*blockDim.x+threadIdx.x;
	if (node<cells && representatives[node]==node) ids[node]=atomicAdd(count,1u);
}

__global__ void PackSets(Coefficients coeff, const unsigned int* representatives, const unsigned int* ids,
                         float* table, void* index, unsigned int mode)
{
	const unsigned int node=blockIdx.x*blockDim.x+threadIdx.x;
	if (node>=coeff.cells) return;
	const unsigned int set=ids[representatives[node]];
	if (mode==1) static_cast<unsigned short*>(index)[node]=static_cast<unsigned short>(set);
	else static_cast<unsigned int*>(index)[node]=set;
	if (representatives[node]==node)
		for (unsigned int k=0;k<coeff.width;++k) table[coeff.width*set+k]=__uint_as_float(coeff.Bits(node,k));
}

__global__ void ReadPacked(const float* palette, const void* indices, unsigned int mode, unsigned int cells,
                           unsigned int kind, unsigned int offset, unsigned int count, float* output)
{
	const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i>=count) return;
	const unsigned int source=offset+i,node=source%cells,component=source/cells;
	const unsigned int set=mode==1 ? static_cast<const unsigned short*>(indices)[node] : static_cast<const unsigned int*>(indices)[node];
	output[i]=palette[12*set+3*kind+component];
}

class Workspace
{
	std::shared_ptr<HIP_Context> context;
	std::vector<void*> buffers;
	size_t bytes=0;
public:
	explicit Workspace(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	size_t Bytes() const {return bytes;}
	~Workspace() { hipStreamSynchronize(context->stream); for (void* buffer:buffers) hipFree(buffer); }
	template<class T> T* Allocate(size_t count)
	{
		T* buffer=NULL;
		const size_t size=GPU_SetupMultiply(count,sizeof(T));
		const hipError_t status=HIP_AllocateDevice(context->stream,&buffer,size);
		if (status==hipErrorOutOfMemory) { hipGetLastError(); throw GPU_SetupUnavailable("insufficient device memory for coefficient packing"); }
		HIP_Check(status,"coefficient packing allocation");
		try { buffers.push_back(buffer); } catch (...) { hipFree(buffer); throw; }
		bytes=GPU_SetupAdd(bytes,size);
		HIP_Check(hipMemsetAsync(buffer,0,size,context->stream),"coefficient packing clear");
		return buffer;
	}
};

void AllocateOutput(hipStream_t stream,void** ptr, size_t bytes)
{
	const hipError_t status=HIP_AllocateDevice(stream,ptr,bytes);
	if (status==hipErrorOutOfMemory) { hipGetLastError(); throw GPU_SetupUnavailable("insufficient device memory for coefficient palette"); }
	HIP_Check(status,"coefficient palette allocation");
}
}

size_t HIP_SetupPackingBytes(size_t cells,unsigned int width)
{
	const size_t words=GPU_SetupAdd(GPU_SetupAdd(3*cells,HashSlots(cells)),1);
	return GPU_SetupAdd(GPU_SetupMultiply(words,sizeof(unsigned int)),GPU_SetupMultiply(MaxSets(cells,width),width*sizeof(float)));
}

void HIP_SetupResult::ReadMaterials(unsigned int kind,size_t offset,size_t count,float* values) const
{
	const size_t size=3ull*dims[0]*dims[1]*dims[2];
	if (kind>=4 || !materials[kind] || offset>size || count>size-offset) throw std::out_of_range("requested material storage range");
	HIP_Check(hipMemcpyAsync(values,materials[kind]+offset,count*sizeof(float),hipMemcpyDeviceToHost,context->stream),"requested material storage readback");
	HIP_Check(hipStreamSynchronize(context->stream),"requested material storage completion");
	requested_readback_bytes=GPU_SetupAdd(requested_readback_bytes,count*sizeof(float));
}

void HIP_SetupResult::ReadCoefficients(unsigned int kind,size_t offset,size_t count,float* values) const
{
	const size_t cells=GPU_SetupMultiply(GPU_SetupMultiply(dims[0],dims[1]),dims[2]),size=3*cells;
	if (kind>=4 || offset>size || count>size-offset || (count && !values)) throw std::out_of_range("coefficient materialization range");
	if (!count) return;
	if (!mode)
	{
		HIP_Check(hipMemcpyAsync(values,coefficients[kind]+offset,count*sizeof(float),hipMemcpyDeviceToHost,context->stream),"requested dense coefficient readback");
		HIP_Check(hipStreamSynchronize(context->stream),"requested coefficient completion");
	}
	else
	{
		Workspace workspace(context);
		const size_t capacity=std::min(count,size_t(65536));
		float* temporary=workspace.Allocate<float>(capacity);
		for (size_t copied=0;copied<count;)
		{
			const unsigned int part=static_cast<unsigned int>(std::min(count-copied,capacity));
			ReadPacked<<<(part+255u)/256,256,0,context->stream>>>(palette,index,mode,static_cast<unsigned int>(cells),kind,static_cast<unsigned int>(offset+copied),part,temporary);
			HIP_Check(hipGetLastError(),"requested palette decoding");
			HIP_Check(hipMemcpyAsync(values+copied,temporary,part*sizeof(float),hipMemcpyDeviceToHost,context->stream),"requested packed coefficient readback");
			HIP_Check(hipStreamSynchronize(context->stream),"requested coefficient completion");copied+=part;
		}
	}
	requested_readback_bytes=GPU_SetupAdd(requested_readback_bytes,count*sizeof(float));
}

void HIP_SetupResult::WriteCoefficient(unsigned int kind,size_t offset,float value)
{
	const size_t cells=GPU_SetupMultiply(GPU_SetupMultiply(dims[0],dims[1]),dims[2]),count=3*cells;
	if (kind>=4 || offset>=count) throw std::out_of_range("coefficient write index");
	// An explicit setter may expand compression. Preserve the buffers already
	// adopted by an engine, just as CPU construction leaves its engine snapshot.
	if (mode || engine_adopted)
	{
		coefficient_snapshots.reserve(coefficient_snapshots.size()+6);
		HIP_SetupResult replacement(context);
		for (unsigned int k=0;k<4;++k)
		{
			AllocateOutput(context->stream,reinterpret_cast<void**>(&replacement.coefficients[k]),count*sizeof(float));
			if (mode)
			{
				ReadPacked<<<(count+255)/256,256,0,context->stream>>>(palette,index,mode,static_cast<unsigned int>(cells),k,0,static_cast<unsigned int>(count),replacement.coefficients[k]);
				HIP_Check(hipGetLastError(),"requested writable coefficient decoding");
			}
			else HIP_Check(hipMemcpyAsync(replacement.coefficients[k],coefficients[k],count*sizeof(float),hipMemcpyDeviceToDevice,context->stream),"requested writable coefficient snapshot");
		}
		HIP_Check(hipStreamSynchronize(context->stream),"writable coefficient completion");
		for (unsigned int k=0;k<4;++k)
		{
			if (coefficients[k]) coefficient_snapshots.push_back(coefficients[k]);
			coefficients[k]=replacement.coefficients[k];replacement.coefficients[k]=nullptr;
		}
		if (palette) coefficient_snapshots.push_back(palette);
		if (index) coefficient_snapshots.push_back(index);
		palette=nullptr;index=nullptr;mode=0;sets=0;engine_adopted=false;
	}
	HIP_Check(hipMemcpyAsync(coefficients[kind]+offset,&value,sizeof(value),hipMemcpyHostToDevice,context->stream),"requested operator coefficient write");
	HIP_Check(hipStreamSynchronize(context->stream),"requested operator coefficient write completion");
	if (std::getenv("OPENEMS_GPU_DIAGNOSTICS") && std::atoi(std::getenv("OPENEMS_GPU_DIAGNOSTICS")))
		std::cout << "GPU requested operator coefficient write: h2d_bytes=" << sizeof(value) << std::endl;
}

static size_t Pack(Coefficients coeff,const std::shared_ptr<HIP_Context>& context,unsigned int& mode,unsigned int& sets,float*& palette,void*& index)
{
	SetupTimer timer("GPU construction");
	const size_t cells=coeff.cells;
	if (!cells || cells>std::numeric_limits<unsigned int>::max()/3) throw std::logic_error("invalid construction result dimensions");
	Workspace workspace(context);
	const unsigned int capacity=HashSlots(cells),groups=static_cast<unsigned int>((cells+255)/256);
	unsigned int* slots=workspace.Allocate<unsigned int>(capacity);
	unsigned int* representatives=workspace.Allocate<unsigned int>(cells);
	unsigned int* ids=workspace.Allocate<unsigned int>(cells);
	unsigned int* count=workspace.Allocate<unsigned int>(1);
	const hipStream_t stream=context->stream;
	FindRepresentatives<<<groups,256,0,stream>>>(coeff,slots,capacity,representatives);
	HIP_Check(hipGetLastError(),"coefficient hash kernel");
	NumberSets<<<groups,256,0,stream>>>(coeff.cells,representatives,ids,count);
	HIP_Check(hipGetLastError(),"coefficient numbering kernel");
	HIP_Check(hipMemcpyAsync(&sets,count,sizeof(unsigned int),hipMemcpyDeviceToHost,stream),"coefficient palette count");
	HIP_Check(hipStreamSynchronize(stream),"coefficient palette count synchronization");
	if (sets>MaxSets(cells,coeff.width))
	{
		timer.Report("device palette rejected; dense coefficients retained");
		return workspace.Bytes();
	}
	mode=sets<=65536?1:2;
	AllocateOutput(stream,reinterpret_cast<void**>(&palette),GPU_SetupMultiply(sets,coeff.width*sizeof(float)));
	AllocateOutput(stream,&index,GPU_SetupMultiply(cells,mode==1?sizeof(unsigned short):sizeof(unsigned int)));
	PackSets<<<groups,256,0,stream>>>(coeff,representatives,ids,palette,index,mode);
	HIP_Check(hipGetLastError(),"coefficient packing kernel");
	HIP_Check(hipStreamSynchronize(stream),"coefficient packing synchronization");
	timer.Report("device coefficient palettes");
	return GPU_SetupAdd(workspace.Bytes(),GPU_SetupAdd(GPU_SetupMultiply(sets,coeff.width*sizeof(float)),GPU_SetupMultiply(cells,mode==1?sizeof(unsigned short):sizeof(unsigned int))));
}

size_t HIP_PackSetupCoefficients(HIP_SetupResult& result)
{
	const size_t cells=GPU_SetupMultiply(GPU_SetupMultiply(result.dims[0],result.dims[1]),result.dims[2]);
	Coefficients coeff={{result.coefficients[0],result.coefficients[1],result.coefficients[2],result.coefficients[3],nullptr,nullptr},static_cast<unsigned int>(cells),12};
	const size_t bytes=Pack(coeff,result.context,result.mode,result.sets,result.palette,result.index);
	if (result.mode) for (int k=0;k<4;++k) {HIP_Check(hipFree(result.coefficients[k]),"dense coefficient release");result.coefficients[k]=nullptr;}
	return bytes;
}

size_t HIP_PackSetupUPML(HIP_SetupUPML& result)
{
	const size_t cells=GPU_SetupMultiply(GPU_SetupMultiply(result.dims[0],result.dims[1]),result.dims[2]);
	Coefficients coeff={{},static_cast<unsigned int>(cells),18};
	for (int k=0;k<6;++k) coeff.values[k]=result.values+3*cells*k;
	return Pack(coeff,result.context,result.mode,result.sets,result.palette,result.index);
}

HIP_SetupUPML::~HIP_SetupUPML()
{
	hipStreamSynchronize(context->stream);
	if (values) hipFree(values);if (palette) hipFree(palette);if (index) hipFree(index);
}

HIP_SetupArray::~HIP_SetupArray()
{
	hipStreamSynchronize(context->stream);if (values) hipFree(values);
}

void HIP_SetupArray::Read(float* output,size_t offset,size_t length,bool diagnostics) const
{
	if (offset>count || length>count-offset) throw std::out_of_range("extension materialization range");
	const size_t bytes=length*sizeof(float);
	if (bytes) HIP_Check(hipMemcpyAsync(output,values+offset,bytes,hipMemcpyDeviceToHost,context->stream),"requested extension tables");
	HIP_Check(hipStreamSynchronize(context->stream),"requested extension completion");
	if (diagnostics) std::cout << "GPU requested host extension: d2h_bytes=" << bytes << std::endl;
}

HIP_SetupSparse::~HIP_SetupSparse()
{
	hipStreamSynchronize(context->stream);
	for (const auto& block:blocks) if (block.nodes) hipFree(block.nodes);
}

void HIP_SetupSparse::ReadNodes(unsigned int index,std::vector<unsigned int>& nodes,bool diagnostics) const
{
	const auto& block=blocks.at(index);nodes.resize(block.count);
	const size_t bytes=nodes.size()*sizeof(unsigned int);
	if (bytes) HIP_Check(hipMemcpyAsync(nodes.data(),block.nodes,bytes,hipMemcpyDeviceToHost,context->stream),"requested sparse positions");
	HIP_Check(hipStreamSynchronize(context->stream),"requested sparse completion");
	if (diagnostics) std::cout << "GPU requested host sparse positions: d2h_bytes=" << bytes << std::endl;
}

void HIP_SetupUPML::Materialize(float* outputs[6],bool diagnostics) const
{
	const size_t count=3ull*dims[0]*dims[1]*dims[2],bytes=count*sizeof(float);
	for (int k=0;k<6;++k) HIP_Check(hipMemcpyAsync(outputs[k],values+k*count,bytes,hipMemcpyDeviceToHost,context->stream),"requested UPML materialization");
	HIP_Check(hipStreamSynchronize(context->stream),"requested UPML completion");
	if (diagnostics) std::cout << "GPU requested host UPML: d2h_bytes=" << 6*bytes << std::endl;
}
