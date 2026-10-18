/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "FDTD/hip/hip_internal.h"
#include "FDTD/gpu_setup_geometry.h"
#include "FDTD/gpu_setup_candidates.h"
#include "FDTD/hip/setup_geometry_index_hip.h"
#include "openems_global.h"
#include <iostream>

namespace
{
__global__ void GeometrySamples(const GPU_SetupPrimitive* descriptor, const double* data, const GPU_SetupBVHNode* tree, const double* coordinates,
                                 unsigned int count, unsigned char* result, GPU_SetupExpressionView expressions, int* errors,
                                 uint32_t* maximum_triangle_tests)
{
	const unsigned int sample=blockIdx.x*blockDim.x+threadIdx.x;
	if (sample<count)
	{
		int error=0;
		uint32_t triangle_tests=0;
		expressions.stack+=size_t(sample)*expressions.capacity; expressions.error=&error;
		result[sample]=GPU_SetupContains(*descriptor,data,coordinates+3*sample,tree,&expressions,&triangle_tests);
		atomicMax(maximum_triangle_tests,triangle_tests);
		errors[sample]=error;
	}
}

struct TestBuffers
{
	std::vector<void*> buffers;
	~TestBuffers() {hipDeviceSynchronize();for (void* buffer:buffers) hipFree(buffer);}
	template<class T> T* Allocate(size_t count, const T* data=NULL)
	{
		T* buffer=NULL;
		HIP_Check(hipMalloc(&buffer,std::max(size_t(1),count)*sizeof(T)),"geometry test allocation");
		try {buffers.push_back(buffer);} catch (...) {hipFree(buffer);throw;}
		if (data && count) HIP_Check(hipMemcpy(buffer,data,count*sizeof(T),hipMemcpyHostToDevice),"geometry test upload");
		return buffer;
	}
};
}

void GPU_TestGeometrySamples(const GPU_SetupPrimitive& descriptor, const std::vector<double>& data,
                             const GPU_SetupExpressionBank& bank, const std::vector<double>& coordinates, std::vector<unsigned char>& results)
{
	TestBuffers buffers;
	double* geometry=buffers.Allocate(data.size(),data.data());
	double* samples=buffers.Allocate(coordinates.size(),coordinates.data());
	unsigned char* output=buffers.Allocate<unsigned char>(results.size());
	GPU_SetupBVHNode* tree=buffers.Allocate<GPU_SetupBVHNode>(descriptor.index_leaves ? 2ull*descriptor.index_leaves-1 : 1);
	uint64_t* keys=buffers.Allocate<uint64_t>(descriptor.index_leaves);
	GPU_SetupExpressionView expressions={buffers.Allocate(bank.programs.size(),bank.programs.data()),
	                                    buffers.Allocate(bank.instructions.size(),bank.instructions.data()),
	                                    buffers.Allocate(bank.constants.size(),bank.constants.data()),
	                                    buffers.Allocate<double>(results.size()*bank.stack_size),bank.stack_size,nullptr};
	int* errors=buffers.Allocate<int>(results.size());
	const uint32_t zero=0;
	uint32_t* maximum=buffers.Allocate<uint32_t>(1,&zero);
	auto* device_descriptor=buffers.Allocate(size_t(1),&descriptor);
	HIP_BuildSetupBounds(device_descriptor,1,geometry,0);
	GPU_SetupPrimitive computed={};HIP_Check(hipMemcpy(&computed,device_descriptor,sizeof(computed),hipMemcpyDeviceToHost),"bounds oracle readback");
	if (descriptor.shape!=GPU_SetupShape::Box || descriptor.coordinate_system==descriptor.input_coordinate_system)
		for (int n=0;n<6;++n)
			if (!std::isfinite(computed.bounds[n]) || std::abs(computed.bounds[n]-descriptor.bounds[n])>2e-13*std::max(1.0,std::abs(descriptor.bounds[n])))
				throw std::runtime_error("device geometry bounds differ from CSXCAD: shape="+std::to_string(static_cast<uint32_t>(descriptor.shape))+" axis="+std::to_string(n)+" expected="+std::to_string(descriptor.bounds[n])+" actual="+std::to_string(computed.bounds[n]));
	HIP_BuildSetupGeometryIndex(std::vector<GPU_SetupPrimitive>(1,descriptor),device_descriptor,geometry,tree,keys,0);
	GeometrySamples<<<static_cast<unsigned int>((results.size()+255)/256),256>>>(device_descriptor,geometry,tree,samples,static_cast<unsigned int>(results.size()),output,expressions,errors,maximum);
	HIP_Check(hipGetLastError(),"HIP geometry query kernel");
	HIP_Check(hipMemcpy(results.data(),output,results.size(),hipMemcpyDeviceToHost),"geometry test readback");
	std::vector<int> status(results.size());
	HIP_Check(hipMemcpy(status.data(),errors,status.size()*sizeof(int),hipMemcpyDeviceToHost),"geometry expression status readback");
	for (int error:status) if (error) throw std::runtime_error("device geometry expression failed");
	if (descriptor.shape==GPU_SetupShape::Polyhedron && descriptor.count>=10000)
	{
		uint32_t tested=0;
		HIP_Check(hipMemcpy(&tested,maximum,sizeof(tested),hipMemcpyDeviceToHost),"triangle query statistics");
		if (2ull*descriptor.index_leaves-1>=4ull*descriptor.count || tested>=descriptor.count/16)
			throw std::runtime_error("polyhedron index exceeds its memory or query-work bound");
		std::cout << "HIP large solid: triangles=" << descriptor.count << " nodes=" << 2ull*descriptor.index_leaves-1
		          << " maximum triangles per query=" << tested << std::endl;
	}
	std::cout << "HIP geometry query active" << std::endl;
}

__global__ static void PropertyIDs(const GPU_SetupPrimitive* shapes,uint32_t shape_count,const double* geometry,const double* coordinates,uint32_t count,uint32_t* output)
{
	const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if (i>=count) return;
	const double* point=coordinates+3*i;double bounds[6];for (int n=0;n<3;++n) bounds[2*n]=bounds[2*n+1]=point[n];
	output[i]=0;
	for (uint32_t p=0;p<shape_count;++p)
		if (GPU_SetupCandidate(shapes[p],bounds) && GPU_SetupContains(shapes[p],geometry,point)) {output[i]=shapes[p].property_id+1;break;}
}
void GPU_TestPropertyIDs(const std::vector<GPU_SetupPrimitive>& shapes,const std::vector<double>& data,const std::vector<double>& coordinates,std::vector<uint32_t>& output)
{
	TestBuffers buffers;auto* device_shapes=buffers.Allocate(shapes.size(),shapes.data());
	const auto* geometry=buffers.Allocate(data.size(),data.data());const auto* points=buffers.Allocate(coordinates.size(),coordinates.data());
	auto* values=buffers.Allocate<uint32_t>(output.size());
	HIP_BuildSetupBounds(device_shapes,uint32_t(shapes.size()),geometry,0);
	PropertyIDs<<<(output.size()+255)/256,256>>>(device_shapes,uint32_t(shapes.size()),geometry,points,uint32_t(output.size()),values);
	HIP_Check(hipGetLastError(),"material property ID snapshot");
	HIP_Check(hipMemcpy(output.data(),values,output.size()*sizeof(uint32_t),hipMemcpyDeviceToHost),"material property ID snapshot readback");
}

#ifdef OPENEMS_STANDALONE_SETUP_TEST
extern "C" int openEMS_TestGPUSetupGeometryDevice();
#else
extern "C" OPENEMS_EXPORT int openEMS_TestGPUSetupGeometryDevice();
#endif
#define OPENEMS_DEVICE_GEOMETRY_TEST
#define main openEMS_TestGPUSetupGeometryDevice
#include "test_gpu_setup_geometry.cpp"
#undef main
