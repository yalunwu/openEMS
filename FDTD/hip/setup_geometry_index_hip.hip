/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_internal.h"
#include "setup_geometry_index_hip.h"
#include "FDTD/gpu_setup_bounds.h"
#include <cfloat>

namespace
{
__global__ void PrimitiveBounds(GPU_SetupPrimitive* shapes,uint32_t count,const double* geometry)
{
	const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if (i<count) GPU_SetupBuildBounds(shapes[i],geometry);
}
__device__ uint32_t Interleave(uint32_t value)
{
	value=(value|(value<<16))&0x030000ff;
	value=(value|(value<<8))&0x0300f00f;
	value=(value|(value<<4))&0x030c30c3;
	return (value|(value<<2))&0x09249249;
}

__global__ void TriangleKeys(const GPU_SetupPrimitive* shape, const double* geometry, uint64_t* keys)
{
	const GPU_SetupPrimitive& d=*shape;
	const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i>=d.index_leaves) return;
	if (i>=d.count) { keys[i]=UINT64_MAX; return; }
	uint32_t q[3];
	for (int n=0;n<3;++n)
	{
		const double* t=geometry+d.offset+9*i;
		const double center=(t[n]+t[n+3]+t[n+6])/3;
		const double extent=d.bounds[2*n+1]-d.bounds[2*n];
		q[n]=extent>0 ? static_cast<uint32_t>(fmax(0.0,fmin(1023.0,1023*(center-d.bounds[2*n])/extent))) : 0;
	}
	const uint32_t morton=Interleave(q[0])|(Interleave(q[1])<<1)|(Interleave(q[2])<<2);
	keys[i]=(uint64_t(morton)<<32)|i;
}

__global__ void SortKeys(uint64_t* keys, uint32_t count, uint32_t width, uint32_t stride)
{
	const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i>=count) return;
	const uint32_t j=i^stride;
	if (j>i)
	{
		const uint64_t a=keys[i],b=keys[j];
		if ((i&width) ? a<b : a>b) {keys[i]=b;keys[j]=a;}
	}
}

__global__ void TriangleLeaves(const GPU_SetupPrimitive* shape, const double* geometry, const uint64_t* keys, GPU_SetupBVHNode* tree)
{
	const GPU_SetupPrimitive& d=*shape;
	const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i>=d.index_leaves) return;
	GPU_SetupBVHNode& leaf=tree[d.index_offset+d.index_leaves-1+i];
	leaf.triangle=static_cast<uint32_t>(keys[i]);
	for (int n=0;n<3;++n)
	{
		leaf.bounds[2*n]=DBL_MAX; leaf.bounds[2*n+1]=-DBL_MAX;
		if (leaf.triangle>=d.count) continue;
		const double* t=geometry+d.offset+9*leaf.triangle;
		leaf.bounds[2*n]=fmin(t[n],fmin(t[n+3],t[n+6]));
		leaf.bounds[2*n+1]=fmax(t[n],fmax(t[n+3],t[n+6]));
	}
}

__global__ void TriangleParents(const GPU_SetupPrimitive* shape, GPU_SetupBVHNode* tree, uint32_t width)
{
	const GPU_SetupPrimitive& d=*shape;
	const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i>=width) return;
	const uint32_t node=width-1+i;
	GPU_SetupBVHNode& parent=tree[d.index_offset+node];
	const GPU_SetupBVHNode& a=tree[d.index_offset+2*node+1];
	const GPU_SetupBVHNode& b=tree[d.index_offset+2*node+2];
	parent.triangle=UINT32_MAX;
	for (int n=0;n<3;++n)
	{
		parent.bounds[2*n]=fmin(a.bounds[2*n],b.bounds[2*n]);
		parent.bounds[2*n+1]=fmax(a.bounds[2*n+1],b.bounds[2*n+1]);
	}
}
}

void HIP_BuildSetupBounds(GPU_SetupPrimitive* shapes,uint32_t count,const double* geometry,hipStream_t stream)
{
	if (!count) return;
	if (count>UINT32_MAX-255u) throw GPU_SetupUnavailable("geometry bounds dispatch range exceeded");
	PrimitiveBounds<<<(count+255u)/256,256,0,stream>>>(shapes,count,geometry);
	HIP_Check(hipGetLastError(),"primitive bounds construction");
}

void HIP_BuildSetupGeometryIndex(const std::vector<GPU_SetupPrimitive>& shapes, const GPU_SetupPrimitive* device_shapes,const double* geometry,
                                 GPU_SetupBVHNode* tree, uint64_t* keys, hipStream_t stream)
{
	for (size_t i=0;i<shapes.size();++i)
	{
		const GPU_SetupPrimitive& d=shapes[i];
		if (d.shape!=GPU_SetupShape::Polyhedron || !d.index_leaves) continue;
		const uint32_t blocks=(d.index_leaves+255u)/256;
		TriangleKeys<<<blocks,256,0,stream>>>(device_shapes+i,geometry,keys);
		for (uint32_t width=2;width<=d.index_leaves;width*=2)
			for (uint32_t stride=width/2;stride;stride/=2) SortKeys<<<blocks,256,0,stream>>>(keys,d.index_leaves,width,stride);
		TriangleLeaves<<<blocks,256,0,stream>>>(device_shapes+i,geometry,keys,tree);
		for (uint32_t width=d.index_leaves/2;width;width/=2)
			TriangleParents<<<(width+255u)/256,256,0,stream>>>(device_shapes+i,tree,width);
		HIP_Check(hipGetLastError(),"polyhedron spatial index construction");
	}
}
