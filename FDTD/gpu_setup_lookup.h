/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_LOOKUP_H
#define GPU_SETUP_LOOKUP_H
#include "gpu_setup_descriptors.h"
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_LOOKUP __host__ __device__
#else
#define GPU_SETUP_LOOKUP
#endif

GPU_SETUP_LOOKUP inline int GPU_SetupDatabasePosition(const GPU_SetupDiscreteMaterial& d, const float* data,
                                                      const uint8_t* indices, const double* input)
{
	double cartesian[3]={input[0],input[1],input[2]},p[3];
	if (d.coordinate_system==1) {cartesian[0]=input[0]*cos(input[1]);cartesian[1]=input[0]*sin(input[1]);}
	for (int n=0;n<3;++n)
	{
		p[n]=d.inverse[4*n+3];
		for (int k=0;k<3;++k) p[n]+=d.inverse[4*n+k]*cartesian[k];
		p[n]/=d.scale;
	}
	uint32_t pos[3];
	for (int n=0;n<3;++n)
	{
		const float* mesh=data+d.mesh[n];
		if (p[n]<mesh[0] || p[n]>mesh[d.dims[n]-1]) return -1;
		// Preserve the CPU's exact-upper-limit convention: its search leaves
		// the voxel index at zero when the coordinate equals the final line.
		if (p[n]==mesh[d.dims[n]-1]) {pos[n]=0;continue;}
		uint32_t lo=1,hi=d.dims[n]-1;
		while (lo<hi) {const uint32_t middle=lo+(hi-lo)/2;if (p[n]<mesh[middle]) hi=middle;else lo=middle+1;}
		pos[n]=lo-1;
	}
	const uint32_t voxel=pos[0]+pos[1]*(d.dims[0]-1)+pos[2]*(d.dims[0]-1)*(d.dims[1]-1);
	const uint32_t index=indices[d.indices+voxel];
	if ((!d.database_background && !index) || index>=d.database_size) return -1;
	return static_cast<int>(index);
}

GPU_SETUP_LOOKUP inline double GPU_SetupInterpolateFile(const GPU_SetupFileWeight& d, const double* data,
                                                        int component, const double* p)
{
	if (component==int(d.propagation_axis)) return 0;
	const int axis=(d.propagation_axis+1)%3,other=(d.propagation_axis+2)%3;
	double c[2]={p[axis],p[other]};
	uint32_t pos[2];
	for (int n=0;n<2;++n)
	{
		const double* grid=data+d.axes[n];
		c[n]=fmax(grid[0],fmin(grid[d.dims[n]-1],c[n]));
		uint32_t lo=0,hi=d.dims[n]-1;
		while (lo<hi) {const uint32_t middle=lo+(hi-lo)/2;if (grid[middle]<c[n]) lo=middle+1;else hi=middle;}
		pos[n]=lo ? lo-1 : 0;
		if (pos[n]>d.dims[n]-2) pos[n]=d.dims[n]-2;
	}
	const double x1=data[d.axes[0]+pos[0]],x2=data[d.axes[0]+pos[0]+1];
	const double y1=data[d.axes[1]+pos[1]],y2=data[d.axes[1]+pos[1]+1];
	const double* field=data+d.fields[component==other ? 1 : 0];
	const uint32_t index=pos[0]*d.dims[1]+pos[1];
	const double factor=1/((x2-x1)*(y2-y1));
	return factor*(field[index]*(x2-c[0])*(y2-c[1])+field[index+d.dims[1]]*(c[0]-x1)*(y2-c[1])+
	               field[index+1]*(x2-c[0])*(c[1]-y1)+field[index+d.dims[1]+1]*(c[0]-x1)*(c[1]-y1));
}

#undef GPU_SETUP_LOOKUP
#endif
