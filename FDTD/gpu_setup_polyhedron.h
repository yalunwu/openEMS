/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_POLYHEDRON_H
#define GPU_SETUP_POLYHEDRON_H

#include "gpu_setup_descriptors.h"
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_POLY_QUERY __host__ __device__
#else
#define GPU_SETUP_POLY_QUERY
#endif

GPU_SETUP_POLY_QUERY inline double GPU_SetupDot(const double* a, const double* b)
{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

GPU_SETUP_POLY_QUERY inline void GPU_SetupCross(const double* a, const double* b, double* result)
{
	result[0]=a[1]*b[2]-a[2]*b[1]; result[1]=a[2]*b[0]-a[0]*b[2]; result[2]=a[0]*b[1]-a[1]*b[0];
}

// Count intersected triangles, as CSXCAD does, rather than coalescing hits into
// repaired topology. Surface points have an explicit inclusive convention.
GPU_SETUP_POLY_QUERY inline unsigned int GPU_SetupTriangleHit(const double* triangle, const double* p, const double* endpoint)
{
	double a[3],b[3],q[3],direction[3],normal[3],cross[3];
	for (int n=0;n<3;++n)
	{
		a[n]=triangle[3+n]-triangle[n]; b[n]=triangle[6+n]-triangle[n];
		q[n]=p[n]-triangle[n]; direction[n]=endpoint[n]-p[n];
	}
	GPU_SetupCross(a,b,normal);
	int major=0;
	if (fabs(normal[1])>fabs(normal[major])) major=1;
	if (fabs(normal[2])>fabs(normal[major])) major=2;
	if (normal[major]==0) return 0;
	if (GPU_SetupDot(normal,q)==0)
	{
		const int x=(major+1)%3,y=(major+2)%3;
		const double determinant=a[x]*b[y]-a[y]*b[x];
		const double u=q[x]*b[y]-q[y]*b[x],v=a[x]*q[y]-a[y]*q[x];
		if (determinant>0 ? (u>=0 && v>=0 && u+v<=determinant) : (u<=0 && v<=0 && u+v>=determinant)) return 2;
	}
	GPU_SetupCross(direction,b,cross);
	const double determinant=GPU_SetupDot(a,cross);
	if (determinant==0) return 0;
	const double u=GPU_SetupDot(q,cross);
	GPU_SetupCross(q,a,cross);
	const double v=GPU_SetupDot(direction,cross),t=GPU_SetupDot(b,cross);
	if (determinant>0) return u>=0 && v>=0 && u+v<=determinant && t>=0 && t<=determinant ? 1 : 0;
	return u<=0 && v<=0 && u+v>=determinant && t<=0 && t>=determinant ? 1 : 0;
}

GPU_SETUP_POLY_QUERY inline bool GPU_SetupSegmentBounds(const double* bounds, const double* p, const double* endpoint)
{
	double lo=0,hi=1;
	for (int n=0;n<3;++n)
	{
		if (bounds[2*n]>bounds[2*n+1]) return false;
		const double direction=endpoint[n]-p[n];
		if (direction==0) { if (p[n]<bounds[2*n] || p[n]>bounds[2*n+1]) return false; }
		else
		{
			const double a=(bounds[2*n]-p[n])/direction,b=(bounds[2*n+1]-p[n])/direction;
			lo=fmax(lo,fmin(a,b)); hi=fmin(hi,fmax(a,b));
			if (lo>hi) return false;
		}
	}
	return true;
}

GPU_SETUP_POLY_QUERY inline bool GPU_SetupPolyhedronContains(const GPU_SetupPrimitive& d, const double* data,
                                                             const GPU_SetupBVHNode* tree, const double* p,
                                                             uint32_t* triangle_tests=nullptr)
{
	if (d.dimension<3 || !d.count) return false;
	for (int n=0;n<3;++n) if (p[n]<d.bounds[2*n] || p[n]>d.bounds[2*n+1]) return false;
	unsigned int hits=0;
	if (!tree)
	{
		// The host query oracle can run without a device index. Production HIP
		// always supplies the tree built below; it never scans every triangle.
		for (uint32_t t=0;t<d.count;++t)
		{
			if (triangle_tests) ++*triangle_tests;
			const unsigned int hit=GPU_SetupTriangleHit(data+d.offset+9*t,p,d.stop);
			if (hit==2) return true;
			hits+=hit;
		}
		return hits%2!=0;
	}
	uint32_t node=0;
	for (;;)
	{
		const GPU_SetupBVHNode& current=tree[d.index_offset+node];
		if (GPU_SetupSegmentBounds(current.bounds,p,d.stop))
		{
			if (node<d.index_leaves-1) { node=2*node+1; continue; }
			if (current.triangle<d.count)
			{
				if (triangle_tests) ++*triangle_tests;
				const unsigned int hit=GPU_SetupTriangleHit(data+d.offset+9*current.triangle,p,d.stop);
				if (hit==2) return true;
				hits+=hit;
			}
		}
		while (node && node%2==0) node=(node-1)/2;
		if (!node) break;
		++node;
	}
	return hits%2!=0;
}

#undef GPU_SETUP_POLY_QUERY
#endif
