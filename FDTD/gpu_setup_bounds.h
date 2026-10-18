/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_BOUNDS_H
#define GPU_SETUP_BOUNDS_H
#include "gpu_setup_descriptors.h"
#include <cmath>
#include <cfloat>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_BOUNDS __host__ __device__
#else
#define GPU_SETUP_BOUNDS
#endif

// Local query bounds use CSXCAD's arithmetic and coordinate conventions.
// Transform/coordinate-mismatched candidate boxes remain conservative.
GPU_SETUP_BOUNDS inline void GPU_SetupBuildBounds(GPU_SetupPrimitive& d,const double* data)
{
	for (int n=0;n<6;++n) d.bounds[n]=0;
	switch (d.shape)
	{
	case GPU_SetupShape::Box:
		for (int n=0;n<3;++n) {d.bounds[2*n]=fmin(d.start[n],d.stop[n]);d.bounds[2*n+1]=fmax(d.start[n],d.stop[n]);}
		break;
	case GPU_SetupShape::Point:
		for (int n=0;n<3;++n) d.bounds[2*n]=d.bounds[2*n+1]=d.start[n];
		break;
	case GPU_SetupShape::Sphere:case GPU_SetupShape::SphericalShell:
	{
		const double radius=d.radius+(d.shape==GPU_SetupShape::SphericalShell ? d.shell_width/2 : 0);
		for (int n=0;n<3;++n) {d.bounds[2*n]=d.start[n]-radius;d.bounds[2*n+1]=d.start[n]+radius;}
		break;
	}
	case GPU_SetupShape::Cylinder:case GPU_SetupShape::CylindricalShell:
	{
		const double radius=d.radius+(d.shape==GPU_SetupShape::CylindricalShell ? d.shell_width/2 : 0);
		double v[3];int nonzero=-1,zeros=0;
		for (int n=0;n<3;++n) {v[n]=d.stop[n]-d.start[n];if (v[n]==0) ++zeros;else nonzero=n;}
		if (zeros==2)
		{
			for (int n=0;n<3;++n)
			{
				d.bounds[2*n]=n==nonzero ? fmin(d.start[n],d.stop[n]) : d.start[n]-radius;
				d.bounds[2*n+1]=n==nonzero ? fmax(d.start[n],d.stop[n]) : d.stop[n]+radius;
			}
			break;
		}
		const double length=sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
		if (!length) break;
		for (int n=0;n<3;++n) v[n]/=length;
		double a[3];for (int n=0;n<3;++n) a[n]=n==nonzero ? (v[(n+1)%3]+v[(n+2)%3])/v[n] : -1;
		const double norm=sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);for (int n=0;n<3;++n) a[n]/=norm;
		const double b[3]={a[1]*v[2]-a[2]*v[1],-a[0]*v[2]+a[2]*v[0],a[0]*v[1]-a[1]*v[0]};
		for (int n=0;n<3;++n)
		{
			d.bounds[2*n]=d.bounds[2*n+1]=d.start[n];const double angle=atan(b[n]/a[n]);
			for (int end=0;end<2;++end)
			{
				const double origin=end ? d.stop[n] : d.start[n];
				const double x=origin+radius*a[n]*cos(angle)+radius*b[n]*sin(angle);
				const double y=origin+radius*a[n]*cos(angle+acos(-1.0))+radius*b[n]*sin(angle+acos(-1.0));
				d.bounds[2*n]=fmin(d.bounds[2*n],fmin(x,y));d.bounds[2*n+1]=fmax(d.bounds[2*n+1],fmax(x,y));
			}
		}
		break;
	}
	case GPU_SetupShape::Polygon:case GPU_SetupShape::RotPoly:
		if (d.count)
		{
			const int a=(d.normal+1)%3,b=(d.normal+2)%3;
			d.bounds[2*d.normal]=d.bounds[2*d.normal+1]=d.start[0];
			d.bounds[2*a]=d.bounds[2*a+1]=data[d.offset];d.bounds[2*b]=d.bounds[2*b+1]=data[d.offset+1];
			for (uint32_t i=1;i<d.count;++i) for (int k=0;k<2;++k)
			{
				const int n=k ? b : a;const double value=data[d.offset+2*i+k];
				d.bounds[2*n]=fmin(d.bounds[2*n],value);d.bounds[2*n+1]=fmax(d.bounds[2*n+1],value);
			}
		}
		if (d.geometry_flags&1u) {d.bounds[2*d.normal]=d.start[0]+fmin(0.0,d.stop[0]);d.bounds[2*d.normal+1]=d.start[0]+fmax(0.0,d.stop[0]);}
		break;
	case GPU_SetupShape::MultiBox:
		for (uint32_t i=0;i<d.count;++i) for (int n=0;n<3;++n)
		{
			const double a=data[d.offset+6*i+2*n],b=data[d.offset+6*i+2*n+1];
			d.bounds[2*n]=i ? fmin(d.bounds[2*n],fmin(a,b)) : fmin(a,b);
			d.bounds[2*n+1]=i ? fmax(d.bounds[2*n+1],fmax(a,b)) : fmax(a,b);
		}
		break;
	case GPU_SetupShape::Curve:case GPU_SetupShape::Wire:
		for (uint32_t i=0;i<d.count;++i) for (int n=0;n<3;++n)
		{
			const double value=data[d.offset+3*i+n],raw=data[d.bounds_offset+3*i+n];
			if (!i) d.bounds[2*n]=d.bounds[2*n+1]=value;
			if (raw<d.bounds[2*n]) d.bounds[2*n]=value;else if (raw>d.bounds[2*n+1]) d.bounds[2*n+1]=value;
		}
		if (d.shape==GPU_SetupShape::Wire) for (int n=0;n<3;++n) {d.bounds[2*n]-=d.radius;d.bounds[2*n+1]+=d.radius;}
		break;
	case GPU_SetupShape::UserDefined:
		for (int n=0;n<3;++n) {d.bounds[2*n]=-DBL_MAX;d.bounds[2*n+1]=DBL_MAX;}
		break;
	case GPU_SetupShape::Polyhedron:
		for (uint32_t i=0;i<d.bounds_count;++i) for (int n=0;n<3;++n)
		{
			const double value=data[d.bounds_offset+3*i+n];
			d.bounds[2*n]=i ? fmin(d.bounds[2*n],value) : value;d.bounds[2*n+1]=i ? fmax(d.bounds[2*n+1],value) : value;
		}
		break;
	default:break;
	}
}
#undef GPU_SETUP_BOUNDS
#endif
