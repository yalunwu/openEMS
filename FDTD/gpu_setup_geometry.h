/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_GEOMETRY_H
#define GPU_SETUP_GEOMETRY_H

#include "gpu_setup_descriptors.h"
#include "gpu_setup_polyhedron.h"
#include "gpu_setup_expression.h"
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_QUERY __host__ __device__
#else
#define GPU_SETUP_QUERY
#endif

GPU_SETUP_QUERY inline bool GPU_SetupInBounds(const double* bounds, const double* p)
{
	for (int n=0;n<3;++n) if (p[n]<bounds[2*n] || p[n]>bounds[2*n+1]) return false;
	return true;
}

GPU_SETUP_QUERY inline bool GPU_SetupPolygon(const GPU_SetupPrimitive& d, const double* data, const double* p)
{
	if (!d.count || !GPU_SetupInBounds(d.bounds,p)) return false;
	const double x=p[(d.normal+1)%3], y=p[(d.normal+2)%3];
	double x1=data[d.offset+2*(d.count-1)], y1=data[d.offset+2*(d.count-1)+1];
	bool startover=y1>=y;
	int winding=0;
	for (uint32_t i=0;i<d.count;++i)
	{
		const double x2=data[d.offset+2*i], y2=data[d.offset+2*i+1];
		if (x2==x1 && x1==x && ((y<y1 && y>y2) || (y>y1 && y<y2))) return true;
		if (y2==y1 && y1==y && ((x<x1 && x>x2) || (x>x1 && x<x2))) return true;
		const bool endover=y2>=y;
		if (startover!=endover)
		{
			if ((y2-y)*(x2-x1)<=(y2-y1)*(x2-x)) { if (endover) ++winding; }
			else if (!endover) --winding;
		}
		startover=endover; x1=x2; y1=y2;
	}
	return winding!=0;
}

GPU_SETUP_QUERY inline double GPU_SetupDistance(const double* p, const double* a)
{
	return sqrt((p[0]-a[0])*(p[0]-a[0])+(p[1]-a[1])*(p[1]-a[1])+(p[2]-a[2])*(p[2]-a[2]));
}

GPU_SETUP_QUERY inline double GPU_SetupLineDistance(const double* p, const double* a, const double* b, double& foot)
{
	double dir[3], length=0, projection=0, point[3];
	for (int n=0;n<3;++n) { dir[n]=b[n]-a[n]; length+=dir[n]*dir[n]; projection+=(p[n]-a[n])*dir[n]; }
	foot=projection/length;
	for (int n=0;n<3;++n) point[n]=a[n]+foot*dir[n];
	return GPU_SetupDistance(p,point);
}

GPU_SETUP_QUERY inline void GPU_SetupConvertCoordinates(double p[3], uint32_t from, uint32_t to)
{
	if (from==to) return;
	const double x=p[0],y=p[1];
	if (from==1) {p[0]=x*cos(y);p[1]=x*sin(y);}
	else {p[0]=sqrt(x*x+y*y);p[1]=atan2(y,x);}
}

// The same FP64 query code is exercised by the CPU oracle tests and HIP.
// Point/Curve containment is deliberately empty; PEC curves need a path pass.
GPU_SETUP_QUERY inline bool GPU_SetupContains(const GPU_SetupPrimitive& d, const double* data, const double* coord,
                                              const GPU_SetupBVHNode* tree=nullptr, const GPU_SetupExpressionView* expressions=nullptr,
                                              uint32_t* triangle_tests=nullptr)
{
	const double pi=3.1415926535897932384626433832795;
	double input[3]={coord[0],coord[1],coord[2]},p[3];
	const bool meshQuery=d.shape==GPU_SetupShape::Box || d.shape==GPU_SetupShape::MultiBox;
	if (!meshQuery || d.has_transform) GPU_SetupConvertCoordinates(input,d.input_coordinate_system,0);
	for (int n=0;n<3;++n)
	{
		p[n]=0;
		for (int k=0;k<3;++k) p[n]+=d.inverse[4*n+k]*input[k];
		p[n]+=d.inverse[4*n+3];
	}
	if (meshQuery)
	{
		if (d.has_transform) GPU_SetupConvertCoordinates(p,0,d.input_coordinate_system);
		if (d.shape==GPU_SetupShape::Box) GPU_SetupConvertCoordinates(p,d.input_coordinate_system,d.coordinate_system);
	}
	else if (d.shape==GPU_SetupShape::Polygon && (d.geometry_flags&1u) && d.has_transform && d.input_coordinate_system==1)
	{
		// LinPoly transforms in mesh coordinates before its Polygon base query.
		GPU_SetupConvertCoordinates(p,0,1);GPU_SetupConvertCoordinates(p,1,0);
	}
	switch (d.shape)
	{
	case GPU_SetupShape::UserDefined:
	{
		if (!expressions) return false;
		double variables[6]={p[0]-d.start[0],p[1]-d.start[1],p[2]-d.start[2],0,0,0};
		if (d.axis)
		{
			const double rho=sqrt(variables[0]*variables[0]+variables[1]*variables[1]);
			variables[3]=d.axis==1 ? rho : sqrt(variables[0]*variables[0]+variables[1]*variables[1]+variables[2]*variables[2]);
			variables[4]=atan2(variables[1],variables[0]);
			if (d.axis==2) variables[5]=asin(1.0)-atan(variables[2]/rho);
		}
		return GPU_SetupEvaluate(expressions->programs[d.normal],expressions->instructions,expressions->constants,variables,
		                         expressions->stack,expressions->capacity,*expressions->error)==1;
	}
	case GPU_SetupShape::Polyhedron:
		return GPU_SetupPolyhedronContains(d,data,tree,p,triangle_tests);
	case GPU_SetupShape::Box:
		if (d.coordinate_system==1)
		{
			const double lo=fmin(d.start[1],d.stop[1]),hi=fmax(d.start[1],d.stop[1]);
			if (p[1]<lo) { while (p[1]<lo) p[1]+=2*pi; }
			else if (p[1]>hi) { while (p[1]>hi) p[1]-=2*pi; }
		}
		for (int n=0;n<3;++n) if (p[n]<fmin(d.start[n],d.stop[n]) || p[n]>fmax(d.start[n],d.stop[n])) return false;
		return true;
	case GPU_SetupShape::Sphere:
		return GPU_SetupDistance(p,d.start)<d.radius;
	case GPU_SetupShape::SphericalShell:
		return fabs(GPU_SetupDistance(p,d.start)-d.radius)<d.shell_width/2;
	case GPU_SetupShape::Cylinder:
	case GPU_SetupShape::CylindricalShell:
	{
		if (!GPU_SetupInBounds(d.bounds,p)) return false;
		double foot, distance=GPU_SetupLineDistance(p,d.start,d.stop,foot);
		if (foot<0 || foot>1) return false;
		return d.shape==GPU_SetupShape::Cylinder ? distance<=d.radius : fabs(distance-d.radius)<=d.shell_width/2;
	}
	case GPU_SetupShape::Polygon:
		return GPU_SetupPolygon(d,data,p);
	case GPU_SetupShape::MultiBox:
		for (uint32_t b=0;b<d.count;++b)
		{
			bool inside=true;
			for (int n=0;n<3;++n)
			{
				const double a=data[d.offset+6*b+2*n], z=data[d.offset+6*b+2*n+1];
				if (p[n]<fmin(a,z) || p[n]>fmax(a,z)) { inside=false; break; }
			}
			if (inside) return true;
		}
		return false;
	case GPU_SetupShape::Wire:
		if (!GPU_SetupInBounds(d.bounds,p)) return false;
		for (uint32_t v=0;v<d.count;++v)
		{
			const double* a=data+d.offset+3*v;
			const double distance=GPU_SetupDistance(p,a);
			if (distance<d.radius) return true;
			if (v+1<d.count && distance<GPU_SetupDistance(a,a+3)+d.radius)
			{
				double foot, perpendicular=GPU_SetupLineDistance(p,a,a+3,foot);
				if (foot>0 && foot<1 && perpendicular<d.radius) return true;
			}
		}
		return false;
	case GPU_SetupShape::RotPoly:
	{
		const uint32_t a=(d.axis+1)%3,b=(d.axis+2)%3;
		double radius=sqrt(p[a]*p[a]+p[b]*p[b]);
		double alpha=atan2(p[b],p[a]);
		if (a==d.normal) alpha-=pi/2;
		if (alpha<0) alpha+=2*pi;
		double q[3]={radius,radius,radius}; q[d.normal]=0; q[d.axis]=p[d.axis];
		if (alpha<d.angles[0]) alpha+=2*pi;
		// RotPoly passes this intermediate point to Polygon::IsInside, which
		// performs its own mesh-coordinate conversion on the CPU too.
		double sample[3]={q[0],q[1],q[2]};GPU_SetupConvertCoordinates(sample,d.input_coordinate_system,0);
		if (GPU_SetupPolygon(d,data,sample) && alpha<d.angles[1]) return true;
		alpha+=pi;
		if (alpha>2*pi) alpha-=2*pi;
		if (alpha<d.angles[0]) alpha+=2*pi;
		if (alpha>d.angles[1]) return false;
		q[0]=q[1]=q[2]=-radius; q[d.normal]=0; q[d.axis]=p[d.axis];
		GPU_SetupConvertCoordinates(q,d.input_coordinate_system,0);
		return GPU_SetupPolygon(d,data,q);
	}
	default:
		return false;
	}
}

#undef GPU_SETUP_QUERY
#endif
