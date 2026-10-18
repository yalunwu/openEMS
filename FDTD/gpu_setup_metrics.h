/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_METRICS_H
#define GPU_SETUP_METRICS_H
#include "gpu_setup_path.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_METRIC __device__
#else
#define GPU_SETUP_METRIC
#endif

struct GPU_SetupMetrics
{
	GPU_SetupMeshView mesh;
	double unit;
	uint32_t cylindrical,closed_alpha,axis,first_radius;
	const double* child_alpha;
	uint32_t child_alpha_count;
	GPU_SETUP_METRIC double Line(int n,int p,bool dual=false) const
	{
		return p>=0 && uint32_t(p)<mesh.dims[n] ? mesh.Line(n,uint32_t(p),dual ? 1 : 0) : 0;
	}
	GPU_SETUP_METRIC double RawEdge(int n,int p,bool dual) const
	{
		return p>=0 && uint32_t(p)<mesh.dims[n] ? mesh.Line(n,uint32_t(p),dual ? 5 : 4) : 0;
	}
	GPU_SETUP_METRIC bool Map(const int* input,int* p) const
	{
		for (int n=0;n<3;++n) p[n]=input[n];
		if (p[0]<0 || p[2]<0 || (p[1]<0 && !closed_alpha)) return false;
		if (p[1]<0) p[1]+=int(mesh.dims[1])-2;
		return true;
	}
	GPU_SETUP_METRIC double EdgeLength(int n,const int* p,bool dual=false) const
	{
		const double length=RawEdge(n,p[n],dual);
		return cylindrical && n==1 ? length*Line(0,p[0],dual) : length;
	}
	GPU_SETUP_METRIC double NodeWidth(int n,const int* input,bool dual=false) const
	{
		int p[3];if (!Map(input,p)) return 0;
		const double width=RawEdge(n,p[n],!dual);
		return cylindrical && n==1 ? width*Line(0,p[0],dual) : width;
	}
	GPU_SETUP_METRIC double NodeArea(int n,const int* input,bool dual=false) const
	{
		int p[3];if (!Map(input,p)) return 0;
		if (cylindrical)
		{
			if (uint32_t(p[n])>=mesh.dims[n] || uint32_t(p[0])>=mesh.dims[0]) return 0;
			if (n==2)
			{
				const double angle=RawEdge(1,p[1],dual)/unit;
				const double outer=dual ? Line(0,p[0])*unit+EdgeLength(0,p,false) : Line(0,p[0],true)*unit;
				const double inner=dual ? Line(0,p[0])*unit : outer-EdgeLength(0,p,true);
				return angle/2*(outer*outer-(inner<=0 ? 0 : inner*inner));
			}
		}
		return NodeWidth((n+1)%3,p,dual)*NodeWidth((n+2)%3,p,dual);
	}
	GPU_SETUP_METRIC double EdgeArea(int n,const int* p,bool dual=false) const
	{
		if (cylindrical && n==0) return EdgeLength(1,p,!dual)*EdgeLength(2,p,!dual);
		return NodeArea(n,p,dual);
	}
	GPU_SETUP_METRIC void Wrap(double* coord,bool yee=false) const
	{
		if (!closed_alpha) return;
		const double first=Line(1,0),period=6.283185307179586476925286766559;
		if (yee ? coord[1]>=first+period : coord[1]>first+period) coord[1]-=period;
		if (coord[1]<first) coord[1]+=period;
	}
	GPU_SETUP_METRIC bool Yee(int n,const int* p,double* coord,bool dual=false) const
	{
		for (int k=0;k<3;++k) coord[k]=Line(k,p[k],k==n ? !dual : dual);
		Wrap(coord,true);
		if (uint32_t(p[0])<first_radius) return false;
		return dual ? uint32_t(p[(n+1)%3])+1<mesh.dims[(n+1)%3] && uint32_t(p[(n+2)%3])+1<mesh.dims[(n+2)%3] : uint32_t(p[n])+1<mesh.dims[n];
	}
	GPU_SETUP_METRIC bool CellCenter(const int* input,double* coord) const
	{
		int p[3]={input[0],input[1],input[2]};
		if (closed_alpha && (p[1]<0 || p[1]>=int(mesh.dims[1])-2)) p[1]+=p[1]<0 ? int(mesh.dims[1])-2 : -int(mesh.dims[1])+2;
		for (int n=0;n<3;++n)
		{
			if (p[n]<0 || uint32_t(p[n])>=mesh.dims[n]) return false;
			coord[n]=Line(n,p[n],true);
		}
		if (first_radius && uint32_t(p[0])<first_radius+1)
		{
			const uint32_t a=uint32_t(p[1])/2;if (a>=child_alpha_count) return false;
			coord[1]=child_alpha[6*a+1];
		}
		return true;
	}
};

GPU_SETUP_METRIC inline void GPU_SetupInitializeLine(double* grid,uint32_t count,uint32_t p,double unit,bool closed)
{
	const double current=grid[6*p],previous=grid[6*(p ? p-1 : 0)],next=grid[6*(p+1<count ? p+1 : p)];
	const double first=grid[0],second=grid[6];
	const double dual=p+1<count ? 0.5*(current+next) : current+0.5*(current-previous);
	const double previous_dual=p ? 0.5*(previous+current) : first;
	grid[6*p+1]=dual;
	grid[6*p+2]=p+1<count ? next-current : closed ? grid[12]-second : previous-current;
	grid[6*p+3]=p ? current-previous : closed ? grid[6*(count-2)]-grid[6*(count-3)] : first-second;
	grid[6*p+4]=(p+1<count ? next-current : current-previous)*unit;
	grid[6*p+5]=(p ? dual-previous_dual : second-first)*unit;
}
#undef GPU_SETUP_METRIC
#endif
