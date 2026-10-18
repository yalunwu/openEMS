/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_MULTIGRID_H
#define GPU_SETUP_MULTIGRID_H
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_MULTIGRID __device__
#else
#define GPU_SETUP_MULTIGRID
#endif

struct GPU_SetupInterpolationPoint {uint32_t positions[2];float weights[2];};

GPU_SETUP_MULTIGRID inline double GPU_SetupAlphaDelta(const double* grid,uint32_t count,uint32_t p,bool dual)
{
	if (dual) return p ? grid[6*p+1]-grid[6*(p-1)+1] : grid[6]-grid[0];
	return p+1<count ? grid[6*(p+1)]-grid[6*p] : grid[6*p]-grid[6*(p-1)];
}

GPU_SETUP_MULTIGRID inline GPU_SetupInterpolationPoint GPU_SetupInterpolateAlpha(const double* base,uint32_t count,
                                                                                 const double* child,uint32_t child_count,
                                                                                 uint32_t a,uint32_t component,bool currents,bool closed)
{
	GPU_SetupInterpolationPoint result={};
	const uint32_t odd=a%2,even=1-odd;
	uint32_t& p=result.positions[0];uint32_t& pp=result.positions[1];
	p=a/2;pp=a/2+((currents && component==0) || (!currents && component==1) ? odd-even : odd);
	if (currents && component==0)
	{
		if (!a) pp=closed ? child_count-3 : 0;
		if (!closed && a==count-2) pp=a/2-1;
		if (!closed && a==count-1) p=pp=a/2;
		double left=child[6*p+1],right=child[6*pp+1];
		if (!a && closed) right-=6.283185307179586476925286766559;
		result.weights[0]=p==pp ? 1.0f : float((right-base[6*a+1])/(right-left));
		result.weights[1]=p==pp ? 0.0f : float((base[6*a+1]-left)/(right-left));
	}
	else if (!currents && component==0)
	{
		const double left=child[6*p],right=child[6*pp];
		result.weights[0]=p==pp ? 1.0f : float((right-base[6*a])/(right-left));
		result.weights[1]=p==pp ? 0.0f : float((base[6*a]-left)/(right-left));
	}
	else if (currents)
	{
		if (!closed && a==1) p=2;
		if (!closed && a==count-2) pp=a/2-1;
		const double delta=GPU_SetupAlphaDelta(base,count,a,true);
		const double left=child[6*p],right=child[6*pp];
		if (p==pp) result.weights[0]=float(delta/GPU_SetupAlphaDelta(child,child_count,p,true));
		else
		{
			result.weights[0]=float((right-base[6*a])/(right-left));result.weights[0]*=delta/GPU_SetupAlphaDelta(child,child_count,p,true);
			result.weights[1]=float((base[6*a]-left)/(right-left));result.weights[1]*=delta/GPU_SetupAlphaDelta(child,child_count,pp,true);
		}
	}
	else
	{
		if (!a) pp=closed ? child_count-3 : 1;
		if (!closed && a==count-2) pp=a/2-1;
		if (!closed && a==count-1) p=pp=0;
		const double delta=GPU_SetupAlphaDelta(base,count,a,false),left=child[6*p+1],right=child[6*pp+1];
		if (p!=pp)
		{
			result.weights[0]=float((right-base[6*a+1])/(right-left));result.weights[0]*=delta/GPU_SetupAlphaDelta(child,child_count,p,false);
			result.weights[1]=float((base[6*a+1]-left)/(right-left));result.weights[1]*=delta/GPU_SetupAlphaDelta(child,child_count,pp,false);
		}
	}
	return result;
}
#undef GPU_SETUP_MULTIGRID
#endif
