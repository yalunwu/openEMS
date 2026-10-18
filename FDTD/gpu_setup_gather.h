/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_GATHER_H
#define GPU_SETUP_GATHER_H
#include "gpu_setup_metrics.h"
#include "engine_field_gather.h"
#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_GATHER __device__
#else
#define GPU_SETUP_GATHER
#endif

struct GPU_SetupGatherBuilder
{
	GPU_SetupMetrics metrics;
	int component;
	bool magnetic;
	GPU_SETUP_GATHER void Set(Engine_GatherEntry& entry,int slot,const int* p) const
	{
		const auto* dims=metrics.mesh.dims;
		entry.idx[slot]=((uint32_t(component)*dims[0]+uint32_t(p[0]))*dims[1]+uint32_t(p[1]))*dims[2]+uint32_t(p[2]);
		entry.delta[slot]=metrics.EdgeLength(component,p,magnetic);
	}
	GPU_SETUP_GATHER Engine_GatherEntry Build(const int* position,uint32_t interpolation) const
	{
		Engine_GatherEntry entry={};int p[3]={position[0],position[1],position[2]};
		if (metrics.closed_alpha)
		{
			if (interpolation==1 && p[1]==0) p[1]=int(metrics.mesh.dims[1])-2;
			else if (interpolation==2 && p[1]==int(metrics.mesh.dims[1])-2) p[1]=0;
		}
		const int n=component,a=(n+1)%3,b=(n+2)%3;const auto* dims=metrics.mesh.dims;
		const bool upper=uint32_t(p[0])+1==dims[0] || uint32_t(p[1])+1==dims[1] || uint32_t(p[2])+1==dims[2];
		if (interpolation==0) {entry.form=Engine_GatherEntry::RAW;Set(entry,0,p);return entry;}
		if ((!magnetic && interpolation==1) || (magnetic && interpolation==2))
		{
			if (uint32_t(p[n])+1==dims[n])
			{
				if (!magnetic) {--p[n];entry.form=Engine_GatherEntry::RAW;Set(entry,0,p);}return entry;
			}
			const double delta=metrics.EdgeLength(n,p,magnetic);
			if (!magnetic && delta==0) return entry;
			entry.form=Engine_GatherEntry::RAW;Set(entry,0,p);
			if (!magnetic && p[n]==0) return entry;
			p[n]+=magnetic ? 1 : -1;
			entry.form=Engine_GatherEntry::LERP;entry.rel=delta/(delta+metrics.EdgeLength(n,p,magnetic));Set(entry,1,p);
			return entry;
		}
		if (upper || (magnetic && (p[a]==0 || p[b]==0))) return entry;
		entry.form=Engine_GatherEntry::AVG4;Set(entry,0,p);
		p[a]+=magnetic ? -1 : 1;Set(entry,1,p);p[b]+=magnetic ? -1 : 1;Set(entry,2,p);
		p[a]+=magnetic ? 1 : -1;Set(entry,3,p);return entry;
	}
};
#undef GPU_SETUP_GATHER
#endif
