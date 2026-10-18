/*
*	Copyright (C) 2026 Sean Mollet (sean@malmoset.com)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <iostream>

#include "gpu_backend.h"
#include "gpu_backend_reference.h"
#include "gpu_setup_multigrid_plan.h"
#include "gpu_setup_processing.h"
#include "gpu_setup_controller.h"
#include "operator_cylinder.h"
#ifdef OPENEMS_WITH_HIP
#include "hip/gpu_backend_hip.h"
bool HIP_SetupMultiGridSplit(GPU_Backend*,const double*,unsigned int,double,unsigned int&);
std::shared_ptr<Engine_SamplingMesh> HIP_SetupSamplingMesh(GPU_SetupResult*,GPU_Backend*,const Operator_Base*,const unsigned int*,const unsigned int*,bool,bool,const unsigned int*,const double*);
bool HIP_SetupProcessingSnap(GPU_SetupResult*,GPU_Backend*,const Operator_Base*,const double*,const double*,unsigned int*,unsigned int*,bool,bool,int,bool,bool*,bool*,int&);
bool HIP_SetupProcessingIndices(GPU_SetupResult*,GPU_Backend*,const Operator_Base*,bool,bool,const double*,const double*,unsigned int*,unsigned int*,int&,int&);
void HIP_FinishInitialization(GPU_Backend*);
#endif
#include "engine.h"
#include "operator.h"
#include "extensions/engine_ext_upml.h"
#include "extensions/engine_ext_steadystate.h"
#ifdef OPENEMS_WITH_METAL
#include "metal/gpu_backend_metal.h"
#endif

GPU_Backend* GPU_Backend::New(const std::string& name)
{
	if (name=="reference")
		return new GPU_Backend_Reference();
	if (name!="auto")
		std::cerr << "GPU_Backend::New: Warning: unknown GPU backend \"" << name << "\", using the best available" << std::endl;

#ifdef OPENEMS_WITH_HIP
	GPU_Backend* hip = GPU_Backend_HIP::New();
	if (hip)
		return hip;
	std::cerr << "GPU_Backend::New: Warning: no usable HIP device found" << std::endl;
#endif

#ifdef OPENEMS_WITH_METAL
	GPU_Backend* metal = GPU_Backend_Metal::New();
	if (metal)
		return metal;
	std::cerr << "GPU_Backend::New: Warning: no Metal device found" << std::endl;
#endif

#if !defined(OPENEMS_WITH_HIP) && !defined(OPENEMS_WITH_METAL)
	std::cerr << "GPU_Backend::New: Warning: openEMS was built without a GPU backend (HIP or Metal)" << std::endl;
#endif
	// not the reference backend: it is far slower than the CPU engines, the caller has to say so
	return NULL;
}

bool GPU_SetupMultiGridSplit(GPU_Backend* backend,const double* lines,unsigned int count,double radius,unsigned int& split)
{
#ifdef OPENEMS_WITH_HIP
	return HIP_SetupMultiGridSplit(backend,lines,count,radius,split);
#else
	return false;
#endif
}

unsigned int GPU_SetupCoordinateFlags(const Operator_Base* op)
{
	const auto* cylinder=dynamic_cast<const Operator_Cylinder*>(op);
	return cylinder ? 1u|(cylinder->GetClosedAlpha() ? 2u : 0u)|(cylinder->GetR0Included() ? 4u : 0u) : 0;
}

std::shared_ptr<Engine_SamplingMesh> GPU_SetupSamplingMesh(const Operator_Base* op,const unsigned int start[3],const unsigned int stop[3],
    bool dual,bool optimal,const unsigned int step[3],const double resolution[3])
{
#ifdef OPENEMS_WITH_HIP
	const auto* fdtd=dynamic_cast<const Operator*>(op);
	if (auto* owner=fdtd ? fdtd->GetConstruction() : nullptr)
		return HIP_SetupSamplingMesh(owner->Result().get(),owner->Backend(),op,start,stop,dual,optimal,step,resolution);
#endif
	return {};
}

bool GPU_SetupProcessingSnap(const Operator_Base* op,const double start[3],const double stop[3],unsigned int first[3],unsigned int last[3],
    bool dual,bool full,int method,bool point,bool* first_inside,bool* last_inside,int& dimension)
{
#ifdef OPENEMS_WITH_HIP
	const auto* fdtd=dynamic_cast<const Operator*>(op);
	if (auto* owner=fdtd ? fdtd->GetConstruction() : nullptr)
		return HIP_SetupProcessingSnap(owner->Result().get(),owner->Backend(),op,start,stop,first,last,dual,full,method,point,first_inside,last_inside,dimension);
#endif
	return false;
}

bool GPU_SetupProcessingIndices(const Operator_Base* op,bool mode_match,bool dual,const double* start,const double* stop,
    unsigned int first[3],unsigned int last[3],int& normal,int& dimension)
{
#ifdef OPENEMS_WITH_HIP
	const auto* fdtd=dynamic_cast<const Operator*>(op);
	if (auto* owner=fdtd ? fdtd->GetConstruction() : nullptr)
		return HIP_SetupProcessingIndices(owner->Result().get(),owner->Backend(),op,mode_match,dual,start,stop,first,last,normal,dimension);
#endif
	return false;
}

bool GPU_SetupFusionOrder(Engine* eng,size_t regions)
{
	// the extension order: UPML (and steady-state) first, so that nothing else
	// touches the fields between the UPML hooks and the main update
	size_t num_upml = 0;
	size_t last_upml = 0;
	for (size_t n=0; n<eng->GetExtensionCount(); ++n)
		if (dynamic_cast<Engine_Ext_UPML*>(eng->GetExtension(n)))
		{
			++num_upml;
			last_upml = n;
		}
	if ((num_upml==0) || (num_upml!=regions))
		return false;
	for (size_t n=0; n<last_upml; ++n)
	{
		Engine_Extension* ext = eng->GetExtension(n);
		if (!dynamic_cast<Engine_Ext_UPML*>(ext) && !dynamic_cast<Engine_Ext_SteadyState*>(ext))
			return false;
	}
	return true;
}

void GPU_FinishInitialization(GPU_Backend* backend)
{
#ifdef OPENEMS_WITH_HIP
	HIP_FinishInitialization(backend);
#endif
}

bool GPU_UPMLFusionBox(Engine* eng, const std::vector<GPU_UPMLRegion>& regions, const unsigned int numLines[3],
                       unsigned int start[3], unsigned int stop[3])
{
	if (!GPU_SetupFusionOrder(eng,regions.size())) return false;

	// mark the region nodes, the others must form a box
	const size_t nx = numLines[0], ny = numLines[1], nz = numLines[2];
	std::vector<unsigned char> mask(nx*ny*nz, 0);
	for (size_t r=0; r<regions.size(); ++r)
	{
		const GPU_UPMLRegion& R = regions[r];
		for (int n=0; n<3; ++n)
			if (R.start[n]+R.size[n]>numLines[n])
				return false;
		for (size_t x=R.start[0]; x<R.start[0]+R.size[0]; ++x)
			for (size_t y=R.start[1]; y<R.start[1]+R.size[1]; ++y)
				for (size_t z=R.start[2]; z<R.start[2]+R.size[2]; ++z)
				{
					unsigned char& m = mask[(x*ny + y)*nz + z];
					if (m)
						return false;   // overlapping regions
					m = 1;
				}
	}
	unsigned int lo[3] = {numLines[0], numLines[1], numLines[2]};
	unsigned int hi[3] = {0, 0, 0};
	size_t free_nodes = 0;
	for (unsigned int x=0; x<nx; ++x)
		for (unsigned int y=0; y<ny; ++y)
			for (unsigned int z=0; z<nz; ++z)
				if (!mask[((size_t)x*ny + y)*nz + z])
				{
					++free_nodes;
					const unsigned int pos[3] = {x, y, z};
					for (int n=0; n<3; ++n)
					{
						lo[n] = std::min(lo[n], pos[n]);
						hi[n] = std::max(hi[n], pos[n]+1);
					}
				}
	if (free_nodes==0)
		return false;
	// the nodes left over must fill their bounding box, else they are no single box
	if (free_nodes!=(size_t)(hi[0]-lo[0])*(hi[1]-lo[1])*(hi[2]-lo[2]))
		return false;

	for (int n=0; n<3; ++n)
	{
		start[n] = lo[n];
		stop[n] = hi[n];
	}
	return true;
}

static FDTD_FLOAT GPU_Coefficient(const Operator* op, int c, int n, unsigned int x, unsigned int y, unsigned int z)
{
	switch (c)
	{
	case 0: return op->GetVV(n, x, y, z);
	case 1: return op->GetVI(n, x, y, z);
	case 2: return op->GetII(n, x, y, z);
	default: return op->GetIV(n, x, y, z);
	}
}

void GPU_NodeCoefficients(const Operator* op, const unsigned int numLines[3], size_t node, float values[12])
{
	const unsigned int x = node/((size_t)numLines[1]*numLines[2]);
	const unsigned int y = (node/numLines[2]) % numLines[1];
	const unsigned int z = node % numLines[2];
	for (int c=0; c<4; ++c)
		for (int n=0; n<3; ++n)
			values[3*c+n] = GPU_Coefficient(op, c, n, x, y, z);
}

void GPU_CoefficientArray(const Operator* op, const unsigned int numLines[3], int c, std::vector<float>& values)
{
	values.resize(3*(size_t)numLines[0]*numLines[1]*numLines[2]);
	size_t idx = 0;
	for (int n=0; n<3; ++n)
		for (unsigned int x=0; x<numLines[0]; ++x)
			for (unsigned int y=0; y<numLines[1]; ++y)
				for (unsigned int z=0; z<numLines[2]; ++z, ++idx)
					values[idx] = GPU_Coefficient(op, c, n, x, y, z);
}
