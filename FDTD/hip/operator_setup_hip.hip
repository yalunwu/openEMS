/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_internal.h"
#include "FDTD/gpu_setup_upml.h"
#include "FDTD/gpu_operator_setup.h"
#include "FDTD/gpu_setup_adapters.h"
#include "FDTD/gpu_setup_geometry.h"
#include "FDTD/gpu_setup_memory.h"
#include "FDTD/gpu_setup_path.h"
#include "FDTD/gpu_setup_metrics.h"
#include "FDTD/gpu_setup_multigrid.h"
#include "FDTD/gpu_setup_constant.h"
#include "FDTD/gpu_setup_lookup.h"
#include "FDTD/gpu_setup_candidates.h"
#include "setup_geometry_index_hip.h"
#include "setup_candidates_hip.h"
#include "setup_scan_hip.h"
#include "setup_metadata_hip.h"
#include "setup_cell_material_hip.h"
#include "tools/setup_timer.h"
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iostream>
#include "FDTD/extensions/operator_ext_upml.h"
#include "FDTD/extensions/operator_ext_excitation.h"
#include "FDTD/extensions/operator_ext_mur_abc.h"
#include "FDTD/extensions/operator_ext_absorbing_bc.h"
#include "FDTD/extensions/operator_ext_conductingsheet.h"
#include "FDTD/extensions/operator_ext_debyematerial.h"
#include "FDTD/extensions/operator_ext_lorentzmaterial.h"
#include "FDTD/extensions/operator_ext_lumpedRLC.h"
#include "FDTD/extensions/operator_ext_tfsf.h"
#include "FDTD/extensions/operator_ext_steadystate.h"
#include "FDTD/extensions/operator_ext_cylinder.h"
#include "CSPropExcitation.h"
#include "CSPropLorentzMaterial.h"
#include "CSPropDebyeMaterial.h"
#include "CSPropLumpedElement.h"
#include "CSPrimBox.h"
#include <map>
#include <cstring>
#include <climits>
#include <typeinfo>


struct SetupDevice
{
	unsigned int dims[3], offsets[3], cells, average, stores, primitives;
	unsigned int cylindrical,closed_alpha,axis,first_radius;
	double grid_unit;
	float* axis_coefficients;
	const double* child_alpha;
	uint32_t child_alpha_count;
	int boundary[6];
	unsigned int bc_size[6];
	const double *grid, *geometry;
	const GPU_SetupPrimitive* shapes;
	const GPU_SetupProperty* properties;
	const GPU_SetupBVHNode* geometry_index;
	const GPU_SetupExpression* expressions;
	const GPU_SetupInstruction* expression_code;
	const double* expression_constants;
	double* expression_stack;
	unsigned int expression_stack_size;
	const GPU_SetupDiscreteMaterial* discrete_materials;
	const float* material_data;
	const uint8_t* material_indices;
	const GPU_SetupFileWeight* file_weights;
	const GPU_SetupDispersionPole* dispersion;
	const unsigned int* candidates;
	const unsigned int* candidate_order;
	unsigned int candidate_offsets[4];
	unsigned int* used;
	float *ec[4], *material[4], *coeff[4];
	float* stored_material[4];
};

__device__ static unsigned int ECIndex(const SetupDevice& s, int n, const int* p)
{
	return n*s.cells+(p[2]*s.dims[1]+p[1])*s.dims[0]+p[0];
}
__device__ static double Grid(const SetupDevice& s, int n, int p, int kind)
{
	return s.grid[s.offsets[n]+6*p+kind];
}
__host__ __device__ static GPU_SetupMetrics Metrics(const SetupDevice& s)
{
	GPU_SetupMetrics result={{{s.dims[0],s.dims[1],s.dims[2]},{s.offsets[0],s.offsets[1],s.offsets[2]},s.grid},
	                         s.grid_unit,s.cylindrical,s.closed_alpha,s.axis,s.first_radius,s.child_alpha,s.child_alpha_count};return result;
}

__global__ static void SetupGridMetrics(SetupDevice s,bool child)
{
	const uint32_t p=blockIdx.x*blockDim.x+threadIdx.x,n=blockIdx.y;
	const uint32_t count=child ? s.child_alpha_count : s.dims[n];if (p>=count) return;
	double* grid=const_cast<double*>(child ? s.child_alpha : s.grid+s.offsets[n]);
	GPU_SetupInitializeLine(grid,count,p,s.grid_unit,s.closed_alpha && (child || n==1));
	for (uint32_t k=1;k<6;++k) if (!isfinite(grid[6*p+k])) atomicOr(s.used+s.primitives+3,1u);
}
__device__ static double Width(const SetupDevice& s,int n,const int* p,bool dual)
{
	return Metrics(s).EdgeLength(n,p,dual);
}
__device__ static double Width(const SetupDevice& s,int n,const uint32_t* p,bool dual)
{
	const int pos[3]={int(p[0]),int(p[1]),int(p[2])};return Width(s,n,pos,dual);
}
__device__ static double Area(const SetupDevice& s, int n, const int* p, bool dual)
{
	return Metrics(s).EdgeArea(n,p,dual);
}
__device__ static bool Inside(const SetupDevice& s, unsigned int prim, const double* coord)
{
	int error=0;
	const size_t thread=blockIdx.x*blockDim.x+threadIdx.x;
	GPU_SetupExpressionView expressions={s.expressions,s.expression_code,s.expression_constants,
	                                    s.expression_stack+thread*s.expression_stack_size,s.expression_stack_size,&error};
	const bool inside=GPU_SetupContains(s.shapes[prim],s.geometry,coord,s.geometry_index,&expressions);
	if (error) atomicOr(s.used+s.primitives+3,1u);
	return inside;
}

struct CurveCount
{
	unsigned int count;
	__device__ void operator()(int, const uint32_t*) {++count;}
};

struct CurveWrite
{
	SetupDevice s;
	__device__ void operator()(int direction, const uint32_t* p)
	{
		const unsigned int id=((direction*s.dims[0]+p[0])*s.dims[1]+p[1])*s.dims[2]+p[2];
		atomicExch(s.coeff[0]+id,0.0f); atomicExch(s.coeff[1]+id,0.0f);
		atomicAdd(s.used+s.primitives+direction,1u);
	}
};

__global__ static void SetupCurves(SetupDevice s)
{
	const unsigned int primitive=blockIdx.x*blockDim.x+threadIdx.x;
	if (primitive>=s.primitives || !(s.properties[primitive].type&CSProperties::METAL)) return;
	const GPU_SetupPrimitive& d=s.shapes[primitive];
	if (d.shape!=GPU_SetupShape::Curve) return;
	GPU_SetupMeshView mesh={{s.dims[0],s.dims[1],s.dims[2]},{s.offsets[0],s.offsets[1],s.offsets[2]},s.grid};
	for (uint32_t segment=1;segment<d.count;++segment)
	{
		double start[3],stop[3];
		for (int n=0;n<3;++n)
		{
			start[n]=stop[n]=0;
			for (int k=0;k<3;++k)
			{
				start[n]+=d.forward[4*n+k]*s.geometry[d.offset+3*(segment-1)+k];
				stop[n]+=d.forward[4*n+k]*s.geometry[d.offset+3*segment+k];
			}
			start[n]+=d.forward[4*n+3]; stop[n]+=d.forward[4*n+3];
		}
		if (s.cylindrical) {GPU_SetupConvertCoordinates(start,0,1);GPU_SetupConvertCoordinates(stop,0,1);}
		if (!isfinite(start[0]) || !isfinite(start[1]) || !isfinite(start[2]) ||
		    !isfinite(stop[0]) || !isfinite(stop[1]) || !isfinite(stop[2]))
		{ atomicOr(s.used+s.primitives+3,1u); continue; }
		CurveCount counter={0};
		if (!GPU_SetupFindPath(mesh,start,stop,counter,s.cylindrical,s.closed_alpha) || !counter.count) continue;
		CurveWrite output={s};GPU_SetupFindPath(mesh,start,stop,output,s.cylindrical,s.closed_alpha);
		atomicOr(s.used+primitive,1u);
	}
}
__device__ static unsigned int Property(const SetupDevice& s, const double* coord, const int* p, unsigned int which)
{
	unsigned int row=which*(s.dims[0]*s.dims[1]+1)+p[0]*s.dims[1]+p[1];
	for (unsigned int i=s.candidates[row];i<s.candidates[row+1];++i) {
		unsigned int prim=s.candidates[i];
		if (Inside(s,prim,coord)) { atomicOr(s.used+prim,1u); return prim; }
	}
	return 0;
}

__device__ static double Evaluate(const SetupDevice& s, uint32_t expression, const double* variables)
{
	if (!expression) return 1;
	int error=0;
	const size_t thread=blockIdx.x*blockDim.x+threadIdx.x;
	const double value=GPU_SetupEvaluate(s.expressions[expression],s.expression_code,s.expression_constants,variables,
	                                     s.expression_stack+thread*s.expression_stack_size,s.expression_stack_size,error);
	if (error) atomicOr(s.used+s.primitives+3,1u);
	return value;
}

__device__ static void WeightCoordinates(const GPU_SetupProperty& property, const double* coord, double* variables)
{
	for (int n=0;n<3;++n) variables[n]=coord[n];
	if (property.coordinate_system==1)
	{
		variables[0]=coord[0]*cos(coord[1]); variables[1]=coord[0]*sin(coord[1]);
	}
	for (int n=0;n<3;++n) variables[n]-=property.weight_origin[n];
	variables[3]=sqrt(variables[0]*variables[0]+variables[1]*variables[1]);
	variables[4]=sqrt(variables[0]*variables[0]+variables[1]*variables[1]+variables[2]*variables[2]);
	variables[5]=atan2(variables[1],variables[0]);
	variables[6]=asin(1.0)-atan(variables[2]/variables[3]);
	if (property.coordinate_system==1 && !(property.type&CSProperties::EXCITATION))
	{
		variables[3]=coord[0]; variables[4]=sqrt(coord[0]*coord[0]+coord[2]*coord[2]);
		variables[5]=coord[1]; variables[6]=asin(1.0)-atan(coord[2]/coord[0]);
	}
}

__device__ static double MaterialValue(const SetupDevice& s, unsigned int prim, int component, int kind, const double* coord)
{
	const GPU_SetupProperty& property=s.properties[prim];
	const double base=kind==4 ? property.density : property.values[component][kind];
	if (property.discrete)
	{
		const GPU_SetupDiscreteMaterial& d=s.discrete_materials[property.discrete];
		const int index=GPU_SetupDatabasePosition(d,s.material_data,s.material_indices,coord);
		if (index>=0 && d.channels[kind]!=UINT32_MAX) return s.material_data[d.channels[kind]+index];
	}
	const uint32_t expression=kind==4 ? property.density_weight : property.weights[component][kind];
	if (!expression && !property.file_weight) return base;
	double variables[7]; WeightCoordinates(property,coord,variables);
	if (property.file_weight) return base*GPU_SetupInterpolateFile(s.file_weights[property.file_weight],s.geometry,component,variables);
	return base*Evaluate(s,expression,variables);
}

__device__ static double DispersionValue(const SetupDevice& s, unsigned int prim, unsigned int pole, int component, int magnetic, int kind, const double* coord)
{
	const GPU_SetupProperty& property=s.properties[prim];
	if (pole>=property.dispersion_count) return 0;
	const GPU_SetupDispersionPole& d=s.dispersion[property.dispersion_offset+pole];
	const uint32_t expression=d.weights[component][magnetic][kind];
	if (!expression) return d.values[component][magnetic][kind];
	double variables[7]; WeightCoordinates(property,coord,variables);
	return d.values[component][magnetic][kind]*Evaluate(s,expression,variables);
}

struct CurveSourceWrite
{
	SetupDevice s;
	unsigned int primitive;
	double dt;
	GPU_SetupSourcePoint* points;
	unsigned long long capacity, count;
	__device__ void operator()(int n, const uint32_t* p)
	{
		const GPU_SetupProperty& prop=s.properties[primitive];
		if (prop.excitation_type>1 || !prop.active[n] || p[n]>=s.dims[n]-1) return;
		const int pos[3]={int(p[0]),int(p[1]),int(p[2])};double coord[3];if (!Metrics(s).Yee(n,pos,coord)) return;
		const float amplitude=static_cast<float>(MaterialValue(s,primitive,n,0,coord)*Width(s,n,p,false));
		if (!isfinite(amplitude)) atomicOr(s.used+s.primitives+3,1u);
		const unsigned int index=((n*s.dims[0]+p[0])*s.dims[1]+p[1])*s.dims[2]+p[2];
		if (points && prop.excitation_type==1) {s.coeff[0][index]=0;s.coeff[1][index]=0;}
		if (amplitude==0) return;
		if (points)
		{
			if (count>=capacity) {atomicOr(s.used+s.primitives+3,1u);return;}
			GPU_SetupSourcePoint point={index,amplitude,static_cast<uint32_t>(prop.delay/dt),0,0,0};
			points[count]=point;
		}
		++count;
	}
};

// Curve contributions are ordered, including duplicates at segment junctions.
// Count and fill run on the device; the host receives only the count before
// allocating the exact output. The stepping backend preserves duplicate order.
__global__ static void SetupSourceCurves(SetupDevice s, double dt, GPU_SetupSourcePoint* output,
                                         unsigned long long capacity, unsigned long long* total)
{
	if (blockIdx.x || threadIdx.x) return;
	GPU_SetupMeshView mesh={{s.dims[0],s.dims[1],s.dims[2]},{s.offsets[0],s.offsets[1],s.offsets[2]},s.grid};
	CurveSourceWrite writer={s,0,dt,output,capacity,0};
	for (unsigned int primitive=1;primitive<s.primitives;++primitive)
	{
		const GPU_SetupProperty& prop=s.properties[primitive];
		const GPU_SetupPrimitive& d=s.shapes[primitive];
		if (!(prop.type&CSProperties::EXCITATION) || !prop.enabled || d.shape!=GPU_SetupShape::Curve) continue;
		writer.primitive=primitive;
		for (uint32_t segment=1;segment<d.count;++segment)
		{
			double start[3],stop[3];
			for (int n=0;n<3;++n)
			{
				start[n]=stop[n]=0;
				for (int k=0;k<3;++k)
				{
					start[n]+=d.forward[4*n+k]*s.geometry[d.offset+3*(segment-1)+k];
					stop[n]+=d.forward[4*n+k]*s.geometry[d.offset+3*segment+k];
				}
				start[n]+=d.forward[4*n+3];stop[n]+=d.forward[4*n+3];
			}
			if (s.cylindrical) {GPU_SetupConvertCoordinates(start,0,1);GPU_SetupConvertCoordinates(stop,0,1);}
			if (!isfinite(start[0])||!isfinite(start[1])||!isfinite(start[2])||!isfinite(stop[0])||!isfinite(stop[1])||!isfinite(stop[2]))
			{atomicOr(s.used+s.primitives+3,1u);continue;}
			CurveCount path={0};
			if (!GPU_SetupFindPath(mesh,start,stop,path,s.cylindrical,s.closed_alpha) || !path.count) continue;
			atomicOr(s.used+primitive,1u);
			GPU_SetupFindPath(mesh,start,stop,writer,s.cylindrical,s.closed_alpha);
		}
	}
	*total=writer.count;
}

__global__ static void SetupSourceSummary(SetupDevice s,const GPU_SetupSourcePoint* points,uint32_t count,uint32_t* summary)
{
	const uint32_t q=blockIdx.x*blockDim.x+threadIdx.x;
	if (q>=count) return;
	const uint32_t index=points[q].index,node=index%s.cells;
	if (!isfinite(points[q].amplitude)) atomicOr(s.used+s.primitives+3,1u);
	atomicAdd(summary+index/s.cells,1u);
	const uint32_t p[3]={node/(s.dims[1]*s.dims[2]),(node/s.dims[2])%s.dims[1],node%s.dims[2]};
	if (p[0]+1>=s.dims[0] || p[1]+1>=s.dims[1] || p[2]+1>=s.dims[2]) atomicOr(summary+3,1u);
}

__global__ static void SetupTFSFDelay(const float* points,uint32_t count,uint32_t* maximum)
{
	const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i<count) atomicMax(maximum,__float_as_uint(points[3*i]));
}

__global__ static void SetupSteadyStateIndices(const uint32_t* input,uint32_t* output,uint32_t count,uint32_t nx,uint32_t ny,uint32_t nz)
{
	const uint32_t q=blockIdx.x*blockDim.x+threadIdx.x;
	if (q<count) output[q]=((input[4*q+3]*nx+input[4*q])*ny+input[4*q+1])*nz+input[4*q+2];
}

__global__ static void SetupRLCFinalize(const float* input,uint32_t count,uint32_t offset,uint32_t total,uint32_t* indices,float* coefficients)
{
	const uint32_t q=blockIdx.x*blockDim.x+threadIdx.x;
	if (q>=count) return;
	indices[offset+q]=__float_as_uint(input[10ull*q]);
	for (uint32_t k=0;k<9;++k) coefficients[size_t(k)*total+offset+q]=input[10ull*q+k+1];
}

__global__ static void SetupRLCDuplicates(const uint32_t* indices,uint32_t count,uint32_t* slots,uint32_t capacity,uint32_t* duplicate)
{
	const uint32_t q=blockIdx.x*blockDim.x+threadIdx.x;
	if (q>=count) return;
	const uint32_t value=indices[q];uint32_t slot=(value*2654435761u)&(capacity-1);
	for (uint32_t probe=0;probe<capacity;++probe,slot=(slot+1)&(capacity-1))
	{
		const uint32_t previous=atomicCAS(slots+slot,UINT32_MAX,value);
		if (previous==UINT32_MAX) return;
		if (previous==value) {atomicOr(duplicate,1u);return;}
	}
	atomicOr(duplicate,2u);
}
__device__ static void Effective(const SetupDevice& s, int n, const int* p, double* m)
{
	int a=(n+1)%3,b=(n+2)%3;
	double area=0,length=0,coord[3]={Grid(s,0,p[0],0),Grid(s,1,p[1],0),Grid(s,2,p[2],0)};
	for (int k=0;k<4;++k) m[k]=0;
	for (int q=0;q<4;++q) {
		int loc[3]={p[0],p[1],p[2]}; loc[a]-=q%2; loc[b]-=q/2;
		double c[3];
		if (s.average==0) {
			for (int k=0;k<3;++k) c[k]=coord[k];
			c[n]+=Grid(s,n,p[n],2)*0.5;
			c[a]+=(q%2==0?Grid(s,a,p[a],2):-Grid(s,a,p[a],3))*0.25;
			c[b]+=(q/2==0?Grid(s,b,p[b],2):-Grid(s,b,p[b],3))*0.25;
		} else {
			if (!Metrics(s).CellCenter(loc,c)) continue;
		}
		Metrics(s).Wrap(c);
		double weight=Metrics(s).NodeArea(n,loc,true);
		const unsigned int prim=Property(s,c,p,0);
		const double raw[]={MaterialValue(s,prim,n,0,c),MaterialValue(s,prim,n,1,c)};
		m[0]+=raw[0]*weight; m[1]+=raw[1]*weight; area+=weight;
	}
	m[0]*=8.85418781762e-12/area; m[1]/=area;
	for (int q=0;q<2;++q) {
		int loc[3]={p[0],p[1],p[2]}; loc[n]-=1-q;
		double c[3];
		if (s.average==0) {
			for (int k=0;k<3;++k) c[k]=coord[k];
			c[n]+=(q==0?-Grid(s,n,p[n],3):Grid(s,n,p[n],2))*0.25;
			c[a]+=Grid(s,a,p[a],2)*0.5; c[b]+=Grid(s,b,p[b],2)*0.5;
		} else {
			if (!Metrics(s).CellCenter(loc,c)) continue;
		}
		Metrics(s).Wrap(c);
		double weight=Metrics(s).NodeWidth(n,loc,true);
		const unsigned int prim=Property(s,c,p,0);
		const double mu=MaterialValue(s,prim,n,2,c),sigma=MaterialValue(s,prim,n,3,c);
		m[2]+=weight/mu;
		if (sigma!=0) m[3]+=weight/sigma; else m[3]=0;
		length+=weight;
	}
	m[2]=length*1.256637062e-6/m[2]; if (m[3]!=0) m[3]=length/m[3];
}
__device__ static void Position(const SetupDevice& s, unsigned int node, int* p)
{
	p[0]=node/(s.dims[1]*s.dims[2]); p[1]=(node/s.dims[2])%s.dims[1]; p[2]=node%s.dims[2];
}
__global__ static void SetupEC(SetupDevice s)
{
	unsigned int id=blockIdx.x*blockDim.x+threadIdx.x;
	if (id>=3*s.cells) return;
	int n=id/s.cells,p[3]; Position(s,id%s.cells,p);
	double m[4]; Effective(s,n,p,m);
	for (int k=0;k<4;++k) {
		if (!isfinite(m[k])) atomicOr(s.used+s.primitives+3,1u);
		if (s.stored_material[k]) s.stored_material[k][id]=float(m[k]);
	}
	unsigned int e=ECIndex(s,n,p);
	if (uint32_t(p[0])<s.first_radius) return;
	double length=Width(s,n,p,false),area=Area(s,n,p,false);
	s.ec[0][e]=length ? float(m[0]*area/length) : 0;s.ec[1][e]=length ? float(m[1]*area/length) : 0;
	length=Width(s,n,p,true);area=Area(s,n,p,true);
	s.ec[2][e]=length ? float(m[2]*area/length) : 0;s.ec[3][e]=length ? float(m[3]*area/length) : 0;
}
__device__ static unsigned int ShiftedEC(const SetupDevice& s, int n, const int* p)
{
	int c[3]; for (int k=0;k<3;++k) c[k]=max(0,min(p[k],int(s.dims[k])-1));
	return ECIndex(s,n,c);
}
__device__ static float Reciprocal(const SetupDevice& s, int l, const int* lp, int c, const int* cp)
{
	float product=s.ec[2][ShiftedEC(s,l,lp)]*s.ec[0][ShiftedEC(s,c,cp)];
	return 1.0f/product;
}
__device__ static double Timestep(const SetupDevice& s, int n, const int* p, int method)
{
	int a=(n+1)%3,b=(n+2)%3,pa[3]={p[0],p[1],p[2]},pb[3]={p[0],p[1],p[2]},pab[3]={p[0],p[1],p[2]};
	--pa[a]; --pb[b]; --pab[a]; --pab[b];
	if (method!=3) {
		float sum=4.0f/s.ec[2][ECIndex(s,a,p)];
		sum+=4.0f/s.ec[2][ShiftedEC(s,a,pb)]; sum+=4.0f/s.ec[2][ECIndex(s,b,p)]; sum+=4.0f/s.ec[2][ShiftedEC(s,b,pa)];
		float quotient=sum/s.ec[0][ECIndex(s,n,p)];
		return double(2.0f/sqrtf(quotient));
	}
	int plusa[3]={p[0],p[1],p[2]},plusb[3]={p[0],p[1],p[2]}; ++plusa[a]; ++plusb[b];
	float pair=Reciprocal(s,b,p,n,plusa)+Reciprocal(s,b,p,n,p); double w=pair;
	pair=Reciprocal(s,a,p,n,plusb)+Reciprocal(s,a,p,n,p); w+=double(pair);
	pair=Reciprocal(s,b,pa,n,p)+Reciprocal(s,b,pa,n,pa); w+=double(pair);
	pair=Reciprocal(s,a,pab,n,pa)+Reciprocal(s,a,pab,n,pab); w+=double(pair);
	double other[2];
	for (int q=0;q<2;++q) {
		int cp[3]={p[0],p[1],p[2]}; cp[n]+=q;
		double t[4]={double(Reciprocal(s,b,p,a,cp)),double(Reciprocal(s,b,pa,a,cp)),
		             double(Reciprocal(s,a,p,b,cp)),double(Reciprocal(s,a,pb,b,cp))};
		other[q]=t[0]+t[1]+t[2]+t[3]-2*fmin(fmin(t[0],t[1]),fmin(t[2],t[3]));
	}
	return 2/sqrt(w+other[0]+other[1]);
}
__global__ static void SetupTimestep(SetupDevice s, int method, double* result)
{
	__shared__ double minima[256];
	unsigned int id=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x;
	double value=1e200;
	if (id<3*s.cells) { int p[3]; Position(s,id%s.cells,p); double v=Timestep(s,id/s.cells,p,method); if (v>0 && v<value) value=v; }
	minima[lane]=value; __syncthreads();
	for (unsigned int step=128;step;step/=2) {
		if (lane<step) minima[lane]=fmin(minima[lane],minima[lane+step]);
		__syncthreads();
	}
	if (!lane) result[blockIdx.x]=minima[0];
}

__global__ static void ReduceTimestep(const double* input, unsigned int count, double* output)
{
	__shared__ double minima[256];
	const unsigned int id=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x;
	minima[lane]=id<count?input[id]:1e200;
	__syncthreads();
	for (unsigned int step=128;step;step/=2)
	{
		if (lane<step) minima[lane]=fmin(minima[lane],minima[lane+step]);
		__syncthreads();
	}
	if (!lane) output[blockIdx.x]=minima[0];
}
__global__ static void SetupCoefficients(SetupDevice s, double dt)
{
	unsigned int id=blockIdx.x*blockDim.x+threadIdx.x;
	if (id>=3*s.cells) return;
	int n=id/s.cells,p[3]; Position(s,id%s.cells,p);
	if (uint32_t(p[0])<s.first_radius) return;
	unsigned int e=ECIndex(s,n,p);
	double C=s.ec[0][e],G=s.ec[1][e],L=s.ec[2][e],R=s.ec[3][e];
	s.coeff[0][id]=C>0?float((1-dt*G/2/C)/(1+dt*G/2/C)):0;
	s.coeff[1][id]=C>0?float((dt/C)/(1+dt*G/2/C)):0;
	s.coeff[2][id]=L>0?float((1-dt*R/2/L)/(1+dt*R/2/L)):0;
	s.coeff[3][id]=L>0?float((dt/L)/(1+dt*R/2/L)):0;
	for (int a=0;a<3;++a) {
		if ((a!=n && p[a]==0 && s.boundary[2*a]!=-1) || (p[a]==int(s.dims[a])-1 && s.boundary[2*a+1]!=-1)) { s.coeff[0][id]=0; s.coeff[1][id]=0; }
		const bool lower_pmc=s.boundary[2*a]==1 && !(a==1 && s.closed_alpha) && !(a==0 && s.axis);
		const bool upper_pmc=s.boundary[2*a+1]==1 && !(a==1 && s.closed_alpha);
		if (p[a]==int(s.dims[a])-1 || (p[a]==0 && lower_pmc) || (a!=n && p[a]==int(s.dims[a])-2 && upper_pmc)) { s.coeff[2][id]=0; s.coeff[3][id]=0; }
	}
	double coord[3];Metrics(s).Yee(n,p,coord);
	unsigned int prim=Property(s,coord,p,1);
	if (s.properties[prim].metal) { s.coeff[0][id]=0; s.coeff[1][id]=0; atomicAdd(s.used+s.primitives+n,1u); }
}

__global__ static void SetupCylinderAxis(SetupDevice s,double dt)
{
	const uint32_t z=blockIdx.x*blockDim.x+threadIdx.x;if (z>=s.dims[2]) return;
	int p[3]={0,0,int(z)};double capacitance=0,conductance=0;
	for (p[1]=0;p[1]<int(s.dims[1])-2;++p[1])
	{
		double material[4];Effective(s,2,p,material);
		const double scale=Area(s,2,p,false)/Width(s,2,p,false);
		capacitance+=material[0]*scale;conductance+=material[1]*scale;
	}
	const double denominator=1+dt*conductance/2/capacitance;
	float vv=float((1-dt*conductance/2/capacitance)/denominator),vi=float((dt/capacitance)/denominator);
	for (p[1]=0;p[1]<int(s.dims[1]);++p[1])
	{
		const uint32_t e=ECIndex(s,2,p);s.ec[0][e]=float(capacitance);s.ec[1][e]=float(conductance);
	}
	p[1]=0;const uint32_t index=2*s.cells+z;s.coeff[0][index]=1;
	double coord[3];Metrics(s).Yee(2,p,coord);
	const uint32_t primitive=Property(s,coord,p,1);
	if (s.properties[primitive].metal)
	{
		vv=0;vi=0;s.coeff[0][index]=0;const uint32_t e=ECIndex(s,2,p);s.ec[0][e]=0;s.ec[1][e]=0;
	}
	s.axis_coefficients[z]=vv;s.axis_coefficients[s.dims[2]+z]=vi;
	if (!isfinite(vv) || !isfinite(vi)) atomicOr(s.used+s.primitives+3,1u);
}

struct SetupMultiGridOutput {uint32_t* positions[4];float* weights[4];};
__global__ static void SetupMultiGridTables(SetupDevice s,SetupMultiGridOutput output)
{
	const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x,alpha=s.dims[1];if (id>=2*alpha) return;
	for (uint32_t current=0;current<2;++current)
	{
		const auto point=GPU_SetupInterpolateAlpha(s.grid+s.offsets[1],alpha,s.child_alpha,s.child_alpha_count,id%alpha,id/alpha,current!=0,s.closed_alpha!=0);
		for (uint32_t k=0;k<2;++k)
		{
			output.positions[2*current+k][id]=point.positions[k];output.weights[2*current+k][id]=point.weights[k];
			if (point.positions[k]>=s.child_alpha_count || !isfinite(point.weights[k])) atomicOr(s.used+s.primitives+3,1u);
		}
	}
}


// The Vulkan setup equations use small indexed vectors; EC storage remains AdrOp's x-fast layout.
struct HIP_SetupI3 {
    int x,y,z;
    __device__ HIP_SetupI3(int v=0) : x(v),y(v),z(v) {}
    __device__ HIP_SetupI3(int a,int b,int c) : x(a),y(b),z(c) {}
    __device__ int& operator[](int n) { return n==0?x:n==1?y:z; }
    __device__ int operator[](int n) const { return n==0?x:n==1?y:z; }
    __device__ HIP_SetupI3 operator+(HIP_SetupI3 other) const { return HIP_SetupI3(x+other.x,y+other.y,z+other.z); }
};
struct HIP_SetupD2 {
    double x,y;
    __device__ HIP_SetupD2(double a=0,double b=0) : x(a),y(b) {}
    __device__ double operator[](int n) const { return n==0?x:y; }
};
struct HIP_SetupD3 {
    double x,y,z;
    __device__ HIP_SetupD3(double v=0) : x(v),y(v),z(v) {}
    __device__ HIP_SetupD3(double a,double b,double c) : x(a),y(b),z(c) {}
    __device__ double& operator[](int n) { return n==0?x:n==1?y:z; }
    __device__ double operator[](int n) const { return n==0?x:n==1?y:z; }
};
struct HIP_SetupD4 { double x,y,z,w; };
struct HIP_SetupCircuit {
    const SetupDevice* s; int kind;
    __device__ float& operator[](unsigned int id) const {
        int p[3]; Position(*s,id%s->cells,p);
        return s->ec[kind][ECIndex(*s,id/s->cells,p)];
    }
};
struct HIP_SetupExtensionParams { unsigned int mode,reserved; double dt; };
struct HIP_SetupExtensionKernel {
    typedef unsigned int uint;
    typedef HIP_SetupI3 I3;
    typedef HIP_SetupD2 D2;
    typedef HIP_SetupD3 D3;
    typedef HIP_SetupD4 D4;
    SetupDevice s;
    HIP_SetupExtensionParams pc;
    const double *aux;
    const unsigned int* cand;
    const unsigned int* BC_SIZE;
    unsigned int *used,*winners,CELLS,PEC_INDEX,ERROR_INDEX;
    float *material0,*material1,*material2,*ext,*vv,*vi,*ii,*iv;
    double* reduction;
    I3 DIMS;
    HIP_SetupCircuit ec0,ec1,ec2,ec3;
    __device__ HIP_SetupExtensionKernel(SetupDevice device,HIP_SetupExtensionParams params,const double* a,
        float* output,double* sums,unsigned int* scan) : s(device),pc(params),aux(a),cand(s.candidates),
        BC_SIZE(s.bc_size),used(s.used),winners(scan),CELLS(s.cells),PEC_INDEX(s.primitives),ERROR_INDEX(s.primitives+3),
        material0(s.material[0]),material1(s.material[1]),material2(s.material[2]),ext(output),
        vv(s.coeff[0]),vi(s.coeff[1]),ii(s.coeff[2]),iv(s.coeff[3]),reduction(sums),
        DIMS(s.dims[0],s.dims[1],s.dims[2]),ec0{&s,0},ec1{&s,1},ec2{&s,2},ec3{&s,3} {}
    __device__ uint linear(I3 p) const { return (p.x*s.dims[1]+p.y)*s.dims[2]+p.z; }
    __device__ I3 position(uint id) const { return I3(id/(s.dims[1]*s.dims[2]),(id/s.dims[2])%s.dims[1],id%s.dims[2]); }
    __device__ double g(int n,int p,int kind) const { return Grid(s,n,p,kind); }
    __device__ double width(int n,I3 p,bool dual) const {int pos[3]={p.x,p.y,p.z};return Width(s,n,pos,dual);}
    __device__ bool yee(int n,I3 p,D3& c,bool dual=false) const {int pos[3]={p.x,p.y,p.z};double coord[3];const bool valid=Metrics(s).Yee(n,pos,coord,dual);c=D3(coord[0],coord[1],coord[2]);return valid;}
    __device__ double area(int n,I3 p,bool dual) const { int pos[3]={p.x,p.y,p.z}; return Area(s,n,pos,dual); }
    __device__ bool inside(uint prim,D3 c) const { double coord[3]={c.x,c.y,c.z}; return Inside(s,prim,coord); }
    __device__ double value(uint prim,int n,int kind,D3 c) const {
        double coord[3]={c.x,c.y,c.z}; return MaterialValue(s,prim,n,kind,coord);
    }
    __device__ double dispersion(uint prim,uint pole,int n,int magnetic,int kind,D3 c) const {
        double coord[3]={c.x,c.y,c.z}; return DispersionValue(s,prim,pole,n,magnetic,kind,coord);
    }
    __device__ uint property(D3 c,I3 p,uint which) const {
        double coord[3]={c.x,c.y,c.z}; int pos[3]={p.x,p.y,p.z}; return Property(s,coord,pos,which);
    }
    __device__ uint sourceProperty(D3 c,I3 p) const {
        uint row=2*(s.dims[0]*s.dims[1]+1)+p.z*s.dims[1]+p.y;
        for(uint q=cand[row];q<cand[row+1];++q) {
            uint prim=cand[q]; if(inside(prim,c)) { atomicOr(used+prim,1u); return prim; }
        }
        return 0;
    }
    __device__ D4 effective(int n,I3 p) const {
        int pos[3]={p.x,p.y,p.z}; double values[4]; Effective(s,n,pos,values);
        D4 result={values[0],values[1],values[2],values[3]}; return result;
    }
__device__ void lorentz(int n,I3 p,uint pole,D3& v,D3& i) {
    v=D3(1.0,0.0,0.0); i=v;
    uint cell=uint(n)*CELLS+linear(p);
    for(int magnetic=0;magnetic<2;++magnetic) {
        D3 c;bool valid=yee(n,p,c,magnetic!=0);
        if(s.axis && n==2 && p.x==0 && !magnetic) c.y=g(1,0,0);
        double coefficient=double(magnetic==0?vi[cell]:iv[cell]);
        if(!valid||coefficient==0.0) continue;
        uint prim=property(c,p,1);
        if (!(s.properties[prim].type&CSProperties::LORENTZMATERIAL)) continue;
        double plasma=dispersion(prim,pole,n,magnetic,0,c)*2.0*3.1415926535897932384626433832795;
        double capacitance=double(magnetic==0?ec0[cell]:ec2[cell]);
        if(plasma<=0.0||capacitance<=0.0) continue;
        double reactance=1.0/(plasma*plasma*capacitance);
        const double tau=dispersion(prim,pole,n,magnetic,1,c);
        double loss=tau>0.0?reactance/tau:0.0;
        double omega=dispersion(prim,pole,n,magnetic,2,c)*2.0*3.1415926535897932384626433832795;
        double other=omega>0.0?1.0/(omega*omega*reactance):0.0;
        D3 result=D3((2.0*reactance-pc.dt*loss)/(2.0*reactance+pc.dt*loss),
            pc.dt/(reactance+pc.dt*loss/2.0)*(s.axis && !magnetic && n==2 && p.x==0 ? double(s.axis_coefficients[s.dims[2]+p.z]) : coefficient),other>0.0?pc.dt/other/coefficient:0.0);
        if(!isfinite(result.x)||!isfinite(result.y)||!isfinite(result.z)) atomicOr(&used[ERROR_INDEX],1u);
        if(magnetic==0) v=result; else i=result;
    }
}
__device__ void debye(int n,I3 p,uint pole,D2& result) {
    result=D2(0.0);
    uint cell=uint(n)*CELLS+linear(p);
    D3 c;if(!yee(n,p,c)) return;
    if(s.axis && n==2 && p.x==0) c.y=g(1,0,0);
    const double coefficient=s.axis && n==2 && p.x==0 ? double(s.axis_coefficients[s.dims[2]+p.z]) : double(vi[cell]);
    if(coefficient==0.0) return;
    uint prim=property(c,p,1);
    if (!(s.properties[prim].type&CSProperties::DEBYEMATERIAL)) return;
    double delta=dispersion(prim,pole,n,0,0,c),tau=dispersion(prim,pole,n,0,1,c);
    if(delta<=0.0||tau<=0.0) return;
    if(tau<3.0*pc.dt) { atomicOr(&used[ERROR_INDEX],2); return; }
    double C=8.85418781762e-12*delta*(area(n,p,false)/width(n,p,false));
    result=D2((2.0*tau-pc.dt)/(2.0*tau+pc.dt),C*coefficient/(2.0*tau+pc.dt));
    if(!isfinite(result.x)||!isfinite(result.y)) atomicOr(&used[ERROR_INDEX],1u);
}
__device__ int sheetTangent(uint prim,int n) {
    if(prim==0) return -1;
    int a=(n+1)%3,b=(n+2)%3,result=-1;
    if(s.shapes[prim].bounds[2*a]!=s.shapes[prim].bounds[2*a+1]) result=a;
    if(s.shapes[prim].bounds[2*b]!=s.shapes[prim].bounds[2*b+1]) result=b;
    return result;
}
__device__ void Run(uint id) {
if(pc.mode==19) {
        uint lane=threadIdx.x,selectedCell=0;
        if(id<CELLS) {
            I3 p=position(id); bool disabled=false;
            for(int axis=0;axis<3;++axis) if(uint(p[axis])<=BC_SIZE[2*axis]||uint(p[axis])>=DIMS[axis]-BC_SIZE[2*axis+1]-1) disabled=true;
            for(int n=0;n<3;++n) {
                uint cell=uint(n)*CELLS+id,prim=0;
                D3 c;
                if(yee(n,p,c,false)) {
                    if(s.axis && n==2 && p.x==0) disabled=true;
                    uint selected=property(c,p,1);
                    const GPU_SetupProperty& prop=s.properties[selected];
                    if(prop.sheet) {
                        if(s.shapes[selected].dimension!=2||disabled||float(prop.conductivity)<=0.0||float(prop.thickness)<=0.0) {
                            vv[cell]=0.0; vi[cell]=0.0; atomicAdd(&used[PEC_INDEX+uint(n)],1);
                        } else { prim=selected; selectedCell=1; }
                    }
                }
                material2[cell]=__uint_as_float(prim);
            }
            material0[id]=__uint_as_float(selectedCell);
        }
        winners[lane]=selectedCell; __syncthreads();
        for(uint step=1;step<256;step*=2) {
            uint value=lane>=step?winners[lane-step]:0;
            __syncthreads(); winners[lane]+=value; __syncthreads();
        }
        if(id<CELLS) material1[id]=__uint_as_float(winners[lane]-selectedCell);
        if(lane==255) reduction[id/256]=double(winners[lane]);
        return;
    }
if(pc.mode==20) {
        if(id>=3*CELLS) return;
        uint node=id%CELLS; int n=int(id/CELLS);
        if(__float_as_uint(material0[node])==0) return;
        uint slot=uint(reduction[node/256])+__float_as_uint(material1[node]);
        uint outIndex=uint(n)*pc.reserved+slot,total=3*pc.reserved;
        for(uint kind=0;kind<4;++kind) ext[kind*total+outIndex]=0.0;
        uint prim=__float_as_uint(material2[id]); int tangent=sheetTangent(prim,n);
        if(tangent<0) return;
        I3 p=position(node),previous=p; --previous[tangent];
        float conductivity=float(s.properties[prim].conductivity),thickness=float(s.properties[prim].thickness);
        float G0=conductivity*thickness;
        float w0=float(8.0/double(G0)/double(thickness)/1.256637062e-6);
        float wstop=float(aux[0]*2.0*3.1415926535897932384626433832795),omega=wstop/w0;
        uint table=0; while(table<29&&aux[2+table*6]<=double(omega)) ++table;
        float factor=(sheetTangent(__float_as_uint(material2[uint(tangent)*CELLS+node]),tangent)<0||
            sheetTangent(__float_as_uint(material2[uint(tangent)*CELLS+linear(previous)]),tangent)<0)?2.0:1.0;
        const int metricPosition[3]={p.x,p.y,p.z};
        float wtl=float(width(n,p,false)/Metrics(s).NodeWidth(tangent,metricPosition,false));
        float L1=float(aux[4+table*6]/double(G0)/double(w0)*double(factor));
        float L2=float(aux[5+table*6]/double(G0)/double(w0)*double(factor));
        float R1=float(aux[6+table*6]/double(G0)*double(factor));
        float R2=float(aux[7+table*6]/double(G0)*double(factor));
        float conductance=float(double(G0)*aux[3+table*6]/double(factor));
        L1*=wtl; L2*=wtl; R1*=wtl; R2*=wtl; conductance/=wtl;
        ec1[id]=conductance;
        ec0[id]=float(pc.dt*pc.dt/4.0*(16.0/double(min(L1,L2))+double(1.0f/L1)+double(1.0f/L2)));
        double C=double(ec0[id]),G=double(ec1[id]);
        vv[id]=float((1.0-pc.dt*G/2.0/C)/(1.0+pc.dt*G/2.0/C));
        vi[id]=float((pc.dt/C)/(1.0+pc.dt*G/2.0/C));
        double inductance=double(ec2[id]),resistance=double(ec3[id]);
        ii[id]=inductance>0.0?float((1.0-pc.dt*resistance/2.0/inductance)/(1.0+pc.dt*resistance/2.0/inductance)):0.0;
        iv[id]=inductance>0.0?float((pc.dt/inductance)/(1.0+pc.dt*resistance/2.0/inductance)):0.0;
        ext[outIndex]=float((2.0*double(L1)-pc.dt*double(R1))/(2.0*double(L1)+pc.dt*double(R1)));
        ext[total+outIndex]=float(pc.dt/(double(L1)+pc.dt*double(R1)/2.0)*double(vi[id]));
        ext[2*total+outIndex]=float((2.0*double(L2)-pc.dt*double(R2))/(2.0*double(L2)+pc.dt*double(R2)));
        ext[3*total+outIndex]=float(pc.dt/(double(L2)+pc.dt*double(R2)/2.0)*double(vi[id]));
        return;
    }
if(pc.mode==12) {
        uint lane=threadIdx.x,flags=0;
        if(id<CELLS) {
            I3 p=position(id);
            for(int n=0;n<3;++n) {
                if(aux[0]==0.0) {
                    D3 v,i; lorentz(n,p,uint(aux[2]),v,i);
                    if(v.y!=0.0) flags|=1; if(i.y!=0.0) flags|=2;
                    if(v.z!=0.0) flags|=4; if(i.z!=0.0) flags|=8;
                } else for(uint pole=0;pole<uint(aux[1]);++pole) {
                    D2 coeff; debye(n,p,pole,coeff); if(coeff.y!=0.0) flags|=1;
                }
            }
            material0[id]=__uint_as_float(flags);
            if(aux[0]==0.0) atomicOr(&used[ERROR_INDEX],flags<<4);
        }
        winners[lane]=flags!=0?1:0; __syncthreads();
        for(uint step=1;step<256;step*=2) {
            uint value=lane>=step?winners[lane-step]:0;
            __syncthreads(); winners[lane]+=value; __syncthreads();
        }
        if(id<CELLS) material1[id]=__uint_as_float(winners[lane]-(flags!=0?1:0));
        if(lane==255) reduction[id/256]=double(winners[lane]);
        return;
    }
if(pc.mode==13||pc.mode==14) {
        if(id>=3*CELLS) return;
        uint node=id%CELLS; int n=int(id/CELLS);
        if(__float_as_uint(material0[node])==0) return;
        uint slot=uint(reduction[node/256])+__float_as_uint(material1[node]);
        if(pc.mode==14) { if(n==0) ext[slot]=__uint_as_float(node); return; }
        I3 p=position(node);
        double value=0.0;
        uint kind=uint(aux[3]);
        if(aux[0]==0.0) {
            D3 v,i; lorentz(n,p,uint(aux[2]),v,i);
            value=kind<3?v[kind]:i[kind-3];
        } else if(kind==2) {
            double sum=0.0;
            for(uint pole=0;pole<uint(aux[1]);++pole) { D2 result; debye(n,p,pole,result); sum+=result.y; }
            value=1.0/(1.0+sum);
        } else { D2 result; debye(n,p,uint(aux[2]),result); value=result[kind]; }
        ext[uint(n)*pc.reserved+slot]=float(value);
        return;
    }
if(pc.mode==15) {
        if(id>=4*pc.reserved) return;
        float phase=float(aux[0]*pc.dt);
        I3 metricPoint=I3(0);const int axis=int(aux[2]);metricPoint[axis]=int(aux[3+axis]);
        double distance=fabs(width(axis,metricPoint,false));
        ext[id]=id<2*pc.reserved?float((double(phase)-distance)/(double(phase)+distance)):float(double(phase)/distance);
        return;
    }
if(pc.mode==16) {
        uint count=pc.reserved;
        if(id>=count) return;
        I3 start=I3(aux[0],aux[1],aux[2]),stop=I3(aux[3],aux[4],aux[5]),p=start;
        int n=int(aux[6]),a=(n+1)%3,b=(n+2)%3;
        uint nb=uint(stop[b]-start[b]+1),na=uint(stop[a]-start[a]+1);
        p[n]+=int(id/(na*nb)); p[a]+=int((id/nb)%na); p[b]+=int(id%nb);
        uint node=linear(p),cell=uint(n)*CELLS+node;
        for(uint k=1;k<10;++k) ext[id*10+k]=0.0;
        ext[id*10]=__uint_as_float(cell); ext[id*10+6]=1.0;
        double omega=2.0*3.1415926535897932384626433832795*aux[14];
        const double series=stop[n]-start[n],parallel=double(na)*nb;
        const double L=isnan(aux[8]) ? 0 : aux[8]*parallel/series,R=isnan(aux[9]) ? 0 : aux[9]*parallel/series;
        const double G=isnan(aux[9]) || aux[9]==0 ? 0 : (1/aux[9])*series/parallel,C=isnan(aux[10]) ? 0 : aux[10]*series/parallel;
        double naturalC=double(ec0[cell]);
        if(aux[7]==0.0) {
            double cellC=C>0.0?C:naturalC;
            if(C>0.0) ec0[cell]=float(C);
            else {
                double Z=max(R,omega*L);
                if(1.0/(omega*cellC)<20.0*Z) { cellC=1.0/(omega*Z*20.0); ec0[cell]=float(cellC); }
            }
            ext[id*10+2]=float((pc.dt/cellC)/(1.0+pc.dt*G/(2.0*cellC)));
            ext[id*10+1]=aux[13]!=0.0?float(pc.dt/L):0.0;
            if(aux[12]!=0.0) ec1[cell]=float(G);
        } else {
            ec1[cell]=0.0;
            float cellC=ec0[cell];
            double reactance=omega*L-(C!=0.0?1.0/(C*omega):0.0);
            double Z=sqrt(R*R+reactance*reactance);
            if(1.0/(omega*double(cellC))<20.0*Z) { cellC=float(1.0/(omega*Z*20.0)); ec0[cell]=cellC; }
            const double ib0=C==0 ? pc.dt/(2*L+pc.dt*R) : 2*pc.dt*C/(4*L*C+2*pc.dt*R*C+pc.dt*pc.dt);
            const double b1=C==0 ? -4*L/pc.dt : (pc.dt*pc.dt-4*L*C)/(pc.dt*C);
            const double b2=C==0 ? (2*L-pc.dt*R)/pc.dt : (4*L*C-2*pc.dt*R*C+pc.dt*pc.dt)/(2*pc.dt*C);
            ext[id*10+3]=float(0.5*pc.dt*ib0/double(cellC));
            ext[id*10+4]=float(0.5*pc.dt*(b1*ib0-1.0)/double(cellC));
            ext[id*10+5]=float(0.5*pc.dt*b2*ib0/double(cellC));
            ext[id*10+6]=float(1.0/(1.0+0.5*pc.dt*ib0/double(cellC)));
            ext[id*10+7]=float(ib0); ext[id*10+8]=float(b1); ext[id*10+9]=float(b2);
        }
        double cap=double(ec0[cell]),conductance=double(ec1[cell]);
        vv[cell]=cap>0.0?float((1.0-pc.dt*conductance/2.0/cap)/(1.0+pc.dt*conductance/2.0/cap)):0.0;
        vi[cell]=cap>0.0?float((pc.dt/cap)/(1.0+pc.dt*conductance/2.0/cap)):0.0;
        double inductance=double(ec2[cell]),resistance=double(ec3[cell]);
        ii[cell]=inductance>0.0?float((1.0-pc.dt*resistance/2.0/inductance)/(1.0+pc.dt*resistance/2.0/inductance)):0.0;
        iv[cell]=inductance>0.0?float((pc.dt/inductance)/(1.0+pc.dt*resistance/2.0/inductance)):0.0;
        for(uint k=1;k<10;++k) if(!isfinite(ext[id*10+k])) atomicOr(&used[ERROR_INDEX],1u);
        return;
    }
if(pc.mode==18) {
        uint count=pc.reserved;
        if(id>=4*count) return;
        uint part=id/count,node=id%count;
        bool voltage=part>=2; int component=int(part%2),axis=int(aux[0]),side=int(aux[1]);
        int a=(axis+1)%3,b=(axis+2)%3,incident=component==0?b:a,updated=component==0?a:b;
        I3 start=I3(aux[2],aux[3],aux[4]),stop=I3(aux[5],aux[6],aux[7]),p=start;
        p[a]+=int(node/uint(stop[b]-start[b]+1)); p[b]+=int(node%uint(stop[b]-start[b]+1));
        p[axis]=side==0?start[axis]-(voltage?1:0):stop[axis];
        D3 c;
        for(int n=0;n<3;++n) c[n]=g(n,p[n],(n==incident)!=voltage?1:0);
        double distance=0.0;
        for(int n=0;n<3;++n) distance+=abs((c[n]-aux[8+uint(n)])*aux[11+uint(n)]);
        double delay=distance*aux[21]/aux[20]/pc.dt+(voltage?1.0:0.0)+aux[22]/pc.dt;
        if(!isfinite(delay)||delay<0.0||delay>double(UINT32_MAX-1)) {atomicOr(&used[ERROR_INDEX],1u);return;}
        float amplitude=float(aux[(voltage?17:14)+uint(incident)]*width(incident,p,voltage));
        if(side==0) p[axis]+=voltage?1:-1;
        amplitude*=voltage?vi[uint(updated)*CELLS+linear(p)]:iv[uint(updated)*CELLS+linear(p)];
        if((voltage&&component==1&&side==0)||(voltage&&component==0&&side==1)||
            (!voltage&&component==0&&side==0)||(!voltage&&component==1&&side==1)) amplitude*=-1.0;
        if((p[a]==stop[a]&&component==(voltage?0:1))||(p[b]==stop[b]&&component==(voltage?1:0))) amplitude=0.0;
        ext[id*3]=__uint_as_float(uint(floor(delay))); ext[id*3+1]=float(delay-floor(delay)); ext[id*3+2]=amplitude;
        return;
    }
if(pc.mode==10) {
        uint count=pc.reserved;
        if(id>=2*count) return;
        int axis=int(aux[0]),a=(axis+1)%3,b=(axis+2)%3;
        uint node=id%count, row=node/DIMS[b];
        I3 p=I3(0); p[axis]=int(aux[1]); p[a]=int(row); p[b]=int(node%DIMS[b]);
        I3 metricPoint=I3(0);metricPoint[axis]=p[axis];
        const double distance=fabs(width(axis,metricPoint,false));
        D3 c=D3(g(0,p.x,0),g(1,p.y,0),g(2,p.z,0));
        c[axis]=g(axis,p[axis],0)+(p[axis]==0 ? distance : -distance)/2.0/s.grid_unit;
        uint prim=0;
        double bounds[6];
        for(int n=0;n<3;++n) {
            const int fixed=n==axis ? int(aux[1]) : n==a ? int(row) : -1;
            bounds[2*n]=g(n,fixed<0 ? 0 : max(0,fixed-1),0);
            bounds[2*n+1]=g(n,fixed<0 ? DIMS[n]-1 : min(DIMS[n]-1,fixed+1),0);
        }
        for(uint q=s.candidate_offsets[0];q<s.candidate_offsets[1];++q) {
            uint candidate=s.candidate_order[q];
            if(GPU_SetupCandidate(s.shapes[candidate],bounds) && inside(candidate,c)) {prim=candidate;break;}
        }
        int component=id<count?a:b;
        double epsilon=value(prim,component,0,c),mu=value(prim,component,2,c),phase;
        if(aux[2]>0.0) phase=aux[2]*pc.dt;
        else if(prim!=0) phase=299792458.0*pc.dt/sqrt(epsilon*mu);
        else phase=299792458.0/sqrt(epsilon*mu)*pc.dt;
        ext[id]=float((phase-distance)/(phase+distance));
        return;
    }
if(pc.mode==8) {
        if(id!=0) return;
        I3 start=I3(aux[0],aux[1],aux[2]),stop=I3(aux[3],aux[4],aux[5]);
        int n=int(aux[6]),a=(n+1)%3,b=(n+2)%3;
        double total=0.0;
        I3 p=start;
        for(p[n]=start[n];p[n]<stop[n];++p[n]) {
            double plane=0.0;
            for(p[a]=start[a];p[a]<=stop[a];++p[a]) for(p[b]=start[b];p[b]<=stop[b];++p[b])
                plane+=area(n,p,false)/width(n,p,false);
            total+=1.0/plane;
        }
        reduction[0]=1.0/total;
        return;
    }
if(pc.mode==6) {
        if(id!=0) return;
        uint groups=(3*CELLS+255)/256;
        double sum=0.0;
        for(uint q=0;q<groups;++q) { double count=reduction[q]; reduction[q]=sum; sum+=count; }
        reduction[groups]=sum;
        return;
    }
if(pc.mode==5||pc.mode==7) {
        uint lane=threadIdx.x;
        int n=int(id%3); uint node=id/3;
        I3 p=I3(node%DIMS.x,(node/DIMS.x)%DIMS.y,node/(DIMS.x*DIMS.y));
        bool magnetic=pc.reserved!=0;
        uint cell=uint(n)*CELLS+(id<3*CELLS?linear(p):0);
        if(pc.mode==7) {
            if(id>=3*CELLS) return;
            uint prim=__float_as_uint(material0[id]); if(prim==0) return;
            uint slot=uint(reduction[id/256])+__float_as_uint(material1[id]);
            D3 c;yee(n,p,c,magnetic);
            if(s.axis && !magnetic && n==2 && p.x==0) c.y=g(1,0,0);
            float amplitude=float(value(prim,n,0,c)*width(n,p,magnetic));
            ext[slot*6]=__uint_as_float(cell); ext[slot*6+1]=float(amplitude);
            ext[slot*6+2]=__uint_as_float(uint(s.properties[prim].delay/pc.dt));
            ext[slot*6+3]=0.0; ext[slot*6+4]=0.0; ext[slot*6+5]=0.0;
            return;
        }
        uint prim=0;
        if(id<3*CELLS) {
            D3 c;bool valid=yee(n,p,c,magnetic) && (!magnetic || (p.x<DIMS.x-1&&p.y<DIMS.y-1&&p.z<DIMS.z-1));
            if(s.axis && !magnetic && n==2 && p.x==0) c.y=g(1,0,0);
            if(s.axis && !magnetic && n==1 && p.x==0) valid=false;
            if(valid) {
                uint selected=sourceProperty(c,p);
                const GPU_SetupProperty& prop=s.properties[selected];
                bool type=magnetic?(prop.excitation_type==2||prop.excitation_type==3):(prop.excitation_type==0||prop.excitation_type==1);
                if(selected!=0&&prop.enabled&&prop.active[n]&&type) {
                    if(prop.excitation_type==1) { vv[cell]=0.0; vi[cell]=0.0; }
                    if(prop.excitation_type==3) { ii[cell]=0.0; iv[cell]=0.0; }
                    if(float(value(selected,n,0,c)*width(n,p,magnetic))!=0.0f) prim=selected;
                }
            }
            material0[id]=__uint_as_float(prim);
        }
        winners[lane]=prim!=0?1:0; __syncthreads();
        // Inclusive workgroup scan, with an explicit read-before-write barrier.
        for(uint step=1;step<256;step*=2) {
            uint value=lane>=step?winners[lane-step]:0;
            __syncthreads(); winners[lane]+=value; __syncthreads();
        }
        if(id<3*CELLS) material1[id]=__uint_as_float(winners[lane]-(prim!=0?1:0));
        if(lane==255) reduction[id/256]=double(winners[lane]);
        return;
    }
if(pc.mode==4) {
        uint count=pc.reserved;
        if(id>=3*count) return;
        int n=int(id/count),a=(n+1)%3,b=(n+2)%3;
        I3 dims=I3(aux[3],aux[4],aux[5]);
        I3 loc=I3(id%count/uint(dims.y*dims.z),(id%count/uint(dims.z))%uint(dims.y),id%count%uint(dims.z));
        I3 p=loc+I3(aux[0],aux[1],aux[2]);
        uint cell=uint(n)*CELLS+linear(p);
        D3 kv,ki;
        for(int axis=0;axis<3;++axis) {
            kv[axis]=ki[axis]=0.0;
            const uint lower=uint(aux[12+2*axis]),upper=uint(aux[13+2*axis]);
            double width=0.0,depth=0.0;
            bool high=false;
            uint size=0;
            if(aux[6+2*axis]==3.0 && uint(p[axis])<=lower) {
                size=lower;
                width=(g(axis,size,0)-g(axis,0,0))*aux[18];
                depth=width-(g(axis,p[axis],0)-g(axis,0,0))*aux[18];
            } else if(aux[7+2*axis]==3.0 && uint(p[axis])>=uint(DIMS[axis])-1-upper) {
                size=upper; high=true;
                width=(g(axis,DIMS[axis]-1,0)-g(axis,DIMS[axis]-1-size,0))*aux[18];
                depth=width-(g(axis,DIMS[axis]-1,0)-g(axis,p[axis],0))*aux[18];
            } else continue;
            if(!size || !isfinite(width) || width<=0.0) {atomicOr(&used[ERROR_INDEX],1u);continue;}
            if(s.cylindrical && axis==1) {width*=g(0,p.x,0);depth*=g(0,p.x,0);}
            const double half=0.5*this->width(axis,p,false);
            if(axis==n) depth+=high ? half : -half;
            double variables[5]={depth,width/size,width,376.730313461,double(size)};
            if(depth>0.0) kv[axis]=GPU_SetupDefaultGradingValue(variables[0],variables[1],variables[2],variables[3]);
            if(axis==n) depth+=high ? -half : half;
            if(axis!=n) depth+=high ? half : -half;
            if((high && depth>width) || (!high && depth<0.0)) depth=0.0;
            variables[0]=depth;
            if(depth>0.0) ki[axis]=GPU_SetupDefaultGradingValue(variables[0],variables[1],variables[2],variables[3]);
        }
        D4 m=effective(n,p);
        double E=8.85418781762e-12;
        uint total=3*count;
        for(uint k=0;k<6;++k) ext[k*total+id]=0.0;
        if((kv.x+kv.y+kv.z)!=0.0&&m.y<1e3) {
            if((vv[cell]+vi[cell])!=0.0) {
                vv[cell]=float((2.0*E-kv[a]*pc.dt)/(2.0*E+kv[a]*pc.dt));
                vi[cell]=float((2.0*E*pc.dt)/(2.0*E+kv[a]*pc.dt)*width(n,p,false)/area(n,p,false));
                ext[id]=float((2.0*E-kv[b]*pc.dt)/(2.0*E+kv[b]*pc.dt));
                ext[total+id]=float((2.0*E-kv[n]*pc.dt)/(2.0*E+kv[b]*pc.dt)/m.x);
                ext[2*total+id]=float((2.0*E+kv[n]*pc.dt)/(2.0*E+kv[b]*pc.dt)/m.x);
            }
        } else {
            ext[id]=vv[cell]; vv[cell]=0.0; ext[2*total+id]=1.0;
        }
        if((ki.x+ki.y+ki.z)!=0.0) {
            if((ii[cell]+iv[cell])!=0.0) {
                ii[cell]=float((2.0*E-ki[a]*pc.dt)/(2.0*E+ki[a]*pc.dt));
                iv[cell]=float((2.0*E*pc.dt)/(2.0*E+ki[a]*pc.dt)*width(n,p,true)/area(n,p,true));
                ext[3*total+id]=float((2.0*E-ki[b]*pc.dt)/(2.0*E+ki[b]*pc.dt));
                ext[4*total+id]=float((2.0*E-ki[n]*pc.dt)/(2.0*E+ki[b]*pc.dt)/m.z);
                ext[5*total+id]=float((2.0*E+ki[n]*pc.dt)/(2.0*E+ki[b]*pc.dt)/m.z);
            }
        } else {
            ext[3*total+id]=ii[cell]; ii[cell]=0.0; ext[5*total+id]=1.0;
        }
        return;
    }
if(id>=3*CELLS) return;
int n=int(id/CELLS); I3 p=position(id%CELLS);
if(pc.mode==9||pc.mode==17) {
        I3 start=I3(aux[0],aux[1],aux[2]),stop=I3(aux[3],aux[4],aux[5]);
        int direction=int(aux[6]);
        if(p.x<start.x||p.y<start.y||p.z<start.z||p.x>stop.x||p.y>stop.y||p.z>stop.z) return;
        if(pc.mode==9&&n==direction&&p[n]<stop[n]) {
            if(aux[7]>0.0) ec0[id]=float(aux[9]*area(n,p,false)/width(n,p,false));
            if(aux[8]>0.0) ec1[id]=float(aux[10]*area(n,p,false)/width(n,p,false));
            if(aux[8]==0.0) { vv[id]=0.0; vi[id]=0.0; }
            else {
                double C=double(ec0[id]),G=double(ec1[id]);
                vv[id]=C>0.0?float((1.0-pc.dt*G/2.0/C)/(1.0+pc.dt*G/2.0/C)):0.0;
                vi[id]=C>0.0?float((pc.dt/C)/(1.0+pc.dt*G/2.0/C)):0.0;
            }
        }
        if(aux[11]!=0.0&&n!=direction&&(p[direction]==start[direction]||p[direction]==stop[direction])&&p[n]<stop[n]) {
            vv[id]=0.0; vi[id]=0.0; atomicAdd(&used[PEC_INDEX+uint(n)],1);
        }
    }
}
};

__global__ static void SetupExtensions(SetupDevice s,HIP_SetupExtensionParams params,const double* aux,
    float* output,double* reduction) {
    __shared__ unsigned int scan[256];
    HIP_SetupExtensionKernel kernel(s,params,aux,output,reduction,scan);
    kernel.Run(blockIdx.x*blockDim.x+threadIdx.x);
}

__global__ static void ValidateSetupOutput(const float* values,uint32_t count,uint32_t mode,uint32_t* status)
{
	const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i>=count || mode==14 || (mode==16 && i%10==0) || (mode==18 && i%3==0)) return;
	if (!isfinite(values[i])) atomicOr(status,1u);
}

class HIP_OperatorSetup : public GPU_OperatorSetup
{
	std::shared_ptr<HIP_Context> context;
	GPU_Backend_HIP::Impl* backend;
	SetupDevice s;
	std::vector<void*> allocations;
	double* reduction;
	double* timestep_workspace;
	double* scalar_metadata=nullptr;
	size_t allocation_bytes=0, peak_bytes=0, uploaded_bytes=0, downloaded_bytes=0;
	bool retain_cell_inputs=false;
	size_t allocation_limit=std::numeric_limits<size_t>::max();
	std::vector<double> host_reduction;
	std::vector<unsigned int> host_used;

	std::map<const Operator_Extension*,std::shared_ptr<HIP_SetupUPML>> upml_results;
	std::map<const Operator_Extension*,std::shared_ptr<HIP_SetupSources>> source_results;
	std::map<const Operator_Extension*,std::shared_ptr<HIP_SetupTFSF>> tfsf_results;
	std::map<const Operator_Extension*,std::shared_ptr<HIP_SetupArray>> extension_tables;
	std::map<const Operator_Extension*,std::shared_ptr<HIP_SetupSteadyState>> steady_state_results;
	std::map<const Operator_Extension*,std::shared_ptr<HIP_SetupSparse>> sparse_results;
	std::shared_ptr<HIP_SetupMultiGrid> multigrid_result;
	unsigned int groups;
	template<typename T> T* Allocate(size_t count, const T* host=NULL)
	{
		T* ptr=NULL;
		const size_t bytes=GPU_SetupMultiply(std::max(size_t(1),count),sizeof(T));
		if (bytes>allocation_limit || allocation_bytes>allocation_limit-bytes)
			throw GPU_SetupUnavailable("insufficient device memory budget for construction allocation");
		hipError_t status=HIP_AllocateDevice(context->stream,&ptr,bytes);
		if (status==hipErrorOutOfMemory) { hipGetLastError(); throw GPU_SetupUnavailable("insufficient device memory for operator construction"); }
		HIP_Check(status,"operator setup allocation");
		try { allocations.push_back(ptr); } catch (...) { hipFree(ptr); throw; }
		allocation_bytes=GPU_SetupAdd(allocation_bytes,bytes); peak_bytes=std::max(peak_bytes,allocation_bytes);
		if (host && count) {
			HIP_Check(hipMemcpyAsync(ptr,host,count*sizeof(T),hipMemcpyHostToDevice,context->stream),"operator setup upload");
			uploaded_bytes=GPU_SetupAdd(uploaded_bytes,count*sizeof(T));
		}
		else HIP_Check(hipMemsetAsync(ptr,0,bytes,context->stream),"operator setup clear");
		return ptr;
	}
	void Flush() { backend->Flush();op.CheckSetupCancellation(); }
	void Release(void* buffer, size_t bytes)
	{
		auto allocation=std::find(allocations.begin(),allocations.end(),buffer);
		if (allocation==allocations.end()) throw std::logic_error("unowned construction workspace");
		HIP_Check(hipFree(buffer),"construction workspace release");
		allocations.erase(allocation); allocation_bytes-=bytes;
	}
	template<class T> T* Transfer(T* buffer)
	{
		auto allocation=std::find(allocations.begin(),allocations.end(),buffer);
		if (allocation==allocations.end()) throw std::logic_error("unowned persistent construction buffer");
		allocations.erase(allocation);return buffer;
	}
	void Copy(void* host, const void* device, size_t bytes)
	{
		HIP_Check(hipMemcpyAsync(host,device,bytes,hipMemcpyDeviceToHost,context->stream),"operator construction readback");
		downloaded_bytes=GPU_SetupAdd(downloaded_bytes,bytes);
	}
public:
	std::unique_ptr<GPU_SetupResult> Finish() override
	{
		std::unique_ptr<HIP_SetupResult> result(new HIP_SetupResult(context));

		result->upml=upml_results;
		result->sources=source_results;
		result->tfsf=tfsf_results;
		result->extension_tables=extension_tables;
		result->steady_state=steady_state_results;
		result->sparse=sparse_results;
		result->multigrid=multigrid_result;
		std::copy(s.dims,s.dims+3,result->dims);
		result->cylindrical=s.cylindrical;result->closed_alpha=s.closed_alpha;result->axis=s.axis;
		if (retain_cell_inputs)
		{
			auto query=std::make_shared<HIP_CellQuery>(context);auto& input=query->device;
			input.metrics=Metrics(s);input.shapes=s.shapes;input.properties=s.properties;input.geometry_index=s.geometry_index;
			input.geometry=s.geometry;input.candidates=s.candidates;input.programs=s.expressions;input.instructions=s.expression_code;
			input.constants=s.expression_constants;input.stack_size=s.expression_stack_size;input.discrete=s.discrete_materials;
			input.material_data=s.material_data;input.material_indices=s.material_indices;
			auto retain=[&](const void* buffer) {
				if (!buffer) return;
				auto found=std::find(allocations.begin(),allocations.end(),buffer);
				if (found==allocations.end()) throw std::logic_error("cell material input ownership was lost");
				query->allocations.push_back(*found);allocations.erase(found);
			};
			const void* inputs[]={s.grid,s.child_alpha,s.shapes,s.properties,s.geometry_index,s.geometry,
			                      s.candidates,s.expressions,s.expression_code,s.expression_constants,s.discrete_materials,s.material_data,s.material_indices};
			for (const void* buffer:inputs) retain(buffer);
			result->cell_query=query;
		}
		for (int k=0;k<4;++k)
		{
			if (s.stored_material[k]) result->materials[k]=Transfer(s.stored_material[k]);
			std::vector<void*>::iterator allocation=std::find(allocations.begin(),allocations.end(),s.coeff[k]);
			if (allocation==allocations.end()) throw std::logic_error("HIP construction result already consumed");
			result->coefficients[k]=s.coeff[k]; allocations.erase(allocation); s.coeff[k]=NULL;
		}
		peak_bytes=std::max(peak_bytes,GPU_SetupAdd(allocation_bytes,HIP_PackSetupCoefficients(*result)));
		downloaded_bytes=GPU_SetupAdd(downloaded_bytes,sizeof(unsigned int));
		const size_t pending=GPU_SetupAdd(extension_history_bytes,GPU_SetupMultiply(cells,12*sizeof(float)));
		const size_t future=GPU_SetupAdd(context->setup_future_bytes,pending);
		result->pending_runtime_bytes=pending;context->setup_future_bytes=future;
		return std::unique_ptr<GPU_SetupResult>(result.release());
	}
	bool BuildMultiGrid() override
	{
		Preparation(s.first_radius && s.child_alpha_count==s.dims[1]/2+1,"inconsistent multigrid angular hierarchy");
		const auto result=std::make_shared<HIP_SetupMultiGrid>(context);result->alpha=s.dims[1];result->split=s.first_radius+1;
		SetupMultiGridOutput output={};
		for (uint32_t k=0;k<4;++k)
		{
			output.positions[k]=result->positions[k]=Transfer(Allocate<uint32_t>(2ull*s.dims[1]));
			output.weights[k]=result->weights[k]=Transfer(Allocate<float>(2ull*s.dims[1]));
		}
		SetupMultiGridTables<<<(2ull*s.dims[1]+255)/256,256,0,context->stream>>>(s,output);
		HIP_Check(hipGetLastError(),"multigrid interpolation construction");
		uint32_t status=0;Copy(&status,s.used+primCount+3,sizeof(status));Flush();Require(!(status&1),"invalid multigrid interpolation");
		multigrid_result=result;std::cout << "HIP multigrid interpolation construction active" << std::endl;return true;
	}
	explicit HIP_OperatorSetup(GPU_Backend_HIP::Impl* impl,const GPU_SetupModel& model) : context(impl->ctx), backend(impl), s(), reduction(NULL), timestep_workspace(NULL), groups(0),
        op(*model.op),primitives(model.primitives),cells(model.dims[0]*model.dims[1]*model.dims[2]),primCount(model.primitive_count),
        auxiliary(NULL),auxiliary_capacity(0) {
        static_assert(sizeof(SourcePoint)==24,"source construction layout");
        for(size_t n=0;n<primitives.size();++n) indices[primitives[n]]=static_cast<uint32_t>(n+1);
    }
	~HIP_OperatorSetup() {
		hipStreamSynchronize(context->stream);
		if (backend->diagnostics)
			std::cout << "GPU setup diagnostics (HIP): h2d_bytes=" << uploaded_bytes << " d2h_bytes=" << downloaded_bytes
			          << " setup_peak_bytes=" << peak_bytes << std::endl;
		for (void* ptr:allocations) hipFree(ptr);
	}
	void Prepare(const GPU_SetupModel& model, size_t candidate_budget, size_t construction_limit)
	{
		retain_cell_inputs=model.cell_materials;
		allocation_limit=construction_limit;

		std::copy(model.dims,model.dims+3,s.dims);std::copy(model.offsets,model.offsets+3,s.offsets);
		s.cylindrical=model.cylindrical;s.closed_alpha=model.closed_alpha;s.axis=model.axis;s.first_radius=model.first_radius;s.grid_unit=model.grid_unit;
		std::copy(model.boundary,model.boundary+6,s.boundary);
		std::copy(model.bc_size,model.bc_size+6,s.bc_size);
		s.cells=s.dims[0]*s.dims[1]*s.dims[2]; s.average=model.average; s.stores=model.stores; s.primitives=model.primitive_count;
		groups=static_cast<unsigned int>((3ull*s.cells+255)/256);
		host_reduction.resize(size_t(groups)+1); host_used.resize(size_t(s.primitives)+4);
		s.used=Allocate<unsigned int>(host_used.size());
		s.grid=Allocate(model.grid.size(),model.grid.data()); s.geometry=Allocate(model.geometry.size(),model.geometry.data());
		s.child_alpha_count=static_cast<uint32_t>(model.child_alpha.size()/6);s.child_alpha=Allocate(model.child_alpha.size(),model.child_alpha.data());
		const uint32_t lines=std::max(s.dims[0],std::max(s.dims[1],s.dims[2]));
		SetupGridMetrics<<<dim3((lines+255u)/256,3),256,0,context->stream>>>(s,false);
		if (s.child_alpha_count) SetupGridMetrics<<<(s.child_alpha_count+255u)/256,256,0,context->stream>>>(s,true);
		HIP_Check(hipGetLastError(),"mesh metric construction");
		s.shapes=Allocate(model.shapes.size(),model.shapes.data()); s.properties=Allocate(model.properties.size(),model.properties.data());
		HIP_BuildSetupBounds(const_cast<GPU_SetupPrimitive*>(s.shapes),s.primitives,s.geometry,context->stream);
		std::cout << "HIP primitive bounds construction active" << std::endl;
		GPU_SetupBVHNode* tree=Allocate<GPU_SetupBVHNode>(model.geometry_index_nodes);
		s.geometry_index=tree;
		uint64_t* keys=Allocate<uint64_t>(model.geometry_index_scratch);
		HIP_BuildSetupGeometryIndex(model.shapes,s.shapes,s.geometry,tree,keys,context->stream);
		if (model.geometry_index_nodes) std::cout << "HIP polyhedron spatial index active: " << model.geometry_index_nodes << " nodes" << std::endl;
		const size_t rows=HIP_SetupCandidateRows(s.dims);
		auto* order=Allocate(model.candidate_order.size(),model.candidate_order.data());
		s.candidate_order=order;std::copy(model.candidate_offsets,model.candidate_offsets+4,s.candidate_offsets);
		auto* counts=Allocate<uint64_t>(rows+1);
		auto* scan=Allocate<uint64_t>(HIP_SetupScanWorkspace(rows+1));
		HIP_SetupCandidateInput input={};
		std::copy(s.dims,s.dims+3,input.dims);std::copy(s.offsets,s.offsets+3,input.grid_offsets);
		std::copy(model.candidate_offsets,model.candidate_offsets+4,input.order_offsets);
		input.grid=s.grid;input.shapes=s.shapes;input.order=order;
		HIP_CountSetupCandidates(input,counts,context->stream);
		HIP_SetupExclusiveScan(counts,rows+1,scan,context->stream);
		uint64_t count=0;Copy(&count,counts+rows,sizeof(count));Flush();
		Preparation(count<=UINT32_MAX-255ull-rows,"candidate index range exceeded");
		const size_t bytes=GPU_SetupMultiply(rows+static_cast<size_t>(count),sizeof(uint32_t));
		Preparation(bytes<=candidate_budget,"insufficient device memory budget for candidate lists");
		auto* candidates=Allocate<uint32_t>(rows+static_cast<size_t>(count));s.candidates=candidates;
		HIP_FillSetupCandidates(input,counts,candidates,context->stream);
		Flush();
		Release(counts,(rows+1)*sizeof(uint64_t));Release(scan,HIP_SetupScanWorkspace(rows+1)*sizeof(uint64_t));
		std::cout << "HIP candidate construction active: " << count << " entries" << std::endl;
		s.expressions=Allocate(model.expressions.programs.size(),model.expressions.programs.data());
		s.expression_code=Allocate(model.expressions.instructions.size(),model.expressions.instructions.data());
		s.expression_constants=Allocate(model.expressions.constants.size(),model.expressions.constants.data());
		s.expression_stack_size=model.expressions.stack_size;
		s.expression_stack=Allocate<double>(GPU_SetupMultiply(3ull*s.cells,s.expression_stack_size));
		s.discrete_materials=Allocate(model.discrete_materials.size(),model.discrete_materials.data());
		s.material_data=Allocate(model.material_data.size(),model.material_data.data());
		s.material_indices=Allocate(model.material_indices.size(),model.material_indices.data());
		s.file_weights=Allocate(model.file_weights.size(),model.file_weights.data());
		s.dispersion=Allocate(model.dispersion.size(),model.dispersion.data());
		if (model.expressions.programs.size()>1) std::cout << "HIP constant expression inputs active" << std::endl;
		if (model.discrete_materials.size()>1) std::cout << "HIP discrete material lookup active" << std::endl;
		if (model.file_weights.size()>1) std::cout << "HIP file source interpolation active" << std::endl;
		for (int k=0;k<4;++k) {
			s.ec[k]=Allocate<float>(3ull*s.cells); s.coeff[k]=Allocate<float>(3ull*s.cells);
			s.material[k]=Allocate<float>(3ull*s.cells);
			s.stored_material[k]=(s.stores&(1u<<k)) ? Allocate<float>(3ull*s.cells) : nullptr;
		}
		reduction=Allocate<double>(host_reduction.size());
		timestep_workspace=Allocate<double>((groups+255ull)/256);
		auxiliary=Allocate<double>(auxiliary_capacity);Flush();
		scalar_metadata=Allocate<double>(16);
		s.axis_coefficients=s.axis ? Allocate<float>(2ull*s.dims[2]) : nullptr;
		uint32_t mesh_status=0;Copy(&mesh_status,s.used+primCount+3,sizeof(mesh_status));Flush();Require(!mesh_status,"non-finite mesh metric");
	}
	void CalcEC(float* ec[4][3], float* materials[4]) override
	{
		SetupTimer timer("GPU construction");
		const bool timed=HIP_KernelTimes::Enabled();
		if (timed) HIP_KernelTimes::Begin(context->stream);
		SetupEC<<<groups,256,0,context->stream>>>(s);
		HIP_Check(hipGetLastError(),"material construction kernel");
		if (timed) { HIP_KernelTimes::End(context->stream,"setup_materials"); timer.Report("material kernel"); }
		for (int k=0;k<4;++k) {
			for (int n=0;n<3;++n) if (ec[k][n]) Copy(ec[k][n],s.ec[k]+size_t(n)*s.cells,size_t(s.cells)*sizeof(float));
			if (materials[k]) Copy(materials[k],s.stored_material[k],3ull*s.cells*sizeof(float));
		}
		Copy(host_used.data(),s.used,host_used.size()*sizeof(unsigned int)); Flush();
		timer.Report(timed?"EC/material readback":"material kernel and EC/material readback");
		if (host_used.back()) throw std::runtime_error("HIP construction: effective material is NaN/Inf");
	}
	double CalcTimestep(int method) override
	{
		SetupTimer timer("GPU construction");
		const bool timed=HIP_KernelTimes::Enabled();
		if (timed) HIP_KernelTimes::Begin(context->stream);
		SetupTimestep<<<groups,256,0,context->stream>>>(s,method,reduction);
		HIP_Check(hipGetLastError(),"timestep construction kernel");
		if (timed) { HIP_KernelTimes::End(context->stream,"setup_timestep"); timer.Report("timestep kernel"); }
		unsigned int count=groups;
		double* input=reduction; double* output=timestep_workspace;
		while (count>1)
		{
			const unsigned int blocks=(count+255u)/256;
			ReduceTimestep<<<blocks,256,0,context->stream>>>(input,count,output);
			HIP_Check(hipGetLastError(),"final timestep reduction kernel");
			count=blocks; std::swap(input,output);
		}
		double dt=0; Copy(&dt,input,sizeof(dt)); Flush();
		timer.Report(timed?"device reduction and scalar readback":"timestep kernels and scalar readback");
		if (!(dt>0 && dt<1e200)) throw std::runtime_error("HIP construction: no positive stable timestep");
		return dt;
	}
	void CalcCoefficients(double dt, float* coefficients[4], std::vector<unsigned int>& used) override
	{
		SetupTimer timer("GPU construction");
		const bool timed=HIP_KernelTimes::Enabled();
		if (timed) HIP_KernelTimes::Begin(context->stream);
		SetupCoefficients<<<groups,256,0,context->stream>>>(s,dt);
		HIP_Check(hipGetLastError(),"coefficient/PEC construction kernel");
		SetupCurves<<<static_cast<unsigned int>((s.primitives+255ull)/256),256,0,context->stream>>>(s);
		HIP_Check(hipGetLastError(),"metal curve construction kernel");
		if (timed) { HIP_KernelTimes::End(context->stream,"setup_coefficients_pec"); timer.Report("coefficient/PEC kernel"); }
		for (int k=0;k<4;++k) if (coefficients[k]) Copy(coefficients[k],s.coeff[k],3ull*s.cells*sizeof(float));
		Copy(host_used.data(),s.used,host_used.size()*sizeof(unsigned int)); Flush(); used=host_used;
		if (host_used.back()) throw std::runtime_error("HIP construction: invalid geometry/coefficient result");
		timer.Report(timed?"coefficient/PEC readback":"coefficient/PEC kernel and readback");
	}
	Operator& op;
	std::vector<CSPrimitives*> primitives;
	std::map<CSPrimitives*,uint32_t> indices;

	uint32_t cells,primCount;
	double* auxiliary;
	size_t auxiliary_capacity,extension_history_bytes=0;
	typedef GPU_SetupSourcePoint SourcePoint;
	static void Require(bool success,const char* reason) {
		if (!success) throw std::runtime_error(std::string("HIP construction: ")+reason);
	}
	static void Preparation(bool success,const char* reason) {
		if (!success) throw GPU_SetupUnavailable(reason);
	}
public:
	size_t ExtensionBytes() const {return (auxiliary_capacity+16)*sizeof(double);}
	size_t RuntimeBytes() const {return extension_history_bytes;}
	void ReserveRuntime(size_t bytes)
	{
		Preparation(bytes<=allocation_limit && allocation_bytes<=allocation_limit-bytes,"insufficient device memory budget for measured extension histories");
		allocation_limit-=bytes;extension_history_bytes=GPU_SetupAdd(extension_history_bytes,bytes);
	}
	void Preflight() {
		auxiliary_capacity=182;
		for(size_t n=0;n<op.GetNumberOfExtentions();++n) {
			Operator_Extension* extension=op.GetExtension(n);
			GPU_PrepareExtensionAdapter(*extension);
			if(auto* upml=dynamic_cast<Operator_Ext_UPML*>(extension)) {
				size_t count=1;
				for(int k=0;k<3;++k) {
					Preparation(upml->m_StartPos[k]<=op.numLines[k] && upml->m_numLines[k]<=op.numLines[k]-upml->m_StartPos[k],"invalid UPML range");
					count*=upml->m_numLines[k];
				}
				Preparation(GPU_SetupMultiply(count,18)<=UINT32_MAX,"UPML output index range exceeded");
				extension_history_bytes=GPU_SetupAdd(extension_history_bytes,GPU_SetupMultiply(count,9*sizeof(float)));
			}
		}
		Preparation(auxiliary_capacity<=UINT32_MAX,"extension metadata index range exceeded");
	}
	void Upload(unsigned int index,const void* host,size_t bytes) {
		Require(index==17 && bytes<=auxiliary_capacity*sizeof(double),"extension metadata range exceeded");
		HIP_Check(hipMemcpyAsync(auxiliary,host,bytes,hipMemcpyHostToDevice,context->stream),"extension metadata upload"); Flush();
		uploaded_bytes=GPU_SetupAdd(uploaded_bytes,bytes);
	}
	void SetAuxiliary(const std::vector<double>& values) { Upload(17,values.data(),values.size()*sizeof(double)); }
	void ClearFlags() {
		unsigned int status=0; Copy(&status,s.used+primCount+3,sizeof(status)); Flush();
		if (status&1u) throw std::runtime_error("HIP construction: invalid numerical expression result");
		HIP_Check(hipMemsetAsync(s.used+primCount+3,0,sizeof(unsigned int),context->stream),"extension flags reset");
	}
	void Download(unsigned int index,void* host,size_t bytes,size_t offset=0) {
		const void* device=index==3?static_cast<const void*>(s.used):static_cast<const void*>(reduction);
		const size_t capacity=index==3?host_used.size()*sizeof(unsigned int):host_reduction.size()*sizeof(double);
		Require((index==3||index==16) && offset<=capacity && bytes<=capacity-offset,"extension readback range exceeded");
		Copy(host,static_cast<const char*>(device)+offset,bytes); Flush();
	}
	void Run(uint32_t mode,uint32_t count=0,float* persistent=nullptr) {
		const unsigned int width=mode==4?18:mode==10?2:mode==13?3:mode==14?1:mode==15?4:mode==16?10:(mode==18||mode==20)?12:0;
		if (width && count && !persistent) throw std::logic_error("missing persistent extension output");
		const size_t output_count=GPU_SetupMultiply(count,width);
		Preparation(output_count<=UINT32_MAX-255ull,"extension output index range exceeded");
		const size_t invocations=mode==16?count:mode==10?2ull*count:(mode==15||mode==18)?4ull*count:3ull*(mode==4?count:cells);
		Require(invocations<=UINT32_MAX-255ull,"extension dispatch range exceeded");
		const unsigned int blocks=(mode==6||mode==8)?1:static_cast<unsigned int>((invocations+255)/256);
		if(!blocks) return;
		static const char* names[]={"unused","unused","unused","unused","setup_upml","setup_source_scan","setup_scan_offsets","setup_source_pack",
			"setup_rc_geometry","setup_rc","setup_mur","unused","setup_dispersion_scan","setup_dispersion_coefficients","setup_sparse_positions",
			"setup_local_absorber","setup_rlc","setup_rlc_caps","setup_tfsf","setup_sheet_scan","setup_sheet_coefficients"};
		const bool timed=HIP_KernelTimes::Enabled();
		if(timed) HIP_KernelTimes::Begin(context->stream);
		HIP_SetupExtensionParams params={mode,count,op.dT};
		SetupExtensions<<<blocks,256,0,context->stream>>>(s,params,auxiliary,persistent,reduction);
		HIP_Check(hipGetLastError(),names[mode]);
		if (output_count) ValidateSetupOutput<<<static_cast<uint32_t>((output_count+255)/256),256,0,context->stream>>>(persistent,static_cast<uint32_t>(output_count),mode,s.used+primCount+3);
		HIP_Check(hipGetLastError(),"extension numerical validation");
		if(timed) HIP_KernelTimes::End(context->stream,names[mode]);
	}
	int SnapBox2Mesh(const double* start, const double* stop, unsigned int* first, unsigned int* last, bool dual, bool full, int method) override
	{
		std::vector<double> input(9);
		for (int n=0;n<3;++n)
		{
			if (!std::isfinite(start[n]) || !std::isfinite(stop[n])) throw std::invalid_argument("non-finite snapping coordinate");
			input[n]=start[n];input[3+n]=stop[n];
		}
		input[6]=dual ? 1 : 0;input[7]=full ? 1 : 0;input[8]=method;SetAuxiliary(input);
		GPU_SetupMeshView mesh={{s.dims[0],s.dims[1],s.dims[2]},{s.offsets[0],s.offsets[1],s.offsets[2]},s.grid};
		HIP_BuildSetupMetadata(0,mesh,auxiliary,scalar_metadata,op.dT,context->stream,s.cylindrical,s.closed_alpha);
		double output[7];Copy(output,scalar_metadata,sizeof(output));Flush();
		for (int n=0;n<3;++n) {first[n]=static_cast<unsigned int>(output[1+n]);last[n]=static_cast<unsigned int>(output[4+n]);}
		return static_cast<int>(output[0]);
	}
	double CalcNumericPhaseVelocity(const unsigned int* start, const unsigned int* stop, const double* direction, float frequency) override
	{
		std::vector<double> input(12);
		for (int n=0;n<3;++n) {input[n]=start[n];input[3+n]=stop[n];input[6+n]=direction[n];}
		input[9]=frequency;input[10]=op.gridDelta;input[11]=C0/sqrt(op.GetBackgroundEpsR()*op.GetBackgroundMueR());SetAuxiliary(input);
		GPU_SetupMeshView mesh={{s.dims[0],s.dims[1],s.dims[2]},{s.offsets[0],s.offsets[1],s.offsets[2]},s.grid};
		HIP_BuildSetupMetadata(1,mesh,auxiliary,scalar_metadata,op.dT,context->stream);
		double output=0;Copy(&output,scalar_metadata,sizeof(output));Flush();return output;
	}
	bool BuildExtension(Operator_Extension& extension) override {
		SetupTimer timer("GPU construction");
		bool active=false;
		switch (GPU_PrepareExtensionAdapter(extension)) {
		case GPU_SetupExtensionKind::Cylinder: active=BuildCylinder(dynamic_cast<Operator_Ext_Cylinder&>(extension));break;
		case GPU_SetupExtensionKind::Excitation: active=BuildExcitation(dynamic_cast<Operator_Ext_Excitation&>(extension));break;
		case GPU_SetupExtensionKind::Mur: active=BuildMur(dynamic_cast<Operator_Ext_Mur_ABC&>(extension));break;
		case GPU_SetupExtensionKind::Absorbing: active=BuildAbsorber(dynamic_cast<Operator_Ext_Absorbing_BC&>(extension));break;
		case GPU_SetupExtensionKind::Sheet: active=BuildSheet(dynamic_cast<Operator_Ext_ConductingSheet&>(extension));break;
		case GPU_SetupExtensionKind::Debye: active=BuildDebye(dynamic_cast<Operator_Ext_DebyeMaterial&>(extension));break;
		case GPU_SetupExtensionKind::RLC: active=BuildRLC(dynamic_cast<Operator_Ext_LumpedRLC&>(extension));break;
		case GPU_SetupExtensionKind::Lorentz: active=BuildLorentz(dynamic_cast<Operator_Ext_LorentzMaterial&>(extension));break;
		case GPU_SetupExtensionKind::UPML: active=BuildUPML(dynamic_cast<Operator_Ext_UPML&>(extension));break;
		case GPU_SetupExtensionKind::SteadyState: active=BuildSteadyState(dynamic_cast<Operator_Ext_SteadyState&>(extension));break;
		case GPU_SetupExtensionKind::TFSF: active=dynamic_cast<Operator_Ext_TFSF&>(extension).Operator_Ext_TFSF::BuildExtension();break;
		}
		timer.Report(extension.GetExtensionName().c_str());
		unsigned int numerical_status=0;
		Copy(&numerical_status,s.used+primCount+3,sizeof(numerical_status)); Flush();
		if (numerical_status&1u) throw std::runtime_error("HIP extension construction: invalid numerical expression result");
		const auto* source=dynamic_cast<Operator_Ext_Excitation*>(&extension);
		const auto* load=dynamic_cast<Operator_Ext_LumpedRLC*>(&extension);
		if(active &&
		   (!source || source->GetVoltCount()+source->GetCurrCount()>0) && (!load || load->RLC_count>0))
			std::cout << "HIP extension construction active: " << extension.GetExtensionName() << std::endl;
		return active;
	}
	bool BuildCylinder(Operator_Ext_Cylinder& cylinder)
	{
		const auto result=std::make_shared<HIP_SetupArray>(context);
		if (cylinder.CC_R0_included)
		{
			Require(s.axis && s.axis_coefficients,"missing cylindrical axis construction storage");
			result->count=2ull*s.dims[2];result->values=Transfer(s.axis_coefficients);
			SetupCylinderAxis<<<(s.dims[2]+255u)/256,256,0,context->stream>>>(s,op.dT);
			HIP_Check(hipGetLastError(),"cylindrical axis coefficients");
		}
		extension_tables[&cylinder]=result;const bool diagnostics=backend->diagnostics;const uint32_t count=s.dims[2];
		cylinder.SetHostMaterializer([result,&cylinder,count,diagnostics]() {
			if (!result->count) return;
			cylinder.vv_R0=new float[count];cylinder.vi_R0=new float[count];
			result->Read(cylinder.vv_R0,0,count,diagnostics);result->Read(cylinder.vi_R0,count,count,diagnostics);
		});
		return true;
	}
	bool BuildSteadyState(Operator_Ext_SteadyState& steady)
	{
		SetAuxiliary({steady.m_T_period});
		GPU_SetupMeshView mesh={{s.dims[0],s.dims[1],s.dims[2]},{s.offsets[0],s.offsets[1],s.offsets[2]},s.grid};
		HIP_BuildSetupMetadata(2,mesh,auxiliary,scalar_metadata,op.dT,context->stream);
		double metadata[2];Copy(metadata,scalar_metadata,sizeof(metadata));Flush();
		Require(metadata[0]>=1,"invalid steady-state period");
		const auto result=std::make_shared<HIP_SetupSteadyState>(context);
		result->period=static_cast<uint32_t>(metadata[0]);steady.m_TS_period=result->period;steady.m_T_period=metadata[1];
		Preparation(steady.m_E_probe_dir.size()<=UINT32_MAX-255u,"steady-state probe index range exceeded");
		result->count=static_cast<uint32_t>(steady.m_E_probe_dir.size());
		ReserveRuntime(GPU_SetupAdd(GPU_SetupMultiply(std::max(1u,result->count),2*sizeof(double)),2*sizeof(double)));
		ReserveRuntime(GPU_SetupMultiply(result->count,sizeof(size_t)));
		const size_t records=GPU_SetupMultiply(std::max(1u,result->count),GPU_SetupMultiply(2,result->period));
		Preparation(records<=UINT32_MAX,"steady-state sampling index range exceeded");
		result->records=Transfer(Allocate<float>(records));result->indices=Transfer(Allocate<uint32_t>(result->count));
		std::vector<uint32_t> inputs(GPU_SetupMultiply(result->count,4));
		for (uint32_t q=0;q<result->count;++q)
		{
			inputs[4*q]=steady.m_E_probe_pos[0][q];inputs[4*q+1]=steady.m_E_probe_pos[1][q];inputs[4*q+2]=steady.m_E_probe_pos[2][q];
			inputs[4*q+3]=steady.m_E_probe_dir[q];
		}
		auto* raw=Allocate(inputs.size(),inputs.data());
		if (result->count) SetupSteadyStateIndices<<<(result->count+255u)/256,256,0,context->stream>>>(raw,result->indices,result->count,s.dims[0],s.dims[1],s.dims[2]);
		HIP_Check(hipGetLastError(),"steady-state sampling index construction");Flush();Release(raw,std::max(size_t(1),inputs.size())*sizeof(uint32_t));
		steady_state_results[&steady]=result;steady.SetHostMaterializer([](){});return true;
	}
	bool BuildUPML(Operator_Ext_UPML& upml) {
		std::vector<double> tables(20);
		for(int n=0;n<3;++n) {tables[n]=upml.m_StartPos[n];tables[3+n]=upml.m_numLines[n];}
		for(int n=0;n<6;++n) {tables[6+n]=upml.m_BC[n];tables[12+n]=upml.m_Size[n];}
		tables[18]=op.gridDelta;
		const uint32_t count=upml.m_numLines[0]*upml.m_numLines[1]*upml.m_numLines[2];
		const auto result=std::make_shared<HIP_SetupUPML>(context);
		std::copy(upml.m_numLines,upml.m_numLines+3,result->dims);
		result->values=Transfer(Allocate<float>(GPU_SetupMultiply(count,18)));
		SetAuxiliary(tables);Run(4,count,result->values);Flush();
		const size_t packing=HIP_SetupPackingBytes(count,18);
		Preparation(packing<=allocation_limit && allocation_bytes<=allocation_limit-packing,"insufficient device memory budget for UPML palettes");
		peak_bytes=std::max(peak_bytes,GPU_SetupAdd(allocation_bytes,HIP_PackSetupUPML(*result)));
		if (result->mode) allocation_bytes=GPU_SetupAdd(allocation_bytes,GPU_SetupAdd(GPU_SetupMultiply(result->sets,18*sizeof(float)),GPU_SetupMultiply(count,result->mode==1 ? 2 : 4)));
		downloaded_bytes=GPU_SetupAdd(downloaded_bytes,sizeof(unsigned int));
		upml_results[&upml]=result;
		const bool diagnostics=backend->diagnostics;
		upml.SetHostMaterializer([result,&upml,diagnostics]() {
			ArrayLib::ArrayNIJK<float>* arrays[6]={&upml.vv,&upml.vvfo,&upml.vvfn,&upml.ii,&upml.iifo,&upml.iifn};
			float* outputs[6];
			for (int n=0;n<6;++n) {arrays[n]->Init("requested HIP UPML",upml.m_numLines);outputs[n]=arrays[n]->data();}
			result->Materialize(outputs,diagnostics);
		});
		return true;
	}
	void SynchronizeOperator() override {
		SetupTimer timer("GPU construction");
		float* coefficients[4]={op.vv_ptr->data(),op.vi_ptr->data(),op.ii_ptr->data(),op.iv_ptr->data()};
		for(int k=0;k<4;++k) if (coefficients[k]) Copy(coefficients[k],s.coeff[k],3ull*cells*sizeof(float));
		for(int n=0;n<3;++n) {
			if (op.EC_C[n]) Copy(op.EC_C[n],s.ec[0]+size_t(n)*cells,size_t(cells)*sizeof(float));
			if (op.EC_G[n]) Copy(op.EC_G[n],s.ec[1]+size_t(n)*cells,size_t(cells)*sizeof(float));
		}
		Copy(host_used.data(),s.used,host_used.size()*sizeof(unsigned int)); Flush();
		for(size_t n=0;n<primitives.size();++n) if(host_used[n+1]) primitives[n]->SetPrimitiveUsed(true);
		std::copy(host_used.begin()+primCount,host_used.begin()+primCount+3,op.m_Nr_PEC);
		timer.Report("construction status and requested host mirrors");
	}
	void ReadCoefficients(unsigned int kind,size_t offset,size_t count,float* values) override
	{
		const size_t size=3ull*cells;
		if (kind>=4 || offset>size || count>size-offset) throw std::out_of_range("coefficient materialization range");
		Copy(values,s.coeff[kind]+offset,count*sizeof(float));Flush();
		if (backend->diagnostics) std::cout << "GPU requested host coefficients: kind=" << kind << " d2h_bytes=" << count*sizeof(float) << std::endl;
	}
	void WriteCoefficient(unsigned int kind,size_t index,float value) override
	{
		if (kind>=4 || index>=3ull*cells) throw std::out_of_range("coefficient write index");
		HIP_Check(hipMemcpyAsync(s.coeff[kind]+index,&value,sizeof(value),hipMemcpyHostToDevice,context->stream),"requested coefficient write");
		uploaded_bytes=GPU_SetupAdd(uploaded_bytes,sizeof(value));Flush();
	}
	void ReadMaterials(unsigned int kind,size_t offset,size_t count,float* values) override
	{
		const size_t size=3ull*cells;
		if (kind>=4 || !s.stored_material[kind] || offset>size || count>size-offset) throw std::out_of_range("material storage readback range");
		Copy(values,s.stored_material[kind]+offset,count*sizeof(float));Flush();
		if (backend->diagnostics) std::cout << "GPU requested host materials: kind=" << kind << " d2h_bytes=" << count*sizeof(float) << std::endl;
	}

	void CompactDeviceNodes(std::vector<double>& metadata,HIP_SetupSparse::Block& block)
	{
		SetAuxiliary(metadata);Run(12);Run(6);
		double count=0;Download(16,&count,sizeof(count),((3ull*cells+255)/256)*sizeof(double));
		Require(count>=0 && count<=cells,"invalid sparse construction count");block.count=static_cast<uint32_t>(count);
		block.nodes=Transfer(Allocate<uint32_t>(block.count));Run(14,block.count,reinterpret_cast<float*>(block.nodes));
	}
	std::shared_ptr<HIP_SetupArray> SparseCoefficients(std::vector<double>& metadata,uint32_t count,uint32_t poles=1)
	{
		const auto output=std::make_shared<HIP_SetupArray>(context);output->count=GPU_SetupMultiply(GPU_SetupMultiply(count,3),poles);
		Preparation(output->count<=UINT32_MAX,"sparse coefficient index range exceeded");output->values=Transfer(Allocate<float>(output->count));
		for (uint32_t pole=0;pole<poles;++pole)
		{
			if (poles>1) metadata[2]=pole;
			SetAuxiliary(metadata);Run(13,count,output->values+3ull*pole*count);
		}
		return output;
	}
	static void MaterializeSparsePositions(Operator_Ext_Dispersive& material,unsigned int order,const std::vector<uint32_t>& nodes,uint32_t ny,uint32_t nz)
	{
		material.m_LM_pos[order]=new unsigned int*[3]();
		for (unsigned int n=0;n<3;++n)
		{
			material.m_LM_pos[order][n]=new unsigned int[nodes.size()];
			for (size_t q=0;q<nodes.size();++q) material.m_LM_pos[order][n][q]=n==0 ? nodes[q]/(ny*nz) : n==1 ? (nodes[q]/nz)%ny : nodes[q]%nz;
		}
	}
	static void MaterializeSparseCoefficients(const HIP_SetupArray& source,size_t offset,uint32_t count,float**& destination,bool diagnostics)
	{
		std::vector<float> values(3ull*count);source.Read(values.data(),offset,values.size(),diagnostics);
		destination=new float*[3]();
		for (unsigned int n=0;n<3;++n) {destination[n]=new float[count];std::copy(values.begin()+size_t(n)*count,values.begin()+size_t(n+1)*count,destination[n]);}
	}
	bool BuildLorentz(Operator_Ext_LorentzMaterial& material) {
		Require(material.m_LM_pos==nullptr,"rebuilding a constructed Lorentz extension requires a fresh operator");
		int order=0;
		for (CSProperties* property:op.CSX->GetPropertyByType(CSProperties::LORENTZMATERIAL))
			order=std::max(order,static_cast<CSPropLorentzMaterial*>(property)->GetDispersionOrder());
		if (!order) return false;
		std::vector<double> metadata(4,0.0); metadata[1]=order;
		material.m_Order=order; material.m_LM_Count.assign(order,0); material.m_LM_pos=new unsigned int**[order]();
		material.m_volt_ADE_On=new bool[order](); material.m_curr_ADE_On=new bool[order]();
		material.m_volt_Lor_ADE_On=new bool[order](); material.m_curr_Lor_ADE_On=new bool[order]();
		float**** outputs[6]={&material.v_int_ADE,&material.v_ext_ADE,&material.v_Lor_ADE,&material.i_int_ADE,&material.i_ext_ADE,&material.i_Lor_ADE};
		for (auto output:outputs) *output=new float**[order]();
		const auto result=std::make_shared<HIP_SetupSparse>(context);result->blocks.resize(order);
		for (int pole=0;pole<order;++pole) {
			metadata[2]=pole;
			uint32_t flags=0; ClearFlags();
			auto& block=result->blocks[pole];CompactDeviceNodes(metadata,block);material.m_LM_Count[pole]=block.count;
			Download(3,&flags,sizeof(flags),size_t(primCount+3)*sizeof(uint32_t));
			if (flags&1u) throw std::runtime_error("HIP Lorentz construction: invalid numerical expression result");
			flags>>=4;
			block.flags=flags;block.coefficients.resize(6);
			material.m_volt_ADE_On[pole]=(flags&1)!=0; material.m_curr_ADE_On[pole]=(flags&2)!=0;
			material.m_volt_Lor_ADE_On[pole]=(flags&4)!=0; material.m_curr_Lor_ADE_On[pole]=(flags&8)!=0;
			const bool enabled[6]={(flags&1)!=0,(flags&1)!=0,(flags&4)!=0,(flags&2)!=0,(flags&2)!=0,(flags&8)!=0};
			const size_t states=3ull*block.count*((flags&1 ? 1 : 0)+(flags&2 ? 1 : 0)+(flags&4 ? 1 : 0)+(flags&8 ? 1 : 0));
			ReserveRuntime(GPU_SetupMultiply(states,sizeof(float)));
			ReserveRuntime(GPU_SetupMultiply(GPU_SetupMultiply(block.count,((flags&1) ? 1u : 0u)+((flags&2) ? 1u : 0u)),sizeof(size_t)));
			for (unsigned int kind=0;kind<6;++kind) if (enabled[kind]) {
				metadata[3]=kind;block.coefficients[kind]=SparseCoefficients(metadata,block.count);
			}
		}
		sparse_results[&material]=result;
		const bool diagnostics=backend->diagnostics;const uint32_t ny=s.dims[1],nz=s.dims[2];
		material.SetHostMaterializer([result,&material,ny,nz,diagnostics]() {
			float**** targets[6]={&material.v_int_ADE,&material.v_ext_ADE,&material.v_Lor_ADE,&material.i_int_ADE,&material.i_ext_ADE,&material.i_Lor_ADE};
			for (unsigned int pole=0;pole<result->blocks.size();++pole)
			{
				std::vector<uint32_t> nodes;result->ReadNodes(pole,nodes,diagnostics);MaterializeSparsePositions(material,pole,nodes,ny,nz);
				const auto& block=result->blocks[pole];
				for (unsigned int kind=0;kind<6;++kind) if (block.coefficients[kind]) MaterializeSparseCoefficients(*block.coefficients[kind],0,block.count,(*targets[kind])[pole],diagnostics);
			}
		});
		return true;
	}
	bool BuildDebye(Operator_Ext_DebyeMaterial& material) {
		Require(material.m_LM_pos==nullptr,"rebuilding a constructed Debye extension requires a fresh operator");
		int poles=0;
		for (CSProperties* property:op.CSX->GetPropertyByType(CSProperties::DEBYEMATERIAL))
			poles=std::max(poles,static_cast<CSPropDebyeMaterial*>(property)->GetDispersionOrder());
		if (!poles) return false;
		std::vector<double> metadata(4,0.0); metadata[0]=1; metadata[1]=poles;
		const auto result=std::make_shared<HIP_SetupSparse>(context);result->blocks.resize(1);
		auto& block=result->blocks[0];CompactDeviceNodes(metadata,block);if (!block.count) return false;
		ReserveRuntime(GPU_SetupMultiply(GPU_SetupMultiply(3ull*block.count,size_t(poles)+2),sizeof(float)));
		ReserveRuntime(GPU_SetupMultiply(block.count,sizeof(size_t)));
		material.m_PoleCount=poles;material.m_Order=1;material.m_LM_Count.assign(1,block.count);
		material.m_LM_pos=new unsigned int**[1]();
		material.m_volt_ADE_On=new bool[1]{true}; material.m_curr_ADE_On=new bool[1]{false};
		material.v_relax_ADE=new float**[poles](); material.v_drive_ADE=new float**[poles]();
		block.coefficients.resize(3);metadata[3]=2;block.coefficients[0]=SparseCoefficients(metadata,block.count);
		metadata[3]=0;block.coefficients[1]=SparseCoefficients(metadata,block.count,poles);
		metadata[3]=1;block.coefficients[2]=SparseCoefficients(metadata,block.count,poles);
		sparse_results[&material]=result;
		const bool diagnostics=backend->diagnostics;const uint32_t ny=s.dims[1],nz=s.dims[2];
		material.SetHostMaterializer([result,&material,ny,nz,poles,diagnostics]() {
			std::vector<uint32_t> nodes;result->ReadNodes(0,nodes,diagnostics);MaterializeSparsePositions(material,0,nodes,ny,nz);
			const auto& block=result->blocks[0];MaterializeSparseCoefficients(*block.coefficients[0],0,block.count,material.v_solve_ADE,diagnostics);
			for (int pole=0;pole<poles;++pole)
			{
				MaterializeSparseCoefficients(*block.coefficients[1],3ull*pole*block.count,block.count,material.v_relax_ADE[pole],diagnostics);
				MaterializeSparseCoefficients(*block.coefficients[2],3ull*pole*block.count,block.count,material.v_drive_ADE[pole],diagnostics);
			}
		});
		uint32_t flags=0; Download(3,&flags,sizeof(flags),size_t(primCount+3)*sizeof(uint32_t));
		if (flags&2) std::cerr << "GPU Debye setup: relaxation time needs at least 3 timesteps; skipping this pole." << std::endl;
		return true;
	}
	bool BuildAbsorber(Operator_Ext_Absorbing_BC& abc) {
		if (abc.m_PendingPrimitive)
		{
			auto* primitive=abc.m_PendingPrimitive;auto* property=abc.m_PendingProperty;
			abc.m_PendingPrimitive=nullptr;abc.m_PendingProperty=nullptr;
			if (!abc.SetInitParams(primitive,property)) return false;
		}
		Require(abc.m_ny>=0 && abc.m_ny<3 && abc.m_phaseVelocity>0,"invalid local absorber metadata");
		abc.m_nyP=(abc.m_ny+1)%3; abc.m_nyPP=(abc.m_ny+2)%3;
		abc.m_numLines[0]=abc.m_sheetX1[abc.m_nyP]-abc.m_sheetX0[abc.m_nyP]+1;
		abc.m_numLines[1]=abc.m_sheetX1[abc.m_nyPP]-abc.m_sheetX0[abc.m_nyPP]+1;
		abc.m_numCells=abc.m_numLines[0]*abc.m_numLines[1];
		ReserveRuntime(GPU_SetupMultiply(abc.m_numCells,4*sizeof(float)));
		const auto result=std::make_shared<HIP_SetupArray>(context);
		result->count=GPU_SetupMultiply(abc.m_numCells,4);result->values=Transfer(Allocate<float>(result->count));
		SetAuxiliary({abc.m_phaseVelocity,0,double(abc.m_ny),
			double(abc.m_sheetX0[0]),double(abc.m_sheetX0[1]),double(abc.m_sheetX0[2]),
			double(abc.m_numLines[0]),double(abc.m_numLines[1]),abc.m_normalSignPositive?1.0:0.0});Run(15,abc.m_numCells,result->values);
		extension_tables[&abc]=result;
		const bool diagnostics=backend->diagnostics;
		abc.SetHostMaterializer([result,&abc,diagnostics]() {
		ArrayLib::ArrayIJ<float>* outputs[4]={&abc.m_K1_nyP,&abc.m_K1_nyPP,&abc.m_K2_nyP,&abc.m_K2_nyPP};
		const unsigned int arrays=abc.m_ABCtype==Operator_Ext_Absorbing_BC::MUR_1ST_SA?4:2;
		for (unsigned int kind=0;kind<arrays;++kind) {
			outputs[kind]->Init("GPU local absorber",abc.m_numLines);
			result->Read(outputs[kind]->data(),size_t(kind)*abc.m_numCells,abc.m_numCells,diagnostics);
		}
		});
		return true;
	}
	bool BuildSheet(Operator_Ext_ConductingSheet& sheet) {
		// These fixed fitting tables are also used by the reference constructor.
#include "FDTD/extensions/cond_sheet_parameter.h"
		Require(sheet.m_LM_pos==nullptr,"rebuilding a constructed sheet extension requires a fresh operator");
		std::vector<double> metadata(2+numOptPara*6); metadata[0]=sheet.m_f_max; metadata[1]=numOptPara;
		for (unsigned int q=0;q<numOptPara;++q) {
			metadata[2+q*6]=omega_stop[q]; metadata[3+q*6]=g[q]; metadata[4+q*6]=l1[q];
			metadata[5+q*6]=l2[q]; metadata[6+q*6]=r1[q]; metadata[7+q*6]=r2[q];
		}
		 SetAuxiliary(metadata);
		Run(19); Run(6);
		double total=0; Download(16,&total,sizeof(total),((3ull*cells+255)/256)*sizeof(double));
		Require(total>=0 && total<=cells,"invalid conducting sheet cell count");
		const uint32_t count=uint32_t(total);if (!count) return false;
		ReserveRuntime(GPU_SetupMultiply(count,6*sizeof(float)));
		const auto result=std::make_shared<HIP_SetupSparse>(context);result->blocks.resize(2);
		for (auto& block:result->blocks)
		{
			block.count=count;block.flags=1;block.coefficients.resize(6);
			block.nodes=Transfer(Allocate<uint32_t>(count));Run(14,count,reinterpret_cast<float*>(block.nodes));
		}
		float* values=Allocate<float>(GPU_SetupMultiply(count,12));Run(20,count,values);
		sheet.m_Order=2;sheet.m_LM_Count.assign(2,count);sheet.m_LM_pos=new unsigned int**[2]();
		sheet.m_volt_ADE_On=new bool[2]{true,true}; sheet.m_curr_ADE_On=new bool[2]{false,false};
		sheet.m_volt_Lor_ADE_On=new bool[2]{false,false}; sheet.m_curr_Lor_ADE_On=new bool[2]{false,false};
		sheet.v_int_ADE=new float**[2](); sheet.v_ext_ADE=new float**[2]();
		for (unsigned int order=0;order<2;++order) for (unsigned int kind=0;kind<2;++kind)
		{
			const auto output=std::make_shared<HIP_SetupArray>(context);output->count=GPU_SetupMultiply(count,3);
			output->values=Transfer(Allocate<float>(output->count));
			HIP_Check(hipMemcpyAsync(output->values,values+(order*6ull+kind*3)*count,output->count*sizeof(float),hipMemcpyDeviceToDevice,context->stream),"sheet coefficient handoff");
			result->blocks[order].coefficients[kind]=output;
		}
		Flush();Release(values,12ull*count*sizeof(float));sparse_results[&sheet]=result;
		const bool diagnostics=backend->diagnostics;const uint32_t ny=s.dims[1],nz=s.dims[2];
		sheet.SetHostMaterializer([result,&sheet,ny,nz,diagnostics]() {
			for (unsigned int order=0;order<2;++order)
			{
				std::vector<uint32_t> nodes;result->ReadNodes(order,nodes,diagnostics);MaterializeSparsePositions(sheet,order,nodes,ny,nz);
				const auto& block=result->blocks[order];
				MaterializeSparseCoefficients(*block.coefficients[0],0,block.count,sheet.v_int_ADE[order],diagnostics);
				MaterializeSparseCoefficients(*block.coefficients[1],0,block.count,sheet.v_ext_ADE[order],diagnostics);
			}
		});
		return true;
	}
	bool BuildRLC(Operator_Ext_LumpedRLC& rlc) {
		Require(rlc.RLC_count==0,"rebuilding a constructed RLC extension requires a fresh operator");
		struct Region {float* values;uint32_t count;};
		std::vector<Region> regions;size_t total=0;
		for (CSProperties* property:op.CSX->GetPropertyByType(CSProperties::LUMPED_ELEMENT)) {
			auto* load=static_cast<CSPropLumpedElement*>(property);
			if (!rlc.IsLElumpedRLC(load)) continue;
			const int n=load->GetDirection(),a=(n+1)%3,b=(n+2)%3;
			if (n<0||n>2) { std::cerr << "GPU RLC setup: invalid direction; skipping load." << std::endl; continue; }
			double R=load->GetResistance(),L=load->GetInductance(),C=load->GetCapacity();
			for (double* value:{&R,&L,&C}) if (*value<0) { *value=0; std::cerr << "GPU RLC setup: negative component value clamped to zero." << std::endl; }
			for (CSPrimitives* primitive:load->GetAllPrimitives()) {
				auto* box=dynamic_cast<CSPrimBox*>(primitive); if (!box) continue;
				unsigned int start[3],stop[3];
				if (op.SnapBox2Mesh(box->GetStartCoord()->GetCoords(op.m_MeshType),box->GetStopCoord()->GetCoords(op.m_MeshType),start,stop,false,true)<=0 || start[n]==stop[n]) continue;
				const size_t count=GPU_SetupMultiply(stop[n]-start[n],GPU_SetupMultiply(stop[a]-start[a]+1,stop[b]-start[b]+1));
				Preparation(count<=UINT32_MAX/10 && total<=UINT32_MAX/9-count,"RLC construction index range exceeded");total+=count;
				std::vector<double> metadata(15);
				for (int axis=0;axis<3;++axis) { metadata[axis]=start[axis]; metadata[3+axis]=stop[axis]; }
				metadata[6]=n; metadata[7]=load->GetLEtype()==CSPropLumpedElement::SERIES?1:0;
				metadata[8]=L;metadata[9]=R;metadata[10]=C;metadata[12]=R>=0?1:0;
				metadata[13]=L>0?1:0; metadata[14]=op.GetExcitationSignal()->GetCenterFreq()+op.GetExcitationSignal()->GetCutOffFreq();
				float* values=Allocate<float>(GPU_SetupMultiply(count,10));regions.push_back({values,static_cast<uint32_t>(count)});
				SetAuxiliary(metadata);Run(16,static_cast<uint32_t>(count),values);

				if (load->GetCaps()) {
					std::vector<double> caps(12,0.0);
					for (int axis=0;axis<3;++axis) { caps[axis]=start[axis]; caps[3+axis]=stop[axis]; }
					caps[6]=n; caps[11]=1; SetAuxiliary(caps); Run(17);
				}
				primitive->SetPrimitiveUsed(true);
			}
		}
		rlc.RLC_count=static_cast<uint32_t>(total);if (!total) return true;
		ReserveRuntime(GPU_SetupMultiply(total,7*sizeof(float)));
		ReserveRuntime(GPU_SetupMultiply(total,sizeof(size_t)));
		const auto result=std::make_shared<HIP_SetupSparse>(context);result->blocks.resize(1);
		auto& block=result->blocks[0];block.count=rlc.RLC_count;block.nodes=Transfer(Allocate<uint32_t>(total));
		const auto values=std::make_shared<HIP_SetupArray>(context);values->count=GPU_SetupMultiply(total,9);values->values=Transfer(Allocate<float>(values->count));
		block.coefficients.push_back(values);
		uint32_t offset=0;
		for (const auto& region:regions)
		{
			SetupRLCFinalize<<<(region.count+255u)/256,256,0,context->stream>>>(region.values,region.count,offset,block.count,block.nodes,values->values);
			HIP_Check(hipGetLastError(),"RLC device table handoff");offset+=region.count;
		}
		Flush();for (const auto& region:regions) Release(region.values,10ull*region.count*sizeof(float));
		uint32_t capacity=32;while (capacity<2ull*total) capacity*=2;
		auto* slots=Allocate<uint32_t>(capacity);auto* duplicate=Allocate<uint32_t>(1);
		HIP_Check(hipMemsetAsync(slots,255,size_t(capacity)*sizeof(uint32_t),context->stream),"RLC edge index clear");
		SetupRLCDuplicates<<<(block.count+255u)/256,256,0,context->stream>>>(block.nodes,block.count,slots,capacity,duplicate);
		HIP_Check(hipGetLastError(),"RLC overlap detection");Copy(&block.flags,duplicate,sizeof(block.flags));Flush();
		Require(!(block.flags&2),"RLC overlap hash exhausted");Release(slots,size_t(capacity)*sizeof(uint32_t));Release(duplicate,sizeof(uint32_t));
		sparse_results[&rlc]=result;
		const bool diagnostics=backend->diagnostics;const uint32_t ny=s.dims[1],nz=s.dims[2],numcells=cells;
		rlc.SetHostMaterializer([result,&rlc,ny,nz,numcells,diagnostics]() {
			std::vector<uint32_t> indices;result->ReadNodes(0,indices,diagnostics);
			const auto& block=result->blocks[0];const uint32_t count=block.count;
			rlc.v_RLC_dir=new int[count];rlc.v_RLC_pos=new unsigned int*[3]();
			for (int n=0;n<3;++n) rlc.v_RLC_pos[n]=new unsigned int[count];
			for (uint32_t q=0;q<count;++q)
			{
				rlc.v_RLC_dir[q]=indices[q]/numcells;const uint32_t node=indices[q]%numcells;
				rlc.v_RLC_pos[0][q]=node/(ny*nz);rlc.v_RLC_pos[1][q]=(node/nz)%ny;rlc.v_RLC_pos[2][q]=node%nz;
			}
			float** targets[9]={&rlc.v_RLC_ilv,&rlc.v_RLC_i2v,&rlc.v_RLC_vv2,&rlc.v_RLC_vj1,&rlc.v_RLC_vj2,&rlc.v_RLC_vvd,&rlc.v_RLC_ib0,&rlc.v_RLC_b1,&rlc.v_RLC_b2};
			for (int kind=0;kind<9;++kind) {*targets[kind]=new float[count];block.coefficients[0]->Read(*targets[kind],size_t(kind)*count,count,diagnostics);}
		});
		return true;
	}
	bool BuildTFSF(Operator_Ext_TFSF& source, const double origin[3]) override {
		double device_origin[3];
		if (!origin)
		{
			std::vector<double> input(16);
			for (int n=0;n<3;++n) {input[n]=source.m_Start[n];input[3+n]=source.m_Stop[n];input[6+n]=source.m_PropDir[n];input[9+n]=source.m_E_Amp[n];}
			input[12]=source.m_Frequency;input[13]=op.GetBackgroundEpsR();input[14]=op.GetBackgroundMueR();input[15]=op.gridDelta;SetAuxiliary(input);
			GPU_SetupMeshView mesh={{s.dims[0],s.dims[1],s.dims[2]},{s.offsets[0],s.offsets[1],s.offsets[2]},s.grid};
			HIP_BuildSetupMetadata(3,mesh,auxiliary,scalar_metadata,op.dT,context->stream);
			double output[16];Copy(output,scalar_metadata,sizeof(output));Flush();
			if (output[14])
			{
				std::cerr << "Operator_Ext_TFSF::BuildExtension: Warning, " << (output[14]==1 ? "plane wave direction is zero" : "plane wave direction and polarization is identical") << ", skipping..." << std::endl;
				source.SetActive(false);return false;
			}
			if (output[15]) std::cerr << "Operator_Ext_TFSF::BuildExtension: Warning, invalid phase velocity found, resetting to c0! " << std::endl;
			if (output[13]!=90) std::cerr << "Operator_Ext_TFSF::BuildExtension: Warning, resetting non-transverse E-polarization" << std::endl;
			for (int n=0;n<3;++n)
			{
				source.m_PropDir[n]=output[n];source.m_E_Amp[n]=output[3+n];source.m_H_Amp[n]=output[6+n];device_origin[n]=output[9+n];
				source.m_numLines[n]=source.m_Stop[n]-source.m_Start[n]+1;source.m_IncLow[n]=source.m_PropDir[n]>=0;
				source.m_ActiveDir[n][0]=source.m_Start[n]!=0;source.m_ActiveDir[n][1]=source.m_Stop[n]!=op.GetNumberOfLines(n,true)-1;
			}
			source.m_PhVel=output[12];origin=device_origin;std::cout << "HIP TFSF numerical metadata active" << std::endl;
		}
		source.m_maxDelay=0;
		const auto result=std::make_shared<HIP_SetupTFSF>(context);
		auto* maximum=Allocate<uint32_t>(1);
		for (int axis=0;axis<3;++axis) for (int side=0;side<2;++side) {
			if (!source.m_ActiveDir[axis][side]) continue;
			const int a=(axis+1)%3,b=(axis+2)%3;
			const uint32_t count=source.m_numLines[a]*source.m_numLines[b];
			std::vector<double> metadata(23); metadata[0]=axis; metadata[1]=side;
			for (int n=0;n<3;++n) {
				metadata[2+n]=source.m_Start[n]; metadata[5+n]=source.m_Stop[n]; metadata[8+n]=origin[n];
				metadata[11+n]=source.m_PropDir[n]; metadata[14+n]=source.m_E_Amp[n]; metadata[17+n]=source.m_H_Amp[n];
			}
			metadata[20]=source.m_PhVel; metadata[21]=op.GetGridDelta(); metadata[22]=source.m_SourceDelay;
			result->counts[axis][side]=count;
			result->planes[axis][side]=Transfer(Allocate<float>(GPU_SetupMultiply(count,12)));
			SetAuxiliary(metadata);Run(18,count,result->planes[axis][side]);
			SetupTFSFDelay<<<(4ull*count+255)/256,256,0,context->stream>>>(result->planes[axis][side],4*count,maximum);
			HIP_Check(hipGetLastError(),"TFSF delay reduction");
		}
		Copy(&source.m_maxDelay,maximum,sizeof(source.m_maxDelay));Flush();Release(maximum,sizeof(uint32_t));
		++source.m_maxDelay;tfsf_results[&source]=result;
		const bool diagnostics=backend->diagnostics;
		source.SetHostMaterializer([result,&source,diagnostics]() {
			for (unsigned int axis=0;axis<3;++axis) for (unsigned int side=0;side<2;++side)
			{
				const uint32_t count=result->counts[axis][side];if (!count) continue;
				std::vector<float> points;result->Materialize(axis,side,points,diagnostics);
				for (unsigned int part=0;part<4;++part)
				{
					const unsigned int component=part%2;
					unsigned int*& delays=part<2?source.m_CurrDelay[axis][side][component]:source.m_VoltDelay[axis][side][component];
					float*& deltas=part<2?source.m_CurrDelayDelta[axis][side][component]:source.m_VoltDelayDelta[axis][side][component];
					float*& amplitudes=part<2?source.m_CurrAmp[axis][side][component]:source.m_VoltAmp[axis][side][component];
					delays=new unsigned int[count];deltas=new float[count];amplitudes=new float[count];
					for (uint32_t q=0;q<count;++q)
					{
						const float* point=&points[(part*count+q)*3];std::memcpy(&delays[q],point,sizeof(uint32_t));
						deltas[q]=point[1];amplitudes[q]=point[2];
					}
				}
			}
		});
		return true;
	}
	bool BuildMur(Operator_Ext_Mur_ABC& mur) {
		if (mur.m_ny<0) return false;
		std::vector<double> metadata(3);
		metadata[0]=mur.m_ny; metadata[1]=mur.m_LineNr; metadata[2]=mur.m_v_phase;
		Upload(17,metadata.data(),metadata.size()*sizeof(double));
		const uint32_t count=mur.m_numLines[0]*mur.m_numLines[1];
		ReserveRuntime(GPU_SetupMultiply(count,2*sizeof(float)));
		const auto result=std::make_shared<HIP_SetupArray>(context);
		result->count=GPU_SetupMultiply(count,2);result->values=Transfer(Allocate<float>(result->count));
		Run(10,count,result->values);extension_tables[&mur]=result;
		const bool diagnostics=backend->diagnostics;
		mur.SetHostMaterializer([result,&mur,count,diagnostics]() {
			mur.m_Mur_Coeff_nyP.Init("requested Mur coefficients",mur.m_numLines);
			mur.m_Mur_Coeff_nyPP.Init("requested Mur coefficients",mur.m_numLines);
			result->Read(mur.m_Mur_Coeff_nyP.data(),0,count,diagnostics);
			result->Read(mur.m_Mur_Coeff_nyPP.data(),count,count,diagnostics);
		});
		return true;
	}
	bool BuildExcitation(Operator_Ext_Excitation& source) {
		source.Reset(); source.m_Exc=op.GetExcitationSignal();
		if (!source.m_Exc || op.dT==0) return false;
		if (op.CSX->GetPropertyByType(CSProperties::EXCITATION).empty()) {
			std::cerr << "Operator::CalcFieldExcitation: Warning, no excitation properties found" << std::endl;
			return false;
		}
		for (CSProperties* property:op.CSX->GetPropertyByType(CSProperties::EXCITATION))
			Require(static_cast<CSPropExcitation*>(property)->GetDelay()/op.dT<=INT32_MAX,"source delay exceeds runtime index range");
		const auto result=std::make_shared<HIP_SetupSources>(context);
		auto* curve_total=Allocate<unsigned long long>(1);
		auto* summary=Allocate<uint32_t>(4);
		for (unsigned int magnetic=0;magnetic<2;++magnetic) {
			 Run(5,magnetic); Run(6);
			double pointCount=0;
			Download(16,&pointCount,sizeof(pointCount),((3ull*cells+255)/256)*sizeof(double));
			const uint32_t count=uint32_t(pointCount);
			unsigned long long curve_count=0;
			if (!magnetic)
			{
				SetupSourceCurves<<<1,1,0,context->stream>>>(s,op.dT,nullptr,0,curve_total);
				HIP_Check(hipGetLastError(),"curve source counting kernel");
				Copy(&curve_count,curve_total,sizeof(curve_count)); Flush();
				if (curve_count>UINT32_MAX-count) throw GPU_SetupUnavailable("curve source table index range exceeded");
			}
			Preparation(count+curve_count<=UINT32_MAX-255ull,"source dispatch range exceeded");
			result->count[magnetic]=count+static_cast<uint32_t>(curve_count);
			result->serial[magnetic]=curve_count!=0;
			result->points[magnetic]=Transfer(Allocate<SourcePoint>(result->count[magnetic]));
			Run(7,magnetic,reinterpret_cast<float*>(result->points[magnetic]));
			if (curve_count)
			{
				SetupSourceCurves<<<1,1,0,context->stream>>>(s,op.dT,result->points[magnetic]+count,curve_count,curve_total);
				HIP_Check(hipGetLastError(),"curve source filling kernel");
				std::cout << "HIP curve source paths active: " << curve_count << " contributions" << std::endl;
			}
			HIP_Check(hipMemsetAsync(summary,0,4*sizeof(uint32_t),context->stream),"source summary reset");
			if (result->count[magnetic]) SetupSourceSummary<<<(result->count[magnetic]+255u)/256,256,0,context->stream>>>(s,result->points[magnetic],result->count[magnetic],summary);
			HIP_Check(hipGetLastError(),"source summary kernel");
			uint32_t counts[4];Copy(counts,summary,sizeof(counts));Flush();
			if (magnetic) {source.Curr_Count=result->count[1];std::copy(counts,counts+3,source.Curr_Count_Dir);if (counts[3]) ++backend->fused_blockers;}
			else {source.Volt_Count=result->count[0];std::copy(counts,counts+3,source.Volt_Count_Dir);}
		}
		Release(curve_total,sizeof(unsigned long long));Release(summary,4*sizeof(uint32_t));
		source_results[&source]=result;
		const bool diagnostics=backend->diagnostics;
		const uint32_t nx=s.dims[0],ny=s.dims[1],nz=s.dims[2];
		source.m_BoundaryDelay=[result,nx,ny,nz,diagnostics](unsigned int normal,unsigned int line) {
			const unsigned int dims[3]={nx,ny,nz};return result->BoundaryDelay(normal,line,dims,diagnostics);
		};
		source.SetHostMaterializer([result,&source,nx,ny,nz,diagnostics]() {
			const uint32_t cells=nx*ny*nz;
			for (unsigned int current=0;current<2;++current)
			{
				std::vector<SourcePoint> points;result->Materialize(current,points,diagnostics);
				std::vector<unsigned int> positions[3],delays,directions;
				std::vector<float> amplitudes;
				for (const auto& point:points)
				{
					const uint32_t node=point.index%cells;
					directions.push_back(point.index/cells);delays.push_back(point.delay);amplitudes.push_back(point.amplitude);
					positions[0].push_back(node/(ny*nz));positions[1].push_back((node/nz)%ny);positions[2].push_back(node%nz);
				}
				if (current) source.setupCurrentExcitation(positions,amplitudes,delays,directions);
				else source.setupVoltageExcitation(positions,amplitudes,delays,directions);
			}
		});
		return true;
	}
	double LumpedGeometry(int direction, const unsigned int start[3], const unsigned int stop[3]) override {
		double data[7];
		for (int n=0;n<3;++n) { data[n]=start[n]; data[3+n]=stop[n]; } data[6]=direction; Upload(17,data,sizeof(data));  Run(8);
		double result=0; Download(16,&result,sizeof(result)); return result;
	}
	void ApplyLumpedRC(int direction, const unsigned int start[3], const unsigned int stop[3],
		double capacitance, double resistance, double epsilon, double kappa, bool caps) override {
		double data[12];
		for (int n=0;n<3;++n) { data[n]=start[n]; data[3+n]=stop[n]; }
		data[6]=direction; data[7]=capacitance; data[8]=resistance; data[9]=epsilon; data[10]=kappa; data[11]=caps?1:0; Upload(17,data,sizeof(data));  Run(9);
		std::cout << "HIP RC construction active" << std::endl;
	}

};

GPU_OperatorSetup* GPU_Backend_HIP::NewOperatorSetup(const GPU_SetupModel& model, std::string& reason)
{
	if (!model.op) {reason="construction operator is missing"; return NULL;}
    std::unique_ptr<HIP_OperatorSetup> setup(new HIP_OperatorSetup(d,model));
    setup->Preflight();
	hipDeviceProp_t prop; HIP_Check(hipGetDeviceProperties(&prop,d->ctx->device),"construction device properties");
#ifdef __HIP_PLATFORM_AMD__
	const bool fp64=prop.arch.hasDoubles;
#else
	const bool fp64=prop.major>=2;
#endif
	if (!fp64) { reason="FP64 is required for GPU construction"; return NULL; }
	if (prop.maxThreadsPerBlock<256 || (3ull*model.dims[0]*model.dims[1]*model.dims[2]+255)/256>static_cast<unsigned int>(prop.maxGridSize[0]))
	{ reason="construction dispatch exceeds device limits"; return NULL; }
	size_t available=0,total=0; HIP_Check(hipMemGetInfo(&available,&total),"construction memory budget");
	const size_t device_free=available;if (!d->ctx->setup_start_free) d->ctx->setup_start_free=device_free;
	const unsigned int stores=4;
	const size_t cells=size_t(model.dims[0])*model.dims[1]*model.dims[2];
	size_t estimate=GPU_SetupMultiply(cells,3*(8+stores)*sizeof(float));
	for (unsigned int k=0;k<4;++k) if (model.stores&(1u<<k)) estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(3*cells,sizeof(float)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply((3*cells+255)/256+1,sizeof(double)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(((3*cells+255)/256+255)/256,sizeof(double)));
	estimate=GPU_SetupAdd(estimate,setup->ExtensionBytes());
	if (model.axis) estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(model.dims[2],2*sizeof(float)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(GPU_SetupAdd(GPU_SetupAdd(model.grid.size(),model.child_alpha.size()),model.geometry.size()),sizeof(double)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(model.shapes.size(),sizeof(GPU_SetupPrimitive)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(model.geometry_index_nodes,sizeof(GPU_SetupBVHNode)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(model.geometry_index_scratch,sizeof(uint64_t)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(model.expressions.programs.size(),sizeof(GPU_SetupExpression)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(model.expressions.instructions.size(),sizeof(GPU_SetupInstruction)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(model.expressions.constants.size(),sizeof(double)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(GPU_SetupMultiply(3*cells,model.expressions.stack_size),sizeof(double)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(model.discrete_materials.size(),sizeof(GPU_SetupDiscreteMaterial)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(model.material_data.size(),sizeof(float)));
	estimate=GPU_SetupAdd(estimate,model.material_indices.size());
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(model.file_weights.size(),sizeof(GPU_SetupFileWeight)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(model.dispersion.size(),sizeof(GPU_SetupDispersionPole)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(model.properties.size(),sizeof(GPU_SetupProperty)));
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(GPU_SetupAdd(model.candidate_order.size(),size_t(model.primitive_count)+4),sizeof(unsigned int)));
	const size_t candidate_rows=HIP_SetupCandidateRows(model.dims);
	estimate=GPU_SetupAdd(estimate,GPU_SetupMultiply(GPU_SetupAdd(candidate_rows+1,HIP_SetupScanWorkspace(candidate_rows+1)),sizeof(uint64_t)));
	const size_t runtime_reserve=GPU_SetupAdd(setup->RuntimeBytes(),GPU_SetupAdd(HIP_SetupPackingBytes(cells),GPU_SetupAdd(GPU_SetupMultiply(cells,12*sizeof(float)),64ull*1024*1024)));
	estimate=GPU_SetupAdd(estimate,runtime_reserve);
	// Reserve both field pairs: the existing fused step can allocate the next pair.
	{
		const size_t limit=GPU_SetupMemoryLimit();
		const size_t live=d->ctx->setup_start_free>device_free ? d->ctx->setup_start_free-device_free : 0;
		available=static_cast<size_t>(std::min<unsigned long long>(available,limit>live ? limit-live : 0));
	}
	if (d->ctx->setup_future_bytes>available) {reason="insufficient device memory budget for pending multigrid fields and histories";return nullptr;}
	available-=d->ctx->setup_future_bytes;
	if (estimate>available-available/5) { reason="insufficient device memory budget for operator construction"; return NULL; }
	if (d->diagnostics)
	{
		int runtime=0,driver=0;
		HIP_Check(hipRuntimeGetVersion(&runtime),"construction runtime version");
		HIP_Check(hipDriverGetVersion(&driver),"construction driver version");
		std::cout << "GPU setup environment: device=" << prop.name << " runtime=" << runtime << " driver=" << driver
		          << " solver=" << GIT_VERSION << " reserved_bytes=" << estimate << " available_bytes=" << available
		          << " hierarchy_future_bytes=" << d->ctx->setup_future_bytes;
#ifdef __CUDACC_VER_MAJOR__
		std::cout << " compiler=CUDA-" << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__;
#elif defined(__clang_version__)
		std::cout << " compiler=" << __clang_version__;
#endif
		std::cout << std::endl;
	}
	setup->Prepare(model,available-available/5-estimate,available-available/5-runtime_reserve);
	return setup.release();
}
