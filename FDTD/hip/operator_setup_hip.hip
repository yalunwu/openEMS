/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_internal.h"
#include "FDTD/gpu_operator_setup.h"
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
	int boundary[6];
	unsigned int bc_size[6];
	const double *grid, *geometry;
	const unsigned int* candidates;
	unsigned int* used;
	float *ec[4], *material[4], *coeff[4];
};

__device__ static unsigned int ECIndex(const SetupDevice& s, int n, const int* p)
{
	return n*s.cells+(p[2]*s.dims[1]+p[1])*s.dims[0]+p[0];
}
__device__ static double Grid(const SetupDevice& s, int n, int p, int kind)
{
	return s.grid[s.offsets[n]+6*p+kind];
}
__device__ static double Width(const SetupDevice& s, int n, int p, bool dual)
{
	return Grid(s,n,p,dual?5:4);
}
__device__ static double Area(const SetupDevice& s, int n, const int* p, bool dual)
{
	if (p[0]<0 || p[1]<0 || p[2]<0) return 0;
	int a=(n+1)%3,b=(n+2)%3;
	return Width(s,a,p[a],!dual)*Width(s,b,p[b],!dual);
}
__device__ static bool Inside(const SetupDevice& s, unsigned int prim, const double* coord)
{
	const double* d=s.geometry+48*prim;
	double p[3],distance=0;
	for (int n=0;n<3;++n) {
		p[n]=0;
		for (int k=0;k<3;++k) p[n]+=d[4+n*4+k]*coord[k];
		p[n]+=d[4+n*4+3];
	}
	if (d[0]==1) {
		for (int n=0;n<3;++n) if (p[n]<fmin(d[16+n],d[19+n]) || p[n]>fmax(d[16+n],d[19+n])) return false;
		return true;
	}
	if (d[0]==2) {
		for (int n=0;n<3;++n) distance+=(p[n]-d[16+n])*(p[n]-d[16+n]);
		return sqrt(distance)<d[22];
	}
	for (int n=0;n<3;++n) if (p[n]<d[23+n*2] || p[n]>d[24+n*2]) return false;
	if (d[0]==4) {
		unsigned int offset=static_cast<unsigned int>(d[17]),count=static_cast<unsigned int>(d[18]);
		if (!count) return false;
		int normal=static_cast<int>(d[16]),winding=0;
		double x=p[(normal+1)%3],y=p[(normal+2)%3];
		double x1=s.geometry[offset+2*(count-1)],y1=s.geometry[offset+2*(count-1)+1];
		bool startover=y1>=y;
		for (unsigned int i=0;i<count;++i) {
			double x2=s.geometry[offset+2*i],y2=s.geometry[offset+2*i+1];
			if (x2==x1 && x1==x && ((y<y1 && y>y2) || (y>y1 && y<y2))) return true;
			if (y2==y1 && y1==y && ((x<x1 && x>x2) || (x>x1 && x<x2))) return true;
			bool endover=y2>=y;
			if (startover!=endover) {
				double left=(y2-y)*(x2-x1),right=(y2-y1)*(x2-x);
				if (left<=right) { if (endover) ++winding; }
				else if (!endover) --winding;
			}
			startover=endover; x1=x2; y1=y2;
		}
		return winding!=0;
	}
	double dir[3],ll=0,projection=0;
	for (int n=0;n<3;++n) { dir[n]=d[19+n]-d[16+n]; ll+=dir[n]*dir[n]; projection+=(p[n]-d[16+n])*dir[n]; }
	double foot=projection/ll;
	if (foot<0 || foot>1) return false;
	for (int n=0;n<3;++n) { double delta=p[n]-(d[16+n]+foot*dir[n]); distance+=delta*delta; }
	return sqrt(distance)<=d[22];
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
			if (loc[0]<0 || loc[1]<0 || loc[2]<0) continue;
			for (int k=0;k<3;++k) c[k]=Grid(s,k,loc[k],1);
		}
		double weight=Area(s,n,loc,true);
		const double* raw=s.geometry+48*Property(s,c,p,0)+32+n*4;
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
			if (loc[0]<0 || loc[1]<0 || loc[2]<0) continue;
			for (int k=0;k<3;++k) c[k]=Grid(s,k,loc[k],1);
		}
		double weight=loc[n]<0?0.0:Width(s,n,loc[n],false);
		const double* raw=s.geometry+48*Property(s,c,p,0)+32+n*4;
		m[2]+=weight/raw[2];
		if (raw[3]!=0) m[3]+=weight/raw[3]; else m[3]=0;
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
		if (s.stores&(1u<<k)) s.material[k][id]=float(m[k]);
	}
	unsigned int e=ECIndex(s,n,p);
	double length=Width(s,n,p[n],false),area=Area(s,n,p,false);
	s.ec[0][e]=float(m[0]*area/length); s.ec[1][e]=float(m[1]*area/length);
	length=Width(s,n,p[n],true); area=Area(s,n,p,true);
	s.ec[2][e]=float(m[2]*area/length); s.ec[3][e]=float(m[3]*area/length);
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
__global__ static void SetupCoefficients(SetupDevice s, double dt)
{
	unsigned int id=blockIdx.x*blockDim.x+threadIdx.x;
	if (id>=3*s.cells) return;
	int n=id/s.cells,p[3]; Position(s,id%s.cells,p);
	unsigned int e=ECIndex(s,n,p);
	double C=s.ec[0][e],G=s.ec[1][e],L=s.ec[2][e],R=s.ec[3][e];
	s.coeff[0][id]=C>0?float((1-dt*G/2/C)/(1+dt*G/2/C)):0;
	s.coeff[1][id]=C>0?float((dt/C)/(1+dt*G/2/C)):0;
	s.coeff[2][id]=L>0?float((1-dt*R/2/L)/(1+dt*R/2/L)):0;
	s.coeff[3][id]=L>0?float((dt/L)/(1+dt*R/2/L)):0;
	for (int a=0;a<3;++a) {
		if ((a!=n && p[a]==0 && s.boundary[2*a]!=-1) || (p[a]==int(s.dims[a])-1 && s.boundary[2*a+1]!=-1)) { s.coeff[0][id]=0; s.coeff[1][id]=0; }
		if (p[a]==int(s.dims[a])-1 || (p[a]==0 && s.boundary[2*a]==1) || (a!=n && p[a]==int(s.dims[a])-2 && s.boundary[2*a+1]==1)) { s.coeff[2][id]=0; s.coeff[3][id]=0; }
	}
	double coord[3]={Grid(s,0,p[0],0),Grid(s,1,p[1],0),Grid(s,2,p[2],0)}; coord[n]=Grid(s,n,p[n],1);
	unsigned int prim=Property(s,coord,p,1);
	if (s.geometry[48*prim+1]==1) { s.coeff[0][id]=0; s.coeff[1][id]=0; atomicAdd(s.used+s.primitives+n,1u); }
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
    const double *aux,*geo;
    const unsigned int* cand;
    const unsigned int* BC_SIZE;
    unsigned int *used,*winners,CELLS,PEC_INDEX,ERROR_INDEX;
    float *material0,*material1,*material2,*ext,*vv,*vi,*ii,*iv;
    double* reduction;
    I3 DIMS;
    HIP_SetupCircuit ec0,ec1,ec2,ec3;
    __device__ HIP_SetupExtensionKernel(SetupDevice device,HIP_SetupExtensionParams params,const double* a,
        float* output,double* sums,unsigned int* scan) : s(device),pc(params),aux(a),geo(s.geometry),cand(s.candidates),
        BC_SIZE(s.bc_size),used(s.used),winners(scan),CELLS(s.cells),PEC_INDEX(s.primitives),ERROR_INDEX(s.primitives+3),
        material0(s.material[0]),material1(s.material[1]),material2(s.material[2]),ext(output),
        vv(s.coeff[0]),vi(s.coeff[1]),ii(s.coeff[2]),iv(s.coeff[3]),reduction(sums),
        DIMS(s.dims[0],s.dims[1],s.dims[2]),ec0{&s,0},ec1{&s,1},ec2{&s,2},ec3{&s,3} {}
    __device__ uint linear(I3 p) const { return (p.x*s.dims[1]+p.y)*s.dims[2]+p.z; }
    __device__ I3 position(uint id) const { return I3(id/(s.dims[1]*s.dims[2]),(id/s.dims[2])%s.dims[1],id%s.dims[2]); }
    __device__ double g(int n,int p,int kind) const { return Grid(s,n,p,kind); }
    __device__ double width(int n,int p,bool dual) const { return Width(s,n,p,dual); }
    __device__ double area(int n,I3 p,bool dual) const { int pos[3]={p.x,p.y,p.z}; return Area(s,n,pos,dual); }
    __device__ bool inside(uint prim,D3 c) const { double coord[3]={c.x,c.y,c.z}; return Inside(s,prim,coord); }
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
        int a=(n+1)%3,b=(n+2)%3;
        bool valid=magnetic==0?p[n]<int(DIMS[n])-1:
            p[a]<int(DIMS[a])-1&&p[b]<int(DIMS[b])-1;
        double coefficient=double(magnetic==0?vi[cell]:iv[cell]);
        if(!valid||coefficient==0.0) continue;
        D3 c;
        for(int axis=0;axis<3;++axis) c[axis]=g(axis,p[axis],(axis==n)!=(magnetic!=0)?1:0);
        uint prim=property(c,p,1);
        uint offset=4+(prim*uint(aux[1])+pole)*18+uint(n)*6+uint(magnetic)*3;
        double plasma=aux[offset]*2.0*3.1415926535897932384626433832795;
        double capacitance=double(magnetic==0?ec0[cell]:ec2[cell]);
        if(plasma<=0.0||capacitance<=0.0) continue;
        double reactance=1.0/(plasma*plasma*capacitance);
        double loss=aux[offset+1]>0.0?reactance/aux[offset+1]:0.0;
        double omega=aux[offset+2]*2.0*3.1415926535897932384626433832795;
        double other=omega>0.0?1.0/(omega*omega*reactance):0.0;
        D3 result=D3((2.0*reactance-pc.dt*loss)/(2.0*reactance+pc.dt*loss),
            pc.dt/(reactance+pc.dt*loss/2.0)*coefficient,other>0.0?pc.dt/other/coefficient:0.0);
        if(magnetic==0) v=result; else i=result;
    }
}
__device__ void debye(int n,I3 p,uint pole,D2& result) {
    result=D2(0.0);
    uint cell=uint(n)*CELLS+linear(p);
    if(p[n]>=int(DIMS[n])-1||vi[cell]==0.0) return;
    D3 c=D3(g(0,p.x,0),g(1,p.y,0),g(2,p.z,0)); c[n]=g(n,p[n],1);
    uint prim=property(c,p,1),offset=4+(prim*uint(aux[1])+pole)*6+uint(n)*2;
    double delta=aux[offset],tau=aux[offset+1];
    if(delta<=0.0||tau<=0.0) return;
    if(tau<3.0*pc.dt) { atomicOr(&used[ERROR_INDEX],2); return; }
    double C=8.85418781762e-12*delta*(area(n,p,false)/width(n,p[n],false));
    result=D2((2.0*tau-pc.dt)/(2.0*tau+pc.dt),C*double(vi[cell])/(2.0*tau+pc.dt));
}
__device__ int sheetTangent(uint prim,int n) {
    if(prim==0) return -1;
    uint offset=prim*48; int a=(n+1)%3,b=(n+2)%3,result=-1;
    if(geo[offset+23+uint(a)*2]!=geo[offset+24+uint(a)*2]) result=a;
    if(geo[offset+23+uint(b)*2]!=geo[offset+24+uint(b)*2]) result=b;
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
                if(p[n]<int(DIMS[n])-1) {
                    D3 c=D3(g(0,p.x,0),g(1,p.y,0),g(2,p.z,0)); c[n]=g(n,p[n],1);
                    uint selected=property(c,p,1),offset=selected*48;
                    if(geo[offset+29]!=0.0) {
                        if(geo[offset+31]!=2.0||disabled||float(geo[offset+32])<=0.0||float(geo[offset+33])<=0.0) {
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
        float conductivity=float(geo[prim*48+32]),thickness=float(geo[prim*48+33]);
        float G0=conductivity*thickness;
        float w0=float(8.0/double(G0)/double(thickness)/1.256637062e-6);
        float wstop=float(aux[0]*2.0*3.1415926535897932384626433832795),omega=wstop/w0;
        uint table=0; while(table<29&&aux[2+table*6]<=double(omega)) ++table;
        float factor=(sheetTangent(__float_as_uint(material2[uint(tangent)*CELLS+node]),tangent)<0||
            sheetTangent(__float_as_uint(material2[uint(tangent)*CELLS+linear(previous)]),tangent)<0)?2.0:1.0;
        float wtl=float(width(n,p[n],false)/width(tangent,p[tangent],true));
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
        double distance=aux[1];
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
        ext[id*10]=__uint_as_float(node); ext[id*10+6]=1.0;
        double omega=2.0*3.1415926535897932384626433832795*aux[14];
        double L=aux[8],R=aux[9],G=aux[10],C=aux[11],naturalC=double(ec0[cell]);
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
            double ib0=aux[15],b1=aux[16],b2=aux[17];
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
        double delay=distance*aux[21]/aux[20]/pc.dt+(voltage?1.0:0.0);
        float amplitude=float(aux[(voltage?17:14)+uint(incident)]*width(incident,p[incident],voltage));
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
        D3 c=D3(g(0,p.x,0),g(1,p.y,0),g(2,p.z,0)); c[axis]=aux[3];
        uint prim=0;
        for(uint q=uint(aux[5+row]);q<uint(aux[6+row]);++q) {
            uint candidate=uint(aux[q]); if(inside(candidate,c)) { prim=candidate; break; }
        }
        int component=id<count?a:b;
        uint offset=prim*48+32+uint(component)*4;
        double epsilon=geo[offset],mu=geo[offset+2],distance=aux[4],phase;
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
                plane+=area(n,p,false)/width(n,p[n],false);
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
            double amplitude=geo[prim*48+32+uint(n)*4]*width(n,p[n],magnetic);
            ext[slot*6]=__uint_as_float(cell); ext[slot*6+1]=float(amplitude);
            ext[slot*6+2]=__uint_as_float(uint(geo[prim*48+30]/pc.dt));
            ext[slot*6+3]=0.0; ext[slot*6+4]=0.0; ext[slot*6+5]=0.0;
            return;
        }
        uint prim=0;
        if(id<3*CELLS) {
            bool valid=magnetic?(p.x<DIMS.x-1&&p.y<DIMS.y-1&&p.z<DIMS.z-1):p[n]<int(DIMS[n])-1;
            if(valid) {
                D3 c;
                for(int axis=0;axis<3;++axis) c[axis]=g(axis,p[axis],(axis==n)!=magnetic?1:0);
                uint selected=sourceProperty(c,p);
                uint b=selected*48;
                bool type=magnetic?(geo[b+3]==2.0||geo[b+3]==3.0):(geo[b+3]==0.0||geo[b+3]==1.0);
                if(selected!=0&&geo[b+2]!=0.0&&geo[b+33+uint(n)*4]!=0.0&&type) {
                    if(geo[b+3]==1.0) { vv[cell]=0.0; vi[cell]=0.0; }
                    if(geo[b+3]==3.0) { ii[cell]=0.0; iv[cell]=0.0; }
                    if(geo[b+32+uint(n)*4]*width(n,p[n],magnetic)!=0.0) prim=selected;
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
        uint stride=DIMS.x+DIMS.y+DIMS.z;
        D3 kv,ki;
        for(int axis=0;axis<3;++axis) {
            uint offset=axis==0?0:axis==1?DIMS.x:DIMS.x+DIMS.y;
            kv[axis]=aux[6+uint(n)*2*stride+offset+uint(p[axis])];
            ki[axis]=aux[6+(uint(n)*2+1)*stride+offset+uint(p[axis])];
        }
        D4 m=effective(n,p);
        double E=8.85418781762e-12;
        uint total=3*count;
        for(uint k=0;k<6;++k) ext[k*total+id]=0.0;
        if((kv.x+kv.y+kv.z)!=0.0&&m.y<1e3) {
            if((vv[cell]+vi[cell])!=0.0) {
                vv[cell]=float((2.0*E-kv[a]*pc.dt)/(2.0*E+kv[a]*pc.dt));
                vi[cell]=float((2.0*E*pc.dt)/(2.0*E+kv[a]*pc.dt)*width(n,p[n],false)/area(n,p,false));
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
                iv[cell]=float((2.0*E*pc.dt)/(2.0*E+ki[a]*pc.dt)*width(n,p[n],true)/area(n,p,true));
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
            if(aux[7]>0.0) ec0[id]=float(aux[9]*area(n,p,false)/width(n,p[n],false));
            if(aux[8]>0.0) ec1[id]=float(aux[10]*area(n,p,false)/width(n,p[n],false));
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

class HIP_OperatorSetup : public GPU_OperatorSetup
{
	std::shared_ptr<HIP_Context> context;
	GPU_Backend_HIP::Impl* backend;
	SetupDevice s;
	std::vector<void*> allocations;
	double* reduction;
	std::vector<double> host_reduction;
	std::vector<unsigned int> host_used;
	unsigned int groups;
	template<typename T> T* Allocate(size_t count, const T* host=NULL)
	{
		T* ptr=NULL;
		hipError_t status=hipMalloc(&ptr,std::max(size_t(1),count)*sizeof(T));
		if (status==hipErrorOutOfMemory) { hipGetLastError(); throw GPU_SetupUnavailable("insufficient device memory for operator construction"); }
		HIP_Check(status,"operator setup allocation");
		try { allocations.push_back(ptr); } catch (...) { hipFree(ptr); throw; }
		if (host && count) HIP_Check(hipMemcpyAsync(ptr,host,count*sizeof(T),hipMemcpyHostToDevice,context->stream),"operator setup upload");
		else HIP_Check(hipMemsetAsync(ptr,0,std::max(size_t(1),count)*sizeof(T),context->stream),"operator setup clear");
		return ptr;
	}
	void Flush() { backend->Flush(); }
	void Copy(void* host, const void* device, size_t bytes)
	{
		HIP_Check(hipMemcpyAsync(host,device,bytes,hipMemcpyDeviceToHost,context->stream),"operator construction readback");
	}
public:
	explicit HIP_OperatorSetup(GPU_Backend_HIP::Impl* impl,const GPU_SetupModel& model) : context(impl->ctx), backend(impl), s(), reduction(NULL), groups(0),
        op(*model.op),primitives(model.primitives),cells(model.dims[0]*model.dims[1]*model.dims[2]),primCount(model.primitive_count),
        auxiliary(NULL),extension_output(NULL),auxiliary_capacity(0),output_floats(0) {
        static_assert(sizeof(SourcePoint)==24,"source construction layout");
        for(size_t n=0;n<primitives.size();++n) indices[primitives[n]]=static_cast<uint32_t>(n+1);
    }
	~HIP_OperatorSetup() { hipStreamSynchronize(context->stream); for (void* ptr:allocations) hipFree(ptr); }
	void Prepare(const GPU_SetupModel& model)
	{
		std::copy(model.dims,model.dims+3,s.dims); std::copy(model.offsets,model.offsets+3,s.offsets);
		std::copy(model.boundary,model.boundary+6,s.boundary);
		std::copy(model.bc_size,model.bc_size+6,s.bc_size);
		s.cells=s.dims[0]*s.dims[1]*s.dims[2]; s.average=model.average; s.stores=model.stores; s.primitives=model.primitive_count;
		groups=static_cast<unsigned int>((3ull*s.cells+255)/256);
		host_reduction.resize(size_t(groups)+1); host_used.resize(size_t(s.primitives)+4);
		s.grid=Allocate(model.grid.size(),model.grid.data()); s.geometry=Allocate(model.geometry.size(),model.geometry.data());
		s.candidates=Allocate(model.candidates.size(),model.candidates.data()); s.used=Allocate<unsigned int>(host_used.size());
		for (int k=0;k<4;++k) {
			s.ec[k]=Allocate<float>(3ull*s.cells); s.coeff[k]=Allocate<float>(3ull*s.cells);
			s.material[k]=Allocate<float>(3ull*s.cells);
		}
		reduction=Allocate<double>(host_reduction.size());
		auxiliary=Allocate<double>(auxiliary_capacity); extension_output=Allocate<float>(output_floats); Flush();
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
			for (int n=0;n<3;++n) Copy(ec[k][n],s.ec[k]+size_t(n)*s.cells,size_t(s.cells)*sizeof(float));
			if (materials[k]) Copy(materials[k],s.material[k],3ull*s.cells*sizeof(float));
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
		Copy(host_reduction.data(),reduction,size_t(groups)*sizeof(double)); Flush();
		timer.Report(timed?"timestep reduction readback":"timestep kernel and reduction readback");
		double dt=*std::min_element(host_reduction.begin(),host_reduction.begin()+groups);
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
		if (timed) { HIP_KernelTimes::End(context->stream,"setup_coefficients_pec"); timer.Report("coefficient/PEC kernel"); }
		for (int k=0;k<4;++k) Copy(coefficients[k],s.coeff[k],3ull*s.cells*sizeof(float));
		Copy(host_used.data(),s.used,host_used.size()*sizeof(unsigned int)); Flush(); used=host_used;
		timer.Report(timed?"coefficient/PEC readback":"coefficient/PEC kernel and readback");
	}
	Operator& op;
	std::vector<CSPrimitives*> primitives;
	std::map<CSPrimitives*,uint32_t> indices;
	std::map<Operator_Ext_UPML*,std::vector<double>> grading_tables;
	uint32_t cells,primCount;
	double* auxiliary;
	float* extension_output;
	size_t auxiliary_capacity,output_floats;
	struct SourcePoint { uint32_t index; float amplitude; uint32_t delay,offset,length,period; };
	static void Require(bool success,const char* reason) {
		if (!success) throw std::runtime_error(std::string("HIP construction: ")+reason);
	}
	static bool ConstantWeight(const std::string& term) {
		if (term.empty()) return true;
		char* end=NULL; double value=std::strtod(term.c_str(),&end);
		while (end && (*end==' ' || *end=='\t')) ++end;
		return end!=term.c_str() && end && !*end && std::isfinite(value);
	}
	static void Preparation(bool success,const char* reason) {
		if (!success) throw GPU_SetupUnavailable(reason);
	}
	std::vector<double> GradingTable(Operator_Ext_UPML& upml) {
		const size_t stride=op.numLines[0]+op.numLines[1]+op.numLines[2];
		std::vector<double> tables(6+6*stride);
		for(int n=0;n<3;++n) { tables[n]=upml.m_StartPos[n]; tables[3+n]=upml.m_numLines[n]; }
		for(unsigned int component=0;component<3;++component) {
			size_t offset=0;
			for(unsigned int axis=0;axis<3;++axis) {
				for(unsigned int index=0;index<op.numLines[axis];++index) {
					unsigned int p[3]={0,0,0}; p[axis]=index;
					double kv[3]={0,0,0},ki[3]={0,0,0};
					upml.CalcGradingKappa(component,p,Z0,kv,ki);
					Preparation(std::isfinite(kv[axis])&&std::isfinite(ki[axis]),"non-finite UPML grading value");
					tables[6+component*2*stride+offset+index]=kv[axis];
					tables[6+(component*2+1)*stride+offset+index]=ki[axis];
				}
				offset+=op.numLines[axis];
			}
		}
		return tables;
	}
public:
	size_t ExtensionBytes() const {return auxiliary_capacity*sizeof(double)+output_floats*sizeof(float);}
	void Preflight() {
		const size_t maxdim=std::max(op.numLines[0],std::max(op.numLines[1],op.numLines[2]));
		auxiliary_capacity=std::max(size_t(182),6+6*size_t(op.numLines[0]+op.numLines[1]+op.numLines[2]));
		auxiliary_capacity=std::max(auxiliary_capacity,6+maxdim*(size_t(primCount)+1));
		output_floats=1;
		for(size_t n=0;n<op.GetNumberOfExtentions();++n) {
			Operator_Extension* extension=op.GetExtension(n);
			Preparation(typeid(*extension)==typeid(Operator_Ext_UPML)||typeid(*extension)==typeid(Operator_Ext_Excitation)||
				typeid(*extension)==typeid(Operator_Ext_TFSF)||typeid(*extension)==typeid(Operator_Ext_LumpedRLC)||
				typeid(*extension)==typeid(Operator_Ext_SteadyState)||typeid(*extension)==typeid(Operator_Ext_Mur_ABC)||
				typeid(*extension)==typeid(Operator_Ext_Absorbing_BC)||typeid(*extension)==typeid(Operator_Ext_LorentzMaterial)||
				typeid(*extension)==typeid(Operator_Ext_DebyeMaterial)||typeid(*extension)==typeid(Operator_Ext_ConductingSheet),
				"unsupported extension construction type");
			if(auto* upml=dynamic_cast<Operator_Ext_UPML*>(extension)) {
				size_t count=1;
				for(int k=0;k<3;++k) {
					Preparation(upml->m_StartPos[k]<=op.numLines[k] && upml->m_numLines[k]<=op.numLines[k]-upml->m_StartPos[k],"invalid UPML range");
					count*=upml->m_numLines[k];
				}
				output_floats=std::max(output_floats,18*count);
				grading_tables[upml]=GradingTable(*upml);
			} else if(dynamic_cast<Operator_Ext_Excitation*>(extension)) {
				// shortcut: reserve worst-case sparse outputs before mutation; tighten if large meshes fall back.
				output_floats=std::max(output_floats,size_t(18)*cells);
			} else if(dynamic_cast<Operator_Ext_ConductingSheet*>(extension)) output_floats=std::max(output_floats,size_t(12)*cells);
			else if(dynamic_cast<Operator_Ext_LumpedRLC*>(extension)) output_floats=std::max(output_floats,size_t(10)*cells);
			else if(dynamic_cast<Operator_Ext_TFSF*>(extension)) output_floats=std::max(output_floats,size_t(6)*cells);
			else if(!dynamic_cast<Operator_Ext_SteadyState*>(extension)) output_floats=std::max(output_floats,size_t(3)*cells);
		}
		unsigned int plane_waves=0;
		for(CSProperties* property:op.CSX->GetPropertyByType(CSProperties::EXCITATION))
			if(static_cast<CSPropExcitation*>(property)->GetExcitType()==10) ++plane_waves;
		Preparation(plane_waves<=1,"multiple TFSF sources require CPU setup");
		for(CSPrimitives* primitive:primitives) {
			if(auto* mat=dynamic_cast<CSPropLorentzMaterial*>(primitive->GetProperty())) {
				Preparation(mat->GetDispersionOrder()>=0 && size_t(mat->GetDispersionOrder())<=(UINT32_MAX-4ull)/(size_t(primCount)*18),"dispersion index range exceeded");
				auxiliary_capacity=std::max(auxiliary_capacity,4+size_t(primCount)*mat->GetDispersionOrder()*18);
				for(int pole=0;pole<mat->GetDispersionOrder();++pole) for(int n=0;n<3;++n) {
					Preparation(ConstantWeight(mat->GetEpsPlasmaFreqWeightFunction(pole,n))&&ConstantWeight(mat->GetEpsRelaxTimeWeightFunction(pole,n))&&
						ConstantWeight(mat->GetEpsLorPoleFreqWeightFunction(pole,n))&&ConstantWeight(mat->GetMuePlasmaFreqWeightFunction(pole,n))&&
						ConstantWeight(mat->GetMueRelaxTimeWeightFunction(pole,n))&&ConstantWeight(mat->GetMueLorPoleFreqWeightFunction(pole,n)),"spatial dispersion weights require CPU setup");
					double coord[3]={0,0,0};
					const double values[6]={mat->GetEpsPlasmaFreqWeighted(pole,n,coord),mat->GetEpsRelaxTimeWeighted(pole,n,coord),mat->GetEpsLorPoleFreqWeighted(pole,n,coord),
						mat->GetMuePlasmaFreqWeighted(pole,n,coord),mat->GetMueRelaxTimeWeighted(pole,n,coord),mat->GetMueLorPoleFreqWeighted(pole,n,coord)};
					for(double value:values) Preparation(std::isfinite(value),"non-finite dispersion parameter");
				}
			}
			if(auto* mat=dynamic_cast<CSPropDebyeMaterial*>(primitive->GetProperty())) {
				Preparation(mat->GetDispersionOrder()>=0 && size_t(mat->GetDispersionOrder())<=(UINT32_MAX-4ull)/(size_t(primCount)*6),"Debye index range exceeded");
				auxiliary_capacity=std::max(auxiliary_capacity,4+size_t(primCount)*mat->GetDispersionOrder()*6);
				for(int pole=0;pole<mat->GetDispersionOrder();++pole) for(int n=0;n<3;++n) {
					Preparation(ConstantWeight(mat->GetEpsDeltaWeightFunction(pole,n))&&ConstantWeight(mat->GetEpsRelaxTimeWeightFunction(pole,n)),"spatial dispersion weights require CPU setup");
					double coord[3]={0,0,0};
					Preparation(std::isfinite(mat->GetEpsDeltaWeighted(pole,n,coord))&&std::isfinite(mat->GetEpsRelaxTimeWeighted(pole,n,coord)),"non-finite Debye parameter");
				}
			}
		}
		Preparation(auxiliary_capacity<=UINT32_MAX,"extension metadata index range exceeded");
		Preparation(output_floats<=UINT32_MAX,"extension output index range exceeded");
	}
	void Upload(unsigned int index,const void* host,size_t bytes) {
		Require(index==17 && bytes<=auxiliary_capacity*sizeof(double),"extension metadata range exceeded");
		HIP_Check(hipMemcpyAsync(auxiliary,host,bytes,hipMemcpyHostToDevice,context->stream),"extension metadata upload"); Flush();
	}
	void SetAuxiliary(const std::vector<double>& values) { Upload(17,values.data(),values.size()*sizeof(double)); }
	void ClearFlags() {
		HIP_Check(hipMemsetAsync(s.used+primCount+3,0,sizeof(unsigned int),context->stream),"extension flags reset");
	}
	void Download(unsigned int index,void* host,size_t bytes,size_t offset=0) {
		const void* device=index==3?static_cast<const void*>(s.used):index==16?static_cast<const void*>(reduction):static_cast<const void*>(extension_output);
		const size_t capacity=index==3?host_used.size()*sizeof(unsigned int):index==16?host_reduction.size()*sizeof(double):output_floats*sizeof(float);
		Require((index==3||index==16||index==18) && offset<=capacity && bytes<=capacity-offset,"extension readback range exceeded");
		Copy(host,static_cast<const char*>(device)+offset,bytes); Flush();
	}
	void Run(uint32_t mode,uint32_t count=0) {
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
		SetupExtensions<<<blocks,256,0,context->stream>>>(s,params,auxiliary,extension_output,reduction);
		HIP_Check(hipGetLastError(),names[mode]);
		if(timed) HIP_KernelTimes::End(context->stream,names[mode]);
	}
	bool BuildExtension(Operator_Extension& extension) override {
		SetupTimer timer("GPU construction");
		bool active=false;
		if(auto* source=dynamic_cast<Operator_Ext_Excitation*>(&extension)) active=BuildExcitation(*source);
		else if(auto* mur=dynamic_cast<Operator_Ext_Mur_ABC*>(&extension)) active=BuildMur(*mur);
		else if(auto* abc=dynamic_cast<Operator_Ext_Absorbing_BC*>(&extension)) active=BuildAbsorber(*abc);
		else if(auto* sheet=dynamic_cast<Operator_Ext_ConductingSheet*>(&extension)) active=BuildSheet(*sheet);
		else if(auto* debye=dynamic_cast<Operator_Ext_DebyeMaterial*>(&extension)) active=BuildDebye(*debye);
		else if(auto* rlc=dynamic_cast<Operator_Ext_LumpedRLC*>(&extension)) active=BuildRLC(*rlc);
		else if(auto* lorentz=dynamic_cast<Operator_Ext_LorentzMaterial*>(&extension)) active=BuildLorentz(*lorentz);
		else if(auto* upml=dynamic_cast<Operator_Ext_UPML*>(&extension)) active=BuildUPML(*upml);
		else if(dynamic_cast<Operator_Ext_TFSF*>(&extension)||dynamic_cast<Operator_Ext_SteadyState*>(&extension)) active=extension.BuildExtension();
		else throw std::runtime_error("HIP construction: extension changed after preparation");
		timer.Report(extension.GetExtensionName().c_str());
		const auto* source=dynamic_cast<Operator_Ext_Excitation*>(&extension);
		const auto* load=dynamic_cast<Operator_Ext_LumpedRLC*>(&extension);
		if(active && !dynamic_cast<Operator_Ext_SteadyState*>(&extension) &&
		   (!source || source->GetVoltCount()+source->GetCurrCount()>0) && (!load || load->RLC_count>0))
			std::cout << "HIP extension construction active: " << extension.GetExtensionName() << std::endl;
		return active;
	}
	bool BuildUPML(Operator_Ext_UPML& upml) {
		const auto& tables=grading_tables.at(&upml);
		const uint32_t count=upml.m_numLines[0]*upml.m_numLines[1]*upml.m_numLines[2];
		SetAuxiliary(tables); Run(4,count);
		ArrayLib::ArrayNIJK<float>* outputs[6]={&upml.vv,&upml.vvfo,&upml.vvfn,&upml.ii,&upml.iifo,&upml.iifn};
		for(unsigned int n=0;n<6;++n) {
			outputs[n]->Init("HIP UPML",upml.m_numLines);
			Download(18,outputs[n]->data(),3ull*count*sizeof(float),n*3ull*count*sizeof(float));
		}
		return true;
	}
	void SynchronizeOperator() override {
		SetupTimer timer("GPU construction");
		float* coefficients[4]={op.vv_ptr->data(),op.vi_ptr->data(),op.ii_ptr->data(),op.iv_ptr->data()};
		for(int k=0;k<4;++k) Copy(coefficients[k],s.coeff[k],3ull*cells*sizeof(float));
		for(int n=0;n<3;++n) {
			Copy(op.EC_C[n],s.ec[0]+size_t(n)*cells,size_t(cells)*sizeof(float));
			Copy(op.EC_G[n],s.ec[1]+size_t(n)*cells,size_t(cells)*sizeof(float));
		}
		Copy(host_used.data(),s.used,host_used.size()*sizeof(unsigned int)); Flush();
		for(size_t n=0;n<primitives.size();++n) if(host_used[n+1]) primitives[n]->SetPrimitiveUsed(true);
		std::copy(host_used.begin()+primCount,host_used.begin()+primCount+3,op.m_Nr_PEC);
		timer.Report("final coefficients, EC and flags readback");
	}

	std::vector<uint32_t> CompactNodes(std::vector<double>& metadata) {
		SetAuxiliary(metadata);
		Run(12); Run(6);
		double total=0; Download(16,&total,sizeof(total),((3ull*cells+255)/256)*sizeof(double));
		Require(total>=0 && total<=cells,"invalid sparse construction count");
		const uint32_t count=uint32_t(total);
		Run(14,count);
		std::vector<uint32_t> nodes(count);
		if (count) Download(18,nodes.data(),nodes.size()*sizeof(uint32_t));
		return nodes;
	}
	void CopySparsePositions(Operator_Ext_Dispersive& material, unsigned int order, const std::vector<uint32_t>& nodes) {
		material.m_LM_Count[order]=uint32_t(nodes.size());
		material.m_LM_pos[order]=new unsigned int*[3]();
		for (unsigned int n=0;n<3;++n) {
			material.m_LM_pos[order][n]=new unsigned int[nodes.size()];
			for (size_t q=0;q<nodes.size();++q) {
				const uint32_t p[3]={nodes[q]/(op.numLines[1]*op.numLines[2]),(nodes[q]/op.numLines[2])%op.numLines[1],nodes[q]%op.numLines[2]};
				material.m_LM_pos[order][n][q]=p[n];
			}
		}
	}
	void CopySparseCoefficients(std::vector<double>& metadata, uint32_t count, float**& destination) {
		SetAuxiliary(metadata); Run(13,count);
		destination=new float*[3]();
		for (unsigned int n=0;n<3;++n) {
			destination[n]=new float[count];
			if (count) Download(18,destination[n],size_t(count)*sizeof(float),size_t(n)*count*sizeof(float));
		}
	}
	bool BuildLorentz(Operator_Ext_LorentzMaterial& material) {
		Require(material.m_LM_pos==nullptr,"rebuilding a constructed Lorentz extension requires a fresh operator");
		int order=0;
		for (CSProperties* property:op.CSX->GetPropertyByType(CSProperties::LORENTZMATERIAL))
			order=std::max(order,static_cast<CSPropLorentzMaterial*>(property)->GetDispersionOrder());
		if (!order) return false;
		std::vector<double> metadata(4+size_t(primCount)*order*18,0.0); metadata[1]=order;
		double coord[3]={0,0,0};
		for (size_t p=0;p<primitives.size();++p) if (auto* mat=dynamic_cast<CSPropLorentzMaterial*>(primitives[p]->GetProperty()))
			for (int pole=0;pole<mat->GetDispersionOrder();++pole) for (int n=0;n<3;++n) {
				Require(ConstantWeight(mat->GetEpsPlasmaFreqWeightFunction(pole,n)) && ConstantWeight(mat->GetEpsRelaxTimeWeightFunction(pole,n)) &&
					ConstantWeight(mat->GetEpsLorPoleFreqWeightFunction(pole,n)) && ConstantWeight(mat->GetMuePlasmaFreqWeightFunction(pole,n)) &&
					ConstantWeight(mat->GetMueRelaxTimeWeightFunction(pole,n)) && ConstantWeight(mat->GetMueLorPoleFreqWeightFunction(pole,n)),
					"spatial dispersion weights require CPU setup");
				double* data=&metadata[4+((p+1)*order+pole)*18+n*6];
				data[0]=mat->GetEpsPlasmaFreqWeighted(pole,n,coord); data[1]=mat->GetEpsRelaxTimeWeighted(pole,n,coord);
				data[2]=mat->GetEpsLorPoleFreqWeighted(pole,n,coord); data[3]=mat->GetMuePlasmaFreqWeighted(pole,n,coord);
				data[4]=mat->GetMueRelaxTimeWeighted(pole,n,coord); data[5]=mat->GetMueLorPoleFreqWeighted(pole,n,coord);
			}
		material.m_Order=order; material.m_LM_Count.assign(order,0); material.m_LM_pos=new unsigned int**[order]();
		material.m_volt_ADE_On=new bool[order](); material.m_curr_ADE_On=new bool[order]();
		material.m_volt_Lor_ADE_On=new bool[order](); material.m_curr_Lor_ADE_On=new bool[order]();
		float**** outputs[6]={&material.v_int_ADE,&material.v_ext_ADE,&material.v_Lor_ADE,&material.i_int_ADE,&material.i_ext_ADE,&material.i_Lor_ADE};
		for (auto output:outputs) *output=new float**[order]();

		for (int pole=0;pole<order;++pole) {
			metadata[2]=pole;
			uint32_t flags=0; ClearFlags();
			auto nodes=CompactNodes(metadata); CopySparsePositions(material,pole,nodes);
			Download(3,&flags,sizeof(flags),size_t(primCount+3)*sizeof(uint32_t)); flags>>=4;
			material.m_volt_ADE_On[pole]=(flags&1)!=0; material.m_curr_ADE_On[pole]=(flags&2)!=0;
			material.m_volt_Lor_ADE_On[pole]=(flags&4)!=0; material.m_curr_Lor_ADE_On[pole]=(flags&8)!=0;
			const bool enabled[6]={(flags&1)!=0,(flags&1)!=0,(flags&4)!=0,(flags&2)!=0,(flags&2)!=0,(flags&8)!=0};
			for (unsigned int kind=0;kind<6;++kind) if (enabled[kind]) {
				metadata[3]=kind; CopySparseCoefficients(metadata,uint32_t(nodes.size()),(*outputs[kind])[pole]);
			}


		}
		return true;
	}
	bool BuildDebye(Operator_Ext_DebyeMaterial& material) {
		Require(material.m_LM_pos==nullptr,"rebuilding a constructed Debye extension requires a fresh operator");
		int poles=0;
		for (CSProperties* property:op.CSX->GetPropertyByType(CSProperties::DEBYEMATERIAL))
			poles=std::max(poles,static_cast<CSPropDebyeMaterial*>(property)->GetDispersionOrder());
		if (!poles) return false;
		std::vector<double> metadata(4+size_t(primCount)*poles*6,0.0); metadata[0]=1; metadata[1]=poles;
		double coord[3]={0,0,0};
		for (size_t p=0;p<primitives.size();++p) if (auto* mat=dynamic_cast<CSPropDebyeMaterial*>(primitives[p]->GetProperty()))
			for (int pole=0;pole<mat->GetDispersionOrder();++pole) for (int n=0;n<3;++n) {
				Require(ConstantWeight(mat->GetEpsDeltaWeightFunction(pole,n))&&ConstantWeight(mat->GetEpsRelaxTimeWeightFunction(pole,n)),
					"spatial dispersion weights require CPU setup");
				metadata[4+((p+1)*poles+pole)*6+n*2]=mat->GetEpsDeltaWeighted(pole,n,coord);
				metadata[5+((p+1)*poles+pole)*6+n*2]=mat->GetEpsRelaxTimeWeighted(pole,n,coord);
			}
		 auto nodes=CompactNodes(metadata);
		if (nodes.empty()) return false;
		material.m_PoleCount=poles; material.m_Order=1; material.m_LM_Count.assign(1,0);
		material.m_LM_pos=new unsigned int**[1](); CopySparsePositions(material,0,nodes);
		material.m_volt_ADE_On=new bool[1]{true}; material.m_curr_ADE_On=new bool[1]{false};
		material.v_relax_ADE=new float**[poles](); material.v_drive_ADE=new float**[poles]();
		metadata[3]=2; CopySparseCoefficients(metadata,uint32_t(nodes.size()),material.v_solve_ADE);
		for (int pole=0;pole<poles;++pole) {
			metadata[2]=pole; metadata[3]=0; CopySparseCoefficients(metadata,uint32_t(nodes.size()),material.v_relax_ADE[pole]);
			metadata[3]=1; CopySparseCoefficients(metadata,uint32_t(nodes.size()),material.v_drive_ADE[pole]);
		}
		uint32_t flags=0; Download(3,&flags,sizeof(flags),size_t(primCount+3)*sizeof(uint32_t));
		if (flags&2) std::cerr << "GPU Debye setup: relaxation time needs at least 3 timesteps; skipping this pole." << std::endl;
		return true;
	}
	bool BuildAbsorber(Operator_Ext_Absorbing_BC& abc) {
		Require(abc.m_ny>=0 && abc.m_ny<3 && abc.m_phaseVelocity>0,"invalid local absorber metadata");
		abc.m_nyP=(abc.m_ny+1)%3; abc.m_nyPP=(abc.m_ny+2)%3;
		abc.m_numLines[0]=abc.m_sheetX1[abc.m_nyP]-abc.m_sheetX0[abc.m_nyP]+1;
		abc.m_numLines[1]=abc.m_sheetX1[abc.m_nyPP]-abc.m_sheetX0[abc.m_nyPP]+1;
		abc.m_numCells=abc.m_numLines[0]*abc.m_numLines[1];
		unsigned int p[3]={0,0,0}; p[abc.m_ny]=abc.m_sheetX0[abc.m_ny];
		SetAuxiliary({abc.m_phaseVelocity,std::abs(op.GetEdgeLength(abc.m_ny,p)),double(abc.m_ny),
			double(abc.m_sheetX0[0]),double(abc.m_sheetX0[1]),double(abc.m_sheetX0[2]),
			double(abc.m_numLines[0]),double(abc.m_numLines[1]),abc.m_normalSignPositive?1.0:0.0});  Run(15,abc.m_numCells);
		ArrayLib::ArrayIJ<float>* outputs[4]={&abc.m_K1_nyP,&abc.m_K1_nyPP,&abc.m_K2_nyP,&abc.m_K2_nyPP};
		const unsigned int arrays=abc.m_ABCtype==Operator_Ext_Absorbing_BC::MUR_1ST_SA?4:2;
		for (unsigned int kind=0;kind<arrays;++kind) {
			outputs[kind]->Init("GPU local absorber",abc.m_numLines);
			Download(18,outputs[kind]->data(),size_t(abc.m_numCells)*sizeof(float),size_t(kind)*abc.m_numCells*sizeof(float));
		}
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
		const uint32_t count=uint32_t(total); if (!count) return false;  Run(14,count);
		std::vector<uint32_t> nodes(count); Download(18,nodes.data(),nodes.size()*sizeof(uint32_t));
		Run(20,count);
		sheet.m_Order=2; sheet.m_LM_Count.assign(2,0); sheet.m_LM_pos=new unsigned int**[2]();
		for (int order=0;order<2;++order) CopySparsePositions(sheet,order,nodes);
		sheet.m_volt_ADE_On=new bool[2]{true,true}; sheet.m_curr_ADE_On=new bool[2]{false,false};
		sheet.m_volt_Lor_ADE_On=new bool[2]{false,false}; sheet.m_curr_Lor_ADE_On=new bool[2]{false,false};
		sheet.v_int_ADE=new float**[2](); sheet.v_ext_ADE=new float**[2]();
		for (int order=0;order<2;++order) {
			sheet.v_int_ADE[order]=new float*[3](); sheet.v_ext_ADE[order]=new float*[3]();
			for (int n=0;n<3;++n) {
				sheet.v_int_ADE[order][n]=new float[count]; sheet.v_ext_ADE[order][n]=new float[count];
				Download(18,sheet.v_int_ADE[order][n],size_t(count)*sizeof(float),size_t(order*6+n)*count*sizeof(float));
				Download(18,sheet.v_ext_ADE[order][n],size_t(count)*sizeof(float),size_t(order*6+3+n)*count*sizeof(float));
			}
		}
		return true;
	}
	bool BuildRLC(Operator_Ext_LumpedRLC& rlc) {
		Require(rlc.RLC_count==0,"rebuilding a constructed RLC extension requires a fresh operator");
		std::vector<uint32_t> nodes; std::vector<int> directions; std::vector<float> values[9];
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
				const uint32_t series=stop[n]-start[n],parallel=(stop[a]-start[a]+1)*(stop[b]-start[b]+1),count=series*parallel;
				const double dL=std::isnan(L)?0:L*parallel/series,dR=std::isnan(R)?0:R*parallel/series;
				const double dG=std::isnan(R)||R==0?0:(1/R)*series/parallel,dC=std::isnan(C)?0:C*series/parallel;
				double ib0=0,b1=0,b2=0;
				if (load->GetLEtype()==CSPropLumpedElement::SERIES) {
					if (dC==0) { ib0=op.dT/(2*dL+op.dT*dR); b1=-4*dL/op.dT; b2=(2*dL-op.dT*dR)/op.dT; }
					else { ib0=2*op.dT*dC/(4*dL*dC+2*op.dT*dR*dC+op.dT*op.dT); b1=(op.dT*op.dT-4*dL*dC)/(op.dT*dC); b2=(4*dL*dC-2*op.dT*dR*dC+op.dT*op.dT)/(2*op.dT*dC); }
				}
				std::vector<double> metadata(18);
				for (int axis=0;axis<3;++axis) { metadata[axis]=start[axis]; metadata[3+axis]=stop[axis]; }
				metadata[6]=n; metadata[7]=load->GetLEtype()==CSPropLumpedElement::SERIES?1:0;
				metadata[8]=dL; metadata[9]=dR; metadata[10]=dG; metadata[11]=dC; metadata[12]=R>=0?1:0;
				metadata[13]=L>0?1:0; metadata[14]=op.GetExcitationSignal()->GetCenterFreq()+op.GetExcitationSignal()->GetCutOffFreq();
				metadata[15]=ib0; metadata[16]=b1; metadata[17]=b2; SetAuxiliary(metadata);  Run(16,count);
				std::vector<float> result(10ull*count); Download(18,result.data(),result.size()*sizeof(float));
				for (uint32_t q=0;q<count;++q) {
					uint32_t node; std::memcpy(&node,&result[q*10],sizeof(node)); nodes.push_back(node); directions.push_back(n);
					for (int kind=0;kind<9;++kind) values[kind].push_back(result[q*10+kind+1]);
				}

				if (load->GetCaps()) {
					std::vector<double> caps(12,0.0);
					for (int axis=0;axis<3;++axis) { caps[axis]=start[axis]; caps[3+axis]=stop[axis]; }
					caps[6]=n; caps[11]=1; SetAuxiliary(caps); Run(17);
				}
				primitive->SetPrimitiveUsed(true);
			}
		}
		rlc.RLC_count=uint32_t(nodes.size()); if (nodes.empty()) return true;
		rlc.v_RLC_dir=new int[nodes.size()]; std::copy(directions.begin(),directions.end(),rlc.v_RLC_dir);
		rlc.v_RLC_pos=new unsigned int*[3]();
		for (int n=0;n<3;++n) {
			rlc.v_RLC_pos[n]=new unsigned int[nodes.size()];
			for (size_t q=0;q<nodes.size();++q) rlc.v_RLC_pos[n][q]=n==0?nodes[q]/(op.numLines[1]*op.numLines[2]):n==1?(nodes[q]/op.numLines[2])%op.numLines[1]:nodes[q]%op.numLines[2];
		}
		float** targets[9]={&rlc.v_RLC_ilv,&rlc.v_RLC_i2v,&rlc.v_RLC_vv2,&rlc.v_RLC_vj1,&rlc.v_RLC_vj2,&rlc.v_RLC_vvd,&rlc.v_RLC_ib0,&rlc.v_RLC_b1,&rlc.v_RLC_b2};
		for (int kind=0;kind<9;++kind) { *targets[kind]=new float[nodes.size()]; std::copy(values[kind].begin(),values[kind].end(),*targets[kind]); }
		return true;
	}
	bool BuildTFSF(Operator_Ext_TFSF& source, const double origin[3]) override {
		source.m_maxDelay=0;
		for (int axis=0;axis<3;++axis) for (int side=0;side<2;++side) {
			if (!source.m_ActiveDir[axis][side]) continue;
			const int a=(axis+1)%3,b=(axis+2)%3;
			const uint32_t count=source.m_numLines[a]*source.m_numLines[b];
			std::vector<double> metadata(23); metadata[0]=axis; metadata[1]=side;
			for (int n=0;n<3;++n) {
				metadata[2+n]=source.m_Start[n]; metadata[5+n]=source.m_Stop[n]; metadata[8+n]=origin[n];
				metadata[11+n]=source.m_PropDir[n]; metadata[14+n]=source.m_E_Amp[n]; metadata[17+n]=source.m_H_Amp[n];
			}
			metadata[20]=source.m_PhVel; metadata[21]=op.GetGridDelta(); SetAuxiliary(metadata);  Run(18,count);
			std::vector<float> points(12ull*count); Download(18,points.data(),points.size()*sizeof(float));
			for (unsigned int part=0;part<4;++part) {
				const int component=part%2;
				unsigned int*& delays=part<2?source.m_CurrDelay[axis][side][component]:source.m_VoltDelay[axis][side][component];
				float*& deltas=part<2?source.m_CurrDelayDelta[axis][side][component]:source.m_VoltDelayDelta[axis][side][component];
				float*& amplitudes=part<2?source.m_CurrAmp[axis][side][component]:source.m_VoltAmp[axis][side][component];
				delays=new unsigned int[count]; deltas=new float[count]; amplitudes=new float[count];
				for (uint32_t q=0;q<count;++q) {
					const float* point=&points[(part*count+q)*3]; std::memcpy(&delays[q],point,sizeof(uint32_t));
					deltas[q]=point[1]; amplitudes[q]=point[2]; source.m_maxDelay=std::max(source.m_maxDelay,delays[q]);
				}
			}
		}
		++source.m_maxDelay; return true;
	}
	bool BuildMur(Operator_Ext_Mur_ABC& mur) {
		if (mur.m_ny<0) return false;
		unsigned int p[3]={0,0,0}; p[mur.m_ny]=mur.m_LineNr;
		const double delta=std::abs(op.GetEdgeLength(mur.m_ny,p));
		std::vector<double> metadata(6+mur.m_numLines[0]);
		metadata[0]=mur.m_ny; metadata[1]=mur.m_LineNr; metadata[2]=mur.m_v_phase;
		metadata[3]=op.GetDiscLine(mur.m_ny,mur.m_LineNr)+(mur.m_LineNr==0?delta:-delta)/2/op.GetGridDelta();
		metadata[4]=delta;
		for (unsigned int row=0;row<mur.m_numLines[0];++row) {
			metadata[5+row]=double(metadata.size());
			int bounds[3]={-1,-1,-1}; bounds[mur.m_ny]=mur.m_LineNr; bounds[mur.m_nyP]=row;
			for (CSPrimitives* prim:op.GetPrimitivesBoundBox(bounds[0],bounds[1],bounds[2],CSProperties::MATERIAL))
				metadata.push_back(indices.at(prim));
		}
		metadata[5+mur.m_numLines[0]]=double(metadata.size()); Upload(17,metadata.data(),metadata.size()*sizeof(double));
		const uint32_t count=mur.m_numLines[0]*mur.m_numLines[1];  Run(10,count);
		Download(18,mur.m_Mur_Coeff_nyP.data(),size_t(count)*sizeof(float));
		Download(18,mur.m_Mur_Coeff_nyPP.data(),size_t(count)*sizeof(float),size_t(count)*sizeof(float));

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

		for (unsigned int magnetic=0;magnetic<2;++magnetic) {
			 Run(5,magnetic); Run(6);
			double pointCount=0;
			Download(16,&pointCount,sizeof(pointCount),((3ull*cells+255)/256)*sizeof(double));
			const uint32_t count=uint32_t(pointCount);
			 Run(7,magnetic);
			std::vector<SourcePoint> points(count);
			if (count) Download(18,points.data(),points.size()*sizeof(SourcePoint));
			std::vector<unsigned int> positions[3],delays,directions;
			std::vector<float> amplitudes;
			for (const auto& point:points) {
				const uint32_t node=point.index%cells;
				directions.push_back(point.index/cells); delays.push_back(point.delay); amplitudes.push_back(point.amplitude);
				positions[0].push_back(node/(op.numLines[1]*op.numLines[2]));
				positions[1].push_back((node/op.numLines[2])%op.numLines[1]); positions[2].push_back(node%op.numLines[2]);
			}
			if (magnetic) source.setupCurrentExcitation(positions,amplitudes,delays,directions);
			else source.setupVoltageExcitation(positions,amplitudes,delays,directions);
		}
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
	const unsigned int stores=4;
	const size_t cells=size_t(model.dims[0])*model.dims[1]*model.dims[2];
	const size_t estimate=cells*3*(8+stores)*sizeof(float)+((3*cells+255)/256+1)*sizeof(double)+setup->ExtensionBytes()+
	                      (model.grid.size()+model.geometry.size())*sizeof(double)+
	                      (model.candidates.size()+model.primitive_count+4)*sizeof(unsigned int)+64ull*1024*1024;
	if (const char* cap=std::getenv("OPENEMS_GPU_SETUP_MEMORY_LIMIT")) {
		bool digits=*cap!=0;
		for (const char* p=cap;*p;++p) digits=digits && *p>='0' && *p<='9';
		char* end=NULL; errno=0; unsigned long long limit=std::strtoull(cap,&end,10);
		if (!digits || end==cap || *end || errno==ERANGE) throw std::invalid_argument("OPENEMS_GPU_SETUP_MEMORY_LIMIT must be an unsigned byte count");
		available=static_cast<size_t>(std::min<unsigned long long>(available,limit));
	}
	if (estimate>available-available/5) { reason="insufficient device memory budget for operator construction"; return NULL; }
	setup->Prepare(model);
	return setup.release();
}
