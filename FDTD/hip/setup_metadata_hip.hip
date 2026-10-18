/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_internal.h"
#include "setup_metadata_hip.h"
#include "FDTD/gpu_setup_phase_velocity.h"
#include "tools/constants.h"

namespace
{
__global__ void Metadata(unsigned int mode,GPU_SetupMeshView mesh,const double* input,double* output,double dt,bool cylindrical,bool closed)
{
	if (mode==0)
	{
		uint32_t start[3]={},stop[3]={};
		output[0]=GPU_SetupSnapBox(mesh,input,input+3,start,stop,input[6]!=0,int(input[8]),cylindrical,closed,input[7]!=0);
		for (int n=0;n<3;++n) {output[1+n]=start[n];output[4+n]=stop[n];}
	}
	else if (mode==1)
	{
		double spacing[3];
		for (int n=0;n<3;++n)
			spacing[n]=fabs(mesh.Line(n,uint32_t(input[n]))-mesh.Line(n,uint32_t(input[3+n])))*input[10]/fabs(input[3+n]-input[n]);
		output[0]=GPU_SetupPhaseVelocity(spacing,input+6,float(input[9]),dt,input[11]);
	}
	else if (mode==2)
	{
		const double period=round(input[0]/dt);
		output[0]=isfinite(period) && period>=1 && period<=double(UINT32_MAX/2) ? period : 0;
		output[1]=period*dt;
	}
	else if (mode==3)
	{
		const double pi=3.1415926535897932384626433832795,c0=C0,z0=Z0;
		double direction[3]={input[6],input[7],input[8]},electric[3]={input[9],input[10],input[11]},spacing[3];
		const double norm=sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2]);
		for (int n=0;n<16;++n) output[n]=0;
		if (!norm) {output[14]=1;return;}
		for (int n=0;n<3;++n)
		{
			direction[n]/=norm;
			spacing[n]=fabs(mesh.Line(n,uint32_t(input[n]))-mesh.Line(n,uint32_t(input[3+n])))*input[15]/fabs(input[3+n]-input[n]);
		}
		const double background=c0/sqrt(input[13]*input[14]);
		double velocity=input[12]<=0 ? background : GPU_SetupPhaseVelocity(spacing,direction,float(input[12]),dt,background);
		if (velocity<0 || velocity>background || isnan(velocity)) {velocity=background;output[15]=1;}
		const double dot=electric[0]*direction[0]+electric[1]*direction[1]+electric[2]*direction[2];
		const double angle=acos(dot/(electric[0]*electric[0]+electric[1]*electric[1]+electric[2]*electric[2]))/pi*180;
		output[13]=angle;if (angle==0) {output[14]=2;return;}
		if (angle!=90) for (int n=0;n<3;++n) electric[n]-=direction[n]*dot;
		for (int n=0;n<3;++n)
		{
			const int a=(n+1)%3,b=(n+2)%3;
			output[n]=direction[n];output[3+n]=electric[n];
			output[6+n]=(direction[a]*electric[b]-direction[b]*electric[a])/(z0*sqrt(input[14]/input[13]));
			const uint32_t origin=direction[n]>=0 ? uint32_t(input[n])-1 : uint32_t(input[3+n])+1;
			output[9+n]=origin<mesh.dims[n] ? mesh.Line(n,origin) : 0;
		}
		output[12]=velocity;
	}
}
}

void HIP_BuildSetupMetadata(unsigned int mode,GPU_SetupMeshView mesh,const double* input,double* output,double dt,hipStream_t stream,bool cylindrical,bool closed)
{
	Metadata<<<1,1,0,stream>>>(mode,mesh,input,output,dt,cylindrical,closed);
	HIP_Check(hipGetLastError(),"source/lumped numerical metadata");
}
