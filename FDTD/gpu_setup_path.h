/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_PATH_H
#define GPU_SETUP_PATH_H

#include "gpu_setup_geometry.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_PATH __device__
#else
#define GPU_SETUP_PATH
#endif

struct GPU_SetupMeshView
{
	uint32_t dims[3], offsets[3];
	const double* grid;
	GPU_SETUP_PATH double Line(int axis, uint32_t position, int kind=0) const {return grid[offsets[axis]+6*position+kind];}
	GPU_SETUP_PATH uint32_t Snap(int axis, double coordinate, bool dual=false) const
	{
		if (coordinate<Line(axis,0)) return 0;
		if (coordinate>Line(axis,dims[axis]-1)) return dims[axis]-1;
		uint32_t lo=dual ? 1 : 0,hi=dims[axis];
		while (lo<hi)
		{
			const uint32_t mid=lo+(hi-lo)/2;
			if (coordinate<=Line(axis,mid,dual ? 0 : 1)) hi=mid;
			else lo=mid+1;
		}
		return lo<dims[axis] ? lo-(dual ? 1 : 0) : 0;
	}
};

GPU_SETUP_PATH inline double GPU_SetupFitAlpha(const GPU_SetupMeshView& mesh,double angle,bool closed,bool full=false)
{
	const double first=mesh.Line(1,0),last=mesh.Line(1,mesh.dims[1]-1-(closed && !full ? 2 : 0));
	const double period=6.283185307179586476925286766559;
	if (angle>=first && angle<=last) return angle;
	while (angle<first) {angle+=period;if (angle>last) return angle-period;if (angle>first) return angle;}
	while (angle>last) {angle-=period;if (angle<first) return angle+period;if (angle<last) return angle;}
	return angle;
}

GPU_SETUP_PATH inline uint32_t GPU_SetupSnapCoordinate(const GPU_SetupMeshView& mesh,int axis,double coord,bool dual,bool full,bool cylindrical,bool closed)
{
	GPU_SetupMeshView range=mesh;
	if (cylindrical && axis==1) coord=GPU_SetupFitAlpha(mesh,coord,closed);
	if (closed && !full) range.dims[1]-=2;
	return range.Snap(axis,coord,dual);
}

GPU_SETUP_PATH inline int GPU_SetupSnapBox(const GPU_SetupMeshView& mesh, const double* start, const double* stop,
                                          uint32_t first[3], uint32_t last[3], bool dual=false, int method=0,
                                          bool cylindrical=false,bool closed=false,bool full=true,bool* first_inside=nullptr,bool* last_inside=nullptr)
{
	double lower[3],upper[3];
	GPU_SetupMeshView range=mesh;if (closed && !full) range.dims[1]-=2;
	for (int n=0;n<3;++n)
	{
		lower[n]=fmin(start[n],stop[n]);upper[n]=fmax(start[n],stop[n]);
		if (cylindrical && n==1)
		{
			const double size=stop[1]-start[1],center=GPU_SetupFitAlpha(mesh,0.5*(stop[1]+start[1]),closed);
			const double a=fmax(range.Line(1,0),fmin(range.Line(1,range.dims[1]-1),center-size/2));
			const double b=fmax(range.Line(1,0),fmin(range.Line(1,range.dims[1]-1),(center-size/2)+size));
			lower[n]=fmin(a,b);upper[n]=fmax(a,b);
		}
		if (upper[n]<range.Line(n,0) || lower[n]>range.Line(n,range.dims[n]-1)) return -2;
	}
	int dimension=0;
	for (int n=0;n<3;++n)
	{
		const double a=cylindrical && n==1 ? GPU_SetupFitAlpha(mesh,lower[n],closed) : lower[n];
		const double b=cylindrical && n==1 ? GPU_SetupFitAlpha(mesh,upper[n],closed) : upper[n];
		if (first_inside) first_inside[n]=a>=range.Line(n,0) && a<=range.Line(n,range.dims[n]-1);
		if (last_inside) last_inside[n]=b>=range.Line(n,0) && b<=range.Line(n,range.dims[n]-1);
		first[n]=GPU_SetupSnapCoordinate(mesh,n,lower[n],dual,full,cylindrical,closed);last[n]=GPU_SetupSnapCoordinate(mesh,n,upper[n],dual,full,cylindrical,closed);
		if (first[n]<last[n])
		{
			if (method==1)
			{
				if (mesh.Line(n,first[n],dual ? 1 : 0)>lower[n] && first[n]>0) --first[n];
				if (mesh.Line(n,last[n],dual ? 1 : 0)<upper[n] && last[n]+1<range.dims[n]) ++last[n];
			}
			else if (method==2)
			{
				if (mesh.Line(n,first[n],dual ? 1 : 0)<lower[n] && first[n]+1<range.dims[n]) ++first[n];
				if (mesh.Line(n,last[n],dual ? 1 : 0)>upper[n] && last[n]>0) --last[n];
			}
		}
		if (last[n]>first[n]) ++dimension;
	}
	return method>=0 && method<=2 ? dimension : -1;
}

GPU_SETUP_PATH inline double GPU_SetupPathDistance(const double* p,const double* start,const double* stop,double& foot,bool cylindrical)
{
	double point[3]={p[0],p[1],p[2]},a[3]={start[0],start[1],start[2]},b[3]={stop[0],stop[1],stop[2]};
	if (cylindrical) {GPU_SetupConvertCoordinates(point,1,0);GPU_SetupConvertCoordinates(a,1,0);GPU_SetupConvertCoordinates(b,1,0);}
	return GPU_SetupLineDistance(point,a,b,foot);
}

// Mirrors Operator::FindPath, including direction tie order and the final
// closing edges. Call once with a counting visitor before emitting writes:
// an out-of-domain path discards the entire segment on the CPU.
template<class Visitor>
GPU_SETUP_PATH bool GPU_SetupFindMeshPath(const GPU_SetupMeshView& mesh,const double* start,const double* stop,Visitor& visit,bool cylindrical,bool closed)
{
	uint32_t current[3],end[3];
	double a[3],b[3];
	for (int n=0;n<3;++n)
	{
		if ((start[n]<mesh.Line(n,0) && stop[n]<mesh.Line(n,0)) ||
		    (start[n]>mesh.Line(n,mesh.dims[n]-1) && stop[n]>mesh.Line(n,mesh.dims[n]-1))) return false;
		current[n]=GPU_SetupSnapCoordinate(mesh,n,start[n],false,true,cylindrical,closed);end[n]=GPU_SetupSnapCoordinate(mesh,n,stop[n],false,true,cylindrical,closed);
		if (cylindrical && n==1)
		{
			const uint32_t last=mesh.dims[1]-1-(closed ? 1 : 0);
			if (stop[1]>start[1] && end[1]<current[1] && end[1]==0) end[1]=last;
			if (stop[1]<start[1] && end[1]>current[1] && end[1]==last) end[1]=0;
		}
		a[n]=mesh.Line(n,current[n]); b[n]=mesh.Line(n,end[n]);
	}
	double start_foot,stop_foot;
	GPU_SetupPathDistance(a,start,stop,start_foot,cylindrical);
	GPU_SetupPathDistance(b,start,stop,stop_foot,cylindrical);
	double current_foot=start_foot,minimum_foot=start_foot;
	uint64_t iterations=0,limit=uint64_t(mesh.dims[0])*mesh.dims[1]*mesh.dims[2];
	int direction=0;
	bool up=false;
	while (minimum_foot<stop_foot)
	{
		if (++iterations>limit) return false;
		double minimum_distance=1e300;
		for (int n=0;n<3;++n)
		{
			double point[3]={mesh.Line(0,current[0]),mesh.Line(1,current[1]),mesh.Line(2,current[2])};
			if (current[n]>0)
			{
				point[n]=mesh.Line(n,current[n]-1);
				double foot,distance=GPU_SetupPathDistance(point,start,stop,foot,cylindrical);
				if (foot>current_foot && distance<minimum_distance)
				{ minimum_foot=foot; minimum_distance=distance; direction=n; up=false; }
			}
			if (current[n]+1<mesh.dims[n])
			{
				point[n]=mesh.Line(n,current[n]+1);
				double foot,distance=GPU_SetupPathDistance(point,start,stop,foot,cylindrical);
				if (foot>current_foot && distance<minimum_distance)
				{ minimum_foot=foot; minimum_distance=distance; direction=n; up=true; }
			}
		}
		uint32_t position[3]={current[0],current[1],current[2]};
		if (up) ++current[direction];
		else { --current[direction]; --position[direction]; }
		for (int n=0;n<3;++n) if (current[n]>=mesh.dims[n]) return false;
		visit(direction,position);
		current_foot=minimum_foot;
	}
	for (int n=0;n<3;++n)
	{
		if (current[n]>end[n]) { --current[n]; visit(n,current); }
		else if (current[n]<end[n]) visit(n,current);
	}
	return true;
}

GPU_SETUP_PATH inline int GPU_SetupAlphaIntersection(const GPU_SetupMeshView& mesh,double angle,const double* start,const double* stop,double* intersection)
{
	double plane[3][3];for (int p=0;p<3;++p) for (int n=0;n<3;++n) plane[p][n]=mesh.Line(n,0);
	for (int p=0;p<3;++p) plane[p][1]=angle;
	plane[1][0]=mesh.Line(0,mesh.dims[0]-1);plane[2][2]=mesh.Line(2,mesh.dims[2]-1);
	double a[3]={start[0],start[1],start[2]},b[3]={stop[0],stop[1],stop[2]};
	GPU_SetupConvertCoordinates(a,1,0);GPU_SetupConvertCoordinates(b,1,0);
	for (int p=0;p<3;++p) GPU_SetupConvertCoordinates(plane[p],1,0);
	double m[9],inverse[9];
	for (int n=0;n<3;++n) {m[3*n]=a[n]-b[n];m[3*n+1]=plane[1][n]-plane[0][n];m[3*n+2]=plane[2][n]-plane[0][n];intersection[n]=0;}
	const double det=m[0]*m[4]*m[8]+m[1]*m[5]*m[6]+m[2]*m[3]*m[7]-m[2]*m[4]*m[6]-m[1]*m[3]*m[8]-m[0]*m[5]*m[7];
	if (fabs(det)<1e-50) return -1;
	inverse[0]=(m[4]*m[8]-m[5]*m[7])/det;inverse[1]=(m[2]*m[7]-m[1]*m[8])/det;inverse[2]=(m[1]*m[5]-m[2]*m[4])/det;
	inverse[3]=(m[5]*m[6]-m[3]*m[8])/det;inverse[4]=(m[0]*m[8]-m[2]*m[6])/det;inverse[5]=(m[2]*m[3]-m[0]*m[5])/det;
	inverse[6]=(m[3]*m[7]-m[4]*m[6])/det;inverse[7]=(m[1]*m[6]-m[0]*m[7])/det;inverse[8]=(m[0]*m[4]-m[1]*m[3])/det;
	double t=0,u=0,v=0;
	for (int n=0;n<3;++n) {t+=inverse[n]*(a[n]-plane[0][n]);u+=inverse[3+n]*(a[n]-plane[0][n]);v+=inverse[6+n]*(a[n]-plane[0][n]);}
	for (int n=0;n<3;++n) intersection[n]=a[n]*(1-t)+b[n]*t;
	GPU_SetupConvertCoordinates(intersection,0,1);
	return u<0 || u>1 || v<0 || v>1 ? 1 : t<0 || t>1 ? 2 : 0;
}

struct GPU_SetupPathCounter {uint64_t count;GPU_SETUP_PATH void operator()(int,const uint32_t*) {++count;}};
template<class Visitor> struct GPU_SetupSeamVisitor
{
	Visitor& output;uint32_t seam;bool closed;
	GPU_SETUP_PATH void operator()(int n,const uint32_t* p)
	{
		uint32_t position[3]={p[0],p[1],p[2]};if (closed && n!=1 && position[1]==0) position[1]=seam;
		output(n,position);
	}
};

template<class Visitor>
GPU_SETUP_PATH bool GPU_SetupFindPath(const GPU_SetupMeshView& mesh,const double* start,const double* stop,Visitor& visit,bool cylindrical=false,bool closed=false)
{
	if (!cylindrical) return GPU_SetupFindMeshPath(mesh,start,stop,visit,false,false);
	const double pi=3.1415926535897932384626433832795;
	double a[3]={start[0],start[1],start[2]},b[3]={stop[0],stop[1],stop[2]};
	while (fabs(b[1]-a[1])>pi) b[1]+=b[1]>a[1] ? -2*pi : 2*pi;
	if (a[1]>b[1]) for (int n=0;n<3;++n) {const double value=a[n];a[n]=b[n];b[n]=value;}
	const double first=GPU_SetupFitAlpha(mesh,a[1],closed),last=GPU_SetupFitAlpha(mesh,b[1],closed);
	if (last>=first) {a[1]=first;b[1]=last;return GPU_SetupFindMeshPath(mesh,a,b,visit,true,closed);}
	for (int n=0;n<3;++n)
		if ((a[n]<mesh.Line(n,0) && b[n]<mesh.Line(n,0)) || (a[n]>mesh.Line(n,mesh.dims[n]-1) && b[n]>mesh.Line(n,mesh.dims[n]-1))) return false;
	const double upper=mesh.Line(1,mesh.dims[1]-1-(closed ? 1 : 0));double intersection[3];
	int result=GPU_SetupAlphaIntersection(mesh,upper,a,b,intersection);if (result<0) return false;
	GPU_SetupSeamVisitor<Visitor> output={visit,mesh.dims[1]-2,closed};
	if (result==0)
	{
		intersection[1]=upper;a[1]=first;GPU_SetupPathCounter count={0};
		if (GPU_SetupFindMeshPath(mesh,a,intersection,count,true,closed)) GPU_SetupFindMeshPath(mesh,a,intersection,output,true,closed);
	}
	if (!closed) result=GPU_SetupAlphaIntersection(mesh,mesh.Line(1,0),a,b,intersection);
	if (result==0)
	{
		intersection[1]=mesh.Line(1,0);b[1]=last;GPU_SetupPathCounter count={0};
		if (GPU_SetupFindMeshPath(mesh,intersection,b,count,true,closed)) GPU_SetupFindMeshPath(mesh,intersection,b,output,true,closed);
	}
	return true;
}

#undef GPU_SETUP_PATH
#endif
