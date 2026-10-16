/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "operator_gpu.h"
#include "gpu_backend.h"
#include "CSPrimBox.h"
#include "CSPrimSphere.h"
#include "CSPrimCylinder.h"
#include "CSPrimLinPoly.h"
#include "CSPropMaterial.h"
#include "CSPropExcitation.h"
#include "CSPropConductingSheet.h"
#include "CSTransform.h"
#include "tools/setup_timer.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>

static void RequireSetup(bool valid, const std::string& reason)
{
	if (!valid) throw GPU_SetupUnavailable(reason);
}

static bool ConstantWeight(const std::string& term)
{
	if (term.empty()) return true;
	char* end=NULL;
	const double value=std::strtod(term.c_str(), &end);
	while (end && (*end==' ' || *end=='\t')) ++end;
	return end!=term.c_str() && end && !*end && std::isfinite(value);
}

GPU_SetupModel Operator_GPU::SetupModel()
{
	GPU_SetupModel model;
	model.op=this;
	const size_t limit=std::numeric_limits<unsigned int>::max();
	size_t cells=1;
	RequireSetup(CSX->GetCoordInputType()!=CYLINDRICAL,"cylindrical geometry input requires CPU setup");
	RequireSetup(std::isfinite(gridDelta) && gridDelta>0, "invalid mesh unit");
	for (int n=0;n<3;++n)
	{
		RequireSetup(numLines[n]>=2 && cells<=limit/3/numLines[n], "construction index range exceeded");
		cells*=numLines[n]; model.dims[n]=numLines[n];
		model.offsets[n]=static_cast<unsigned int>(model.grid.size());
		for (unsigned int p=0;p<numLines[n];++p)
		{
			RequireSetup(std::isfinite(GetDiscLine(n,p)) && GetDiscDelta(n,p)>0, "non-finite or unordered mesh");
			model.grid.push_back(GetDiscLine(n,p)); model.grid.push_back(GetDiscLine(n,p,true));
			model.grid.push_back(GetRawDiscDelta(n,p)); model.grid.push_back(GetRawDiscDelta(n,int(p)-1));
			model.grid.push_back(GetDiscDelta(n,p)*gridDelta); model.grid.push_back(GetDiscDelta(n,p,true)*gridDelta);
		}
	}
	model.average=static_cast<unsigned int>(m_MatAverageMethod);
	RequireSetup(model.average<=1,"unsupported material averaging method");
	for (double value:model.grid) RequireSetup(std::isfinite(value),"non-finite mesh descriptor");
	model.stores=0;
	for (int n=0;n<4;++n) if (m_StoreMaterial[n]) model.stores|=1u<<n;
	std::copy(m_BC,m_BC+6,model.boundary);
	for (int n=0;n<6;++n) model.bc_size[n]=GetBCSize(n);
	m_SetupPrimitives=CSX->GetAllPrimitives(false,CSProperties::PropertyType(CSProperties::MATERIAL|CSProperties::METAL|CSProperties::EXCITATION));
	model.primitives=m_SetupPrimitives;
	RequireSetup(m_SetupPrimitives.size()<limit/48-1,"primitive index range exceeded");
	model.primitive_count=static_cast<unsigned int>(m_SetupPrimitives.size()+1);
	model.geometry.resize(size_t(model.primitive_count)*48,0);
	const double background[4]={GetBackgroundEpsR(),GetBackgroundKappa(),GetBackgroundMueR(),GetBackgroundSigma()};
	for (int n=0;n<3;++n) std::copy(background,background+4,model.geometry.begin()+32+n*4);
	std::vector<double> vertices;
	std::map<CSPrimitives*,unsigned int> indices;
	for (size_t i=0;i<m_SetupPrimitives.size();++i)
	{
		CSPrimitives* prim=m_SetupPrimitives[i]; indices[prim]=static_cast<unsigned int>(i+1);
		RequireSetup(!(prim->GetProperty()->GetType()&CSProperties::DISCRETE_MATERIAL),"discrete materials require CPU setup");
		double* data=&model.geometry[(i+1)*48]; data[4]=data[9]=data[14]=1;
		RequireSetup(prim->GetCoordinateSystem()!=CYLINDRICAL,"cylindrical primitive coordinates require CPU setup");
		if (prim->HasTransform())
		{
			CSTransform inverse(prim->GetTransform()); double* m=inverse.GetMatrix();
			RequireSetup(m[12]==0 && m[13]==0 && m[14]==0 && m[15]==1,"non-affine primitive transform");
			double determinant=m[0]*(m[5]*m[10]-m[6]*m[9])-m[1]*(m[4]*m[10]-m[6]*m[8])+m[2]*(m[4]*m[9]-m[5]*m[8]);
			RequireSetup(std::isfinite(determinant) && determinant!=0,"singular primitive transform");
			inverse.Invert(); std::copy(inverse.GetMatrix(),inverse.GetMatrix()+12,data+4);
		}
		const double *start=NULL,*stop=NULL;
		if (prim->GetType()==CSPrimitives::BOX)
		{
			CSPrimBox* box=static_cast<CSPrimBox*>(prim); data[0]=1;
			start=box->GetStartCoord()->GetCartesianCoords(); stop=box->GetStopCoord()->GetCartesianCoords();
		}
		else if (prim->GetType()==CSPrimitives::SPHERE)
		{
			CSPrimSphere* sphere=static_cast<CSPrimSphere*>(prim); data[0]=2;
			start=stop=sphere->GetCenter()->GetCartesianCoords(); data[22]=sphere->GetRadius();
		}
		else if (prim->GetType()==CSPrimitives::CYLINDER)
		{
			CSPrimCylinder* cylinder=static_cast<CSPrimCylinder*>(prim); data[0]=3;
			start=cylinder->GetAxisStartCoord()->GetCartesianCoords(); stop=cylinder->GetAxisStopCoord()->GetCartesianCoords();
			data[22]=cylinder->GetRadius(); cylinder->GetBoundBox(data+23);
			RequireSetup(start[0]!=stop[0] || start[1]!=stop[1] || start[2]!=stop[2],"zero-length cylinder");
		}
		else if (prim->GetType()==CSPrimitives::LINPOLY)
		{
			CSPrimLinPoly* poly=static_cast<CSPrimLinPoly*>(prim); data[0]=4;
			const size_t count=poly->GetQtyCoords(), offset=model.geometry.size()+vertices.size();
			RequireSetup(poly->GetNormDir()>=0 && poly->GetNormDir()<3,"invalid LinPoly normal");
			RequireSetup(offset<=limit && count<=(limit-offset)/2,"polygon vertex index range exceeded");
			data[16]=poly->GetNormDir(); data[17]=double(offset); data[18]=double(count); poly->GetBoundBox(data+23);
			for (size_t v=0;v<count;++v) { vertices.push_back(poly->GetCoord(int(2*v))); vertices.push_back(poly->GetCoord(int(2*v+1))); }
		}
		else throw GPU_SetupUnavailable("unsupported material/metal primitive or source: "+prim->GetTypeName());
		if (start) { std::copy(start,start+3,data+16); std::copy(stop,stop+3,data+19); }
		data[1]=prim->GetProperty()->GetType()==CSProperties::METAL?1:0;
		CSPropMaterial* mat=dynamic_cast<CSPropMaterial*>(prim->GetProperty());
		for (int n=0;n<3;++n)
		{
			std::copy(background,background+4,data+32+n*4);
			if (!mat) continue;
			RequireSetup(ConstantWeight(mat->GetEpsilonWeightFunction(n)) && ConstantWeight(mat->GetKappaWeightFunction(n)) &&
			             ConstantWeight(mat->GetMueWeightFunction(n)) && ConstantWeight(mat->GetSigmaWeightFunction(n)),"spatial material weights require CPU setup");
			double coords[3]={0,0,0};
			data[32+n*4]=mat->GetEpsilonWeighted(n,coords); data[33+n*4]=mat->GetKappaWeighted(n,coords);
			data[34+n*4]=mat->GetMueWeighted(n,coords); data[35+n*4]=mat->GetSigmaWeighted(n,coords);
		}
		if (auto* source=dynamic_cast<CSPropExcitation*>(prim->GetProperty()))
		{
			RequireSetup(source->GetWeightFile().empty(),"file-weighted sources require CPU setup");
			RequireSetup(std::isfinite(source->GetDelay()) && source->GetDelay()>=0,"invalid source delay");
			data[2]=source->GetEnabled()?1:0; data[3]=source->GetExcitType(); data[30]=source->GetDelay();
			for (int n=0;n<3;++n)
			{
				RequireSetup(ConstantWeight(source->GetWeightFunction(n)),"spatial source weights require CPU setup");
				double coords[3]={0,0,0}; data[32+n*4]=source->GetWeightedExcitation(n,coords);
				data[33+n*4]=source->GetActiveDir(n)?1:0;
			}
		}
		if (auto* sheet=dynamic_cast<CSPropConductingSheet*>(prim->GetProperty()))
		{
			data[29]=1; data[31]=prim->GetDimension(); prim->GetBoundBox(data+23);
			data[32]=sheet->GetConductivity(); data[33]=sheet->GetThickness();
		}
	}
	model.geometry.insert(model.geometry.end(),vertices.begin(),vertices.end());
	for (double value:model.geometry) RequireSetup(std::isfinite(value),"non-finite geometry/material descriptor");
	const size_t rows=size_t(numLines[0])*numLines[1];
	const size_t sourceRows=size_t(numLines[1])*numLines[2];
	RequireSetup(rows<=(limit-3-sourceRows)/2,"candidate index range exceeded");
	model.candidates.resize(2*(rows+1)+sourceRows+1);
	for (int which=0;which<2;++which)
	{
		for (unsigned int x=0;x<numLines[0];++x) for (unsigned int y=0;y<numLines[1];++y)
		{
			model.candidates[which*(rows+1)+size_t(x)*numLines[1]+y]=static_cast<unsigned int>(model.candidates.size());
			auto list=GetPrimitivesBoundBox(x,y,-1,which==0?CSProperties::MATERIAL:CSProperties::PropertyType(CSProperties::MATERIAL|CSProperties::METAL));
			RequireSetup(list.size()<=limit-model.candidates.size(),"candidate index range exceeded");
			for (CSPrimitives* prim:list) model.candidates.push_back(indices.at(prim));
		}
		model.candidates[which*(rows+1)+rows]=static_cast<unsigned int>(model.candidates.size());
	}
	for (unsigned int z=0;z<numLines[2];++z) for (unsigned int y=0;y<numLines[1];++y)
	{
		model.candidates[2*(rows+1)+size_t(z)*numLines[1]+y]=static_cast<unsigned int>(model.candidates.size());
		auto list=GetPrimitivesBoundBox(-1,y,z,CSProperties::EXCITATION);
		RequireSetup(list.size()<=limit-model.candidates.size(),"source candidate index range exceeded");
		for (CSPrimitives* prim:list) model.candidates.push_back(indices.at(prim));
	}
	model.candidates[2*(rows+1)+sourceRows]=static_cast<unsigned int>(model.candidates.size());
	return model;
}

int Operator_GPU::CalcECOperator(DebugFlags flags)
{
	m_Setup.reset(); m_SetupBackend.reset();
	if (m_SetupMode!="cpu")
	{
		try
		{
			SetupTimer timer("GPU construction");
			GPU_SetupModel model=SetupModel();
			m_SetupBackend.reset(GPU_Backend::New(m_Backend));
			std::string reason;
			m_Setup.reset(m_SetupBackend->NewOperatorSetup(model,reason));
			if (!m_Setup) throw GPU_SetupUnavailable(reason);
			std::cout << "Operator_GPU: HIP construction active (materials, timestep, coefficients, PEC/PMC and extensions)" << std::endl;
			timer.Report("geometry, preflight and allocation");
		}
		catch (const std::bad_alloc&) { throw GPU_SetupUnavailable("insufficient host memory for operator construction preparation"); }
	}
	const int result=Operator::CalcECOperator(flags);
	m_Setup.reset();
	return result;
}

double Operator_GPU::CalcTimestep()
{
	if (!m_Setup) return Operator::CalcTimestep();
	dT=m_Setup->CalcTimestep(m_TimeStepVar);
	m_Used_TS_Name=m_TimeStepVar==3?"Rennings_2":"Rennings_1";
	MainOp->SetReflection2Cell();
	return 0;
}
