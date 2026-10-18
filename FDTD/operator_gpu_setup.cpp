/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "gpu_setup_controller.h"
#include "gpu_setup_upml.h"
#include "operator_cylinder.h"
#include "operator_cylindermultigrid.h"
#include "gpu_backend.h"
#include "gpu_setup_geometry_export.h"
#include "gpu_setup_constant.h"
#include "gpu_setup_input_export.h"
#include "gpu_setup_adapters.h"
#include "excitation.h"
#include "extensions/operator_ext_upml.h"
#include "fparser.hh"
#include "CSPrimBox.h"
#include "CSPrimSphere.h"
#include "CSPrimCylinder.h"
#include "CSPrimLinPoly.h"
#include "CSPropMaterial.h"
#include "CSPropExcitation.h"
#include "CSPropConductingSheet.h"
#include "CSPropMetal.h"
#include "CSPropLorentzMaterial.h"
#include "CSPropDebyeMaterial.h"
#include "CSPropDiscMaterial.h"
#include "CSTransform.h"
#include "tools/setup_timer.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <typeinfo>

static void RequireSetup(bool valid, const std::string& reason)
{
	if (!valid) throw GPU_SetupUnavailable(reason);
}

static bool ConstantWeight(const std::string& term)
{
	return GPU_SetupConstant(term);
}

GPU_SetupController::GPU_SetupController(Operator& owner) : op(owner) {}
GPU_SetupController::~GPU_SetupController() {}
GPU_Backend* GPU_SetupController::ReleaseBackend() {return m_SetupBackend.release();}

GPU_SetupModel GPU_SetupController::SetupModel()
{
	op.CheckSetupCancellation();
	auto* CSX=op.CSX;
	auto* m_Exc=op.m_Exc;
	const auto* numLines=op.numLines;
	const double gridDelta=op.gridDelta;
	const auto m_MatAverageMethod=op.m_MatAverageMethod;
	const auto* m_StoreMaterial=op.m_StoreMaterial;
	const auto* m_BC=op.m_BC;
	GPU_SetupModel model;
	model.cell_materials=op.m_StoreCellMaterials;
	model.op=&op;
	model.grid_unit=gridDelta;model.cylindrical=op.m_MeshType==CYLINDRICAL;
	if (auto* cylinder=dynamic_cast<Operator_Cylinder*>(&op))
	{
		model.closed_alpha=cylinder->GetClosedAlpha();model.axis=cylinder->GetR0Included();
		if (auto* multigrid=dynamic_cast<Operator_CylinderMultiGrid*>(cylinder))
		{
			model.first_radius=multigrid->GetSplitPos()-1;
			const auto* child=multigrid->GetInnerOperator();
			for (unsigned int p=0;p<child->GetNumberOfLines(1,true);++p)
			{
				model.child_alpha.push_back(child->GetDiscLine(1,p));
				for (int k=0;k<5;++k) model.child_alpha.push_back(0);
			}
		}
	}
	RequireSetup(m_Exc && typeid(*m_Exc)==typeid(Excitation),"no device waveform adapter for custom excitation callback");
	const int waveform=m_Exc->GetExciteType();
	RequireSetup(waveform==Excitation::GaussianPulse || waveform==Excitation::Sinusoidal || waveform==Excitation::DiracPulse ||
	             waveform==Excitation::Step || waveform==Excitation::CustomExcite,"no device waveform adapter for excitation type");
	RequireSetup(waveform!=Excitation::CustomExcite,"custom waveforms require CPU operator construction");
	// Padded 256-thread dispatches must not wrap their 32-bit global index.
	const size_t limit=std::numeric_limits<unsigned int>::max()-255u;
	size_t cells=1;
	RequireSetup(std::isfinite(gridDelta) && gridDelta>0, "invalid mesh unit");
	for (int n=0;n<3;++n)
	{
		RequireSetup(numLines[n]>=2 && cells<=limit/3/numLines[n], "construction index range exceeded");
		cells*=numLines[n]; model.dims[n]=numLines[n];
		model.offsets[n]=static_cast<unsigned int>(model.grid.size());
		for (unsigned int p=0;p<numLines[n];++p)
		{
			RequireSetup(std::isfinite(op.GetDiscLine(n,p)) && (!p || op.GetDiscLine(n,p)>op.GetDiscLine(n,p-1)),"non-finite or unordered mesh");
			model.grid.push_back(op.GetDiscLine(n,p));for (int k=0;k<5;++k) model.grid.push_back(0);
		}
	}
	model.average=static_cast<unsigned int>(m_MatAverageMethod);
	RequireSetup(model.average<=1,"unsupported material averaging method");
	for (double value:model.grid) RequireSetup(std::isfinite(value),"non-finite mesh descriptor");
	model.stores=0;
	for (int n=0;n<4;++n) if (m_StoreMaterial[n]) model.stores|=1u<<n;
	std::copy(m_BC,m_BC+6,model.boundary);
	for (int n=0;n<6;++n) model.bc_size[n]=op.GetBCSize(n);
	m_SetupPrimitives=CSX->GetAllPrimitives(false,CSProperties::PropertyType(CSProperties::MATERIAL|CSProperties::METAL|CSProperties::EXCITATION));
	model.primitives=m_SetupPrimitives;
	RequireSetup(m_SetupPrimitives.size()<limit-4,"primitive index range exceeded");
	model.primitive_count=static_cast<unsigned int>(m_SetupPrimitives.size()+1);
	model.shapes.resize(model.primitive_count);
	model.properties.resize(model.primitive_count);
	const double background[4]={op.GetBackgroundEpsR(),op.GetBackgroundKappa(),op.GetBackgroundMueR(),op.GetBackgroundSigma()};
	for (int n=0;n<3;++n) std::copy(background,background+4,model.properties[0].values[n]);
	std::map<CSPrimitives*,unsigned int> indices;
	std::map<CSProperties*,GPU_PropertyAdapter> adapters;
	for (size_t n=0;n<CSX->GetQtyProperties();++n)
	{
		auto* property=CSX->GetProperty(n);
		adapters.emplace(property,GPU_PreparePropertyAdapter(*property));
	}
	std::map<CSProperties*,uint32_t> discrete_ids,file_ids;
	std::map<CSProperties*,std::pair<uint32_t,uint32_t>> dispersion_ids;
	for (size_t i=0;i<m_SetupPrimitives.size();++i)
	{
		op.CheckSetupCancellation();
		CSPrimitives* prim=m_SetupPrimitives[i]; indices[prim]=static_cast<unsigned int>(i+1);
		model.shapes[i+1]=GPU_DescribePrimitive(*prim,model.geometry,&model.expressions);
		GPU_SetupProperty& data=model.properties[i+1];
		data.id=prim->GetProperty()->GetUniqueID(); data.type=prim->GetProperty()->GetType();
		data.coordinate_system=prim->GetProperty()->GetCoordInputType()==1 ? 1 : 0;
		data.metal=prim->GetProperty()->GetType()==CSProperties::METAL?1:0;
		CSPropMaterial* mat=dynamic_cast<CSPropMaterial*>(prim->GetProperty());
		data.isotropic=mat ? (mat->GetIsotropy() ? 1u : 0u) : 1u;
		if (auto* discrete=dynamic_cast<CSPropDiscMaterial*>(mat))
		{
			uint32_t& id=discrete_ids[discrete];
			if (!id) id=GPU_ExportSetupDiscrete(*discrete,model);
			data.discrete=id;
		}
		for (int n=0;n<3;++n)
		{
			std::copy(background,background+4,data.values[n]);
			if (!mat) continue;
			const std::string terms[]={mat->GetEpsilonWeightFunction(n),mat->GetKappaWeightFunction(n),mat->GetMueWeightFunction(n),mat->GetSigmaWeightFunction(n)};
			double coords[3]={0,0,0};
			const double base[]={mat->GetEpsilon(n),mat->GetKappa(n),mat->GetMue(n),mat->GetSigma(n)};
			for (int k=0;k<4;++k)
			{
				if (ConstantWeight(terms[k]))
					data.values[n][k]=k==0 ? mat->CSPropMaterial::GetEpsilonWeighted(n,coords) : k==1 ? mat->CSPropMaterial::GetKappaWeighted(n,coords) :
					                 k==2 ? mat->CSPropMaterial::GetMueWeighted(n,coords) : mat->CSPropMaterial::GetSigmaWeighted(n,coords);
				else
				{
					data.values[n][k]=base[k];
					data.weights[n][k]=GPU_CompileSetupExpression(terms[k],"x,y,z,rho,r,alpha,theta",model.expressions);
				}
			}
		}
		if (mat)
		{
			auto descriptor=dispersion_ids.find(mat);
			if (descriptor==dispersion_ids.end())
			{
				GPU_ExportSetupDispersion(*mat,model,data);
				dispersion_ids[mat]=std::make_pair(data.dispersion_offset,data.dispersion_count);
			}
			else {data.dispersion_offset=descriptor->second.first;data.dispersion_count=descriptor->second.second;}
			if (ConstantWeight(mat->GetDensityWeightFunction()))
			{
				double coords[3]={0,0,0}; data.density=mat->CSPropMaterial::GetDensityWeighted(coords);
			}
			else
			{
				data.density=mat->GetDensity();
				data.density_weight=GPU_CompileSetupExpression(mat->GetDensityWeightFunction(),"x,y,z,rho,r,alpha,theta",model.expressions);
			}
			RequireSetup(std::isfinite(data.density),"non-finite material density");
		}
		if (auto* source=dynamic_cast<CSPropExcitation*>(prim->GetProperty()))
		{
			if (!source->GetWeightFile().empty())
			{
				uint32_t& id=file_ids[source];
				if (!id) id=GPU_ExportSetupFileWeight(*source,model);
				data.file_weight=id;
			}
			RequireSetup(std::isfinite(source->GetDelay()) && source->GetDelay()>=0,"invalid source delay");
			data.enabled=source->GetEnabled()?1:0; data.excitation_type=source->GetExcitType(); data.delay=source->GetDelay();
			for (int n=0;n<3;++n)
			{
				data.weight_origin[n]=source->GetWeightOrigin(n);
				if (data.file_weight) data.values[n][0]=source->GetExcitation(n);
				else if (ConstantWeight(source->GetWeightFunction(n)))
				{
					double coords[3]={0,0,0}; data.values[n][0]=source->GetWeightedExcitation(n,coords);
				}
				else
				{
					data.values[n][0]=source->GetExcitation(n);
					data.weights[n][0]=GPU_CompileSetupExpression(source->GetWeightFunction(n),"x,y,z,rho,r,alpha,theta",model.expressions);
				}
				data.active[n]=source->GetActiveDir(n)?1:0;
			}
		}
		if (auto* sheet=dynamic_cast<CSPropConductingSheet*>(prim->GetProperty()))
		{
			data.sheet=1; data.conductivity=sheet->GetConductivity(); data.thickness=sheet->GetThickness();
		}
		const auto& adapter=adapters.at(prim->GetProperty());
		if (adapter.export_data) adapter.export_data(*prim->GetProperty(),model,data);
		RequireSetup(data.id==prim->GetProperty()->GetUniqueID() && data.type==prim->GetProperty()->GetType(),
		             "property adapter changed stable identity or query mask");
		for (int n=0;n<3;++n) for (double value:data.values[n]) RequireSetup(std::isfinite(value),"non-finite material/source descriptor");
		RequireSetup(std::isfinite(data.conductivity) && std::isfinite(data.thickness),"non-finite conducting sheet descriptor");
		for (double value:data.weight_origin) RequireSetup(std::isfinite(value),"non-finite source weight origin");
	}
	model.geometry_index_nodes=model.geometry_index_scratch=0;
	for (size_t n=0;n<op.GetNumberOfExtentions();++n)
		if (auto* upml=dynamic_cast<Operator_Ext_UPML*>(op.GetExtension(n)))
			RequireSetup(GPU_SetupDefaultGrading(upml->GetGradingFunction()),"custom PML grading requires CPU operator construction");
	for (GPU_SetupPrimitive& shape:model.shapes)
	{
		if (!shape.index_leaves) continue;
		const size_t nodes=2ull*shape.index_leaves-1;
		RequireSetup(nodes<=limit-model.geometry_index_nodes,"combined triangle tree index range exceeded");
		shape.index_offset=static_cast<uint32_t>(model.geometry_index_nodes);
		model.geometry_index_nodes+=nodes;
		model.geometry_index_scratch=std::max(model.geometry_index_scratch,size_t(shape.index_leaves));
	}
	for (double value:model.geometry) RequireSetup(std::isfinite(value),"non-finite geometry/material descriptor");
	const size_t rows=size_t(numLines[0])*numLines[1];
	const size_t sourceRows=size_t(numLines[1])*numLines[2];
	RequireSetup(rows<=(limit-3-sourceRows)/2,"candidate index range exceeded");
	const CSProperties::PropertyType masks[3]={CSProperties::MATERIAL,
		CSProperties::PropertyType(CSProperties::MATERIAL|CSProperties::METAL),CSProperties::EXCITATION};
	for (int which=0;which<3;++which)
	{
		model.candidate_offsets[which]=static_cast<unsigned int>(model.candidate_order.size());
		const auto list=CSX->GetAllPrimitives(true,masks[which]);
		RequireSetup(list.size()<=limit-model.candidate_order.size(),"candidate priority order range exceeded");
		for (CSPrimitives* prim:list) model.candidate_order.push_back(indices.at(prim));
	}
	model.candidate_offsets[3]=static_cast<unsigned int>(model.candidate_order.size());
	return model;
}

int GPU_SetupController::Build(const std::string& backend,const std::string& mode,GPU_Backend* parent,const std::function<int()>& build)
{
	m_Setup.reset(); m_SetupBackend.reset();m_SetupResult.reset();
	m_OriginalPrimitives.clear();
	for (auto& kind:m_MaterialTile) for (auto& component:kind) for (auto& tile:component) tile.clear();
	std::vector<bool> original_used;
	bool multigrid=false;
	try
	{
		if (mode!="cpu")
		{
			SetupTimer timer("GPU construction");
			if (!op.CSX) throw std::invalid_argument("operator geometry is missing");
			m_OriginalPrimitives=op.CSX->GetAllPrimitives();
			for (CSPrimitives* primitive:m_OriginalPrimitives) original_used.push_back(primitive->GetPrimitiveUsed());
			GPU_SetupModel model=SetupModel();
			multigrid=model.first_radius!=0;
			m_SetupBackend.reset(parent ? parent->NewSubGridBackend() : GPU_Backend::New(backend));
			std::string reason;
			m_Setup.reset(m_SetupBackend->NewOperatorSetup(model,reason));
			if (!m_Setup) throw GPU_SetupUnavailable(reason);
			std::cout << "Operator_GPU: HIP construction active (materials, timestep, coefficients, PEC/PMC and extensions)" << std::endl;
			timer.Report("geometry, preflight and allocation");
		}
		const int result=build();
		if (result!=0 && m_Setup)
		{
			for (size_t n=0;n<original_used.size();++n) m_OriginalPrimitives[n]->SetPrimitiveUsed(original_used[n]);
			m_Setup.reset();m_SetupBackend.reset();m_SetupResult.reset();return result;
		}
		if (result==0 && m_Setup)
		{
			if (multigrid && !m_Setup->BuildMultiGrid()) throw GPU_SetupUnavailable("backend has no multigrid construction adapter");
			m_SetupResult=std::shared_ptr<GPU_SetupResult>(m_Setup->Finish());
			if (!m_SetupBackend->SetSetupResult(m_SetupResult)) throw std::runtime_error("GPU construction result could not be adopted by its backend");
			std::cout << "HIP operator construction completed" << std::endl;
		}
		m_Setup.reset();
		m_SetupOriginalUsed=original_used;
		return result;
	}
	catch (const GPU_SetupUnavailable&)
	{
		for (size_t n=0;n<original_used.size();++n) m_OriginalPrimitives[n]->SetPrimitiveUsed(original_used[n]);
		m_Setup.reset(); m_SetupBackend.reset();m_SetupResult.reset();
		throw;
	}
	catch (const std::bad_alloc&)
	{
		if (mode=="cpu") throw;
		for (size_t n=0;n<original_used.size();++n) m_OriginalPrimitives[n]->SetPrimitiveUsed(original_used[n]);
		m_Setup.reset(); m_SetupBackend.reset();m_SetupResult.reset();
		throw GPU_SetupUnavailable("insufficient host memory for operator construction");
	}
	catch (...)
	{
		for (size_t n=0;n<original_used.size();++n) m_OriginalPrimitives[n]->SetPrimitiveUsed(original_used[n]);
		m_Setup.reset();m_SetupBackend.reset();m_SetupResult.reset();throw;
	}
}

bool GPU_SetupController::BuildExcitationSignal(unsigned int maxTS)
{
	if (!m_SetupBackend) return op.Operator::BuildExcitationSignal(maxTS);
	try {return m_SetupBackend->BuildExcitationSignal(*op.m_Exc,maxTS);}
	catch (const GPU_SetupUnavailable&)
	{
		for (size_t n=0;n<m_SetupOriginalUsed.size();++n) m_OriginalPrimitives[n]->SetPrimitiveUsed(m_SetupOriginalUsed[n]);
		m_SetupBackend.reset();
		m_SetupResult.reset();
		throw;
	}
	catch (...)
	{
		for (size_t n=0;n<m_SetupOriginalUsed.size();++n) m_OriginalPrimitives[n]->SetPrimitiveUsed(m_SetupOriginalUsed[n]);
		m_SetupBackend.reset();m_SetupResult.reset();throw;
	}
}

double GPU_SetupController::CalcTimestep()
{
	op.dT=m_Setup->CalcTimestep(op.m_TimeStepVar);
	op.m_Used_TS_Name=op.m_TimeStepVar==3?"Rennings_2":"Rennings_1";
	op.MainOp->SetReflection2Cell();
	return 0;
}

bool GPU_SetupController::CalcEC()
{
	float* ec[4][3];
	for (int n=0;n<3;++n) {ec[0][n]=op.EC_C[n];ec[1][n]=op.EC_G[n];ec[2][n]=op.EC_L[n];ec[3][n]=op.EC_R[n];}
	float* materials[4]={op.m_epsR_ptr?op.m_epsR_ptr->data():nullptr,op.m_kappa_ptr?op.m_kappa_ptr->data():nullptr,
	                     op.m_mueR_ptr?op.m_mueR_ptr->data():nullptr,op.m_sigma_ptr?op.m_sigma_ptr->data():nullptr};
	m_Setup->CalcEC(ec,materials);return true;
}

bool GPU_SetupController::CalcPEC()
{
	std::copy(m_SetupPEC,m_SetupPEC+3,op.m_Nr_PEC);return true;
}

void GPU_SetupController::CalcCoefficients()
{
	float* matrices[4]={op.vv_ptr->data(),op.vi_ptr->data(),op.ii_ptr->data(),op.iv_ptr->data()};
	std::vector<unsigned int> used;m_Setup->CalcCoefficients(op.dT,matrices,used);
	for (size_t n=0;n<m_SetupPrimitives.size();++n) if (used[n+1]) m_SetupPrimitives[n]->SetPrimitiveUsed(true);
	std::copy(used.begin()+m_SetupPrimitives.size()+1,used.begin()+m_SetupPrimitives.size()+4,m_SetupPEC);
	op.MainOp->SetPos(op.numLines[0]-1,op.numLines[1]-1,op.numLines[2]-1);
}

void GPU_SetupController::InitEC()
{
	for (int n=0;n<3;++n)
	{
		delete[] op.EC_C[n];op.EC_C[n]=nullptr;delete[] op.EC_G[n];op.EC_G[n]=nullptr;
		delete[] op.EC_L[n];op.EC_L[n]=nullptr;delete[] op.EC_R[n];op.EC_R[n]=nullptr;
	}
}

void GPU_SetupController::InitOperator()
{
	delete op.vv_ptr;delete op.vi_ptr;delete op.ii_ptr;delete op.iv_ptr;
	op.vv_ptr=new ArrayLib::ArrayNIJK<float>();op.vi_ptr=new ArrayLib::ArrayNIJK<float>();
	op.ii_ptr=new ArrayLib::ArrayNIJK<float>();op.iv_ptr=new ArrayLib::ArrayNIJK<float>();
}

float GPU_SetupController::Coefficient(unsigned int kind,unsigned int n,unsigned int x,unsigned int y,unsigned int z) const
{
	if (kind>=4 || n>=3 || x>=op.numLines[0] || y>=op.numLines[1] || z>=op.numLines[2]) throw std::out_of_range("operator coefficient index");
	ArrayLib::ArrayNIJK<float>* arrays[4]={op.vv_ptr,op.vi_ptr,op.ii_ptr,op.iv_ptr};
	if (arrays[kind] && arrays[kind]->valid()) return (*arrays[kind])(n,x,y,z);
	const size_t index=((size_t(n)*op.numLines[0]+x)*op.numLines[1]+y)*op.numLines[2]+z;
	float value=0;
	if (m_Setup) m_Setup->ReadCoefficients(kind,index,1,&value);
	else if (m_SetupResult) m_SetupResult->ReadCoefficients(kind,index,1,&value);
	else throw std::logic_error("operator coefficients have not been constructed");
	return value;
}

void GPU_SetupController::SetCoefficient(unsigned int kind,unsigned int n,unsigned int x,unsigned int y,unsigned int z,float value)
{
	if (kind>=4 || n>=3 || x>=op.numLines[0] || y>=op.numLines[1] || z>=op.numLines[2]) throw std::out_of_range("operator coefficient index");
	ArrayLib::ArrayNIJK<float>* arrays[4]={op.vv_ptr,op.vi_ptr,op.ii_ptr,op.iv_ptr};
	const size_t index=((size_t(n)*op.numLines[0]+x)*op.numLines[1]+y)*op.numLines[2]+z;
	if (m_Setup) m_Setup->WriteCoefficient(kind,index,value);
	else if (m_SetupResult) m_SetupResult->WriteCoefficient(kind,index,value);
	else if (!arrays[kind] || !arrays[kind]->valid()) throw std::logic_error("operator coefficients have not been constructed");
	if (arrays[kind] && arrays[kind]->valid()) (*arrays[kind])(n,x,y,z)=value;
}

void GPU_SetupController::MaterializeCoefficients()
{
	if (!HasDeviceResult()) return;
	ArrayLib::ArrayNIJK<float>* arrays[4]={op.vv_ptr,op.vi_ptr,op.ii_ptr,op.iv_ptr};
	const char* names[4]={"vv","vi","ii","iv"};
	for (unsigned int kind=0;kind<4;++kind)
	{
		if (!arrays[kind]->valid()) arrays[kind]->Init(names[kind],op.numLines);
		if (m_Setup) m_Setup->ReadCoefficients(kind,0,arrays[kind]->size(),arrays[kind]->data());
		else m_SetupResult->ReadCoefficients(kind,0,arrays[kind]->size(),arrays[kind]->data());
	}
}

void GPU_SetupController::MaterializeMultiGrid(GPU_MultiGridInterpolation& output) const
{
	if (!m_SetupResult) throw std::logic_error("multigrid construction result is missing");
	m_SetupResult->ReadMultiGrid(output);
}

void GPU_SetupController::ReadMaterials(unsigned int kind,size_t offset,size_t count,float* values) const
{
	if (m_Setup) m_Setup->ReadMaterials(kind,offset,count,values);
	else if (m_SetupResult) m_SetupResult->ReadMaterials(kind,offset,count,values);
	else throw std::logic_error("device material storage is unavailable");
}

float GPU_SetupController::Material(unsigned int kind,size_t index) const
{
	std::lock_guard<std::mutex> lock(m_MaterialMutex);
	const size_t size=3ull*op.numLines[0]*op.numLines[1]*op.numLines[2];
	if (kind>=4 || index>=size) throw std::out_of_range("device material index");
	const size_t cells=size/3,component=index/cells,local=index%cells,slot=(local/1024)%8;
	auto& tile=m_MaterialTile[kind][component][slot];auto& offset=m_MaterialOffset[kind][component][slot];
	if (tile.empty() || index<offset || index-offset>=tile.size())
	{
		offset=component*cells+local/1024*1024;
		tile.resize((std::min)((component+1)*cells-offset,size_t(1024)));ReadMaterials(kind,offset,tile.size(),tile.data());
	}
	return tile[index-offset];
}
