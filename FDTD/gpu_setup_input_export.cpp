/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "gpu_setup_input_export.h"
#include "gpu_setup_geometry_export.h"
#include "gpu_operator_setup.h"
#include "CSPropDiscMaterial.h"
#include "CSPropExcitation.h"
#include "CSPropLorentzMaterial.h"
#include "CSPropDebyeMaterial.h"
#include "gpu_setup_constant.h"
#include <cmath>

uint32_t GPU_ExportSetupDiscrete(CSPropDiscMaterial&, GPU_SetupModel&)
{
	throw GPU_SetupUnavailable("discrete materials require CPU operator construction");
}

uint32_t GPU_ExportSetupFileWeight(CSPropExcitation&, GPU_SetupModel&)
{
	throw GPU_SetupUnavailable("file-weighted sources require CPU operator construction");
}

void GPU_ExportSetupDispersion(CSPropMaterial& material, GPU_SetupModel& model, GPU_SetupProperty& property)
{
	CSPropLorentzMaterial* lorentz=dynamic_cast<CSPropLorentzMaterial*>(&material);
	CSPropDebyeMaterial* debye=dynamic_cast<CSPropDebyeMaterial*>(&material);
	if (!lorentz && !debye) return;
	const int count=lorentz ? lorentz->GetDispersionOrder() : debye->GetDispersionOrder();
	if (count<0 || model.dispersion.size()>UINT32_MAX || size_t(count)>UINT32_MAX-model.dispersion.size()) throw GPU_SetupUnavailable("dispersion descriptor index range exceeded");
	property.dispersion_offset=static_cast<uint32_t>(model.dispersion.size()); property.dispersion_count=count;
	double coordinate[3]={0,0,0};
	for (int pole=0;pole<count;++pole)
	{
		GPU_SetupDispersionPole d={};
		for (int n=0;n<3;++n) for (int magnetic=0;magnetic<(lorentz ? 2 : 1);++magnetic)
		{
			std::string terms[3]; double base[3]={},constant[3]={};
			if (lorentz)
			{
				terms[0]=magnetic ? lorentz->GetMuePlasmaFreqWeightFunction(pole,n) : lorentz->GetEpsPlasmaFreqWeightFunction(pole,n);
				terms[1]=magnetic ? lorentz->GetMueRelaxTimeWeightFunction(pole,n) : lorentz->GetEpsRelaxTimeWeightFunction(pole,n);
				terms[2]=magnetic ? lorentz->GetMueLorPoleFreqWeightFunction(pole,n) : lorentz->GetEpsLorPoleFreqWeightFunction(pole,n);
				base[0]=magnetic ? lorentz->GetMuePlasmaFreq(pole,n) : lorentz->GetEpsPlasmaFreq(pole,n);
				base[1]=magnetic ? lorentz->GetMueRelaxTime(pole,n) : lorentz->GetEpsRelaxTime(pole,n);
				base[2]=magnetic ? lorentz->GetMueLorPoleFreq(pole,n) : lorentz->GetEpsLorPoleFreq(pole,n);
			}
			else
			{
				terms[0]=debye->GetEpsDeltaWeightFunction(pole,n); terms[1]=debye->GetEpsRelaxTimeWeightFunction(pole,n);
				base[0]=debye->GetEpsDelta(pole,n); base[1]=debye->GetEpsRelaxTime(pole,n);
			}
			for (int kind=0;kind<(lorentz ? 3 : 2);++kind)
			{
				if (GPU_SetupConstant(terms[kind]))
				{
					if (lorentz)
						constant[kind]=kind==0 ? (magnetic ? lorentz->GetMuePlasmaFreqWeighted(pole,n,coordinate) : lorentz->GetEpsPlasmaFreqWeighted(pole,n,coordinate)) :
						               kind==1 ? (magnetic ? lorentz->GetMueRelaxTimeWeighted(pole,n,coordinate) : lorentz->GetEpsRelaxTimeWeighted(pole,n,coordinate)) :
						                         (magnetic ? lorentz->GetMueLorPoleFreqWeighted(pole,n,coordinate) : lorentz->GetEpsLorPoleFreqWeighted(pole,n,coordinate));
					else constant[kind]=kind==0 ? debye->GetEpsDeltaWeighted(pole,n,coordinate) : debye->GetEpsRelaxTimeWeighted(pole,n,coordinate);
					d.values[n][magnetic][kind]=constant[kind];
				}
				else
				{
					d.values[n][magnetic][kind]=base[kind];
					d.weights[n][magnetic][kind]=GPU_CompileSetupExpression(terms[kind],"x,y,z,rho,r,alpha,theta",model.expressions);
				}
				if (!std::isfinite(d.values[n][magnetic][kind])) throw std::invalid_argument("non-finite dispersion parameter");
			}
		}
		model.dispersion.push_back(d);
	}
}
