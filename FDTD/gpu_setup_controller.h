/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_CONTROLLER_H
#define GPU_SETUP_CONTROLLER_H

#include "operator.h"
#include "gpu_operator_setup.h"
#include <functional>
#include <mutex>

class GPU_Backend;

// Construction storage belongs to the operator, independently of its CPU field
// layout. Cartesian, cylindrical and child operators share this lifecycle.
class GPU_SetupController
{
public:
	explicit GPU_SetupController(Operator& owner);
	~GPU_SetupController();
	GPU_OperatorSetup* Setup() const {return m_Setup.get();}
	bool HasDeviceResult() const {return bool(m_Setup) || bool(m_SetupResult);}
	GPU_Backend* ReleaseBackend();
	GPU_Backend* Backend() const {return m_SetupBackend.get();}
	std::shared_ptr<GPU_SetupResult> Result() const {return m_SetupResult;}
	int Build(const std::string& backend,const std::string& mode,GPU_Backend* parent,const std::function<int()>& build);
	bool BuildExcitationSignal(unsigned int maxTS);
	bool CalcEC();
	bool CalcPEC();
	void CalcCoefficients();
	double CalcTimestep();
	void InitEC();
	void InitOperator();
	float Coefficient(unsigned int kind,unsigned int n,unsigned int x,unsigned int y,unsigned int z) const;
	void SetCoefficient(unsigned int kind,unsigned int n,unsigned int x,unsigned int y,unsigned int z,float value);
	void MaterializeCoefficients();
	float Material(unsigned int kind,size_t index) const;
	void ReadMaterials(unsigned int kind,size_t offset,size_t count,float* values) const;
	void MaterializeMultiGrid(GPU_MultiGridInterpolation& output) const;

private:
	GPU_SetupModel SetupModel();
	Operator& op;
	std::unique_ptr<GPU_Backend> m_SetupBackend;
	std::unique_ptr<GPU_OperatorSetup> m_Setup;
	std::shared_ptr<GPU_SetupResult> m_SetupResult;
	std::vector<CSPrimitives*> m_SetupPrimitives;
	std::vector<CSPrimitives*> m_OriginalPrimitives;
	std::vector<bool> m_SetupOriginalUsed;
	unsigned int m_SetupPEC[3]={};
	mutable size_t m_MaterialOffset[4][3][8]={};
	mutable std::vector<float> m_MaterialTile[4][3][8];
	mutable std::mutex m_MaterialMutex;
};
#endif
