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

#ifndef OPERATOR_GPU_H
#define OPERATOR_GPU_H

#include "operator.h"

#include <vector>
#include <memory>
#include "gpu_operator_setup.h"

class GPU_Backend;
class GPU_SetupController;

//! Operator for the GPU engine
/*!
  CPU construction uses the basic operator layout. Optional HIP construction
  materializes requested host data and transfers its packed core result
  directly to the backend.
  */
class Operator_GPU : public Operator
{
public:
	//! Create a new operator for the GPU backend \a backend (see GPU_Backend::New)
	static Operator_GPU* New(const std::string& backend="auto", unsigned int numThreads=0, const std::string& setup="cpu");
	virtual ~Operator_GPU();

	virtual Engine* CreateEngine();
	virtual int CalcECOperator(DebugFlags debugFlags=None);
	virtual bool BuildExcitationSignal(unsigned int maxTS);
	GPU_OperatorSetup* GetSetupBackend() const override;
	GPU_SetupController* GetConstruction() const override;
	void ApplyElectricBC(bool* dirs) override;
	void ApplyMagneticBC(bool* dirs) override;
	FDTD_FLOAT GetVV(unsigned int n,unsigned int x,unsigned int y,unsigned int z) const override {return Coefficient(0,n,x,y,z);}
	FDTD_FLOAT GetVI(unsigned int n,unsigned int x,unsigned int y,unsigned int z) const override {return Coefficient(1,n,x,y,z);}
	FDTD_FLOAT GetII(unsigned int n,unsigned int x,unsigned int y,unsigned int z) const override {return Coefficient(2,n,x,y,z);}
	FDTD_FLOAT GetIV(unsigned int n,unsigned int x,unsigned int y,unsigned int z) const override {return Coefficient(3,n,x,y,z);}
	using Operator::GetVV;using Operator::GetVI;using Operator::GetII;using Operator::GetIV;
	void SetVV(unsigned int n,unsigned int x,unsigned int y,unsigned int z,FDTD_FLOAT value) override;
	void SetVI(unsigned int n,unsigned int x,unsigned int y,unsigned int z,FDTD_FLOAT value) override;
	void SetII(unsigned int n,unsigned int x,unsigned int y,unsigned int z,FDTD_FLOAT value) override;
	void SetIV(unsigned int n,unsigned int x,unsigned int y,unsigned int z,FDTD_FLOAT value) override;
	void DumpOperator2File(std::string filename) override;
	void DumpPEC2File(std::string filename,unsigned int* range=nullptr) override;

	const std::string& GetBackendName() const {return m_Backend;}

protected:
	//! use New() for creating a new Operator
	Operator_GPU(const std::string& backend, unsigned int numThreads, const std::string& setup="cpu");

	//! Build the basic-layout operator over disjoint x line ranges in parallel
	/*!
	  Operator_GPU derives from the basic Operator, whose storage the host mirror
	  of Engine_GPU uses; Operator_Multithread is an Operator_SSE_Compressed and
	  brings the wrong one.
	  */
	virtual bool Calc_EC();
	virtual bool CalcPEC();
	virtual void CalcCoefficients();
	virtual double CalcTimestep();
	void Init_EC() override;
	void InitOperator() override;
	float Coefficient(unsigned int kind,unsigned int n,unsigned int x,unsigned int y,unsigned int z) const;
	void MaterializeCoefficients();

	//! x line ranges of the threads, at most one per available CPU, bounds inclusive
	void ThreadRanges(std::vector<unsigned int>& start, std::vector<unsigned int>& stop) const;

	std::string m_Backend;
	unsigned int m_NumThreads;
	std::string m_SetupMode;
	std::unique_ptr<GPU_SetupController> m_Construction;
};

#endif // OPERATOR_GPU_H
