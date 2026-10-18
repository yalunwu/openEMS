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

#include <iostream>
#include <algorithm>
#include <exception>
#include <thread>

#include "engine_gpu.h"
#include "operator_gpu.h"
#include "gpu_setup_controller.h"
#include "tools/useful.h"
#include "CSPrimitives.h"

using std::cout;
using std::endl;

Operator_GPU* Operator_GPU::New(const std::string& backend, unsigned int numThreads, const std::string& setup)
{
	if (setup!="cpu" && setup!="auto" && setup!="gpu") throw std::invalid_argument("gpu-setup must be cpu, auto or gpu");
	cout << "Create FDTD operator (GPU)" << endl;
	Operator_GPU* op = new Operator_GPU(backend, numThreads, setup);
	op->Init();
	return op;
}

Operator_GPU::Operator_GPU(const std::string& backend, unsigned int numThreads, const std::string& setup) : Operator(), m_Construction(new GPU_SetupController(*this))
{
	m_Backend = backend;
	m_NumThreads = numThreads;
	m_SetupMode = setup;
}

Operator_GPU::~Operator_GPU()
{
}

Engine* Operator_GPU::CreateEngine()
{
	m_Engine = Engine_GPU::New(this, m_Backend, NULL, m_Construction->ReleaseBackend());
	// The device holds the coefficients now: free the host arrays (48 bytes per node), as
	// Operator_SSE_Compressed frees its full arrays. GetVV() and friends are invalid after this.
	delete vv_ptr; vv_ptr = NULL;
	delete vi_ptr; vi_ptr = NULL;
	delete ii_ptr; ii_ptr = NULL;
	delete iv_ptr; iv_ptr = NULL;
	return m_Engine;
}

void Operator_GPU::ThreadRanges(std::vector<unsigned int>& start, std::vector<unsigned int>& stop) const
{
	const unsigned int available = AvailableThreads();
	const unsigned int count = std::max(1u, std::min(numLines[0], m_NumThreads ? std::min(m_NumThreads, available) : available));
	std::vector<unsigned int> jobs = AssignJobs2Threads(numLines[0], count, true);
	unsigned int line = 0;
	for (size_t n=0; n<jobs.size(); ++n)
	{
		start.push_back(line);
		line += jobs.at(n);
		stop.push_back(line-1);
	}
}

// Join every worker before propagating an exception, including a failed launch.
template <typename Job>
static void RunSetupWorkers(size_t count, Job job)
{
	std::vector<std::exception_ptr> errors(count);
	std::vector<std::thread> threads;
	threads.reserve(count-1);
	auto work = [&](size_t n) {
		try { job(n); } catch (...) { errors[n] = std::current_exception(); }
	};
	try
	{
		for (size_t n=1; n<count; ++n)
			threads.emplace_back(work, n);
	}
	catch (...)
	{
		for (auto& thread : threads) thread.join();
		throw;
	}
	work(0);
	for (auto& thread : threads) thread.join();
	for (const auto& error : errors)
		if (error) std::rethrow_exception(error);
}

bool Operator_GPU::Calc_EC()
{
	if (m_Construction->Setup()) return m_Construction->CalcEC();
	if (CSX==NULL)
	{
		std::cerr << "Operator_GPU::Calc_EC: CSX not given or invalid!!!" << std::endl;
		return false;
	}
	MainOp->SetPos(0,0,0);

	std::vector<unsigned int> start, stop;
	ThreadRanges(start, stop);
	cout << "Operator_GPU: CPU setup using " << start.size() << " thread(s)" << endl;
	RunSetupWorkers(start.size(), [&](size_t n) { Calc_EC_Range(start[n], stop[n]); });
	return true;
}

bool Operator_GPU::CalcPEC()
{
	if (m_Construction->Setup()) return m_Construction->CalcPEC();
	std::vector<unsigned int> start, stop;
	ThreadRanges(start, stop);
	std::vector<unsigned int> counter(3*start.size(), 0);   // three PEC counters per thread, summed up below
	RunSetupWorkers(start.size(), [&](size_t n) { CalcPEC_Range(start[n], stop[n], &counter[3*n]); });

	for (int n=0; n<3; ++n)
	{
		m_Nr_PEC[n] = 0;
		for (size_t t=0; t<start.size(); ++t)
			m_Nr_PEC[n] += counter[3*t+n];
	}
	CalcPEC_Curves();
	return true;
}

void Operator_GPU::CalcCoefficients()
{
	if (m_Construction->Setup()) {m_Construction->CalcCoefficients();return;}
	std::vector<unsigned int> start, stop;
	ThreadRanges(start, stop);
	if (start.size()==1)
	{
		Operator::CalcCoefficients();
		return;
	}
	// GetPos() reads relative to this fixed cursor; SetPos() inside workers races.
	MainOp->SetPos(0,0,0);
	RunSetupWorkers(start.size(), [&](size_t worker) {
		unsigned int pos[3];
		for (int n=0; n<3; ++n)
			for (pos[0]=start[worker]; pos[0]<=stop[worker]; ++pos[0])
				for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
					for (pos[2]=0; pos[2]<numLines[2]; ++pos[2])
						Calc_ECOperatorPos(n, pos, MainOp->GetPos(pos[0],pos[1],pos[2]));
	});
	// Subsequent extensions see the same cursor as after the serial pass.
	MainOp->SetPos(numLines[0]-1,numLines[1]-1,numLines[2]-1);
}

void Operator_GPU::Init_EC()
{
	if (m_Construction->Setup()) m_Construction->InitEC();
	else Operator::Init_EC();
}

void Operator_GPU::InitOperator()
{
	if (m_Construction->Setup()) m_Construction->InitOperator();
	else Operator::InitOperator();
}

void Operator_GPU::ApplyElectricBC(bool* dirs) {if (!m_Construction->Setup()) Operator::ApplyElectricBC(dirs);}
void Operator_GPU::ApplyMagneticBC(bool* dirs) {if (!m_Construction->Setup()) Operator::ApplyMagneticBC(dirs);}

GPU_OperatorSetup* Operator_GPU::GetSetupBackend() const {return m_Construction->Setup();}
GPU_SetupController* Operator_GPU::GetConstruction() const {return m_Construction->HasDeviceResult() ? m_Construction.get() : nullptr;}

int Operator_GPU::CalcECOperator(DebugFlags flags)
{
	if (m_SetupMode!="cpu" && (flags&debugMaterial)) for (int k=0;k<4;++k) SetMaterialStoreFlags(k,true);
	return m_Construction->Build(m_Backend,m_SetupMode,nullptr,[this,flags]() {return Operator::CalcECOperator(flags);});
}

bool Operator_GPU::BuildExcitationSignal(unsigned int maxTS) {return m_Construction->BuildExcitationSignal(maxTS);}

double Operator_GPU::CalcTimestep()
{
	return m_Construction->Setup() ? m_Construction->CalcTimestep() : Operator::CalcTimestep();
}

float Operator_GPU::Coefficient(unsigned int kind,unsigned int n,unsigned int x,unsigned int y,unsigned int z) const
{
	return m_Construction->Coefficient(kind,n,x,y,z);
}

void Operator_GPU::SetVV(unsigned int n,unsigned int x,unsigned int y,unsigned int z,FDTD_FLOAT value) {m_Construction->SetCoefficient(0,n,x,y,z,value);}
void Operator_GPU::SetVI(unsigned int n,unsigned int x,unsigned int y,unsigned int z,FDTD_FLOAT value) {m_Construction->SetCoefficient(1,n,x,y,z,value);}
void Operator_GPU::SetII(unsigned int n,unsigned int x,unsigned int y,unsigned int z,FDTD_FLOAT value) {m_Construction->SetCoefficient(2,n,x,y,z,value);}
void Operator_GPU::SetIV(unsigned int n,unsigned int x,unsigned int y,unsigned int z,FDTD_FLOAT value) {m_Construction->SetCoefficient(3,n,x,y,z,value);}
void Operator_GPU::MaterializeCoefficients() {m_Construction->MaterializeCoefficients();}
void Operator_GPU::DumpOperator2File(std::string filename) {MaterializeCoefficients();Operator::DumpOperator2File(filename);}
void Operator_GPU::DumpPEC2File(std::string filename,unsigned int* range) {MaterializeCoefficients();Operator::DumpPEC2File(filename,range);}
