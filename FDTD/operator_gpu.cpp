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

Operator_GPU::Operator_GPU(const std::string& backend, unsigned int numThreads, const std::string& setup) : Operator()
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
	m_Engine = Engine_GPU::New(this, m_Backend, NULL, m_SetupBackend.release());
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
	if (m_Setup)
	{
		float* ec[4][3];
		for (int n=0;n<3;++n) { ec[0][n]=EC_C[n]; ec[1][n]=EC_G[n]; ec[2][n]=EC_L[n]; ec[3][n]=EC_R[n]; }
		float* materials[4]={m_epsR_ptr?m_epsR_ptr->data():NULL,m_kappa_ptr?m_kappa_ptr->data():NULL,
		                     m_mueR_ptr?m_mueR_ptr->data():NULL,m_sigma_ptr?m_sigma_ptr->data():NULL};
		m_Setup->CalcEC(ec,materials);
		return true;
	}
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
	if (m_Setup)
	{
		std::copy(m_SetupPEC,m_SetupPEC+3,m_Nr_PEC);
		return true;
	}
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
	if (m_Setup)
	{
		float* matrices[4]={vv_ptr->data(),vi_ptr->data(),ii_ptr->data(),iv_ptr->data()};
		std::vector<unsigned int> used;
		m_Setup->CalcCoefficients(dT,matrices,used);
		for (size_t n=0;n<m_SetupPrimitives.size();++n)
			if (used[n+1]) m_SetupPrimitives[n]->SetPrimitiveUsed(true);
		std::copy(used.begin()+m_SetupPrimitives.size()+1,used.begin()+m_SetupPrimitives.size()+4,m_SetupPEC);
		MainOp->SetPos(numLines[0]-1,numLines[1]-1,numLines[2]-1);
		return;
	}
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
