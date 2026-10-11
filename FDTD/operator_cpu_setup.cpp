/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "operator_cpu_setup.h"
#include "ContinuousStructure.h"
#include "CSProperties.h"
#include "CSPrimitives.h"
#include <array>
#include <cstdint>
#include <exception>
#include <future>
#include <map>
#include <stdexcept>

struct CpuOperatorSetup::Worker
{
	const Operator* owner;
	std::vector<std::unique_ptr<CSProperties> > properties;
	std::map<CSPrimitives*, CSPrimitives*> copies;
	std::vector<CSPrimitives*> material, pec;

	Worker(Operator& op, const std::vector<CSPrimitives*>& materialOrder,
	       const std::vector<CSPrimitives*>& pecOrder) : owner(&op)
	{
		// Clone serially: CSXCAD's primitive ID counter is process-global. Each
		// worker then owns mutable query caches, expression parsers and used flags.
		std::map<CSProperties*, CSProperties*> propertyCopies;
		for (CSPrimitives* primitive : pecOrder) {
			CSProperties* source = primitive->GetProperty();
			if (!propertyCopies.count(source)) {
				std::unique_ptr<CSProperties> copy(source->GetCopy(false));
				if (!copy || copy->GetType()!=source->GetType() || !copy->Update())
					throw std::runtime_error("CPU setup could not copy a geometry property");
				propertyCopies[source] = copy.get();
				properties.push_back(std::move(copy));
			}
			CSPrimitives* copy = primitive->GetCopy(propertyCopies[source]);
			if (!copy || copy->GetType()!=primitive->GetType() || !copy->Update())
				throw std::runtime_error("CPU setup could not copy a geometry primitive");
			copies[primitive] = copy;
		}
		// Preserve the reference's separate sorts, including equal-priority ties.
		for (CSPrimitives* primitive : materialOrder) material.push_back(copies.at(primitive));
		for (CSPrimitives* primitive : pecOrder) pec.push_back(copies.at(primitive));
	}
};

thread_local CpuOperatorSetup::Worker* CpuOperatorSetup::s_worker = nullptr;

CpuOperatorSetup::CpuOperatorSetup(Operator& op, unsigned int threads) : m_op(op)
{
	const auto material = op.CSX->GetAllPrimitives(true, CSProperties::MATERIAL);
	const auto pec = op.CSX->GetAllPrimitives(true,
		CSProperties::PropertyType(CSProperties::MATERIAL | CSProperties::METAL));
	for (unsigned int worker=0; worker<threads; ++worker)
		m_workers.emplace_back(new Worker(op, material, pec));
}

CpuOperatorSetup::~CpuOperatorSetup() = default;

void CpuOperatorSetup::Run(const std::function<void(unsigned int, unsigned int, unsigned int)>& job)
{
	const unsigned int count = static_cast<unsigned int>(m_workers.size());
	auto work = [&](unsigned int worker) {
		struct QueryScope {
			Worker* previous;
			QueryScope(Worker* current) : previous(s_worker) { s_worker=current; }
			~QueryScope() { s_worker=previous; }
		} scope(m_workers[worker].get());
		const unsigned int start = uint64_t(m_op.numLines[0])*worker/count;
		const unsigned int stop = uint64_t(m_op.numLines[0])*(worker+1)/count-1;
		job(worker, start, stop);
	};
	std::vector<std::future<void> > jobs;
	// Futures join on destruction if launching a later worker fails.
	for (unsigned int worker=1; worker<count; ++worker)
		jobs.push_back(std::async(std::launch::async, work, worker));
	std::exception_ptr failure;
	try { work(0); } catch (...) { failure=std::current_exception(); }
	for (auto& pending : jobs) {
		try { pending.get(); } catch (...) { if (!failure) failure=std::current_exception(); }
	}
	if (failure) std::rethrow_exception(failure);
}

std::vector<CSPrimitives*> CpuOperatorSetup::PrimitivesBoundBox(const double bounds[6], int type) const
{
	if (!HasWorkerContext())
		return m_op.CSX->GetPrimitivesByBoundBox(bounds, true, CSProperties::PropertyType(type));
	const auto& candidates = type==CSProperties::MATERIAL ? s_worker->material : s_worker->pec;
	std::vector<CSPrimitives*> result;
	for (CSPrimitives* primitive : candidates)
		if (primitive->IsInsideBox(bounds)>=0) result.push_back(primitive);
	return result;
}

bool CpuOperatorSetup::HasWorkerContext() const
{
	return s_worker && s_worker->owner==&m_op;
}

void CpuOperatorSetup::MergeUsedPrimitives()
{
	for (const auto& worker : m_workers)
		for (const auto& entry : worker->copies)
			if (entry.second->GetPrimitiveUsed()) entry.first->SetPrimitiveUsed(true);
}

void CpuOperatorSetup::CalcEC()
{
	m_op.MainOp->SetPos(0,0,0);
	Run([&](unsigned int, unsigned int start, unsigned int stop) { m_op.Calc_EC_Range(start,stop); });
	MergeUsedPrimitives();
}

void CpuOperatorSetup::CalcCoefficients()
{
	m_op.MainOp->SetPos(0,0,0);
	Run([&](unsigned int, unsigned int start, unsigned int stop) {
		unsigned int pos[3];
		for (int n=0; n<3; ++n)
			for (pos[0]=start; pos[0]<=stop; ++pos[0])
				for (pos[1]=0; pos[1]<m_op.numLines[1]; ++pos[1])
					for (pos[2]=0; pos[2]<m_op.numLines[2]; ++pos[2])
						m_op.Calc_ECOperatorPos(n,pos);
	});
	// Leave the same cursor state as the serial loop for subsequent extensions.
	m_op.MainOp->SetPos(m_op.numLines[0]-1,m_op.numLines[1]-1,m_op.numLines[2]-1);
}

void CpuOperatorSetup::CalcPEC()
{
	std::vector<std::array<unsigned int,3> > counters(m_workers.size());
	Run([&](unsigned int worker, unsigned int start, unsigned int stop) {
		counters[worker].fill(0);
		m_op.CalcPEC_Range(start,stop,counters[worker].data());
	});
	for (unsigned int n=0; n<3; ++n) {
		m_op.m_Nr_PEC[n]=0;
		for (const auto& counter : counters) m_op.m_Nr_PEC[n]+=counter[n];
	}
	MergeUsedPrimitives();
}
