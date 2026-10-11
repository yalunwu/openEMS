/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OPERATOR_CPU_SETUP_H
#define OPERATOR_CPU_SETUP_H

#include "operator.h"
#include <functional>
#include <memory>
#include <vector>

// Parallel construction preserves Operator's basic storage and runtime type.
class CpuOperatorSetup
{
public:
	CpuOperatorSetup(Operator& op, unsigned int threads);
	~CpuOperatorSetup();
	void CalcEC();
	void CalcCoefficients();
	void CalcPEC();
	bool HasWorkerContext() const;
	std::vector<CSPrimitives*> PrimitivesBoundBox(const double bounds[6], int type) const;

private:
	struct Worker;
	Operator& m_op;
	std::vector<std::unique_ptr<Worker> > m_workers;
	static thread_local Worker* s_worker;
	void Run(const std::function<void(unsigned int, unsigned int, unsigned int)>& job);
	void MergeUsedPrimitives();
};

#endif
