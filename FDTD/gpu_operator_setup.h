/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_OPERATOR_SETUP_H
#define GPU_OPERATOR_SETUP_H

#include <stdexcept>
#include <string>
#include <vector>

class Operator;
class Operator_Extension;
class Operator_Ext_TFSF;
class CSPrimitives;

struct GPU_SetupModel
{
	Operator* op;
	unsigned int dims[3], offsets[3], average, stores, primitive_count;
	int boundary[6];
	unsigned int bc_size[6];
	std::vector<CSPrimitives*> primitives;
	std::vector<double> grid, geometry;
	std::vector<unsigned int> candidates;
};

// Only preparation may throw this: it precedes EC/waveform changes and allows a fresh CPU rebuild.
class GPU_SetupUnavailable : public std::runtime_error
{
public:
	explicit GPU_SetupUnavailable(const std::string& reason) : std::runtime_error(reason) {}
};

class GPU_OperatorSetup
{
public:
	virtual ~GPU_OperatorSetup() {}
	// EC uses AdrOp's x-fast layout; materials and coefficients use ArrayNIJK.
	virtual void CalcEC(float* ec[4][3], float* materials[4]) = 0;
	virtual double CalcTimestep(int method) = 0;
	virtual void CalcCoefficients(double dt, float* coefficients[4], std::vector<unsigned int>& used) = 0;
	virtual bool BuildExtension(Operator_Extension& extension) = 0;
	virtual bool BuildTFSF(Operator_Ext_TFSF& extension, const double origin[3]) = 0;
	virtual double LumpedGeometry(int direction, const unsigned int start[3], const unsigned int stop[3]) = 0;
	virtual void ApplyLumpedRC(int direction, const unsigned int start[3], const unsigned int stop[3],
	                         double capacitance, double resistance, double epsilon, double kappa, bool caps) = 0;
	virtual void SynchronizeOperator() = 0;
};

#endif
