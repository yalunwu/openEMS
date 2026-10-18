/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_OPERATOR_SETUP_H
#define GPU_OPERATOR_SETUP_H

#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include "gpu_setup_descriptors.h"
#include "gpu_setup_expression_export.h"

class Operator;
class Operator_Extension;
class Operator_Ext_TFSF;
class CSPrimitives;
struct GPU_MultiGridInterpolation;

struct GPU_SetupModel
{
	Operator* op;
	unsigned int dims[3], offsets[3], average, stores, primitive_count;
	unsigned int cylindrical=0,closed_alpha=0,axis=0,first_radius=0;
	bool cell_materials=false;
	double grid_unit=1;
	int boundary[6];
	unsigned int bc_size[6];
	size_t geometry_index_nodes, geometry_index_scratch;
	std::vector<CSPrimitives*> primitives;
	std::vector<double> grid, geometry;
	std::vector<double> child_alpha;
	std::vector<GPU_SetupPrimitive> shapes;
	std::vector<GPU_SetupProperty> properties;
	// Primitive orders are the CPU's priority/tie order, exported once for each
	// property mask. Row filtering and compaction happen on the device.
	std::vector<unsigned int> candidate_order;
	unsigned int candidate_offsets[4];
	GPU_SetupExpressionBank expressions;
	std::vector<GPU_SetupDiscreteMaterial> discrete_materials;
	std::vector<float> material_data;
	std::vector<uint8_t> material_indices;
	std::vector<GPU_SetupFileWeight> file_weights;
	std::vector<GPU_SetupDispersionPole> dispersion;

};

// A result outlives the construction workspace and is owned by the backend.
// Implementations must keep their device/context alive and release every buffer
// on failed initialization as well as on normal engine destruction.
class GPU_SetupResult
{
public:
	virtual ~GPU_SetupResult() {}
	virtual const char* Backend() const = 0;
	virtual void ReadCoefficients(unsigned int kind, size_t offset, size_t count, float* values) const = 0;
	virtual void WriteCoefficient(unsigned int kind,size_t index,float value) {throw std::logic_error("construction result is immutable");}
	virtual void ReadMaterials(unsigned int kind,size_t offset,size_t count,float* values) const {throw std::logic_error("construction result has no material storage");}
	virtual void ReadMultiGrid(GPU_MultiGridInterpolation&) const {throw std::logic_error("construction result has no multigrid interpolation");}
};

// Unsupported input or a recoverable resource shortage requires a fresh CPU
// operator. Never continue construction with partially modified arrays.
class GPU_SetupUnavailable : public std::runtime_error
{
public:
	explicit GPU_SetupUnavailable(const std::string& reason) : std::runtime_error(reason) {}
};

class GPU_SetupTimestepChange : public std::runtime_error
{
public:
	explicit GPU_SetupTimestepChange(double value) : std::runtime_error("multigrid shared timestep changed"),timestep(value) {}
	double timestep;
};

class GPU_OperatorSetup
{
public:
	virtual ~GPU_OperatorSetup() {}
	virtual std::unique_ptr<GPU_SetupResult> Finish() = 0;
	virtual bool BuildMultiGrid() {return false;}
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
	virtual void ReadCoefficients(unsigned int kind, size_t offset, size_t count, float* values) = 0;
	virtual void WriteCoefficient(unsigned int kind,size_t index,float value) {throw std::logic_error("construction workspace is immutable");}
	virtual void ReadMaterials(unsigned int kind,size_t offset,size_t count,float* values) {throw std::logic_error("construction workspace has no material storage");}
	virtual int SnapBox2Mesh(const double* start, const double* stop, unsigned int* first, unsigned int* last, bool dual, bool full, int method) = 0;
	virtual double CalcNumericPhaseVelocity(const unsigned int* start, const unsigned int* stop, const double* direction, float frequency) = 0;
};

#endif
