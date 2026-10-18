/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef HIP_SETUP_RESULT_H
#define HIP_SETUP_RESULT_H

#include "FDTD/gpu_operator_setup.h"

struct HIP_Context;
struct HIP_CellQuery;
class Excitation;

struct HIP_SetupArray
{
	explicit HIP_SetupArray(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	~HIP_SetupArray();
	std::shared_ptr<HIP_Context> context;
	float* values=nullptr;
	size_t count=0;
	void Read(float* output,size_t offset,size_t length,bool diagnostics) const;
};

struct HIP_SetupSources
{
	explicit HIP_SetupSources(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	~HIP_SetupSources();
	std::shared_ptr<HIP_Context> context;
	GPU_SetupSourcePoint* points[2]={};
	unsigned int count[2]={};
	bool serial[2]={};
	void Materialize(unsigned int current,std::vector<GPU_SetupSourcePoint>& output,bool diagnostics) const;
	int BoundaryDelay(unsigned int normal,unsigned int line,const unsigned int dims[3],bool diagnostics) const;
};

struct HIP_SetupUPML
{
	explicit HIP_SetupUPML(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	~HIP_SetupUPML();
	std::shared_ptr<HIP_Context> context;
	unsigned int dims[3]={},mode=0,sets=0;
	float* values=nullptr;
	float* palette=nullptr;
	void* index=nullptr;
	void Materialize(float* outputs[6], bool diagnostics) const;
};

struct HIP_SetupTFSF
{
	explicit HIP_SetupTFSF(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	~HIP_SetupTFSF();
	std::shared_ptr<HIP_Context> context;
	float* planes[3][2]={};
	unsigned int counts[3][2]={};
	void Materialize(unsigned int axis,unsigned int side,std::vector<float>& output,bool diagnostics) const;
};

struct HIP_SetupSteadyState
{
	explicit HIP_SetupSteadyState(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	~HIP_SetupSteadyState();
	std::shared_ptr<HIP_Context> context;
	unsigned int period=0,count=0;
	unsigned int* indices=nullptr;
	float* records=nullptr;
};

struct HIP_SetupSparse
{
	explicit HIP_SetupSparse(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	~HIP_SetupSparse();
	struct Block
	{
		unsigned int count=0,flags=0;
		unsigned int* nodes=nullptr;
		std::vector<std::shared_ptr<HIP_SetupArray>> coefficients;
	};
	std::shared_ptr<HIP_Context> context;
	std::vector<Block> blocks;
	void ReadNodes(unsigned int block,std::vector<unsigned int>& nodes,bool diagnostics) const;
};

struct HIP_SetupMultiGrid
{
	explicit HIP_SetupMultiGrid(const std::shared_ptr<HIP_Context>& ctx) : context(ctx) {}
	~HIP_SetupMultiGrid();
	std::shared_ptr<HIP_Context> context;
	unsigned int alpha=0,split=0;
	unsigned int* positions[4]={};
	float* weights[4]={};
	void Read(GPU_MultiGridInterpolation& output,bool diagnostics) const;
};

struct HIP_SetupResult : GPU_SetupResult
{
	explicit HIP_SetupResult(const std::shared_ptr<HIP_Context>& context);
	~HIP_SetupResult();
	const char* Backend() const override {return "HIP";}
	void ReadCoefficients(unsigned int kind, size_t offset, size_t count, float* values) const override;
	void WriteCoefficient(unsigned int kind,size_t offset,float value) override;
	bool engine_adopted=false;
	std::vector<void*> coefficient_snapshots;
	void ReadMaterials(unsigned int kind,size_t offset,size_t count,float* values) const override;
	void ReadMultiGrid(GPU_MultiGridInterpolation& output) const override;
	mutable size_t requested_readback_bytes=0;
	size_t pending_runtime_bytes=0;
	std::shared_ptr<HIP_Context> context;
	unsigned int dims[3], mode, sets;
	unsigned int cylindrical=0,closed_alpha=0,axis=0;
	float* coefficients[4];
	float* materials[4]={};
	float* palette;
	void* index;

	Excitation* waveform_source;
	unsigned int waveform_length;
	float* waveform[2];
	std::shared_ptr<float> waveform_owner[2];
	std::map<const Operator_Extension*,std::shared_ptr<HIP_SetupUPML>> upml;
	std::map<const Operator_Extension*,std::shared_ptr<HIP_SetupSources>> sources;
	std::map<const Operator_Extension*,std::shared_ptr<HIP_SetupTFSF>> tfsf;
	std::map<const Operator_Extension*,std::shared_ptr<HIP_SetupArray>> extension_tables;
	std::map<const Operator_Extension*,std::shared_ptr<HIP_SetupSteadyState>> steady_state;
	std::map<const Operator_Extension*,std::shared_ptr<HIP_SetupSparse>> sparse;
	std::shared_ptr<HIP_SetupMultiGrid> multigrid;
	std::shared_ptr<HIP_CellQuery> cell_query;
};

// The input is immutable, component-major coefficient arrays. Packing preserves
// the existing GPU_FindSets compression threshold and bitwise set equality.
size_t HIP_PackSetupCoefficients(HIP_SetupResult& result);
size_t HIP_SetupPackingBytes(size_t cells, unsigned int width=12);
size_t HIP_PackSetupUPML(HIP_SetupUPML& result);
bool HIP_BuildSetupWaveform(HIP_SetupResult& result, Excitation& excitation, unsigned int maxTS, bool diagnostics);

#endif
