/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_EXPRESSION_H
#define GPU_SETUP_EXPRESSION_H
#include <cmath>
#include <cstdint>

// Only coordinate-independent constants are prepared by the stock host parser.
// These records belong to openEMS and contain no parser internals or bytecode.
struct GPU_SetupInstruction {uint32_t opcode, operand, operand2;};
struct GPU_SetupExpression
{
	uint32_t code_offset, code_count, immediate_offset, immediate_count, stack_size, variables, flags;
	double epsilon;
};
struct GPU_SetupExpressionView
{
	const GPU_SetupExpression* programs;
	const GPU_SetupInstruction* instructions;
	const double* constants;
	double* stack;
	uint32_t capacity;
	int* error;
};
#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_SETUP_EVAL __host__ __device__
#else
#define GPU_SETUP_EVAL
#endif
GPU_SETUP_EVAL inline bool GPU_SetupFinite(double value)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
	return ::isfinite(value);
#else
	return std::isfinite(value);
#endif
}
GPU_SETUP_EVAL inline double GPU_SetupEvaluate(const GPU_SetupExpression& d,
	const GPU_SetupInstruction* instructions, const double* constants,
	const double*, double*, uint32_t capacity, int& error)
{
	error=0;
	if (!d.code_count) return 1;
	if (d.code_count!=1 || d.immediate_count!=1 || d.stack_size!=1 || d.variables ||
	    capacity<1 || !instructions || !constants || instructions[d.code_offset].opcode!=1)
	{error=6;return 0;}
	const double value=constants[d.immediate_offset];
	if (!GPU_SetupFinite(value)) {error=7;return 0;}
	return value;
}
#undef GPU_SETUP_EVAL
#endif
