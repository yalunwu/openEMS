/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_EXPRESSION_EXPORT_H
#define GPU_SETUP_EXPRESSION_EXPORT_H
#include "gpu_setup_expression.h"
#include <string>
#include <vector>

class ParameterSet;
struct GPU_SetupExpressionBank
{
	std::vector<GPU_SetupExpression> programs;
	std::vector<GPU_SetupInstruction> instructions;
	std::vector<double> constants;
	uint32_t stack_size;
	GPU_SetupExpressionBank() : programs(1), stack_size(0) {}
};

uint32_t GPU_CompileSetupExpression(const std::string& expression, const std::string& variables,
                                   GPU_SetupExpressionBank& bank, ParameterSet* parameters=nullptr, bool csxFunctions=true);
#endif
