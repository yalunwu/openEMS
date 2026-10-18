/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "gpu_setup_expression_export.h"
#include "gpu_operator_setup.h"
#include "CSFunctionParser.h"
#include "ParameterObjects.h"
#include <algorithm>
#include <cmath>

uint32_t GPU_CompileSetupExpression(const std::string& expression, const std::string&,
                                   GPU_SetupExpressionBank& bank, ParameterSet* parameters, bool csxFunctions)
{
	CSFunctionParser csxParser;
	FunctionParser plainParser;
	FunctionParser& parser=csxFunctions ? static_cast<FunctionParser&>(csxParser) : plainParser;
	if (parameters)
		for (size_t n=0;n<parameters->GetQtyParameter();++n)
		{
			Parameter* parameter=parameters->GetParameter(n);
			if (!parser.AddConstant(parameter->GetName(),parameter->GetValue())) throw GPU_SetupUnavailable("invalid expression parameter name");
		}
	if (parser.Parse(expression,"")>=0)
		throw GPU_SetupUnavailable("spatial expressions require CPU operator construction");
	const double value=parser.Eval(nullptr);
	if (parser.EvalError() || !std::isfinite(value))
		throw std::invalid_argument("invalid constant setup expression: "+expression);
	if (bank.instructions.size()>=UINT32_MAX || bank.constants.size()>=UINT32_MAX || bank.programs.size()>=UINT32_MAX)
		throw GPU_SetupUnavailable("constant input index range exceeded");
	GPU_SetupExpression d={static_cast<uint32_t>(bank.instructions.size()),1,
	                       static_cast<uint32_t>(bank.constants.size()),1,1,0,0,0};
	bank.instructions.push_back({1,0,0});bank.constants.push_back(value);
	bank.programs.push_back(d);bank.stack_size=1;
	return static_cast<uint32_t>(bank.programs.size()-1);
}
