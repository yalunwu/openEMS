/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "FDTD/gpu_setup_expression_export.h"
#include "FDTD/gpu_setup_upml.h"
#include "FDTD/gpu_operator_setup.h"
#include "CSFunctionParser.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef OPENEMS_DEVICE_EXPRESSION_TEST
void GPU_TestExpressionSamples(const GPU_SetupExpression&,const GPU_SetupExpressionBank&,
	const std::vector<double>&,std::vector<double>&,std::vector<int>&);
void GPU_TestDefaultGrading(const std::vector<double>&,std::vector<double>&);
#endif

int main()
{
	try
	{
		GPU_SetupExpressionBank bank;
		for (const char* term:{"1/2","if(0,1/0,1/2)","sin(pi/6)","j0(0)+j1(0)","sqrt(4)+log(e)"})
		{
			CSFunctionParser parser;
			if (parser.Parse(term,"")>=0) throw std::runtime_error("constant fixture parse failed");
			const double expected=parser.Eval(nullptr);
			const auto id=GPU_CompileSetupExpression(term,"x,y,z,t",bank);
			const auto d=bank.programs[id];
			std::vector<double> inputs(4*17),values(17);std::vector<int> errors(17);
			for (size_t i=0;i<inputs.size();++i) inputs[i]=double(i)-30;
#ifdef OPENEMS_DEVICE_EXPRESSION_TEST
			GPU_TestExpressionSamples(d,bank,inputs,values,errors);
#else
			double stack=0;
			for (size_t i=0;i<values.size();++i)
				values[i]=GPU_SetupEvaluate(d,bank.instructions.data(),bank.constants.data(),inputs.data()+4*i,&stack,1,errors[i]);
#endif
			for (size_t i=0;i<values.size();++i)
				if (errors[i] || values[i]!=expected) throw std::runtime_error("constant differs from stock parser");
		}
		for (const char* term:{"x","if(x>0,1,0)","j0(x)"})
		{
			const size_t count=bank.programs.size();bool rejected=false;
			try {GPU_CompileSetupExpression(term,"x",bank);} catch (const GPU_SetupUnavailable&) {rejected=true;}
			if (!rejected || bank.programs.size()!=count) throw std::runtime_error("spatial expression was accepted or changed inputs");
		}
		for (const char* term:{"1/0","sqrt(-1)"})
		{
			bool rejected=false;
			try {GPU_CompileSetupExpression(term,"",bank);} catch (const std::invalid_argument&) {rejected=true;}
			if (!rejected) throw std::runtime_error("invalid constant lost its numerical error");
		}
		FunctionParser parser;
		const std::string grading=" -log(1e-6)*log(2.5)/(2*dl*Z*(pow(2.5,W/dl)-1)) * pow(2.5, D/dl) ";
		if (!GPU_SetupDefaultGrading(grading) || GPU_SetupDefaultGrading("D/dl")) throw std::runtime_error("PML grading selection differs");
		if (parser.Parse(grading,"D,dl,W,Z,N")>=0) throw std::runtime_error("PML oracle parse failed");
		std::vector<double> inputs,values;
		for (double count:{3.0,6.0,8.0,12.0}) for (double delta:{1e-6,0.001,0.15})
			for (double fraction:{0.125,0.5,1.0,2.5})
				inputs.insert(inputs.end(),{fraction*delta,delta,count*delta,376.730313461});
		values.resize(inputs.size()/4);
#ifdef OPENEMS_DEVICE_EXPRESSION_TEST
		GPU_TestDefaultGrading(inputs,values);
#else
		for (size_t i=0;i<values.size();++i)
			values[i]=GPU_SetupDefaultGradingValue(inputs[4*i],inputs[4*i+1],inputs[4*i+2],inputs[4*i+3]);
#endif
		for (size_t i=0;i<values.size();++i)
		{
			double input[5]={inputs[4*i],inputs[4*i+1],inputs[4*i+2],inputs[4*i+3],inputs[4*i+2]/inputs[4*i+1]};
			const double expected=parser.Eval(input);
			if (!std::isfinite(values[i]) || std::abs(values[i]-expected)>2e-12*std::max(1.0,std::abs(expected)))
				throw std::runtime_error("default GPU PML grading differs from stock parser");
		}
		std::cout << "PASS stock-parser constants, spatial rejection and default PML grading" << std::endl;
		return 0;
	}
	catch (const std::exception& error) {std::cerr << error.what() << std::endl;return 1;}
}
