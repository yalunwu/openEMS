/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "gpu_setup_mode_export.h"
#include "gpu_setup_input_export.h"
#include "CSPropExcitation.h"
#include "ParameterObjects.h"

GPU_SetupModeInput GPU_ExportModeInput(const Engine_ModeTableInput& input)
{
	GPU_SetupModeInput result;
	if (!input.file.empty())
	{
		ParameterSet parameters;CSPropExcitation source(&parameters);source.SetWeightFile(input.file);
		for (int n=0;n<3;++n) source.SetPropagationDir(n==int(input.normal) ? 1 : 0,n);
		result.file=GPU_ExportSetupFileWeight(source,result.data);
	}
	else for (int n=0;n<2;++n)
		result.expressions[n]=GPU_CompileSetupExpression(input.functions[(input.normal+n+1)%3],"x,y,z,rho,a,r,t",result.data.expressions);
	return result;
}
