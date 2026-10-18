/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "FDTD/gpu_setup_input_export.h"
#include "FDTD/gpu_operator_setup.h"
#include "CSPropDiscMaterial.h"
#include "CSPropExcitation.h"
#include "ParameterObjects.h"
#include <iostream>
#include <stdexcept>

int main()
{
	try
	{
		ParameterSet parameters;GPU_SetupModel model;
		CSPropDiscMaterial material(&parameters);CSPropExcitation source(&parameters);
		bool rejected=false;
		try {GPU_ExportSetupDiscrete(material,model);} catch (const GPU_SetupUnavailable&) {rejected=true;}
		if (!rejected || !model.discrete_materials.empty()) throw std::runtime_error("discrete material was accepted or changed inputs");
		rejected=false;
		try {GPU_ExportSetupFileWeight(source,model);} catch (const GPU_SetupUnavailable&) {rejected=true;}
		if (!rejected || !model.file_weights.empty()) throw std::runtime_error("file-weighted source was accepted or changed inputs");
		std::cout << "PASS stock-dependency input rejection before construction" << std::endl;
		return 0;
	}
	catch (const std::exception& error) {std::cerr << error.what() << std::endl;return 1;}
}
