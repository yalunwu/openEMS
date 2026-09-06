/*
 *  Copyright (C) 2026 openEMS Project
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#include "engine_backend.h"
#include "operator.h"
#include "ContinuousStructure.h"
#include "CSProperties.h"

bool EngineBackend::CheckModelSupport(const Operator* op, const ContinuousStructure* csx, std::string& unsupportedReason)
{
	if (!op)
	{
		unsupportedReason = "Invalid operator pointer.";
		return false;
	}

	if (csx)
	{
		// Check for dispersive media
		if (csx->GetQtyPropertyType(CSProperties::LORENTZMATERIAL) > 0)
		{
			unsupportedReason = "Dispersive material (Lorentz) is not supported on WebGPU.";
			return false;
		}

		if (csx->GetQtyPropertyType(CSProperties::DEBYEMATERIAL) > 0)
		{
			unsupportedReason = "Dispersive material (Debye) is not supported on WebGPU.";
			return false;
		}

		// Check for conducting sheets
		if (csx->GetQtyPropertyType(CSProperties::CONDUCTINGSHEET) > 0)
		{
			unsupportedReason = "Conducting sheets are not supported on WebGPU.";
			return false;
		}
	}

	// Model is compatible with standard Cartesian Yee + UPML + excitation GPU pipeline
	return true;
}
