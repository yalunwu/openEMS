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
#include "extensions/operator_extension.h"
#include "extensions/operator_ext_excitation.h"
#include "extensions/operator_ext_upml.h"
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
		ContinuousStructure* nonConstCSX = const_cast<ContinuousStructure*>(csx);
		// Check for dispersive media
		if (nonConstCSX->GetQtyPropertyType(CSProperties::LORENTZMATERIAL) > 0)
		{
			unsupportedReason = "Dispersive material (Lorentz) is not supported on Vulkan.";
			return false;
		}

		if (nonConstCSX->GetQtyPropertyType(CSProperties::DEBYEMATERIAL) > 0)
		{
			unsupportedReason = "Dispersive material (Debye) is not supported on Vulkan.";
			return false;
		}

		// Check for conducting sheets
		if (nonConstCSX->GetQtyPropertyType(CSProperties::CONDUCTINGSHEET) > 0)
		{
			unsupportedReason = "Conducting sheets are not supported on Vulkan.";
			return false;
		}
	}

	for (size_t i = 0; i < op->GetNumberOfExtentions(); ++i)
	{
		Operator_Extension* extension = op->GetExtension(i);
		if (dynamic_cast<Operator_Ext_Excitation*>(extension))
			continue;
		if (dynamic_cast<Operator_Ext_UPML*>(extension))
			continue;

		unsupportedReason = extension->GetExtensionName() + " is not supported on Vulkan.";
		return false;
	}

	// Model is compatible with the standard Cartesian Yee + UPML + excitation GPU pipeline.
	return true;
}
