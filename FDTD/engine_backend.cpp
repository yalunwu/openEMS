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
#include "extensions/operator_ext_mur_abc.h"
#include "extensions/operator_ext_steadystate.h"
#include "extensions/operator_ext_tfsf.h"
#include "extensions/operator_ext_lumpedRLC.h"
#include "extensions/operator_ext_absorbing_bc.h"
#include "extensions/operator_ext_lorentzmaterial.h"
#include "extensions/operator_ext_conductingsheet.h"
#include "extensions/operator_ext_cylinder.h"
#include "operator_cylindermultigrid.h"

bool EngineBackend::CheckModelSupport(const Operator* op, const ContinuousStructure*, std::string& unsupportedReason)
{
	if (!op)
	{
		unsupportedReason = "Invalid operator pointer.";
		return false;
	}

	if (dynamic_cast<const Operator_CylinderMultiGrid*>(op))
	{
		unsupportedReason = "Cylindrical multi-grid is not supported on Vulkan.";
		return false;
	}

	for (size_t i = 0; i < op->GetNumberOfExtentions(); ++i)
	{
		Operator_Extension* extension = op->GetExtension(i);
		if (dynamic_cast<Operator_Ext_Excitation*>(extension))
			continue;
		if (dynamic_cast<Operator_Ext_UPML*>(extension))
			continue;
		if (dynamic_cast<Operator_Ext_Mur_ABC*>(extension))
			continue;
		if (dynamic_cast<Operator_Ext_SteadyState*>(extension))
			continue;
		if (dynamic_cast<Operator_Ext_TFSF*>(extension))
			continue;
		if (dynamic_cast<Operator_Ext_LumpedRLC*>(extension))
			continue;
		if (dynamic_cast<Operator_Ext_Absorbing_BC*>(extension))
			continue;
		if (dynamic_cast<Operator_Ext_LorentzMaterial*>(extension))
			continue;
		if (dynamic_cast<Operator_Ext_ConductingSheet*>(extension))
			continue;
		if (dynamic_cast<Operator_Ext_Cylinder*>(extension))
			continue;

		unsupportedReason = extension->GetExtensionName() + " is not supported on Vulkan.";
		return false;
	}

	// Every operator extension attached to the model has a Vulkan implementation.
	return true;
}
