/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_GEOMETRY_EXPORT_H
#define GPU_SETUP_GEOMETRY_EXPORT_H

#include "gpu_setup_descriptors.h"
#include "gpu_setup_expression_export.h"
#include <vector>
#include <functional>
#include <string>
#include <typeinfo>

class CSPrimitives;
using GPU_PrimitivePrepare=std::function<bool(CSPrimitives&,std::string&)>;
using GPU_PrimitiveExport=std::function<GPU_SetupPrimitive(CSPrimitives&,std::vector<double>&,GPU_SetupExpressionBank*)>;
void GPU_RegisterPrimitiveAdapter(const std::type_info& type,const std::string& name,GPU_PrimitivePrepare prepare,GPU_PrimitiveExport export_data);
void GPU_UnregisterPrimitiveAdapter(const std::type_info& type);
// An adapter may explicitly reuse a built-in base's semantics after checking
// that its subclass has the same queries. Compatibility is never inherited.
GPU_SetupPrimitive GPU_DescribePrimitiveAs(CSPrimitives& primitive,const std::type_info& semantic_type,std::vector<double>& data,GPU_SetupExpressionBank* expressions=nullptr);
GPU_SetupPrimitive GPU_DescribePrimitive(CSPrimitives& primitive, std::vector<double>& data, GPU_SetupExpressionBank* expressions=nullptr);
void GPU_SetupExportInverse(const double matrix[16], double inverse[12]);

#endif
