/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_ADAPTERS_H
#define GPU_SETUP_ADAPTERS_H
#include <functional>
#include <string>
#include <typeinfo>
class CSProperties;
class Operator_Extension;
class Engine_Extension;
struct GPU_SetupModel;
struct GPU_SetupProperty;

enum class GPU_SetupExtensionKind {UPML, Excitation, TFSF, RLC, SteadyState, Mur, Absorbing, Lorentz, Debye, Sheet, Cylinder};
using GPU_PropertyPrepare=std::function<bool(CSProperties&,std::string&)>;
using GPU_PropertyExport=std::function<void(CSProperties&,GPU_SetupModel&,GPU_SetupProperty&)>;
using GPU_ExtensionPrepare=std::function<bool(Operator_Extension&,std::string&)>;
struct GPU_PropertyAdapter
{
	std::string name;
	GPU_PropertyExport export_data;
};

// Registrations explicitly opt into a built-in base's device semantics. The
// check must reject changed host callbacks that the serialized data cannot
// represent. A property exporter can extend/replace its base descriptor using
// the model's immutable expression, voxel and file buffers. No mesh sweep is
// permitted in preparation or export callbacks.
void GPU_RegisterPropertyAdapter(const std::type_info& type,const std::type_info& semantic,const std::string& name,
                                 GPU_PropertyPrepare prepare,GPU_PropertyExport export_data={});
void GPU_UnregisterPropertyAdapter(const std::type_info& type);
GPU_PropertyAdapter GPU_PreparePropertyAdapter(CSProperties& property);
void GPU_RegisterExtensionAdapter(const std::type_info& type,const std::type_info& semantic,const std::type_info& engine,
                                  const std::string& name,GPU_ExtensionPrepare prepare);
void GPU_UnregisterExtensionAdapter(const std::type_info& type);
GPU_SetupExtensionKind GPU_PrepareExtensionAdapter(Operator_Extension& extension);
void GPU_CheckEngineAdapter(Engine_Extension& extension);
#endif
