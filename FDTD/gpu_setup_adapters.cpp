/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "gpu_setup_adapters.h"
#include "gpu_operator_setup.h"
#include "CSPropMaterial.h"
#include "CSPropMetal.h"
#include "CSPropExcitation.h"
#include "CSPropConductingSheet.h"
#include "CSPropLorentzMaterial.h"
#include "CSPropDebyeMaterial.h"
#include "CSPropDiscMaterial.h"
#include "CSPropLumpedElement.h"
#include "CSPropAbsorbingBC.h"
#include "CSPropProbeBox.h"
#include "CSPropDumpBox.h"
#include "CSPropResBox.h"
#include "CSPropUnknown.h"
#include "extensions/operator_ext_upml.h"
#include "extensions/operator_ext_excitation.h"
#include "extensions/operator_ext_tfsf.h"
#include "extensions/operator_ext_lumpedRLC.h"
#include "extensions/operator_ext_steadystate.h"
#include "extensions/operator_ext_mur_abc.h"
#include "extensions/operator_ext_absorbing_bc.h"
#include "extensions/operator_ext_lorentzmaterial.h"
#include "extensions/operator_ext_debyematerial.h"
#include "extensions/operator_ext_conductingsheet.h"
#include "extensions/operator_ext_cylinder.h"
#include "extensions/engine_ext_upml.h"
#include "extensions/engine_ext_excitation.h"
#include "extensions/engine_ext_tfsf.h"
#include "extensions/engine_ext_lumpedRLC.h"
#include "extensions/engine_ext_steadystate.h"
#include "extensions/engine_ext_mur_abc.h"
#include "extensions/engine_ext_absorbing_bc.h"
#include "extensions/engine_ext_lorentzmaterial.h"
#include "extensions/engine_ext_debyematerial.h"
#include "extensions/engine_ext_cylinder.h"
#include <map>
#include <mutex>
#include "gpu_setup_type_key.h"
#include "gpu_setup_constant.h"

namespace {
struct PropertyEntry
{
	std::string name;
	GPU_PropertyPrepare prepare;
	GPU_PropertyExport export_data;
	bool builtin;
};
struct ExtensionEntry
{
	std::string name;
	GPU_ExtensionPrepare prepare;
	GPU_SetupExtensionKind kind;
	std::string engine;
	std::function<bool(Engine_Extension&)> engine_check;
	bool builtin;
};
std::mutex& AdapterMutex() {static std::mutex mutex;return mutex;}
template<class T> void AddProperty(std::map<std::string,PropertyEntry>& entries,const char* name)
{
	entries.emplace(GPU_SetupTypeKey(typeid(T)),PropertyEntry{name,[](CSProperties& property,std::string& reason) {
		reason="incompatible property base";return dynamic_cast<T*>(&property)!=nullptr;
	},{},true});
}
std::map<std::string,PropertyEntry>& Properties()
{
	static auto entries=[]() {
		std::map<std::string,PropertyEntry> result;
		AddProperty<CSPropMaterial>(result,"Material");AddProperty<CSPropMetal>(result,"Metal");
		AddProperty<CSPropExcitation>(result,"Excitation");AddProperty<CSPropConductingSheet>(result,"ConductingSheet");
		AddProperty<CSPropLorentzMaterial>(result,"LorentzMaterial");AddProperty<CSPropDebyeMaterial>(result,"DebyeMaterial");
		AddProperty<CSPropDiscMaterial>(result,"DiscMaterial");AddProperty<CSPropLumpedElement>(result,"LumpedElement");
		AddProperty<CSPropAbsorbingBC>(result,"AbsorbingBC");AddProperty<CSPropProbeBox>(result,"ProbeBox");
		AddProperty<CSPropDumpBox>(result,"DumpBox");AddProperty<CSPropResBox>(result,"ResBox");
		AddProperty<CSPropUnknown>(result,"Unknown");return result;
	}();return entries;
}
template<class O,class E> void AddExtension(std::map<std::string,ExtensionEntry>& entries,GPU_SetupExtensionKind kind,const char* name)
{
	entries.emplace(GPU_SetupTypeKey(typeid(O)),ExtensionEntry{name,[](Operator_Extension& extension,std::string& reason) {
		reason="incompatible operator base or missing device stepping support";
		return dynamic_cast<O*>(&extension)!=nullptr && extension.IsGPUSave();
	},kind,GPU_SetupTypeKey(typeid(E)),[](Engine_Extension& extension) {return dynamic_cast<E*>(&extension)!=nullptr;},true});
}
std::map<std::string,ExtensionEntry>& Extensions()
{
	static auto entries=[]() {
		std::map<std::string,ExtensionEntry> result;
		AddExtension<Operator_Ext_UPML,Engine_Ext_UPML>(result,GPU_SetupExtensionKind::UPML,"UPML");
		AddExtension<Operator_Ext_Excitation,Engine_Ext_Excitation>(result,GPU_SetupExtensionKind::Excitation,"Excitation");
		AddExtension<Operator_Ext_TFSF,Engine_Ext_TFSF>(result,GPU_SetupExtensionKind::TFSF,"TFSF");
		AddExtension<Operator_Ext_LumpedRLC,Engine_Ext_LumpedRLC>(result,GPU_SetupExtensionKind::RLC,"LumpedRLC");
		AddExtension<Operator_Ext_SteadyState,Engine_Ext_SteadyState>(result,GPU_SetupExtensionKind::SteadyState,"SteadyState");
		AddExtension<Operator_Ext_Mur_ABC,Engine_Ext_Mur_ABC>(result,GPU_SetupExtensionKind::Mur,"Mur");
		AddExtension<Operator_Ext_Absorbing_BC,Engine_Ext_Absorbing_BC>(result,GPU_SetupExtensionKind::Absorbing,"AbsorbingBC");
		AddExtension<Operator_Ext_LorentzMaterial,Engine_Ext_LorentzMaterial>(result,GPU_SetupExtensionKind::Lorentz,"LorentzMaterial");
		AddExtension<Operator_Ext_DebyeMaterial,Engine_Ext_DebyeMaterial>(result,GPU_SetupExtensionKind::Debye,"DebyeMaterial");
		AddExtension<Operator_Ext_ConductingSheet,Engine_Ext_LorentzMaterial>(result,GPU_SetupExtensionKind::Sheet,"ConductingSheet");
		AddExtension<Operator_Ext_Cylinder,Engine_Ext_Cylinder>(result,GPU_SetupExtensionKind::Cylinder,"Cylinder");return result;
	}();return entries;
}
ExtensionEntry FindExtension(Operator_Extension& extension)
{
	std::lock_guard<std::mutex> lock(AdapterMutex());const auto found=Extensions().find(GPU_SetupTypeKey(typeid(extension)));
	if (found==Extensions().end()) throw GPU_SetupUnavailable("no device extension adapter for "+extension.GetExtensionName());
	return found->second;
}
}

void GPU_RegisterPropertyAdapter(const std::type_info& type,const std::type_info& semantic,const std::string& name,
                                 GPU_PropertyPrepare prepare,GPU_PropertyExport export_data)
{
	if (name.empty() || !prepare) throw std::invalid_argument("property adapter needs a name and preparation check");
	std::lock_guard<std::mutex> lock(AdapterMutex());const auto base=Properties().find(GPU_SetupTypeKey(semantic));
	if (base==Properties().end() || !base->second.builtin) throw std::invalid_argument("unknown built-in property semantics");
	if (Properties().count(GPU_SetupTypeKey(type))) throw std::invalid_argument("property adapter is already registered");
	const auto check=base->second.prepare;
	Properties().emplace(GPU_SetupTypeKey(type),PropertyEntry{name,[check,prepare](CSProperties& property,std::string& reason) {
		return check(property,reason) && prepare(property,reason);
	},export_data,false});
}
void GPU_UnregisterPropertyAdapter(const std::type_info& type)
{
	std::lock_guard<std::mutex> lock(AdapterMutex());const auto found=Properties().find(GPU_SetupTypeKey(type));
	if (found!=Properties().end() && found->second.builtin) throw std::invalid_argument("built-in property adapters cannot be removed");
	Properties().erase(GPU_SetupTypeKey(type));
}
GPU_PropertyAdapter GPU_PreparePropertyAdapter(CSProperties& property)
{
	if (dynamic_cast<CSPropDiscMaterial*>(&property))
		throw GPU_SetupUnavailable("discrete materials require CPU operator construction");
	if (auto* probe=dynamic_cast<CSPropProbeBox*>(&property))
		if (probe->GetProbeType()==10 || probe->GetProbeType()==11)
		{
			if (!probe->GetModeFile().empty())
				throw GPU_SetupUnavailable("file-based mode matching requires CPU operator construction");
			for (int n=0;n<3;++n)
				if (!GPU_SetupConstant(probe->GetModeFunction(n)))
					throw GPU_SetupUnavailable("spatial mode functions require CPU operator construction");
		}
	PropertyEntry entry;
	{
		std::lock_guard<std::mutex> lock(AdapterMutex());const auto found=Properties().find(GPU_SetupTypeKey(typeid(property)));
		if (found==Properties().end()) throw GPU_SetupUnavailable("no device property adapter for "+property.GetTypeXMLString()+" ("+typeid(property).name()+")");
		entry=found->second;
	}
	std::string reason;if (!entry.prepare(property,reason)) throw GPU_SetupUnavailable(entry.name+": "+reason);
	return {entry.name,entry.export_data};
}
void GPU_RegisterExtensionAdapter(const std::type_info& type,const std::type_info& semantic,const std::type_info& engine,
                                  const std::string& name,GPU_ExtensionPrepare prepare)
{
	if (name.empty() || !prepare) throw std::invalid_argument("extension adapter needs a name and preparation check");
	std::lock_guard<std::mutex> lock(AdapterMutex());const auto base=Extensions().find(GPU_SetupTypeKey(semantic));
	if (base==Extensions().end() || !base->second.builtin) throw std::invalid_argument("unknown built-in extension semantics");
	if (Extensions().count(GPU_SetupTypeKey(type))) throw std::invalid_argument("extension adapter is already registered");
	auto entry=base->second;const auto check=entry.prepare;
	entry.name=name;entry.builtin=false;entry.engine=GPU_SetupTypeKey(engine);
	entry.prepare=[check,prepare](Operator_Extension& extension,std::string& reason) {return check(extension,reason) && prepare(extension,reason);};
	Extensions().emplace(GPU_SetupTypeKey(type),entry);
}
void GPU_UnregisterExtensionAdapter(const std::type_info& type)
{
	std::lock_guard<std::mutex> lock(AdapterMutex());const auto found=Extensions().find(GPU_SetupTypeKey(type));
	if (found!=Extensions().end() && found->second.builtin) throw std::invalid_argument("built-in extension adapters cannot be removed");
	Extensions().erase(GPU_SetupTypeKey(type));
}
GPU_SetupExtensionKind GPU_PrepareExtensionAdapter(Operator_Extension& extension)
{
	const auto entry=FindExtension(extension);std::string reason;
	if (!entry.prepare(extension,reason)) throw GPU_SetupUnavailable(entry.name+": "+reason);
	return entry.kind;
}
void GPU_CheckEngineAdapter(Engine_Extension& extension)
{
	if (!extension.GetOperatorExtension()) throw std::runtime_error("device engine extension has no operator extension");
	const auto entry=FindExtension(*extension.GetOperatorExtension());
	if (entry.engine!=GPU_SetupTypeKey(typeid(extension)) || !entry.engine_check(extension))
		throw std::runtime_error(entry.name+": engine extension does not match the registered device adapter");
}
