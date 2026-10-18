/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_TYPE_KEY_H
#define GPU_SETUP_TYPE_KEY_H
#include <string>
#include <typeinfo>
// MinGW can retain separate RTTI name addresses across a DLL boundary, including
// for an explicitly passed type_info. Compare the complete ABI type name rather
// than its address/order; this still requires an exact registered concrete type.
inline std::string GPU_SetupTypeKey(const std::type_info& type) {return type.name();}
#endif
