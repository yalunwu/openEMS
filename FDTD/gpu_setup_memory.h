/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_MEMORY_H
#define GPU_SETUP_MEMORY_H

#include "gpu_operator_setup.h"
#include <limits>
#include <cerrno>
#include <cstdlib>

inline size_t GPU_SetupMemoryLimit()
{
	const char* value=std::getenv("OPENEMS_GPU_SETUP_MEMORY_LIMIT");
	if (!value) return std::numeric_limits<size_t>::max();
	bool digits=*value!=0;
	for (const char* p=value;*p;++p) digits=digits && *p>='0' && *p<='9';
	char* end=nullptr;errno=0;const unsigned long long limit=std::strtoull(value,&end,10);
	if (!digits || end==value || *end || errno==ERANGE || limit>std::numeric_limits<size_t>::max())
		throw std::invalid_argument("OPENEMS_GPU_SETUP_MEMORY_LIMIT must be an unsigned byte count");
	return static_cast<size_t>(limit);
}

inline size_t GPU_SetupAdd(size_t a, size_t b)
{
	if (b>std::numeric_limits<size_t>::max()-a) throw GPU_SetupUnavailable("construction byte count overflow");
	return a+b;
}

inline size_t GPU_SetupMultiply(size_t a, size_t b)
{
	if (b && a>std::numeric_limits<size_t>::max()/b) throw GPU_SetupUnavailable("construction byte count overflow");
	return a*b;
}

#endif
