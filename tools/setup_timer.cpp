/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#include "setup_timer.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

size_t SetupHostPeakBytes()
{
#ifdef _WIN32
	PROCESS_MEMORY_COUNTERS memory={};memory.cb=sizeof(memory);
	return K32GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory)) ? size_t(memory.PeakWorkingSetSize) : 0;
#elif defined(__unix__) || defined(__APPLE__)
	rusage usage={};if (getrusage(RUSAGE_SELF,&usage)!=0) return 0;
#ifdef __APPLE__
	return size_t(usage.ru_maxrss);
#else
	return size_t(usage.ru_maxrss)*1024;
#endif
#else
	return 0;
#endif
}
