/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef HIP_SETUP_SCAN_H
#define HIP_SETUP_SCAN_H
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

size_t HIP_SetupScanWorkspace(size_t count);
void HIP_SetupExclusiveScan(uint64_t* values, size_t count, uint64_t* workspace, hipStream_t stream);
#endif
