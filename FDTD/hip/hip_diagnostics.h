/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef HIP_DIAGNOSTICS_H
#define HIP_DIAGNOSTICS_H
#include <hip/hip_runtime.h>
#include <cstddef>

struct HIP_Context;
struct HIP_SetupDiagnostics
{
	bool enabled=false,reported=false;
	size_t h2d=0,d2h=0,d2d=0,zeroed=0,live=0,peak=0,pinned=0,pinned_peak=0,allocations=0;
};
void HIP_RegisterDiagnostics(HIP_Context* context);
void HIP_UnregisterDiagnostics(HIP_Context* context);
void HIP_ReportSetupDiagnostics(HIP_Context* context);
hipError_t HIP_AllocateDeviceRaw(hipStream_t stream,void** output,size_t bytes);
hipError_t HIP_AllocateHostRaw(hipStream_t stream,void** output,size_t bytes);
template<class T> hipError_t HIP_AllocateDevice(hipStream_t stream,T** output,size_t bytes)
{return HIP_AllocateDeviceRaw(stream,reinterpret_cast<void**>(output),bytes);}
template<class T> hipError_t HIP_AllocateHost(hipStream_t stream,T** output,size_t bytes)
{return HIP_AllocateHostRaw(stream,reinterpret_cast<void**>(output),bytes);}
hipError_t HIP_TrackedFree(void* pointer);
hipError_t HIP_TrackedHostFree(void* pointer);
hipError_t HIP_TrackedMemcpy(void* output,const void* input,size_t bytes,hipMemcpyKind kind);
hipError_t HIP_TrackedMemcpyAsync(void* output,const void* input,size_t bytes,hipMemcpyKind kind,hipStream_t stream=0);
hipError_t HIP_TrackedMemsetAsync(void* output,int value,size_t bytes,hipStream_t stream=0);
#ifndef OPENEMS_HIP_RAW_RUNTIME
#define hipFree HIP_TrackedFree
#define hipHostFree HIP_TrackedHostFree
#define hipMemcpy HIP_TrackedMemcpy
#define hipMemcpyAsync HIP_TrackedMemcpyAsync
#define hipMemsetAsync HIP_TrackedMemsetAsync
#endif
#endif
