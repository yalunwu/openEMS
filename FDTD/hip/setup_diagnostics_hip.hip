/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#define OPENEMS_HIP_RAW_RUNTIME
#include "hip_internal.h"
#include <atomic>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>

namespace
{
struct Allocation {HIP_Context* context;size_t bytes;bool pinned;};
std::mutex mutex;
std::map<hipStream_t,HIP_Context*> streams;
std::map<uintptr_t,Allocation> allocations;
std::atomic<unsigned int> active{0};
size_t Add(size_t a,size_t b) {return b>std::numeric_limits<size_t>::max()-a ? std::numeric_limits<size_t>::max() : a+b;}
HIP_Context* Owner(const void* pointer)
{
	const auto address=reinterpret_cast<uintptr_t>(pointer);auto found=allocations.upper_bound(address);
	if (found==allocations.begin()) return nullptr;
	--found;return !found->second.pinned && address-found->first<found->second.bytes ? found->second.context : nullptr;
}
HIP_Context* StreamOwner(hipStream_t stream)
{const auto found=streams.find(stream);return found==streams.end() ? nullptr : found->second;}
void Transfer(void* output,const void* input,size_t bytes,hipMemcpyKind kind,hipStream_t stream)
{
	if (!active.load(std::memory_order_relaxed)) return;
	std::lock_guard<std::mutex> lock(mutex);auto* context=StreamOwner(stream);
	if (!context) context=kind==hipMemcpyHostToDevice ? Owner(output) : Owner(input);
	if (!context) return;
	auto& stats=context->setup_diagnostics;
	if (kind==hipMemcpyDefault)
	{
		const bool from=Owner(input)!=nullptr,to=Owner(output)!=nullptr;
		kind=from ? (to ? hipMemcpyDeviceToDevice : hipMemcpyDeviceToHost) : (to ? hipMemcpyHostToDevice : hipMemcpyHostToHost);
	}
	if (kind==hipMemcpyHostToDevice) stats.h2d=Add(stats.h2d,bytes);
	if (kind==hipMemcpyDeviceToHost) stats.d2h=Add(stats.d2h,bytes);
	if (kind==hipMemcpyDeviceToDevice) stats.d2d=Add(stats.d2d,bytes);
}
hipError_t Allocate(hipStream_t stream,void** output,size_t bytes,bool pinned)
{
	const auto status=pinned ? hipHostMalloc(output,bytes) : hipMalloc(output,bytes);
	if (status!=hipSuccess || !active.load(std::memory_order_relaxed)) return status;
	try
	{
		std::lock_guard<std::mutex> lock(mutex);auto* context=StreamOwner(stream);if (!context) return status;
		allocations.emplace(reinterpret_cast<uintptr_t>(*output),Allocation{context,bytes,pinned});
		auto& stats=context->setup_diagnostics;auto& live=pinned ? stats.pinned : stats.live;auto& peak=pinned ? stats.pinned_peak : stats.peak;
		live=Add(live,bytes);peak=(std::max)(peak,live);++stats.allocations;
	}
	catch (...) {if (pinned) hipHostFree(*output);else hipFree(*output);*output=nullptr;return hipErrorOutOfMemory;}
	return status;
}
void Released(void* pointer)
{
	if (!active.load(std::memory_order_relaxed)) return;
	std::lock_guard<std::mutex> lock(mutex);const auto found=allocations.find(reinterpret_cast<uintptr_t>(pointer));
	if (found==allocations.end()) return;
	auto& stats=found->second.context->setup_diagnostics;auto& live=found->second.pinned ? stats.pinned : stats.live;
	live-=found->second.bytes;allocations.erase(found);
}
}

void HIP_RegisterDiagnostics(HIP_Context* context)
{
	if (!context->setup_diagnostics.enabled) return;
	std::lock_guard<std::mutex> lock(mutex);
	try {streams.emplace(context->stream,context);streams.emplace(context->copy_stream,context);}
	catch (...) {streams.erase(context->stream);streams.erase(context->copy_stream);throw;}
	++active;
}
void HIP_UnregisterDiagnostics(HIP_Context* context)
{
	std::lock_guard<std::mutex> lock(mutex);const auto found=streams.find(context->stream);
	if (found==streams.end() || found->second!=context) return;
	streams.erase(context->stream);streams.erase(context->copy_stream);
	for (auto item=allocations.begin();item!=allocations.end();) {if (item->second.context==context) item=allocations.erase(item);else ++item;}
	--active;
}
void HIP_ReportSetupDiagnostics(HIP_Context* context)
{
	if (!context->setup_diagnostics.enabled) return;
	std::lock_guard<std::mutex> lock(mutex);auto& s=context->setup_diagnostics;if (s.reported) return;s.reported=true;
	std::cout << "GPU whole-context setup diagnostics: h2d_bytes=" << s.h2d << " d2h_bytes=" << s.d2h << " d2d_bytes=" << s.d2d
	          << " device_allocation_peak_bytes=" << s.peak << " device_live_bytes=" << s.live << " pinned_host_peak_bytes=" << s.pinned_peak
	          << " device_zero_bytes=" << s.zeroed << " allocations=" << s.allocations << std::endl;
}
hipError_t HIP_AllocateDeviceRaw(hipStream_t stream,void** output,size_t bytes) {return Allocate(stream,output,bytes,false);}
hipError_t HIP_AllocateHostRaw(hipStream_t stream,void** output,size_t bytes) {return Allocate(stream,output,bytes,true);}
hipError_t HIP_TrackedFree(void* pointer) {const auto status=hipFree(pointer);if (status==hipSuccess) Released(pointer);return status;}
hipError_t HIP_TrackedHostFree(void* pointer) {const auto status=hipHostFree(pointer);if (status==hipSuccess) Released(pointer);return status;}
hipError_t HIP_TrackedMemcpy(void* output,const void* input,size_t bytes,hipMemcpyKind kind)
{const auto status=hipMemcpy(output,input,bytes,kind);if (status==hipSuccess) Transfer(output,input,bytes,kind,0);return status;}
hipError_t HIP_TrackedMemcpyAsync(void* output,const void* input,size_t bytes,hipMemcpyKind kind,hipStream_t stream)
{const auto status=hipMemcpyAsync(output,input,bytes,kind,stream);if (status==hipSuccess) Transfer(output,input,bytes,kind,stream);return status;}
hipError_t HIP_TrackedMemsetAsync(void* output,int value,size_t bytes,hipStream_t stream)
{
	const auto status=hipMemsetAsync(output,value,bytes,stream);
	if (status==hipSuccess && active.load(std::memory_order_relaxed))
	{
		std::lock_guard<std::mutex> lock(mutex);auto* context=StreamOwner(stream);if (!context) context=Owner(output);
		if (context) context->setup_diagnostics.zeroed=Add(context->setup_diagnostics.zeroed,bytes);
	}
	return status;
}
