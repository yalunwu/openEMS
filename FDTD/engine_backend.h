/*
 *  Copyright (C) 2026 openEMS Project
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#ifndef ENGINE_BACKEND_H
#define ENGINE_BACKEND_H

#include <string>
#include <memory>
#include <vector>
#include "tools/global.h"
#include "tools/constants.h"
#include "openems_global.h"

class Operator;
class ContinuousStructure;
class Engine;
class ProcessingArray;

//! Abstract base class for all pluggable simulation backends (CPU, Vulkan, etc.)
class OPENEMS_EXPORT EngineBackend
{
public:
	virtual ~EngineBackend() = default;

	//! Initialize backend resources (buffers, devices, queues)
	virtual bool Initialize() = 0;

	//! Reset the backend state
	virtual void Reset() = 0;

	//! Iterate a specified number of timesteps on this backend
	virtual bool IterateTS(unsigned int iterTS) = 0;

	//! Get total completed timesteps
	virtual unsigned int GetNumberOfTimesteps() const = 0;

	//! Notify backend of interval speed update
	virtual void NextInterval(float curr_speed) { UNUSED(curr_speed); }

	//! Access voltage (electric field) component
	virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const = 0;

	//! Access current (magnetic field) component
	virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const = 0;

	//! Set voltage component
	virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val) = 0;

	//! Set current component
	virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val) = 0;

	//! Synchronize on-device field buffers to host CPU memory (e.g. for field dumps)
	virtual bool SyncFieldsToHost() = 0;

	//! Synchronize point probe and line integral values from device to host
	virtual bool SyncProbesToHost() = 0;

	//! Register probe points/integrals for efficient on-device extraction
	virtual void RegisterProbes(const ProcessingArray* pa) { UNUSED(pa); }

	//! Human-readable backend name (e.g. "CPU-Multithreaded", "Vulkan (NVIDIA GeForce RTX 3070)")
	virtual std::string GetBackendName() const = 0;

	//! Capability scanner: returns true if the model features are supported on this backend type
	static bool CheckModelSupport(const Operator* op, const ContinuousStructure* csx, std::string& unsupportedReason);
};

#endif // ENGINE_BACKEND_H
