/*
 *  Copyright (C) 2026 openEMS Project
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#ifndef ENGINE_CPU_H
#define ENGINE_CPU_H

#include "engine_backend.h"
#include "engine.h"

//! CPU simulation backend wrapping openEMS's existing multithreaded / SSE / basic FDTD engines.
class EngineCPU : public EngineBackend
{
public:
	explicit EngineCPU(Engine* existingEngine, bool ownsEngine = true);
	virtual ~EngineCPU();

	bool Initialize() override;
	void Reset() override;
	bool IterateTS(unsigned int iterTS) override;
	unsigned int GetNumberOfTimesteps() const override;
	void NextInterval(float curr_speed) override;

	FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
	FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const override;
	void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val) override;
	void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val) override;

	bool SyncFieldsToHost() override;
	bool SyncProbesToHost() override;

	std::string GetBackendName() const override;

	Engine* GetEngine() const { return m_engine; }

private:
	Engine* m_engine;
	bool m_ownsEngine;
};

#endif // ENGINE_CPU_H
