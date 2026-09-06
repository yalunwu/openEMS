/*
 *  Copyright (C) 2026 openEMS Project
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#include "engine_cpu.h"
#include <iostream>

EngineCPU::EngineCPU(Engine* existingEngine, bool ownsEngine)
	: m_engine(existingEngine), m_ownsEngine(ownsEngine)
{
}

EngineCPU::~EngineCPU()
{
	if (m_ownsEngine && m_engine)
	{
		delete m_engine;
		m_engine = nullptr;
	}
}

bool EngineCPU::Initialize()
{
	if (m_engine)
	{
		m_engine->Init();
		return true;
	}
	return false;
}

void EngineCPU::Reset()
{
	if (m_engine)
		m_engine->Reset();
}

bool EngineCPU::IterateTS(unsigned int iterTS)
{
	if (m_engine)
		return m_engine->IterateTS(iterTS);
	return false;
}

unsigned int EngineCPU::GetNumberOfTimesteps() const
{
	if (m_engine)
		return m_engine->GetNumberOfTimesteps();
	return 0;
}

void EngineCPU::NextInterval(float curr_speed)
{
	if (m_engine)
		m_engine->NextInterval(curr_speed);
}

FDTD_FLOAT EngineCPU::GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	if (m_engine)
		return m_engine->GetVolt(n, x, y, z);
	return 0.0f;
}

FDTD_FLOAT EngineCPU::GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
{
	if (m_engine)
		return m_engine->GetCurr(n, x, y, z);
	return 0.0f;
}

void EngineCPU::SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
{
	if (m_engine)
		m_engine->SetVolt(n, x, y, z, val);
}

void EngineCPU::SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
{
	if (m_engine)
		m_engine->SetCurr(n, x, y, z, val);
}

bool EngineCPU::SyncFieldsToHost()
{
	// CPU engine fields are already resident in host RAM
	return true;
}

bool EngineCPU::SyncProbesToHost()
{
	// CPU engine probes are sampled directly in host RAM
	return true;
}

std::string EngineCPU::GetBackendName() const
{
	return "CPU";
}
