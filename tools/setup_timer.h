/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SETUP_TIMER_H
#define SETUP_TIMER_H

#include <chrono>
#include <iostream>
#include "global.h"

// Phase wall times without adding synchronization to device work.
class SetupTimer
{
public:
	explicit SetupTimer(const char* owner) : m_Owner(owner), m_Start(Clock::now()) {}
	void Report(const char* phase)
	{
		const auto now = Clock::now();
		if (g_settings.GetVerboseLevel()>0)
			std::cout << m_Owner << ": " << phase << " time: "
			          << std::chrono::duration<double>(now-m_Start).count() << " s" << std::endl;
		m_Start = now;
	}
private:
	typedef std::chrono::steady_clock Clock;
	const char* m_Owner;
	Clock::time_point m_Start;
};

#endif
