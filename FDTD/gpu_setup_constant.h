/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_CONSTANT_H
#define GPU_SETUP_CONSTANT_H

#include "CSFunctionParser.h"
#include <cmath>
#include <stdexcept>

// Fold coordinate-independent input with the same parser as CSXCAD. Spatial
// programs still need a device adapter; do not probe them at a dummy position.
inline bool GPU_SetupConstant(const std::string& term)
{
	if (term.empty()) return true;
	CSFunctionParser parser;
	if (parser.Parse(term,"")>=0) return false;
	const double value=parser.Eval(NULL);
	if (parser.EvalError() || !std::isfinite(value))
		throw std::invalid_argument("invalid constant setup weight: "+term);
	return true;
}

#endif
