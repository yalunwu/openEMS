/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_MULTIGRID_PLAN_H
#define GPU_SETUP_MULTIGRID_PLAN_H
class GPU_Backend;
bool GPU_SetupMultiGridSplit(GPU_Backend* backend,const double* lines,unsigned int count,double radius,unsigned int& split);
#endif
