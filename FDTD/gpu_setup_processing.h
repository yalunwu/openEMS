/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_PROCESSING_H
#define GPU_SETUP_PROCESSING_H
#include "Common/engine_setup_tables.h"
#include <memory>
class Operator_Base;
class Engine;
bool GPU_SetupFusionOrder(Engine* engine,size_t regions);
class GPU_Backend;
void GPU_FinishInitialization(GPU_Backend* backend);
unsigned int GPU_SetupCoordinateFlags(const Operator_Base* op);
std::shared_ptr<Engine_SamplingMesh> GPU_SetupSamplingMesh(const Operator_Base* op,const unsigned int start[3],const unsigned int stop[3],
    bool dual,bool optimal,const unsigned int step[3],const double resolution[3]);
// False means this operator uses CPU construction. Successful device dispatch
// reports scalar indices/inside flags needed for processing registration.
bool GPU_SetupProcessingSnap(const Operator_Base* op,const double start[3],const double stop[3],unsigned int first[3],unsigned int last[3],
    bool dual,bool full,int method,bool point,bool* first_inside,bool* last_inside,int& dimension);
bool GPU_SetupProcessingIndices(const Operator_Base* op,bool mode_match,bool dual,const double* start,const double* stop,
    unsigned int first[3],unsigned int last[3],int& normal,int& dimension);
#endif
