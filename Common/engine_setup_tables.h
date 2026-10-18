/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef ENGINE_SETUP_TABLES_H
#define ENGINE_SETUP_TABLES_H
#include <cstddef>
#include <string>

class Engine_SamplingMesh
{
public:
	virtual ~Engine_SamplingMesh() {}
	virtual unsigned int Lines(int axis) const=0;
	// Requested mesh metadata for file encoding or CPU processing. Device
	// sampling consumers retain the original indices rather than reuploading.
	virtual void Read(unsigned int* positions[3],double* coordinates[3]) const=0;
};

// Numerical processing inputs have been prepared on the device. A CPU
// processing consumer requests only the values it actually uses.
class Engine_SetupTable
{
public:
	virtual ~Engine_SetupTable() {}
	virtual size_t Count() const=0;
	virtual void Read(double* values,size_t count) const=0;
};

class Engine_CellMaterialTable : public Engine_SetupTable
{
public:
	virtual bool AnyAnisotropic() const=0;
};

struct Engine_ModeTableInput
{
	unsigned int start[3],stop[3],normal;
	bool magnetic;
	double origin[3];
	std::string functions[3],file;
};
#endif
