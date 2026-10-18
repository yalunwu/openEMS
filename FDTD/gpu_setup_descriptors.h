/* Copyright (C) 2026 openEMS Project; SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef GPU_SETUP_DESCRIPTORS_H
#define GPU_SETUP_DESCRIPTORS_H

#include <cstdint>
#include <type_traits>

// Uploaded records contain values and buffer offsets, never CSXCAD pointers.
enum class GPU_SetupShape : uint32_t
{
	Empty, Box, Sphere, Cylinder, Polygon, MultiBox, RotPoly,
	SphericalShell, CylindricalShell, Wire, Point, Curve, Polyhedron, UserDefined
};

struct GPU_SetupPrimitive
{
	uint32_t version, id, property_id;
	GPU_SetupShape shape;
	uint32_t coordinate_system, offset, count, normal, axis;
	uint32_t input_coordinate_system, candidate_bounds, has_transform, geometry_flags;
	uint32_t index_offset, index_leaves;
	uint32_t bounds_offset, bounds_count;
	int32_t dimension;
	double inverse[12], forward[12], start[3], stop[3], bounds[6];
	double radius, shell_width, angles[2];
};

// A complete balanced tree uses heap children 2*i+1 and 2*i+2. Empty padded
// leaves have inverted bounds. Traversal needs no per-thread stack.
struct GPU_SetupBVHNode
{
	double bounds[6];
	uint32_t triangle;
};

struct GPU_SetupProperty
{
	uint32_t id, type, metal, enabled, excitation_type, sheet, active[3];
	uint32_t coordinate_system, weights[3][4];
	uint32_t discrete, file_weight;
	uint32_t density_weight;
	uint32_t dispersion_offset, dispersion_count;
	uint32_t isotropic;
	double values[3][4], delay, conductivity, thickness;
	double density;
	double weight_origin[3];
};

struct GPU_SetupDiscreteMaterial
{
	uint32_t dims[3], mesh[3], channels[5], indices, database_size, database_background, coordinate_system;
	double scale, inverse[12];
};

struct GPU_SetupDispersionPole
{
	double values[3][2][3];
	uint32_t weights[3][2][3];
};

struct GPU_SetupFileWeight
{
	uint32_t dims[2], axes[2], fields[2], propagation_axis;
};

struct GPU_SetupSourcePoint
{
	uint32_t index;
	float amplitude;
	uint32_t delay, offset, length, period;
};

static_assert(std::is_trivially_copyable<GPU_SetupPrimitive>::value,"geometry descriptors must be plain device records");
static_assert(std::is_trivially_copyable<GPU_SetupProperty>::value,"property descriptors must be plain device records");

#endif
