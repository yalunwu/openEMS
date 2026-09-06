/*
 *  Copyright (C) 2026 openEMS Project
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#ifndef SHADERS_EMBEDDED_H
#define SHADERS_EMBEDDED_H

namespace OpenEMS_WebGPU
{
	static const char* kShaderYeeUpdate = R"(
struct GridUniforms {
    dimX: u32,
    dimY: u32,
    dimZ: u32,
    numCells: u32,
};

@group(0) @binding(0) var<uniform> grid: GridUniforms;
@group(0) @binding(1) var<storage, read_write> volt: array<f32>;
@group(0) @binding(2) var<storage, read_write> curr: array<f32>;
@group(0) @binding(3) var<storage, read> vv: array<f32>;
@group(0) @binding(4) var<storage, read> vi: array<f32>;
@group(0) @binding(5) var<storage, read> ii: array<f32>;
@group(0) @binding(6) var<storage, read> iv: array<f32>;

fn get_index(n: u32, x: u32, y: u32, z: u32) -> u32 {
    let stride_n = grid.dimX * grid.dimY * grid.dimZ;
    let stride_x = grid.dimY * grid.dimZ;
    return n * stride_n + x * stride_x + y * grid.dimZ + z;
}

@compute @workgroup_size(8, 8, 4)
fn update_voltages(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= grid.dimX || y >= grid.dimY || z >= grid.dimZ) {
        return;
    }

    let shift_x: u32 = select(0u, 1u, x > 0u);
    let shift_y: u32 = select(0u, 1u, y > 0u);
    let shift_z: u32 = select(0u, 1u, z > 0u);

    // Update Ex (volt component 0)
    let idx_ex = get_index(0u, x, y, z);
    let curl_h_x = (curr[get_index(2u, x, y, z)] - curr[get_index(2u, x, y - shift_y, z)]) -
                   (curr[get_index(1u, x, y, z)] - curr[get_index(1u, x, y, z - shift_z)]);
    volt[idx_ex] = volt[idx_ex] * vv[idx_ex] + vi[idx_ex] * curl_h_x;

    // Update Ey (volt component 1)
    let idx_ey = get_index(1u, x, y, z);
    let curl_h_y = (curr[get_index(0u, x, y, z)] - curr[get_index(0u, x, y, z - shift_z)]) -
                   (curr[get_index(2u, x, y, z)] - curr[get_index(2u, x - shift_x, y, z)]);
    volt[idx_ey] = volt[idx_ey] * vv[idx_ey] + vi[idx_ey] * curl_h_y;

    // Update Ez (volt component 2)
    let idx_ez = get_index(2u, x, y, z);
    let curl_h_z = (curr[get_index(1u, x, y, z)] - curr[get_index(1u, x - shift_x, y, z)]) -
                   (curr[get_index(0u, x, y, z)] - curr[get_index(0u, x, y - shift_y, z)]);
    volt[idx_ez] = volt[idx_ez] * vv[idx_ez] + vi[idx_ez] * curl_h_z;
}

@compute @workgroup_size(8, 8, 4)
fn update_currents(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    // Currents are evaluated on the dual grid (dim - 1)
    if (x >= (grid.dimX - 1u) || y >= (grid.dimY - 1u) || z >= (grid.dimZ - 1u)) {
        return;
    }

    // Update Hx (curr component 0)
    let idx_hx = get_index(0u, x, y, z);
    let curl_e_x = (volt[get_index(2u, x, y, z)] - volt[get_index(2u, x, y + 1u, z)]) -
                   (volt[get_index(1u, x, y, z)] - volt[get_index(1u, x, y, z + 1u)]);
    curr[idx_hx] = curr[idx_hx] * ii[idx_hx] + iv[idx_hx] * curl_e_x;

    // Update Hy (curr component 1)
    let idx_hy = get_index(1u, x, y, z);
    let curl_e_y = (volt[get_index(0u, x, y, z)] - volt[get_index(0u, x, y, z + 1u)]) -
                   (volt[get_index(2u, x, y, z)] - volt[get_index(2u, x + 1u, y, z)]);
    curr[idx_hy] = curr[idx_hy] * ii[idx_hy] + iv[idx_hy] * curl_e_y;

    // Update Hz (curr component 2)
    let idx_hz = get_index(2u, x, y, z);
    let curl_e_z = (volt[get_index(1u, x, y, z)] - volt[get_index(1u, x + 1u, y, z)]) -
                   (volt[get_index(0u, x, y, z)] - volt[get_index(0u, x, y + 1u, z)]);
    curr[idx_hz] = curr[idx_hz] * ii[idx_hz] + iv[idx_hz] * curl_e_z;
}
)";

	static const char* kShaderUPML = R"(
struct UPMLParams {
    startX: u32,
    startY: u32,
    startZ: u32,
    numX: u32,
    numY: u32,
    numZ: u32,
    fullDimX: u32,
    fullDimY: u32,
    fullDimZ: u32,
};

@group(0) @binding(0) var<uniform> upml: UPMLParams;
@group(0) @binding(1) var<storage, read_write> field: array<f32>;
@group(0) @binding(2) var<storage, read_write> flux: array<f32>;
@group(0) @binding(3) var<storage, read> coef1: array<f32>;
@group(0) @binding(4) var<storage, read> coef2: array<f32>;

fn get_field_idx(n: u32, x: u32, y: u32, z: u32) -> u32 {
    let stride_n = upml.fullDimX * upml.fullDimY * upml.fullDimZ;
    let stride_x = upml.fullDimY * upml.fullDimZ;
    return n * stride_n + x * stride_x + y * upml.fullDimZ + z;
}

fn get_pml_idx(n: u32, x: u32, y: u32, z: u32) -> u32 {
    let stride_n = upml.numX * upml.numY * upml.numZ;
    let stride_x = upml.numY * upml.numZ;
    return n * stride_n + x * stride_x + y * upml.numZ + z;
}

@compute @workgroup_size(8, 8, 4)
fn upml_pre_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let lx = global_id.x;
    let ly = global_id.y;
    let lz = global_id.z;

    if (lx >= upml.numX || ly >= upml.numY || lz >= upml.numZ) {
        return;
    }

    let gx = upml.startX + lx;
    let gy = upml.startY + ly;
    let gz = upml.startZ + lz;

    for (var n: u32 = 0u; n < 3u; n = n + 1u) {
        let f_idx = get_field_idx(n, gx, gy, gz);
        let p_idx = get_pml_idx(n, lx, ly, lz);

        let prev_flux = flux[p_idx];
        flux[p_idx] = prev_flux * coef1[p_idx] + field[f_idx] * coef2[p_idx];
    }
}
)";

	static const char* kShaderProbe = R"(
struct ProbeSegment {
    startX: u32,
    startY: u32,
    startZ: u32,
    component: u32,
    length: u32,
    weight: f32,
};

struct ProbeParams {
    numProbes: u32,
    currentTimestep: u32,
    dimX: u32,
    dimY: u32,
    dimZ: u32,
};

@group(0) @binding(0) var<uniform> params: ProbeParams;
@group(0) @binding(1) var<storage, read> segments: array<ProbeSegment>;
@group(0) @binding(2) var<storage, read> field: array<f32>;
@group(0) @binding(3) var<storage, read_write> probeHistory: array<f32>;

fn get_field_idx(n: u32, x: u32, y: u32, z: u32) -> u32 {
    let stride_n = params.dimX * params.dimY * params.dimZ;
    let stride_x = params.dimY * params.dimZ;
    return n * stride_n + x * stride_x + y * params.dimZ + z;
}

@compute @workgroup_size(64, 1, 1)
fn accumulate_probes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let probe_idx = global_id.x;
    if (probe_idx >= params.numProbes) {
        return;
    }

    let seg = segments[probe_idx];
    var integral: f32 = 0.0;

    var x = seg.startX;
    var y = seg.startY;
    var z = seg.startZ;

    for (var i: u32 = 0u; i < seg.length; i = i + 1u) {
        let f_idx = get_field_idx(seg.component, x, y, z);
        integral = integral + field[f_idx] * seg.weight;

        if (seg.component == 0u) { x = x + 1u; }
        else if (seg.component == 1u) { y = y + 1u; }
        else { z = z + 1u; }
    }

    let history_idx = params.currentTimestep * params.numProbes + probe_idx;
    probeHistory[history_idx] = integral;
}
)";

	static const char* kShaderExcitation = R"(
struct ExcitationPoint {
    index: u32,
    value: f32,
};

struct ExcitationParams {
    count: u32,
};

@group(0) @binding(0) var<uniform> excParams: ExcitationParams;
@group(0) @binding(1) var<storage, read_write> field: array<f32>;
@group(0) @binding(2) var<storage, read> points: array<ExcitationPoint>;

@compute @workgroup_size(64, 1, 1)
fn inject_excitation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= excParams.count) {
        return;
    }
    let p = points[i];
    field[p.index] = field[p.index] + p.value;
}
)";
}

#endif // SHADERS_EMBEDDED_H
