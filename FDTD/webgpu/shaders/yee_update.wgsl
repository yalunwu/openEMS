// WGSL Compute Shaders for openEMS FDTD Field Updates (Yee Stencil)
// Matches ArrayLib contiguous N-I-J-K striding for coalesced GPU execution.

struct GridUniforms {
    dimX: u32,
    dimY: u32,
    dimZ: u32,
    numCells: u32,
};

@group(0) @binding(0) var<uniform> grid: GridUniforms;
@group(0) @binding(1) var<storage, read_write> volt: array<f32>;
@group(0) @binding(2) var<storage, read> curr: array<f32>;
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
