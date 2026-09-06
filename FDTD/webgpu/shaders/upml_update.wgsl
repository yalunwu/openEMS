// WGSL Compute Shader for UPML (Uniaxial Perfectly Matched Layer) Boundaries
// Handles auxiliary flux density equations on boundary cells.

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

        // Pre-update PML auxiliary flux calculation
        let prev_flux = flux[p_idx];
        flux[p_idx] = prev_flux * coef1[p_idx] + field[f_idx] * coef2[p_idx];
    }
}
