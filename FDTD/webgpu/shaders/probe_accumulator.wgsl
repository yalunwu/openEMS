// WGSL Compute Shader for Asynchronous On-Device Probe & Port Line-Integral Accumulation

struct ProbeSegment {
    startX: u32,
    startY: u32,
    startZ: u32,
    component: u32, // 0 = X, 1 = Y, 2 = Z
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

    // Accumulate field along the integration path
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

    // Record sample in circular/linear history buffer on device
    let history_idx = params.currentTimestep * params.numProbes + probe_idx;
    probeHistory[history_idx] = integral;
}
