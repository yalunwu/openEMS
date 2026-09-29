#ifndef SHADERS_GLSL_H
#define SHADERS_GLSL_H

namespace VulkanShaders {

static const char* kShaderVoltageUpdate = R"(#version 450
layout(local_size_x = 32, local_size_y = 4, local_size_z = 2) in;

layout(push_constant) uniform PushConstants {
    uint dimX;
    uint dimY;
    uint dimZ;
    uint numCells;
} pc;

layout(std430, binding = 0) readonly buffer BufVv { float vv[]; };
layout(std430, binding = 1) readonly buffer BufVi { float vi[]; };
layout(std430, binding = 2) readonly buffer BufIi { float ii[]; };
layout(std430, binding = 3) readonly buffer BufIv { float iv[]; };
layout(std430, binding = 4) buffer BufVolt { float volt[]; };
layout(std430, binding = 5) readonly buffer BufCurr { float curr[]; };

void main() {
    uint z = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint x = gl_GlobalInvocationID.z;

    if (x >= pc.dimX || y >= pc.dimY || z >= pc.dimZ) {
        return;
    }

    uint sliceSize = pc.dimY * pc.dimZ;
    uint idx = x * sliceSize + y * pc.dimZ + z;
    uint nCells = pc.numCells;

    uint idx_ym1 = (y > 0u) ? (idx - pc.dimZ) : idx;
    uint idx_zm1 = (z > 0u) ? (idx - 1u) : idx;
    uint idx_xm1 = (x > 0u) ? (idx - sliceSize) : idx;

    // Vx update: vi_x * [ (Iz(x,y,z) - Iz(x,y-1,z)) - (Iy(x,y,z) - Iy(x,y,z-1)) ]
    float curl_x = (curr[2u * nCells + idx] - curr[2u * nCells + idx_ym1])
                 - (curr[1u * nCells + idx] - curr[1u * nCells + idx_zm1]);
    volt[idx] = vv[idx] * volt[idx] + vi[idx] * curl_x;

    // Vy update: vi_y * [ (Ix(x,y,z) - Ix(x,y,z-1)) - (Iz(x,y,z) - Iz(x-1,y,z)) ]
    float curl_y = (curr[idx] - curr[idx_zm1])
                 - (curr[2u * nCells + idx] - curr[2u * nCells + idx_xm1]);
    volt[nCells + idx] = vv[nCells + idx] * volt[nCells + idx] + vi[nCells + idx] * curl_y;

    // Vz update: vi_z * [ (Iy(x,y,z) - Iy(x-1,y,z)) - (Ix(x,y,z) - Ix(x,y-1,z)) ]
    float curl_z = (curr[1u * nCells + idx] - curr[1u * nCells + idx_xm1])
                 - (curr[idx] - curr[idx_ym1]);
    volt[2u * nCells + idx] = vv[2u * nCells + idx] * volt[2u * nCells + idx] + vi[2u * nCells + idx] * curl_z;
}
)";

static const char* kShaderCurrentUpdate = R"(#version 450
layout(local_size_x = 32, local_size_y = 4, local_size_z = 2) in;

layout(push_constant) uniform PushConstants {
    uint dimX;
    uint dimY;
    uint dimZ;
    uint numCells;
} pc;

layout(std430, binding = 0) readonly buffer BufVv { float vv[]; };
layout(std430, binding = 1) readonly buffer BufVi { float vi[]; };
layout(std430, binding = 2) readonly buffer BufIi { float ii[]; };
layout(std430, binding = 3) readonly buffer BufIv { float iv[]; };
layout(std430, binding = 4) readonly buffer BufVolt { float volt[]; };
layout(std430, binding = 5) buffer BufCurr { float curr[]; };

void main() {
    uint z = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint x = gl_GlobalInvocationID.z;

    if (x >= (pc.dimX - 1u) || y >= (pc.dimY - 1u) || z >= (pc.dimZ - 1u)) {
        return;
    }

    uint sliceSize = pc.dimY * pc.dimZ;
    uint idx = x * sliceSize + y * pc.dimZ + z;
    uint nCells = pc.numCells;

    uint idx_yp1 = idx + pc.dimZ;
    uint idx_zp1 = idx + 1u;
    uint idx_xp1 = idx + sliceSize;

    // Ix update: iv_x * [ (Vz(x,y,z) - Vz(x,y+1,z)) - (Vy(x,y,z) - Vy(x,y,z+1)) ]
    float curl_x = (volt[2u * nCells + idx] - volt[2u * nCells + idx_yp1])
                 - (volt[1u * nCells + idx] - volt[1u * nCells + idx_zp1]);
    curr[idx] = ii[idx] * curr[idx] + iv[idx] * curl_x;

    // Iy update: iv_y * [ (Vx(x,y,z) - Vx(x,y,z+1)) - (Vz(x,y,z) - Vz(x+1,y,z)) ]
    float curl_y = (volt[idx] - volt[idx_zp1])
                 - (volt[2u * nCells + idx] - volt[2u * nCells + idx_xp1]);
    curr[nCells + idx] = ii[nCells + idx] * curr[nCells + idx] + iv[nCells + idx] * curl_y;

    // Iz update: iv_z * [ (Vy(x,y,z) - Vy(x+1,y,z)) - (Vx(x,y,z) - Vx(x,y+1,z)) ]
    float curl_z = (volt[1u * nCells + idx] - volt[1u * nCells + idx_xp1])
                 - (volt[idx] - volt[idx_yp1]);
    curr[2u * nCells + idx] = ii[2u * nCells + idx] * curr[2u * nCells + idx] + iv[2u * nCells + idx] * curl_z;
}
)";

static const char* kShaderExcitation = R"(#version 450
layout(local_size_x = 64) in;

struct ExcPoint {
    uint index;
    float value;
};

layout(push_constant) uniform ExcParams {
    uint count;
} ep;

layout(std430, binding = 0) readonly buffer PointsBuffer {
    ExcPoint points[];
};

layout(std430, binding = 1) buffer FieldBuffer {
    float fieldData[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= ep.count) {
        return;
    }
    fieldData[points[idx].index] += points[idx].value;
}
)";

static const char* kShaderProbeGather = R"(#version 450
layout(local_size_x = 64) in;

struct ProbePoint {
    uint linear_index;
    uint field_type; // 0 = volt, 1 = curr
};

layout(push_constant) uniform ProbeParams {
    uint count;
} pp;

layout(std430, binding = 0) readonly buffer PointsBuffer {
    ProbePoint points[];
};

layout(std430, binding = 1) readonly buffer VoltBuffer {
    float voltData[];
};

layout(std430, binding = 2) readonly buffer CurrBuffer {
    float currData[];
};

layout(std430, binding = 3) writeonly buffer ValuesBuffer {
    float outValues[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pp.count) {
        return;
    }
    if (points[idx].field_type == 0u) {
        outValues[idx] = voltData[points[idx].linear_index];
    } else {
        outValues[idx] = currData[points[idx].linear_index];
    }
}
)";

static const char* kShaderUpmlPre = R"(#version 450
layout(local_size_x = 256) in;

layout(push_constant) uniform UpmlParams {
    uint count;
} pc;

layout(std430, binding = 0) readonly buffer IndicesBuffer {
    uint fieldIndices[];
};

layout(std430, binding = 1) readonly buffer CoeffsBuffer {
    vec4 coeffs[];
};

layout(std430, binding = 2) buffer FluxBuffer {
    float fluxData[];
};

layout(std430, binding = 3) buffer FieldBuffer {
    float fieldData[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.count) {
        return;
    }
    uint f_idx = fieldIndices[idx];
    vec4 c = coeffs[idx];

    float f_val = fieldData[f_idx];
    float fl_val = fluxData[idx];

    float f_help = c.x * f_val - c.y * fl_val;
    fieldData[f_idx] = fl_val;
    fluxData[idx] = f_help;
}
)";

static const char* kShaderUpmlPost = R"(#version 450
layout(local_size_x = 256) in;

layout(push_constant) uniform UpmlParams {
    uint count;
} pc;

layout(std430, binding = 0) readonly buffer IndicesBuffer {
    uint fieldIndices[];
};

layout(std430, binding = 1) readonly buffer CoeffsBuffer {
    vec4 coeffs[];
};

layout(std430, binding = 2) buffer FluxBuffer {
    float fluxData[];
};

layout(std430, binding = 3) buffer FieldBuffer {
    float fieldData[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.count) {
        return;
    }
    uint f_idx = fieldIndices[idx];
    vec4 c = coeffs[idx];

    float fl_val = fluxData[idx];
    float f_val = fieldData[f_idx];

    fluxData[idx] = f_val;
    fieldData[f_idx] = fl_val + c.z * f_val;
}
)";

static const char* kShaderMurPre = R"(#version 450
layout(local_size_x = 256) in;

struct MurPoint {
    uint pos_idx;
    uint shift_idx;
    float coeff;
    uint start_TS;
};

layout(std430, binding = 0) readonly buffer MurParams {
    MurPoint points[];
};

layout(std430, binding = 1) readonly buffer VoltField {
    float voltData[];
};

layout(std430, binding = 2) writeonly buffer MurStore {
    float storeData[];
};

layout(push_constant) uniform MurPushConstants {
    uint offset;
    uint count;
    uint current_TS;
    uint pad;
} pc;

void main() {
    uint idx = pc.offset + gl_GlobalInvocationID.x;
    if (idx >= pc.offset + pc.count) {
        return;
    }
    MurPoint p = points[idx];
    if (pc.current_TS < p.start_TS) {
        return;
    }
    storeData[idx] = voltData[p.shift_idx] - p.coeff * voltData[p.pos_idx];
}
)";

static const char* kShaderMurPost = R"(#version 450
layout(local_size_x = 256) in;

struct MurPoint {
    uint pos_idx;
    uint shift_idx;
    float coeff;
    uint start_TS;
};

layout(std430, binding = 0) readonly buffer MurParams {
    MurPoint points[];
};

layout(std430, binding = 1) readonly buffer VoltField {
    float voltData[];
};

layout(std430, binding = 2) buffer MurStore {
    float storeData[];
};

layout(push_constant) uniform MurPushConstants {
    uint offset;
    uint count;
    uint current_TS;
    uint pad;
} pc;

void main() {
    uint idx = pc.offset + gl_GlobalInvocationID.x;
    if (idx >= pc.offset + pc.count) {
        return;
    }
    MurPoint p = points[idx];
    if (pc.current_TS < p.start_TS) {
        return;
    }
    storeData[idx] += p.coeff * voltData[p.shift_idx];
}
)";

static const char* kShaderMurApply = R"(#version 450
layout(local_size_x = 256) in;

struct MurPoint {
    uint pos_idx;
    uint shift_idx;
    float coeff;
    uint start_TS;
};

layout(std430, binding = 0) readonly buffer MurParams {
    MurPoint points[];
};

layout(std430, binding = 1) buffer VoltField {
    float voltData[];
};

layout(std430, binding = 2) readonly buffer MurStore {
    float storeData[];
};

layout(push_constant) uniform MurPushConstants {
    uint offset;
    uint count;
    uint current_TS;
    uint pad;
} pc;

void main() {
    uint idx = pc.offset + gl_GlobalInvocationID.x;
    if (idx >= pc.offset + pc.count) {
        return;
    }
    MurPoint p = points[idx];
    if (pc.current_TS < p.start_TS) {
        return;
    }
    voltData[p.pos_idx] = storeData[idx];
}
)";

static const char* kShaderTFSF = R"(#version 450
layout(local_size_x = 256) in;

struct TfsfPoint {
    uint pos_idx;
    uint delay;
    float w0;
    float w1;
};

layout(std430, binding = 0) buffer FieldBuffer {
    float fieldData[];
};

layout(std430, binding = 1) readonly buffer TfsfParams {
    TfsfPoint points[];
};

layout(std430, binding = 2) readonly buffer SignalBuffer {
    float signalData[];
};

layout(push_constant) uniform TfsfPushConstants {
    uint offset;
    uint count;
    uint current_TS;
    uint sigLength;
    int p;
} pc;

void main() {
    uint local_idx = gl_GlobalInvocationID.x;
    if (local_idx >= pc.count) {
        return;
    }
    uint idx = pc.offset + local_idx;
    TfsfPoint pt = points[idx];

    // Match Engine_Ext_TFSF's delay lookup exactly. Before a delayed
    // sample starts (and after a finite signal ends), the CPU selects
    // sample zero rather than injecting a literal zero.
    float s0 = signalData[0];
    if (pc.current_TS >= pt.delay) {
        uint diff0 = pc.current_TS - pt.delay;
        if (pc.p > 0) {
            uint rem0 = diff0 % uint(pc.p);
            s0 = signalData[rem0];
        } else if (diff0 < pc.sigLength) {
            s0 = signalData[diff0];
        }
    }

    float s1 = signalData[0];
    uint delay1 = pt.delay + 1u;
    if (pc.current_TS >= delay1) {
        uint diff1 = pc.current_TS - delay1;
        if (pc.p > 0) {
            uint rem1 = diff1 % uint(pc.p);
            s1 = signalData[rem1];
        } else if (diff1 < pc.sigLength) {
            s1 = signalData[diff1];
        }
    }

    fieldData[pt.pos_idx] += pt.w0 * s0 + pt.w1 * s1;
}
)";

static const char* kShaderRLC = R"(#version 450
layout(local_size_x = 64) in;

layout(push_constant) uniform RlcPushConstants {
    uint count;
    uint mode;
} pc;

struct GpuRlcParam {
    uint field_index;
    float ilv_i2v;
    float vvd;
    float vv2;
    float vj1;
    float vj2;
    float ib0;
    float b1_ib0;
    float b2_ib0;
    float pad0;
    float pad1;
    float pad2;
};

struct GpuRlcState {
    float v_Il;
    float vd0;
    float vd1;
    float vd2;
    float j0;
    float j1;
    float j2;
    float pad;
};

layout(std430, binding = 0) buffer FieldVoltBuf {
    float fieldVolt[];
};

layout(std430, binding = 1) readonly buffer RlcParamsBuf {
    GpuRlcParam rlcParams[];
};

layout(std430, binding = 2) buffer RlcStateBuf {
    GpuRlcState rlcState[];
};

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= pc.count) {
        return;
    }

    if (pc.mode == 0u) {
        rlcState[id].vd2 = rlcState[id].vd1;
        rlcState[id].vd1 = rlcState[id].vd0;
        rlcState[id].v_Il += rlcParams[id].ilv_i2v * rlcState[id].vd1;
    } else if (pc.mode == 1u) {
        rlcState[id].j2 = rlcState[id].j1;
        rlcState[id].j1 = rlcState[id].j0;

        uint fIdx = rlcParams[id].field_index;
        float v0 = fieldVolt[fIdx];

        float v_new = rlcParams[id].vvd * (
            v0 - rlcState[id].v_Il
            + rlcParams[id].vv2 * rlcState[id].vd2
            + rlcParams[id].vj1 * rlcState[id].j1
            + rlcParams[id].vj2 * rlcState[id].j2
        );

        float j_new = rlcParams[id].ib0 * (v_new - rlcState[id].vd2)
            - rlcParams[id].b1_ib0 * rlcState[id].j1
            - rlcParams[id].b2_ib0 * rlcState[id].j2;

        rlcState[id].vd0 = v_new;
        rlcState[id].j0 = j_new;

    } else if (rlcParams[id].pad0 != 0.0) {
        // All elements calculate before any writes. Only the final element
        // at a duplicated node writes, matching the CPU's last-write-wins
        // behavior without a shader race.
        fieldVolt[rlcParams[id].field_index] = rlcState[id].vd0;
    }
}
)";

static const char* kShaderAbcVolt = R"(#version 450
layout(local_size_x = 256) in;

struct AbcVoltPoint {
    uint pos_idx;
    uint shift_idx;
    float k1;
    uint pad;
};

layout(std430, binding = 0) readonly buffer ParamsBuffer {
    AbcVoltPoint points[];
};

layout(std430, binding = 1) buffer VoltBuffer {
    float voltData[];
};

layout(std430, binding = 2) buffer StoreBuffer {
    float storeData[];
};

layout(push_constant) uniform AbcPushConstants {
    uint offset;
    uint count;
    uint mode; // 0 = pre, 1 = post, 2 = apply
} pc;

void main() {
    uint local_idx = gl_GlobalInvocationID.x;
    if (local_idx >= pc.count) {
        return;
    }
    uint idx = pc.offset + local_idx;
    AbcVoltPoint pt = points[idx];
    if (pc.mode == 0u) {
        storeData[idx] = voltData[pt.shift_idx] - pt.k1 * voltData[pt.pos_idx];
    } else if (pc.mode == 1u) {
        storeData[idx] += pt.k1 * voltData[pt.shift_idx];
    } else if (pc.mode == 2u) {
        voltData[pt.pos_idx] = storeData[idx];
    }
}
)";

static const char* kShaderAbcCurr = R"(#version 450
layout(local_size_x = 256) in;

struct AbcCurrPoint {
    uint pos_idx;
    uint shift_idx;
    float k1;
    float k2;
};

layout(std430, binding = 0) readonly buffer ParamsBuffer {
    AbcCurrPoint points[];
};

layout(std430, binding = 1) buffer CurrBuffer {
    float currData[];
};

layout(std430, binding = 2) buffer StoreBuffer {
    float storeData[];
};

layout(push_constant) uniform AbcPushConstants {
    uint offset;
    uint count;
    uint mode; // 0 = pre, 1 = post, 2 = apply
} pc;

void main() {
    uint local_idx = gl_GlobalInvocationID.x;
    if (local_idx >= pc.count) {
        return;
    }
    uint idx = pc.offset + local_idx;
    AbcCurrPoint pt = points[idx];
    if (pc.mode == 0u) {
        storeData[idx] = currData[pt.shift_idx] - pt.k1 * currData[pt.pos_idx];
    } else if (pc.mode == 1u) {
        storeData[idx] += pt.k1 * currData[pt.shift_idx];
    } else if (pc.mode == 2u) {
        currData[pt.pos_idx] = (storeData[idx] * pt.k2 + currData[pt.pos_idx]) / (pt.k2 + 1.0);
    }
}
)";

static const char* kShaderDispersive = R"(#version 450
layout(local_size_x = 256) in;

struct DispersivePoint {
    uint pos_idx;
    float int_coeff;
    float ext_coeff;
    float lor_coeff;
};

struct DispersiveState {
    float ade_val;
    float lor_val;
};

layout(std430, binding = 0) readonly buffer ParamsBuffer {
    DispersivePoint points[];
};

layout(std430, binding = 1) buffer FieldBuffer {
    float fieldData[];
};

layout(std430, binding = 2) buffer StateBuffer {
    DispersiveState states[];
};

layout(push_constant) uniform DispersivePushConstants {
    uint offset;
    uint count;
    uint mode; // 0 = pre-update, 1 = apply
} pc;

void main() {
    uint local_idx = gl_GlobalInvocationID.x;
    if (local_idx >= pc.count) {
        return;
    }
    uint idx = pc.offset + local_idx;
    DispersivePoint pt = points[idx];
    if (pc.mode == 0u) {
        float lor = states[idx].lor_val + pt.lor_coeff * states[idx].ade_val;
        float f = fieldData[pt.pos_idx];
        float ade = states[idx].ade_val * pt.int_coeff + pt.ext_coeff * (f - lor);
        states[idx].lor_val = lor;
        states[idx].ade_val = ade;
    } else if (pc.mode == 1u) {
        fieldData[pt.pos_idx] -= states[idx].ade_val;
    }
}
)";

static const char* kShaderCylinder = R"(#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer VoltBuffer {
    float voltData[];
};

layout(std430, binding = 1) buffer CurrBuffer {
    float currData[];
};

layout(std430, binding = 2) readonly buffer R0CoeffBuffer {
    vec2 r0Coeffs[];
};

layout(push_constant) uniform CylPushConstants {
    uint dimX;        // Nr
    uint dimY;        // Nalpha
    uint dimZ;        // Nz
    uint numCells;
    uint last_A_Line; // dimY - 2
    uint hasR0;       // 1 if CC_R0_included, else 0
    uint mode;        // 0: post-volt R0, 1: post-volt alpha wrap, 2: post-curr alpha wrap
} pc;

void main() {
    if (pc.mode == 0u) {
        // Post-Volt R0 pass: 1D thread index over z in [0, dimZ - 1]
        uint z = gl_WorkGroupID.x * 256u + gl_LocalInvocationIndex;
        if (z >= pc.dimZ || pc.hasR0 == 0u) return;

        // Vz(r=0, a=0, z): dir=2, r=0, a=0 => 2 * numCells + z
        uint ez_0 = 2u * pc.numCells + z;
        float volt = voltData[ez_0] * r0Coeffs[z].x;
        float vi = r0Coeffs[z].y;

        // Accumulate I_alpha(r=0, a, z) across a in [0, dimY - 2]
        // dir=1, r=0 => numCells + a * dimZ + z
        for (uint a = 0u; a < pc.dimY - 1u; ++a) {
            uint i_idx = pc.numCells + a * pc.dimZ + z;
            volt += vi * currData[i_idx];
        }

        // Set E_alpha(r=0, a, z) = 0 and E_z(r=0, a, z) = volt for all alpha in [0, dimY - 1]
        for (uint a = 0u; a < pc.dimY; ++a) {
            uint ea_idx = pc.numCells + a * pc.dimZ + z;
            uint ez_idx = 2u * pc.numCells + a * pc.dimZ + z;
            voltData[ea_idx] = 0.0;
            voltData[ez_idx] = volt;
        }
    } else if (pc.mode == 1u) {
        // Post-Volt Alpha Wrap: (r, z) in [0, dimX - 1] x [0, dimZ - 1]
        uint r = gl_GlobalInvocationID.x;
        uint z = gl_GlobalInvocationID.y;
        if (r >= pc.dimX || z >= pc.dimZ) return;

        uint slice = r * (pc.dimY * pc.dimZ);
        uint src_offset = slice + pc.last_A_Line * pc.dimZ + z;
        uint dst_offset = slice + z; // a = 0

        // Copy Er (dir=0)
        voltData[dst_offset] = voltData[src_offset];
        // Copy Ez (dir=2)
        voltData[2u * pc.numCells + dst_offset] = voltData[2u * pc.numCells + src_offset];
    } else if (pc.mode == 2u) {
        // Post-Curr Alpha Wrap: (r, z) in [0, dimX - 2] x [0, dimZ - 2]
        uint r = gl_GlobalInvocationID.x;
        uint z = gl_GlobalInvocationID.y;
        if (r >= pc.dimX - 1u || z >= pc.dimZ - 1u) return;

        uint slice = r * (pc.dimY * pc.dimZ);
        uint src_offset = slice + z; // a = 0
        uint dst_offset = slice + pc.last_A_Line * pc.dimZ + z;

        // Copy Ir (dir=0)
        currData[dst_offset] = currData[src_offset];
        // Copy Iz (dir=2)
        currData[2u * pc.numCells + dst_offset] = currData[2u * pc.numCells + src_offset];
    }
}
)";

} // namespace VulkanShaders

#endif // SHADERS_GLSL_H
