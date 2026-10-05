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
	uint startX;
	uint countX;
} pc;

#ifdef COEFFICIENT_PALETTE
layout(std430, binding = 0) readonly buffer Palette { float coefficients[]; };
layout(std430, binding = 1) readonly buffer Indices { uint coefficientIndex[]; };
#define VV(n) coefficients[coefficientBase + 4u * (n)]
#define VI(n) coefficients[coefficientBase + 4u * (n) + 1u]
#define II(n) coefficients[coefficientBase + 4u * (n) + 2u]
#define IV(n) coefficients[coefficientBase + 4u * (n) + 3u]
#else
layout(std430, binding = 0) readonly buffer BufVv { float vv[]; };
layout(std430, binding = 1) readonly buffer BufVi { float vi[]; };
layout(std430, binding = 2) readonly buffer BufIi { float ii[]; };
layout(std430, binding = 3) readonly buffer BufIv { float iv[]; };
#define VV(n) vv[(n) * nCells + idx]
#define VI(n) vi[(n) * nCells + idx]
#define II(n) ii[(n) * nCells + idx]
#define IV(n) iv[(n) * nCells + idx]
#endif
layout(std430, binding = 4) buffer BufVolt { float volt[]; };
layout(std430, binding = 5) readonly buffer BufCurr { float curr[]; };

void main() {
    uint z = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint localX = gl_GlobalInvocationID.z;
    if (localX >= pc.countX) {
        return;
    }
    uint x = pc.startX + localX;

    if (x >= pc.dimX || y >= pc.dimY || z >= pc.dimZ) {
        return;
    }

    uint sliceSize = pc.dimY * pc.dimZ;
    uint idx = x * sliceSize + y * pc.dimZ + z;
    uint nCells = pc.numCells;
#ifdef COEFFICIENT_PALETTE
    uint coefficientBase = 12u * coefficientIndex[idx];
#endif

    uint idx_ym1 = (y > 0u) ? (idx - pc.dimZ) : idx;
    uint idx_zm1 = (z > 0u) ? (idx - 1u) : idx;
    uint idx_xm1 = (x > 0u) ? (idx - sliceSize) : idx;

    // Vx update: vi_x * [ (Iz(x,y,z) - Iz(x,y-1,z)) - (Iy(x,y,z) - Iy(x,y,z-1)) ]
    float curl_x = (curr[2u * nCells + idx] - curr[2u * nCells + idx_ym1])
                 - (curr[1u * nCells + idx] - curr[1u * nCells + idx_zm1]);
    volt[idx] = VV(0u) * volt[idx] + VI(0u) * curl_x;

    // Vy update: vi_y * [ (Ix(x,y,z) - Ix(x,y,z-1)) - (Iz(x,y,z) - Iz(x-1,y,z)) ]
    float curl_y = (curr[idx] - curr[idx_zm1])
                 - (curr[2u * nCells + idx] - curr[2u * nCells + idx_xm1]);
    volt[nCells + idx] = VV(1u) * volt[nCells + idx] + VI(1u) * curl_y;

    // Vz update: vi_z * [ (Iy(x,y,z) - Iy(x-1,y,z)) - (Ix(x,y,z) - Ix(x,y-1,z)) ]
    float curl_z = (curr[1u * nCells + idx] - curr[1u * nCells + idx_xm1])
                 - (curr[idx] - curr[idx_ym1]);
    volt[2u * nCells + idx] = VV(2u) * volt[2u * nCells + idx] + VI(2u) * curl_z;
}
)";

static const char* kShaderCurrentUpdate = R"(#version 450
layout(local_size_x = 32, local_size_y = 4, local_size_z = 2) in;

layout(push_constant) uniform PushConstants {
    uint dimX;
    uint dimY;
    uint dimZ;
    uint numCells;
	uint startX;
	uint countX;
} pc;

#ifdef COEFFICIENT_PALETTE
layout(std430, binding = 0) readonly buffer Palette { float coefficients[]; };
layout(std430, binding = 1) readonly buffer Indices { uint coefficientIndex[]; };
#define VV(n) coefficients[coefficientBase + 4u * (n)]
#define VI(n) coefficients[coefficientBase + 4u * (n) + 1u]
#define II(n) coefficients[coefficientBase + 4u * (n) + 2u]
#define IV(n) coefficients[coefficientBase + 4u * (n) + 3u]
#else
layout(std430, binding = 0) readonly buffer BufVv { float vv[]; };
layout(std430, binding = 1) readonly buffer BufVi { float vi[]; };
layout(std430, binding = 2) readonly buffer BufIi { float ii[]; };
layout(std430, binding = 3) readonly buffer BufIv { float iv[]; };
#define VV(n) vv[(n) * nCells + idx]
#define VI(n) vi[(n) * nCells + idx]
#define II(n) ii[(n) * nCells + idx]
#define IV(n) iv[(n) * nCells + idx]
#endif
layout(std430, binding = 4) readonly buffer BufVolt { float volt[]; };
layout(std430, binding = 5) buffer BufCurr { float curr[]; };

void main() {
    uint z = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint localX = gl_GlobalInvocationID.z;
    if (localX >= pc.countX) {
        return;
    }
    uint x = pc.startX + localX;

    if (x >= (pc.dimX - 1u) || y >= (pc.dimY - 1u) || z >= (pc.dimZ - 1u)) {
        return;
    }

    uint sliceSize = pc.dimY * pc.dimZ;
    uint idx = x * sliceSize + y * pc.dimZ + z;
    uint nCells = pc.numCells;
#ifdef COEFFICIENT_PALETTE
    uint coefficientBase = 12u * coefficientIndex[idx];
#endif

    uint idx_yp1 = idx + pc.dimZ;
    uint idx_zp1 = idx + 1u;
    uint idx_xp1 = idx + sliceSize;

    // Ix update: iv_x * [ (Vz(x,y,z) - Vz(x,y+1,z)) - (Vy(x,y,z) - Vy(x,y,z+1)) ]
    float curl_x = (volt[2u * nCells + idx] - volt[2u * nCells + idx_yp1])
                 - (volt[1u * nCells + idx] - volt[1u * nCells + idx_zp1]);
    curr[idx] = II(0u) * curr[idx] + IV(0u) * curl_x;

    // Iy update: iv_y * [ (Vx(x,y,z) - Vx(x,y,z+1)) - (Vz(x,y,z) - Vz(x+1,y,z)) ]
    float curl_y = (volt[idx] - volt[idx_zp1])
                 - (volt[2u * nCells + idx] - volt[2u * nCells + idx_xp1]);
    curr[nCells + idx] = II(1u) * curr[nCells + idx] + IV(1u) * curl_y;

    // Iz update: iv_z * [ (Vy(x,y,z) - Vy(x+1,y,z)) - (Vx(x,y,z) - Vx(x,y+1,z)) ]
    float curl_z = (volt[1u * nCells + idx] - volt[1u * nCells + idx_xp1])
                 - (volt[idx] - volt[idx_yp1]);
    curr[2u * nCells + idx] = II(2u) * curr[2u * nCells + idx] + IV(2u) * curl_z;
}
)";

static const char* kShaderExcitation = R"(#version 450
layout(local_size_x = 64) in;

struct ExcPoint {
    uint index;
    float amplitude;
    uint delay;
    uint signalOffset;
    uint signalLength;
    uint period;
};

layout(push_constant) uniform ExcParams {
    uint count;
    uint timestep;
} ep;

layout(std430, binding = 0) readonly buffer PointsBuffer {
    ExcPoint points[];
};

layout(std430, binding = 1) buffer FieldBuffer {
    float fieldData[];
};

layout(std430, binding = 2) readonly buffer SignalsBuffer {
    float signals[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= ep.count) {
        return;
    }
    ExcPoint point = points[idx];
    if (ep.timestep < point.delay) return;
    uint position = ep.timestep - point.delay;
    if (point.period > 0u) position %= point.period;
    if (position >= point.signalLength) position = 0u;
    float value = point.amplitude * signals[point.signalOffset + position];
    if (value != 0.0) fieldData[point.index] += value;
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
    uint offset;
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
        outValues[pp.offset + idx] = voltData[points[idx].linear_index];
    } else {
        outValues[pp.offset + idx] = currData[points[idx].linear_index];
    }
}
)";

// Preserve float field multiplication, then reduce partial sums. CPU applies
// EPS0/MUE0 in double. SSE energy uses a different ordered float accumulator.
static const char* kShaderFastEnergy = R"(#version 450
#ifdef USE_FP64
#extension GL_ARB_gpu_shader_fp64 : require
#define ACC double
#else
#define ACC float
#endif
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer Voltage { float volt[]; };
layout(std430, binding = 1) readonly buffer Current { float curr[]; };
layout(std430, binding = 2) writeonly buffer Partials { ACC sums[]; };
layout(push_constant) uniform EnergyParams {
    uint dimX; uint dimY; uint dimZ; uint numCells;
    uint extentX; uint extentY; uint extentZ; uint groups;
} pc;
shared ACC electric[256];
shared ACC magnetic[256];
shared uint nonzero[256];
void main() {
    uint lane = gl_LocalInvocationID.x;
    uint group = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    uint item = group * 256u + lane;
    uint count = pc.extentX * pc.extentY * pc.extentZ;
    precise ACC e = ACC(0), h = ACC(0);
    uint fieldBits = 0u;
    if (item < count) {
        uint z = item % pc.extentZ;
        uint y = (item / pc.extentZ) % pc.extentY;
        uint x = item / (pc.extentZ * pc.extentY);
        uint index = x * pc.dimY * pc.dimZ + y * pc.dimZ + z;
        for (uint component = 0u; component < 3u; ++component) {
            uint position = index + component * pc.numCells;
            fieldBits |= (floatBitsToUint(volt[position]) | floatBitsToUint(curr[position])) & 0x7fffffffu;
            precise float ve = volt[position] * volt[position];
            precise float ih = curr[position] * curr[position];
            e += ACC(ve); h += ACC(ih);
        }
    }
    electric[lane] = e; magnetic[lane] = h;
    nonzero[lane] = fieldBits;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            electric[lane] += electric[lane + stride];
            magnetic[lane] += magnetic[lane + stride];
            nonzero[lane] |= nonzero[lane + stride];
        }
        barrier();
    }
    if (lane == 0u && group < pc.groups) {
        sums[3u * group] = electric[0];
        sums[3u * group + 1u] = magnetic[0];
        sums[3u * group + 2u] = ACC(nonzero[0] != 0u ? 1 : 0);
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

static const char* kShaderDebye = R"(#version 450
layout(local_size_x = 256) in;

struct DebyePoint {
    uint pos_idx;
    uint pole_offset;
    uint pole_count;
    float solve_coeff;
};

layout(std430, binding = 0) readonly buffer ParamsBuffer { DebyePoint points[]; };
layout(std430, binding = 1) buffer VoltBuffer { float voltData[]; };
layout(std430, binding = 2) readonly buffer PoleBuffer { vec2 poles[]; };
// First count entries hold each cell's pre-update sum / joint correction.
// The remaining entries hold the independent capacitor states of its poles.
layout(std430, binding = 3) buffer StateBuffer { float states[]; };
layout(push_constant) uniform DebyePushConstants {
    uint count;
    uint mode; // 0 = pre, 1 = post, 2 = apply
} pc;

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= pc.count) return;
    DebyePoint pt = points[id];
    float volt = voltData[pt.pos_idx];
    if (pc.mode == 0u) {
        float pre = 0.0;
        for (uint o = 0u; o < pt.pole_count; ++o) {
            uint p = pt.pole_offset + o;
            float old = states[pc.count + p];
            float w = poles[p].x * old + poles[p].y * volt;
            pre += w - old;
            states[pc.count + p] = w;
        }
        states[id] = pre;
    } else if (pc.mode == 1u) {
        float solved = (volt - states[id]) * pt.solve_coeff;
        states[id] = volt - solved;
        for (uint o = 0u; o < pt.pole_count; ++o) {
            uint p = pt.pole_offset + o;
            states[pc.count + p] += poles[p].y * solved;
        }
    } else {
        voltData[pt.pos_idx] -= states[id];
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

static const char* kShaderCylinderMultigrid = R"(#version 450
layout(local_size_x = 32, local_size_y = 4, local_size_z = 1) in;

struct Interpolation {
    uvec4 posP;
    uvec4 posPP;
    vec4 coeffP;
    vec4 coeffPP;
};

layout(std430, binding = 0) buffer ParentVolt { float parentVolt[]; };
layout(std430, binding = 1) buffer ParentCurr { float parentCurr[]; };
layout(std430, binding = 2) buffer ChildVolt { float childVolt[]; };
layout(std430, binding = 3) buffer ChildCurr { float childCurr[]; };
layout(std430, binding = 4) readonly buffer InterpolationBuffer { Interpolation interp[]; };

layout(push_constant) uniform MultigridPushConstants {
    uint parentDimX;
    uint parentDimY;
    uint parentDimZ;
    uint parentCells;
    uint childDimX;
    uint childDimY;
    uint childDimZ;
    uint childCells;
    uint radialStart;
    uint radialCount;
    uint mode;
} pc;

uint parentIndex(uint component, uint r, uint a, uint z) {
    return component * pc.parentCells + r * (pc.parentDimY * pc.parentDimZ) + a * pc.parentDimZ + z;
}

uint childIndex(uint component, uint r, uint a, uint z) {
    return component * pc.childCells + r * (pc.childDimY * pc.childDimZ) + a * pc.childDimZ + z;
}

void main() {
    uint z = gl_GlobalInvocationID.x;
    uint a = gl_GlobalInvocationID.y;
    uint localR = gl_GlobalInvocationID.z;

    if (pc.mode == 0u) {
        uint childAlphaCount = pc.parentDimY / 2u;
        if (z >= pc.childDimZ || a >= childAlphaCount || localR != 0u) return;
        uint parentA = 2u * a;
        uint r = pc.radialStart;
        childVolt[childIndex(0u, r, a, z)] = 0.0;
        childVolt[childIndex(2u, r, a, z)] = parentVolt[parentIndex(2u, r, parentA, z)];
        childVolt[childIndex(1u, r, a, z)] = parentVolt[parentIndex(1u, r, parentA, z)]
                                                 + parentVolt[parentIndex(1u, r, parentA + 1u, z)];
        return;
    }

    if (z >= pc.parentDimZ || a >= pc.parentDimY || localR >= pc.radialCount) return;
    uint r = pc.radialStart + localR;
    Interpolation ip = interp[a];
    bool current = (pc.mode == 1u || pc.mode == 3u);
    uint slot0 = current ? 2u : 0u;
    uint slot1 = current ? 3u : 1u;

    for (uint component = 0u; component < 3u; ++component) {
        uint slot = (component == 1u) ? slot1 : slot0;
        uint p = ip.posP[slot];
        uint pp = ip.posPP[slot];
        float value;
        if (current) {
            value = ip.coeffP[slot] * childCurr[childIndex(component, r, p, z)]
                  + ip.coeffPP[slot] * childCurr[childIndex(component, r, pp, z)];
            parentCurr[parentIndex(component, r, a, z)] = value;
        } else {
            value = ip.coeffP[slot] * childVolt[childIndex(component, r, p, z)]
                  + ip.coeffPP[slot] * childVolt[childIndex(component, r, pp, z)];
            parentVolt[parentIndex(component, r, a, z)] = value;
        }
    }
}
)";

} // namespace VulkanShaders

#endif // SHADERS_GLSL_H
