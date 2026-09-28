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

} // namespace VulkanShaders

#endif // SHADERS_GLSL_H
