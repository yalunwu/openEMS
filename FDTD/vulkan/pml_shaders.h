#ifndef VULKAN_PML_SHADERS_H
#define VULKAN_PML_SHADERS_H

#include <string>
#include <stdexcept>

namespace VulkanPml {
inline void Replace(std::string& source, const std::string& before, const std::string& after)
{
	const size_t at = source.find(before);
	if (at == std::string::npos) throw std::logic_error("PML shader template mismatch");
	source.replace(at, before.size(), after);
}

inline std::string DirectShader(const char* original)
{
	std::string source(original);
	Replace(source, "uint count;", "uint count; uint regionCount; uint regionIndex; uint perRegion;");
	Replace(source, "layout(std430, binding = 0) readonly buffer IndicesBuffer {\n    uint fieldIndices[];\n};", R"(
struct Region { uvec4 originComponent; uvec4 extentOffset; uvec4 strideCountBase; };
layout(std430, binding = 0) readonly buffer Regions { Region regions[]; };
)");
	Replace(source, "uint f_idx = fieldIndices[idx];", R"(
    uint selected = pc.regionIndex;
    if (pc.perRegion == 0u) {
        selected = 0u;
        while (selected + 1u < pc.regionCount && idx >= regions[selected + 1u].extentOffset.w) ++selected;
    }
    Region r = regions[selected];
    if (pc.perRegion != 0u) idx += r.extentOffset.w;
    uint local = idx - r.extentOffset.w;
    uint yz = r.extentOffset.y * r.extentOffset.z;
    uint f_idx = r.strideCountBase.w + (local / yz) * r.strideCountBase.x
               + ((local % yz) / r.extentOffset.z) * r.strideCountBase.y + local % r.extentOffset.z;
)");
	return source;
}

inline std::string FusedShader(const char* original, bool voltage, bool palette)
{
	std::string source(original);
	if (palette) Replace(source, "#version 450", "#version 450\n#define COEFFICIENT_PALETTE");
	Replace(source, "uint countX;", "uint countX; uint startY; uint countY; uint startZ; uint countZ; uint regionCount; uint regionIndex;");
	Replace(source, "void main() {", R"(
struct Region { uvec4 originComponent; uvec4 extentOffset; uvec4 strideCountBase; };
layout(std430, binding = 6) readonly buffer PmlCoefficients { vec4 pmlCoefficients[]; };
layout(std430, binding = 7) buffer PmlFlux { float pmlFlux[]; };
layout(std430, binding = 8) readonly buffer Regions { Region regions[]; };
uint selectedRegion;
uvec3 node;
float updateComponent(uint component, float oldField, float curl, float a, float b, bool updateYee) {
    if (selectedRegion == 0xfffffffeu) return updateYee ? a * oldField + b * curl : oldField;
    Region r = regions[selectedRegion + component];
    uvec3 local = node - r.originComponent.xyz;
    uint offset = r.extentOffset.w + (local.x * r.extentOffset.y + local.y) * r.extentOffset.z + local.z;
    vec4 c = pmlCoefficients[offset];
    float oldFlux = pmlFlux[offset];
    float p = c.x * oldField - c.y * oldFlux;
    float u = updateYee ? a * oldFlux + b * curl : oldFlux;
    pmlFlux[offset] = u;
    return p + c.z * u;
}
void main() {
)");
	Replace(source, "uint z = gl_GlobalInvocationID.x;", "uint z = pc.startZ + gl_GlobalInvocationID.x;");
	Replace(source, "uint y = gl_GlobalInvocationID.y;", "uint y = pc.startY + gl_GlobalInvocationID.y;");
	Replace(source, "if (localX >= pc.countX)", "if (localX >= pc.countX || gl_GlobalInvocationID.x >= pc.countZ || gl_GlobalInvocationID.y >= pc.countY)");
	if (!voltage) Replace(source, "x >= (pc.dimX - 1u) || y >= (pc.dimY - 1u) || z >= (pc.dimZ - 1u)",
	                      "x >= pc.dimX || y >= pc.dimY || z >= pc.dimZ");
	Replace(source, "uint sliceSize = pc.dimY * pc.dimZ;", R"(
    node = uvec3(x, y, z);
    selectedRegion = pc.regionIndex;
    if (selectedRegion == 0xffffffffu) {
        selectedRegion = 0xfffffffeu;
        for (uint i = 0u; i < pc.regionCount; i += 3u) {
            Region r = regions[i];
            if (all(greaterThanEqual(node, r.originComponent.xyz)) &&
                all(lessThan(node - r.originComponent.xyz, r.extentOffset.xyz))) {
                selectedRegion = i; break;
            }
        }
    }
    uint sliceSize = pc.dimY * pc.dimZ;
)");
	Replace(source, "uint nCells = pc.numCells;", voltage ? "uint nCells = pc.numCells; bool updateYee = true;" :
	        "uint nCells = pc.numCells; bool updateYee = x < pc.dimX - 1u && y < pc.dimY - 1u && z < pc.dimZ - 1u;");
	const char* addresses[] = {"idx", "nCells + idx", "2u * nCells + idx"};
	const char* curls[] = {"curl_x", "curl_y", "curl_z"};
	for (unsigned int c = 0; c < 3u; ++c) {
		const std::string component = std::to_string(c) + "u";
		const std::string field = std::string(voltage ? "volt[" : "curr[") + addresses[c] + "]";
		const std::string a = std::string(voltage ? "VV(" : "II(") + component + ")";
		const std::string b = std::string(voltage ? "VI(" : "IV(") + component + ")";
		Replace(source, field + " = " + a + " * " + field + " + " + b + " * " + curls[c] + ";",
		        field + " = updateComponent(" + component + ", " + field + ", " + curls[c] + ", " + a + ", " + b + ", updateYee);");
		if (!voltage) {
			const size_t begin = source.find(std::string("float ") + curls[c] + " = ");
			const size_t end = source.find(';', begin);
			if (begin == std::string::npos || end == std::string::npos) throw std::logic_error("Missing current curl");
			const size_t value = source.find(" = ", begin) + 3u;
			source.insert(end, ") : 0.0");
			source.insert(value, "updateYee ? (");
		}
	}
	if (!voltage) {
		// Opposite-phase fields are immutable. Explicit center reuse survives the
		// separate guarded curls, where the optimizer otherwise repeats three loads.
		const char* centers[] = {"centerX", "centerY", "centerZ"};
		for (unsigned int c = 0; c < 3u; ++c) {
			const std::string field = std::string("volt[") + (c == 0 ? "idx" : c == 1 ? "1u * nCells + idx" : "2u * nCells + idx") + "]";
			Replace(source, field, centers[c]);
			Replace(source, field, centers[c]);
		}
		Replace(source, "uint idx_yp1 = idx + pc.dimZ;", R"(
    float centerX = volt[idx];
    float centerY = volt[1u * nCells + idx];
    float centerZ = volt[2u * nCells + idx];
    uint idx_yp1 = idx + pc.dimZ;
)");
	}
	return source;
}
}
#endif
