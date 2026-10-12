#ifndef VULKAN_PML_REGIONS_H
#define VULKAN_PML_REGIONS_H

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace VulkanPml {
// std430: three uvec4 records. Offsets retain region/component/x/y/z order.
struct alignas(16) Region {
	uint32_t origin[3], component;
	uint32_t extent[3], offset;
	uint32_t strideX, strideY, count, fieldBase;
};
static_assert(sizeof(Region) == 48, "PML descriptor layout mismatch");

inline uint32_t Validate(std::vector<Region>& regions, const uint32_t dims[3])
{
	uint64_t cells = 1;
	for (unsigned int d = 0; d < 3; ++d) {
		if (!dims[d] || cells > UINT32_MAX / dims[d])
			throw std::out_of_range("[openEMS Vulkan] Invalid PML grid size");
		cells *= dims[d];
	}
	if (cells > UINT32_MAX / 3u) throw std::out_of_range("[openEMS Vulkan] PML field size overflow");
	uint64_t offset = 0;
	for (size_t i = 0; i < regions.size(); ++i) {
		Region& r = regions[i];
		if (r.component >= 3u) throw std::out_of_range("[openEMS Vulkan] Invalid PML component");
		uint64_t count = 1;
		for (unsigned int d = 0; d < 3; ++d) {
			if (r.origin[d] >= dims[d] || !r.extent[d] || r.extent[d] > dims[d] - r.origin[d])
				throw std::out_of_range("[openEMS Vulkan] UPML range is outside the field grid");
			count *= r.extent[d]; // bounded above by the validated grid product
		}
		if (offset + count > UINT32_MAX) throw std::out_of_range("[openEMS Vulkan] PML offset overflow");
		r.offset = static_cast<uint32_t>(offset);
		r.count = static_cast<uint32_t>(count);
		r.strideX = dims[1] * dims[2];
		r.strideY = dims[2];
		r.fieldBase = static_cast<uint32_t>(r.component * cells + r.origin[0] * r.strideX +
		                                    r.origin[1] * r.strideY + r.origin[2]);
		offset += count;
		for (size_t j = 0; j < i; ++j) {
			const Region& s = regions[j];
			if (r.component != s.component) continue;
			bool overlap = true;
			for (unsigned int d = 0; d < 3; ++d)
				overlap = overlap && r.origin[d] < s.origin[d] + s.extent[d] && s.origin[d] < r.origin[d] + r.extent[d];
			if (overlap) throw std::out_of_range("[openEMS Vulkan] Overlapping UPML regions have multiple field writers");
		}
	}
	return static_cast<uint32_t>(offset);
}

inline uint32_t FieldIndex(const Region& r, uint32_t local)
{
	const uint32_t yz = r.extent[1] * r.extent[2];
	return r.fieldBase + (local / yz) * r.strideX + ((local % yz) / r.extent[2]) * r.strideY + local % r.extent[2];
}

struct Box { uint32_t origin[3], extent[3]; };

// Subtract disjoint PML boxes from the domain. Every remaining node has one writer.
inline std::vector<Box> Interior(const std::vector<Region>& regions, const uint32_t dims[3])
{
	if (regions.size() % 3u) throw std::logic_error("Incomplete PML component group");
	Box domain = {{0, 0, 0}, {dims[0], dims[1], dims[2]}};
	std::vector<Box> boxes(1, domain);
	for (size_t i = 0; i < regions.size(); i += 3u) {
		const Region& r = regions[i];
		for (unsigned int c = 0; c < 3u; ++c) {
			const Region& s = regions[i + c];
			if (s.component != c) throw std::logic_error("Unordered PML components");
			for (unsigned int d = 0; d < 3u; ++d)
				if (s.origin[d] != r.origin[d] || s.extent[d] != r.extent[d])
					throw std::logic_error("Mismatched PML component boxes");
		}
		std::vector<Box> next;
		for (Box box : boxes) {
			uint32_t lo[3], hi[3];
			bool intersects = true;
			for (unsigned int d = 0; d < 3u; ++d) {
				lo[d] = std::max(box.origin[d], r.origin[d]);
				hi[d] = std::min(box.origin[d] + box.extent[d], r.origin[d] + r.extent[d]);
				intersects = intersects && lo[d] < hi[d];
			}
			if (!intersects) { next.push_back(box); continue; }
			for (unsigned int d = 0; d < 3u; ++d) {
				const uint32_t end = box.origin[d] + box.extent[d];
				if (box.origin[d] < lo[d]) {
					Box part = box; part.extent[d] = lo[d] - box.origin[d]; next.push_back(part);
				}
				if (hi[d] < end) {
					Box part = box; part.origin[d] = hi[d]; part.extent[d] = end - hi[d]; next.push_back(part);
				}
				box.origin[d] = lo[d]; box.extent[d] = hi[d] - lo[d];
			}
		}
		boxes.swap(next);
	}
	return boxes;
}
}
#endif
