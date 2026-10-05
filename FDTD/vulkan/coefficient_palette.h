#ifndef VULKAN_COEFFICIENT_PALETTE_H
#define VULKAN_COEFFICIENT_PALETTE_H

#include "FDTD/operator.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace VulkanCoefficients {
// Direction is the position in this tuple. Geometry, anisotropy and boundary
// corrections are already incorporated in these final operator coefficients.
using Node = std::array<uint32_t, 12>;

template<size_t N> struct BitHash {
	size_t operator()(const std::array<uint32_t, N>& key) const {
		size_t hash = 2166136261u;
		for (uint32_t word : key) hash = (hash ^ word) * 16777619u;
		return hash;
	}
};

inline Node ReadNode(const Operator& op, unsigned int x, unsigned int y, unsigned int z)
{
	Node result;
	for (unsigned int n = 0; n < 3; ++n) {
		const float values[] = {op.GetVV(n, x, y, z), op.GetVI(n, x, y, z),
		                        op.GetII(n, x, y, z), op.GetIV(n, x, y, z)};
		static_assert(sizeof(float) == sizeof(uint32_t), "Palette requires 32-bit floats");
		std::memcpy(result.data() + 4u * n, values, sizeof(values));
	}
	return result;
}

struct Palette {
	std::vector<Node> tuples;
	std::vector<uint32_t> indices;
	bool complete = true;
	uint64_t Bytes() const { return uint64_t(tuples.size()) * 48u + uint64_t(indices.size()) * 4u; }
};

// A bounded candidate stops before a mostly unique mesh exhausts host memory.
// Incomplete candidates must never be uploaded or used for reconstruction.
template<class Reader>
Palette Build(uint32_t count, Reader read, uint32_t maxUnique = UINT32_MAX)
{
	Palette result;
	std::unordered_map<Node, uint32_t, BitHash<12>> lookup;
	result.indices.reserve(count);
	for (uint32_t i = 0; i < count; ++i) {
		const Node tuple = read(i);
		auto found = lookup.find(tuple);
		if (found == lookup.end()) {
			if (result.tuples.size() >= maxUnique) {
				result.complete = false;
				return result;
			}
			const uint32_t index = static_cast<uint32_t>(result.tuples.size());
			lookup.emplace(tuple, index);
			result.tuples.push_back(tuple);
			result.indices.push_back(index);
		} else result.indices.push_back(found->second);
	}
	return result;
}

inline Palette Build(const Operator& op, uint32_t maxUnique = UINT32_MAX)
{
	const uint32_t ny = op.GetNumberOfLines(1, true), nz = op.GetNumberOfLines(2, true);
	const uint64_t count = uint64_t(op.GetNumberOfLines(0, true)) * ny * nz;
	if (!ny || !nz || count > UINT32_MAX) throw std::length_error("Coefficient grid exceeds 32-bit indexing");
	return Build(static_cast<uint32_t>(count), [&](uint32_t i) {
		return ReadNode(op, i / (ny * nz), (i / nz) % ny, i % nz);
	}, maxUnique);
}

inline uint64_t CountComponents(const Operator& op)
{
	uint64_t count = 0;
	// Separate dictionaries preserve direction without storing a direction word.
	for (unsigned int n = 0; n < 3; ++n) {
		std::unordered_set<std::array<uint32_t, 4>, BitHash<4>> tuples;
		for (unsigned int x = 0; x < op.GetNumberOfLines(0, true); ++x)
		for (unsigned int y = 0; y < op.GetNumberOfLines(1, true); ++y)
		for (unsigned int z = 0; z < op.GetNumberOfLines(2, true); ++z) {
			const Node node = ReadNode(op, x, y, z);
			std::array<uint32_t, 4> tuple;
			std::copy(node.begin() + 4u * n, node.begin() + 4u * (n + 1u), tuple.begin());
			tuples.insert(tuple);
		}
		count += tuples.size();
	}
	return count;
}
}
#endif
