#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

using Vec3 = std::array<uint32_t, 3>;

uint32_t Extract(EmitterState& s, uint32_t type, uint32_t vector, uint32_t index) {
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpCompositeExtract, type, result, vector, index);
	return result;
}

uint32_t Vector(EmitterState& s, uint32_t type, std::span<const uint32_t> components) {
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpCompositeConstruct, type, result, components);
	return result;
}

uint32_t NodeWord(EmitterState& s, uint32_t block, uint32_t word) {
	const auto offset = Binary(s, spv::OpShiftLeftLogical, TypeU32(s), word, ConstantU32(s, 2));
	const auto address = Binary(s, spv::OpIAdd, TypeScalarU64(s), block,
	                            Unary(s, spv::OpUConvert, TypeScalarU64(s), offset));
	const auto pointer = Unary(s, spv::OpConvertUToPtr, TypePhysicalU32Pointer(s), address);
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpLoad, TypeU32(s), result, pointer, spv::MemoryAccessAlignedMask, 4u);
	return result;
}

// `count` bits starting at bit `first` of the 64-byte node at `block`. Every caller passes a
// compile-time bit position, so the shifts and the mask fold away.
uint32_t NodeBits(EmitterState& s, uint32_t block, uint32_t first, uint32_t count) {
	const auto u = TypeU32(s), word = first / 32u, shift = first % 32u;
	auto value = NodeWord(s, block, ConstantU32(s, word));
	if (shift != 0u)
		value = Binary(s, spv::OpShiftRightLogical, u, value, ConstantU32(s, shift));
	if (shift + count > 32u)
		value = Binary(s, spv::OpBitwiseOr, u, value,
		    Binary(s, spv::OpShiftLeftLogical, u, NodeWord(s, block, ConstantU32(s, word + 1u)),
		           ConstantU32(s, 32u - shift)));
	return Binary(s, spv::OpBitwiseAnd, u, value, ConstantU32(s, (1u << count) - 1u));
}

// One bound of a PS5 type-6 node (the shared-exponent box): an 18-bit sign-magnitude `field` (bit
// 17 is the sign) on the fixed grid of the axis' shared biased `exponent`, renormalized to f32.
// The low mantissa bits the shift truncates are filled with ones when the bound rounds away from
// zero (minima towards -inf, maxima towards +inf), so the box only ever grows. Integer only.
uint32_t SharedExpBound(EmitterState& s, uint32_t field, uint32_t exponent, bool is_min) {
	const auto u = TypeU32(s), b = TypeBool(s);
	const auto sign = Binary(s, spv::OpShiftRightLogical, u, field, ConstantU32(s, 17));
	const auto magnitude = Binary(s, spv::OpBitwiseAnd, u, field, ConstantU32(s, 0x1ffffu));
	const auto negative = Binary(s, spv::OpINotEqual, b, sign, ConstantU32(s, 0));
	const auto away = is_min ? negative : Unary(s, spv::OpLogicalNot, b, negative);
	// A nonzero magnitude shifts its leading one up to the implicit bit, lowering the exponent (not
	// below zero: the result is denormal then). FindUMsb yields -1 for a zero magnitude, so `lead`
	// wraps to 17 and `down` becomes min(17, exponent) -- the constant case handled further down.
	const auto msb = EmitGlsl<GLSLstd450FindUMsb, IR::Type::U32>(s, magnitude);
	const auto lead = Binary(s, spv::OpISub, u, ConstantU32(s, 16), msb);
	const auto down = Select(s, u, Binary(s, spv::OpULessThan, b, lead, exponent), lead, exponent);
	const auto exp_n = Binary(s, spv::OpISub, u, exponent, down);
	const auto shift = Binary(s, spv::OpIAdd, u, ConstantU32(s, 7), down);
	const auto fill = Select(s, u,
	    Binary(s, spv::OpLogicalAnd, b, away,
	           Binary(s, spv::OpINotEqual, b, exp_n, ConstantU32(s, 0xffu))),
	    Binary(s, spv::OpISub, u,
	           Binary(s, spv::OpShiftLeftLogical, u, ConstantU32(s, 1), shift), ConstantU32(s, 1)),
	    ConstantU32(s, 0));
	const auto mant_n = Binary(s, spv::OpBitwiseAnd, u,
	    Binary(s, spv::OpBitwiseOr, u,
	           Binary(s, spv::OpShiftLeftLogical, u, magnitude, shift), fill),
	    ConstantU32(s, 0x7fffffu));
	// A zero magnitude rounding away is mantissa 0x7fffff at exp_n; otherwise a signed zero.
	const auto zero = Binary(s, spv::OpIEqual, b, magnitude, ConstantU32(s, 0));
	const auto mant = Select(s, u, Binary(s, spv::OpLogicalAnd, b, zero, away),
	                         ConstantU32(s, 0x7fffffu), mant_n);
	const auto exp = Select(s, u,
	    Binary(s, spv::OpLogicalAnd, b, zero, Unary(s, spv::OpLogicalNot, b, away)),
	    ConstantU32(s, 0), exp_n);
	return Unary(s, spv::OpBitcast, TypeF32(s), Binary(s, spv::OpBitwiseOr, u,
	    Binary(s, spv::OpBitwiseOr, u,
	           Binary(s, spv::OpShiftLeftLogical, u, sign, ConstantU32(s, 31)),
	           Binary(s, spv::OpShiftLeftLogical, u, exp, ConstantU32(s, 23))),
	    mant));
}

// The four child pointers of a PS5 type-6 node: bits 0-28 hold the first child's base in 64-byte
// units, each following child starts `SharedExpUnits(type)` blocks after the previous one, and the
// two bits at 41+2c mark a child absent when they are 3.
std::array<uint32_t, 4> BoxChildren6(EmitterState& s, uint32_t block) {
	const auto u = TypeU32(s), b = TypeBool(s);
	auto running = Binary(s, spv::OpBitwiseAnd, u, NodeWord(s, block, ConstantU32(s, 0)),
	                      ConstantU32(s, 0x1fffffffu));
	std::array<uint32_t, 4> children;
	for (uint32_t child = 0; child < 4; ++child) {
		const auto type = NodeBits(s, block, 29u + 3u * child, 3u);
		const auto empty = Binary(s, spv::OpIEqual, b, NodeBits(s, block, 41u + 2u * child, 2u),
		                          ConstantU32(s, 3));
		children[child] = Select(s, u, empty, ConstantU32(s, ~0u),
		    Binary(s, spv::OpBitwiseOr, u,
		           Binary(s, spv::OpShiftLeftLogical, u, running, ConstantU32(s, 3)), type));
		const auto is_type = [&](uint32_t value) {
			return Binary(s, spv::OpIEqual, b, type, ConstantU32(s, value));
		};
		const auto units = Select(s, u, is_type(2), ConstantU32(s, 4),
		    Select(s, u, Binary(s, spv::OpLogicalOr, b, is_type(3),
		                        Binary(s, spv::OpLogicalOr, b, is_type(5), is_type(6))),
		           ConstantU32(s, 2), ConstantU32(s, 1)));
		running = Binary(s, spv::OpIAdd, u, running, units);
	}
	return children;
}

// The six bounds (lo xyz then hi xyz) of one type-6 child: the three shared exponents at bits
// 56/64/72 and the child's six 18-bit fields from bit 80.
std::array<uint32_t, 6> Box6(EmitterState& s, uint32_t block, uint32_t child) {
	std::array<uint32_t, 3> exponent;
	for (uint32_t axis = 0; axis < 3; ++axis)
		exponent[axis] = NodeBits(s, block, 56u + 8u * axis, 8u);
	std::array<uint32_t, 6> result;
	for (uint32_t k = 0; k < 6; ++k)
		result[k] = SharedExpBound(s, NodeBits(s, block, 80u + 18u * (6u * child + k), 18u),
		                           exponent[k % 3u], k < 3u);
	return result;
}

uint32_t Triangle(EmitterState& s, uint32_t block, uint32_t kind, uint32_t bary,
                  Vec3 origin, Vec3 direction) {
	const auto f = TypeF32(s), u = TypeU32(s), b = TypeBool(s);
	const auto zero = ConstantF32Value(s, 0.0f);
	const auto op = [&](spv::Op code, uint32_t x, uint32_t y) {
		const auto value = Binary(s, code, f, x, y);
		s.builder.AddAnnotation(spv::OpDecorate, value, spv::DecorationNoContraction);
		return value;
	};
	const auto cmp = [&](spv::Op code, uint32_t x, uint32_t y) { return Binary(s, code, b, x, y); };
	const auto either = [&](uint32_t x, uint32_t y) { return Binary(s, spv::OpLogicalOr, b, x, y); };
	const auto both = [&](uint32_t x, uint32_t y) { return Binary(s, spv::OpLogicalAnd, b, x, y); };
	const auto abs = [&](uint32_t x) { return EmitGlsl<GLSLstd450FAbs, IR::Type::F32>(s, x); };
	const auto is_kind = [&](uint32_t n) { return cmp(spv::OpIEqual, kind, ConstantU32(s, n)); };
	// The four triangles are (0,1,2), (1,3,2), (2,3,4), (2,4,0).
	const std::array vertices {
		EmitGlsl<GLSLstd450UMin, IR::Type::U32>(s, kind, ConstantU32(s, 2)),
		Select(s, u, is_kind(0), ConstantU32(s, 1),
		       Select(s, u, is_kind(3), ConstantU32(s, 4), ConstantU32(s, 3))),
		Select(s, u, is_kind(2), ConstantU32(s, 4),
		       Select(s, u, is_kind(3), ConstantU32(s, 0), ConstantU32(s, 2))) };
	std::array<Vec3, 3> positions;
	for (uint32_t i = 0; i < 3; ++i) {
		const auto first = Binary(s, spv::OpIMul, u, vertices[i], ConstantU32(s, 3));
		for (uint32_t axis = 0; axis < 3; ++axis)
			positions[i][axis] = Unary(s, spv::OpBitcast, f,
			    NodeWord(s, block, Binary(s, spv::OpIAdd, u, first, ConstantU32(s, axis))));
	}
	const auto flag = NodeWord(s, block, ConstantU32(s, 15));
	const auto y_largest = cmp(spv::OpFOrdLessThan, abs(direction[0]), abs(direction[1]));
	const auto z_largest = cmp(spv::OpFOrdLessThan,
	    Select(s, f, y_largest, abs(direction[1]), abs(direction[0])), abs(direction[2]));
	const auto rotate = [&](Vec3 value) {
		Vec3 result;
		for (uint32_t i = 0; i < 3; ++i)
			result[i] = Select(s, f, z_largest, value[i],
			    Select(s, f, y_largest, value[(i + 2) % 3], value[(i + 1) % 3]));
		return result;
	};
	origin = rotate(origin);
	direction = rotate(direction);
	std::array<Vec3, 3> projected;
	for (uint32_t i = 0; i < 3; ++i) {
		const auto position = rotate(positions[i]);
		Vec3 relative;
		for (uint32_t axis = 0; axis < 3; ++axis)
			relative[axis] = op(spv::OpFSub, position[axis], origin[axis]);
		projected[i] = {op(spv::OpFSub, op(spv::OpFMul, relative[0], direction[2]),
		                              op(spv::OpFMul, direction[0], relative[2])),
		                op(spv::OpFSub, op(spv::OpFMul, relative[1], direction[2]),
		                              op(spv::OpFMul, direction[1], relative[2])), relative[2]};
	}
	Vec3 edge, weighted;
	for (uint32_t i = 0; i < 3; ++i) {
		const auto& first = projected[(i + 1) % 3];
		const auto& second = projected[(i + 2) % 3];
		edge[i] = op(spv::OpFSub, op(spv::OpFMul, second[0], first[1]),
		                        op(spv::OpFMul, second[1], first[0]));
		weighted[i] = op(spv::OpFMul, edge[i], direction[2]);
	}
	auto numerator = op(spv::OpFAdd, op(spv::OpFAdd,
	    op(spv::OpFMul, edge[0], projected[0][2]), op(spv::OpFMul, edge[1], projected[1][2])),
	    op(spv::OpFMul, edge[2], projected[2][2]));
	auto denominator = op(spv::OpFAdd, op(spv::OpFAdd, weighted[0], weighted[1]), weighted[2]);
	uint32_t negative = ConstantBool(s, false), positive = negative;
	for (auto value: edge) {
		negative = either(negative, cmp(spv::OpFOrdLessThan, value, zero));
		positive = either(positive, cmp(spv::OpFOrdGreaterThan, value, zero));
	}
	const auto winding = cmp(spv::OpFOrdGreaterThan, denominator, zero);
	auto missed = either(both(negative, positive), cmp(spv::OpFOrdEqual, denominator, zero));
	missed = either(missed, Unary(s, spv::OpIsNan, b, numerator));
	missed = either(missed, cmp(spv::OpFOrdLessThan,
	    Select(s, f, winding, numerator, Unary(s, spv::OpFNegate, f, numerator)), zero));
	for (uint32_t i = 0; i < 3; ++i) {
		const auto first = projected[(i + 1) % 3][1], second = projected[(i + 2) % 3][1];
		const auto first_zero = cmp(spv::OpFOrdEqual, first, zero);
		const auto horizontal = both(first_zero, cmp(spv::OpFOrdEqual, second, zero));
		const auto below = either(cmp(spv::OpFOrdLessThan, first, zero),
		    both(first_zero, cmp(spv::OpFOrdGreaterThan, second, zero)));
		const auto right = Binary(s, spv::OpLogicalNotEqual, b, below, winding);
		const auto excluded = Select(s, b, horizontal,
		    cmp(spv::OpFOrdGreaterThan, projected[i][1], zero), right);
		missed = either(missed, both(cmp(spv::OpFOrdEqual, edge[i], zero), excluded));
	}
	const auto remap = Binary(s, spv::OpShiftRightLogical, u, flag,
	    Binary(s, spv::OpIMul, u, kind, ConstantU32(s, 8)));
	const auto coordinate = [&](uint32_t shift) {
		const auto index = Binary(s, spv::OpBitwiseAnd, u,
		    Binary(s, spv::OpShiftRightLogical, u, remap, ConstantU32(s, shift)), ConstantU32(s, 3));
		return Unary(s, spv::OpBitcast, u, Select(s, f,
		    cmp(spv::OpIEqual, index, ConstantU32(s, 1)), weighted[1],
		    Select(s, f, cmp(spv::OpIEqual, index, ConstantU32(s, 2)), weighted[2], weighted[0])));
	};
	const std::array result {
		Unary(s, spv::OpBitcast, u, Select(s, f, missed, ConstantF32(s, 0x7f800000), numerator)),
		Unary(s, spv::OpBitcast, u, Select(s, f, missed, ConstantF32Value(s, 1.0f), denominator)),
		Select(s, u, bary, coordinate(0), Binary(s, spv::OpIAdd, u, flag, kind)),
		Select(s, u, bary, coordinate(2), Select(s, u, missed, ConstantU32(s, 0), ConstantU32(s, 1))) };
	return Vector(s, TypeU32Vector(s, 4), result);
}

std::array<uint32_t, 6> Box(EmitterState& s, uint32_t first, uint32_t second,
                           uint32_t half, uint32_t child) {
	const auto f = TypeF32(s);
	const auto half_label = s.builder.AllocateId(), full_label = s.builder.AllocateId();
	const auto merge = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
	s.builder.AddFunction(spv::OpBranchConditional, half, half_label, full_label);
	EmitLabel(s, half_label);
	std::array<uint32_t, 6> packed, full, result;
	for (uint32_t pair = 0; pair < 3; ++pair) {
		const auto word = NodeWord(s, first, ConstantU32(s, 4 + child * 3 + pair));
		const auto unpacked = EmitGlsl<GLSLstd450UnpackHalf2x16, IR::Type::F32x2>(s, word);
		packed[pair * 2] = Extract(s, f, unpacked, 0);
		packed[pair * 2 + 1] = Extract(s, f, unpacked, 1);
	}
	s.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(s, full_label);
	for (uint32_t component = 0; component < 6; ++component) {
		const auto word = 4 + child * 6 + component;
		full[component] = Unary(s, spv::OpBitcast, f,
		    NodeWord(s, word < 16 ? first : second, ConstantU32(s, word % 16)));
	}
	s.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(s, merge);
	for (uint32_t component = 0; component < 6; ++component) {
		result[component] = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpPhi, f, result[component], packed[component], half_label,
		                      full[component], full_label);
	}
	return result;
}

uint32_t BoxSlabs(EmitterState& s, const std::array<std::array<uint32_t, 6>, 4>& bounds,
                  const std::array<uint32_t, 4>& pointers, uint32_t grow, uint32_t sort,
                  uint32_t extent, const Vec3& origin, const Vec3& inverse) {
	const auto f = TypeF32(s), u = TypeU32(s), b = TypeBool(s);
	const auto zero = ConstantF32Value(s, 0.0f);
	std::array<uint32_t, 4> depths, indices;
	Vec3 forward;
	for (uint32_t axis = 0; axis < 3; ++axis)
		forward[axis] = Binary(s, spv::OpFOrdGreaterThanEqual, b, inverse[axis], zero);
	for (uint32_t child = 0; child < 4; ++child) {
		const auto& box = bounds[child];
		Vec3 near, far;
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const auto lo = Binary(s, spv::OpFMul, f,
			    Binary(s, spv::OpFSub, f, box[axis], origin[axis]), inverse[axis]);
			const auto hi = Binary(s, spv::OpFMul, f,
			    Binary(s, spv::OpFSub, f, box[axis + 3], origin[axis]), inverse[axis]);
			near[axis] = Select(s, f, forward[axis], lo, hi);
			far[axis] = Select(s, f, forward[axis], hi, lo);
		}
		const auto entry = EmitGlsl<GLSLstd450FMax, IR::Type::F32>(s,
		    EmitGlsl<GLSLstd450FMax, IR::Type::F32>(s, near[0], near[1]), near[2]);
		const auto exit = EmitGlsl<GLSLstd450FMin, IR::Type::F32>(s,
		    EmitGlsl<GLSLstd450FMin, IR::Type::F32>(s, far[0], far[1]), far[2]);
		// Valid hits have nonnegative exit distance. Saturate ULP growth at +infinity.
		const auto bits = Unary(s, spv::OpBitcast, u, Select(s, f,
		    Binary(s, spv::OpFOrdEqual, b, exit, zero), zero, exit));
		const auto grown_bits = EmitGlsl<GLSLstd450UMin, IR::Type::U32>(s,
		    Binary(s, spv::OpIAdd, u, bits, grow), ConstantU32(s, 0x7f800000));
		const auto grown = Unary(s, spv::OpBitcast, f, grown_bits);
		auto hit = Binary(s, spv::OpLogicalAnd, b,
		    Binary(s, spv::OpFOrdGreaterThanEqual, b, exit, zero),
		    Binary(s, spv::OpFOrdLessThan, b, entry, extent));
		hit = Binary(s, spv::OpLogicalAnd, b, hit,
		    Binary(s, spv::OpFOrdLessThanEqual, b, entry, grown));
		depths[child] = Unary(s, spv::OpBitcast, u, Select(s, f, hit,
		    Select(s, f, Binary(s, spv::OpFOrdGreaterThan, b, entry, zero), entry, zero),
		    ConstantF32(s, 0x7f800000)));
		indices[child] = Select(s, u, hit, pointers[child], ConstantU32(s, ~0u));
	}
	// A select network keeps the original child order when sorting is disabled.
	constexpr std::array<std::pair<uint32_t, uint32_t>, 5> network {{{0,1}, {2,3}, {0,2}, {1,3}, {1,2}}};
	for (size_t step = 0; step < network.size(); ++step) {
		const auto [a, c] = network[step];
		const auto swap = Binary(s, spv::OpLogicalAnd, b, sort,
		    Binary(s, spv::OpUGreaterThan, b, depths[a], depths[c]));
		const auto depth = depths[a], index = indices[a];
		const auto later = [&](uint32_t position) {
			return std::ranges::any_of(std::span(network).subspan(step + 1), [=](auto pair) {
				return pair.first == position || pair.second == position;
			});
		};
		if (later(a)) depths[a] = Select(s, u, swap, depths[c], depth);
		if (later(c)) depths[c] = Select(s, u, swap, depth, depths[c]);
		indices[a] = Select(s, u, swap, indices[c], index);
		indices[c] = Select(s, u, swap, index, indices[c]);
	}
	return Vector(s, TypeU32Vector(s, 4), indices);
}

// RDNA2 box nodes: the four child pointers are the node's first four dwords, and the bounds live
// in the same block (half floats, node type 4) or the block after it (f32, node type 5).
uint32_t Boxes(EmitterState& s, uint32_t first, uint32_t second, uint32_t half,
               uint32_t grow, uint32_t sort, uint32_t extent, const Vec3& origin, const Vec3& inverse) {
	std::array<std::array<uint32_t, 6>, 4> bounds;
	std::array<uint32_t, 4> pointers;
	for (uint32_t child = 0; child < 4; ++child) {
		bounds[child] = Box(s, first, second, half, child);
		pointers[child] = NodeWord(s, first, ConstantU32(s, child));
	}
	return BoxSlabs(s, bounds, pointers, grow, sort, extent, origin, inverse);
}

// PS5 type-6 box node: children and their shared-exponent bounds all in this one 64-byte block.
uint32_t Boxes6(EmitterState& s, uint32_t first, uint32_t grow, uint32_t sort, uint32_t extent,
                const Vec3& origin, const Vec3& inverse) {
	std::array<std::array<uint32_t, 6>, 4> bounds;
	for (uint32_t child = 0; child < 4; ++child)
		bounds[child] = Box6(s, first, child);
	return BoxSlabs(s, bounds, BoxChildren6(s, first), grow, sort, extent, origin, inverse);
}

} // namespace

void DefineBvhIntersect(EmitterState& s) {
	if (!s.requirements.bvh) return;
	const auto u = TypeU32(s), f = TypeF32(s), b = TypeBool(s), wide = TypeScalarU64(s);
	const auto vec4 = TypeU32Vector(s, 4), vec3 = TypeF32Vector(s, 3);
	const auto signature = s.builder.Type(spv::OpTypeFunction, vec4, vec4, wide, f, vec3, vec3, vec3);
	s.bvh_intersect_function = s.builder.AllocateId();
	s.builder.AddName(s.bvh_intersect_function, "bvh_intersect");
	s.builder.AddFunction(spv::OpFunction, vec4, s.bvh_intersect_function,
	                      spv::FunctionControlMaskNone, signature);
	std::array<uint32_t, 6> args;
	const std::array types {vec4, wide, f, vec3, vec3, vec3};
	for (uint32_t i = 0; i < args.size(); ++i) {
		args[i] = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpFunctionParameter, types[i], args[i]);
	}
	EmitLabel(s, s.builder.AllocateId());
	const auto descriptor = args[0], node = args[1], extent = args[2];
	Vec3 origin, direction, inverse;
	for (uint32_t i = 0; i < 3; ++i) {
		origin[i] = Extract(s, f, args[3], i);
		direction[i] = Extract(s, f, args[4], i);
		inverse[i] = Extract(s, f, args[5], i);
	}
	std::array<uint32_t, 4> words;
	for (uint32_t i = 0; i < 4; ++i) words[i] = Extract(s, u, descriptor, i);
	const auto and_bits = [&](uint32_t value, uint32_t mask) {
		return Binary(s, spv::OpBitwiseAnd, u, value, ConstantU32(s, mask));
	};
	const auto shr = [&](uint32_t value, uint32_t shift) {
		return Binary(s, spv::OpShiftRightLogical, u, value, ConstantU32(s, shift));
	};
	const auto kind = and_bits(Unary(s, spv::OpUConvert, u, node), 7);
	const auto triangle = Binary(s, spv::OpULessThan, b, kind, ConstantU32(s, 4));
	const auto full = Binary(s, spv::OpIEqual, b, kind, ConstantU32(s, 5));
	const auto half = Binary(s, spv::OpIEqual, b, kind, ConstantU32(s, 4));
	// Node type 6 is the PS5 shared-exponent box; it is not an RDNA2 node type.
	const auto shared = Binary(s, spv::OpIEqual, b, kind, ConstantU32(s, 6));
	const auto bary = Binary(s, spv::OpINotEqual, b, and_bits(words[3], 0x01000000), ConstantU32(s, 0));
	const auto sort = Binary(s, spv::OpINotEqual, b, and_bits(words[1], 0x80000000), ConstantU32(s, 0));
	const auto grow = and_bits(shr(words[1], 23), 0xff);
	const auto base = Binary(s, spv::OpShiftLeftLogical, wide,
	    DeviceAddressFromWords(s, words[0], and_bits(words[1], 0xff)), ConstantDeviceAddress(s, 8));
	const auto index = Binary(s, spv::OpShiftRightLogical, wide, node, ConstantDeviceAddress(s, 3));
	const auto last_index = Binary(s, spv::OpIAdd, wide, index,
	    Select(s, wide, full, ConstantDeviceAddress(s, 1), ConstantDeviceAddress(s, 0)));
	const auto address = Binary(s, spv::OpIAdd, wide, base,
	    Binary(s, spv::OpShiftLeftLogical, wide, index, ConstantDeviceAddress(s, 6)));
	const auto last_address = Binary(s, spv::OpIAdd, wide, base,
	    Binary(s, spv::OpShiftLeftLogical, wide, last_index, ConstantDeviceAddress(s, 6)));
	auto valid = Binary(s, spv::OpIEqual, b, shr(words[3], 28), ConstantU32(s, 8));
	const auto require = [&](uint32_t condition) {
		valid = Binary(s, spv::OpLogicalAnd, b, valid, condition);
	};
	require(Binary(s, spv::OpULessThanEqual, b, kind, ConstantU32(s, 6)));
	require(Binary(s, spv::OpULessThanEqual, b, last_index, DeviceAddressFromWords(s, words[2], and_bits(words[3], 0x3ff))));
	require(Binary(s, spv::OpULessThan, b, last_address, ConstantDeviceAddress(s, uint64_t{1} << 48)));
	require(Unary(s, spv::OpLogicalNot, b, Unary(s, spv::OpIsNan, b, extent)));
	for (uint32_t i = 0; i < 3; ++i) {
		for (const auto value: {origin[i], direction[i], inverse[i]})
			require(Unary(s, spv::OpLogicalNot, b, Unary(s, spv::OpIsNan, b, value)));
		require(Unary(s, spv::OpLogicalNot, b, Unary(s, spv::OpIsInf, b, origin[i])));
	}
	const std::array triangle_invalid {ConstantU32(s, 0x7f800000), ConstantU32(s, 0x3f800000),
	                                  ConstantU32(s, ~0u), ConstantU32(s, 0)};
	const std::array box_invalid {ConstantU32(s, ~0u), ConstantU32(s, ~0u),
	                             ConstantU32(s, ~0u), ConstantU32(s, ~0u)};
	std::array<uint32_t, 4> invalid_words;
	for (uint32_t i = 0; i < 4; ++i)
		invalid_words[i] = Select(s, u, triangle, triangle_invalid[i], box_invalid[i]);
	const auto invalid = Vector(s, vec4, invalid_words);
	const auto result = EmitValueOrDefaultIfCondition(s, valid, vec4, invalid, [&] {
		const auto first = GetBdaPointer(s, address);
		const auto second = EmitValueOrDefaultIfCondition(s, full, wide, ConstantDeviceAddress(s, 0), [&] {
			return GetBdaPointer(s, Binary(s, spv::OpIAdd, wide, address, ConstantDeviceAddress(s, 64)));
		});
		const auto present = Binary(s, spv::OpLogicalAnd, b,
		    Binary(s, spv::OpINotEqual, b, first, ConstantDeviceAddress(s, 0)),
		    Binary(s, spv::OpLogicalOr, b, Unary(s, spv::OpLogicalNot, b, full),
		        Binary(s, spv::OpINotEqual, b, second, ConstantDeviceAddress(s, 0))));
		return EmitValueOrDefaultIfCondition(s, present, vec4, invalid, [&] {
			const auto tri_label = s.builder.AllocateId(), box_label = s.builder.AllocateId();
			const auto merge = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
			s.builder.AddFunction(spv::OpBranchConditional, triangle, tri_label, box_label);
			EmitLabel(s, tri_label);
			const auto tri_value = Triangle(s, first, kind, bary, origin, direction);
			const auto tri_exit = s.current_label;
			s.builder.AddFunction(spv::OpBranch, merge);
			EmitLabel(s, box_label);
			// The node's own layout decides: a type-6 node keeps its children and bounds here, a
			// type-4/5 node keeps pointers here and its bounds here or in the following block.
			const auto shared_label = s.builder.AllocateId(), plain_label = s.builder.AllocateId();
			const auto box_merge = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpSelectionMerge, box_merge, spv::SelectionControlMaskNone);
			s.builder.AddFunction(spv::OpBranchConditional, shared, shared_label, plain_label);
			EmitLabel(s, shared_label);
			const auto shared_value = Boxes6(s, first, grow, sort, extent, origin, inverse);
			const auto shared_exit = s.current_label;
			s.builder.AddFunction(spv::OpBranch, box_merge);
			EmitLabel(s, plain_label);
			const auto plain_value = Boxes(s, first, second, half, grow, sort, extent, origin, inverse);
			const auto plain_exit = s.current_label;
			s.builder.AddFunction(spv::OpBranch, box_merge);
			EmitLabel(s, box_merge);
			const auto box_value = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpPhi, vec4, box_value, shared_value, shared_exit,
			                      plain_value, plain_exit);
			const auto box_exit = s.current_label;
			s.builder.AddFunction(spv::OpBranch, merge);
			EmitLabel(s, merge);
			const auto value = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpPhi, vec4, value, tri_value, tri_exit, box_value, box_exit);
			return value;
		});
	});
	s.builder.AddFunction(spv::OpReturnValue, result);
	s.builder.AddFunction(spv::OpFunctionEnd);
}

uint32_t EmitBvhIntersect(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& s = ctx.state;
	const auto* ray = ctx.ImageAddress(inst.Arg(1));
	const auto node_words = inst.Flags<uint32_t>();
	if (ray == nullptr || (node_words != 1u && node_words != 2u) ||
	    ray->NumArgs() < node_words + 10u || s.bvh_intersect_function == 0)
		ctx.Fail(inst, "BVH intersection requires its node pointer, ray components and shared function");
	const auto vector = [&](uint32_t first) {
		Vec3 values;
		for (uint32_t i = 0; i < 3; ++i)
			values[i] = Unary(s, spv::OpBitcast, TypeF32(s), ctx.Arg(*ray, first + i));
		return Vector(s, TypeF32Vector(s, 3), values);
	};
	const auto descriptor = ctx.Arg(inst, 0);
	const auto node = node_words == 2u
	                      ? DeviceAddressFromWords(s, ctx.Arg(*ray, 0), ctx.Arg(*ray, 1))
	                      : Unary(s, spv::OpUConvert, TypeScalarU64(s), ctx.Arg(*ray, 0));
	const auto extent = Unary(s, spv::OpBitcast, TypeF32(s), ctx.Arg(*ray, node_words));
	const auto origin = vector(node_words + 1u), direction = vector(node_words + 4u),
	           inverse = vector(node_words + 7u);
	return EmitValueOrDefaultIfCondition(s, ctx.Arg(inst, 2), TypeU32Vector(s, 4),
	    ConstantU32CompositeZero(s, 4), [&] {
		const auto result = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpFunctionCall, TypeU32Vector(s, 4), result,
		    s.bvh_intersect_function, descriptor, node, extent, origin, direction, inverse);
		return result;
	});
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
