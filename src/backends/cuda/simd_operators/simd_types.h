#pragma once

#include <common/bfloat16.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/image/pixelTypes.h>
#include <common/tupel.h>
#include <concepts>

namespace opp::cuda::simd
{
using byte1_8   = Tupel<Vector1<byte>, 8>;
using sbyte1_8  = Tupel<Vector1<sbyte>, 8>;
using byte2_4   = Tupel<Vector2<byte>, 4>;
using sbyte2_4  = Tupel<Vector2<sbyte>, 4>;
using ushort1_4 = Tupel<Vector1<ushort>, 4>;
using short1_4  = Tupel<Vector1<short>, 4>;
using bfloat1_4 = Tupel<Vector1<BFloat16>, 4>;
using hfloat1_4 = Tupel<Vector1<HalfFp16>, 4>;

// one byte types: tupels of 8 vector1 and 4 vector2 (vector2 has no SIMD for 1 byte types)
// two byte types: tupels of 4 vector1 (vector2 has already SIMD for 2 byte types)

template <typename T>
concept IsNativeSignedSimdType = std::same_as<T, sbyte1_8> || std::same_as<T, sbyte2_4> || std::same_as<T, short1_4>;

template <typename T>
concept IsNativeUnsignedSimdType = std::same_as<T, byte1_8> || std::same_as<T, byte2_4> || std::same_as<T, ushort1_4>;

template <typename T>
concept IsNativeSimdType = IsNativeSignedSimdType<T> || IsNativeUnsignedSimdType<T>;

template <typename T>
concept IsNonNativeSimdType = std::same_as<T, bfloat1_4> || std::same_as<T, hfloat1_4>;

template <typename T>
concept IsSignedSimdType = IsNativeSignedSimdType<T> || IsNonNativeSimdType<T>;

template <typename T>
concept IsSimdType = IsNativeSimdType<T> || IsNonNativeSimdType<T>;

} // namespace opp::cuda::simd
