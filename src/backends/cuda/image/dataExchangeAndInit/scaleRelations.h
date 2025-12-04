#pragma once
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/half_fp16.h>
#include <common/vector_typetraits.h>
#include <concepts>
#include <type_traits>

namespace mpp::image::cuda
{
// Defines what scale operations are implemented:

template <typename TFrom, typename TTo> struct scaleImplemented : std::false_type
{
};
// 8s
template <> struct scaleImplemented<sbyte, byte> : std::true_type
{
};
template <> struct scaleImplemented<sbyte, ushort> : std::true_type
{
};
template <> struct scaleImplemented<sbyte, short> : std::true_type
{
};
template <> struct scaleImplemented<sbyte, uint> : std::true_type
{
};
template <> struct scaleImplemented<sbyte, int> : std::true_type
{
};
template <> struct scaleImplemented<sbyte, HalfFp16> : std::true_type
{
};
template <> struct scaleImplemented<sbyte, BFloat16> : std::true_type
{
};
template <> struct scaleImplemented<sbyte, float> : std::true_type
{
};
template <> struct scaleImplemented<sbyte, double> : std::true_type
{
};
// 8u
template <> struct scaleImplemented<byte, sbyte> : std::true_type
{
};
template <> struct scaleImplemented<byte, ushort> : std::true_type
{
};
template <> struct scaleImplemented<byte, short> : std::true_type
{
};
template <> struct scaleImplemented<byte, uint> : std::true_type
{
};
template <> struct scaleImplemented<byte, int> : std::true_type
{
};
template <> struct scaleImplemented<byte, HalfFp16> : std::true_type
{
};
template <> struct scaleImplemented<byte, BFloat16> : std::true_type
{
};
template <> struct scaleImplemented<byte, float> : std::true_type
{
};
template <> struct scaleImplemented<byte, double> : std::true_type
{
};
// 16s
template <> struct scaleImplemented<short, byte> : std::true_type
{
};
template <> struct scaleImplemented<short, sbyte> : std::true_type
{
};
template <> struct scaleImplemented<short, ushort> : std::true_type
{
};
template <> struct scaleImplemented<short, uint> : std::true_type
{
};
template <> struct scaleImplemented<short, int> : std::true_type
{
};
template <> struct scaleImplemented<short, HalfFp16> : std::true_type
{
};
template <> struct scaleImplemented<short, BFloat16> : std::true_type
{
};
template <> struct scaleImplemented<short, float> : std::true_type
{
};
template <> struct scaleImplemented<short, double> : std::true_type
{
};
// 16u
template <> struct scaleImplemented<ushort, byte> : std::true_type
{
};
template <> struct scaleImplemented<ushort, sbyte> : std::true_type
{
};
template <> struct scaleImplemented<ushort, short> : std::true_type
{
};
template <> struct scaleImplemented<ushort, uint> : std::true_type
{
};
template <> struct scaleImplemented<ushort, int> : std::true_type
{
};
template <> struct scaleImplemented<ushort, HalfFp16> : std::true_type
{
};
template <> struct scaleImplemented<ushort, BFloat16> : std::true_type
{
};
template <> struct scaleImplemented<ushort, float> : std::true_type
{
};
template <> struct scaleImplemented<ushort, double> : std::true_type
{
};
// 32s
template <> struct scaleImplemented<int, byte> : std::true_type
{
};
template <> struct scaleImplemented<int, sbyte> : std::true_type
{
};
template <> struct scaleImplemented<int, ushort> : std::true_type
{
};
template <> struct scaleImplemented<int, short> : std::true_type
{
};
template <> struct scaleImplemented<int, uint> : std::true_type
{
};
template <> struct scaleImplemented<int, HalfFp16> : std::true_type
{
};
template <> struct scaleImplemented<int, BFloat16> : std::true_type
{
};
template <> struct scaleImplemented<int, float> : std::true_type
{
};
template <> struct scaleImplemented<int, double> : std::true_type
{
};
// 32u
template <> struct scaleImplemented<uint, byte> : std::true_type
{
};
template <> struct scaleImplemented<uint, sbyte> : std::true_type
{
};
template <> struct scaleImplemented<uint, ushort> : std::true_type
{
};
template <> struct scaleImplemented<uint, short> : std::true_type
{
};
template <> struct scaleImplemented<uint, int> : std::true_type
{
};
template <> struct scaleImplemented<uint, HalfFp16> : std::true_type
{
};
template <> struct scaleImplemented<uint, BFloat16> : std::true_type
{
};
template <> struct scaleImplemented<uint, float> : std::true_type
{
};
template <> struct scaleImplemented<uint, double> : std::true_type
{
};
// 16f
template <> struct scaleImplemented<HalfFp16, byte> : std::true_type
{
};
template <> struct scaleImplemented<HalfFp16, sbyte> : std::true_type
{
};
template <> struct scaleImplemented<HalfFp16, ushort> : std::true_type
{
};
template <> struct scaleImplemented<HalfFp16, short> : std::true_type
{
};
template <> struct scaleImplemented<HalfFp16, uint> : std::true_type
{
};
template <> struct scaleImplemented<HalfFp16, int> : std::true_type
{
};
template <> struct scaleImplemented<HalfFp16, BFloat16> : std::true_type
{
};
template <> struct scaleImplemented<HalfFp16, float> : std::true_type
{
};
template <> struct scaleImplemented<HalfFp16, double> : std::true_type
{
};
// 16bf
template <> struct scaleImplemented<BFloat16, byte> : std::true_type
{
};
template <> struct scaleImplemented<BFloat16, sbyte> : std::true_type
{
};
template <> struct scaleImplemented<BFloat16, ushort> : std::true_type
{
};
template <> struct scaleImplemented<BFloat16, short> : std::true_type
{
};
template <> struct scaleImplemented<BFloat16, uint> : std::true_type
{
};
template <> struct scaleImplemented<BFloat16, int> : std::true_type
{
};
template <> struct scaleImplemented<BFloat16, HalfFp16> : std::true_type
{
};
template <> struct scaleImplemented<BFloat16, float> : std::true_type
{
};
template <> struct scaleImplemented<BFloat16, double> : std::true_type
{
};
// 16sc
template <> struct scaleImplemented<c_short, c_int> : std::true_type
{
};
template <> struct scaleImplemented<c_short, c_float> : std::true_type
{
};
// 32sc
template <> struct scaleImplemented<c_int, c_short> : std::true_type
{
};
template <> struct scaleImplemented<c_int, c_float> : std::true_type
{
};
// 32f
template <> struct scaleImplemented<float, byte> : std::true_type
{
};
template <> struct scaleImplemented<float, sbyte> : std::true_type
{
};
template <> struct scaleImplemented<float, ushort> : std::true_type
{
};
template <> struct scaleImplemented<float, short> : std::true_type
{
};
template <> struct scaleImplemented<float, uint> : std::true_type
{
};
template <> struct scaleImplemented<float, int> : std::true_type
{
};
template <> struct scaleImplemented<float, BFloat16> : std::true_type
{
};
template <> struct scaleImplemented<float, HalfFp16> : std::true_type
{
};
template <> struct scaleImplemented<float, double> : std::true_type
{
};
// 64f
template <> struct scaleImplemented<double, byte> : std::true_type
{
};
template <> struct scaleImplemented<double, sbyte> : std::true_type
{
};
template <> struct scaleImplemented<double, ushort> : std::true_type
{
};
template <> struct scaleImplemented<double, short> : std::true_type
{
};
template <> struct scaleImplemented<double, uint> : std::true_type
{
};
template <> struct scaleImplemented<double, int> : std::true_type
{
};
template <> struct scaleImplemented<double, BFloat16> : std::true_type
{
};
template <> struct scaleImplemented<double, HalfFp16> : std::true_type
{
};
template <> struct scaleImplemented<double, float> : std::true_type
{
};
// 32fc
template <> struct scaleImplemented<c_float, c_short> : std::true_type
{
};
template <> struct scaleImplemented<c_float, c_int> : std::true_type
{
};

template <typename TFrom, typename TTo>
constexpr inline bool scaleImplemented_v = scaleImplemented<remove_vector_t<TFrom>, remove_vector_t<TTo>>::value;

template <typename TFrom, typename TTo>
concept ScaleImplemented = (scaleImplemented_v<TFrom, TTo>);

} // namespace mpp::image::cuda