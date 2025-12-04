#pragma once
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/half_fp16.h>
#include <common/vector_typetraits.h>
#include <concepts>
#include <type_traits>

namespace mpp::image::cuda
{
// Defines what conversion operations are implemented:

template <typename TFrom, typename TTo> struct conversionImplemented : std::false_type
{
};
// 8s
template <> struct conversionImplemented<sbyte, byte> : std::true_type
{
};
template <> struct conversionImplemented<sbyte, ushort> : std::true_type
{
};
template <> struct conversionImplemented<sbyte, short> : std::true_type
{
};
template <> struct conversionImplemented<sbyte, uint> : std::true_type
{
};
template <> struct conversionImplemented<sbyte, int> : std::true_type
{
};
template <> struct conversionImplemented<sbyte, HalfFp16> : std::true_type
{
};
template <> struct conversionImplemented<sbyte, BFloat16> : std::true_type
{
};
template <> struct conversionImplemented<sbyte, float> : std::true_type
{
};
template <> struct conversionImplemented<sbyte, double> : std::true_type
{
};
// 8u
template <> struct conversionImplemented<byte, sbyte> : std::true_type
{
};
template <> struct conversionImplemented<byte, ushort> : std::true_type
{
};
template <> struct conversionImplemented<byte, short> : std::true_type
{
};
template <> struct conversionImplemented<byte, uint> : std::true_type
{
};
template <> struct conversionImplemented<byte, int> : std::true_type
{
};
template <> struct conversionImplemented<byte, HalfFp16> : std::true_type
{
};
template <> struct conversionImplemented<byte, BFloat16> : std::true_type
{
};
template <> struct conversionImplemented<byte, float> : std::true_type
{
};
template <> struct conversionImplemented<byte, double> : std::true_type
{
};
// 16s
template <> struct conversionImplemented<short, byte> : std::true_type
{
};
template <> struct conversionImplemented<short, sbyte> : std::true_type
{
};
template <> struct conversionImplemented<short, ushort> : std::true_type
{
};
template <> struct conversionImplemented<short, uint> : std::true_type
{
};
template <> struct conversionImplemented<short, int> : std::true_type
{
};
template <> struct conversionImplemented<short, HalfFp16> : std::true_type
{
};
template <> struct conversionImplemented<short, BFloat16> : std::true_type
{
};
template <> struct conversionImplemented<short, float> : std::true_type
{
};
template <> struct conversionImplemented<short, double> : std::true_type
{
};
// 16u
template <> struct conversionImplemented<ushort, byte> : std::true_type
{
};
template <> struct conversionImplemented<ushort, sbyte> : std::true_type
{
};
template <> struct conversionImplemented<ushort, short> : std::true_type
{
};
template <> struct conversionImplemented<ushort, uint> : std::true_type
{
};
template <> struct conversionImplemented<ushort, int> : std::true_type
{
};
template <> struct conversionImplemented<ushort, HalfFp16> : std::true_type
{
};
template <> struct conversionImplemented<ushort, BFloat16> : std::true_type
{
};
template <> struct conversionImplemented<ushort, float> : std::true_type
{
};
template <> struct conversionImplemented<ushort, double> : std::true_type
{
};
// 32s
template <> struct conversionImplemented<int, byte> : std::true_type
{
};
template <> struct conversionImplemented<int, sbyte> : std::true_type
{
};
template <> struct conversionImplemented<int, ushort> : std::true_type
{
};
template <> struct conversionImplemented<int, short> : std::true_type
{
};
template <> struct conversionImplemented<int, uint> : std::true_type
{
};
template <> struct conversionImplemented<int, HalfFp16> : std::true_type
{
};
template <> struct conversionImplemented<int, BFloat16> : std::true_type
{
};
template <> struct conversionImplemented<int, float> : std::true_type
{
};
template <> struct conversionImplemented<int, double> : std::true_type
{
};
// 32u
template <> struct conversionImplemented<uint, byte> : std::true_type
{
};
template <> struct conversionImplemented<uint, sbyte> : std::true_type
{
};
template <> struct conversionImplemented<uint, ushort> : std::true_type
{
};
template <> struct conversionImplemented<uint, short> : std::true_type
{
};
template <> struct conversionImplemented<uint, int> : std::true_type
{
};
template <> struct conversionImplemented<uint, HalfFp16> : std::true_type
{
};
template <> struct conversionImplemented<uint, BFloat16> : std::true_type
{
};
template <> struct conversionImplemented<uint, float> : std::true_type
{
};
template <> struct conversionImplemented<uint, double> : std::true_type
{
};
// 16f
template <> struct conversionImplemented<HalfFp16, float> : std::true_type
{
};
template <> struct conversionImplemented<HalfFp16, double> : std::true_type
{
};
// 16bf
template <> struct conversionImplemented<BFloat16, float> : std::true_type
{
};
template <> struct conversionImplemented<BFloat16, double> : std::true_type
{
};
// 16sc
template <> struct conversionImplemented<c_short, c_int> : std::true_type
{
};
template <> struct conversionImplemented<c_short, c_float> : std::true_type
{
};
// 32sc
template <> struct conversionImplemented<c_int, c_short> : std::true_type
{
};
template <> struct conversionImplemented<c_int, c_float> : std::true_type
{
};
// 32f
template <> struct conversionImplemented<float, HalfFp16> : std::true_type
{
};
template <> struct conversionImplemented<float, BFloat16> : std::true_type
{
};
template <> struct conversionImplemented<float, double> : std::true_type
{
};
// 64f
template <> struct conversionImplemented<double, HalfFp16> : std::true_type
{
};
template <> struct conversionImplemented<double, BFloat16> : std::true_type
{
};
template <> struct conversionImplemented<double, float> : std::true_type
{
};

template <typename TFrom, typename TTo>
constexpr inline bool conversionImplemented_v =
    conversionImplemented<remove_vector_t<TFrom>, remove_vector_t<TTo>>::value;

template <typename TFrom, typename TTo>
concept ConversionImplemented = (conversionImplemented_v<TFrom, TTo>);

template <typename TFrom, typename TTo> struct conversionRoundImplemented : std::false_type
{
};
// 16f
template <> struct conversionRoundImplemented<HalfFp16, sbyte> : std::true_type
{
};
template <> struct conversionRoundImplemented<HalfFp16, byte> : std::true_type
{
};
template <> struct conversionRoundImplemented<HalfFp16, short> : std::true_type
{
};
template <> struct conversionRoundImplemented<HalfFp16, ushort> : std::true_type
{
};
template <> struct conversionRoundImplemented<HalfFp16, int> : std::true_type
{
};
template <> struct conversionRoundImplemented<HalfFp16, uint> : std::true_type
{
};
// 16bf
template <> struct conversionRoundImplemented<BFloat16, sbyte> : std::true_type
{
};
template <> struct conversionRoundImplemented<BFloat16, byte> : std::true_type
{
};
template <> struct conversionRoundImplemented<BFloat16, short> : std::true_type
{
};
template <> struct conversionRoundImplemented<BFloat16, ushort> : std::true_type
{
};
template <> struct conversionRoundImplemented<BFloat16, int> : std::true_type
{
};
template <> struct conversionRoundImplemented<BFloat16, uint> : std::true_type
{
};
// 32fc
template <> struct conversionRoundImplemented<c_float, c_short> : std::true_type
{
};
template <> struct conversionRoundImplemented<c_float, c_int> : std::true_type
{
};
// 32f
template <> struct conversionRoundImplemented<float, sbyte> : std::true_type
{
};
template <> struct conversionRoundImplemented<float, byte> : std::true_type
{
};
template <> struct conversionRoundImplemented<float, short> : std::true_type
{
};
template <> struct conversionRoundImplemented<float, ushort> : std::true_type
{
};
template <> struct conversionRoundImplemented<float, int> : std::true_type
{
};
template <> struct conversionRoundImplemented<float, uint> : std::true_type
{
};
template <> struct conversionRoundImplemented<float, HalfFp16> : std::true_type
{
};
template <> struct conversionRoundImplemented<float, BFloat16> : std::true_type
{
};
// 64f
template <> struct conversionRoundImplemented<double, sbyte> : std::true_type
{
};
template <> struct conversionRoundImplemented<double, byte> : std::true_type
{
};
template <> struct conversionRoundImplemented<double, short> : std::true_type
{
};
template <> struct conversionRoundImplemented<double, ushort> : std::true_type
{
};
template <> struct conversionRoundImplemented<double, int> : std::true_type
{
};
template <> struct conversionRoundImplemented<double, uint> : std::true_type
{
};

template <typename TFrom, typename TTo>
constexpr inline bool conversionRoundImplemented_v =
    conversionRoundImplemented<remove_vector_t<TFrom>, remove_vector_t<TTo>>::value;

template <typename TFrom, typename TTo>
concept ConversionRoundImplemented = (conversionRoundImplemented_v<TFrom, TTo>);

template <typename TFrom, typename TTo> struct conversionRoundScaleImplemented : std::false_type
{
};
// 8u
template <> struct conversionRoundScaleImplemented<byte, sbyte> : std::true_type
{
};
// 16u
template <> struct conversionRoundScaleImplemented<ushort, sbyte> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<ushort, byte> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<ushort, short> : std::true_type
{
};
// 16s
template <> struct conversionRoundScaleImplemented<short, sbyte> : std::true_type
{
};
// 32u
template <> struct conversionRoundScaleImplemented<uint, sbyte> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<uint, byte> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<uint, short> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<uint, ushort> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<uint, int> : std::true_type
{
};
// 32s
template <> struct conversionRoundScaleImplemented<int, sbyte> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<int, byte> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<int, short> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<int, ushort> : std::true_type
{
};
// 32sc
template <> struct conversionRoundScaleImplemented<c_int, c_short> : std::true_type
{
};
// 32f
template <> struct conversionRoundScaleImplemented<float, sbyte> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<float, byte> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<float, short> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<float, ushort> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<float, int> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<float, uint> : std::true_type
{
};
// 32fc
template <> struct conversionRoundScaleImplemented<c_float, c_short> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<c_float, c_int> : std::true_type
{
};
// 64f
template <> struct conversionRoundScaleImplemented<double, sbyte> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<double, byte> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<double, short> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<double, ushort> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<double, int> : std::true_type
{
};
template <> struct conversionRoundScaleImplemented<double, uint> : std::true_type
{
};

template <typename TFrom, typename TTo>
constexpr inline bool conversionRoundScaleImplemented_v =
    conversionRoundScaleImplemented<remove_vector_t<TFrom>, remove_vector_t<TTo>>::value;

template <typename TFrom, typename TTo>
concept ConversionRoundScaleImplemented = (conversionRoundScaleImplemented_v<TFrom, TTo>);

} // namespace mpp::image::cuda