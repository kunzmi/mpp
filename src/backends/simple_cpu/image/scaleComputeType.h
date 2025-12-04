#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/image/pixelTypes.h>

namespace mpp::image::cpuSimple
{
// use scalar types here:
template <typename T, typename TTo> struct compute_type_scale_for
{
    using type                             = remove_vector_t<default_floating_compute_type_for_t<Vector1<T>>>;
    static constexpr bool use_int_division = false;
};
template <> struct compute_type_scale_for<byte, int>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<sbyte, int>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<short, int>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<ushort, int>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<int, byte>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<int, sbyte>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<int, short>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<int, ushort>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<uint, int>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<byte, uint>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<sbyte, uint>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<short, uint>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<ushort, uint>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<uint, byte>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<uint, sbyte>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<uint, short>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<uint, ushort>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<int, uint>
{
    using type                             = long64;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<c_short, c_int>
{
    using type                             = c_long;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<c_int, c_short>
{
    using type                             = c_long;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_scale_for<int, float>
{
    using type                             = float;
    static constexpr bool use_int_division = false;
};
template <> struct compute_type_scale_for<int, BFloat16>
{
    using type                             = float;
    static constexpr bool use_int_division = false;
};
template <> struct compute_type_scale_for<int, HalfFp16>
{
    using type                             = float;
    static constexpr bool use_int_division = false;
};
template <> struct compute_type_scale_for<uint, float>
{
    using type                             = float;
    static constexpr bool use_int_division = false;
};
template <> struct compute_type_scale_for<uint, BFloat16>
{
    using type                             = float;
    static constexpr bool use_int_division = false;
};
template <> struct compute_type_scale_for<uint, HalfFp16>
{
    using type                             = float;
    static constexpr bool use_int_division = false;
};
template <typename T> struct compute_type_scale_for<T, double>
{
    using type                             = double;
    static constexpr bool use_int_division = false;
};

template <typename T, typename TTo>
using compute_type_scale_for_t =
    same_vector_size_different_type_t<T,
                                      typename compute_type_scale_for<remove_vector_t<T>, remove_vector_t<TTo>>::type>;

template <typename T, typename TTo>
inline constexpr bool use_int_division_for_scale_v =
    compute_type_scale_for<remove_vector_t<T>, remove_vector_t<TTo>>::use_int_division;

} // namespace mpp::image::cpuSimple