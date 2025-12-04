#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/image/pixelTypes.h>

namespace mpp::image::cpuSimple
{
// use scalar types here:
template <typename T, typename TTo> struct compute_type_convertScaleRound_for
{
    using type                             = remove_vector_t<default_floating_compute_type_for_t<Vector1<T>>>;
    static constexpr bool use_int_division = false;
};
template <typename TTo> struct compute_type_convertScaleRound_for<int, TTo>
{
    // TTo is either 8u/8s/16s or 16u, keeping ComputeT int is enough
    using type                             = int;
    static constexpr bool use_int_division = true;
};
template <typename TTo> struct compute_type_convertScaleRound_for<uint, TTo>
{
    // TTo is either 8u/8s/16s or 16u, keeping ComputeT uint is enough
    using type                             = uint;
    static constexpr bool use_int_division = true;
};
template <> struct compute_type_convertScaleRound_for<c_int, c_short>
{
    using type                             = c_int;
    static constexpr bool use_int_division = true;
};

template <typename T, typename TTo>
using compute_type_convertScaleRound_for_t = same_vector_size_different_type_t<
    T, typename compute_type_convertScaleRound_for<remove_vector_t<T>, remove_vector_t<TTo>>::type>;

template <typename T, typename TTo>
inline constexpr bool use_int_division_for_convertScaleRound_v =
    compute_type_convertScaleRound_for<remove_vector_t<T>, remove_vector_t<TTo>>::use_int_division;

} // namespace mpp::image::cpuSimple