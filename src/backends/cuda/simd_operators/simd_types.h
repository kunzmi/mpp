#pragma once

#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <concepts>

namespace opp::image::cuda
{

template <typename T> struct is_simd_type : std::false_type
{
};
template <> struct is_simd_type<opp::image::Pixel8uC1> : std::true_type
{
};
template <> struct is_simd_type<opp::image::Pixel8uC2> : std::true_type
{
};
template <> struct is_simd_type<opp::image::Pixel8sC1> : std::true_type
{
};
template <> struct is_simd_type<opp::image::Pixel8sC2> : std::true_type
{
};
template <> struct is_simd_type<opp::image::Pixel16uC1> : std::true_type
{
};
template <> struct is_simd_type<opp::image::Pixel16uC2> : std::true_type
{
};
template <> struct is_simd_type<opp::image::Pixel16sC1> : std::true_type
{
};
template <> struct is_simd_type<opp::image::Pixel16sC2> : std::true_type
{
};

template <typename T> inline constexpr bool is_simd_type_v = is_simd_type<T>::value;

template <typename T>
concept IsSIMDType = is_simd_type<T>::value;

template <typename T>
concept IsNOT_A_SIMDType = !is_simd_type<T>::value;

} // namespace opp::image::cuda