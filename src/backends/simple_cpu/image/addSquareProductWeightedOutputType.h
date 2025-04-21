#pragma once
#include <common/bfloat16.h>
#include <common/half_fp16.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace opp::image::cpuSimple
{
// Get the output type for AddSquare, AddProduct and AddWeighted functions:
template <AnyVector T> struct add_spw_output_for
{
    using type = same_vector_size_different_type<T, float>::type;
};
template <AnyVector T>
    requires std::same_as<remove_vector_t<T>, int>
struct add_spw_output_for<T>
{
    using type = same_vector_size_different_type<T, double>::type;
};
template <AnyVector T>
    requires std::same_as<remove_vector_t<T>, uint>
struct add_spw_output_for<T>
{
    using type = same_vector_size_different_type<T, double>::type;
};
template <AnyVector T>
    requires std::same_as<remove_vector_t<T>, double>
struct add_spw_output_for<T>
{
    using type = same_vector_size_different_type<T, double>::type;
};
template <AnyVector T>
    requires std::same_as<remove_vector_t<T>, HalfFp16>
struct add_spw_output_for<T>
{
    using type = same_vector_size_different_type<T, HalfFp16>::type;
};
template <AnyVector T>
    requires std::same_as<remove_vector_t<T>, BFloat16>
struct add_spw_output_for<T>
{
    using type = same_vector_size_different_type<T, BFloat16>::type;
};
template <AnyVector T>
    requires ComplexVector<T>
struct add_spw_output_for<T>
{
    using type = same_vector_size_different_type<T, Complex<float>>::type;
};

template <typename T> using add_spw_output_for_t = typename add_spw_output_for<T>::type;
} // namespace opp::image::cpuSimple