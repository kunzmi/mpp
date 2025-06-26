#pragma once
#include <common/bfloat16.h>
#include <common/half_fp16.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace mpp::image::cpuSimple
{
// level types for histogram even:
template <typename SrcT> struct hist_even_level_types_scalar_for
{
    using levelType = int;
};
template <> struct hist_even_level_types_scalar_for<uint>
{
    using levelType = ulong64;
};
template <> struct hist_even_level_types_scalar_for<int>
{
    using levelType = long64;
};

template <> struct hist_even_level_types_scalar_for<HalfFp16>
{
    using levelType = float;
};
template <> struct hist_even_level_types_scalar_for<BFloat16>
{
    using levelType = float;
};

template <> struct hist_even_level_types_scalar_for<float>
{
    using levelType = float;
};
template <> struct hist_even_level_types_scalar_for<double>
{
    using levelType = double;
};

// compute and result types for sum reduction:
template <typename SrcT> struct hist_even_level_types_for
{
    using levelType =
        same_vector_size_different_type_t<SrcT,
                                          typename hist_even_level_types_scalar_for<remove_vector_t<SrcT>>::levelType>;
};

template <typename T> using hist_even_level_types_for_t = typename hist_even_level_types_for<T>::levelType;

// level types for histogram range:
template <typename SrcT> struct hist_range_types_scalar_for
{
    using levelType = int;
};
template <> struct hist_range_types_scalar_for<uint>
{
    using levelType = ulong64;
};
template <> struct hist_range_types_scalar_for<int>
{
    using levelType = long64;
};

template <> struct hist_range_types_scalar_for<HalfFp16>
{
    using levelType = float;
};
template <> struct hist_range_types_scalar_for<BFloat16>
{
    using levelType = float;
};

template <> struct hist_range_types_scalar_for<float>
{
    using levelType = float;
};
template <> struct hist_range_types_scalar_for<double>
{
    using levelType = double;
};

// compute and result types for histogram range:
template <typename SrcT> struct hist_range_types_for
{
    using levelType = typename hist_range_types_scalar_for<remove_vector_t<SrcT>>::levelType;
};

template <typename T>
using hist_range_types_for_t = same_vector_size_different_type_t<T, typename hist_range_types_for<T>::levelType>;

} // namespace mpp::image::cpuSimple