#pragma once
#include <common/defines.h>
#include <common/vector_typetraits.h>
#include <common/vector1.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace mpp::image::cuda
{
template <typename T> struct morph_compute_type
{
    using type = T;
};
template <> struct morph_compute_type<byte>
{
    using type = int;
};
template <> struct morph_compute_type<sbyte>
{
    using type = int;
};
template <> struct morph_compute_type<short>
{
    using type = int;
};
template <> struct morph_compute_type<ushort>
{
    using type = int;
};

template <typename T>
using morph_compute_type_t =
    same_vector_size_different_type_t<T, typename morph_compute_type<remove_vector_t<T>>::type>;

template <typename T> using morph_gray_compute_type_t = Vector1<typename morph_compute_type<remove_vector_t<T>>::type>;
} // namespace mpp::image::cuda
