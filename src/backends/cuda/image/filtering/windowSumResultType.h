#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h>

namespace mpp::image::cuda
{
template <typename T> struct window_sum_result_type
{
    using type = same_vector_size_different_type_t<T, float>;
};
template <typename T>
    requires Is16BitFloat<remove_vector_t<T>>
struct window_sum_result_type<T>
{
    using type = same_vector_size_different_type_t<T, remove_vector_t<T>>;
};
template <typename T>
    requires ComplexVector<T>
struct window_sum_result_type<T>
{
    using type = same_vector_size_different_type_t<T, Complex<float>>;
};
template <typename T>
    requires(sizeof(complex_basetype_t<remove_vector_t<T>>) >= 8)
struct window_sum_result_type<T>
{
    using type = same_vector_size_different_type_t<T, double>;
};

template <typename T> using window_sum_result_type_t = typename window_sum_result_type<T>::type;

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
