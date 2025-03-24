#pragma once
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace opp::image
{
enum class ReductionInitValue
{
    Zero,
    Min,
    Max
};

template <ReductionInitValue redVal, typename T> struct reduction_init_value
{
    static constexpr complex_basetype_t<remove_vector_t<T>> value =
        static_cast<complex_basetype_t<remove_vector_t<T>>>(0);
};

template <typename T> struct reduction_init_value<ReductionInitValue::Min, T>
{
    static constexpr complex_basetype_t<remove_vector_t<T>> value =
        numeric_limits<complex_basetype_t<remove_vector_t<T>>>::lowest();
};

// template <> struct reduction_init_value<ReductionInitValue::Min, Pixel16fC4A>
//{
//     // static constexpr HalfFp16 value = numeric_limits<HalfFp16>::lowest();
//     static constexpr float value = numeric_limits<float>::lowest();
// };

template <typename T> struct reduction_init_value<ReductionInitValue::Max, T>
{
    static constexpr complex_basetype_t<remove_vector_t<T>> value =
        numeric_limits<complex_basetype_t<remove_vector_t<T>>>::max();
};

// template <> struct reduction_init_value<ReductionInitValue::Max, Pixel16fC4A>
//{
//     static constexpr float value = numeric_limits<float>::max();
// };
template <ReductionInitValue redVal, typename T>
inline constexpr auto reduction_init_value_v = reduction_init_value<redVal, T>::value;

// template <typename T> struct reduction_init
//{
//     using type = complex_basetype_t<remove_vector_t<T>>;
// };
// template <typename T>
//     requires Is16BitFloat<complex_basetype_t<remove_vector_t<T>>>
// struct reduction_init<T>
//{
//     using type = float;
// };
// template <typename T>
//     requires RealSignedIntegral<complex_basetype_t<remove_vector_t<T>>>
// struct reduction_init<T>
//{
//     using type = long64;
// };
// template <typename T>
//     requires RealUnsignedIntegral<complex_basetype_t<remove_vector_t<T>>>
// struct reduction_init<T>
//{
//     using type = ulong64;
// };

// template <typename T> using reduction_init_t = typename reduction_init<T>::type;
//
// template <typename T> struct reduction_init_zero
//{
//     static constexpr complex_basetype_t<remove_vector_t<T>> value =
//         static_cast<complex_basetype_t<remove_vector_t<T>>>(0);
// };
// template <typename T>
//     requires Is16BitFloat<complex_basetype_t<remove_vector_t<T>>>
// struct reduction_init_zero<T>
//{
//     static constexpr float value = 0.0f;
// };
//
// template <class T> inline constexpr reduction_init_t<T> reduction_init_zero_v = reduction_init_zero<T>::value;
//
// template <typename T> struct reduction_init_max
//{
//     static constexpr complex_basetype_t<remove_vector_t<T>> value =
//         numeric_limits<complex_basetype_t<remove_vector_t<T>>>::max();
// };
// template <typename T>
//     requires Is16BitFloat<complex_basetype_t<remove_vector_t<T>>>
// struct reduction_init_max<T>
//{
//     static constexpr float value = numeric_limits<float>::max();
// };
//
// template <class T> inline constexpr reduction_init_t<T> reduction_init_max_v = reduction_init_max<T>::value;
//
// template <typename T> struct reduction_init_min
//{
//     static constexpr complex_basetype_t<remove_vector_t<T>> value =
//         numeric_limits<complex_basetype_t<remove_vector_t<T>>>::lowest();
// };
// template <typename T>
//     requires Is16BitFloat<complex_basetype_t<remove_vector_t<T>>>
// struct reduction_init_min<T>
//{
//     static constexpr float value = numeric_limits<float>::lowest();
// };
//
// template <class T> inline constexpr reduction_init_t<T> reduction_init_min_v = reduction_init_min<T>::value;

} // namespace opp::image
