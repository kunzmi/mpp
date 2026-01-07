#pragma once
#include "dllexport_common.h"
#include <common/defines.h>
#include <common/numberTypes.h>
#include <concepts>
#include <type_traits>

namespace mpp
{

// forward declaration:
template <Number T> struct Vector1;
template <Number T> struct Vector2;
template <Number T> struct Vector3;
template <Number T> struct Vector4;
template <Number T> struct Vector4A;

template <typename T> struct remove_vector
{
    using type = T;
};
template <typename T> struct remove_vector<Vector1<T>>
{
    using type = T;
};
template <typename T> struct remove_vector<Vector2<T>>
{
    using type = T;
};
template <typename T> struct remove_vector<Vector3<T>>
{
    using type = T;
};
template <typename T> struct remove_vector<Vector4<T>>
{
    using type = T;
};
template <typename T> struct remove_vector<Vector4A<T>>
{
    using type = T;
};

template <typename T> using remove_vector_t = typename remove_vector<T>::type;

template <typename T> struct vector_size : std::integral_constant<int, 0>
{
};
template <typename T> struct vector_size<Vector1<T>> : std::integral_constant<int, 1>
{
};
template <typename T> struct vector_size<Vector2<T>> : std::integral_constant<int, 2>
{
};
template <typename T> struct vector_size<Vector3<T>> : std::integral_constant<int, 3>
{
};
template <typename T> struct vector_size<Vector4<T>> : std::integral_constant<int, 4>
{
};
template <typename T> struct vector_size<Vector4A<T>> : std::integral_constant<int, 4>
{
};

template <class T> inline constexpr int vector_size_v = vector_size<T>::value;

// the number of channel components used for computations (3 for Vector4A, N for all others)
template <typename T> struct vector_active_size : std::integral_constant<int, 0>
{
};
template <typename T> struct vector_active_size<Vector1<T>> : std::integral_constant<int, 1>
{
};
template <typename T> struct vector_active_size<Vector2<T>> : std::integral_constant<int, 2>
{
};
template <typename T> struct vector_active_size<Vector3<T>> : std::integral_constant<int, 3>
{
};
template <typename T> struct vector_active_size<Vector4<T>> : std::integral_constant<int, 4>
{
};
template <typename T> struct vector_active_size<Vector4A<T>> : std::integral_constant<int, 3>
{
};

template <class T> inline constexpr int vector_active_size_v = vector_active_size<T>::value;

template <typename TVector, typename TOther> struct same_vector_size_different_type
{
    using type = TOther;
};
template <typename TVector, typename TOther> struct same_vector_size_different_type<Vector1<TVector>, TOther>
{
    using type = Vector1<TOther>;
};
template <typename TVector, typename TOther> struct same_vector_size_different_type<Vector2<TVector>, TOther>
{
    using type = Vector2<TOther>;
};
template <typename TVector, typename TOther> struct same_vector_size_different_type<Vector3<TVector>, TOther>
{
    using type = Vector3<TOther>;
};
template <typename TVector, typename TOther> struct same_vector_size_different_type<Vector4<TVector>, TOther>
{
    using type = Vector4<TOther>;
};
template <typename TVector, typename TOther> struct same_vector_size_different_type<Vector4A<TVector>, TOther>
{
    using type = Vector4A<TOther>;
};

template <typename TVector, typename TOther>
using same_vector_size_different_type_t = typename same_vector_size_different_type<TVector, TOther>::type;

template <typename T>
concept AnyVector = (vector_size_v<T> > 0) && (vector_size_v<T> <= 4);

template <typename T>
concept RealVector = AnyVector<T> && RealNumber<remove_vector_t<T>>;

template <typename T>
concept RealIntVector = AnyVector<T> && RealIntegral<remove_vector_t<T>>;

template <typename T>
concept RealSignedVector = AnyVector<T> && RealSignedNumber<remove_vector_t<T>>;

template <typename T>
concept RealUnsignedVector = AnyVector<T> && RealUnsignedNumber<remove_vector_t<T>>;

template <typename T>
concept RealFloatingVector = AnyVector<T> && RealFloatingPoint<remove_vector_t<T>>;

template <typename T>
concept ComplexVector = AnyVector<T> && ComplexNumber<remove_vector_t<T>>;

template <typename T>
concept RealOrComplexVector = AnyVector<T> && (RealNumber<remove_vector_t<T>> || ComplexNumber<remove_vector_t<T>>);

template <typename T>
concept RealOrComplexFloatingVector = RealOrComplexVector<T> && RealOrComplexFloatingPoint<remove_vector_t<T>>;

template <typename T>
concept RealOrComplexIntVector = RealOrComplexVector<T> && RealOrComplexIntegral<remove_vector_t<T>>;

} // namespace mpp