#pragma once
#include <common/defines.h>
#include <concepts>
#include <type_traits>

namespace opp
{

// forward declaration:
template <SignedNumber T> struct Complex;
template <typename T> struct isComplexType;

template <Number T> struct Vector1;
template <Number T> struct Vector2;
template <Number T> struct Vector3;
template <Number T> struct Vector4;

template <typename T> struct remove_vector
{
    using type = void;
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

template <typename T>
concept VectorType = (vector_size<T>::value > 0) && (vector_size<T>::value <= 4);

template <typename T>
concept IntVectorType =
    (vector_size<T>::value > 0) && (vector_size<T>::value <= 4) && Integral<typename remove_vector<T>::type>;

template <typename T>
concept FloatingVectorType =
    (vector_size<T>::value > 0) && (vector_size<T>::value <= 4) && FloatingPoint<typename remove_vector<T>::type>;

template <typename T>
concept VectorOrComplexType = VectorType<T> || isComplexType<T>::value;

} // namespace opp