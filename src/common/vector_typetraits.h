#pragma once
#include <common/defines.h>
#include <common/numberTypes.h>
#include <concepts>
#include <type_traits>

namespace opp
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

template <typename T>
concept VectorType = (vector_size_v<T> > 0) && (vector_size_v<T> <= 4);

template <typename T>
concept IntVectorType = VectorType<T> && RealIntegral<remove_vector_t<T>>;

template <typename T>
concept SignedVectorType = VectorType<T> && RealSignedNumber<remove_vector_t<T>>;

template <typename T>
concept FloatingVectorType = VectorType<T> && RealFloatingPoint<remove_vector_t<T>>;

template <typename T>
concept ComplexVector = VectorType<T> || ComplexNumber<remove_vector_t<T>>;

template <typename T>
concept RealOrComplexFloatingVector =
    FloatingVectorType<T> || (ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>);

} // namespace opp