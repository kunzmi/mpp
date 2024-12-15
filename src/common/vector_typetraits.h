#pragma once
#include <common/complex_typetraits.h>
#include <common/defines.h>
#include <concepts>
#include <type_traits>

namespace opp
{

// forward declaration:
template <ComplexOrNumber T> struct Vector1;
template <ComplexOrNumber T> struct Vector2;
template <ComplexOrNumber T> struct Vector3;
template <ComplexOrNumber T> struct Vector4;
template <ComplexOrNumber T> struct Vector4A;

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

//// well, Complex is not per se a vector, but we use C1 directly as complex, but C2 to C4 as vector...
// template <typename T> struct remove_vector<Complex<T>>
//{
//     using type = T;
// };

template <typename T> using remove_vector_t = typename remove_vector<T>::type;

template <typename T> struct vector_size : std::integral_constant<int, 0>
{
};
template <typename T> struct vector_size<Vector1<T>> : std::integral_constant<int, 1>
{
};
template <typename T> struct vector_size<Complex<T>> : std::integral_constant<int, 1>
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
concept IntVectorType = (vector_size_v<T> > 0) && (vector_size_v<T> <= 4) && Integral<remove_vector_t<T>>;

template <typename T>
concept SignedVectorType = (vector_size_v<T> > 0) && (vector_size_v<T> <= 4) && SignedNumber<remove_vector_t<T>>;

template <typename T>
concept FloatingVectorType = (vector_size_v<T> > 0) && (vector_size_v<T> <= 4) && FloatingPoint<remove_vector_t<T>>;

template <typename T>
concept VectorOrComplexType = VectorType<T> || ComplexType<T>;

template <typename T>
concept FloatingVectorOrComplexType =
    FloatingVectorType<T> || FloatingComplexType<T> || (VectorType<T> && ComplexType<remove_vector_t<T>>);

} // namespace opp