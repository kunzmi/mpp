#pragma once
#include "defines.h"
#include <concepts>
#include <type_traits>

namespace opp
{
// forward declaration
template <SignedNumber T> struct Complex;

template <typename T> struct isComplexType : std::false_type
{
};
template <typename T> struct isComplexType<Complex<T>> : std::true_type
{
};

template <typename T> struct remove_complex
{
    using type = void;
};
template <typename T> struct remove_complex<Complex<T>>
{
    using type = T;
};

template <typename T> using remove_complex_t = typename remove_complex<T>::type;

template <typename T>
concept ComplexType = isComplexType<T>::value;

template <typename T>
concept ComplexOrNumber = Number<T> || ComplexType<T>;

template <typename T>
concept IntComplexType = isComplexType<T>::value && Integral<remove_complex_t<T>>;

template <typename T>
concept FloatingComplexType = isComplexType<T>::value && FloatingPoint<remove_complex_t<T>>;
} // namespace opp
