#pragma once

#include "defines.h"
#include <concepts>
#include <type_traits>

// definition of some concepts that classify the different number types used:

namespace opp
{
// forward declaration for HalfFp16, BFloat16
class HalfFp16;
class BFloat16;

// Is of type BFloat16, shortcut for std::same...
template <typename T>
concept IsBFloat16 = std::same_as<T, BFloat16>;

// Is of type HalfFp16, shortcut for std::same...
template <typename T>
concept IsHalfFp16 = std::same_as<T, HalfFp16>;

// Is of type HalfFp16 or BFloat16
template <typename T>
concept Is16BitFloat = IsBFloat16<T> || IsHalfFp16<T>;

// Define our own FP concept as HalfFp16 and BFloat16 are not part of std::floating_point and we don't want to modify
// std namespace
template <typename T>
concept RealFloatingPoint = std::floating_point<T> || Is16BitFloat<T>;

// Floating point number of native type, i.e. float or double, but no HalfFp16, BFloat16 etc.
template <typename T>
concept NativeFloatingPoint = std::floating_point<T>;

// Integer number of native type, i.e. int, short, ushort etc
template <typename T>
concept NativeIntegral = std::integral<T>;

// Define our own integral concept, who knows, maybe some future new int types... int4?
template <typename T>
concept RealIntegral = NativeIntegral<T>;

// all supported number formats: floating point and integral types, but not complex
template <typename T>
concept RealNumber = RealFloatingPoint<T> || RealIntegral<T>;

// all supported signed integer types
template <typename T>
concept RealSignedIntegral = std::signed_integral<T>;

// all supported unsigned integer types
template <typename T>
concept RealUnsignedIntegral = std::unsigned_integral<T>;

// all supported native number formats: floating point and integral types but not fp16, int4 etc
template <typename T>
concept NativeNumber = NativeFloatingPoint<T> || NativeIntegral<T>;

// floating point and signed integral types
template <typename T>
concept RealSignedNumber = RealFloatingPoint<T> || RealSignedIntegral<T>;

// Now that we have real signed numbers, we can declare our complex type and according concepts:
template <RealSignedNumber T> struct Complex;

template <typename T> struct is_complex_type : std::false_type
{
};
template <typename T> struct is_complex_type<Complex<T>> : std::true_type
{
};
template <class T> inline constexpr bool is_complex_type_v = is_complex_type<T>::value;

template <typename T> struct complex_basetype
{
    using type = T;
};
template <typename T> struct complex_basetype<Complex<T>>
{
    using type = T;
};

template <typename T> using complex_basetype_t = typename complex_basetype<T>::type;

template <typename T>
concept ComplexNumber = is_complex_type_v<T>;

template <typename T>
concept ComplexIntegral = ComplexNumber<T> && RealIntegral<complex_basetype_t<T>>;

template <typename T>
concept ComplexFloatingPoint = ComplexNumber<T> && RealFloatingPoint<complex_basetype_t<T>>;

template <typename T>
concept RealOrComplexIntegral = RealIntegral<T> || (ComplexNumber<T> && RealIntegral<complex_basetype_t<T>>);

template <typename T>
concept RealOrComplexFloatingPoint =
    RealFloatingPoint<T> || (ComplexNumber<T> && RealFloatingPoint<complex_basetype_t<T>>);

// all supported number formats: floating point and integral types, and complex
template <typename T>
concept Number = RealFloatingPoint<T> || RealIntegral<T> || ComplexNumber<T>;

// All types that are non native C++ types, currently HalfFp16, BFloat16 and Complex but have an interface that fits
// Vector1..4
template <typename T>
concept NonNativeNumber = Is16BitFloat<T> || ComplexNumber<T>;

// All types that are non native C++ types, currently HalfFp16, BFloat16 and real valued but have an interface that fits
// Vector1..4
template <typename T>
concept RealNonNativeNumber = Is16BitFloat<T>;

// currently HalfFp16, BFloat16 and complex floating point
template <typename T>
concept NonNativeFloatingPoint = Is16BitFloat<T> || (ComplexNumber<T> && RealFloatingPoint<complex_basetype_t<T>>);

//// All types that are native C++ types, all but HalfFp16, BFloat16 and Complex
// template <typename T>
// concept NativeNumber = !NonNativeNumber<T>;

// some more shortcuts:
template <typename T>
concept IsByte = std::same_as<T, byte>;
template <typename T>
concept IsSByte = std::same_as<T, sbyte>;
template <typename T>
concept IsUShort = std::same_as<T, ushort>;
template <typename T>
concept IsShort = std::same_as<T, short>;
template <typename T>
concept IsUInt = std::same_as<T, uint>;
template <typename T>
concept IsInt = std::same_as<T, int>;
template <typename T>
concept IsFloat = std::same_as<T, float>;
template <typename T>
concept IsDouble = std::same_as<T, double>;

} // namespace opp