#pragma once
#include "defines.h"
#include "dllexport_common.h"
#include "mpp_defs.h"
#include "needSaturationClamp.h"
#include "numberTypes.h"
#include "vector_typetraits.h"
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/half_fp16.h>
#include <concepts>
#include <iostream>
#include <type_traits>

namespace mpp
{

// forward declaration:
template <Number T> struct MPPEXPORTFWDDECL_COMMON Vector1;
template <Number T> struct MPPEXPORTFWDDECL_COMMON Vector2;
template <Number T> struct MPPEXPORTFWDDECL_COMMON Vector3;
template <Number T> struct MPPEXPORTFWDDECL_COMMON Vector4;

enum class Axis3D // NOLINT(performance-enum-size)
{
    X = 0,
    Y = 1,
    Z = 2
};

MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Axis3D &aAxis);
MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Axis3D &aAxis);

/// <summary>
/// A three T component vector. Can replace CUDA's vector3 types
/// </summary>
template <Number T> struct alignas(sizeof(T)) MPPEXPORT_COMMON Vector3
{
    T x; // NOLINT
    T y; // NOLINT
    T z; // NOLINT

#pragma region Constructors
    /// <summary>
    /// Default constructor does not initialize the members
    /// </summary>
    Vector3() noexcept = default;

    /// <summary>
    /// Initializes vector to all components = aVal
    /// </summary>
    DEVICE_CODE constexpr Vector3(T aVal) noexcept : x(aVal), y(aVal), z(aVal) // NOLINT
    {
    }

    /// <summary>
    /// Initializes vector to all components = aVal (especially when set to 0)
    /// </summary>
    DEVICE_CODE constexpr Vector3(int aVal) noexcept // NOLINT
        requires(!IsInt<T>)
        : x(static_cast<T>(aVal)), y(static_cast<T>(aVal)), z(static_cast<T>(aVal))
    {
    }

    /// <summary>
    /// Initializes vector to all components = [aVal[0], aVal[1], aVal[2]]
    /// </summary>
    DEVICE_CODE constexpr explicit Vector3(T aVal[3]) // NOLINT
        : x(aVal[0]), y(aVal[1]), z(aVal[2])
    {
    }

    /// <summary>
    /// Initializes vector to [aX, aY, aZ]
    /// </summary>
    DEVICE_CODE constexpr Vector3(T aX, T aY, T aZ) noexcept : x(aX), y(aY), z(aZ) // NOLINT
    {
    }

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.
    /// </summary>
    template <Number T2> DEVICE_CODE Vector3(const Vector3<T2> &aVec) noexcept; // NOLINT

    /// <summary>
    /// Type conversion with saturation if needed<para/>
    /// E.g.: when converting int to byte, values are clamped to 0..255<para/>
    /// But when converting byte to int, no clamping operation is performed.<para/>
    /// If we can modify the input variable, no need to allocate temporary storage for clamping.
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector3(Vector3<T2> &aVec) noexcept // NOLINT &&
        requires(!std::same_as<T, T2>);

    /// <summary>
    /// Type conversion using CUDA intrinsics for float to BFloat/Halffloat and complex
    /// </summary>
    template <Number T2>
    DEVICE_CODE Vector3(const Vector3<T2> &aVec, RoundingMode aRoundingMode)
        requires((IsBFloat16<T> || IsHalfFp16<T>) && IsFloat<T2>) ||
                (ComplexFloatingPoint<T> && ComplexFloatingPoint<T2> &&
                 NonNativeFloatingPoint<complex_basetype_t<remove_vector_t<T>>> &&
                 std::same_as<float, complex_basetype_t<remove_vector_t<T2>>>);

    ~Vector3() = default;

    Vector3(const Vector3 &) noexcept            = default;
    Vector3(Vector3 &&) noexcept                 = default;
    Vector3 &operator=(const Vector3 &) noexcept = default;
    Vector3 &operator=(Vector3 &&) noexcept      = default;
#pragma endregion

#pragma region Operators
    // don't use space-ship operator as it returns true if any comparison returns true.
    // But NPP only returns true if all channels fulfill the comparison.
    // auto operator<=>(const Vector3 &) const = default;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector3 &aLeft, const Vector3 &aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector3 &aLeft, const Vector3 &aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector3 &aLeft, const Vector3 &aRight, T aEpsilon)
        requires Is16BitFloat<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] static bool EqEps(const Vector3 &aLeft, const Vector3 &aRight,
                                                complex_basetype_t<T> aEpsilon)
        requires ComplexFloatingPoint<T>;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<(const Vector3 &aOther) const
        requires RealNumber<T>;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<=(const Vector3 &aOther) const
        requires RealNumber<T>;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>(const Vector3 &aOther) const
        requires RealNumber<T>;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>=(const Vector3 &aOther) const
        requires RealNumber<T>;

    /// <summary>
    /// Returns true if each element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator==(const Vector3 &aOther) const;

    /// <summary>
    /// Returns true if any element comparison is true
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator!=(const Vector3 &aOther) const;

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3 operator-() const
        requires RealSignedNumber<T> || ComplexNumber<T>;

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector3 &operator+=(T aOther);

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector3 &operator+=(complex_basetype_t<T> aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE Vector3 &operator+=(const Vector3 &aOther);

    /// <summary>
    /// Component wise addition
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3 operator+(const Vector3 &aOther) const;

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector3 &operator-=(T aOther);

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector3 &operator-=(complex_basetype_t<T> aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE Vector3 &operator-=(const Vector3 &aOther);

    /// <summary>
    /// Component wise subtraction (inverted inplace sub: this = aOther - this)
    /// </summary>
    DEVICE_CODE Vector3 &SubInv(const Vector3 &aOther);

    /// <summary>
    /// Component wise subtraction
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3 operator-(const Vector3 &aOther) const;

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector3 &operator*=(T aOther);

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector3 &operator*=(complex_basetype_t<T> aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE Vector3 &operator*=(const Vector3 &aOther);

    /// <summary>
    /// Component wise multiplication
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3 operator*(const Vector3 &aOther) const;

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector3 &operator/=(T aOther);

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector3 &operator/=(complex_basetype_t<T> aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE Vector3 &operator/=(const Vector3 &aOther);

    /// <summary>
    /// Component wise division (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector3 &DivInv(const Vector3 &aOther);

    /// <summary>
    /// Component wise division
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3 operator/(const Vector3 &aOther) const;

    /// <summary>
    /// Inplace Integer division with element wise round()
    /// </summary>
    DEVICE_CODE Vector3 &DivRound(const Vector3 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector3 &DivRoundNearest(const Vector3 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector3 &DivRoundZero(const Vector3 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise floor()
    /// </summary>
    DEVICE_CODE Vector3 &DivFloor(const Vector3 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE Vector3 &DivCeil(const Vector3 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector3 &DivInvRound(const Vector3 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round nearest ties to even (inverted inplace div: this =
    /// aOther / this)
    /// </summary>
    DEVICE_CODE Vector3 &DivInvRoundNearest(const Vector3 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round toward zero (inverted inplace div: this = aOther /
    /// this)
    /// </summary>
    DEVICE_CODE Vector3 &DivInvRoundZero(const Vector3 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise floor() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector3 &DivInvFloor(const Vector3 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise ceil() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector3 &DivInvCeil(const Vector3 &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3 DivRound(const Vector3 &aLeft, const Vector3 &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3 DivRoundNearest(const Vector3 &aLeft, const Vector3 &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3 DivRoundZero(const Vector3 &aLeft, const Vector3 &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3 DivFloor(const Vector3 &aLeft, const Vector3 &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3 DivCeil(const Vector3 &aLeft, const Vector3 &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round() (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector3 &DivScaleRound(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round nearest ties to even (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector3 &DivScaleRoundNearest(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round toward zero (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector3 &DivScaleRoundZero(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise floor (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector3 &DivScaleFloor(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise ceil() (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector3 &DivScaleCeil(T aScale)
        requires RealIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round()
    /// </summary>
    DEVICE_CODE Vector3 &DivRound(const Vector3 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector3 &DivRoundNearest(const Vector3 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector3 &DivRoundZero(const Vector3 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise floor()
    /// </summary>
    DEVICE_CODE Vector3 &DivFloor(const Vector3 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE Vector3 &DivCeil(const Vector3 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector3 &DivInvRound(const Vector3 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round nearest ties to even (inverted inplace div: this =
    /// aOther / this)
    /// </summary>
    DEVICE_CODE Vector3 &DivInvRoundNearest(const Vector3 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise round toward zero (inverted inplace div: this = aOther /
    /// this)
    /// </summary>
    DEVICE_CODE Vector3 &DivInvRoundZero(const Vector3 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise floor() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector3 &DivInvFloor(const Vector3 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace Integer division with element wise ceil() (inverted inplace div: this = aOther / this)
    /// </summary>
    DEVICE_CODE Vector3 &DivInvCeil(const Vector3 &aOther)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3 DivRound(const Vector3 &aLeft, const Vector3 &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3 DivRoundNearest(const Vector3 &aLeft, const Vector3 &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3 DivRoundZero(const Vector3 &aLeft, const Vector3 &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3 DivFloor(const Vector3 &aLeft, const Vector3 &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Integer division with element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3 DivCeil(const Vector3 &aLeft, const Vector3 &aRight)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round() (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector3 &DivScaleRound(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round nearest ties to even (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector3 &DivScaleRoundNearest(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise round toward zero (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector3 &DivScaleRoundZero(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise floor (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector3 &DivScaleFloor(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// Inplace integer division with element wise ceil() (for scaling operations)
    /// </summary>
    DEVICE_CODE Vector3 &DivScaleCeil(complex_basetype_t<T> aScale)
        requires ComplexIntegral<T>;

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T &operator[](Axis3D aAxis) const
        requires DeviceCode<T>;

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] const T &operator[](Axis3D aAxis) const
        requires HostCode<T>;

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    DEVICE_CODE [[nodiscard]] T &operator[](Axis3D aAxis)
        requires DeviceCode<T>;

    /// <summary>
    /// returns the element corresponding to the given axis
    /// </summary>
    [[nodiscard]] T &operator[](Axis3D aAxis)
        requires HostCode<T>;
#pragma endregion

#pragma region Convert Methods
    /// <summary>
    /// Type conversion without saturation, direct type conversion
    /// </summary>
    template <Number T2> [[nodiscard]] static Vector3<T> DEVICE_CODE Convert(const Vector3<T2> &aVec);
#pragma endregion

#pragma region Integral only Methods
    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE Vector3<T> &LShift(const Vector3<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> LShift(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE Vector3<T> &RShift(const Vector3<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> RShift(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE Vector3<T> &LShift(uint aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise left shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> LShift(const Vector3<T> &aLeft, uint aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE Vector3<T> &RShift(uint aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise right shift
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> RShift(const Vector3<T> &aLeft, uint aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE Vector3<T> &And(const Vector3<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise And
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> And(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE Vector3<T> &Or(const Vector3<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise Or
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Or(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE Vector3<T> &Xor(const Vector3<T> &aOther)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise Xor
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Xor(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE Vector3<T> &Not()
        requires RealIntegral<T>;

    /// <summary>
    /// Element wise bitwise negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Not(const Vector3<T> &aVec)
        requires RealIntegral<T>;
#pragma endregion

#pragma region Methods
#pragma region Exp
    /// <summary>
    /// Element wise exponential
    /// </summary>
    Vector3<T> &Exp()
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE Vector3<T> &Exp()
        requires NonNativeNumber<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE Vector3<T> &Exp()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Exp(const Vector3<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Exp(const Vector3<T> &aVec)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise exponential
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Exp(const Vector3<T> &aVec)
        requires NonNativeNumber<T>;
#pragma endregion

#pragma region Log
    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    Vector3<T> &Ln()
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE Vector3<T> &Ln()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE Vector3<T> &Ln()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Ln(const Vector3<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Ln(const Vector3<T> &aVec)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise natural logarithm
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Ln(const Vector3<T> &aVec)
        requires NonNativeFloatingPoint<T>;
#pragma endregion

#pragma region Sqr
    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE Vector3<T> &Sqr();

    /// <summary>
    /// Element wise square
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Sqr(const Vector3<T> &aVec);
#pragma endregion

#pragma region Sqrt
    /// <summary>
    /// Element wise square root
    /// </summary>
    Vector3<T> &Sqrt()
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE Vector3<T> &Sqrt()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE Vector3<T> &Sqrt()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Sqrt(const Vector3<T> &aVec)
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Sqrt(const Vector3<T> &aVec)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise square root
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Sqrt(const Vector3<T> &aVec)
        requires NonNativeFloatingPoint<T>;
#pragma endregion

#pragma region Abs
    /// <summary>
    /// Element wise absolute
    /// </summary>
    Vector3<T> &Abs()
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE Vector3<T> &Abs()
        requires RealSignedNumber<T> && NonNativeNumber<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE Vector3<T> &Abs()
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Abs(const Vector3<T> &aVec)
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Abs(const Vector3<T> &aVec)
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Abs(const Vector3<T> &aVec)
        requires RealSignedNumber<T> && NonNativeNumber<T>;
#pragma endregion

#pragma region AbsDiff
    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector3<T> &AbsDiff(const Vector3<T> &aOther)
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> AbsDiff(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector3<T> &AbsDiff(const Vector3<T> &aOther)
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> AbsDiff(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE Vector3<T> &AbsDiff(const Vector3<T> &aOther)
        requires RealSignedNumber<T> && NonNativeNumber<T>;

    /// <summary>
    /// Element wise absolute difference
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> AbsDiff(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealSignedNumber<T> && NonNativeNumber<T>;
#pragma endregion

#pragma region Methods for Complex types
    /// <summary>
    /// Conjugate complex per element
    /// </summary>
    DEVICE_CODE Vector3<T> &Conj()
        requires ComplexNumber<T>;

    /// <summary>
    /// Conjugate complex per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Conj(const Vector3<T> &aValue)
        requires ComplexNumber<T>;

    /// <summary>
    /// Conjugate complex multiplication: this * conj(aOther)  per element
    /// </summary>
    DEVICE_CODE Vector3<T> &ConjMul(const Vector3<T> &aOther)
        requires ComplexNumber<T>;

    /// <summary>
    /// Conjugate complex multiplication: aLeft * conj(aRight) per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> ConjMul(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires ComplexNumber<T>;

    /// <summary>
    /// Complex magnitude per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<complex_basetype_t<T>> Magnitude() const
        requires ComplexFloatingPoint<T>;

    /// <summary>
    /// Complex magnitude squared per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<complex_basetype_t<T>> MagnitudeSqr() const
        requires ComplexFloatingPoint<T>;

    /// <summary>
    /// Angle between real and imaginary of a complex number (atan2(image, real)) per element
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector3<complex_basetype_t<T>> Angle() const
        requires ComplexFloatingPoint<T>;
#pragma endregion

#pragma region Clamp
    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector3<T> &Clamp(T aMinVal, T aMaxVal)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    Vector3<T> &Clamp(T aMinVal, T aMaxVal)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector3<T> &Clamp(T aMinVal, T aMaxVal)
        requires NonNativeNumber<T> && (!ComplexNumber<T>);

    /// <summary>
    /// Component wise clamp to value range
    /// </summary>
    DEVICE_CODE Vector3<T> &Clamp(complex_basetype_t<T> aMinVal, complex_basetype_t<T> aMaxVal)
        requires ComplexNumber<T>;

    /// <summary>
    /// Component wise clamp to maximum value range of given target type
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Vector3<T> &ClampToTargetType() noexcept
        requires(need_saturation_clamp_v<T, TTarget>);

    /// <summary>
    /// Component wise clamp to maximum value range of given target type<para/>
    /// NOP in case no saturation clamping is needed.
    /// </summary>
    template <Number TTarget>
    DEVICE_CODE Vector3<T> &ClampToTargetType() noexcept
        requires(!need_saturation_clamp_v<T, TTarget>);
#pragma endregion

#pragma region Min
    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE Vector3<T> &Min(const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum
    /// </summary>
    Vector3<T> &Min(const Vector3<T> &aRight)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE Vector3<T> &Min(const Vector3<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>);

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Min(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum
    /// </summary>
    [[nodiscard]] static Vector3<T> Min(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise minimum
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static Vector3<T> Min(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>);

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Min() const
        requires NonNativeNumber<T> && (!ComplexNumber<T>);

    /// <summary>
    /// Returns the minimum component of the vector
    /// </summary>
    [[nodiscard]] T Min() const
        requires HostCode<T> && NativeNumber<T>;
#pragma endregion

#pragma region Max
    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE Vector3<T> &Max(const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise maximum
    /// </summary>
    Vector3<T> &Max(const Vector3<T> &aRight)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE Vector3<T> &Max(const Vector3<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>);

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Max(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise maximum
    /// </summary>
    [[nodiscard]] static Vector3<T> Max(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires HostCode<T> && NativeNumber<T>;

    /// <summary>
    /// Component wise maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Max(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires NonNativeNumber<T> && (!ComplexNumber<T>);

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires DeviceCode<T> && NativeNumber<T>;

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    DEVICE_CODE [[nodiscard]] T Max() const
        requires NonNativeNumber<T> && (!ComplexNumber<T>);

    /// <summary>
    /// Returns the maximum component of the vector
    /// </summary>
    [[nodiscard]] T Max() const
        requires HostCode<T> && NativeNumber<T>;
#pragma endregion

#pragma region Round
    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Round(const Vector3<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE Vector3<T> &Round()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round()
    /// </summary>
    DEVICE_CODE Vector3<T> &Round()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round()
    /// </summary>
    Vector3<T> &Round()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Floor(const Vector3<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_CODE Vector3<T> &Floor()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    Vector3<T> &Floor()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise floor()
    /// </summary>
    DEVICE_ONLY_CODE Vector3<T> &Floor()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> Ceil(const Vector3<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_CODE Vector3<T> &Ceil()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    Vector3<T> &Ceil()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise ceil()
    /// </summary>
    DEVICE_ONLY_CODE Vector3<T> &Ceil()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> RoundNearest(const Vector3<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_CODE Vector3<T> &RoundNearest()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even<para/>
    /// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    Vector3<T> &RoundNearest()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round nearest ties to even
    /// </summary>
    DEVICE_ONLY_CODE Vector3<T> &RoundNearest()
        requires NonNativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<T> RoundZero(const Vector3<T> &aValue)
        requires RealOrComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_CODE Vector3<T> &RoundZero()
        requires DeviceCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    Vector3<T> &RoundZero()
        requires HostCode<T> && NativeFloatingPoint<T>;

    /// <summary>
    /// Element wise round toward zero
    /// </summary>
    DEVICE_ONLY_CODE Vector3<T> &RoundZero()
        requires NonNativeFloatingPoint<T>;
#pragma endregion

#pragma region Compare per element
    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    [[nodiscard]] static Vector3<byte> CompareEQEps(const Vector3<T> &aLeft, const Vector3<T> &aRight, T aEpsilon)
        requires NativeFloatingPoint<T> && HostCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareEQEps(const Vector3<T> &aLeft, const Vector3<T> &aRight,
                                                                T aEpsilon)
        requires NativeFloatingPoint<T> && DeviceCode<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareEQEps(const Vector3<T> &aLeft, const Vector3<T> &aRight,
                                                                T aEpsilon)
        requires Is16BitFloat<T>;

    /// <summary>
    /// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
    /// the element, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareEQEps(const Vector3<T> &aLeft, const Vector3<T> &aRight,
                                                                complex_basetype_t<T> aEpsilon)
        requires ComplexFloatingPoint<T>;

    /// <summary>
    /// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareEQ(const Vector3<T> &aLeft, const Vector3<T> &aRight);

    /// <summary>
    /// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareGE(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealNumber<T>;

    /// <summary>
    /// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareGT(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealNumber<T>;

    /// <summary>
    /// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareLE(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealNumber<T>;

    /// <summary>
    /// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareLT(const Vector3<T> &aLeft, const Vector3<T> &aRight)
        requires RealNumber<T>;

    /// <summary>
    /// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false
    /// </summary>
    DEVICE_CODE [[nodiscard]] static Vector3<byte> CompareNEQ(const Vector3<T> &aLeft, const Vector3<T> &aRight);
#pragma endregion

#pragma region Data accessors
    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<T> XY() const;

    /// <summary>
    /// return sub-vector elements
    /// </summary>
    DEVICE_CODE [[nodiscard]] Vector2<T> YZ() const;

    /// <summary>
    /// Provide a smiliar accessor to inner data as for std container
    /// </summary>
    DEVICE_CODE [[nodiscard]] T *data();

    /// <summary>
    /// Provide a smiliar accessor to inner data as for std container
    /// </summary>
    DEVICE_CODE [[nodiscard]] const T *data() const;
#pragma endregion
#pragma endregion
};

template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator+(const Vector3<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft.x + aRight), static_cast<T>(aLeft.y + aRight),
                      static_cast<T>(aLeft.z + aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator+(T2 aLeft, const Vector3<T> &aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft + aRight.x), static_cast<T>(aLeft + aRight.y),
                      static_cast<T>(aLeft + aRight.z)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator-(const Vector3<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft.x - aRight), static_cast<T>(aLeft.y - aRight),
                      static_cast<T>(aLeft.z - aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator-(T2 aLeft, const Vector3<T> &aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft - aRight.x), static_cast<T>(aLeft - aRight.y),
                      static_cast<T>(aLeft - aRight.z)};
}

template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator*(const Vector3<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft.x * aRight), static_cast<T>(aLeft.y * aRight),
                      static_cast<T>(aLeft.z * aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator*(T2 aLeft, const Vector3<T> &aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft * aRight.x), static_cast<T>(aLeft * aRight.y),
                      static_cast<T>(aLeft * aRight.z)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator/(const Vector3<T> &aLeft, T2 aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft.x / aRight), static_cast<T>(aLeft.y / aRight),
                      static_cast<T>(aLeft.z / aRight)};
}
template <typename T, typename T2>
DEVICE_CODE Vector3<T> operator/(T2 aLeft, const Vector3<T> &aRight)
    requires Number<T2>
{
    return Vector3<T>{static_cast<T>(aLeft / aRight.x), static_cast<T>(aLeft / aRight.y),
                      static_cast<T>(aLeft / aRight.z)};
}

template <HostCode T2> MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<T2> &aVec);

template <HostCode T2> MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<T2> &aVec);

// byte and sbyte are treated as characters and not numbers:
template <HostCode T2>
MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2>
MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2> MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<T2> &aVec);

template <HostCode T2> MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<T2> &aVec);

template <HostCode T2>
MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<T2> &aVec)
    requires ByteSizeType<T2>;

template <HostCode T2>
MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<T2> &aVec)
    requires ByteSizeType<T2>;

#ifdef IS_HOST_COMPILER
extern template struct Vector3<sbyte>;
extern template struct Vector3<byte>;
extern template struct Vector3<short>;
extern template struct Vector3<ushort>;
extern template struct Vector3<int>;
extern template struct Vector3<uint>;
extern template struct Vector3<long64>;
extern template struct Vector3<ulong64>;

extern template struct Vector3<BFloat16>;
extern template struct Vector3<HalfFp16>;
extern template struct Vector3<float>;
extern template struct Vector3<double>;

extern template struct Vector3<Complex<sbyte>>;
extern template struct Vector3<Complex<short>>;
extern template struct Vector3<Complex<int>>;
extern template struct Vector3<Complex<long64>>;
extern template struct Vector3<Complex<BFloat16>>;
extern template struct Vector3<Complex<HalfFp16>>;
extern template struct Vector3<Complex<float>>;
extern template struct Vector3<Complex<double>>;

extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(const Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(const Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(const Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(const Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(const Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(const Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(const Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(const Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(const Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(const Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(const Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(const Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(const Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(const Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(const Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(const Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(const Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(const Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(const Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(const Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(const Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(const Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<short>::Vector3(const Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(const Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(const Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(const Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(const Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(const Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(const Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(const Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(const Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(const Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(const Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(const Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(const Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(const Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(const Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(const Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(const Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(const Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(const Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(const Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(const Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(const Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<int>::Vector3(const Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(const Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(const Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(const Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(const Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(const Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(const Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(const Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(const Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(const Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(const Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(const Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(const Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(const Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(const Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(const Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(const Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(const Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(const Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(const Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(const Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(const Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(const Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(const Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(const Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(const Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(const Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(const Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(const Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(const Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(const Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(const Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(const Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(const Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(const Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(const Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(const Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(const Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(const Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(const Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(const Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(const Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(const Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(const Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(const Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(const Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(const Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(const Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(const Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(const Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(const Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(const Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(const Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(const Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(const Vector3<double> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(const Vector3<float> &, RoundingMode);

extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(const Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(const Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(const Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(const Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(const Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(const Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(const Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(const Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(const Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(const Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(const Vector3<double> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(const Vector3<float> &, RoundingMode);

extern template MPPEXPORT_COMMON Vector3<float>::Vector3(const Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(const Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(const Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(const Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(const Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(const Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(const Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(const Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(const Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(const Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(const Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<double>::Vector3(const Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(const Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(const Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(const Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(const Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(const Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(const Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(const Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(const Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(const Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(const Vector3<float> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(const Vector3<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(const Vector3<sbyte> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(const Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(const Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(const Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(const Vector3<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(const Vector3<short> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(const Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(const Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(const Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(const Vector3<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(const Vector3<int> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(const Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(const Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(const Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(const Vector3<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(const Vector3<long64> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(const Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(const Vector3<Complex<float>> &, RoundingMode);

extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(const Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(const Vector3<Complex<float>> &, RoundingMode);

extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(const Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(const Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(const Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(const Vector3<Complex<double>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(const Vector3<float> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(const Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(const Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(const Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(const Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(const Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(const Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(const Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(const Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte>::Vector3(Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<byte>::Vector3(Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<short>::Vector3(Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<short>::Vector3(Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort>::Vector3(Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<int>::Vector3(Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<int>::Vector3(Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<uint>::Vector3(Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<long64>::Vector3(Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64>::Vector3(Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16>::Vector3(Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(Vector3<float> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16>::Vector3(Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<float>::Vector3(Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<float>::Vector3(Vector3<double> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<double>::Vector3(Vector3<sbyte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(Vector3<byte> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(Vector3<short> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(Vector3<ushort> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(Vector3<int> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(Vector3<uint> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(Vector3<long64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(Vector3<ulong64> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(Vector3<BFloat16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(Vector3<HalfFp16> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<double>::Vector3(Vector3<float> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>>::Vector3(Vector3<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>>::Vector3(Vector3<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>>::Vector3(Vector3<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>>::Vector3(Vector3<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>>::Vector3(Vector3<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<float>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>>::Vector3(Vector3<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>>::Vector3(Vector3<Complex<double>> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(Vector3<Complex<sbyte>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(Vector3<Complex<short>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(Vector3<Complex<int>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(Vector3<Complex<long64>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(Vector3<Complex<BFloat16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(Vector3<Complex<HalfFp16>> &) noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>>::Vector3(Vector3<Complex<float>> &) noexcept;

extern template MPPEXPORT_COMMON Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<sbyte> &Vector3<sbyte>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<byte> &Vector3<byte>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<byte> &Vector3<byte>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<byte> &Vector3<byte>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<byte> &Vector3<byte>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<byte> &Vector3<byte>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<byte> &Vector3<byte>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<byte> &Vector3<byte>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<byte> &Vector3<byte>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<byte> &Vector3<byte>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<byte> &Vector3<byte>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<byte> &Vector3<byte>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<byte> &Vector3<byte>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<short> &Vector3<short>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<short> &Vector3<short>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<short> &Vector3<short>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<short> &Vector3<short>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<short> &Vector3<short>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<short> &Vector3<short>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<short> &Vector3<short>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<short> &Vector3<short>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<short> &Vector3<short>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<short> &Vector3<short>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<short> &Vector3<short>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<short> &Vector3<short>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<ushort> &Vector3<ushort>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort> &Vector3<ushort>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort> &Vector3<ushort>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort> &Vector3<ushort>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort> &Vector3<ushort>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort> &Vector3<ushort>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort> &Vector3<ushort>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort> &Vector3<ushort>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort> &Vector3<ushort>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort> &Vector3<ushort>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort> &Vector3<ushort>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ushort> &Vector3<ushort>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<int> &Vector3<int>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<int> &Vector3<int>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<int> &Vector3<int>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<int> &Vector3<int>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<int> &Vector3<int>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<int> &Vector3<int>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<int> &Vector3<int>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<int> &Vector3<int>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<int> &Vector3<int>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<int> &Vector3<int>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<int> &Vector3<int>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<int> &Vector3<int>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<uint> &Vector3<uint>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<uint> &Vector3<uint>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<uint> &Vector3<uint>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<uint> &Vector3<uint>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<uint> &Vector3<uint>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<uint> &Vector3<uint>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<uint> &Vector3<uint>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<uint> &Vector3<uint>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<uint> &Vector3<uint>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<uint> &Vector3<uint>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<uint> &Vector3<uint>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<uint> &Vector3<uint>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<long64> &Vector3<long64>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<long64> &Vector3<long64>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<long64> &Vector3<long64>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<long64> &Vector3<long64>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<long64> &Vector3<long64>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<long64> &Vector3<long64>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<long64> &Vector3<long64>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<long64> &Vector3<long64>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<long64> &Vector3<long64>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<long64> &Vector3<long64>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<long64> &Vector3<long64>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<long64> &Vector3<long64>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<ulong64> &Vector3<ulong64>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<BFloat16> &Vector3<BFloat16>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<HalfFp16> &Vector3<HalfFp16>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<float> &Vector3<float>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<float> &Vector3<float>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<float> &Vector3<float>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<float> &Vector3<float>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<float> &Vector3<float>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<float> &Vector3<float>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<float> &Vector3<float>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<float> &Vector3<float>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<float> &Vector3<float>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<float> &Vector3<float>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<float> &Vector3<float>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<float> &Vector3<float>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<double> &Vector3<double>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<double> &Vector3<double>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<double> &Vector3<double>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<double> &Vector3<double>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<double> &Vector3<double>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<double> &Vector3<double>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<double> &Vector3<double>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<double> &Vector3<double>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<double> &Vector3<double>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<double> &Vector3<double>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<double> &Vector3<double>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<double> &Vector3<double>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<sbyte>> &Vector3<Complex<sbyte>>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<short>> &Vector3<Complex<short>>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<int>> &Vector3<Complex<int>>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<long64>> &Vector3<Complex<long64>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<
    byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<
    int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<
    uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<BFloat16>> &Vector3<Complex<BFloat16>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<
    byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<
    int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<
    uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<HalfFp16>> &Vector3<Complex<HalfFp16>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<float>> &Vector3<Complex<float>>::ClampToTargetType<double>() noexcept;

extern template MPPEXPORT_COMMON Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<
    sbyte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<byte>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<
    short>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<
    ushort>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<int>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<uint>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<
    long64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<
    ulong64>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<
    BFloat16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<
    HalfFp16>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<
    float>() noexcept;
extern template MPPEXPORT_COMMON Vector3<Complex<double>> &Vector3<Complex<double>>::ClampToTargetType<
    double>() noexcept;

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<sbyte> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<sbyte> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<sbyte> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<sbyte> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<byte> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<byte> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<byte> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<byte> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<short> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<short> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<short> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<short> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<ushort> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<ushort> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<ushort> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<ushort> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<int> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<int> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<int> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<int> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<uint> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<uint> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<uint> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<uint> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<long64> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<long64> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<long64> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<long64> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<ulong64> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<ulong64> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<ulong64> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<ulong64> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<BFloat16> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<BFloat16> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<BFloat16> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<BFloat16> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<HalfFp16> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<HalfFp16> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<HalfFp16> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<HalfFp16> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<float> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<float> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<float> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<float> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<double> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<double> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<double> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<double> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<sbyte>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<sbyte>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<Complex<sbyte>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<sbyte>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<short>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<short>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<Complex<short>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<short>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<int>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<int>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<Complex<int>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<int>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<long64>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<long64>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<Complex<long64>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<long64>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<BFloat16>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<BFloat16>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<Complex<BFloat16>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<BFloat16>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<HalfFp16>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<HalfFp16>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<Complex<HalfFp16>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<HalfFp16>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<float>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<float>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<Complex<float>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<float>> &aVec);

extern template MPPEXPORT_COMMON std::ostream &operator<<(std::ostream &aOs, const Vector3<Complex<double>> &aVec);
extern template MPPEXPORT_COMMON std::wostream &operator<<(std::wostream &aOs, const Vector3<Complex<double>> &aVec);
extern template MPPEXPORT_COMMON std::istream &operator>>(std::istream &aIs, Vector3<Complex<double>> &aVec);
extern template MPPEXPORT_COMMON std::wistream &operator>>(std::wistream &aIs, Vector3<Complex<double>> &aVec);
#endif // IS_HOST_COMPILER

} // namespace mpp