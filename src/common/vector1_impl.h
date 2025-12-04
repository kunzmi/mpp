#pragma once
#include "defines.h"
#include "exception.h"
#include "mpp_defs.h"
#include "needSaturationClamp.h"
#include "numberTypes.h"
#include "numeric_limits.h"
#include "safeCast.h"
#include "staticCast.h"
#include "vector_typetraits.h"
#include "vector1.h"
#include <cmath>
#include <common/utilities.h>
#include <concepts>
#include <iostream>
#include <type_traits>

namespace mpp
{

#pragma region Constructors

/// <summary>
/// Type conversion with saturation if needed<para/>
/// E.g.: when converting int to byte, values are clamped to 0..255<para/>
/// But when converting byte to int, no clamping operation is performed.
/// </summary>
template <Number T> template <Number T2> DEVICE_CODE Vector1<T>::Vector1(const Vector1<T2> &aVec) noexcept
{
    if constexpr (need_saturation_clamp_v<T2, T>)
    {
        Vector1<T2> temp(aVec);
        temp.template ClampToTargetType<T>();
        x = StaticCast<T2, T>(temp.x);
    }
    else
    {
        x = StaticCast<T2, T>(aVec.x);
    }
}

/// <summary>
/// Type conversion with saturation if needed<para/>
/// E.g.: when converting int to byte, values are clamped to 0..255<para/>
/// But when converting byte to int, no clamping operation is performed.<para/>
/// If we can modify the input variable, no need to allocate temporary storage for clamping.
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector1<T>::Vector1(Vector1<T2> &aVec) noexcept
    requires(!std::same_as<T, T2>)
{
    if constexpr (need_saturation_clamp_v<T2, T>)
    {
        aVec.template ClampToTargetType<T>();
    }
    x = StaticCast<T2, T>(aVec.x);
}

/// <summary>
/// Type conversion using CUDA intrinsics for float to BFloat/Halffloat and complex
/// </summary>
template <Number T>
template <Number T2>
DEVICE_CODE Vector1<T>::Vector1(const Vector1<T2> &aVec, RoundingMode aRoundingMode)
    requires((IsBFloat16<T> || IsHalfFp16<T>) && IsFloat<T2>) ||
            (ComplexFloatingPoint<T> && ComplexFloatingPoint<T2> &&
             NonNativeFloatingPoint<complex_basetype_t<remove_vector_t<T>>> &&
             std::same_as<float, complex_basetype_t<remove_vector_t<T2>>>)
    : x(T(aVec.x, aRoundingMode))
{
}
#pragma endregion

#pragma region Operators
// don't use space-ship operator as it returns true if any comparison returns true.
// But NPP only returns true if all channels fulfill the comparison.
// auto operator<=>(const Vector1 &) const = default;

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector1<T>::EqEps(const Vector1 &aLeft, const Vector1 &aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && HostCode<T>
{
    Vector1<T> left  = aLeft;
    Vector1<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);

    return std::abs(left.x - right.x) <= aEpsilon;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector1<T>::EqEps(const Vector1 &aLeft, const Vector1 &aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && DeviceCode<T>
{
    Vector1<T> left  = aLeft;
    Vector1<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);

    return abs(left.x - right.x) <= aEpsilon;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector1<T>::EqEps(const Vector1 &aLeft, const Vector1 &aRight, T aEpsilon)
    requires Is16BitFloat<T>
{
    Vector1<T> left  = aLeft;
    Vector1<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);

    return T::Abs(left.x - right.x) <= aEpsilon;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector1<T>::EqEps(const Vector1 &aLeft, const Vector1 &aRight, complex_basetype_t<T> aEpsilon)
    requires ComplexFloatingPoint<T>
{
    return T::EqEps(aLeft.x, aRight.x, aEpsilon);
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector1<T>::operator<(const Vector1 &aOther) const
    requires RealNumber<T>
{
    return x < aOther.x;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector1<T>::operator<=(const Vector1 &aOther) const
    requires RealNumber<T>
{
    return x <= aOther.x;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector1<T>::operator>(const Vector1 &aOther) const
    requires RealNumber<T>
{
    return x > aOther.x;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T>
DEVICE_CODE bool Vector1<T>::operator>=(const Vector1 &aOther) const
    requires RealNumber<T>
{
    return x >= aOther.x;
}

/// <summary>
/// Returns true if each element comparison is true
/// </summary>
template <Number T> DEVICE_CODE bool Vector1<T>::operator==(const Vector1 &aOther) const
{
    return x == aOther.x;
}

/// <summary>
/// Returns true if any element comparison is true
/// </summary>
template <Number T> DEVICE_CODE bool Vector1<T>::operator!=(const Vector1 &aOther) const
{
    return x != aOther.x;
}

/// <summary>
/// Negation
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::operator-() const
    requires RealSignedNumber<T> || ComplexNumber<T>
{
    return Vector1<T>(-x);
}

/// <summary>
/// Component wise addition
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> &Vector1<T>::operator+=(T aOther)
{
    x += aOther;
    return *this;
}

/// <summary>
/// Component wise addition
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::operator+=(complex_basetype_t<T> aOther)
    requires ComplexNumber<T>
{
    x += aOther;
    return *this;
}

/// <summary>
/// Component wise addition
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> &Vector1<T>::operator+=(const Vector1 &aOther)
{
    x += aOther.x;
    return *this;
}

/// <summary>
/// Component wise addition
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> Vector1<T>::operator+(const Vector1 &aOther) const
{
    return Vector1<T>{T(x + aOther.x)};
}

/// <summary>
/// Component wise subtraction
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> &Vector1<T>::operator-=(T aOther)
{
    x -= aOther;
    return *this;
}

/// <summary>
/// Component wise subtraction
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::operator-=(complex_basetype_t<T> aOther)
    requires ComplexNumber<T>
{
    x -= aOther;
    return *this;
}

/// <summary>
/// Component wise subtraction
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> &Vector1<T>::operator-=(const Vector1 &aOther)
{
    x -= aOther.x;
    return *this;
}

/// <summary>
/// Component wise subtraction (inverted inplace sub: this = aOther - this)
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> &Vector1<T>::SubInv(const Vector1 &aOther)
{
    x = aOther.x - x;
    return *this;
}

/// <summary>
/// Component wise subtraction
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> Vector1<T>::operator-(const Vector1 &aOther) const
{
    return Vector1<T>{T(x - aOther.x)};
}

/// <summary>
/// Component wise multiplication
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> &Vector1<T>::operator*=(T aOther)
{
    x *= aOther;
    return *this;
}

/// <summary>
/// Component wise multiplication
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::operator*=(complex_basetype_t<T> aOther)
    requires ComplexNumber<T>
{
    x *= aOther;
    return *this;
}

/// <summary>
/// Component wise multiplication
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> &Vector1<T>::operator*=(const Vector1 &aOther)
{
    x *= aOther.x;
    return *this;
}

/// <summary>
/// Component wise multiplication
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> Vector1<T>::operator*(const Vector1 &aOther) const
{
    return Vector1<T>{T(x * aOther.x)};
}

/// <summary>
/// Component wise division
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> &Vector1<T>::operator/=(T aOther)
{
    x /= aOther;
    return *this;
}

/// <summary>
/// Component wise division
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::operator/=(complex_basetype_t<T> aOther)
    requires ComplexNumber<T>
{
    x /= aOther;
    return *this;
}

/// <summary>
/// Component wise division
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> &Vector1<T>::operator/=(const Vector1 &aOther)
{
    x /= aOther.x;
    return *this;
}

/// <summary>
/// Component wise division (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> &Vector1<T>::DivInv(const Vector1 &aOther)
{
    x = aOther.x / x;
    return *this;
}

/// <summary>
/// Component wise division
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> Vector1<T>::operator/(const Vector1 &aOther) const
{
    return Vector1<T>{T(x / aOther.x)};
}

/// <summary>
/// Inplace integer division with element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivRound(const Vector1 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTiesAwayFromZero(x, aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivRoundNearest(const Vector1 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundNearestEven(x, aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivRoundZero(const Vector1 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardZero(x, aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivFloor(const Vector1 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardNegInf(x, aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivCeil(const Vector1 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardPosInf(x, aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivInvRound(const Vector1 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTiesAwayFromZero(aOther.x, x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even (inverted inplace div: this =
/// aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivInvRoundNearest(const Vector1 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundNearestEven(aOther.x, x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero (inverted inplace div: this = aOther /
/// this)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivInvRoundZero(const Vector1 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardZero(aOther.x, x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivInvFloor(const Vector1 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardNegInf(aOther.x, x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivInvCeil(const Vector1 &aOther)
    requires RealIntegral<T>
{
    x = DivRoundTowardPosInf(aOther.x, x);
    return *this;
}

/// <summary>
/// Integer division with element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::DivRound(const Vector1 &aLeft, const Vector1 &aRight)
    requires RealIntegral<T>
{
    return Vector1<T>{DivRoundTiesAwayFromZero(aLeft.x, aRight.x)};
}

/// <summary>
/// Integer division with element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::DivRoundNearest(const Vector1 &aLeft, const Vector1 &aRight)
    requires RealIntegral<T>
{
    return Vector1<T>{DivRoundNearestEven(aLeft.x, aRight.x)};
}

/// <summary>
/// Integer division with element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::DivRoundZero(const Vector1 &aLeft, const Vector1 &aRight)
    requires RealIntegral<T>
{
    return Vector1<T>{DivRoundTowardZero(aLeft.x, aRight.x)};
}

/// <summary>
/// Integer division with element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::DivFloor(const Vector1 &aLeft, const Vector1 &aRight)
    requires RealIntegral<T>
{
    return Vector1<T>{DivRoundTowardNegInf(aLeft.x, aRight.x)};
}

/// <summary>
/// Integer division with element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::DivCeil(const Vector1 &aLeft, const Vector1 &aRight)
    requires RealIntegral<T>
{
    return Vector1<T>{DivRoundTowardPosInf(aLeft.x, aRight.x)};
}

/// <summary>
/// Inplace integer division with element wise round() (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivScaleRound(T aScale)
    requires RealIntegral<T>
{
    x = DivScaleRoundTiesAwayFromZero(x, aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivScaleRoundNearest(T aScale)
    requires RealIntegral<T>
{
    x = DivScaleRoundNearestEven(x, aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivScaleRoundZero(T aScale)
    requires RealIntegral<T>
{
    x = DivScaleRoundTowardZero(x, aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivScaleFloor(T aScale)
    requires RealIntegral<T>
{
    x = DivScaleRoundTowardNegInf(x, aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil() (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivScaleCeil(T aScale)
    requires RealIntegral<T>
{
    x = DivScaleRoundTowardPosInf(x, aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivRound(const Vector1 &aOther)
    requires ComplexIntegral<T>
{
    x.DivRound(aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivRoundNearest(const Vector1 &aOther)
    requires ComplexIntegral<T>
{
    x.DivRoundNearest(aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivRoundZero(const Vector1 &aOther)
    requires ComplexIntegral<T>
{
    x.DivRoundZero(aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivFloor(const Vector1 &aOther)
    requires ComplexIntegral<T>
{
    x.DivFloor(aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivCeil(const Vector1 &aOther)
    requires ComplexIntegral<T>
{
    x.DivCeil(aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivInvRound(const Vector1 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvRound(aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even (inverted inplace div: this =
/// aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivInvRoundNearest(const Vector1 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvRoundNearest(aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero (inverted inplace div: this = aOther /
/// this)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivInvRoundZero(const Vector1 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvRoundZero(aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivInvFloor(const Vector1 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvFloor(aOther.x);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil() (inverted inplace div: this = aOther / this)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivInvCeil(const Vector1 &aOther)
    requires ComplexIntegral<T>
{
    x.DivInvCeil(aOther.x);
    return *this;
}

/// <summary>
/// Integer division with element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::DivRound(const Vector1 &aLeft, const Vector1 &aRight)
    requires ComplexIntegral<T>
{
    return Vector1<T>{T::DivRound(aLeft.x, aRight.x)};
}

/// <summary>
/// Integer division with element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::DivRoundNearest(const Vector1 &aLeft, const Vector1 &aRight)
    requires ComplexIntegral<T>
{
    return Vector1<T>{T::DivRoundNearest(aLeft.x, aRight.x)};
}

/// <summary>
/// Integer division with element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::DivRoundZero(const Vector1 &aLeft, const Vector1 &aRight)
    requires ComplexIntegral<T>
{
    return Vector1<T>{T::DivRoundZero(aLeft.x, aRight.x)};
}

/// <summary>
/// Integer division with element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::DivFloor(const Vector1 &aLeft, const Vector1 &aRight)
    requires ComplexIntegral<T>
{
    return Vector1<T>{T::DivFloor(aLeft.x, aRight.x)};
}

/// <summary>
/// Integer division with element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::DivCeil(const Vector1 &aLeft, const Vector1 &aRight)
    requires ComplexIntegral<T>
{
    return Vector1<T>{T::DivCeil(aLeft.x, aRight.x)};
}

/// <summary>
/// Inplace integer division with element wise round() (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivScaleRound(complex_basetype_t<T> aScale)
    requires ComplexIntegral<T>
{
    x.DivScaleRound(aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round nearest ties to even (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivScaleRoundNearest(complex_basetype_t<T> aScale)
    requires ComplexIntegral<T>
{
    x.DivScaleRoundNearest(aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise round toward zero (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivScaleRoundZero(complex_basetype_t<T> aScale)
    requires ComplexIntegral<T>
{
    x.DivScaleRoundZero(aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise floor (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivScaleFloor(complex_basetype_t<T> aScale)
    requires ComplexIntegral<T>
{
    x.DivScaleFloor(aScale);
    return *this;
}

/// <summary>
/// Inplace integer division with element wise ceil() (for scaling operations)
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::DivScaleCeil(complex_basetype_t<T> aScale)
    requires ComplexIntegral<T>
{
    x.DivScaleCeil(aScale);
    return *this;
}

/// <summary>
/// returns the element corresponding to the given axis
/// </summary>
template <Number T>
DEVICE_CODE const T &Vector1<T>::operator[](Axis1D aAxis) const
    requires DeviceCode<T>
{
    switch (aAxis)
    {
        case Axis1D::X:
            return x;
    }
    return x;
}

/// <summary>
/// returns the element corresponding to the given axis
/// </summary>
template <Number T>
const T &Vector1<T>::operator[](Axis1D aAxis) const
    requires HostCode<T>
{
    switch (aAxis)
    {
        case Axis1D::X:
            return x;
    }

    throw INVALIDARGUMENT(aAxis, aAxis);
}

/// <summary>
/// returns the element corresponding to the given axis
/// </summary>
template <Number T>
DEVICE_CODE T &Vector1<T>::operator[](Axis1D aAxis)
    requires DeviceCode<T>
{
    switch (aAxis)
    {
        case Axis1D::X:
            return x;
    }
    return x;
}

/// <summary>
/// returns the element corresponding to the given axis
/// </summary>
template <Number T>
T &Vector1<T>::operator[](Axis1D aAxis)
    requires HostCode<T>
{
    switch (aAxis)
    {
        case Axis1D::X:
            return x;
    }

    throw INVALIDARGUMENT(aAxis, aAxis);
}
#pragma region Convert Methods
/// <summary>
/// Type conversion without saturation, direct type conversion
/// </summary>
template <Number T> template <Number T2> Vector1<T> DEVICE_CODE Vector1<T>::Convert(const Vector1<T2> &aVec)
{
    return {StaticCast<T2, T>(aVec.x)};
}
#pragma endregion

#pragma region Integral only Methods
/// <summary>
/// Element wise bitwise left shift
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::LShift(const Vector1<T> &aOther)
    requires RealIntegral<T>
{
    x = x << aOther.x; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise left shift
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::LShift(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires RealIntegral<T>
{
    Vector1<T> ret;              // NOLINT
    ret.x = aLeft.x << aRight.x; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise right shift
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::RShift(const Vector1<T> &aOther)
    requires RealIntegral<T>
{
    x = x >> aOther.x; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise right shift
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::RShift(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires RealIntegral<T>
{
    Vector1<T> ret;              // NOLINT
    ret.x = aLeft.x >> aRight.x; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise left shift
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::LShift(uint aOther)
    requires RealIntegral<T>
{
    x = x << aOther; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise left shift
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::LShift(const Vector1<T> &aLeft, uint aRight)
    requires RealIntegral<T>
{
    Vector1<T> ret;            // NOLINT
    ret.x = aLeft.x << aRight; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise right shift
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::RShift(uint aOther)
    requires RealIntegral<T>
{
    x = x >> aOther; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise right shift
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::RShift(const Vector1<T> &aLeft, uint aRight)
    requires RealIntegral<T>
{
    Vector1<T> ret;            // NOLINT
    ret.x = aLeft.x >> aRight; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise And
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::And(const Vector1<T> &aOther)
    requires RealIntegral<T>
{
    x = x & aOther.x; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise And
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::And(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires RealIntegral<T>
{
    Vector1<T> ret;             // NOLINT
    ret.x = aLeft.x & aRight.x; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise Or
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Or(const Vector1<T> &aOther)
    requires RealIntegral<T>
{
    x = x | aOther.x; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise Or
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Or(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires RealIntegral<T>
{
    Vector1<T> ret;             // NOLINT
    ret.x = aLeft.x | aRight.x; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise Xor
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Xor(const Vector1<T> &aOther)
    requires RealIntegral<T>
{
    x = x ^ aOther.x; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise Xor
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Xor(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires RealIntegral<T>
{
    Vector1<T> ret;             // NOLINT
    ret.x = aLeft.x ^ aRight.x; // NOLINT
    return ret;
}

/// <summary>
/// Element wise bitwise negation
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Not()
    requires RealIntegral<T>
{
    x = ~x; // NOLINT
    return *this;
}

/// <summary>
/// Element wise bitwise negation
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Not(const Vector1<T> &aVec)
    requires RealIntegral<T>
{
    Vector1<T> ret;  // NOLINT
    ret.x = ~aVec.x; // NOLINT
    return ret;
}
#pragma endregion

#pragma region Methods
#pragma region Exp
/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
Vector1<T> &Vector1<T>::Exp()
    requires HostCode<T> && NativeNumber<T>
{
    x = std::exp(x);
    return *this;
}
/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Exp()
    requires NonNativeNumber<T>
{
    x = T::Exp(x);
    return *this;
}

/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Exp()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = exp(x);
    return *this;
}

/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Exp(const Vector1<T> &aVec)
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    Vector1<T> ret;
    ret.x = exp(aVec.x);
    return ret;
}

/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Exp(const Vector1<T> &aVec)
    requires HostCode<T> && NativeNumber<T>
{
    Vector1<T> ret; // NOLINT
    ret.x = std::exp(aVec.x);
    return ret;
}

/// <summary>
/// Element wise exponential
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Exp(const Vector1<T> &aVec)
    requires NonNativeNumber<T>
{
    Vector1<T> ret; // NOLINT
    ret.x = T::Exp(aVec.x);
    return ret;
}
#pragma endregion

#pragma region Log
/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
Vector1<T> &Vector1<T>::Ln()
    requires HostCode<T> && NativeNumber<T>
{
    x = std::log(x);
    return *this;
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Ln()
    requires NonNativeFloatingPoint<T>
{
    x = T::Ln(x);
    return *this;
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Ln()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = log(x);
    return *this;
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Ln(const Vector1<T> &aVec)
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    Vector1<T> ret;
    ret.x = log(aVec.x);
    return ret;
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Ln(const Vector1<T> &aVec)
    requires HostCode<T> && NativeNumber<T>
{
    Vector1<T> ret; // NOLINT
    ret.x = std::log(aVec.x);
    return ret;
}

/// <summary>
/// Element wise natural logarithm
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Ln(const Vector1<T> &aVec)
    requires NonNativeFloatingPoint<T>
{
    Vector1<T> ret; // NOLINT
    ret.x = T::Ln(aVec.x);
    return ret;
}
#pragma endregion

#pragma region Sqr
/// <summary>
/// Element wise square
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> &Vector1<T>::Sqr()
{
    x = x * x;
    return *this;
}

/// <summary>
/// Element wise square
/// </summary>
template <Number T> DEVICE_CODE Vector1<T> Vector1<T>::Sqr(const Vector1<T> &aVec)
{
    Vector1<T> ret; // NOLINT
    ret.x = aVec.x * aVec.x;
    return ret;
}
#pragma endregion

#pragma region Sqrt
/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
Vector1<T> &Vector1<T>::Sqrt()
    requires HostCode<T> && NativeNumber<T>
{
    x = std::sqrt(x);
    return *this;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Sqrt()
    requires NonNativeFloatingPoint<T>
{
    x = T::Sqrt(x);
    return *this;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Sqrt()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = sqrt(x);
    return *this;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Sqrt(const Vector1<T> &aVec)
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    Vector1<T> ret;
    ret.x = sqrt(aVec.x);
    return ret;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Sqrt(const Vector1<T> &aVec)
    requires HostCode<T> && NativeNumber<T>
{
    Vector1<T> ret; // NOLINT
    ret.x = std::sqrt(aVec.x);
    return ret;
}

/// <summary>
/// Element wise square root
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Sqrt(const Vector1<T> &aVec)
    requires NonNativeFloatingPoint<T>
{
    Vector1<T> ret; // NOLINT
    ret.x = T::Sqrt(aVec.x);
    return ret;
}
#pragma endregion

#pragma region Abs
/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
Vector1<T> &Vector1<T>::Abs()
    requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    if constexpr (sizeof(T) < 4)
    {
        // short and sbyte are computed as int
        x = static_cast<T>(std::abs(x));
    }
    else
    {
        x = std::abs(x);
    }
    return *this;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Abs()
    requires RealSignedNumber<T> && NonNativeNumber<T>
{
    x = T::Abs(x);
    return *this;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Abs()
    requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    x = abs(x);
    return *this;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Abs(const Vector1<T> &aVec)
    requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    Vector1<T> ret;
    ret.x = abs(aVec.x);
    return ret;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Abs(const Vector1<T> &aVec)
    requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    Vector1<T> ret; // NOLINT
    if constexpr (sizeof(T) < 4)
    {
        // short and sbyte are computed as int
        ret.x = static_cast<T>(std::abs(aVec.x));
    }
    else
    {
        ret.x = std::abs(aVec.x);
    }
    return ret;
}

/// <summary>
/// Element wise absolute
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Abs(const Vector1<T> &aVec)
    requires RealSignedNumber<T> && NonNativeNumber<T>
{
    Vector1<T> ret; // NOLINT
    ret.x = T::Abs(aVec.x);
    return ret;
}
#pragma endregion

#pragma region AbsDiff
/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::AbsDiff(const Vector1<T> &aOther)
    requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    x = std::abs(x - aOther.x);
    return *this;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::AbsDiff(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires HostCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    Vector1<T> ret; // NOLINT
    ret.x = std::abs(aLeft.x - aRight.x);
    return ret;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::AbsDiff(const Vector1<T> &aOther)
    requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    x = abs(x - aOther.x);
    return *this;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::AbsDiff(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires DeviceCode<T> && RealSignedNumber<T> && NativeNumber<T>
{
    Vector1<T> ret;
    ret.x = abs(aLeft.x - aRight.x);
    return ret;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::AbsDiff(const Vector1<T> &aOther)
    requires RealSignedNumber<T> && NonNativeNumber<T>
{
    x = T::Abs(x - aOther.x);
    return *this;
}

/// <summary>
/// Element wise absolute difference
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::AbsDiff(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires RealSignedNumber<T> && NonNativeNumber<T>
{
    Vector1<T> ret; // NOLINT
    ret.x = T::Abs(aLeft.x - aRight.x);
    return ret;
}
#pragma endregion

#pragma region Methods for Complex types
/// <summary>
/// Conjugate complex per element
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Conj()
    requires ComplexNumber<T>
{
    x.Conj();
    return *this;
}

/// <summary>
/// Conjugate complex per element
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Conj(const Vector1<T> &aValue)
    requires ComplexNumber<T>
{
    return {T::Conj(aValue.x)};
}

/// <summary>
/// Conjugate complex multiplication: this * conj(aOther)  per element
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::ConjMul(const Vector1<T> &aOther)
    requires ComplexNumber<T>
{
    x.ConjMul(aOther.x);
    return *this;
}

/// <summary>
/// Conjugate complex multiplication: aLeft * conj(aRight) per element
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::ConjMul(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires ComplexNumber<T>
{
    return {T::ConjMul(aLeft.x, aRight.x)};
}

/// <summary>
/// Complex magnitude per element
/// </summary>
template <Number T>
DEVICE_CODE Vector1<complex_basetype_t<T>> Vector1<T>::Magnitude() const
    requires ComplexFloatingPoint<T>
{
    Vector1<complex_basetype_t<T>> ret; // NOLINT
    ret.x = x.Magnitude();
    return ret;
}

/// <summary>
/// Complex magnitude squared per element
/// </summary>
template <Number T>
DEVICE_CODE Vector1<complex_basetype_t<T>> Vector1<T>::MagnitudeSqr() const
    requires ComplexFloatingPoint<T>
{
    Vector1<complex_basetype_t<T>> ret; // NOLINT
    ret.x = x.MagnitudeSqr();
    return ret;
}

/// <summary>
/// Angle between real and imaginary of a complex number (atan2(image, real)) per element
/// </summary>
template <Number T>
DEVICE_CODE Vector1<complex_basetype_t<T>> Vector1<T>::Angle() const
    requires ComplexFloatingPoint<T>
{
    Vector1<complex_basetype_t<T>> ret; // NOLINT
    ret.x = x.Angle();
    return ret;
}
#pragma endregion

#pragma region Clamp
/// <summary>
/// Component wise clamp to value range
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Clamp(T aMinVal, T aMaxVal)
    requires DeviceCode<T> && NativeNumber<T>
{
    x = max(aMinVal, min(x, aMaxVal));
    return *this;
}

/// <summary>
/// Component wise clamp to value range
/// </summary>
template <Number T>
Vector1<T> &Vector1<T>::Clamp(T aMinVal, T aMaxVal)
    requires HostCode<T> && NativeNumber<T>
{
    x = std::max(aMinVal, std::min(x, aMaxVal));
    return *this;
}

/// <summary>
/// Component wise clamp to value range
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Clamp(T aMinVal, T aMaxVal)
    requires NonNativeNumber<T> && (!ComplexNumber<T>)
{
    x = T::Max(aMinVal, T::Min(x, aMaxVal));
    return *this;
}

/// <summary>
/// Component wise clamp to value range
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Clamp(complex_basetype_t<T> aMinVal, complex_basetype_t<T> aMaxVal)
    requires ComplexNumber<T>
{
    x.Clamp(aMinVal, aMaxVal);
    return *this;
}

/// <summary>
/// Component wise clamp to maximum value range of given target type
/// </summary>
template <Number T>
template <Number TTarget>
DEVICE_CODE Vector1<T> &Vector1<T>::ClampToTargetType() noexcept
    requires(need_saturation_clamp_v<T, TTarget>)
{
    return Clamp(numeric_limits_conversion<T, TTarget>::lowest(), numeric_limits_conversion<T, TTarget>::max());
}

/// <summary>
/// Component wise clamp to maximum value range of given target type<para/>
/// NOP in case no saturation clamping is needed.
/// </summary>
template <Number T>
template <Number TTarget>
DEVICE_CODE Vector1<T> &Vector1<T>::ClampToTargetType() noexcept
    requires(!need_saturation_clamp_v<T, TTarget>)
{
    return *this;
}
#pragma endregion

#pragma region Min
/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
Vector1<T> &Vector1<T>::Min(const Vector1<T> &aRight)
    requires HostCode<T> && NativeNumber<T>
{
    x = std::min(x, aRight.x);
    return *this;
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Min(const Vector1<T> &aRight)
    requires NonNativeNumber<T> && (!ComplexNumber<T>)
{
    x.Min(aRight.x);
    return *this;
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Min(const Vector1<T> &aRight)
    requires DeviceCode<T> && NativeNumber<T>
{
    x = min(x, aRight.x);
    return *this;
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Min(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires DeviceCode<T> && NativeNumber<T>
{
    return Vector1<T>{T(min(aLeft.x, aRight.x))};
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
Vector1<T> Vector1<T>::Min(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires HostCode<T> && NativeNumber<T>
{
    return Vector1<T>{std::min(aLeft.x, aRight.x)};
}

/// <summary>
/// Component wise minimum
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Min(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires NonNativeNumber<T> && (!ComplexNumber<T>)
{
    return Vector1<T>{T::Min(aLeft.x, aRight.x)};
}
#pragma endregion

#pragma region Max
/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Max(const Vector1<T> &aRight)
    requires DeviceCode<T> && NativeNumber<T>
{
    x = max(x, aRight.x);
    return *this;
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
Vector1<T> &Vector1<T>::Max(const Vector1<T> &aRight)
    requires HostCode<T> && NativeNumber<T>
{
    x = std::max(x, aRight.x);
    return *this;
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Max(const Vector1<T> &aRight)
    requires NonNativeNumber<T> && (!ComplexNumber<T>)
{
    x.Max(aRight.x);
    return *this;
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Max(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires DeviceCode<T> && NativeNumber<T>
{
    return Vector1<T>{T(max(aLeft.x, aRight.x))};
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
Vector1<T> Vector1<T>::Max(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires HostCode<T> && NativeNumber<T>
{
    return Vector1<T>{std::max(aLeft.x, aRight.x)};
}

/// <summary>
/// Component wise maximum
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Max(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires NonNativeNumber<T> && (!ComplexNumber<T>)
{
    return Vector1<T>{T::Max(aLeft.x, aRight.x)};
}
#pragma endregion

#pragma region Round
/// <summary>
/// Element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Round(const Vector1<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector1<T> ret = aValue;
    ret.Round();
    return ret;
}

/// <summary>
/// Element wise round()
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector1<T> &Vector1<T>::Round()
    requires NonNativeFloatingPoint<T>
{
    x.Round();
    return *this;
}

/// <summary>
/// Element wise round()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Round()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = round(x);
    return *this;
}

/// <summary>
/// Element wise round()
/// </summary>
template <Number T>
Vector1<T> &Vector1<T>::Round()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::round(x);
    return *this;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Floor(const Vector1<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector1<T> ret = aValue;
    ret.Floor();
    return ret;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Floor()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = floor(x);
    return *this;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <Number T>
Vector1<T> &Vector1<T>::Floor()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::floor(x);
    return *this;
}

/// <summary>
/// Element wise floor()
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector1<T> &Vector1<T>::Floor()
    requires NonNativeFloatingPoint<T>
{
    x.Floor();
    return *this;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::Ceil(const Vector1<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector1<T> ret = aValue;
    ret.Ceil();
    return ret;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::Ceil()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = ceil(x);
    return *this;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <Number T>
Vector1<T> &Vector1<T>::Ceil()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::ceil(x);
    return *this;
}

/// <summary>
/// Element wise ceil()
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector1<T> &Vector1<T>::Ceil()
    requires NonNativeFloatingPoint<T>
{
    x.Ceil();
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even<para/>
/// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::RoundNearest(const Vector1<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector1<T> ret = aValue;
    ret.RoundNearest();
    return ret;
}

/// <summary>
/// Element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::RoundNearest()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = rint(x);
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even<para/>
/// Note: this host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
template <Number T>
Vector1<T> &Vector1<T>::RoundNearest()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::nearbyint(x);
    return *this;
}

/// <summary>
/// Element wise round nearest ties to even
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector1<T> &Vector1<T>::RoundNearest()
    requires NonNativeFloatingPoint<T>
{
    x.RoundNearest();
    return *this;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> Vector1<T>::RoundZero(const Vector1<T> &aValue)
    requires RealOrComplexFloatingPoint<T>
{
    Vector1<T> ret = aValue;
    ret.RoundZero();
    return ret;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <Number T>
DEVICE_CODE Vector1<T> &Vector1<T>::RoundZero()
    requires DeviceCode<T> && NativeFloatingPoint<T>
{
    x = trunc(x);
    return *this;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <Number T>
Vector1<T> &Vector1<T>::RoundZero()
    requires HostCode<T> && NativeFloatingPoint<T>
{
    x = std::trunc(x);
    return *this;
}

/// <summary>
/// Element wise round toward zero
/// </summary>
template <Number T>
DEVICE_ONLY_CODE Vector1<T> &Vector1<T>::RoundZero()
    requires NonNativeFloatingPoint<T>
{
    x.RoundZero();
    return *this;
}
#pragma endregion

#pragma region Data accessors
/// <summary>
/// Provide a smiliar accessor to inner data as for std container
/// </summary>
template <Number T> DEVICE_CODE T *Vector1<T>::data()
{
    return &x;
}

/// <summary>
/// Provide a smiliar accessor to inner data as for std container
/// </summary>
template <Number T> DEVICE_CODE const T *Vector1<T>::data() const
{
    return &x;
}
#pragma endregion

#pragma region Compare per element
/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
Vector1<byte> Vector1<T>::CompareEQEps(const Vector1<T> &aLeft, const Vector1<T> &aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && HostCode<T>
{
    Vector1<T> left  = aLeft;
    Vector1<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);

    Vector1<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(std::abs(left.x - right.x) <= aEpsilon) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector1<byte> Vector1<T>::CompareEQEps(const Vector1<T> &aLeft, const Vector1<T> &aRight, T aEpsilon)
    requires NativeFloatingPoint<T> && DeviceCode<T>
{
    Vector1<T> left  = aLeft;
    Vector1<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);

    Vector1<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(abs(left.x - right.x) <= aEpsilon) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector1<byte> Vector1<T>::CompareEQEps(const Vector1<T> &aLeft, const Vector1<T> &aRight, T aEpsilon)
    requires Is16BitFloat<T>
{
    Vector1<T> left  = aLeft;
    Vector1<T> right = aRight;
    MakeNANandINFValid(left.x, right.x);

    Vector1<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(T::Abs(left.x - right.x) <= aEpsilon) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal with epsilon margin, if both elements to compare are NAN/INF result is true for
/// the element, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector1<byte> Vector1<T>::CompareEQEps(const Vector1<T> &aLeft, const Vector1<T> &aRight,
                                                   complex_basetype_t<T> aEpsilon)
    requires ComplexFloatingPoint<T>
{
    Vector1<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(T::EqEps(aLeft.x, aRight.x, aEpsilon)) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison equal, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T> DEVICE_CODE Vector1<byte> Vector1<T>::CompareEQ(const Vector1<T> &aLeft, const Vector1<T> &aRight)
{
    Vector1<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x == aRight.x) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison greater or equal, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector1<byte> Vector1<T>::CompareGE(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires RealNumber<T>
{
    Vector1<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x >= aRight.x) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison greater than, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector1<byte> Vector1<T>::CompareGT(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires RealNumber<T>
{
    Vector1<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x > aRight.x) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison less or equal, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector1<byte> Vector1<T>::CompareLE(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires RealNumber<T>
{
    Vector1<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x <= aRight.x) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison less than, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T>
DEVICE_CODE Vector1<byte> Vector1<T>::CompareLT(const Vector1<T> &aLeft, const Vector1<T> &aRight)
    requires RealNumber<T>
{
    Vector1<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x < aRight.x) * TRUE_VALUE);
    return ret;
}

/// <summary>
/// Element wise comparison not equal, returns 0xFF per element for true, 0x00 for false
/// </summary>
template <Number T> DEVICE_CODE Vector1<byte> Vector1<T>::CompareNEQ(const Vector1<T> &aLeft, const Vector1<T> &aRight)
{
    Vector1<byte> ret; // NOLINT
    ret.x = static_cast<byte>(static_cast<int>(aLeft.x != aRight.x) * TRUE_VALUE);
    return ret;
}
#pragma endregion
#pragma endregion

template <HostCode T2> std::ostream &operator<<(std::ostream &aOs, const Vector1<T2> &aVec)
{
    aOs << '(' << aVec.x << ')';
    return aOs;
}

template <HostCode T2> std::wostream &operator<<(std::wostream &aOs, const Vector1<T2> &aVec)
{
    aOs << '(' << aVec.x << ')';
    return aOs;
}

// byte and sbyte are treated as characters and not numbers:
template <HostCode T2>
std::ostream &operator<<(std::ostream &aOs, const Vector1<T2> &aVec)
    requires ByteSizeType<T2>
{
    aOs << '(' << static_cast<int>(aVec.x) << ')';
    return aOs;
}

template <HostCode T2>
std::wostream &operator<<(std::wostream &aOs, const Vector1<T2> &aVec)
    requires ByteSizeType<T2>
{
    aOs << '(' << static_cast<int>(aVec.x) << ')';
    return aOs;
}

template <HostCode T2> std::istream &operator>>(std::istream &aIs, Vector1<T2> &aVec)
{
    aIs >> aVec.x;
    return aIs;
}

template <HostCode T2> std::wistream &operator>>(std::wistream &aIs, Vector1<T2> &aVec)
{
    aIs >> aVec.x;
    return aIs;
}

template <HostCode T2>
std::istream &operator>>(std::istream &aIs, Vector1<T2> &aVec)
    requires ByteSizeType<T2>
{
    int temp = 0;
    aIs >> temp;
    aVec.x = static_cast<T2>(temp);
    return aIs;
}

template <HostCode T2>
std::wistream &operator>>(std::wistream &aIs, Vector1<T2> &aVec)
    requires ByteSizeType<T2>
{
    int temp = 0;
    aIs >> temp;
    aVec.x = static_cast<T2>(temp);
    return aIs;
}

} // namespace mpp