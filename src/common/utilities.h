#pragma once

#include "defines.h"
#include "numberTypes.h"
#include "numeric_limits.h"
#include <cfloat>
#include <cmath>
#include <numbers>
#include <vector>

namespace mpp
{

/// <summary>
/// returns as result a value res so that res * aDiv is always >= aValue. Basically a floating point division with
/// ceiling as rounding to int.
/// </summary>
DEVICE_CODE constexpr uint DIV_UP(uint aValue, uint aDiv)
{
    // yes this could theoretically overflow, but we use this mainly for grid launch config, so numbers are way smaller
    // than UINT_MAX...
    return (aValue + aDiv - 1) / aDiv;
}

/// <summary>
/// Converts degrees to radians (float)
/// </summary>
DEVICE_CODE constexpr float DEG_TO_RAD(float aDeg)
{
    return aDeg * std::numbers::pi_v<float> / 180.0f;
}

/// <summary>
/// Converts degrees to radians (double)
/// </summary>
DEVICE_CODE constexpr double DEG_TO_RAD(double aDeg)
{
    return aDeg * std::numbers::pi_v<double> / 180.0;
}

/// <summary>
/// Converts radians to degrees (float)
/// </summary>
DEVICE_CODE constexpr float RAD_TO_DEG(float aDeg)
{
    return aDeg * 180.0f / std::numbers::pi_v<float>;
}

/// <summary>
/// Converts radians to degrees (double)
/// </summary>
DEVICE_CODE constexpr double RAD_TO_DEG(double aDeg)
{
    return aDeg * 180.0 / std::numbers::pi_v<double>;
}

/// <summary>
/// Normalized sinc function: sin(pi * x) / (pi * x). Uses __sinf intrinsic.
/// </summary>
DEVICE_ONLY_CODE inline float sinc(float aVal)
{
    if (aVal == 0.0f)
    {
        return 1.0f;
    }
    aVal *= std::numbers::pi_v<float>;
#ifdef IS_CUDA_COMPILER
    return __sinf(aVal) / aVal;
#endif
#ifdef IS_HOST_COMPILER
    return std::sin(aVal) / aVal;
#endif
}

/// <summary>
/// Normalized sinc function: sin(pi * x) / (pi * x)
/// </summary>
DEVICE_CODE inline double sinc(double aVal)
{
    if (aVal == 0.0)
    {
        return 1.0;
    }
    aVal *= std::numbers::pi_v<double>;
#ifdef IS_CUDA_COMPILER
    return sin(aVal) / aVal;
#endif
#ifdef IS_HOST_COMPILER
    return std::sin(aVal) / aVal;
#endif
}

/// <summary>
/// Normalized sinc function: sin(pi * x) / (pi * x). Does not perform check for x == 0. Uses __sinf intrinsic.
/// </summary>
DEVICE_ONLY_CODE inline float sinc_never0(float aVal)
{
    aVal *= std::numbers::pi_v<float>;
#ifdef IS_CUDA_COMPILER
    return __sinf(aVal) / aVal;
#endif
#ifdef IS_HOST_COMPILER
    return std::sin(aVal) / aVal;
#endif
}

/// <summary>
/// Normalized sinc function: sin(pi * x) / (pi * x). Does not perform check for x == 0.
/// </summary>
DEVICE_CODE inline double sinc_never0(double aVal)
{
    aVal *= std::numbers::pi_v<double>;
#ifdef IS_CUDA_COMPILER
    return sin(aVal) / aVal;
#endif
#ifdef IS_HOST_COMPILER
    return std::sin(aVal) / aVal;
#endif
}

/// <summary>
/// Converts the integer scaling argument to a double multiplication factor according to the NPP/IPP convention:
/// factor = 2^-aScale
/// </summary>
inline double GetScaleFactor(int aScale)
{
    return std::pow(2.0, static_cast<double>(-aScale));
}

/// <summary>
/// Get the sign of a floating point value as float, i.e. either +1.0f or -1.0f
/// </summary>
DEVICE_CODE inline float GetSign(float aFloatVal)
{
    constexpr uint ONE_AS_FLOAT = 0x3F800000;
    constexpr uint SIGN_BIT     = 0x80000000;
    uint floatbits              = *reinterpret_cast<uint *>(&aFloatVal);
    floatbits &= SIGN_BIT;
    floatbits |= ONE_AS_FLOAT;
    return *reinterpret_cast<float *>(&floatbits);
}

/// <summary>
/// Get the sign of a floating point value as double, i.e. either +1.0 or -1.0
/// </summary>
DEVICE_CODE inline double GetSign(double aDoubleVal)
{
    constexpr ulong64 ONE_AS_DOUBLE = 0x3FF0000000000000ULL;
    constexpr ulong64 SIGN_BIT      = 0x8000000000000000ULL;
    ulong64 doublebits              = *reinterpret_cast<ulong64 *>(&aDoubleVal);
    doublebits &= SIGN_BIT;
    doublebits |= ONE_AS_DOUBLE;
    return *reinterpret_cast<double *>(&doublebits);
}

/// <summary>
/// Get the sign of an integer value as T, i.e. either +1 or -1
/// </summary>
template <Number T>
    requires RealSignedIntegral<T>
DEVICE_CODE constexpr T GetSign(T aVal)
{
    return +1 | ((aVal) >> mpp::numeric_limits<T>::bitCountMinus1());
}

/// <summary>
/// Get the sign of an integer multiplication or division result with value as T, i.e. either +1 or -1<para/>
/// If both values are positive or both negative the result is +1, or -1 if only one value is negative.
/// </summary>
template <Number T>
    requires RealSignedIntegral<T>
DEVICE_CODE constexpr T GetSign(T aSrc0, T aSrc1)
{
    return +1 | ((aSrc0 ^ aSrc1) >> mpp::numeric_limits<T>::bitCountMinus1());
}

DEVICE_CODE inline bool isnan(float aFloat)
{
#ifdef IS_CUDA_COMPILER
    return ::isnan(aFloat);
#endif
#ifdef IS_HOST_COMPILER
    return std::isnan(aFloat);
#endif
}
DEVICE_CODE inline bool isinf(float aFloat)
{
#ifdef IS_CUDA_COMPILER
    return ::isinf(aFloat);
#endif
#ifdef IS_HOST_COMPILER
    return std::isinf(aFloat);
#endif
}

DEVICE_CODE inline bool isnan(double aFloat)
{
#ifdef IS_CUDA_COMPILER
    return ::isnan(aFloat);
#endif
#ifdef IS_HOST_COMPILER
    return std::isnan(aFloat);
#endif
}
DEVICE_CODE inline bool isinf(double aFloat)
{
#ifdef IS_CUDA_COMPILER
    return ::isinf(aFloat);
#endif
#ifdef IS_HOST_COMPILER
    return std::isinf(aFloat);
#endif
}

/// <summary>
/// For floating point comparison: if both values are NAN or INF, we want to accept that both values are equal, if only
/// one of the values is NAN or INF, the comparison should return false. For this we replace the actual NAN/INF by some
/// valid value.
/// </summary>
template <RealFloatingPoint T> DEVICE_CODE void MakeNANandINFValid(T &aLeft, T &aRight)
{
    if (isnan(aLeft) || isinf(aLeft))
    {
        if (isnan(aRight) || isinf(aRight))
        {
            // if left and right are nan/inf, set them to same value and result will be true for any aEpsilon
            aLeft  = T(0.0f);
            aRight = T(0.0f);
        }
        else
        {
            // if only left is nan/inf, set them to max valid value difference and result will be false for any
            // (reasonable) aEpsilon
            aLeft  = T(0.0f); // aEpsilon should be < numeric_limits<T>::max() / 2
            aRight = mpp::numeric_limits<T>::max();
        }
    }
    else
    {
        if (isnan(aRight) || isinf(aRight))
        {
            // if only right is nan/inf, set them to max valid value difference and result will be false for any
            // (reasonable) aEpsilon
            aLeft  = T(0.0f); // aEpsilon should be < numeric_limits<T>::max() / 2
            aRight = mpp::numeric_limits<T>::max();
        }
    }
}

template <Number T>
    requires RealFloatingPoint<T>
DEVICE_CODE T GetAlphaAsFloat(T aValue)
{
    return aValue;
}
template <Number T>
    requires RealIntegral<T> // byte, sybte, ushort and short
DEVICE_CODE float GetAlphaAsFloat(T aValue)
{
    // Note: The results from NPP are not coherent. The result from AlphaComp with AlphaComposition::Over and an empty
    // second image is not the same as AlphaPremul - but they should give same results. So instead of trying getting the
    // exact same result as NPP we'll just do simple math as described in the IPP functions.
    return (static_cast<float>(aValue)) / static_cast<float>(mpp::numeric_limits<T>::max());
}
template <Number T>
    requires RealIntegral<T> && FourBytesSizeType<T> // int32 and uint32
DEVICE_CODE double GetAlphaAsFloat(T aValue)
{
    return (static_cast<double>(aValue)) / static_cast<double>(mpp::numeric_limits<T>::max());
}

// Integer division with rounding

/// <summary>
/// same as floating point nearbyint() in c++
/// </summary>
template <Number T>
    requires RealSignedIntegral<T>
DEVICE_CODE T DivRoundNearestEven(T aSrc0, T aSrc1)
{
    if (aSrc1 == 0)
    {
        if (aSrc0 < 0)
        {
            return mpp::numeric_limits<T>::lowest();
        }
        else
        {
            return mpp::numeric_limits<T>::max();
        }
    }

    const T sign = GetSign(aSrc0, aSrc1);

#ifdef IS_CUDA_COMPILER
    aSrc0 = abs(aSrc0);
    aSrc1 = abs(aSrc1);
#endif
#ifdef IS_HOST_COMPILER
    aSrc0 = std::abs(aSrc0);
    aSrc1 = std::abs(aSrc1);
#endif

    const T src1_half = aSrc1 / 2;

    const T offset     = aSrc0 + src1_half;
    const T roundup    = offset / aSrc1;
    const T isMultiple = roundup * aSrc1 == offset;
    const T res        = (aSrc1 | (roundup ^ isMultiple)) & roundup;
    return res * sign;
}

/// <summary>
/// same as floating point round() in c++
/// </summary>
template <Number T>
    requires RealSignedIntegral<T>
DEVICE_CODE T DivRoundTiesAwayFromZero(T aSrc0, T aSrc1)
{
    if (aSrc1 == 0)
    {
        if (aSrc0 < 0)
        {
            return mpp::numeric_limits<T>::lowest();
        }
        else
        {
            return mpp::numeric_limits<T>::max();
        }
    }
    T sign = GetSign(aSrc0, aSrc1);

#ifdef IS_CUDA_COMPILER
    aSrc0 = abs(aSrc0);
    aSrc1 = abs(aSrc1);
#endif
#ifdef IS_HOST_COMPILER
    aSrc0 = std::abs(aSrc0);
    aSrc1 = std::abs(aSrc1);
#endif

    const T src1_half = aSrc1 / 2;
    return (aSrc0 + src1_half) / aSrc1 * sign;
}

/// <summary>
/// same as floating point truncate() in c++
/// </summary>
template <Number T>
    requires RealSignedIntegral<T>
DEVICE_CODE T DivRoundTowardZero(T aSrc0, T aSrc1)
{
    if (aSrc1 == 0)
    {
        if (aSrc0 < 0)
        {
            return mpp::numeric_limits<T>::lowest();
        }
        else
        {
            return mpp::numeric_limits<T>::max();
        }
    }

    return aSrc0 / aSrc1;
}

/// <summary>
/// same as floating point floor() in c++
/// </summary>
template <Number T>
    requires RealSignedIntegral<T>
DEVICE_CODE T DivRoundTowardNegInf(T aSrc0, T aSrc1)
{
    if (aSrc1 == 0)
    {
        if (aSrc0 < 0)
        {
            return mpp::numeric_limits<T>::lowest();
        }
        else
        {
            return mpp::numeric_limits<T>::max();
        }
    }

    const T sign  = GetSign(aSrc0, aSrc1);
    const T sign2 = ((aSrc0 ^ aSrc1) >> mpp::numeric_limits<T>::bitCountMinus1());

#ifdef IS_CUDA_COMPILER
    aSrc0 = abs(aSrc0);
    aSrc1 = abs(aSrc1);
#endif
#ifdef IS_HOST_COMPILER
    aSrc0 = std::abs(aSrc0);
    aSrc1 = std::abs(aSrc1);
#endif

    const T src1Minus = (aSrc1 - 1) * sign2;
    return (aSrc0 - src1Minus) / aSrc1 * sign;
}

/// <summary>
/// same as floating point ceil() in c++
/// </summary>
template <Number T>
    requires RealSignedIntegral<T>
DEVICE_CODE T DivRoundTowardPosInf(T aSrc0, T aSrc1)
{
    if (aSrc1 == 0)
    {
        if (aSrc0 < 0)
        {
            return mpp::numeric_limits<T>::lowest();
        }
        else
        {
            return mpp::numeric_limits<T>::max();
        }
    }
    T sign  = GetSign(aSrc0, aSrc1);
    T sign2 = 1 + ((aSrc0 ^ aSrc1) >> mpp::numeric_limits<T>::bitCountMinus1());

#ifdef IS_CUDA_COMPILER
    aSrc0 = abs(aSrc0);
    aSrc1 = abs(aSrc1);
#endif
#ifdef IS_HOST_COMPILER
    aSrc0 = std::abs(aSrc0);
    aSrc1 = std::abs(aSrc1);
#endif

    T src1Plus = (aSrc1 - 1) * sign2;
    return (aSrc0 + src1Plus) / aSrc1 * sign;
}

template <Number T>
    requires RealUnsignedIntegral<T>
DEVICE_CODE T DivRoundNearestEven(T aSrc0, T aSrc1)
{
    if (aSrc1 == 0)
    {
        return mpp::numeric_limits<T>::max();
    }

    const T src1_half = aSrc1 / 2;

    const T offset     = aSrc0 + src1_half;
    const T roundup    = offset / aSrc1;
    const T isMultiple = roundup * aSrc1 == offset;
    const T res        = (aSrc1 | (roundup ^ isMultiple)) & roundup;
    return res;
}

template <Number T>
    requires RealUnsignedIntegral<T>
DEVICE_CODE T DivRoundTiesAwayFromZero(T aSrc0, T aSrc1)
{
    if (aSrc1 == 0)
    {
        return mpp::numeric_limits<T>::max();
    }

    const T src1_half = aSrc1 / 2;
    return (aSrc0 + src1_half) / aSrc1;
}

template <Number T>
    requires RealUnsignedIntegral<T>
DEVICE_CODE T DivRoundTowardZero(T aSrc0, T aSrc1)
{
    if (aSrc1 == 0)
    {
        return mpp::numeric_limits<T>::max();
    }

    return aSrc0 / aSrc1;
}

template <Number T>
    requires RealUnsignedIntegral<T>
DEVICE_CODE T DivRoundTowardNegInf(T aSrc0, T aSrc1)
{
    return DivRoundTowardZero(aSrc0, aSrc1);
}

template <Number T>
    requires RealUnsignedIntegral<T>
DEVICE_CODE T DivRoundTowardPosInf(T aSrc0, T aSrc1)
{
    if (aSrc1 == 0)
    {
        return mpp::numeric_limits<T>::max();
    }

    T src1Minus = (aSrc1 - 1);
    return (aSrc0 + src1Minus) / aSrc1;
}

// Div round for scaling --> aSrc1 is always positive and > 0, we can skip some checks
template <Number T>
    requires RealSignedIntegral<T>
DEVICE_CODE T DivScaleRoundNearestEven(T aSrc0, T aSrc1)
{
    const T sign = GetSign(aSrc0);

#ifdef IS_CUDA_COMPILER
    aSrc0 = abs(aSrc0);
#endif
#ifdef IS_HOST_COMPILER
    aSrc0 = std::abs(aSrc0);
#endif

    const T src1_half = aSrc1 / 2;

    const T offset     = aSrc0 + src1_half;
    const T roundup    = offset / aSrc1;
    const T isMultiple = roundup * aSrc1 == offset;
    const T res        = (aSrc1 | (roundup ^ isMultiple)) & roundup;
    return res * sign;
}

// Div round for scaling --> aSrc1 is always positive and > 0, we can skip some checks
template <Number T>
    requires RealSignedIntegral<T>
DEVICE_CODE T DivScaleRoundTiesAwayFromZero(T aSrc0, T aSrc1)
{
    T sign = GetSign(aSrc0);

#ifdef IS_CUDA_COMPILER
    aSrc0 = abs(aSrc0);
#endif
#ifdef IS_HOST_COMPILER
    aSrc0 = std::abs(aSrc0);
#endif

    const T src1_half = aSrc1 / 2;
    return (aSrc0 + src1_half) / aSrc1 * sign;
}

// Div round for scaling --> aSrc1 is always positive and > 0, we can skip some checks
template <Number T>
    requires RealSignedIntegral<T>
DEVICE_CODE T DivScaleRoundTowardZero(T aSrc0, T aSrc1)
{
    return aSrc0 / aSrc1;
}

// Div round for scaling --> aSrc1 is always positive and > 0, we can skip some checks
template <Number T>
    requires RealSignedIntegral<T>
DEVICE_CODE T DivScaleRoundTowardNegInf(T aSrc0, T aSrc1)
{
    const T sign  = GetSign(aSrc0);
    const T sign2 = aSrc0 >> mpp::numeric_limits<T>::bitCountMinus1();

#ifdef IS_CUDA_COMPILER
    aSrc0 = abs(aSrc0);
#endif
#ifdef IS_HOST_COMPILER
    aSrc0 = std::abs(aSrc0);
#endif

    const T src1Minus = (aSrc1 - 1) * sign2;
    return (aSrc0 - src1Minus) / aSrc1 * sign;
}

// Div round for scaling --> aSrc1 is always positive and > 0, we can skip some checks
template <Number T>
    requires RealSignedIntegral<T>
DEVICE_CODE T DivScaleRoundTowardPosInf(T aSrc0, T aSrc1)
{
    T sign  = GetSign(aSrc0);
    T sign2 = 1 + (aSrc0 >> mpp::numeric_limits<T>::bitCountMinus1());

#ifdef IS_CUDA_COMPILER
    aSrc0 = abs(aSrc0);
#endif
#ifdef IS_HOST_COMPILER
    aSrc0 = std::abs(aSrc0);
#endif

    T src1Plus = (aSrc1 - 1) * sign2;
    return (aSrc0 + src1Plus) / aSrc1 * sign;
}

template <Number T>
    requires RealUnsignedIntegral<T>
DEVICE_CODE T DivScaleRoundNearestEven(T aSrc0, T aSrc1)
{
    const T src1_half = aSrc1 / 2;

    const T offset     = aSrc0 + src1_half;
    const T roundup    = offset / aSrc1;
    const T isMultiple = roundup * aSrc1 == offset;
    const T res        = (aSrc1 | (roundup ^ isMultiple)) & roundup;
    return res;
}

template <Number T>
    requires RealUnsignedIntegral<T>
DEVICE_CODE T DivScaleRoundTiesAwayFromZero(T aSrc0, T aSrc1)
{
    const T src1_half = aSrc1 / 2;
    return (aSrc0 + src1_half) / aSrc1;
}

template <Number T>
    requires RealUnsignedIntegral<T>
DEVICE_CODE T DivScaleRoundTowardZero(T aSrc0, T aSrc1)
{
    return aSrc0 / aSrc1;
}

template <Number T>
    requires RealUnsignedIntegral<T>
DEVICE_CODE T DivScaleRoundTowardNegInf(T aSrc0, T aSrc1)
{
    return DivRoundTowardZero(aSrc0, aSrc1);
}

template <Number T>
    requires RealUnsignedIntegral<T>
DEVICE_CODE T DivScaleRoundTowardPosInf(T aSrc0, T aSrc1)
{
    T src1Minus = (aSrc1 - 1);
    return (aSrc0 + src1Minus) / aSrc1;
}

} // namespace mpp