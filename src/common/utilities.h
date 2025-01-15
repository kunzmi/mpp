#pragma once

#include "defines.h"
#include "numberTypes.h"
#include "numeric_limits.h"
#include <cfloat>
#include <cmath>
#include <numbers>
#include <vector>

namespace opp
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
/// Converts the integer scaling argument to a float multiplication factor according to the NPP/IPP convention:
/// factor = 2^-aScale
/// </summary>
inline float GetScaleFactor(int aScale)
{
    return std::pow(2.0f, static_cast<float>(-aScale));
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

template <typename T> struct numeric_limits;

#ifdef IS_CUDA_COMPILER
DEVICE_CODE inline bool isnan(float aFloat)
{
    return ::isnan(aFloat);
}
DEVICE_CODE inline bool isinf(float aFloat)
{
    return ::isinf(aFloat);
}
#endif

/// <summary>
/// For floating point comparison: if both values are NAN or INF, we want to accept that both values are equal, if only
/// one of the values is NAN or INF, the comparison should return false. For this we replace the actual NAN/INF by some
/// valid value.
/// </summary>
template <RealFloatingPoint T> DEVICE_CODE void MakeNANandINFValid(T &aLeft, T &aRight)
{
#ifdef IS_HOST_COMPILER
    using namespace std;
#endif
#ifdef IS_CUDA_COMPILER
    using namespace opp;
#endif

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
            aRight = opp::numeric_limits<T>::max();
        }
    }
    else
    {
        if (isnan(aRight) || isinf(aRight))
        {
            // if only right is nan/inf, set them to max valid value difference and result will be false for any
            // (reasonable) aEpsilon
            aLeft  = T(0.0f); // aEpsilon should be < numeric_limits<T>::max() / 2
            aRight = opp::numeric_limits<T>::max();
        }
    }
}

} // namespace opp