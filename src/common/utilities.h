#pragma once

#include "defines.h"
#include <cfloat>
#include <cmath>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <numbers>

DEVICE_CODE constexpr opp::uint DIV_UP(opp::uint aValue, opp::uint aDiv)
{
    return (aValue + aDiv - 1) / aDiv;
}

DEVICE_CODE constexpr float DEG_TO_RAD(float aDeg)
{
    return aDeg * std::numbers::pi_v<float> / 180.0f;
}

DEVICE_CODE constexpr double DEG_TO_RAD(double aDeg)
{
    return aDeg * std::numbers::pi_v<double> / 180.0;
}

DEVICE_CODE constexpr float RAD_TO_DEG(float aDeg)
{
    return aDeg * 180.0f / std::numbers::pi_v<float>;
}

DEVICE_CODE constexpr double RAD_TO_DEG(double aDeg)
{
    return aDeg * 180.0 / std::numbers::pi_v<double>;
}

inline double GetScaleFactor(int aScale)
{
    if (aScale < 0)
    {
        aScale = -aScale;
        return 1.0f / std::pow(2, double(aScale));
    }
    return std::pow(2, double(aScale));
}

DEVICE_CODE inline float GetSign(float aFloatVal)
{
    constexpr opp::uint ONE_AS_FLOAT = 0x3F800000;
    constexpr opp::uint SIGN_BIT     = 0x80000000;
    opp::uint floatbits              = *reinterpret_cast<opp::uint *>(&aFloatVal);
    floatbits &= SIGN_BIT;
    floatbits |= ONE_AS_FLOAT;
    return *reinterpret_cast<float *>(&floatbits);
}

DEVICE_CODE inline double GetSign(double aDoubleVal)
{
    constexpr opp::ulong64 ONE_AS_DOUBLE = 0x3FF0000000000000ULL;
    constexpr opp::ulong64 SIGN_BIT      = 0x8000000000000000ULL;
    opp::ulong64 doublebits              = *reinterpret_cast<opp::ulong64 *>(&aDoubleVal);
    doublebits &= SIGN_BIT;
    doublebits |= ONE_AS_DOUBLE;
    return *reinterpret_cast<double *>(&doublebits);
}

template <opp::RealFloatingPoint T> DEVICE_CODE void MakeNANandINFValid(T &aLeft, T &aRight)
{
#ifdef IS_HOST_COMPILER
    using namespace std;
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
            // if only left is nan/inf, set them to max valid value difference and result will be false any
            // (reasonable) aEpsilon
            aLeft  = T(0.0f); // aEpsilon should be < numeric_limits<T>::max() / 2
            aRight = opp::numeric_limits<T>::max();
        }
    }
    else
    {
        if (isnan(aRight) || isinf(aRight))
        {
            // if only right is nan/inf, set them to max valid value difference and result will be false any
            // (reasonable) aEpsilon
            aLeft  = T(0.0f); // aEpsilon should be < numeric_limits<T>::max() / 2
            aRight = opp::numeric_limits<T>::max();
        }
    }
}