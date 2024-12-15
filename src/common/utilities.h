#pragma once

#include "defines.h"
#include <cmath>
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