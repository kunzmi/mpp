#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>

namespace opp
{
template <RoudingMode roundingMode, typename T> struct RoundFunctor
{
    DEVICE_CODE void operator()(T &aVec)
        requires FloatingVectorOrComplexType<T>
    {
        if constexpr (roundingMode == RoudingMode::NearestTiesToEven)
        {
            aVec.RoundNearest();
            return;
        }
        else if constexpr (roundingMode == RoudingMode::NearestTiesAwayFromZero)
        {
            aVec.Round();
            return;
        }
        else if constexpr (roundingMode == RoudingMode::TowardZero)
        {
            aVec.RoundZero();
            return;
        }
        else if constexpr (roundingMode == RoudingMode::TowardNegativeInfinity)
        {
            aVec.Floor();
            return;
        }
        else if constexpr (roundingMode == RoudingMode::TowardPositiveInfinity)
        {
            aVec.Ceil();
            return;
        }
        else if constexpr (roundingMode == RoudingMode::None)
        {
            return;
        }
        else
        {
            static_assert(AlwaysFalse<T>::value, "Unknown rounding mode");
        }
    }
};
} // namespace opp
