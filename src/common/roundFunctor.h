#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/vectorTypes.h>
#include <concepts>

namespace opp
{

//// for all other undefined types, just do nothing:
// template <RoudingMode roundingMode, typename T> struct RoundFunctor
//{
// };

template <RoudingMode roundingMode, VectorType T> struct RoundFunctor
{
    DEVICE_CODE void operator()(T &aVec)
        requires std::floating_point<typename remove_vector<T>::type>
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
        else
        {
            static_assert(AlwaysFalse<T>::value, "Unknown rounding mode");
        }
    }
};
} // namespace opp
