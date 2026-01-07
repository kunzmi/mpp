#pragma once
#include <common/defines.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>

namespace mpp
{
template <RoundingMode roundingMode, typename T> struct RoundFunctor
{
    DEVICE_CODE void operator()(T &aVec) const
        requires RealOrComplexFloatingVector<T>
    {
        if constexpr (roundingMode == RoundingMode::NearestTiesToEven)
        {
            aVec.RoundNearest();
            return;
        }
        else if constexpr (roundingMode == RoundingMode::NearestTiesAwayFromZero)
        {
            aVec.Round();
            return;
        }
        else if constexpr (roundingMode == RoundingMode::TowardZero)
        {
            aVec.RoundZero();
            return;
        }
        else if constexpr (roundingMode == RoundingMode::TowardNegativeInfinity)
        {
            aVec.Floor();
            return;
        }
        else if constexpr (roundingMode == RoundingMode::TowardPositiveInfinity)
        {
            aVec.Ceil();
            return;
        }
        else if constexpr (roundingMode == RoundingMode::None)
        {
            return;
        }
        else if constexpr (roundingMode == RoundingMode::NearestTiesAwayFromZeroPositive)
        {
            aVec += static_cast<complex_basetype_t<remove_vector_t<T>>>(0.5);
            aVec.Floor();
            return;
        }
        else
        {
            static_assert(AlwaysFalse<T>::value, "Unknown rounding mode");
        }
    }

    DEVICE_CODE void operator()(T & /*aVec*/) const
        requires RealOrComplexIntVector<T>
    { // NOP
    }
};
} // namespace mpp
