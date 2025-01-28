#pragma once
#include <common/complex.h>
#include <common/defines.h>
#include <common/opp_defs.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace opp
{
template <AnyVector TFrom, AnyVector TTo> struct Convert
{
    // not const TFrom as we want to clamp value range inplace in aSrc1
    DEVICE_CODE void operator()(TFrom &aSrc1, TTo &aDst)
    {
        aDst = static_cast<TTo>(aSrc1);
    }
};

template <AnyVector TFrom, AnyVector TTo> struct ConvertRound
{
    RoundingMode mRoundingMode;

    ConvertRound(RoundingMode aRoundingMode) : mRoundingMode(aRoundingMode)
    {
    }

    DEVICE_CODE void operator()(const TFrom &aSrc1, TTo &aDst)
    {
        aDst = TTo(aSrc1, mRoundingMode);
    }
};

} // namespace opp
