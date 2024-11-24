#pragma once
#include "imageFunctors.h"
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace opp::image
{
/// <summary>
/// Computes an output pixel from one srcDst pixel -> srcDst pixel inplace
/// </summary>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename ComputeT, typename DstT, typename operation, typename ComputeT_SIMD = void,
          typename operation_SIMD = NullOp<void>, RoudingMode roundingMode = RoudingMode::NearestTiesAwayFromZero>
struct InplaceFunctor : public ImageFunctor<true>
{
    operation Op;
    operation_SIMD OpSIMD;

    RoundFunctor<roundingMode, ComputeT> round;

    InplaceFunctor()
    {
    }

    InplaceFunctor(operation aOp) : Op(aOp)
    {
    }

    InplaceFunctor(operation aOp, operation_SIMD aOpSIMD) : Op(aOp), OpSIMD(aOpSIMD)
    {
    }

    DEVICE_CODE void operator()(int /*aPixelX*/, int /*aPixelY*/, DstT &aDst)
        requires Integral<pixel_basetype_t<DstT>> && //
                 FloatingPoint<pixel_basetype_t<ComputeT>>
    {
        ComputeT temp(aDst);
        Op(temp);
        round(temp);
        // DstT constructor will clamp temp to value range of DstT
        aDst = DstT(temp);
    }

    DEVICE_CODE void operator()(int /*aPixelX*/, int /*aPixelY*/, Tupel<DstT, tupelSize> &aDst)
        requires Integral<pixel_basetype_t<DstT>> &&          //
                 FloatingPoint<pixel_basetype_t<ComputeT>> && //
                 (tupelSize > 1)
    {
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp(aDst.value[i]);
            Op(temp);
            round(temp);
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = DstT(temp);
        }
    }

    DEVICE_CODE void operator()(int /*aPixelX*/, int /*aPixelY*/, DstT &aDst)
        requires std::same_as<ComputeT, DstT>
    {
        Op(aDst);
    }

    DEVICE_CODE void operator()(int /*aPixelX*/, int /*aPixelY*/, Tupel<DstT, tupelSize> &aDst)
        requires std::same_as<ComputeT, DstT> && //
                 (tupelSize > 1)
    {
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(aDst.value[i]);
        }
    }
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
