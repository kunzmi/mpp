#pragma once
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <concepts>

#ifndef __restrict__
#define __restrict__
#define myDef
#endif

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace opp::image
{
/// <summary>
/// Computes an output pixel from one srcDst pixel and one src array -> srcDst pixel inplace with float scaling of
/// result
/// </summary>
/// <typeparam name="ComputeType"></typeparam>
/// <typeparam name="ResultType"></typeparam>
/// <typeparam name="SrcType"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename ComputeType, typename ResultType, typename SrcType, typename operation,
          RoudingMode roundingMode = RoudingMode::NearestTiesAwayFromZero>
struct InplaceSrcScaleFunctor
{
    const SrcType *__restrict__ Src1;
    size_t SrcPitch1;

    operation Op;
    float ScaleFactor;

    RoundFunctor<roundingMode, ComputeType> round;

    InplaceSrcScaleFunctor()
    {
    }

    InplaceSrcScaleFunctor(SrcType *aSrc1, size_t aSrcPitch1, operation aOp, float aScaleFactor)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), Op(aOp), ScaleFactor(aScaleFactor)
    {
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, ResultType &aDst)
        requires Integral<typename remove_vector<ResultType>::type> &&
                 FloatingPoint<typename remove_vector<ComputeType>::type>
    {
        const SrcType *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        ComputeType temp(aDst);
        Op(ComputeType(*pixelSrc1), temp);
        temp *= ScaleFactor;
        round(temp);
        // ResultType constructor will clamp temp to value range of ResultType
        aDst = ResultType(temp);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<ResultType, tupelSize> &aDst)
        requires Integral<typename remove_vector<ResultType>::type> &&
                 FloatingPoint<typename remove_vector<ComputeType>::type> && (tupelSize > 1)
    {
        const SrcType *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcType, tupelSize> tupelSrc1 = Tupel<SrcType, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeType temp(aDst.value[i]);
            Op(ComputeType(tupelSrc1.value[i]), temp);
            temp *= ScaleFactor;
            round(temp);
            // ResultType constructor will clamp temp to value range of ResultType
            aDst.value[i] = ResultType(temp);
        }
    }
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>

#ifdef myDef
#undef __restrict__
#undef myDef
#endif // myDef