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
/// Computes an output pixel from one srcDst pixel and one src array -> srcDst pixel inplace
/// </summary>
/// <typeparam name="ComputeType"></typeparam>
/// <typeparam name="ResultType"></typeparam>
/// <typeparam name="SrcType"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename ComputeType, typename ResultType, typename SrcType, typename operation,
          RoudingMode roundingMode = RoudingMode::NearestTiesAwayFromZero>
struct InplaceSrcFunctor
{
    const SrcType *__restrict__ Src1;
    size_t SrcPitch1;

    operation Op;

    RoundFunctor<roundingMode, ComputeType> round;

    InplaceSrcFunctor()
    {
    }

    InplaceSrcFunctor(SrcType *aSrc1, size_t aSrcPitch1, operation aOp) : Src1(aSrc1), SrcPitch1(aSrcPitch1), Op(aOp)
    {
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, ResultType &aDst)
        requires std::same_as<ComputeType, ResultType>
    {
        const SrcType *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        Op(ComputeType(*pixelSrc1), aDst);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<ResultType, tupelSize> &aDst)
        requires std::same_as<ComputeType, ResultType> && (tupelSize > 1)
    {
        const SrcType *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcType, tupelSize> tupelSrc1 = Tupel<SrcType, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(ComputeType(tupelSrc1.value[i]), aDst.value[i]);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, ResultType &aDst)
        requires Integral<typename remove_vector<ResultType>::type> &&
                 FloatingPoint<typename remove_vector<ComputeType>::type>
    {
        const SrcType *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        ComputeType temp(aDst);
        Op(ComputeType(*pixelSrc1), temp);
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
            round(temp);
            // ResultType constructor will clamp temp to value range of ResultType
            aDst.value[i] = ResultType(temp);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, ResultType &aDst)
        requires(!std::same_as<ComputeType, ResultType>) && Integral<typename remove_vector<ResultType>::type> &&
                Integral<typename remove_vector<ComputeType>::type>
    {
        const SrcType *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        ComputeType temp(aDst);
        Op(ComputeType(*pixelSrc1), temp);
        // ResultType constructor will clamp temp to value range of ResultType
        aDst = ResultType(temp);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<ResultType, tupelSize> &aDst)
        requires(!std::same_as<ComputeType, ResultType>) && Integral<typename remove_vector<ResultType>::type> &&
                Integral<typename remove_vector<ComputeType>::type> && (tupelSize > 1)
    {
        const SrcType *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcType, tupelSize> tupelSrc1 = Tupel<SrcType, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeType temp(aDst.value[i]);
            Op(ComputeType(tupelSrc1.value[i]), temp);
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