#pragma once
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace opp::image
{
/// <summary>
/// Computes an output pixel from one srcDst pixel and one device memory constant value -> srcDst pixel inplace
/// </summary>
/// <typeparam name="ComputeType"></typeparam>
/// <typeparam name="ResultType"></typeparam>
/// <typeparam name="SrcType"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename ComputeType, typename ResultType, typename SrcType, typename operation,
          RoudingMode roundingMode = RoudingMode::NearestTiesAwayFromZero>
struct InplaceDevConstantFunctor
{
    ComputeType *Constant;
    operation Op;

    RoundFunctor<roundingMode, ComputeType> round;

    InplaceDevConstantFunctor()
    {
    }

    InplaceDevConstantFunctor(ComputeType *aConstant, operation aOp) : Constant(aConstant), Op(aOp)
    {
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, ResultType &aDst)
        requires Integral<typename remove_vector<ResultType>::type> &&
                 FloatingPoint<typename remove_vector<ComputeType>::type>
    {
        ComputeType temp(aDst);
        Op(*Constant, temp);
        round(temp);
        // ResultType constructor will clamp temp to value range of ResultType
        aDst = ResultType(temp);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<ResultType, tupelSize> &aDst)
        requires Integral<typename remove_vector<ResultType>::type> &&
                 FloatingPoint<typename remove_vector<ComputeType>::type> && (tupelSize > 1)
    {
        ComputeType _constant = *Constant;
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeType temp(aDst.value[i]);
            Op(_constant, temp);
            round(temp);
            // ResultType constructor will clamp temp to value range of ResultType
            aDst.value[i] = ResultType(temp);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, ResultType &aDst)
        requires(!std::same_as<ComputeType, ResultType>) && Integral<typename remove_vector<ResultType>::type> &&
                Integral<typename remove_vector<ComputeType>::type>
    {
        ComputeType temp(aDst);
        Op(*Constant, temp);
        // ResultType constructor will clamp temp to value range of ResultType
        aDst = ResultType(temp);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<ResultType, tupelSize> &aDst)
        requires(!std::same_as<ComputeType, ResultType>) && Integral<typename remove_vector<ResultType>::type> &&
                Integral<typename remove_vector<ComputeType>::type> && (tupelSize > 1)
    {
        ComputeType _constant = *Constant;
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeType temp(aDst.value[i]);
            Op(_constant, temp);
            // ResultType constructor will clamp temp to value range of ResultType
            aDst.value[i] = ResultType(temp);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, ResultType &aDst)
        requires std::same_as<ComputeType, ResultType>
    {
        Op(*Constant, aDst);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<ResultType, tupelSize> &aDst)
        requires std::same_as<ComputeType, ResultType> && (tupelSize > 1)
    {
        ComputeType _constant = *Constant;
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(_constant, aDst.value[i]);
        }
    }
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>