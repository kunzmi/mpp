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
/// Computes an output pixel from one srcDst pixel and one constant value -> srcDst pixel inplace
/// </summary>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename ComputeT, typename DstT, typename operation, typename ComputeT_SIMD = ComputeT,
          typename operation_SIMD = NullOp<void>, RoudingMode roundingMode = RoudingMode::NearestTiesAwayFromZero>
struct InplaceConstantFunctor : public ImageFunctor<true>
{
    ComputeT Constant;
    ComputeT_SIMD ConstantSIMD;

    operation Op;
    operation_SIMD OpSIMD;

    RoundFunctor<roundingMode, ComputeT> round;

    InplaceConstantFunctor()
    {
    }

    InplaceConstantFunctor(ComputeT aConstant, operation aOp) : Op(aOp), Constant(aConstant)
    {
    }

    InplaceConstantFunctor(ComputeT aConstant, operation aOp, operation_SIMD aOpSIMD)
        : Op(aOp), OpSIMD(aOpSIMD), Constant(aConstant)
    {
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires Integral<pixel_basetype_t<DstT>> && //
                 FloatingPoint<pixel_basetype_t<ComputeT>>
    {
        ComputeT temp(aDst);
        Op(Constant, temp);
        round(temp);
        // DstT constructor will clamp temp to value range of DstT
        aDst = DstT(temp);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires Integral<pixel_basetype_t<DstT>> &&          //
                 FloatingPoint<pixel_basetype_t<ComputeT>> && //
                 (tupelSize > 1)
    {
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp(aDst.value[i]);
            Op(Constant, temp);
            round(temp);
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = DstT(temp);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires(!std::same_as<ComputeT, DstT>) &&  //
                Integral<pixel_basetype_t<DstT>> && //
                Integral<pixel_basetype_t<ComputeT>>
    {
        ComputeT temp(aDst);
        Op(Constant, temp);
        // DstT constructor will clamp temp to value range of DstT
        aDst = DstT(temp);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires(!std::same_as<ComputeT, DstT>) &&      //
                Integral<pixel_basetype_t<DstT>> &&     //
                Integral<pixel_basetype_t<ComputeT>> && //
                (tupelSize > 1)
    {
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp(aDst.value[i]);
            Op(Constant, temp);
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = DstT(temp);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires std::same_as<ComputeT, DstT>
    {
        Op(Constant, aDst);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires std::same_as<ComputeT, DstT> && //
                 (tupelSize > 1)
    {
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(Constant, aDst.value[i]);
        }
    }
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
