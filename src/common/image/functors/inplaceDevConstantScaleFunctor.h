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
/// Computes an output pixel from one srcDst pixel and one device memory constant value -> srcDst pixel inplace with
/// float scaling of result
/// </summary>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename ComputeT, typename DstT, typename operation,
          RoudingMode roundingMode = RoudingMode::NearestTiesAwayFromZero>
struct InplaceDevConstantScaleFunctor : public ImageFunctor<true>
{
    const DstT *RESTRICT Constant;

    remove_vector_t<ComputeT> ScaleFactor;

    [[no_unique_address]] operation Op;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

    InplaceDevConstantScaleFunctor()
    {
    }

    InplaceDevConstantScaleFunctor(const DstT *aConstant, operation aOp, remove_vector_t<ComputeT> aScaleFactor)
        : Constant(aConstant), ScaleFactor(aScaleFactor), Op(aOp)
    {
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires Integral<pixel_basetype_t<DstT>> && //
                 FloatingPoint<pixel_basetype_t<ComputeT>>
    {
        ComputeT temp(aDst);
        Op(ComputeT(*Constant), temp);
        temp *= ScaleFactor;
        round(temp);
        // DstT constructor will clamp temp to value range of DstT
        aDst = DstT(temp);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires Integral<pixel_basetype_t<DstT>> &&          //
                 FloatingPoint<pixel_basetype_t<ComputeT>> && //
                 (tupelSize > 1)
    {
        ComputeT _constant = *Constant;
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp(aDst.value[i]);
            Op(_constant, temp);
            temp *= ScaleFactor;
            round(temp);
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = DstT(temp);
        }
    }
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
