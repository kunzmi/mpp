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
/// Computes an output pixel from one src array and one constant value -> dst pixel
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename ComputeT, typename DstT, typename operation,
          typename ComputeT_SIMD = void *, typename operation_SIMD = NullOp<void>,
          RoudingMode roundingMode = RoudingMode::NearestTiesAwayFromZero>
struct SrcConstantFunctor : public ImageFunctor<false>
{
    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    ComputeT Constant;
    ComputeT_SIMD ConstantSIMD;

    operation Op;
    operation_SIMD OpSIMD;

    RoundFunctor<roundingMode, ComputeT> round;

    SrcConstantFunctor()
    {
    }

    SrcConstantFunctor(const SrcT *aSrc1, size_t aSrcPitch1, ComputeT aConstant, operation aOp)
        : Op(aOp), Constant(aConstant), Src1(aSrc1), SrcPitch1(aSrcPitch1)
    {
    }

    SrcConstantFunctor(const SrcT *aSrc1, size_t aSrcPitch1, ComputeT aConstant, operation aOp, operation_SIMD aOpSIMD)
        : Op(aOp), OpSIMD(aOpSIMD), Constant(aConstant), Src1(aSrc1), SrcPitch1(aSrcPitch1)
    {
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires std::same_as<ComputeT, DstT>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        Op(ComputeT(*pixelSrc1), Constant, aDst);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires std::same_as<ComputeT, DstT> && //
                 (tupelSize > 1)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(ComputeT(tupelSrc1.value[i]), Constant, aDst.value[i]);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires Integral<pixel_basetype_t<DstT>> && //
                 FloatingPoint<pixel_basetype_t<ComputeT>>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        ComputeT temp;
        Op(ComputeT(*pixelSrc1), Constant, temp);
        round(temp);
        // DstT constructor will clamp temp to value range of DstT
        aDst = DstT(temp);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires Integral<pixel_basetype_t<DstT>> &&          //
                 FloatingPoint<pixel_basetype_t<ComputeT>> && //
                 (tupelSize > 1)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp;
            Op(ComputeT(tupelSrc1.value[i]), Constant, temp);
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
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        ComputeT temp;
        Op(ComputeT(*pixelSrc1), Constant, temp);
        // DstT constructor will clamp temp to value range of DstT
        aDst = DstT(temp);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires(!std::same_as<ComputeT, DstT>) &&      //
                Integral<pixel_basetype_t<DstT>> &&     //
                Integral<pixel_basetype_t<ComputeT>> && //
                (tupelSize > 1)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp;
            Op(ComputeT(tupelSrc1.value[i]), Constant, temp);
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = DstT(temp);
        }
    }
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
