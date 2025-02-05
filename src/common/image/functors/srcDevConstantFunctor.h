#pragma once
#include "imageFunctors.h"
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
#include <common/opp_defs.h>
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
/// Computes an output pixel from one src array and one device memory constant value -&gt; dst pixel
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename ComputeT, typename DstT, typename operation,
          RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero, bool SrcIsSameAsCompute = false>
struct SrcDevConstantFunctor : public ImageFunctor<false>
{
    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    const SrcT *RESTRICT Constant;

    [[no_unique_address]] operation Op;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

#pragma region Constructors
    SrcDevConstantFunctor()
    {
    }

    SrcDevConstantFunctor(const SrcT *aSrc1, size_t aSrcPitch1, const SrcT *aConstant, operation aOp)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), Constant(aConstant), Op(aOp)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires std::same_as<ComputeT, DstT> && (!SrcIsSameAsCompute)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        Op(static_cast<ComputeT>(*pixelSrc1), static_cast<ComputeT>(*Constant), aDst);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires std::same_as<SrcT, ComputeT> && (SrcIsSameAsCompute)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        Op(static_cast<ComputeT>(*pixelSrc1), static_cast<ComputeT>(*Constant), aDst);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires(!std::same_as<ComputeT, DstT>) && (!SrcIsSameAsCompute)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        ComputeT temp;
        Op(static_cast<ComputeT>(*pixelSrc1), static_cast<ComputeT>(*Constant), temp);
        round(temp); // NOP for integer ComputeT
        // DstT constructor will clamp temp to value range of DstT
        aDst = static_cast<DstT>(temp);
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires std::same_as<ComputeT, DstT> && (!SrcIsSameAsCompute)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

        ComputeT _constant = static_cast<ComputeT>(*Constant);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(static_cast<ComputeT>(tupelSrc1.value[i]), _constant, aDst.value[i]);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires std::same_as<SrcT, ComputeT> && (SrcIsSameAsCompute)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

        ComputeT _constant = static_cast<ComputeT>(*Constant);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(static_cast<ComputeT>(tupelSrc1.value[i]), _constant, aDst.value[i]);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires(!std::same_as<ComputeT, DstT>) && (!SrcIsSameAsCompute)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

        ComputeT _constant = static_cast<ComputeT>(*Constant);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp;
            Op(static_cast<ComputeT>(tupelSrc1.value[i]), _constant, temp);
            round(temp); // NOP for integer ComputeT
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }
#pragma endregion
};

} // namespace opp::image
#include <common/disableWarningsEnd.h>
