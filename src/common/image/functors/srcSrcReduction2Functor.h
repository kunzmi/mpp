#pragma once
#include "imageFunctors.h"
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/maskTupel.h>
#include <common/numberTypes.h>
#include <common/mpp_defs.h>
#include <common/roundFunctor.h>
#include <common/statistics/operators.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace mpp::image
{
/// <summary>
/// Performs a reduction operation with two input images and two seperate output values
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename DstT1, typename DstT2, typename operation1, typename operation2>
struct SrcSrcReduction2Functor
{
    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    const SrcT *RESTRICT Src2;
    size_t SrcPitch2;

    [[no_unique_address]] operation1 Op1;
    [[no_unique_address]] operation2 Op2;

#pragma region Constructors
    SrcSrcReduction2Functor()
    {
    }

    SrcSrcReduction2Functor(const SrcT *aSrc1, size_t aSrcPitch1, const SrcT *aSrc2, size_t aSrcPitch2, operation1 aOp1,
                            operation2 aOp2)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), Src2(aSrc2), SrcPitch2(aSrcPitch2), Op1(aOp1), Op2(aOp2)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT1 &aDst1, DstT2 &aDst2) const
    {
        const SrcT pixelSrc1 = *gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcT pixelSrc2 = *gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);
        Op1(pixelSrc1, pixelSrc2, aDst1);
        Op2(pixelSrc1, pixelSrc2, aDst2);
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT1 &aDst1, DstT2 &aDst2, bool /*isTupel*/) const
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcT *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);

        // we know for src1 that it is aligned to tupels, no need to check:
        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::LoadAligned(pixelSrc1);

        // but we don't know if src2 is also aligned to tupels, so do the check:
        Tupel<SrcT, tupelSize> tupelSrc2 = Tupel<SrcT, tupelSize>::Load(pixelSrc2);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op1(tupelSrc1.value[i], tupelSrc2.value[i], aDst1);
            Op2(tupelSrc1.value[i], tupelSrc2.value[i], aDst2);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT1 &aDst1, DstT2 &aDst2,
                                const MaskTupel<tupelSize> &aMaskTupel, int &maskCount) const
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcT *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);

        // we know for src1 that it is aligned to tupels, no need to check:
        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::LoadAligned(pixelSrc1);

        // but we don't know if src2 is also aligned to tupels, so do the check:
        Tupel<SrcT, tupelSize> tupelSrc2 = Tupel<SrcT, tupelSize>::Load(pixelSrc2);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            if (aMaskTupel.value[i])
            {
                Op1(tupelSrc1.value[i], tupelSrc2.value[i], aDst1);
                Op2(tupelSrc1.value[i], tupelSrc2.value[i], aDst2);
                maskCount++;
            }
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT1 &aDst1, DstT2 &aDst2,
                                const MaskTupel<tupelSize> &aMaskTupel) const
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcT *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);

        // we know for src1 that it is aligned to tupels, no need to check:
        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::LoadAligned(pixelSrc1);

        // but we don't know if src2 is also aligned to tupels, so do the check:
        Tupel<SrcT, tupelSize> tupelSrc2 = Tupel<SrcT, tupelSize>::Load(pixelSrc2);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            if (aMaskTupel.value[i])
            {
                Op1(tupelSrc1.value[i], tupelSrc2.value[i], aDst1);
                Op2(tupelSrc1.value[i], tupelSrc2.value[i], aDst2);
            }
        }
    }
#pragma endregion
};
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
