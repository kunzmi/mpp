#pragma once
#include "imageFunctors.h"
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/maskTupel.h>
#include <common/numberTypes.h>
#include <common/opp_defs.h>
#include <common/roundFunctor.h>
#include <common/statistics/operators.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace opp::image
{
/// <summary>
/// Performs a reduction operation with two input images and one output values
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename DstT, typename operation> struct SrcSrcReductionFunctor
{
    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    const SrcT *RESTRICT Src2;
    size_t SrcPitch2;

    [[no_unique_address]] operation Op;

#pragma region Constructors
    SrcSrcReductionFunctor()
    {
    }

    SrcSrcReductionFunctor(const SrcT *aSrc1, size_t aSrcPitch1, const SrcT *aSrc2, size_t aSrcPitch2, operation aOp)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), Src2(aSrc2), SrcPitch2(aSrcPitch2), Op(aOp)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
    {
        const SrcT pixelSrc1 = *gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcT pixelSrc2 = *gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);
        Op(pixelSrc1, pixelSrc2, aDst);
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst, bool /*isTupel*/)
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
            Op(tupelSrc1.value[i], tupelSrc2.value[i], aDst);
        }
    }
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst, const MaskTupel<tupelSize> &aMaskTupel,
                                int &maskCount)
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
                Op(tupelSrc1.value[i], tupelSrc2.value[i], aDst);
                maskCount++;
            }
        }
    }
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst, const MaskTupel<tupelSize> &aMaskTupel)
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
                Op(tupelSrc1.value[i], tupelSrc2.value[i], aDst);
            }
        }
    }
#pragma endregion
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
