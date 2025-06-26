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
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace mpp::image
{
/// <summary>
/// Computes an output pixel from one src array -&gt; dst pixel
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT> struct SrcReductionMinIdxFunctor
{
    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    [[no_unique_address]] mpp::MinIdx<SrcT> OpMin;

#pragma region Constructors
    SrcReductionMinIdxFunctor()
    {
    }

    DEVICE_CODE SrcReductionMinIdxFunctor(const SrcT *aSrc1, size_t aSrcPitch1) : Src1(aSrc1), SrcPitch1(aSrcPitch1)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, SrcT &aDstMin,
                                same_vector_size_different_type_t<SrcT, int> &aIdxXMin) const
    {
        const SrcT pixelSrc1 = *gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        OpMin(pixelSrc1, aPixelX, aDstMin, aIdxXMin);
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, SrcT &aDstMin,
                                same_vector_size_different_type_t<SrcT, int> &aIdxXMin, bool /*isTupel*/) const
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        // we know for src1 that it is aligned to tupels, no need to check:
        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::LoadAligned(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            OpMin(tupelSrc1.value[i], aPixelX + static_cast<int>(i), aDstMin, aIdxXMin);
        }
    }
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, SrcT &aDstMin,
                                same_vector_size_different_type_t<SrcT, int> &aIdxXMin,
                                const MaskTupel<tupelSize> &aMaskTupel) const
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        // we know for src1 that it is aligned to tupels, no need to check:
        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::LoadAligned(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            if (aMaskTupel.value[i])
            {
                OpMin(tupelSrc1.value[i], aPixelX + static_cast<int>(i), aDstMin, aIdxXMin);
            }
        }
    }
#pragma endregion
};
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
