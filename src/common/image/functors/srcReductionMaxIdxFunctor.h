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
template <size_t tupelSize, typename SrcT> struct SrcReductionMaxIdxFunctor
{
    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    [[no_unique_address]] mpp::MaxIdx<SrcT> OpMax;

#pragma region Constructors
    SrcReductionMaxIdxFunctor()
    {
    }

    DEVICE_CODE SrcReductionMaxIdxFunctor(const SrcT *aSrc1, size_t aSrcPitch1) : Src1(aSrc1), SrcPitch1(aSrcPitch1)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, SrcT &aDstMax,
                                same_vector_size_different_type_t<SrcT, int> &aIdxXMax) const
    {
        const SrcT pixelSrc1 = *gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        OpMax(pixelSrc1, aPixelX, aDstMax, aIdxXMax);
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, SrcT &aDstMax,
                                same_vector_size_different_type_t<SrcT, int> &aIdxXMax, bool /*isTupel*/) const
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        // we know for src1 that it is aligned to tupels, no need to check:
        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::LoadAligned(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            OpMax(tupelSrc1.value[i], aPixelX + static_cast<int>(i), aDstMax, aIdxXMax);
        }
    }
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, SrcT &aDstMax,
                                same_vector_size_different_type_t<SrcT, int> &aIdxXMax,
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
                OpMax(tupelSrc1.value[i], aPixelX + static_cast<int>(i), aDstMax, aIdxXMax);
            }
        }
    }
#pragma endregion
};
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
