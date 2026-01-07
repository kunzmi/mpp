#pragma once
#include "imageFunctors.h"
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace mpp::image
{

enum class Src422C2Layout
{
    YCrCb,
    YCbCr,
    CbYCr,
    CrYCb // this variant is technically possible but doesn't seem to exist in real world scenarios.
};

/// <summary>
/// Computes an output pixel from 422 downsampled src array -&gt; dst pixel<para/>
/// First channel is Luma channel at full resolution, second and third channel are chroma with half width and full
/// height. Chroma is interleaved at every second pixel in the second channel (YCbCr/YCrCb/CbYCr/CrYCb).
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename ComputeT, typename DstT, typename operation,
          Src422C2Layout pixelLayout, RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero>
struct Src422C2Functor : public ImageFunctor<false>
{
    using SrcPlane     = Vector2<remove_vector_t<SrcT>>;
    using ComputePlane = Vector2<remove_vector_t<ComputeT>>;
    const SrcPlane *RESTRICT Src1;
    size_t SrcPitch1;

    [[no_unique_address]] operation Op;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

#pragma region Constructors
    Src422C2Functor()
    {
    }

    Src422C2Functor(const SrcPlane *aSrc1, size_t aSrcPitch1, operation aOp)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), Op(aOp)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
    {
        ComputeT pixel;
        const SrcPlane *pixelSrcY = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        int cbX, crX;
        getChromaSampleLocation(aPixelX, cbX, crX);
        const SrcPlane *pixelSrcCb = gotoPtr(Src1, SrcPitch1, cbX, aPixelY);
        const SrcPlane *pixelSrcCr = gotoPtr(Src1, SrcPitch1, crX, aPixelY);

        if constexpr (pixelLayout == Src422C2Layout::YCbCr || pixelLayout == Src422C2Layout::YCrCb)
        {
            pixel = static_cast<ComputeT>(SrcT(pixelSrcY->x, pixelSrcCb->y, pixelSrcCr->y));
        }
        if constexpr (pixelLayout == Src422C2Layout::CrYCb || pixelLayout == Src422C2Layout::CbYCr)
        {
            pixel = static_cast<ComputeT>(SrcT(pixelSrcY->y, pixelSrcCb->x, pixelSrcCr->x));
        }

        if constexpr (std::same_as<ComputeT, DstT>)
        {
            Op(pixel, aDst);
        }
        else
        {
            ComputeT temp;
            Op(pixel, temp);
            round(temp); // NOP for integer ComputeT
            // DstT constructor will clamp temp to value range of DstT
            aDst = static_cast<DstT>(temp);
        }
        return true;
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
    {
#pragma unroll
        for (int i = 0; i < static_cast<int>(tupelSize); i++)
        {
            ComputeT pixel;
            const SrcPlane *pixelSrcY = gotoPtr(Src1, SrcPitch1, aPixelX + i, aPixelY);

            int cbX, crX;
            getChromaSampleLocation(aPixelX + i, cbX, crX);
            const SrcPlane *pixelSrcCb = gotoPtr(Src1, SrcPitch1, cbX, aPixelY);
            const SrcPlane *pixelSrcCr = gotoPtr(Src1, SrcPitch1, crX, aPixelY);

            if constexpr (pixelLayout == Src422C2Layout::YCbCr || pixelLayout == Src422C2Layout::YCrCb)
            {
                pixel = static_cast<ComputeT>(SrcT(pixelSrcY->x, pixelSrcCb->y, pixelSrcCr->y));
            }
            if constexpr (pixelLayout == Src422C2Layout::CrYCb || pixelLayout == Src422C2Layout::CbYCr)
            {
                pixel = static_cast<ComputeT>(SrcT(pixelSrcY->y, pixelSrcCb->x, pixelSrcCr->x));
            }

            if constexpr (std::same_as<ComputeT, DstT>)
            {
                Op(pixel, aDst.value[i]);
            }
            else
            {
                ComputeT temp;
                Op(pixel, temp);
                round(temp); // NOP for integer ComputeT
                // DstT constructor will clamp temp to value range of DstT
                aDst.value[i] = static_cast<DstT>(temp);
            }
        }
    }
#pragma endregion
  private:
    DEVICE_CODE static void getChromaSampleLocation(int aLumaX, int &aCbX, int &aCrX)
    {
        int offsetX  = aLumaX & 1;
        int iChromaX = aLumaX - offsetX;

        if constexpr (pixelLayout == Src422C2Layout::CbYCr || pixelLayout == Src422C2Layout::YCbCr)
        {
            aCbX = iChromaX;
            aCrX = iChromaX + 1;
        }

        if constexpr (pixelLayout == Src422C2Layout::CrYCb || pixelLayout == Src422C2Layout::YCrCb)
        {
            aCbX = iChromaX + 1;
            aCrX = iChromaX;
        }
    }
};
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
