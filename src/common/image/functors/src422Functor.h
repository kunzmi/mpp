#pragma once
#include "imageFunctors.h"
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/functors/borderControlHorizontal.h>
#include <common/image/functors/interpolatorHorizontal.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/vector_typetraits.h>
#include <common/vector1.h>
#include <common/vectorTypes.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace mpp::image
{
/// <summary>
/// Computes an output pixel from 422 downsampled src arrays -&gt; dst pixel<para/>
/// First channel is Luma channel at full resolution, second and third channel are chroma with half width and full
/// height. Chroma can be either one 2 channel plane or 2 one channel planar planes.
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename ComputeT, typename DstT, typename operation,
          ChromaSubsamplePos chromaSubsamplePos, InterpolationMode interpolationMode, bool swapChroma,
          bool planarChroma, RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZeroPositive>
struct Src422Functor : public ImageFunctor<false>
{
    using Src1Plane     = Vector1<remove_vector_t<SrcT>>;
    using Src2Plane     = Vector2<remove_vector_t<SrcT>>;
    using Compute1Plane = Vector1<remove_vector_t<ComputeT>>;
    using Compute2Plane = Vector2<remove_vector_t<ComputeT>>;

    const Src1Plane *RESTRICT Src1;
    size_t SrcPitch1;

    InterpolatorHorizontal<
        Compute2Plane,
        BorderControlHorizontal<Src2Plane, mpp::BorderType::Replicate, false, false, false, planarChroma>, float,
        interpolationMode>
        Src2;

    [[no_unique_address]] operation Op;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

#pragma region Constructors
    Src422Functor(const Src1Plane *aSrc1, size_t aSrcPitch1, const Src2Plane *aSrc2, size_t aSrcPitch2,
                  const Size2D &aSize2, operation aOp)
        requires(planarChroma == false)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1),
          Src2(BorderControlHorizontal<Src2Plane, mpp::BorderType::Replicate, false, false, false, planarChroma>(
              aSrc2, aSrcPitch2, aSize2.x, 0)),
          Op(aOp)
    {
    }
    Src422Functor(const Src1Plane *aSrc1, size_t aSrcPitch1, const Src1Plane *aSrc21, size_t aSrcPitch21,
                  const Src1Plane *aSrc22, size_t aSrcPitch22, const Size2D &aSize2, operation aOp)
        requires(planarChroma == true)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1),
          Src2(BorderControlHorizontal<Src2Plane, mpp::BorderType::Replicate, false, false, false, planarChroma>(
              aSrc21, aSrcPitch21, aSrc22, aSrcPitch22, aSize2.x, 0)),
          Op(aOp)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
    {
        const Src1Plane *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        float chromaX;
        float chromaY;
        getChromaSampleLocation(aPixelX, aPixelY, chromaX, chromaY);

        const Src2Plane pixelSrc2 = Src2(chromaX, chromaY);

        SrcT pixelSrc;
        pixelSrc.x = pixelSrc1->x;
        if constexpr (swapChroma)
        {
            pixelSrc.y = pixelSrc2.y;
            pixelSrc.z = pixelSrc2.x;
        }
        else
        {
            pixelSrc.y = pixelSrc2.x;
            pixelSrc.z = pixelSrc2.y;
        }

        const ComputeT pixelSrcC(pixelSrc);

        if constexpr (std::same_as<ComputeT, DstT>)
        {
            Op(pixelSrcC, aDst);
        }
        else
        {
            ComputeT temp;
            Op(pixelSrcC, temp);
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
            const Src1Plane *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX + i, aPixelY);

            float chromaX;
            float chromaY;
            getChromaSampleLocation(aPixelX + i, aPixelY, chromaX, chromaY);

            const Src2Plane pixelSrc2 = Src2(chromaX, chromaY);

            SrcT pixelSrc;
            pixelSrc.x = pixelSrc1->x;
            if constexpr (swapChroma)
            {
                pixelSrc.y = pixelSrc2.y;
                pixelSrc.z = pixelSrc2.x;
            }
            else
            {
                pixelSrc.y = pixelSrc2.x;
                pixelSrc.z = pixelSrc2.y;
            }

            const ComputeT pixelSrcC(pixelSrc);

            if constexpr (std::same_as<ComputeT, DstT>)
            {
                Op(pixelSrcC, aDst.value[i]);
            }
            else
            {
                ComputeT temp;
                Op(pixelSrcC, temp);
                round(temp); // NOP for integer ComputeT
                // DstT constructor will clamp temp to value range of DstT
                aDst.value[i] = static_cast<DstT>(temp);
            }
        }
    }
#pragma endregion

  private:
    DEVICE_CODE static void getChromaSampleLocation(int aLumaX, int aLumaY, float &aChromaX, float &aChromaY)
    {
        int iChromaX = aLumaX / 2; // integer division with floor()
        int iChromaY = aLumaY;
        aChromaX     = static_cast<float>(iChromaX);
        aChromaY     = static_cast<float>(iChromaY);

        if constexpr (interpolationMode != InterpolationMode::NearestNeighbor)
        {
            // the pixel index in the 2x1 YUV pixel block:
            int offsetX = aLumaX & 1;

            if constexpr ((chromaSubsamplePos == ChromaSubsamplePos::Undefined ||
                           chromaSubsamplePos == ChromaSubsamplePos::Center))
            {
                aChromaX += 0.5f * static_cast<float>(offsetX) - 0.25f;
            }
            // TopLeft should never occur, but we treat it as Left:
            if constexpr (chromaSubsamplePos == ChromaSubsamplePos::Left ||
                          chromaSubsamplePos == ChromaSubsamplePos::TopLeft)
            {
                aChromaX += 0.5f * static_cast<float>(offsetX);
            }
        }
    }
};
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
