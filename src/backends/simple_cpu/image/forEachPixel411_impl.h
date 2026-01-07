#pragma once
#include "forEachPixel411.h"
#include <backends/cuda/cudaException.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/image/channel.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vector1.h>
#include <iostream>

namespace mpp::image::cpuSimple
{
/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 2 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixel411(ImageView<Vector1<remove_vector_t<DstT>>> &aDst1, ImageView<Vector2<remove_vector_t<DstT>>> &aDst2,
                     ChromaSubsamplePos aChromaSubsamplePos, const functor &aFunctor)
{
    using DstPlaneLumaT         = Vector1<remove_vector_t<DstT>>;
    using DstPlaneChromaT       = Vector2<remove_vector_t<DstT>>;
    constexpr int subSampleSize = 4;

    const Size2D sizeLoop = aDst1.SizeRoi() / Vec2i(subSampleSize, 1);

    for (const auto &elem : sizeLoop)
    {
        const int pixelX       = elem.Pixel.x * subSampleSize;
        const int pixelXChroma = elem.Pixel.x;
        const int pixelY       = elem.Pixel.y;

        DstT res[subSampleSize];
        DstPlaneLumaT *pixelOutLuma     = gotoPtr(aDst1.PointerRoi(), aDst1.Pitch(), pixelX, pixelY);
        DstPlaneChromaT *pixelOutChroma = gotoPtr(aDst2.PointerRoi(), aDst2.Pitch(), pixelXChroma, pixelY);

        for (int i = 0; i < subSampleSize; i++)
        {
            // ignore functor result, as only transformerFunctor can return false and they are not used in chroma
            // sub-sampled kernels
            aFunctor(pixelX + i, pixelY, res[i]);
            *(pixelOutLuma + i) = res[i].x;
        }

        DstPlaneChromaT resChroma;
        if (aChromaSubsamplePos == ChromaSubsamplePos::Center)
        {
            // average chroma values:
            if constexpr (RealFloatingVector<DstT>)
            {
                resChroma = res[0].YZ();
                resChroma += res[1].YZ();
                resChroma += res[2].YZ();
                resChroma += res[3].YZ();
                resChroma /= static_cast<remove_vector_t<DstT>>(subSampleSize);
            }
            else
            {
                Vector2<int> temp = static_cast<Vector2<int>>(res[0].YZ());
                temp += static_cast<Vector2<int>>(res[1].YZ());
                temp += static_cast<Vector2<int>>(res[2].YZ());
                temp += static_cast<Vector2<int>>(res[3].YZ());
                temp.DivScaleRoundZero(subSampleSize); // simple integer division, same as in NPP
                resChroma = static_cast<DstPlaneChromaT>(temp);
            }
        }
        else // CenterLeft is treated as Left
        {
            resChroma = res[0].YZ();
        }
        *pixelOutChroma = resChroma;
    }
}

/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 3 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixel411(ImageView<Vector1<remove_vector_t<DstT>>> &aDst1, ImageView<Vector1<remove_vector_t<DstT>>> &aDst2,
                     ImageView<Vector1<remove_vector_t<DstT>>> &aDst3, ChromaSubsamplePos aChromaSubsamplePos,
                     const functor &aFunctor)
{
    using DstPlane              = Vector1<remove_vector_t<DstT>>;
    constexpr int subSampleSize = 4;

    const Size2D sizeLoop = aDst1.SizeRoi() / Vec2i(subSampleSize, 1);

    for (const auto &elem : sizeLoop)
    {
        const int pixelX       = elem.Pixel.x * subSampleSize;
        const int pixelXChroma = elem.Pixel.x;
        const int pixelY       = elem.Pixel.y;

        DstT res[subSampleSize];
        DstPlane *pixelOutLuma    = gotoPtr(aDst1.PointerRoi(), aDst1.Pitch(), pixelX, pixelY);
        DstPlane *pixelOutChroma1 = gotoPtr(aDst2.PointerRoi(), aDst2.Pitch(), pixelXChroma, pixelY);
        DstPlane *pixelOutChroma2 = gotoPtr(aDst3.PointerRoi(), aDst3.Pitch(), pixelXChroma, pixelY);

        for (int i = 0; i < subSampleSize; i++)
        {
            // ignore functor result, as only transformerFunctor can return false and they are not used in chroma
            // sub-sampled kernels
            aFunctor(pixelX + i, pixelY, res[i]);
            *(pixelOutLuma + i) = res[i].x;
        }

        DstPlane resChroma1;
        DstPlane resChroma2;
        if (aChromaSubsamplePos == ChromaSubsamplePos::Center)
        {
            // average chroma values:
            if constexpr (RealFloatingVector<DstT>)
            {
                resChroma1.x = res[0].y;
                resChroma1.x += res[1].y;
                resChroma1.x += res[2].y;
                resChroma1.x += res[3].y;
                resChroma1.x /= static_cast<remove_vector_t<DstT>>(subSampleSize);

                resChroma2.x = res[0].z;
                resChroma2.x += res[1].z;
                resChroma2.x += res[2].z;
                resChroma2.x += res[3].z;
                resChroma2.x /= static_cast<remove_vector_t<DstT>>(subSampleSize);
            }
            else
            {
                int temp = static_cast<int>(res[0].y);
                temp += static_cast<int>(res[1].y);
                temp += static_cast<int>(res[2].y);
                temp += static_cast<int>(res[3].y);
                temp /= subSampleSize; // simple integer division, same as in NPP
                resChroma1.x = static_cast<remove_vector_t<DstPlane>>(temp);

                temp = static_cast<int>(res[0].z);
                temp += static_cast<int>(res[1].z);
                temp += static_cast<int>(res[2].z);
                temp += static_cast<int>(res[3].z);
                temp /= subSampleSize; // simple integer division, same as in NPP
                resChroma2.x = static_cast<remove_vector_t<DstPlane>>(temp);
            }
        }
        else // CenterLeft is treated as Left
        {
            resChroma1.x = res[0].y;
            resChroma2.x = res[0].z;
        }
        *pixelOutChroma1 = resChroma1;
        *pixelOutChroma2 = resChroma2;
    }
}

} // namespace mpp::image::cpuSimple