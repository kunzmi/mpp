#pragma once
#include "forEachPixel420.h"
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
void forEachPixel420(ImageView<Vector1<remove_vector_t<DstT>>> &aDst1, ImageView<Vector2<remove_vector_t<DstT>>> &aDst2,
                     ChromaSubsamplePos aChromaSubsamplePos, const functor &aFunctor)
{
    using DstPlaneLumaT         = Vector1<remove_vector_t<DstT>>;
    using DstPlaneChromaT       = Vector2<remove_vector_t<DstT>>;
    constexpr int subSampleSize = 2;

    const Size2D sizeLoop = aDst1.SizeRoi() / Vec2i(subSampleSize, subSampleSize);

    for (const auto &elem : sizeLoop)
    {
        const int pixelX       = elem.Pixel.x * subSampleSize;
        const int pixelXChroma = elem.Pixel.x;
        const int pixelY       = elem.Pixel.y * subSampleSize;
        const int pixelYChroma = elem.Pixel.x;

        DstT res[subSampleSize * subSampleSize];
        DstPlaneLumaT *pixelOutLuma1    = gotoPtr(aDst1.PointerRoi(), aDst1.Pitch(), pixelX, pixelY);
        DstPlaneLumaT *pixelOutLuma2    = gotoPtr(aDst1.PointerRoi(), aDst1.Pitch(), pixelX, pixelY + 1);
        DstPlaneChromaT *pixelOutChroma = gotoPtr(aDst2.PointerRoi(), aDst2.Pitch(), pixelXChroma, pixelYChroma);

        DstPlaneLumaT resLuma[subSampleSize][subSampleSize];

        for (int sy = 0; sy < subSampleSize; sy++)
        {
#pragma unroll
            for (int sx = 0; sx < subSampleSize; sx++)
            {
                // ignore functor result, as only transformerFunctor can return false and they are not used in chroma
                // sub-sampled kernels
                aFunctor(pixelX + sx, pixelY + sy, res[sy * subSampleSize + sx]);
                resLuma[sy][sx].x = res[sy * subSampleSize + sx].x;
            }
        }

        *(pixelOutLuma1 + 0) = resLuma[0][0];
        *(pixelOutLuma1 + 1) = resLuma[0][1];
        *(pixelOutLuma2 + 0) = resLuma[1][0];
        *(pixelOutLuma2 + 1) = resLuma[1][1];

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
                resChroma /= static_cast<remove_vector_t<DstT>>(subSampleSize * subSampleSize);
            }
            else
            {
                Vector2<int> temp = static_cast<Vector2<int>>(res[0].YZ());
                temp += static_cast<Vector2<int>>(res[1].YZ());
                temp += static_cast<Vector2<int>>(res[2].YZ());
                temp += static_cast<Vector2<int>>(res[3].YZ());
                temp.DivScaleRoundZero(subSampleSize * subSampleSize); // simple integer division, same as in NPP
                resChroma = static_cast<DstPlaneChromaT>(temp);
            }
        }
        else if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
        {
            // average chroma values:
            if constexpr (RealFloatingVector<DstT>)
            {
                resChroma = res[0].YZ();
                resChroma += res[2].YZ();
                resChroma /= static_cast<remove_vector_t<DstT>>(subSampleSize);
            }
            else
            {
                Vector2<int> temp = static_cast<Vector2<int>>(res[0].YZ());
                temp += static_cast<Vector2<int>>(res[2].YZ());
                temp.DivScaleRoundZero(subSampleSize); // simple integer division, same as in NPP
                resChroma = static_cast<DstPlaneChromaT>(temp);
            }
        }
        else // TopLeft
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
void forEachPixel420(ImageView<Vector1<remove_vector_t<DstT>>> &aDst1, ImageView<Vector1<remove_vector_t<DstT>>> &aDst2,
                     ImageView<Vector1<remove_vector_t<DstT>>> &aDst3, ChromaSubsamplePos aChromaSubsamplePos,
                     const functor &aFunctor)
{
    using DstPlane              = Vector1<remove_vector_t<DstT>>;
    constexpr int subSampleSize = 2;

    const Size2D sizeLoop = aDst1.SizeRoi() / Vec2i(subSampleSize, subSampleSize);

    for (const auto &elem : sizeLoop)
    {
        const int pixelX       = elem.Pixel.x * subSampleSize;
        const int pixelXChroma = elem.Pixel.x;
        const int pixelY       = elem.Pixel.y * subSampleSize;
        const int pixelYChroma = elem.Pixel.x;

        DstT res[subSampleSize * subSampleSize];
        DstPlane *pixelOutLuma1   = gotoPtr(aDst1.PointerRoi(), aDst1.Pitch(), pixelX, pixelY);
        DstPlane *pixelOutLuma2   = gotoPtr(aDst1.PointerRoi(), aDst1.Pitch(), pixelX, pixelY + 1);
        DstPlane *pixelOutChroma1 = gotoPtr(aDst2.PointerRoi(), aDst2.Pitch(), pixelXChroma, pixelYChroma);
        DstPlane *pixelOutChroma2 = gotoPtr(aDst3.PointerRoi(), aDst3.Pitch(), pixelXChroma, pixelYChroma);

        DstPlane resLuma[subSampleSize][subSampleSize];

        for (int sy = 0; sy < subSampleSize; sy++)
        {
#pragma unroll
            for (int sx = 0; sx < subSampleSize; sx++)
            {
                // ignore functor result, as only transformerFunctor can return false and they are not used in chroma
                // sub-sampled kernels
                aFunctor(pixelX + sx, pixelY + sy, res[sy * subSampleSize + sx]);
                resLuma[sy][sx].x = res[sy * subSampleSize + sx].x;
            }
        }

        *(pixelOutLuma1 + 0) = resLuma[0][0];
        *(pixelOutLuma1 + 1) = resLuma[0][1];
        *(pixelOutLuma2 + 0) = resLuma[1][0];
        *(pixelOutLuma2 + 1) = resLuma[1][1];

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
                resChroma1.x /= static_cast<remove_vector_t<DstT>>(subSampleSize * subSampleSize);

                resChroma2.x = res[0].z;
                resChroma2.x += res[1].z;
                resChroma2.x += res[2].z;
                resChroma2.x += res[3].z;
                resChroma2.x /= static_cast<remove_vector_t<DstT>>(subSampleSize * subSampleSize);
            }
            else
            {
                int temp = static_cast<int>(res[0].y);
                temp += static_cast<int>(res[1].y);
                temp += static_cast<int>(res[2].y);
                temp += static_cast<int>(res[3].y);
                temp /= subSampleSize * subSampleSize; // simple integer division, same as in NPP
                resChroma1.x = static_cast<remove_vector_t<DstPlane>>(temp);

                temp = static_cast<int>(res[0].z);
                temp += static_cast<int>(res[1].z);
                temp += static_cast<int>(res[2].z);
                temp += static_cast<int>(res[3].z);
                temp /= subSampleSize * subSampleSize; // simple integer division, same as in NPP
                resChroma2.x = static_cast<remove_vector_t<DstPlane>>(temp);
            }
        }
        else if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
        {
            // average chroma values:
            if constexpr (RealFloatingVector<DstT>)
            {
                resChroma1.x = res[0].y;
                resChroma1.x += res[2].y;
                resChroma1.x /= static_cast<remove_vector_t<DstT>>(subSampleSize);

                resChroma2.x = res[0].z;
                resChroma2.x += res[2].z;
                resChroma2.x /= static_cast<remove_vector_t<DstT>>(subSampleSize);
            }
            else
            {
                int temp = static_cast<int>(res[0].y);
                temp += static_cast<int>(res[2].y);
                temp /= subSampleSize; // simple integer division, same as in NPP
                resChroma1.x = static_cast<remove_vector_t<DstPlane>>(temp);

                temp = static_cast<int>(res[0].z);
                temp += static_cast<int>(res[2].z);
                temp /= subSampleSize; // simple integer division, same as in NPP
                resChroma2.x = static_cast<remove_vector_t<DstPlane>>(temp);
            }
        }
        else // TopLeft
        {
            resChroma1.x = res[0].y;
            resChroma2.x = res[0].z;
        }
        *pixelOutChroma1 = resChroma1;
        *pixelOutChroma2 = resChroma2;
    }
}
} // namespace mpp::image::cpuSimple