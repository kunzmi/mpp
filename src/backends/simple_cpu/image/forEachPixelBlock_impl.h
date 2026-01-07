#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <common/utilities.h>

#include <iostream>

namespace mpp::image::cpuSimple
{
/// <summary>
/// runs aOp on every pixel of an image. Inplace and outplace operation, no mask.
/// </summary>
template <typename DstT, typename functor> void forEachPixelBlock(ImageView<DstT> &aDst, const functor &aFunctor)
{
    for (int y = 0; y < aDst.HeightRoi(); y += 2)
    {
        int pixelY = y + aDst.ROI().y;
        for (int x = 0; x < aDst.WidthRoi(); x += 2)
        {
            int pixelX = x + aDst.ROI().x;

            // will be optimized away as unused in case of no alpha channel:
            pixel_basetype_t<DstT> alphaChannel[4];

            DstT res[4];
            DstT *pixelOut0 = gotoPtr(aDst.PointerRoi(), aDst.Pitch(), pixelX, pixelY);
            DstT *pixelOut1 = pixelOut0 + 1;
            DstT *pixelOut2 = gotoPtr(aDst.PointerRoi(), aDst.Pitch(), pixelX, pixelY + 1);
            DstT *pixelOut3 = pixelOut2 + 1;

            // load the destination pixel in case of inplace operation or we load the full pixel for alpha operations:
            if constexpr (functor::DoLoadBeforeOp || //
                          (has_alpha_channel_v<DstT> && load_full_vector_for_alpha_v<DstT>))
            {
                res[0] = *pixelOut0;
                res[1] = *pixelOut1;
                res[2] = *pixelOut2;
                res[3] = *pixelOut3;

                // save alpha channel value seperatly:
                if constexpr (has_alpha_channel_v<DstT>)
                {
                    alphaChannel[0] = res[0].w;
                    alphaChannel[1] = res[1].w;
                    alphaChannel[2] = res[2].w;
                    alphaChannel[3] = res[3].w;
                }
            }

            // for nearly all functors aOp will evaluate to constant true.
            // Only transformerFunctor is capable of returning false if a source pixel is outside of the src-roi
            if (aFunctor(pixelX, pixelY, res))
            {
                // if we don't load the pixel anyhow but we still need just the alpha channel, load it:
                if constexpr (!functor::DoLoadBeforeOp && //
                              (has_alpha_channel_v<DstT> && !load_full_vector_for_alpha_v<DstT>))
                {
                    alphaChannel[0] = pixelOut0->w;
                    alphaChannel[1] = pixelOut1->w;
                    alphaChannel[2] = pixelOut2->w;
                    alphaChannel[3] = pixelOut3->w;
                }

                // restore alpha channel value:
                if constexpr (has_alpha_channel_v<DstT>)
                {
                    res[0].w = alphaChannel[0];
                    res[1].w = alphaChannel[1];
                    res[2].w = alphaChannel[2];
                    res[3].w = alphaChannel[3];
                }

                *pixelOut0 = res[0];
                *pixelOut1 = res[1];
                *pixelOut2 = res[2];
                *pixelOut3 = res[3];
            }
        }
    }
}

} // namespace mpp::image::cpuSimple