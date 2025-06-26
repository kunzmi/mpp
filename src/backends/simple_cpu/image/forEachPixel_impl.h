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
template <typename DstT, typename functor> void forEachPixel(ImageView<DstT> &aDst, const functor &aOp)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        // will be optimized away as unused in case of no alpha channel:
        pixel_basetype_t<DstT> alphaChannel;

        DstT res;
        DstT &pixelOut = pixelIterator.Value();

        // load the destination pixel in case of inplace operation or we load the full pixel for alpha operations:
        if constexpr (functor::DoLoadBeforeOp || //
                      (has_alpha_channel_v<DstT> && load_full_vector_for_alpha_v<DstT>))
        {
            res = pixelOut;

            // save alpha channel value seperatly:
            if constexpr (has_alpha_channel_v<DstT>)
            {
                alphaChannel = res.w;
            }
        }

        // for nearly all functors aOp will evaluate to constant true.
        // Only transformerFunctor is capable of returning false if a source pixel is outside of the src-roi
        if (aOp(pixelX, pixelY, res))
        {
            // if we don't load the pixel anyhow but we still need just the alpha channel, load it:
            if constexpr (!functor::DoLoadBeforeOp && //
                          (has_alpha_channel_v<DstT> && !load_full_vector_for_alpha_v<DstT>))
            {
                alphaChannel = pixelOut.w;
            }

            // restore alpha channel value:
            if constexpr (has_alpha_channel_v<DstT>)
            {
                res.w = alphaChannel;
            }

            pixelOut = res;
        }
    }
}

/// <summary>
/// runs aOp on every pixel in roi (0,0, aSize). Inplace operation, no mask.
/// </summary>
template <typename DstT, typename functor, bool xUneven>
void forEachPixel(ImageView<DstT> &aDst, const Size2D aSize, const functor &aOp)
{
    for (auto &pixelIterator : aSize)
    {
        int pixelX = pixelIterator.Pixel.x;
        int pixelY = pixelIterator.Pixel.y;

        int otherX = 0;
        int otherY = 0;

        if constexpr (xUneven)
        {
            if (pixelX == aSize.x - 1 && pixelY > aSize.y / 2)
            {
                continue;
            }
        }

        DstT &pixel = aDst(pixelX, pixelY);

        aOp(pixelX, pixelY, otherX, otherY);

        DstT &otherPixel = aDst(otherX, otherY);

        DstT temp = otherPixel;

        // keep alpha channel value (simpleCPU version):
        if constexpr (has_alpha_channel_v<DstT>)
        {
            otherPixel.x = pixel.x;
            otherPixel.y = pixel.y;
            otherPixel.z = pixel.z;
            pixel.x      = temp.x;
            pixel.y      = temp.y;
            pixel.z      = temp.z;
        }
        else
        {
            otherPixel = pixel;
            pixel      = temp;
        }
    }
}
} // namespace mpp::image::cpuSimple