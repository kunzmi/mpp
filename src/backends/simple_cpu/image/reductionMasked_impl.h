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

namespace opp::image::cpuSimple
{
/// <summary>
/// runs aOp on every pixel of an image. Inplace and outplace operation, no mask.
/// </summary>
template <typename DstT, typename functor>
size_t reduction(const ImageView<Pixel8uC1> &aMask, const Size2D &aSize, DstT &aDst, const functor &aOp)
{
    size_t pixelsInMask = 0;
    for (auto &pixelIterator : aSize)
    {
        int pixelX = pixelIterator.Pixel.x;
        int pixelY = pixelIterator.Pixel.y;

        if (aMask(pixelX, pixelY) != 0)
        {
            pixelsInMask++;
            aOp(pixelX, pixelY, aDst);
        }
    }
    return pixelsInMask;
}
/// <summary>
/// runs aOp on every pixel of an image. Inplace and outplace operation, no mask.
/// </summary>
template <typename DstT, typename functor>
size_t reduction(const ImageView<Pixel8uC1> &aMask, const Size2D &aSize, DstT &aDst1, DstT &aDst2, const functor &aOp)
{
    size_t pixelsInMask = 0;
    for (auto &pixelIterator : aSize)
    {
        int pixelX = pixelIterator.Pixel.x;
        int pixelY = pixelIterator.Pixel.y;

        if (aMask(pixelX, pixelY) != 0)
        {
            pixelsInMask++;
            aOp(pixelX, pixelY, aDst1, aDst2);
        }
    }
    return pixelsInMask;
}
} // namespace opp::image::cpuSimple