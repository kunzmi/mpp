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
template <typename DstT, typename functor> void reduction(const Size2D &aSize, DstT &aDst, const functor &aOp)
{
    for (auto &pixelIterator : aSize)
    {
        int pixelX = pixelIterator.Pixel.x;
        int pixelY = pixelIterator.Pixel.y;

        aOp(pixelX, pixelY, aDst);
    }
}
/// <summary>
/// runs aOp on every pixel of an image. Inplace and outplace operation, no mask.
/// </summary>
template <typename DstT, typename functor>
void reduction(const Size2D &aSize, DstT &aDst1, DstT &aDst2, const functor &aOp)
{
    for (auto &pixelIterator : aSize)
    {
        int pixelX = pixelIterator.Pixel.x;
        int pixelY = pixelIterator.Pixel.y;

        aOp(pixelX, pixelY, aDst1, aDst2);
    }
}
/// <summary>
/// runs aOp on every pixel of an image. Inplace and outplace operation, no mask.
/// </summary>
template <typename DstT, typename functor>
void reduction(const Size2D &aSize, DstT &aDst1, DstT &aDst2, DstT &aDst3, DstT &aDst4, DstT &aDst5, const functor &aOp)
{
    for (auto &pixelIterator : aSize)
    {
        int pixelX = pixelIterator.Pixel.x;
        int pixelY = pixelIterator.Pixel.y;

        aOp(pixelX, pixelY, aDst1, aDst2, aDst3, aDst4, aDst5);
    }
}
} // namespace opp::image::cpuSimple