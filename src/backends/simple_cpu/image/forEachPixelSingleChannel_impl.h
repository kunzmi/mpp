#pragma once
#include "forEachPixelSingleChannel.h"
#include <backends/cuda/cudaException.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/image/channel.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vector1.h>
#include <common/vector_typetraits.h>
#include <iostream>

namespace opp::image::cpuSimple
{
/// <summary>
/// runs aFunctor on every pixel (only one channel of a multi-channel pixel) of an image. no inplace operation, no mask.
/// </summary>
template <typename DstT, typename functor>
void forEachPixelSingleChannel(ImageView<DstT> &aDst, Channel aDstChannel, const functor &aFunctor)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        Vector1<remove_vector_t<DstT>> res;
        DstT &pixelOut = pixelIterator.Value();

        aFunctor(pixelX, pixelY, res);

        pixelOut[aDstChannel] = res.x;
    }
}
} // namespace opp::image::cpuSimple