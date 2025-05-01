#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/image/channel.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <common/utilities.h>

#include <iostream>

namespace opp::image::cpuSimple
{
// forward declaration
template <PixelType T> class ImageView;

/// <summary>
/// runs aFunctor on every pixel (only one channel of a multi-channel pixel) of an image. no inplace operation, no mask.
/// </summary>
template <typename DstT, typename functor>
void forEachPixelSingleChannel(ImageView<DstT> &aDst, Channel aDstChannel, const functor &aOp);
} // namespace opp::image::cpuSimple