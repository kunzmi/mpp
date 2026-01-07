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
// forward declaration
template <PixelType T> class ImageView;

/// <summary>
/// runs aOp on every pixel of an image. Inplace and outplace operation, no mask.
/// </summary>
template <typename DstT, typename functor> void forEachPixelBlock(ImageView<DstT> &aDst, const functor &aFunctor);
} // namespace mpp::image::cpuSimple