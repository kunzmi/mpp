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
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 2 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixelPlanar(ImageView<DstT> &aDst1, ImageView<DstT> &aDst2, const functor &aOp);

/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 3 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixelPlanar(ImageView<DstT> &aDst1, ImageView<DstT> &aDst2, ImageView<DstT> &aDst3, const functor &aOp);

/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 4 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixelPlanar(ImageView<DstT> &aDst1, ImageView<DstT> &aDst2, ImageView<DstT> &aDst3, ImageView<DstT> &aDst4,
                        const functor &aOp);
} // namespace opp::image::cpuSimple