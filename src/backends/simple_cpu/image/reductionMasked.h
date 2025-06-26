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
/// runs reduction aOp on every pixel of an image. Only pixel with mask != 0 are taken into account.
/// Returns the number of active pixels inside the mask.
/// </summary>
template <typename DstT, typename functor>
size_t reduction(const ImageView<Pixel8uC1> &aMask, const Size2D &aSize, DstT &aDst, const functor &aOp);

/// <summary>
/// runs reduction aOp on every pixel of an image. Only pixel with mask != 0 are taken into account.
/// Returns the number of active pixels inside the mask.
/// </summary>
template <typename DstT, typename functor>
size_t reduction(const ImageView<Pixel8uC1> &aMask, const Size2D &aSize, DstT &aDst1, DstT &aDst2, const functor &aOp);
} // namespace mpp::image::cpuSimple