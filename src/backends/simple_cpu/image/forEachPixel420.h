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

namespace mpp::image::cpuSimple
{
// forward declaration
template <PixelType T> class ImageView;

/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 2 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixel420(ImageView<Vector1<remove_vector_t<DstT>>> &aDst1, ImageView<Vector2<remove_vector_t<DstT>>> &aDst2,
                     ChromaSubsamplePos chromaSubsamplePos, const functor &aFunctor);

/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 3 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixel420(ImageView<Vector1<remove_vector_t<DstT>>> &aDst1, ImageView<Vector1<remove_vector_t<DstT>>> &aDst2,
                     ImageView<Vector1<remove_vector_t<DstT>>> &aDst3, ChromaSubsamplePos chromaSubsamplePos,
                     const functor &aFunctor);

} // namespace mpp::image::cpuSimple