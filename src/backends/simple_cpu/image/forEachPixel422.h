#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/defines.h>
#include <common/image/channel.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/mpp_defs.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vector1.h>
#include <common/vector2.h>
#include <iostream>

namespace mpp::image::cpuSimple
{
// forward declaration
template <PixelType T> class ImageView;

enum class Dst422C2Layout : byte
{
    YCrCb,
    YCbCr,
    CbYCr,
    CrYCb // this variant is technically possible but doesn't seem to exist in real world scenarios.
};

/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 2 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixel422(ImageView<Vector1<remove_vector_t<DstT>>> &aDst1, ImageView<Vector2<remove_vector_t<DstT>>> &aDst2,
                     ChromaSubsamplePos aChromaSubsamplePos, const functor &aFunctor);

/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 3 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixel422(ImageView<Vector1<remove_vector_t<DstT>>> &aDst1, ImageView<Vector1<remove_vector_t<DstT>>> &aDst2,
                     ImageView<Vector1<remove_vector_t<DstT>>> &aDst3, ChromaSubsamplePos aChromaSubsamplePos,
                     const functor &aFunctor);

/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 4 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixel422(ImageView<Vector2<remove_vector_t<DstT>>> &aDst1, ChromaSubsamplePos aChromaSubsamplePos,
                     Dst422C2Layout aLayout, const functor &aFunctor);
} // namespace mpp::image::cpuSimple