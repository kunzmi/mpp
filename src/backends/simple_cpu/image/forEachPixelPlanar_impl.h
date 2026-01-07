#pragma once
#include "forEachPixelPlanar.h"
#include <backends/cuda/cudaException.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/image/channel.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vector1.h>
#include <iostream>

namespace mpp::image::cpuSimple
{
/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 2 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixelPlanar(ImageView<Vector1<remove_vector_t<DstT>>> &aDst1,
                        ImageView<Vector1<remove_vector_t<DstT>>> &aDst2, const functor &aFunctor)
{
    using DstPlaneT = Vector1<remove_vector_t<DstT>>;

    auto pixelIterator2 = aDst2.begin();
    for (auto &pixelIterator1 : aDst1)
    {
        int pixelX = pixelIterator1.Pixel().x - aDst1.ROI().x;
        int pixelY = pixelIterator1.Pixel().y - aDst1.ROI().y;

        DstT res;
        DstPlaneT &pixelOut1 = pixelIterator1.Value();
        DstPlaneT &pixelOut2 = pixelIterator2.Value();

        // load the destination pixel in case of inplace operation:
        if constexpr (functor::DoLoadBeforeOp)
        {
            res.x = pixelOut1.x;
            res.y = pixelOut2.x;
        }

        // for nearly all functors aOp will evaluate to constant true.
        // Only transformerFunctor is capable of returning false if a source pixel is outside of the src-roi
        if (aFunctor(pixelX, pixelY, res))
        {
            pixelOut1 = res.x;
            pixelOut2 = res.y;
        }

        ++pixelIterator2;
    }
}

/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 3 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixelPlanar(ImageView<Vector1<remove_vector_t<DstT>>> &aDst1,
                        ImageView<Vector1<remove_vector_t<DstT>>> &aDst2,
                        ImageView<Vector1<remove_vector_t<DstT>>> &aDst3, const functor &aFunctor)
{
    using DstPlaneT = Vector1<remove_vector_t<DstT>>;

    auto pixelIterator2 = aDst2.begin();
    auto pixelIterator3 = aDst3.begin();
    for (auto &pixelIterator1 : aDst1)
    {
        int pixelX = pixelIterator1.Pixel().x - aDst1.ROI().x;
        int pixelY = pixelIterator1.Pixel().y - aDst1.ROI().y;

        DstT res;
        DstPlaneT &pixelOut1 = pixelIterator1.Value();
        DstPlaneT &pixelOut2 = pixelIterator2.Value();
        DstPlaneT &pixelOut3 = pixelIterator3.Value();

        // load the destination pixel in case of inplace operation:
        if constexpr (functor::DoLoadBeforeOp)
        {
            res.x = pixelOut1.x;
            res.y = pixelOut2.x;
            res.z = pixelOut3.x;
        }

        // for nearly all functors aOp will evaluate to constant true.
        // Only transformerFunctor is capable of returning false if a source pixel is outside of the src-roi
        if (aFunctor(pixelX, pixelY, res))
        {
            pixelOut1 = res.x;
            pixelOut2 = res.y;
            pixelOut3 = res.z;
        }

        ++pixelIterator2;
        ++pixelIterator3;
    }
}

/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 4 channel destination.
/// </summary>
template <typename DstT, typename functor>
void forEachPixelPlanar(ImageView<Vector1<remove_vector_t<DstT>>> &aDst1,
                        ImageView<Vector1<remove_vector_t<DstT>>> &aDst2,
                        ImageView<Vector1<remove_vector_t<DstT>>> &aDst3,
                        ImageView<Vector1<remove_vector_t<DstT>>> &aDst4, const functor &aFunctor)
{
    using DstPlaneT = Vector1<remove_vector_t<DstT>>;

    auto pixelIterator2 = aDst2.begin();
    auto pixelIterator3 = aDst3.begin();
    auto pixelIterator4 = aDst4.begin();
    for (auto &pixelIterator1 : aDst1)
    {
        int pixelX = pixelIterator1.Pixel().x - aDst1.ROI().x;
        int pixelY = pixelIterator1.Pixel().y - aDst1.ROI().y;

        DstT res;
        DstPlaneT &pixelOut1 = pixelIterator1.Value();
        DstPlaneT &pixelOut2 = pixelIterator2.Value();
        DstPlaneT &pixelOut3 = pixelIterator3.Value();
        DstPlaneT &pixelOut4 = pixelIterator4.Value();

        // load the destination pixel in case of inplace operation:
        if constexpr (functor::DoLoadBeforeOp)
        {
            res.x = pixelOut1.x;
            res.y = pixelOut2.x;
            res.z = pixelOut3.x;
            res.w = pixelOut4.x;
        }

        // for nearly all functors aOp will evaluate to constant true.
        // Only transformerFunctor is capable of returning false if a source pixel is outside of the src-roi
        if (aFunctor(pixelX, pixelY, res))
        {
            pixelOut1 = res.x;
            pixelOut2 = res.y;
            pixelOut3 = res.z;
            pixelOut4 = res.w;
        }

        ++pixelIterator2;
        ++pixelIterator3;
        ++pixelIterator4;
    }
}
} // namespace mpp::image::cpuSimple