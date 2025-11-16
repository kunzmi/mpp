#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include "imageView.h"
#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/geometryTransforms/geometryTransformsKernel.h>
#include <backends/cuda/streamCtx.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace mpp::image::cuda
{
#pragma region Affine
template <PixelType T>
ImageView<T> &ImageView<T>::WarpAffine(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                                       InterpolationMode aInterpolation, BorderType aBorder,
                                       const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->WarpAffineBack(aDst, aAffine.Inverse(), aInterpolation, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpAffine(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                                       InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                       const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->WarpAffineBack(aDst, aAffine.Inverse(), aInterpolation, aConstant, aBorder, ROI(), aStreamCtx);
}
template <PixelType T>
ImageView<T> &ImageView<T>::WarpAffine(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                                       InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                                       const mpp::cuda::StreamCtx &aStreamCtx) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->WarpAffineBack(aDst, aAffine.Inverse(), aInterpolation, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpAffine(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                                       InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                       const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->WarpAffineBack(aDst, aAffine.Inverse(), aInterpolation, aConstant, aBorder, aAllowedReadRoi,
                                aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                              BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aDst1, aDst2, aAffine.Inverse(), aInterpolation, aBorder, aSrc1.ROI(),
                                 aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                              const T &aConstant, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aDst1, aDst2, aAffine.Inverse(), aInterpolation, aConstant, aBorder,
                                 aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                              BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aDst1, aDst2, aAffine.Inverse(), aInterpolation, {0}, aBorder,
                                 aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                              const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aDst1, aDst2, aAffine.Inverse(), aInterpolation, aConstant, aBorder,
                                 aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                              const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                              BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aAffine.Inverse(), aInterpolation, aBorder,
                                 aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                              const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                              const T &aConstant, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aAffine.Inverse(), aInterpolation, aConstant,
                                 aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffine(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                              const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                              BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aAffine.Inverse(), aInterpolation, {0},
                                 aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffine(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aAffine.Inverse(), aInterpolation, aConstant,
                                 aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffine(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, BorderType aBorder,
    const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aAffine.Inverse(),
                                 aInterpolation, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffine(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aAffine.Inverse(),
                                 aInterpolation, aConstant, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffine(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, BorderType aBorder,
    const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aAffine.Inverse(),
                                 aInterpolation, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffine(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aAffine.Inverse(),
                                 aInterpolation, aConstant, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpAffineBack(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                                           InterpolationMode aInterpolation, BorderType aBorder,
                                           const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->WarpAffineBack(aDst, aAffine, aInterpolation, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpAffineBack(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                                           InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                           const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->WarpAffineBack(aDst, aAffine, aInterpolation, aConstant, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpAffineBack(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                                           InterpolationMode aInterpolation, BorderType aBorder,
                                           const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->WarpAffineBack(aDst, aAffine, aInterpolation, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpAffineBack(ImageView<T> &aDst, const AffineTransformation<double> &aAffine,
                                           InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                           const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeAffineBackSrc(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aAffine, aInterpolation, aBorder,
                        aConstant, roiOffset, aAllowedReadRoi.Size(), SizeRoi(), aDst.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::WarpAffineBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                  const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                                  BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aDst1, aDst2, aAffine, aInterpolation, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffineBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                  const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                                  const T &aConstant, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aDst1, aDst2, aAffine, aInterpolation, aConstant, aBorder, aSrc1.ROI(),
                                 aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffineBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                  const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                                  BorderType aBorder, const Roi &aAllowedReadRoi,
                                  const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aDst1, aDst2, aAffine, aInterpolation, {0}, aBorder, aAllowedReadRoi,
                                 aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffineBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                  const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                                  const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                                  const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());

    const Size2D minSizeAllocSrc = Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeAffineBackSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                        aDst2.PointerRoi(), aDst2.Pitch(), aAffine, aInterpolation, aBorder, aConstant, roiOffset,
                        aAllowedReadRoi.Size(), aSrc1.SizeRoi(), aDst1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffineBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                                  const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                                  BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aAffine, aInterpolation, aBorder,
                                 aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffineBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                                  const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                                  const T &aConstant, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aAffine, aInterpolation, aConstant, aBorder,
                                 aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffineBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, BorderType aBorder,
    const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aAffine, aInterpolation, {0}, aBorder,
                                 aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffineBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI() || aSrc1.ROI() != aSrc3.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2 aSrc3,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.ROI(), aDst3.ROI());

    const Size2D minSizeAllocSrc = Size2D::Min(Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc()), aSrc3.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr3 =
        gotoPtr(aSrc3.Pointer(), aSrc3.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeAffineBackSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                        aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(),
                        aDst3.Pitch(), aAffine, aInterpolation, aBorder, aConstant, roiOffset, aAllowedReadRoi.Size(),
                        aSrc1.SizeRoi(), aDst1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffineBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, BorderType aBorder,
    const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aAffine, aInterpolation,
                                 aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffineBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aAffine, aInterpolation,
                                 aConstant, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffineBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, BorderType aBorder,
    const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aAffine, aInterpolation, {0},
                                 aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpAffineBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aSrc1.ROI() != aSrc2.ROI() || aSrc1.ROI() != aSrc3.ROI() || aSrc1.ROI() != aSrc4.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2 aSrc3 aSrc4,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.ROI(), aDst3.ROI());
    checkSameSize(aDst1.ROI(), aDst4.ROI());

    const Size2D minSizeAllocSrc = Size2D::Min(
        Size2D::Min(Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc()), aSrc3.SizeAlloc()), aSrc4.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr3 =
        gotoPtr(aSrc3.Pointer(), aSrc3.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr4 =
        gotoPtr(aSrc4.Pointer(), aSrc4.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeAffineBackSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(), allowedPtr4,
                        aSrc4.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(),
                        aDst3.PointerRoi(), aDst3.Pitch(), aDst4.PointerRoi(), aDst4.Pitch(), aAffine, aInterpolation,
                        aBorder, aConstant, roiOffset, aAllowedReadRoi.Size(), aSrc1.SizeRoi(), aDst1.SizeRoi(),
                        aStreamCtx);
}
#pragma endregion

#pragma region Perspective
template <PixelType T>
ImageView<T> &ImageView<T>::WarpPerspective(ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective,
                                            InterpolationMode aInterpolation, BorderType aBorder,
                                            const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->WarpPerspectiveBack(aDst, aPerspective.Inverse(), aInterpolation, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpPerspective(ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective,
                                            InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                            const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->WarpPerspectiveBack(aDst, aPerspective.Inverse(), aInterpolation, aConstant, aBorder, ROI(),
                                     aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpPerspective(ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective,
                                            InterpolationMode aInterpolation, BorderType aBorder,
                                            const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->WarpPerspectiveBack(aDst, aPerspective.Inverse(), aInterpolation, {0}, aBorder, aAllowedReadRoi,
                                     aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpPerspective(ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective,
                                            InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                            const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->WarpPerspectiveBack(aDst, aPerspective.Inverse(), aInterpolation, aConstant, aBorder, aAllowedReadRoi,
                                     aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                   const PerspectiveTransformation<double> &aPerspective,
                                   InterpolationMode aInterpolation, BorderType aBorder,
                                   const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aDst1, aDst2, aPerspective.Inverse(), aInterpolation, aBorder,
                                      aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                   const PerspectiveTransformation<double> &aPerspective,
                                   InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                   const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aDst1, aDst2, aPerspective.Inverse(), aInterpolation, aConstant,
                                      aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                   const PerspectiveTransformation<double> &aPerspective,
                                   InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                                   const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aDst1, aDst2, aPerspective.Inverse(), aInterpolation, {0}, aBorder,
                                      aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                   const PerspectiveTransformation<double> &aPerspective,
                                   InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                   const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aDst1, aDst2, aPerspective.Inverse(), aInterpolation, aConstant,
                                      aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
    const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aPerspective.Inverse(), aInterpolation,
                                      aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aPerspective.Inverse(), aInterpolation,
                                      aConstant, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
    const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aPerspective.Inverse(), aInterpolation,
                                      {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aPerspective.Inverse(), aInterpolation,
                                      aConstant, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
    const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aPerspective.Inverse(),
                                      aInterpolation, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aPerspective.Inverse(),
                                      aInterpolation, aConstant, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
    const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aPerspective.Inverse(),
                                      aInterpolation, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aPerspective.Inverse(),
                                      aInterpolation, aConstant, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpPerspectiveBack(ImageView<T> &aDst,
                                                const PerspectiveTransformation<double> &aPerspective,
                                                InterpolationMode aInterpolation, BorderType aBorder,
                                                const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->WarpPerspectiveBack(aDst, aPerspective, aInterpolation, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpPerspectiveBack(ImageView<T> &aDst,
                                                const PerspectiveTransformation<double> &aPerspective,
                                                InterpolationMode aInterpolation, const T &aConstant,
                                                BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->WarpPerspectiveBack(aDst, aPerspective, aInterpolation, aConstant, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpPerspectiveBack(ImageView<T> &aDst,
                                                const PerspectiveTransformation<double> &aPerspective,
                                                InterpolationMode aInterpolation, BorderType aBorder,
                                                const Roi &aAllowedReadRoi,
                                                const mpp::cuda::StreamCtx &aStreamCtx) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->WarpPerspectiveBack(aDst, aPerspective, aInterpolation, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpPerspectiveBack(ImageView<T> &aDst,
                                                const PerspectiveTransformation<double> &aPerspective,
                                                InterpolationMode aInterpolation, const T &aConstant,
                                                BorderType aBorder, const Roi &aAllowedReadRoi,
                                                const mpp::cuda::StreamCtx &aStreamCtx) const
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokePerspectiveBackSrc(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aPerspective, aInterpolation,
                             aBorder, aConstant, roiOffset, aAllowedReadRoi.Size(), SizeRoi(), aDst.SizeRoi(),
                             aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                       const PerspectiveTransformation<double> &aPerspective,
                                       InterpolationMode aInterpolation, BorderType aBorder,
                                       const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aDst1, aDst2, aPerspective, aInterpolation, aBorder, aSrc1.ROI(),
                                      aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                       const PerspectiveTransformation<double> &aPerspective,
                                       InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                       const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aDst1, aDst2, aPerspective, aInterpolation, aConstant, aBorder,
                                      aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                       const PerspectiveTransformation<double> &aPerspective,
                                       InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                                       const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aDst1, aDst2, aPerspective, aInterpolation, {0}, aBorder,
                                      aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                       const PerspectiveTransformation<double> &aPerspective,
                                       InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                       const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());

    const Size2D minSizeAllocSrc = Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokePerspectiveBackSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                             aDst2.PointerRoi(), aDst2.Pitch(), aPerspective, aInterpolation, aBorder, aConstant,
                             roiOffset, aAllowedReadRoi.Size(), aSrc1.SizeRoi(), aDst1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
    const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aPerspective, aInterpolation, aBorder,
                                      aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aPerspective, aInterpolation, aConstant,
                                      aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
    const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aPerspective, aInterpolation, {0},
                                      aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI() || aSrc1.ROI() != aSrc3.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2 aSrc3,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.ROI(), aDst3.ROI());

    const Size2D minSizeAllocSrc = Size2D::Min(Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc()), aSrc3.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr3 =
        gotoPtr(aSrc3.Pointer(), aSrc3.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokePerspectiveBackSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                             aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(),
                             aDst3.Pitch(), aPerspective, aInterpolation, aBorder, aConstant, roiOffset,
                             aAllowedReadRoi.Size(), aSrc1.SizeRoi(), aDst1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
    const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aPerspective,
                                      aInterpolation, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aPerspective,
                                      aInterpolation, aConstant, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
    const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aPerspective,
                                      aInterpolation, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, const T &aConstant,
    BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aSrc1.ROI() != aSrc2.ROI() || aSrc1.ROI() != aSrc3.ROI() || aSrc1.ROI() != aSrc4.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2 aSrc3 aSrc4,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.ROI(), aDst3.ROI());
    checkSameSize(aDst1.ROI(), aDst4.ROI());

    const Size2D minSizeAllocSrc = Size2D::Min(
        Size2D::Min(Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc()), aSrc3.SizeAlloc()), aSrc4.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr3 =
        gotoPtr(aSrc3.Pointer(), aSrc3.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr4 =
        gotoPtr(aSrc4.Pointer(), aSrc4.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokePerspectiveBackSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                             allowedPtr4, aSrc4.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(),
                             aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aDst4.PointerRoi(), aDst4.Pitch(),
                             aPerspective, aInterpolation, aBorder, aConstant, roiOffset, aAllowedReadRoi.Size(),
                             aSrc1.SizeRoi(), aDst1.SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region Rotate

template <PixelType T>
ImageView<T> &ImageView<T>::Rotate(ImageView<T> &aDst, double aAngleInDeg, const Vector2<double> &aShift,
                                   InterpolationMode aInterpolation, BorderType aBorder,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->Rotate(aDst, aAngleInDeg, aShift, aInterpolation, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::Rotate(aSrc1, aSrc2, aDst1, aDst2, aAngleInDeg, aShift, aInterpolation, aBorder, aSrc1.ROI(),
                         aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, double aAngleInDeg,
                          const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder,
                          const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::Rotate(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aAngleInDeg, aShift, aInterpolation, aBorder,
                         aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::Rotate(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aAngleInDeg, aShift, aInterpolation,
                         aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Rotate(ImageView<T> &aDst, double aAngleInDeg, const Vector2<double> &aShift,
                                   InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->Rotate(aDst, aAngleInDeg, aShift, aInterpolation, aConstant, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          const T &aConstant, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::Rotate(aSrc1, aSrc2, aDst1, aDst2, aAngleInDeg, aShift, aInterpolation, aConstant, aBorder,
                         aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, double aAngleInDeg,
                          const Vector2<double> &aShift, InterpolationMode aInterpolation, const T &aConstant,
                          BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::Rotate(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aAngleInDeg, aShift, aInterpolation, aConstant,
                         aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          const T &aConstant, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::Rotate(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aAngleInDeg, aShift, aInterpolation,
                         aConstant, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Rotate(ImageView<T> &aDst, double aAngleInDeg, const Vector2<double> &aShift,
                                   InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Rotate(aDst, aAngleInDeg, aShift, aInterpolation, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Rotate(aSrc1, aSrc2, aDst1, aDst2, aAngleInDeg, aShift, aInterpolation, {0}, aBorder, aAllowedReadRoi,
                         aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, double aAngleInDeg,
                          const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder,
                          const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Rotate(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aAngleInDeg, aShift, aInterpolation, {0}, aBorder,
                         aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Rotate(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aAngleInDeg, aShift, aInterpolation,
                         {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Rotate(ImageView<T> &aDst, double aAngleInDeg, const Vector2<double> &aShift,
                                   InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                   const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    // The rotation and shift are given from source to destination, with shift applied after rotation. As we compute
    // from destination to source, we have to invert the transformation, the shift being now a pre-rotation shift:

    const AffineTransformation<double> rotate =
        AffineTransformation<double>::GetRotation(-aAngleInDeg) * AffineTransformation<double>::GetTranslation(-aShift);
    return this->WarpAffineBack(aDst, rotate, aInterpolation, aConstant, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                          const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    // The rotation and shift are given from source to destination, with shift applied after rotation. As we compute
    // from destination to source, we have to invert the transformation, the shift being now a pre-rotation shift:

    const AffineTransformation<double> rotate =
        AffineTransformation<double>::GetRotation(-aAngleInDeg) * AffineTransformation<double>::GetTranslation(-aShift);
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aDst1, aDst2, rotate, aInterpolation, aConstant, aBorder,
                                 aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, double aAngleInDeg,
                          const Vector2<double> &aShift, InterpolationMode aInterpolation, const T &aConstant,
                          BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    // The rotation and shift are given from source to destination, with shift applied after rotation. As we compute
    // from destination to source, we have to invert the transformation, the shift being now a pre-rotation shift:

    const AffineTransformation<double> rotate =
        AffineTransformation<double>::GetRotation(-aAngleInDeg) * AffineTransformation<double>::GetTranslation(-aShift);
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, rotate, aInterpolation, aConstant, aBorder,
                                 aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Rotate(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                          const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    // The rotation and shift are given from source to destination, with shift applied after rotation. As we compute
    // from destination to source, we have to invert the transformation, the shift being now a pre-rotation shift:

    const AffineTransformation<double> rotate =
        AffineTransformation<double>::GetRotation(-aAngleInDeg) * AffineTransformation<double>::GetTranslation(-aShift);
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, rotate, aInterpolation,
                                 aConstant, aBorder, aAllowedReadRoi, aStreamCtx);
}
#pragma endregion

#pragma region Resize

template <PixelType T>
ImageView<T> &ImageView<T>::Resize(ImageView<T> &aDst, InterpolationMode aInterpolation,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
{
    const Vec2f scaleFactor = Vec2f(aDst.SizeRoi()) / Vec2f(SizeRoi());
    const Vec2f shift       = Vec2f(0);

    return this->Resize(aDst, scaleFactor, shift, aInterpolation, {0}, BorderType::None, ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          InterpolationMode aInterpolation, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    const Vec2f scaleFactor = Vec2f(aDst1.SizeRoi()) / Vec2f(aSrc1.SizeRoi());
    const Vec2f shift       = Vec2f(0);

    ImageView<T>::Resize(aSrc1, aSrc2, aDst1, aDst2, scaleFactor, shift, aInterpolation, {0}, BorderType::None,
                         aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, InterpolationMode aInterpolation,
                          const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    const Vec2f scaleFactor = Vec2f(aDst1.SizeRoi()) / Vec2f(aSrc1.SizeRoi());
    const Vec2f shift       = Vec2f(0);

    ImageView<T>::Resize(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, scaleFactor, shift, aInterpolation, {0},
                         BorderType::None, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                          InterpolationMode aInterpolation, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    const Vec2f scaleFactor = Vec2f(aDst1.SizeRoi()) / Vec2f(aSrc1.SizeRoi());
    const Vec2f shift       = Vec2f(0);

    ImageView<T>::Resize(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, scaleFactor, shift, aInterpolation,
                         {0}, BorderType::None, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T> Vec2f ImageView<T>::ResizeGetNPPShift(ImageView<T> &aDst) const
{
    const Vec2f scaleFactor    = Vec2f(aDst.SizeRoi()) / Vec2d(SizeRoi());
    const Vec2f invScaleFactor = 1.0f / scaleFactor;
    Vec2f shift(0); // no shift if scaling == 1

    if (scaleFactor.x > 1.0f) // upscaling
    {
        shift.x = (0.25f - (1.0f - invScaleFactor.x) / 2.0f) * scaleFactor.x;
    }
    else if (scaleFactor.x < 1.0f) // downscaling
    {
        shift.x = -((1.0f - invScaleFactor.x) / 2.0f) * scaleFactor.x;
    }

    if (scaleFactor.y > 1.0f) // upscaling
    {
        shift.y = (0.25f - (1.0f - invScaleFactor.y) / 2.0f) * scaleFactor.y;
    }
    else if (scaleFactor.y < 1.0f) // downscaling
    {
        shift.y = -((1.0f - invScaleFactor.y) / 2.0f) * scaleFactor.y;
    }

    return shift;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Resize(ImageView<T> &aDst, const Vector2<double> &aScale, const Vector2<double> &aShift,
                                   InterpolationMode aInterpolation, BorderType aBorder,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->Resize(aDst, aScale, aShift, aInterpolation, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          const Vector2<double> &aScale, const Vector2<double> &aShift,
                          InterpolationMode aInterpolation, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::Resize(aSrc1, aSrc2, aDst1, aDst2, aScale, aShift, aInterpolation, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Vector2<double> &aScale,
                          const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder,
                          const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::Resize(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aScale, aShift, aInterpolation, aBorder, aSrc1.ROI(),
                         aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                          const Vector2<double> &aScale, const Vector2<double> &aShift,
                          InterpolationMode aInterpolation, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::Resize(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aScale, aShift, aInterpolation,
                         aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Resize(ImageView<T> &aDst, const Vector2<double> &aScale, const Vector2<double> &aShift,
                                   InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->Resize(aDst, aScale, aShift, aInterpolation, aConstant, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          const Vector2<double> &aScale, const Vector2<double> &aShift,
                          InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                          const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::Resize(aSrc1, aSrc2, aDst1, aDst2, aScale, aShift, aInterpolation, aConstant, aBorder, aSrc1.ROI(),
                         aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Vector2<double> &aScale,
                          const Vector2<double> &aShift, InterpolationMode aInterpolation, const T &aConstant,
                          BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::Resize(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aScale, aShift, aInterpolation, aConstant, aBorder,
                         aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                          const Vector2<double> &aScale, const Vector2<double> &aShift,
                          InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                          const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::Resize(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aScale, aShift, aInterpolation,
                         aConstant, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Resize(ImageView<T> &aDst, const Vector2<double> &aScale, const Vector2<double> &aShift,
                                   InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Resize(aDst, aScale, aShift, aInterpolation, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          const Vector2<double> &aScale, const Vector2<double> &aShift,
                          InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                          const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Resize(aSrc1, aSrc2, aDst1, aDst2, aScale, aShift, aInterpolation, {0}, aBorder, aAllowedReadRoi,
                         aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Vector2<double> &aScale,
                          const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder,
                          const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Resize(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aScale, aShift, aInterpolation, {0}, aBorder,
                         aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                          const Vector2<double> &aScale, const Vector2<double> &aShift,
                          InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                          const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Resize(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aScale, aShift, aInterpolation, {0},
                         aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Resize(ImageView<T> &aDst, const Vector2<double> &aScale, const Vector2<double> &aShift,
                                   InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                   const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    if (aScale.x <= 0 || aScale.y <= 0)
    {
        throw INVALIDARGUMENT(aScale, "Scale factors must be > 0. Provided scaling factors are: " << aScale);
    }

    if (aInterpolation == InterpolationMode::Super && (aScale.x >= 1 || aScale.y >= 1))
    {
        throw INVALIDARGUMENT(aInterpolation & aScale,
                              "For InterpolationMode::Super, scaling in X and Y direction must be a down-sampling, "
                              "i.e, a scaling value > 0 and < 1. Provided scaling factors are: "
                                  << aScale);
    }

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeResizeSrc(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aScale, aShift, aInterpolation, aBorder,
                    aConstant, roiOffset, aAllowedReadRoi.Size(), SizeRoi(), aDst.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          const Vector2<double> &aScale, const Vector2<double> &aShift,
                          InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                          const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());

    if (aScale.x <= 0 || aScale.y <= 0)
    {
        throw INVALIDARGUMENT(aScale, "Scale factors must be > 0. Provided scaling factors are: " << aScale);
    }

    if (aInterpolation == InterpolationMode::Super && (aScale.x >= 1 || aScale.y >= 1))
    {
        throw INVALIDARGUMENT(aInterpolation & aScale,
                              "For InterpolationMode::Super, scaling in X and Y direction must be a down-sampling, "
                              "i.e, a scaling value > 0 and < 1. Provided scaling factors are: "
                                  << aScale);
    }

    const Size2D minSizeAllocSrc = Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeResizeSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                    aDst2.PointerRoi(), aDst2.Pitch(), aScale, aShift, aInterpolation, aBorder, aConstant, roiOffset,
                    aAllowedReadRoi.Size(), aSrc1.SizeRoi(), aDst1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Vector2<double> &aScale,
                          const Vector2<double> &aShift, InterpolationMode aInterpolation, const T &aConstant,
                          BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI() || aSrc1.ROI() != aSrc3.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2 aSrc3,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.ROI(), aDst3.ROI());

    if (aScale.x <= 0 || aScale.y <= 0)
    {
        throw INVALIDARGUMENT(aScale, "Scale factors must be > 0. Provided scaling factors are: " << aScale);
    }

    if (aInterpolation == InterpolationMode::Super && (aScale.x >= 1 || aScale.y >= 1))
    {
        throw INVALIDARGUMENT(aInterpolation & aScale,
                              "For InterpolationMode::Super, scaling in X and Y direction must be a down-sampling, "
                              "i.e, a scaling value > 0 and < 1. Provided scaling factors are: "
                                  << aScale);
    }

    const Size2D minSizeAllocSrc = Size2D::Min(Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc()), aSrc3.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr3 =
        gotoPtr(aSrc3.Pointer(), aSrc3.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeResizeSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                    aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(),
                    aDst3.Pitch(), aScale, aShift, aInterpolation, aBorder, aConstant, roiOffset,
                    aAllowedReadRoi.Size(), aSrc1.SizeRoi(), aDst1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Resize(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                          const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                          const Vector2<double> &aScale, const Vector2<double> &aShift,
                          InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                          const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aSrc1.ROI() != aSrc2.ROI() || aSrc1.ROI() != aSrc3.ROI() || aSrc1.ROI() != aSrc4.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2 aSrc3 aSrc4,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.ROI(), aDst3.ROI());
    checkSameSize(aDst1.ROI(), aDst4.ROI());

    if (aScale.x <= 0 || aScale.y <= 0)
    {
        throw INVALIDARGUMENT(aScale, "Scale factors must be > 0. Provided scaling factors are: " << aScale);
    }

    if (aInterpolation == InterpolationMode::Super && (aScale.x >= 1 || aScale.y >= 1))
    {
        throw INVALIDARGUMENT(aInterpolation & aScale,
                              "For InterpolationMode::Super, scaling in X and Y direction must be a down-sampling, "
                              "i.e, a scaling value > 0 and < 1. Provided scaling factors are: "
                                  << aScale);
    }

    const Size2D minSizeAllocSrc = Size2D::Min(
        Size2D::Min(Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc()), aSrc3.SizeAlloc()), aSrc4.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr3 =
        gotoPtr(aSrc3.Pointer(), aSrc3.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr4 =
        gotoPtr(aSrc4.Pointer(), aSrc4.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeResizeSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(), allowedPtr4,
                    aSrc4.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(),
                    aDst3.PointerRoi(), aDst3.Pitch(), aDst4.PointerRoi(), aDst4.Pitch(), aScale, aShift,
                    aInterpolation, aBorder, aConstant, roiOffset, aAllowedReadRoi.Size(), aSrc1.SizeRoi(),
                    aDst1.SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region Mirror
template <PixelType T>
ImageView<T> &ImageView<T>::Mirror(ImageView<T> &aDst, MirrorAxis aAxis, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeMirrorSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aAxis, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T> ImageView<T> &ImageView<T>::Mirror(MirrorAxis aAxis, const mpp::cuda::StreamCtx &aStreamCtx)
{
    InvokeMirrorInplace(PointerRoi(), Pitch(), aAxis, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Remap

template <PixelType T>
ImageView<T> &ImageView<T>::Remap(ImageView<T> &aDst, const ImageView<Pixel32fC2> &aCoordinateMap,
                                  InterpolationMode aInterpolation, BorderType aBorder,
                                  const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->Remap(aDst, aCoordinateMap, aInterpolation, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::Remap(aSrc1, aSrc2, aDst1, aDst2, aCoordinateMap, aInterpolation, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, const ImageView<Pixel32fC2> &aCoordinateMap,
                         InterpolationMode aInterpolation, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aCoordinateMap, aInterpolation, aBorder, aSrc1.ROI(),
                        aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aCoordinateMap, aInterpolation, aBorder,
                        aSrc1.ROI(), aStreamCtx);
}
template <PixelType T>
ImageView<T> &ImageView<T>::Remap(ImageView<T> &aDst, const ImageView<Pixel32fC2> &aCoordinateMap,
                                  InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                  const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->Remap(aDst, aCoordinateMap, aInterpolation, aConstant, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         const T &aConstant, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::Remap(aSrc1, aSrc2, aDst1, aDst2, aCoordinateMap, aInterpolation, aConstant, aBorder, aSrc1.ROI(),
                        aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, const ImageView<Pixel32fC2> &aCoordinateMap,
                         InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aCoordinateMap, aInterpolation, aConstant, aBorder,
                        aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         const T &aConstant, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aCoordinateMap, aInterpolation,
                        aConstant, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Remap(ImageView<T> &aDst, const ImageView<Pixel32fC2> &aCoordinateMap,
                                  InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                                  const mpp::cuda::StreamCtx &aStreamCtx) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Remap(aDst, aCoordinateMap, aInterpolation, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Remap(aSrc1, aSrc2, aDst1, aDst2, aCoordinateMap, aInterpolation, {0}, aBorder, aAllowedReadRoi,
                        aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, const ImageView<Pixel32fC2> &aCoordinateMap,
                         InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aCoordinateMap, aInterpolation, {0}, aBorder,
                        aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aCoordinateMap, aInterpolation, {0},
                        aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Remap(ImageView<T> &aDst, const ImageView<Pixel32fC2> &aCoordinateMap,
                                  InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                                  const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    checkSameSize(aDst.SizeRoi(), aCoordinateMap.SizeRoi());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeRemapSrc(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aCoordinateMap.PointerRoi(),
                   aCoordinateMap.Pitch(), aInterpolation, aBorder, aConstant, roiOffset, aAllowedReadRoi.Size(),
                   SizeRoi(), aDst.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMap.SizeRoi());

    const Size2D minSizeAllocSrc = Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeRemapSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                   aDst2.PointerRoi(), aDst2.Pitch(), aCoordinateMap.PointerRoi(), aCoordinateMap.Pitch(),
                   aInterpolation, aBorder, aConstant, roiOffset, aAllowedReadRoi.Size(), aSrc1.SizeRoi(),
                   aDst1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, const ImageView<Pixel32fC2> &aCoordinateMap,
                         InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                         const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI() || aSrc1.ROI() != aSrc3.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2 aSrc3,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.ROI(), aDst3.ROI());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMap.SizeRoi());

    const Size2D minSizeAllocSrc = Size2D::Min(Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc()), aSrc3.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr3 =
        gotoPtr(aSrc3.Pointer(), aSrc3.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeRemapSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                   aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(),
                   aDst3.Pitch(), aCoordinateMap.PointerRoi(), aCoordinateMap.Pitch(), aInterpolation, aBorder,
                   aConstant, roiOffset, aAllowedReadRoi.Size(), aSrc1.SizeRoi(), aDst1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aSrc1.ROI() != aSrc2.ROI() || aSrc1.ROI() != aSrc3.ROI() || aSrc1.ROI() != aSrc4.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2 aSrc3 aSrc4,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.ROI(), aDst3.ROI());
    checkSameSize(aDst1.ROI(), aDst4.ROI());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMap.SizeRoi());

    const Size2D minSizeAllocSrc = Size2D::Min(
        Size2D::Min(Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc()), aSrc3.SizeAlloc()), aSrc4.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr3 =
        gotoPtr(aSrc3.Pointer(), aSrc3.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr4 =
        gotoPtr(aSrc4.Pointer(), aSrc4.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeRemapSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(), allowedPtr4,
                   aSrc4.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(),
                   aDst3.PointerRoi(), aDst3.Pitch(), aDst4.PointerRoi(), aDst4.Pitch(), aCoordinateMap.PointerRoi(),
                   aCoordinateMap.Pitch(), aInterpolation, aBorder, aConstant, roiOffset, aAllowedReadRoi.Size(),
                   aSrc1.SizeRoi(), aDst1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Remap(ImageView<T> &aDst, const ImageView<Pixel32fC1> &aCoordinateMapX,
                                  const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                                  BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->Remap(aDst, aCoordinateMapX, aCoordinateMapY, aInterpolation, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::Remap(aSrc1, aSrc2, aDst1, aDst2, aCoordinateMapX, aCoordinateMapY, aInterpolation, aBorder,
                        aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, const ImageView<Pixel32fC1> &aCoordinateMapX,
                         const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                         BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aCoordinateMapX, aCoordinateMapY, aInterpolation,
                        aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aCoordinateMapX, aCoordinateMapY,
                        aInterpolation, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Remap(ImageView<T> &aDst, const ImageView<Pixel32fC1> &aCoordinateMapX,
                                  const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                                  const T &aConstant, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return this->Remap(aDst, aCoordinateMapX, aCoordinateMapY, aInterpolation, aConstant, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    ImageView<T>::Remap(aSrc1, aSrc2, aDst1, aDst2, aCoordinateMapX, aCoordinateMapY, aInterpolation, aConstant,
                        aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, const ImageView<Pixel32fC1> &aCoordinateMapX,
                         const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                         const T &aConstant, BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aCoordinateMapX, aCoordinateMapY, aInterpolation,
                        aConstant, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aCoordinateMapX, aCoordinateMapY,
                        aInterpolation, aConstant, aBorder, aSrc1.ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Remap(ImageView<T> &aDst, const ImageView<Pixel32fC1> &aCoordinateMapX,
                                  const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                                  BorderType aBorder, const Roi &aAllowedReadRoi,
                                  const mpp::cuda::StreamCtx &aStreamCtx) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Remap(aDst, aCoordinateMapX, aCoordinateMapY, aInterpolation, {0}, aBorder, aAllowedReadRoi,
                       aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Remap(aSrc1, aSrc2, aDst1, aDst2, aCoordinateMapX, aCoordinateMapY, aInterpolation, {0}, aBorder,
                        aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, const ImageView<Pixel32fC1> &aCoordinateMapX,
                         const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                         BorderType aBorder, const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aCoordinateMapX, aCoordinateMapY, aInterpolation, {0},
                        aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, BorderType aBorder, const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aCoordinateMapX, aCoordinateMapY,
                        aInterpolation, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Remap(ImageView<T> &aDst, const ImageView<Pixel32fC1> &aCoordinateMapX,
                                  const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                                  const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                                  const mpp::cuda::StreamCtx &aStreamCtx) const
{
    checkSameSize(aDst.SizeRoi(), aCoordinateMapX.SizeRoi());
    checkSameSize(aDst.SizeRoi(), aCoordinateMapY.SizeRoi());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeRemapSrc(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aCoordinateMapX.PointerRoi(),
                   aCoordinateMapX.Pitch(), aCoordinateMapY.PointerRoi(), aCoordinateMapY.Pitch(), aInterpolation,
                   aBorder, aConstant, roiOffset, aAllowedReadRoi.Size(), SizeRoi(), aDst.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                         const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires TwoChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMapX.SizeRoi());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMapY.SizeRoi());

    const Size2D minSizeAllocSrc = Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeRemapSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                   aDst2.PointerRoi(), aDst2.Pitch(), aCoordinateMapX.PointerRoi(), aCoordinateMapX.Pitch(),
                   aCoordinateMapY.PointerRoi(), aCoordinateMapY.Pitch(), aInterpolation, aBorder, aConstant, roiOffset,
                   aAllowedReadRoi.Size(), aSrc1.SizeRoi(), aDst1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, const ImageView<Pixel32fC1> &aCoordinateMapX,
                         const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                         const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi,
                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires ThreeChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI() || aSrc1.ROI() != aSrc3.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2 aSrc3,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.ROI(), aDst3.ROI());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMapX.SizeRoi());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMapY.SizeRoi());

    const Size2D minSizeAllocSrc = Size2D::Min(Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc()), aSrc3.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr3 =
        gotoPtr(aSrc3.Pointer(), aSrc3.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeRemapSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                   aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(),
                   aDst3.Pitch(), aCoordinateMapX.PointerRoi(), aCoordinateMapX.Pitch(), aCoordinateMapY.PointerRoi(),
                   aCoordinateMapY.Pitch(), aInterpolation, aBorder, aConstant, roiOffset, aAllowedReadRoi.Size(),
                   aSrc1.SizeRoi(), aDst1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Remap(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                         const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, const T &aConstant, BorderType aBorder,
                         const Roi &aAllowedReadRoi, const mpp::cuda::StreamCtx &aStreamCtx)
    requires FourChannelNoAlpha<T>
{
    if (aSrc1.ROI() != aSrc2.ROI() || aSrc1.ROI() != aSrc3.ROI() || aSrc1.ROI() != aSrc4.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2 aSrc3 aSrc4,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.ROI(), aDst3.ROI());
    checkSameSize(aDst1.ROI(), aDst4.ROI());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMapX.SizeRoi());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMapY.SizeRoi());

    const Size2D minSizeAllocSrc = Size2D::Min(
        Size2D::Min(Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc()), aSrc3.SizeAlloc()), aSrc4.SizeAlloc());

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAllocSrc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr3 =
        gotoPtr(aSrc3.Pointer(), aSrc3.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr4 =
        gotoPtr(aSrc4.Pointer(), aSrc4.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeRemapSrc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(), allowedPtr4,
                   aSrc4.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(),
                   aDst3.PointerRoi(), aDst3.Pitch(), aDst4.PointerRoi(), aDst4.Pitch(), aCoordinateMapX.PointerRoi(),
                   aCoordinateMapX.Pitch(), aCoordinateMapY.PointerRoi(), aCoordinateMapY.Pitch(), aInterpolation,
                   aBorder, aConstant, roiOffset, aAllowedReadRoi.Size(), aSrc1.SizeRoi(), aDst1.SizeRoi(), aStreamCtx);
}
#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND