#pragma once
#include "imageView.h"
#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/thresholdAndCompare/thresholdAndCompareKernel.h>
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
#include <common/image/validateImage.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace mpp::image::cuda
{
#pragma region Compare
template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::Compare(const ImageView<T> &aSrc2, CompareOp aCompare, ImageView<Pixel8uC1> &aDst,
                                            const mpp::cuda::StreamCtx &aStreamCtx) const
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeCompareSrcSrc<T, T, Pixel8uC1>(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(),
                                         aDst.Pitch(), aCompare, SizeRoi(), aStreamCtx);
    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::Compare(const T &aConst, CompareOp aCompare, ImageView<Pixel8uC1> &aDst,
                                            const mpp::cuda::StreamCtx &aStreamCtx) const
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeCompareSrcC<T, T, Pixel8uC1>(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), aCompare,
                                       SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::Compare(const mpp::cuda::DevVarView<T> &aConst, CompareOp aCompare,
                                            ImageView<Pixel8uC1> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeCompareSrcDevC<T, T, Pixel8uC1>(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(),
                                          aCompare, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::Compare(CompareOp aCompare, ImageView<Pixel8uC1> &aDst,
                                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeCompareSrc<T, T, Pixel8uC1>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aCompare, SizeRoi(),
                                      aStreamCtx);

    return aDst;
}
template <PixelType T>
ImageView<same_vector_size_different_type_t<T, byte>> &ImageView<T>::Compare(
    const ImageView<T> &aSrc2, CompareOp aCompare, ImageView<same_vector_size_different_type_t<T, byte>> &aDst,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_size_v<T> > 1)
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeCompareSrcSrc<T, T, same_vector_size_different_type_t<T, byte>>(
        PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aCompare, SizeRoi(),
        aStreamCtx);
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, byte>> &ImageView<T>::Compare(
    const T &aConst, CompareOp aCompare, ImageView<same_vector_size_different_type_t<T, byte>> &aDst,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_size_v<T> > 1)
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeCompareSrcC<T, T, same_vector_size_different_type_t<T, byte>>(
        PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), aCompare, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, byte>> &ImageView<T>::Compare(
    const mpp::cuda::DevVarView<T> &aConst, CompareOp aCompare,
    ImageView<same_vector_size_different_type_t<T, byte>> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_size_v<T> > 1)
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeCompareSrcDevC<T, T, same_vector_size_different_type_t<T, byte>>(
        PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), aCompare, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, byte>> &ImageView<T>::Compare(
    CompareOp aCompare, ImageView<same_vector_size_different_type_t<T, byte>> &aDst,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T> && (vector_size_v<T> > 1)
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeCompareSrc<T, T, same_vector_size_different_type_t<T, byte>>(PointerRoi(), Pitch(), aDst.PointerRoi(),
                                                                       aDst.Pitch(), aCompare, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CompareEqEps(const ImageView<T> &aSrc2,
                                                 complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                                 ImageView<Pixel8uC1> &aDst,
                                                 const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeCompareEqEpsSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                             aEpsilon, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CompareEqEps(const T &aConst, complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                                 ImageView<Pixel8uC1> &aDst,
                                                 const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeCompareEqEpsSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), aEpsilon, SizeRoi(),
                           aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CompareEqEps(const mpp::cuda::DevVarView<T> &aConst,
                                                 complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                                 ImageView<Pixel8uC1> &aDst,
                                                 const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeCompareEqEpsSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), aEpsilon,
                              SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion
#pragma region Threshold
template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, CompareOp aCompare, ImageView<T> &aDst,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    switch (aCompare)
    {
        case mpp::CompareOp::Less:
            return ThresholdLT(aThreshold, aDst, aStreamCtx);
            break;
        case mpp::CompareOp::Greater:
            return ThresholdGT(aThreshold, aDst, aStreamCtx);
            break;
        default:
            throw INVALIDARGUMENT(
                aCompare,
                "CompareOp " << aCompare << " is not supported for Threshold, only Less and Greater are supported.");
            break;
    }
}
template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const mpp::cuda::DevVarView<T> &aThreshold, CompareOp aCompare,
                                      ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    switch (aCompare)
    {
        case mpp::CompareOp::Less:
            return ThresholdLT(aThreshold, aDst, aStreamCtx);
            break;
        case mpp::CompareOp::Greater:
            return ThresholdGT(aThreshold, aDst, aStreamCtx);
            break;
        default:
            throw INVALIDARGUMENT(
                aCompare,
                "CompareOp " << aCompare << " is not supported for Threshold, only Less and Greater are supported.");
            break;
    }
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold, ImageView<T> &aDst,
                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeThresholdLTSrcC(PointerRoi(), Pitch(), aThreshold, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLT(const mpp::cuda::DevVarView<T> &aThreshold, ImageView<T> &aDst,
                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeThresholdLTSrcDevC(PointerRoi(), Pitch(), aThreshold.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                             aStreamCtx);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, ImageView<T> &aDst,
                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeThresholdGTSrcC(PointerRoi(), Pitch(), aThreshold, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const mpp::cuda::DevVarView<T> &aThreshold, ImageView<T> &aDst,
                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeThresholdGTSrcDevC(PointerRoi(), Pitch(), aThreshold.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                             aStreamCtx);

    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, CompareOp aCompare, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case mpp::CompareOp::Less:
            return ThresholdLT(aThreshold, aStreamCtx);
            break;
        case mpp::CompareOp::Greater:
            return ThresholdGT(aThreshold, aStreamCtx);
            break;
        default:
            throw INVALIDARGUMENT(
                aCompare,
                "CompareOp " << aCompare << " is not supported for Threshold, only Less and Greater are supported.");
            break;
    }
}
template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const mpp::cuda::DevVarView<T> &aThreshold, CompareOp aCompare,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case mpp::CompareOp::Less:
            return ThresholdLT(aThreshold, aStreamCtx);
            break;
        case mpp::CompareOp::Greater:
            return ThresholdGT(aThreshold, aStreamCtx);
            break;
        default:
            throw INVALIDARGUMENT(
                aCompare,
                "CompareOp " << aCompare << " is not supported for Threshold, only Less and Greater are supported.");
            break;
    }
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    validateImage(*this);
    InvokeThresholdLTInplaceC(PointerRoi(), Pitch(), aThreshold, SizeRoi(), aStreamCtx);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLT(const mpp::cuda::DevVarView<T> &aThreshold,
                                        const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    validateImage(*this);
    InvokeThresholdLTInplaceDevC(PointerRoi(), Pitch(), aThreshold.Pointer(), SizeRoi(), aStreamCtx);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    validateImage(*this);
    InvokeThresholdGTInplaceC(PointerRoi(), Pitch(), aThreshold, SizeRoi(), aStreamCtx);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const mpp::cuda::DevVarView<T> &aThreshold,
                                        const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    validateImage(*this);
    InvokeThresholdGTInplaceDevC(PointerRoi(), Pitch(), aThreshold.Pointer(), SizeRoi(), aStreamCtx);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare, ImageView<T> &aDst,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    switch (aCompare)
    {
        case mpp::CompareOp::Less:
            return ThresholdLT(aThreshold, aValue, aDst, aStreamCtx);
            break;
        case mpp::CompareOp::Greater:
            return ThresholdGT(aThreshold, aValue, aDst, aStreamCtx);
            break;
        default:
            throw INVALIDARGUMENT(
                aCompare,
                "CompareOp " << aCompare << " is not supported for Threshold, only Less and Greater are supported.");
            break;
    }
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold, const T &aValue, ImageView<T> &aDst,
                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeThresholdLTValSrcC(PointerRoi(), Pitch(), aThreshold, aValue, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                             aStreamCtx);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, const T &aValue, ImageView<T> &aDst,
                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeThresholdGTValSrcC(PointerRoi(), Pitch(), aThreshold, aValue, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                             aStreamCtx);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case mpp::CompareOp::Less:
            return ThresholdLT(aThreshold, aValue, aStreamCtx);
            break;
        case mpp::CompareOp::Greater:
            return ThresholdGT(aThreshold, aValue, aStreamCtx);
            break;
        default:
            throw INVALIDARGUMENT(
                aCompare,
                "CompareOp " << aCompare << " is not supported for Threshold, only Less and Greater are supported.");
            break;
    }
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold, const T &aValue, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    validateImage(*this);
    InvokeThresholdLTValInplaceC(PointerRoi(), Pitch(), aThreshold, aValue, SizeRoi(), aStreamCtx);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, const T &aValue, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    validateImage(*this);
    InvokeThresholdGTValInplaceC(PointerRoi(), Pitch(), aThreshold, aValue, SizeRoi(), aStreamCtx);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT,
                                          const T &aValueGT, ImageView<T> &aDst,
                                          const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeThresholdLTValGTValSrcC(PointerRoi(), Pitch(), aThresholdLT, aValueLT, aThresholdGT, aValueGT,
                                  aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT,
                                          const T &aValueGT, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    validateImage(*this);
    InvokeThresholdLTValGTValInplaceC(PointerRoi(), Pitch(), aThresholdLT, aValueLT, aThresholdGT, aValueGT, SizeRoi(),
                                      aStreamCtx);
    return *this;
}
#pragma endregion
#pragma region ReplaceIf

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(const ImageView<T> &aSrc2, CompareOp aCompare, const T &aValue,
                                      ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    validateImage(*this);
    validateImage(aSrc2);
    validateImage(aDst);
    checkSameSize(*this, aSrc2);
    checkSameSize(*this, aDst);

    InvokeReplaceIfSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aValue, aDst.PointerRoi(),
                          aDst.Pitch(), aCompare, SizeRoi(), aStreamCtx);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(const T &aConst, CompareOp aCompare, const T &aValue, ImageView<T> &aDst,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeReplaceIfSrcC(PointerRoi(), Pitch(), aConst, aValue, aDst.PointerRoi(), aDst.Pitch(), aCompare, SizeRoi(),
                        aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(const mpp::cuda::DevVarView<T> &aConst, CompareOp aCompare, const T &aValue,
                                      ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeReplaceIfSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aValue, aDst.PointerRoi(), aDst.Pitch(), aCompare,
                           SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(CompareOp aCompare, const T &aValue, ImageView<T> &aDst,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    validateImage(aDst);
    checkSameSize(*this, aDst);

    InvokeReplaceIfSrc(PointerRoi(), Pitch(), aValue, aDst.PointerRoi(), aDst.Pitch(), aCompare, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(const ImageView<T> &aSrc2, CompareOp aCompare, const T &aValue,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
{
    validateImage(*this);
    validateImage(aSrc2);
    checkSameSize(*this, aSrc2);

    InvokeReplaceIfInplaceSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aValue, aCompare, SizeRoi(),
                                 aStreamCtx);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(const T &aConst, CompareOp aCompare, const T &aValue,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
{
    validateImage(*this);
    InvokeReplaceIfInplaceSrcC(PointerRoi(), Pitch(), aConst, aValue, aCompare, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(const mpp::cuda::DevVarView<T> &aConst, CompareOp aCompare, const T &aValue,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
{
    validateImage(*this);
    InvokeReplaceIfInplaceSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aValue, aCompare, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(CompareOp aCompare, const T &aValue, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    validateImage(*this);
    InvokeReplaceIfInplaceSrc(PointerRoi(), Pitch(), aValue, aCompare, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion
} // namespace mpp::image::cuda