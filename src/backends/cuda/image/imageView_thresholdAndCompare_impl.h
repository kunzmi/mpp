#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

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
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/opp_defs.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace opp::image::cuda
{
#pragma region Compare
template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::Compare(const ImageView<T> &aSrc2, CompareOp aCompare, ImageView<Pixel8uC1> &aDst,
                                            const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeCompareSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                        aCompare, SizeRoi(), aStreamCtx);
    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::Compare(const T &aConst, CompareOp aCompare, ImageView<Pixel8uC1> &aDst,
                                            const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeCompareSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), aCompare, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::Compare(const opp::cuda::DevVarView<T> &aConst, CompareOp aCompare,
                                            ImageView<Pixel8uC1> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeCompareSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), aCompare, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CompareEqEps(const ImageView<T> &aSrc2,
                                                 complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                                 ImageView<Pixel8uC1> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeCompareEqEpsSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                             aEpsilon, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CompareEqEps(const T &aConst, complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                                 ImageView<Pixel8uC1> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeCompareEqEpsSrcC(PointerRoi(), Pitch(), aConst, aDst.PointerRoi(), aDst.Pitch(), aEpsilon, SizeRoi(),
                           aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CompareEqEps(const opp::cuda::DevVarView<T> &aConst,
                                                 complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                                 ImageView<Pixel8uC1> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeCompareEqEpsSrcDevC(PointerRoi(), Pitch(), aConst.Pointer(), aDst.PointerRoi(), aDst.Pitch(), aEpsilon,
                              SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion
#pragma region Threshold
template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, CompareOp aCompare, ImageView<T> &aDst,
                                      const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case opp::CompareOp::Less:
            return ThresholdLT(aThreshold, aDst, aStreamCtx);
            break;
        case opp::CompareOp::Greater:
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
ImageView<T> &ImageView<T>::Threshold(const opp::cuda::DevVarView<T> &aThreshold, CompareOp aCompare,
                                      ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case opp::CompareOp::Less:
            return ThresholdLT(aThreshold, aDst, aStreamCtx);
            break;
        case opp::CompareOp::Greater:
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
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeThresholdLTSrcC(PointerRoi(), Pitch(), aThreshold, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLT(const opp::cuda::DevVarView<T> &aThreshold, ImageView<T> &aDst,
                                        const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeThresholdLTSrcDevC(PointerRoi(), Pitch(), aThreshold.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                             aStreamCtx);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeThresholdGTSrcC(PointerRoi(), Pitch(), aThreshold, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const opp::cuda::DevVarView<T> &aThreshold, ImageView<T> &aDst,
                                        const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeThresholdGTSrcDevC(PointerRoi(), Pitch(), aThreshold.Pointer(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                             aStreamCtx);

    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, CompareOp aCompare, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case opp::CompareOp::Less:
            return ThresholdLT(aThreshold, aStreamCtx);
            break;
        case opp::CompareOp::Greater:
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
ImageView<T> &ImageView<T>::Threshold(const opp::cuda::DevVarView<T> &aThreshold, CompareOp aCompare,
                                      const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case opp::CompareOp::Less:
            return ThresholdLT(aThreshold, aStreamCtx);
            break;
        case opp::CompareOp::Greater:
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
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    InvokeThresholdLTInplaceC(PointerRoi(), Pitch(), aThreshold, SizeRoi(), aStreamCtx);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLT(const opp::cuda::DevVarView<T> &aThreshold,
                                        const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    InvokeThresholdLTInplaceDevC(PointerRoi(), Pitch(), aThreshold.Pointer(), SizeRoi(), aStreamCtx);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    InvokeThresholdGTInplaceC(PointerRoi(), Pitch(), aThreshold, SizeRoi(), aStreamCtx);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const opp::cuda::DevVarView<T> &aThreshold,
                                        const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    InvokeThresholdGTInplaceDevC(PointerRoi(), Pitch(), aThreshold.Pointer(), SizeRoi(), aStreamCtx);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare, ImageView<T> &aDst,
                                      const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case opp::CompareOp::Less:
            return ThresholdLT(aThreshold, aValue, aDst, aStreamCtx);
            break;
        case opp::CompareOp::Greater:
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
                                        const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeThresholdLTValSrcC(PointerRoi(), Pitch(), aThreshold, aValue, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                             aStreamCtx);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, const T &aValue, ImageView<T> &aDst,
                                        const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeThresholdGTValSrcC(PointerRoi(), Pitch(), aThreshold, aValue, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                             aStreamCtx);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare,
                                      const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case opp::CompareOp::Less:
            return ThresholdLT(aThreshold, aValue, aStreamCtx);
            break;
        case opp::CompareOp::Greater:
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
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold, const T &aValue, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    InvokeThresholdLTValInplaceC(PointerRoi(), Pitch(), aThreshold, aValue, SizeRoi(), aStreamCtx);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, const T &aValue, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    InvokeThresholdGTValInplaceC(PointerRoi(), Pitch(), aThreshold, aValue, SizeRoi(), aStreamCtx);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT,
                                          const T &aValueGT, ImageView<T> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeThresholdLTValGTValSrcC(PointerRoi(), Pitch(), aThresholdLT, aValueLT, aThresholdGT, aValueGT,
                                  aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT,
                                          const T &aValueGT, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    InvokeThresholdLTValGTValInplaceC(PointerRoi(), Pitch(), aThresholdLT, aValueLT, aThresholdGT, aValueGT, SizeRoi(),
                                      aStreamCtx);
    return *this;
}
#pragma endregion
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND