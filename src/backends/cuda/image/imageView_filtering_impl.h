#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include "imageView.h"
#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/filtering/filteringKernel.h>
#include <backends/cuda/streamCtx.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/filterArea.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/opp_defs.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace opp::image::cuda
{
#pragma region FixedFilter
template <PixelType T>
ImageView<T> &ImageView<T>::FixedFilter(ImageView<T> &aDst, opp::FixedFilter aFilter, MaskSize aMaskSize,
                                        BorderType aBorder, T aConstant, Roi aAllowedReadRoi,
                                        const opp::cuda::StreamCtx &aStreamCtx) const
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aFilter)
    {
        case opp::FixedFilter::Gauss:
            InvokeGaussFixed(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                             aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::HighPass:
            InvokeHighpass(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                           aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::LowPass:
            InvokeLowpass(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                          aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::Laplace:
            InvokeLaplace(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                          aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::PrewittHoriz:
            InvokePrewittHoriz(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                               aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::PrewittVert:
            InvokePrewittVert(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                              aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::RobertsDown:
            InvokeRobertsDown(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                              aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::RobertsUp:
            InvokeRobertsUp(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                            aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::ScharrHoriz:
            InvokeScharrHoriz(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                              aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::ScharrVert:
            InvokeScharrVert(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                             aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::Sharpen:
            InvokeSharpen(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                          aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::SobelCross:
            InvokeSobelCross(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                             aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::SobelHoriz:
            InvokeSobelHoriz(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                             aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::SobelVert:
            InvokeSobelVert(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                            aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::SobelHorizSecond:
            InvokeSobelHorizSecond(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                                   aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::SobelVertSecond:
            InvokeSobelVertSecond(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                                  aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        default:
            break;
    }

    return aDst;
}

template <PixelType T>
ImageView<alternative_filter_output_type_for_t<T>> &ImageView<T>::FixedFilter(
    ImageView<alternative_filter_output_type_for_t<T>> &aDst, opp::FixedFilter aFilter, MaskSize aMaskSize,
    BorderType aBorder, T aConstant, Roi aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires(has_alternative_filter_output_type_for_v<T>)
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aFilter)
    {
        case opp::FixedFilter::HighPass:
            InvokeHighpass(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                           aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::Laplace:
            InvokeLaplace(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                          aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::PrewittHoriz:
            InvokePrewittHoriz(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                               aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::PrewittVert:
            InvokePrewittVert(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                              aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::RobertsDown:
            InvokeRobertsDown(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                              aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::RobertsUp:
            InvokeRobertsUp(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                            aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::ScharrHoriz:
            InvokeScharrHoriz(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                              aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::ScharrVert:
            InvokeScharrVert(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                             aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::Sharpen:
            InvokeSharpen(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                          aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::SobelCross:
            InvokeSobelCross(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                             aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::SobelHoriz:
            InvokeSobelHoriz(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                             aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::SobelVert:
            InvokeSobelVert(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                            aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::SobelHorizSecond:
            InvokeSobelHorizSecond(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                                   aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        case opp::FixedFilter::SobelVertSecond:
            InvokeSobelVertSecond(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aMaskSize, aBorder, aConstant,
                                  aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
            break;
        default:
            throw INVALIDARGUMENT(aFitler, "The filter "
                                               << aFilter
                                               << " is only implemented for the same pixel type for input and output.");
            break;
    }

    return aDst;
}

#pragma endregion

#pragma region SeparableFilter
template <PixelType T>
ImageView<T> &ImageView<T>::SeparableFilter(
    ImageView<T> &aDst, const opp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
    int aFilterSize, int aFilterCenter, BorderType aBorder, T aConstant, Roi aAllowedReadRoi,
    const opp::cuda::StreamCtx &aStreamCtx) const
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    constexpr filtertype_for_t<filter_compute_type_for_t<T>> one =
        static_cast<filtertype_for_t<filter_compute_type_for_t<T>>>(1);

    if (aFilterSize == 3 || aFilterSize == 5 || aFilterSize == 7 || aFilterSize == 9)
    {
        InvokeFixedSizeSeparableFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilter.Pointer(), one,
                                       aFilterSize, aFilterCenter, aBorder, aConstant, aAllowedReadRoi.Size(),
                                       roiOffset, SizeRoi(), aStreamCtx);
    }
    else
    {
        InvokeSeparableFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilter.Pointer(), one, aFilterSize,
                              aFilterCenter, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(),
                              aStreamCtx);
    }

    return aDst;
}

#pragma endregion

#pragma region ColumnFilter
template <PixelType T>
ImageView<T> &ImageView<T>::ColumnFilter(
    ImageView<T> &aDst, const opp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
    int aFilterSize, int aFilterCenter, BorderType aBorder, T aConstant, Roi aAllowedReadRoi,
    const opp::cuda::StreamCtx &aStreamCtx) const
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    constexpr filtertype_for_t<filter_compute_type_for_t<T>> one =
        static_cast<filtertype_for_t<filter_compute_type_for_t<T>>>(1);

    InvokeColumnCoefficientFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilter.Pointer(), one,
                                  aFilterSize, aFilterCenter, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset,
                                  SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<window_sum_result_type_t<T>> &ImageView<T>::ColumnWindowSum(
    ImageView<window_sum_result_type_t<T>> &aDst,
    complex_basetype_t<remove_vector_t<window_sum_result_type_t<T>>> aScalingValue, int aFilterSize, int aFilterCenter,
    BorderType aBorder, T aConstant, Roi aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeColumnWindowSum(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aScalingValue, aFilterSize,
                          aFilterCenter, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);

    return aDst;
}

#pragma endregion

#pragma region RowFilter
template <PixelType T>
ImageView<T> &ImageView<T>::RowFilter(
    ImageView<T> &aDst, const opp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
    int aFilterSize, int aFilterCenter, BorderType aBorder, T aConstant, Roi aAllowedReadRoi,
    const opp::cuda::StreamCtx &aStreamCtx) const
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    constexpr filtertype_for_t<filter_compute_type_for_t<T>> one =
        static_cast<filtertype_for_t<filter_compute_type_for_t<T>>>(1);

    InvokeRowCoefficientFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilter.Pointer(), one,
                               aFilterSize, aFilterCenter, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset,
                               SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<window_sum_result_type_t<T>> &ImageView<T>::RowWindowSum(
    ImageView<window_sum_result_type_t<T>> &aDst,
    complex_basetype_t<remove_vector_t<window_sum_result_type_t<T>>> aScalingValue, int aFilterSize, int aFilterCenter,
    BorderType aBorder, T aConstant, Roi aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeRowWindowSum(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aScalingValue, aFilterSize, aFilterCenter,
                       aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion

#pragma region BoxFilter
template <PixelType T>
ImageView<T> &ImageView<T>::BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                      T aConstant, Roi aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == 3 || aFilterArea.Size == 5 || aFilterArea.Size == 7 || aFilterArea.Size == 9)
    {
        // this implies aFilterArea.Size.x == aFilterArea.Size.y
        InvokeFixedSizeBoxFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea.Size.x,
                                 aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(),
                                 aStreamCtx);
    }
    else
    {
        InvokeBoxFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea, aBorder, aConstant,
                        aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::BoxFilter(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
    T aConstant, Roi aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T>
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == 3 || aFilterArea.Size == 5 || aFilterArea.Size == 7 || aFilterArea.Size == 9)
    {
        // this implies aFilterArea.Size.x == aFilterArea.Size.y
        InvokeFixedSizeBoxFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea.Size.x,
                                 aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(),
                                 aStreamCtx);
    }
    else
    {
        InvokeBoxFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea, aBorder, aConstant,
                        aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<Pixel32fC2> &ImageView<T>::BoxFilter(ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea,
                                               BorderType aBorder, T aConstant, Roi aAllowedReadRoi,
                                               const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8)
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeBoxAndBoxSquare(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea, aBorder, aConstant,
                          aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion

#pragma region Min/Max Filter
template <PixelType T>
ImageView<T> &ImageView<T>::MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                      T aConstant, Roi aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == 3 || aFilterArea.Size == 5 || aFilterArea.Size == 7 || aFilterArea.Size == 9)
    {
        // this implies aFilterArea.Size.x == aFilterArea.Size.y
        InvokeFixedSizeMaxFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea.Size.x,
                                 aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(),
                                 aStreamCtx);
    }
    else
    {
        InvokeMaxFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea, aBorder, aConstant,
                        aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                      T aConstant, Roi aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == 3 || aFilterArea.Size == 5 || aFilterArea.Size == 7 || aFilterArea.Size == 9)
    {
        // this implies aFilterArea.Size.x == aFilterArea.Size.y
        InvokeFixedSizeMinFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea.Size.x,
                                 aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(),
                                 aStreamCtx);
    }
    else
    {
        InvokeMinFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea, aBorder, aConstant,
                        aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }

    return aDst;
}

#pragma endregion

#pragma region Wiener Filter
/// <summary>
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                         const filter_compute_type_for_t<T> &aNoise, BorderType aBorder, T aConstant,
                                         Roi aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeWienerFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea, aNoise, aBorder, aConstant,
                       aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion

#pragma region Threshold Adaptive Box Filter
/// <summary>
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdAdaptiveBoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                       const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
                                                       const T &aValLE, BorderType aBorder, T aConstant,
                                                       Roi aAllowedReadRoi,
                                                       const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeThresholdAdaptiveBoxFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea, aDelta, aValGT,
                                     aValLE, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(),
                                     aStreamCtx);

    return aDst;
}
#pragma endregion

#pragma region Filter
/// <summary>
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::Filter(ImageView<T> &aDst,
                                   const opp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
                                   const FilterArea &aFilterArea, BorderType aBorder, T aConstant, Roi aAllowedReadRoi,
                                   const opp::cuda::StreamCtx &aStreamCtx) const
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterArea.Size == Vec2i{3, 3})
    {
        InvokeFixedSizeFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilter.Pointer(),
                              MaskSize::Mask_3x3, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                              roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{5, 5})
    {
        InvokeFixedSizeFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilter.Pointer(),
                              MaskSize::Mask_5x5, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                              roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{7, 7})
    {
        InvokeFixedSizeFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilter.Pointer(),
                              MaskSize::Mask_7x7, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                              roiOffset, SizeRoi(), aStreamCtx);
    }
    else if (aFilterArea.Size == Vec2i{9, 9})
    {
        InvokeFixedSizeFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilter.Pointer(),
                              MaskSize::Mask_9x9, aFilterArea.Center, aBorder, aConstant, aAllowedReadRoi.Size(),
                              roiOffset, SizeRoi(), aStreamCtx);
    }
    else
    {
        InvokeFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilter.Pointer(), aFilterArea, aBorder,
                     aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
    }

    return aDst;
}
#pragma endregion

#pragma region Bilateral Gauss Filter
template <PixelType T>
void ImageView<T>::PrecomputeBilateralGaussFilter(opp::cuda::DevVarView<float> &aPreCompGeomDistCoeff,
                                                  const FilterArea &aFilterArea, float aPosSquareSigma,
                                                  const opp::cuda::StreamCtx &aStreamCtx) const
{
    if (!aFilterArea.CheckIfValid())
    {
        throw INVALIDARGUMENT(aFilterArea, "Invalid filter area: " << aFilterArea);
    }
    if (aPreCompGeomDistCoeff.Size() < aFilterArea.Size.TotalSize())
    {
        throw INVALIDARGUMENT(aPreCompGeomDistCoeff,
                              "The provided filter array is not large enough for a filter size of " << aFilterArea.Size
                                                                                                    << " pixels.");
    }

    InvokePrecomputeBilateralGaussFilter(aPreCompGeomDistCoeff.Pointer(), aFilterArea, aPosSquareSigma, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                 const opp::cuda::DevVarView<float> &aPreCompGeomDistCoeff,
                                                 float aValSquareSigma, BorderType aBorder, T aConstant,
                                                 Roi aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> &&
             (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>)
{
    if (!aFilterArea.CheckIfValid())
    {
        throw INVALIDARGUMENT(aFilterArea, "Invalid filter area: " << aFilterArea);
    }
    if (aPreCompGeomDistCoeff.Size() < aFilterArea.Size.TotalSize())
    {
        throw INVALIDARGUMENT(aPreCompGeomDistCoeff,
                              "The provided filter array is not large enough for a filter size of " << aFilterArea.Size
                                                                                                    << " pixels.");
    }
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeBilateralGaussFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea,
                               aPreCompGeomDistCoeff.Pointer(), aValSquareSigma, Norm::L1, aBorder, aConstant,
                               aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);

    return aDst;
}
/// <summary>
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                 const opp::cuda::DevVarView<float> &aPreCompGeomDistCoeff,
                                                 float aValSquareSigma, opp::Norm aNorm, BorderType aBorder,
                                                 T aConstant, Roi aAllowedReadRoi,
                                                 const opp::cuda::StreamCtx &aStreamCtx) const
    requires(!SingleChannel<T>) && RealVector<T> &&
            (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>)
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeBilateralGaussFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilterArea,
                               aPreCompGeomDistCoeff.Pointer(), aValSquareSigma, aNorm, aBorder, aConstant,
                               aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion
#pragma region Gradient Vector
/// <summary>
/// </summary>
template <PixelType T>
void ImageView<T>::GradientVectorSobel(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                       ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                       ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                       BorderType aBorder, T aConstant, Roi aAllowedReadRoi,
                                       const opp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    // find first not nullptr output:
    Size2D refSize = aDstX.SizeRoi();
    if (aDstX.Pointer() == nullptr)
    {
        refSize = aDstY.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr)
    {
        refSize = aDstMag.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr)
    {
        refSize = aDstAngle.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr)
    {
        refSize = aDstCovariance.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr && aDstCovariance.Pointer() == nullptr)
    {
        throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle aDstCovariance,
                              "All output images are nullptr, at least one output must be provided.");
    }

    if (aDstX.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstX.SizeRoi());
    }
    if (aDstY.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstY.SizeRoi());
    }
    if (aDstMag.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstMag.SizeRoi());
    }
    if (aDstAngle.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstAngle.SizeRoi());
    }
    if (aDstCovariance.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstCovariance.SizeRoi());
    }

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeGradientVectorSobel(allowedPtr, Pitch(), aDstX.PointerRoi(), aDstX.Pitch(), aDstY.PointerRoi(), aDstY.Pitch(),
                              aDstMag.PointerRoi(), aDstMag.Pitch(), aDstAngle.PointerRoi(), aDstAngle.Pitch(),
                              aDstCovariance.PointerRoi(), aDstCovariance.Pitch(), aNorm, aMaskSize, aBorder, aConstant,
                              aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
}
/// <summary>
/// </summary>
template <PixelType T>
void ImageView<T>::GradientVectorSobel(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                       ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                       ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                       BorderType aBorder, T aConstant, Roi aAllowedReadRoi,
                                       const opp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    // find first not nullptr output:
    Size2D refSize = aDstX.SizeRoi();
    if (aDstX.Pointer() == nullptr)
    {
        refSize = aDstY.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr)
    {
        refSize = aDstMag.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr)
    {
        refSize = aDstAngle.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr)
    {
        refSize = aDstCovariance.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr && aDstCovariance.Pointer() == nullptr)
    {
        throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle aDstCovariance,
                              "All output images are nullptr, at least one output must be provided.");
    }

    if (aDstX.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstX.SizeRoi());
    }
    if (aDstY.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstY.SizeRoi());
    }
    if (aDstMag.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstMag.SizeRoi());
    }
    if (aDstAngle.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstAngle.SizeRoi());
    }
    if (aDstCovariance.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstCovariance.SizeRoi());
    }

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeGradientVectorSobel(allowedPtr, Pitch(), aDstX.PointerRoi(), aDstX.Pitch(), aDstY.PointerRoi(), aDstY.Pitch(),
                              aDstMag.PointerRoi(), aDstMag.Pitch(), aDstAngle.PointerRoi(), aDstAngle.Pitch(),
                              aDstCovariance.PointerRoi(), aDstCovariance.Pitch(), aNorm, aMaskSize, aBorder, aConstant,
                              aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
}

/// <summary>
/// </summary>
template <PixelType T>
void ImageView<T>::GradientVectorScharr(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                        ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                        ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                        BorderType aBorder, T aConstant, Roi aAllowedReadRoi,
                                        const opp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    // find first not nullptr output:
    Size2D refSize = aDstX.SizeRoi();
    if (aDstX.Pointer() == nullptr)
    {
        refSize = aDstY.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr)
    {
        refSize = aDstMag.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr)
    {
        refSize = aDstAngle.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr)
    {
        refSize = aDstCovariance.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr && aDstCovariance.Pointer() == nullptr)
    {
        throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle aDstCovariance,
                              "All output images are nullptr, at least one output must be provided.");
    }

    if (aDstX.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstX.SizeRoi());
    }
    if (aDstY.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstY.SizeRoi());
    }
    if (aDstMag.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstMag.SizeRoi());
    }
    if (aDstAngle.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstAngle.SizeRoi());
    }
    if (aDstCovariance.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstCovariance.SizeRoi());
    }

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeGradientVectorScharr(allowedPtr, Pitch(), aDstX.PointerRoi(), aDstX.Pitch(), aDstY.PointerRoi(),
                               aDstY.Pitch(), aDstMag.PointerRoi(), aDstMag.Pitch(), aDstAngle.PointerRoi(),
                               aDstAngle.Pitch(), aDstCovariance.PointerRoi(), aDstCovariance.Pitch(), aNorm, aMaskSize,
                               aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
}
/// <summary>
/// </summary>
template <PixelType T>
void ImageView<T>::GradientVectorScharr(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                        ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                        ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                        BorderType aBorder, T aConstant, Roi aAllowedReadRoi,
                                        const opp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    // find first not nullptr output:
    Size2D refSize = aDstX.SizeRoi();
    if (aDstX.Pointer() == nullptr)
    {
        refSize = aDstY.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr)
    {
        refSize = aDstMag.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr)
    {
        refSize = aDstAngle.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr)
    {
        refSize = aDstCovariance.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr && aDstCovariance.Pointer() == nullptr)
    {
        throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle aDstCovariance,
                              "All output images are nullptr, at least one output must be provided.");
    }

    if (aDstX.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstX.SizeRoi());
    }
    if (aDstY.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstY.SizeRoi());
    }
    if (aDstMag.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstMag.SizeRoi());
    }
    if (aDstAngle.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstAngle.SizeRoi());
    }
    if (aDstCovariance.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstCovariance.SizeRoi());
    }

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeGradientVectorScharr(allowedPtr, Pitch(), aDstX.PointerRoi(), aDstX.Pitch(), aDstY.PointerRoi(),
                               aDstY.Pitch(), aDstMag.PointerRoi(), aDstMag.Pitch(), aDstAngle.PointerRoi(),
                               aDstAngle.Pitch(), aDstCovariance.PointerRoi(), aDstCovariance.Pitch(), aNorm, aMaskSize,
                               aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
}

/// <summary>
/// </summary>
template <PixelType T>
void ImageView<T>::GradientVectorPrewitt(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                         ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                         ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                         BorderType aBorder, T aConstant, Roi aAllowedReadRoi,
                                         const opp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    // find first not nullptr output:
    Size2D refSize = aDstX.SizeRoi();
    if (aDstX.Pointer() == nullptr)
    {
        refSize = aDstY.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr)
    {
        refSize = aDstMag.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr)
    {
        refSize = aDstAngle.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr)
    {
        refSize = aDstCovariance.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr && aDstCovariance.Pointer() == nullptr)
    {
        throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle aDstCovariance,
                              "All output images are nullptr, at least one output must be provided.");
    }

    if (aDstX.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstX.SizeRoi());
    }
    if (aDstY.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstY.SizeRoi());
    }
    if (aDstMag.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstMag.SizeRoi());
    }
    if (aDstAngle.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstAngle.SizeRoi());
    }
    if (aDstCovariance.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstCovariance.SizeRoi());
    }

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeGradientVectorPrewitt(
        allowedPtr, Pitch(), aDstX.PointerRoi(), aDstX.Pitch(), aDstY.PointerRoi(), aDstY.Pitch(), aDstMag.PointerRoi(),
        aDstMag.Pitch(), aDstAngle.PointerRoi(), aDstAngle.Pitch(), aDstCovariance.PointerRoi(), aDstCovariance.Pitch(),
        aNorm, aMaskSize, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
}
/// <summary>
/// </summary>
template <PixelType T>
void ImageView<T>::GradientVectorPrewitt(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                         ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                         ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                         BorderType aBorder, T aConstant, Roi aAllowedReadRoi,
                                         const opp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    // find first not nullptr output:
    Size2D refSize = aDstX.SizeRoi();
    if (aDstX.Pointer() == nullptr)
    {
        refSize = aDstY.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr)
    {
        refSize = aDstMag.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr)
    {
        refSize = aDstAngle.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr)
    {
        refSize = aDstCovariance.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr && aDstCovariance.Pointer() == nullptr)
    {
        throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle aDstCovariance,
                              "All output images are nullptr, at least one output must be provided.");
    }

    if (aDstX.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstX.SizeRoi());
    }
    if (aDstY.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstY.SizeRoi());
    }
    if (aDstMag.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstMag.SizeRoi());
    }
    if (aDstAngle.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstAngle.SizeRoi());
    }
    if (aDstCovariance.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstCovariance.SizeRoi());
    }

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeGradientVectorPrewitt(
        allowedPtr, Pitch(), aDstX.PointerRoi(), aDstX.Pitch(), aDstY.PointerRoi(), aDstY.Pitch(), aDstMag.PointerRoi(),
        aDstMag.Pitch(), aDstAngle.PointerRoi(), aDstAngle.Pitch(), aDstCovariance.PointerRoi(), aDstCovariance.Pitch(),
        aNorm, aMaskSize, aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region Unsharp Filter
template <PixelType T>
ImageView<T> &ImageView<T>::UnsharpFilter(
    ImageView<T> &aDst, const opp::cuda::DevVarView<filtertype_for_t<filter_compute_type_for_t<T>>> &aFilter,
    int aFilterSize, int aFilterCenter, remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
    remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold, BorderType aBorder, T aConstant,
    Roi aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    if (aFilterSize == 3 || aFilterSize == 5 || aFilterSize == 7 || aFilterSize == 9)
    {
        InvokeFixedSizeUnsharpFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilter.Pointer(), aWeight,
                                     aThreshold, aFilterSize, aFilterCenter, aBorder, aConstant, aAllowedReadRoi.Size(),
                                     roiOffset, SizeRoi(), aStreamCtx);
    }
    else
    {
        InvokeUnsharpFilter(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aFilter.Pointer(), aWeight,
                            aThreshold, aFilterSize, aFilterCenter, aBorder, aConstant, aAllowedReadRoi.Size(),
                            roiOffset, SizeRoi(), aStreamCtx);
    }

    return aDst;
}
#pragma endregion

#pragma region Harris Corner Response
template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::HarrisCornerResponse(ImageView<Pixel32fC1> &aDst, const FilterArea aAvgWindowSize,
                                                          float aK, float aScale, BorderType aBorder, T aConstant,
                                                          Roi aAllowedReadRoi,
                                                          const opp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<T, Pixel32fC4>
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeHarrisCornerResponse(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aAvgWindowSize, aK, aScale,
                               aBorder, aConstant, aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);

    return aDst;
}

#pragma endregion

#pragma region Canny edge
template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CannyEdge(const ImageView<Pixel32fC1> &aSrcAngle, ImageView<Pixel8uC1> &aTemp,
                                              ImageView<Pixel8uC1> &aDst, T aLowThreshold, T aHighThreshold,
                                              Roi aAllowedReadRoi, const opp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<T, Pixel16sC1> || std::same_as<T, Pixel32fC1>
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeCannyEdgeMaxSupression(allowedPtr, Pitch(), aSrcAngle.PointerRoi(), aSrcAngle.Pitch(), aTemp.PointerRoi(),
                                 aTemp.Pitch(), aLowThreshold, aHighThreshold, aAllowedReadRoi.Size(), roiOffset,
                                 SizeRoi(), aStreamCtx);

    InvokeCannyEdgeHysteresis(aTemp.PointerRoi(), aTemp.Pitch(), aSrcAngle.PointerRoi(), aSrcAngle.Pitch(),
                              aDst.PointerRoi(), aDst.Pitch(), aDst.SizeRoi(), Vector2<int>(0, 0), SizeRoi(),
                              aStreamCtx);

    return aDst;
}
#pragma endregion
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND