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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND