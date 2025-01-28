#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include "imageView.h"
#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/dataExchangeAndInit/dataExchangeAndInitKernel.h>
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

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Convert(ImageView<TTo> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires(!std::same_as<T, TTo>) &&
            (RealOrComplexIntVector<T> || (std::same_as<complex_basetype_t<remove_vector_t<T>>, float> &&
                                           (std::same_as<complex_basetype_t<remove_vector_t<TTo>>, BFloat16> ||
                                            std::same_as<complex_basetype_t<remove_vector_t<TTo>>, HalfFp16>)))
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeConvert(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode,
                                      const opp::cuda::StreamCtx &aStreamCtx)
    requires(!std::same_as<T, TTo>) && RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeConvertRound(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aRoundingMode, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode, int aScaleFactor,
                                      const opp::cuda::StreamCtx &aStreamCtx)
    requires(!std::same_as<T, TTo>) && (!std::same_as<TTo, float>) && (!std::same_as<TTo, double>) &&
            (!std::same_as<TTo, Complex<float>>) && (!std::same_as<TTo, Complex<double>>)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<complex_basetype_t<T>, float> &&
                  (std::same_as<complex_basetype_t<T>, HalfFp16> || std::same_as<complex_basetype_t<T>, BFloat16>))
    {
        if (aRoundingMode == RoundingMode::NearestTiesAwayFromZero)
        {
            throw INVALIDARGUMENT(
                aRoundingMode,
                "Unsupported rounding mode: "
                    << aRoundingMode
                    << ". Only NearestTiesToEven, TowardZero, TowardNegativeInfinity and TowardPositiveInfinity are "
                       "supported for float to BFloat16 or HalfFp16 conversions on Cuda devices.");
        }
    }

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);
    InvokeConvertScaleRound(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aRoundingMode, scaleFactorFloat,
                            SizeRoi(), aStreamCtx);
    return aDst;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, const opp::cuda::StreamCtx &aStreamCtx)
    requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<T> && RealOrComplexIntVector<TTo>
{
    checkSameSize(ROI(), aDst.ROI());

    using scaleType            = scalefactor_t<default_compute_type_for_t<T>>;
    constexpr scaleType srcMin = static_cast<scaleType>(numeric_limits<T>::lowest());
    constexpr scaleType srcMax = static_cast<scaleType>(numeric_limits<T>::max());
    constexpr scaleType dstMin = static_cast<scaleType>(numeric_limits<TTo>::lowest());
    constexpr scaleType dstMax = static_cast<scaleType>(numeric_limits<TTo>::max());
    constexpr scaleType factor = (dstMax - dstMin) / (srcMax - srcMin);

    InvokeScale(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), factor, srcMin, dstMin, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax,
                                    const opp::cuda::StreamCtx &aStreamCtx)
    requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using scaleType            = scalefactor_t<default_compute_type_for_t<T>>;
    constexpr scaleType srcMin = static_cast<scaleType>(numeric_limits<T>::lowest());
    constexpr scaleType srcMax = static_cast<scaleType>(numeric_limits<T>::max());
    const scaleType dstMin     = static_cast<scaleType>(aDstMin);
    const scaleType dstMax     = static_cast<scaleType>(aDstMax);
    const scaleType factor     = (dstMax - dstMin) / (srcMax - srcMin);

    InvokeScale(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), factor, srcMin, dstMin, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                                    const opp::cuda::StreamCtx &aStreamCtx)
    requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<TTo>
{
    checkSameSize(ROI(), aDst.ROI());

    using scaleType            = scalefactor_t<default_compute_type_for_t<T>>;
    const scaleType srcMin     = static_cast<scaleType>(aSrcMin);
    const scaleType srcMax     = static_cast<scaleType>(aSrcMax);
    constexpr scaleType dstMin = static_cast<scaleType>(numeric_limits<TTo>::lowest());
    constexpr scaleType dstMax = static_cast<scaleType>(numeric_limits<TTo>::max());
    const scaleType factor     = (dstMax - dstMin) / (srcMax - srcMin);

    InvokeScale(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), factor, srcMin, dstMin, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                                    scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax,
                                    const opp::cuda::StreamCtx &aStreamCtx)
    requires(!std::same_as<T, TTo>)
{
    checkSameSize(ROI(), aDst.ROI());

    using scaleType        = scalefactor_t<default_compute_type_for_t<T>>;
    const scaleType srcMin = static_cast<scaleType>(aSrcMin);
    const scaleType srcMax = static_cast<scaleType>(aSrcMax);
    const scaleType dstMin = static_cast<scaleType>(aDstMin);
    const scaleType dstMax = static_cast<scaleType>(aDstMax);
    const scaleType factor = (dstMax - dstMin) / (srcMax - srcMin);

    InvokeScale(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), factor, srcMin, dstMin, SizeRoi(), aStreamCtx);

    return aDst;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

template <PixelType T> ImageView<T> &ImageView<T>::Set(const T &aConst, const opp::cuda::StreamCtx &aStreamCtx)
{
    InvokeSetC(aConst, PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Set(const opp::cuda::DevVarView<T> &aConst, const opp::cuda::StreamCtx &aStreamCtx)
{
    InvokeSetDevC(aConst.Pointer(), PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Set(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aMask.ROI());
    InvokeSetCMask(aMask.PointerRoi(), aMask.Pitch(), aConst, PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Set(const opp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                const opp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aMask.ROI());
    InvokeSetDevCMask(aMask.PointerRoi(), aMask.Pitch(), aConst.Pointer(), PointerRoi(), Pitch(), SizeRoi(),
                      aStreamCtx);

    return *this;
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND