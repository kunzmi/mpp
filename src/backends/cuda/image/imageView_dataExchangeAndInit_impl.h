#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include "dataExchangeAndInit/conversionRelations.h"
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
#include <common/image/size2D.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace mpp::image::cuda
{
#pragma region Convert
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Convert(ImageView<TTo> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(!std::same_as<T, TTo>) && ConversionImplemented<T, TTo>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeConvert(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(!std::same_as<T, TTo>) && ConversionRoundImplemented<T, TTo>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeConvertRound(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aRoundingMode, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode, int aScaleFactor,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(!std::same_as<T, TTo>) && ConversionRoundScaleImplemented<T, TTo> && (!std::same_as<TTo, float>) &&
            (!std::same_as<TTo, double>) && (!std::same_as<TTo, Complex<float>>) &&
            (!std::same_as<TTo, Complex<double>>)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<complex_basetype_t<T>, float> &&
                  (std::same_as<complex_basetype_t<TTo>, HalfFp16> || std::same_as<complex_basetype_t<TTo>, BFloat16>))
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

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);
    InvokeConvertScaleRound(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aRoundingMode, scaleFactorFloat,
                            SizeRoi(), aStreamCtx);
    return aDst;
}
#pragma endregion

#pragma region Copy
/// <summary>
/// Copy image.
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::Copy(ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeCopy(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

/// <summary>
/// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::CopyMasked(ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                       const mpp::cuda::StreamCtx &aStreamCtx) const
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    InvokeCopyMask(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(),
                   aStreamCtx);

    return aDst;
}

/// <summary>
/// Copy channel aSrcChannel to channel aDstChannel of aDst.
/// </summary>
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Copy(Channel aSrcChannel, ImageView<TTo> &aDst, Channel aDstChannel,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_size_v<T> > 1) &&   //
            (vector_size_v<TTo> > 1) && //
            std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeCopyChannel(PointerRoi(), Pitch(), aSrcChannel, aDst.PointerRoi(), aDst.Pitch(), aDstChannel, SizeRoi(),
                      aStreamCtx);

    return aDst;
}

/// <summary>
/// Copy this single channel image to channel aDstChannel of aDst.
/// </summary>
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Copy(ImageView<TTo> &aDst, Channel aDstChannel,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_size_v<T> == 1) &&  //
            (vector_size_v<TTo> > 1) && //
            std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeCopyChannel(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aDstChannel, SizeRoi(), aStreamCtx);

    return aDst;
}

/// <summary>
/// Copy channel aSrcChannel to single channel image aDst.
/// </summary>
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Copy(Channel aSrcChannel, ImageView<TTo> &aDst,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_size_v<T> > 1) &&    //
            (vector_size_v<TTo> == 1) && //
            std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeCopyChannel(PointerRoi(), Pitch(), aSrcChannel, aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

/// <summary>
/// Copy packed image pixels to planar images.
/// </summary>
template <PixelType T>
void ImageView<T>::Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
                        ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(TwoChannel<T>)
{
    checkSameSize(ROI(), aDstChannel1.ROI());
    checkSameSize(ROI(), aDstChannel2.ROI());

    InvokeCopyPlanar(PointerRoi(), Pitch(), aDstChannel1.PointerRoi(), aDstChannel1.Pitch(), aDstChannel2.PointerRoi(),
                     aDstChannel2.Pitch(), SizeRoi(), aStreamCtx);
}

/// <summary>
/// Copy packed image pixels to planar images.
/// </summary>
template <PixelType T>
void ImageView<T>::Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
                        ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
                        ImageView<Vector1<remove_vector_t<T>>> &aDstChannel3,
                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(ThreeChannel<T>)
{
    checkSameSize(ROI(), aDstChannel1.ROI());
    checkSameSize(ROI(), aDstChannel2.ROI());
    checkSameSize(ROI(), aDstChannel3.ROI());

    InvokeCopyPlanar(PointerRoi(), Pitch(), aDstChannel1.PointerRoi(), aDstChannel1.Pitch(), aDstChannel2.PointerRoi(),
                     aDstChannel2.Pitch(), aDstChannel3.PointerRoi(), aDstChannel3.Pitch(), SizeRoi(), aStreamCtx);
}

/// <summary>
/// Copy packed image pixels to planar images.
/// </summary>
template <PixelType T>
void ImageView<T>::Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
                        ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
                        ImageView<Vector1<remove_vector_t<T>>> &aDstChannel3,
                        ImageView<Vector1<remove_vector_t<T>>> &aDstChannel4,
                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(FourChannelNoAlpha<T>)
{
    checkSameSize(ROI(), aDstChannel1.ROI());
    checkSameSize(ROI(), aDstChannel2.ROI());
    checkSameSize(ROI(), aDstChannel3.ROI());
    checkSameSize(ROI(), aDstChannel4.ROI());

    InvokeCopyPlanar(PointerRoi(), Pitch(), aDstChannel1.PointerRoi(), aDstChannel1.Pitch(), aDstChannel2.PointerRoi(),
                     aDstChannel2.Pitch(), aDstChannel3.PointerRoi(), aDstChannel3.Pitch(), aDstChannel4.PointerRoi(),
                     aDstChannel4.Pitch(), SizeRoi(), aStreamCtx);
}

/// <summary>
/// Copy planar image pixels to packed pixel image.
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2, ImageView<T> &aDst,
                                 const mpp::cuda::StreamCtx &aStreamCtx)
    requires(TwoChannel<T>)
{
    checkSameSize(aSrcChannel1.ROI(), aDst.ROI());
    checkSameSize(aSrcChannel2.ROI(), aDst.ROI());

    InvokeCopyPlanar(aSrcChannel1.PointerRoi(), aSrcChannel1.Pitch(), aSrcChannel2.PointerRoi(), aSrcChannel2.Pitch(),
                     aDst.PointerRoi(), aDst.Pitch(), aDst.SizeRoi(), aStreamCtx);

    return aDst;
}

/// <summary>
/// Copy planar image pixels to packed pixel image.
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel3, ImageView<T> &aDst,
                                 const mpp::cuda::StreamCtx &aStreamCtx)
    requires(ThreeChannel<T>)
{
    checkSameSize(aSrcChannel1.ROI(), aDst.ROI());
    checkSameSize(aSrcChannel2.ROI(), aDst.ROI());
    checkSameSize(aSrcChannel3.ROI(), aDst.ROI());

    InvokeCopyPlanar(aSrcChannel1.PointerRoi(), aSrcChannel1.Pitch(), aSrcChannel2.PointerRoi(), aSrcChannel2.Pitch(),
                     aSrcChannel3.PointerRoi(), aSrcChannel3.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aDst.SizeRoi(),
                     aStreamCtx);

    return aDst;
}

/// <summary>
/// Copy planar image pixels to packed pixel image.
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel3,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel4, ImageView<T> &aDst,
                                 const mpp::cuda::StreamCtx &aStreamCtx)
    requires(FourChannelNoAlpha<T>)
{
    checkSameSize(aSrcChannel1.ROI(), aDst.ROI());
    checkSameSize(aSrcChannel2.ROI(), aDst.ROI());
    checkSameSize(aSrcChannel3.ROI(), aDst.ROI());
    checkSameSize(aSrcChannel4.ROI(), aDst.ROI());

    InvokeCopyPlanar(aSrcChannel1.PointerRoi(), aSrcChannel1.Pitch(), aSrcChannel2.PointerRoi(), aSrcChannel2.Pitch(),
                     aSrcChannel3.PointerRoi(), aSrcChannel3.Pitch(), aSrcChannel4.PointerRoi(), aSrcChannel4.Pitch(),
                     aDst.PointerRoi(), aDst.Pitch(), aDst.SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion

#pragma region Copy (with border control)
template <PixelType T>
ImageView<T> &ImageView<T>::Copy(ImageView<T> &aDst, const Vector2<int> &aLowerBorderSize, BorderType aBorder,
                                 const mpp::cuda::StreamCtx &aStreamCtx) const
{
    InvokeCopyBorder(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aLowerBorderSize, aBorder, {0}, SizeRoi(),
                     aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Copy(ImageView<T> &aDst, const Vector2<int> &aLowerBorderSize, BorderType aBorder,
                                 const T &aConstant, const mpp::cuda::StreamCtx &aStreamCtx) const
{
    InvokeCopyBorder(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aLowerBorderSize, aBorder, aConstant,
                     SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion

#pragma region Copy subpix
template <PixelType T>
ImageView<T> &ImageView<T>::Copy(ImageView<T> &aDst, const Pixel32fC2 &aDelta, InterpolationMode aInterpolation,
                                 const mpp::cuda::StreamCtx &aStreamCtx) const
{
    InvokeCopySubpix(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aDelta, aInterpolation, SizeRoi(),
                     aStreamCtx);

    return aDst;
}
#pragma endregion

#pragma region Dup
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Dup(ImageView<TTo> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_size_v<T> == 1) &&
            (vector_size_v<TTo> > 1) && std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeDupSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion

#pragma region Scale
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<T> && RealOrComplexIntVector<TTo>
{
    checkSameSize(ROI(), aDst.ROI());

    using scaleType            = scalefactor_t<default_floating_compute_type_for_t<T>>;
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
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using scaleType            = scalefactor_t<default_floating_compute_type_for_t<T>>;
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
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<TTo>
{
    checkSameSize(ROI(), aDst.ROI());

    using scaleType            = scalefactor_t<default_floating_compute_type_for_t<T>>;
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
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(!std::same_as<T, TTo>)
{
    checkSameSize(ROI(), aDst.ROI());

    using scaleType        = scalefactor_t<default_floating_compute_type_for_t<T>>;
    const scaleType srcMin = static_cast<scaleType>(aSrcMin);
    const scaleType srcMax = static_cast<scaleType>(aSrcMax);
    const scaleType dstMin = static_cast<scaleType>(aDstMin);
    const scaleType dstMax = static_cast<scaleType>(aDstMax);
    const scaleType factor = (dstMax - dstMin) / (srcMax - srcMin);

    InvokeScale(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), factor, srcMin, dstMin, SizeRoi(), aStreamCtx);

    return aDst;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
#pragma endregion

#pragma region Set
template <PixelType T> ImageView<T> &ImageView<T>::Set(const T &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
{
    InvokeSetC(aConst, PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Set(const mpp::cuda::DevVarView<T> &aConst, const mpp::cuda::StreamCtx &aStreamCtx)
{
    InvokeSetDevC(aConst.Pointer(), PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SetMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aMask.ROI());
    InvokeSetCMask(aMask.PointerRoi(), aMask.Pitch(), aConst, PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SetMasked(const mpp::cuda::DevVarView<T> &aConst, const ImageView<Pixel8uC1> &aMask,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
{
    checkSameSize(ROI(), aMask.ROI());
    InvokeSetDevCMask(aMask.PointerRoi(), aMask.Pitch(), aConst.Pointer(), PointerRoi(), Pitch(), SizeRoi(),
                      aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Set(remove_vector_t<T> aConst, Channel aChannel, const mpp::cuda::StreamCtx &aStreamCtx)
    requires(vector_size_v<T> > 1)
{
    InvokeSetChannelC(aConst, aChannel, PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Set(const mpp::cuda::DevVarView<remove_vector_t<T>> &aConst, Channel aChannel,
                                const mpp::cuda::StreamCtx &aStreamCtx)
    requires(vector_size_v<T> > 1)
{
    InvokeSetChannelDevC(aConst.Pointer(), aChannel, PointerRoi(), Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region Swap Channel
/// <summary>
/// Swap channels
/// </summary>
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::SwapChannel(ImageView<TTo> &aDst,
                                          const ChannelList<vector_active_size_v<TTo>> &aDstChannels,
                                          const mpp::cuda::StreamCtx &aStreamCtx) const
    requires((vector_active_size_v<TTo> <= vector_active_size_v<T>)) && //
            (vector_size_v<T> >= 3) &&                                  //
            (vector_size_v<TTo> >= 3) &&                                //
            (!has_alpha_channel_v<TTo>) &&                              //
            (!has_alpha_channel_v<T>) &&                                //
            std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>
{
    checkSameSize(ROI(), aDst.ROI());
    for (size_t i = 0; i < vector_active_size_v<TTo>; i++)
    {
        if (!aDstChannels.data()[i].template IsInRange<T>())
        {
            throw INVALIDARGUMENT(aDstChannels,
                                  "Channel " << i << " in aDstChannels is out of range. Expected value in range 0.."
                                             << vector_active_size_v<T> - 1 << " but got: " << aDstChannels.data()[i]);
        }
    }

    InvokeSwapChannelSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aDstChannels, SizeRoi(), aStreamCtx);

    return aDst;
}

/// <summary>
/// Swap channels (inplace)
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::SwapChannel(const ChannelList<vector_active_size_v<T>> &aDstChannels,
                                        const mpp::cuda::StreamCtx &aStreamCtx)
    requires(vector_size_v<T> >= 3) && (!has_alpha_channel_v<T>)
{
    for (size_t i = 0; i < vector_active_size_v<T>; i++)
    {
        if (!aDstChannels.data()[i].template IsInRange<T>())
        {
            throw INVALIDARGUMENT(aDstChannels,
                                  "Channel " << i << " in aDstChannels is out of range. Expected value in range 0.."
                                             << vector_active_size_v<T> - 1 << " but got: " << aDstChannels.data()[i]);
        }
    }

    InvokeSwapChannelInplace(PointerRoi(), Pitch(), aDstChannels, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::SwapChannel(ImageView<TTo> &aDst,
                                          const ChannelList<vector_active_size_v<TTo>> &aDstChannels,
                                          remove_vector_t<T> aValue,
                                          const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_size_v<T> == 3) &&          //
            (vector_active_size_v<TTo> == 4) && //
            (!has_alpha_channel_v<TTo>) &&      //
            (!has_alpha_channel_v<T>) &&        //
            std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeSwapChannelSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aDstChannels, aValue, SizeRoi(),
                         aStreamCtx);

    return aDst;
}
#pragma endregion

#pragma region Transpose
template <PixelType T>
ImageView<T> &ImageView<T>::Transpose(ImageView<T> &aDst, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires NoAlpha<T>
{
    if (SizeRoi() != Size2D(aDst.SizeRoi().y, aDst.SizeRoi().x))
    {
        throw ROIEXCEPTION(
            "Width of destination image ROI must be the same as the height of the source image ROI, and height "
            "of the destination image ROI must be the same as the width of the source image ROI. Source ROI size: "
            << SizeRoi() << " provided destination image ROI size: " << aDst.SizeRoi());
    }

    InvokeTransposeSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aDst.SizeRoi(), aStreamCtx);

    return aDst;
}
#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND