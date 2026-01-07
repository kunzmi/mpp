#pragma once
#include "conversionRelations.h"
#include "scaleRelations.h"
#include <backends/simple_cpu/image/convertScaleRoundComputeType.h>
#include <backends/simple_cpu/image/forEachPixel.h>
#include <backends/simple_cpu/image/forEachPixelMasked.h>
#include <backends/simple_cpu/image/forEachPixelPlanar.h>
#include <backends/simple_cpu/image/forEachPixelSingleChannel.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/image/scaleComputeType.h>
#include <backends/simple_cpu/operator_random.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/dataExchangeAndInit/operators.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/border.h>
#include <common/image/channel.h>
#include <common/image/channelList.h>
#include <common/image/functors/constantFunctor.h>
#include <common/image/functors/convertFunctor.h>
#include <common/image/functors/convertScaleFunctor.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/functors/inplaceConstantFunctor.h>
#include <common/image/functors/inplaceConstantScaleFunctor.h>
#include <common/image/functors/inplaceDevConstantFunctor.h>
#include <common/image/functors/inplaceDevConstantScaleFunctor.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/inplaceSrcScaleFunctor.h>
#include <common/image/functors/scaleConversionFunctor.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcConstantScaleFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
#include <common/image/functors/srcDevConstantScaleFunctor.h>
#include <common/image/functors/srcDstAsSrcFunctor.h>
#include <common/image/functors/srcFunctor.h>
#include <common/image/functors/srcPlanar2Functor.h>
#include <common/image/functors/srcPlanar3Functor.h>
#include <common/image/functors/srcPlanar4Functor.h>
#include <common/image/functors/srcScaleFunctor.h>
#include <common/image/functors/srcSingleChannelFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/functors/srcSrcScaleFunctor.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/safeCast.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vector1.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace mpp::image::cpuSimple
{
#pragma region Convert
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Convert(ImageView<TTo> &aDst) const
    requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && ConversionImplemented<T, TTo>
{
    checkSameSize(ROI(), aDst.ROI());

    using convert = ConvertFunctor<1, T, TTo, RoundingMode::NearestTiesToEven>;

    const convert functor(PointerRoi(), Pitch());
    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode) const
    requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && ConversionRoundImplemented<T, TTo> &&
            RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    if constexpr (std::same_as<remove_vector_t<T>, float> && std::same_as<remove_vector_t<TTo>, BFloat16>)
    {
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using convert = ConvertFunctor<1, T, TTo, RoundingMode::NearestTiesToEven>;
                const convert functor(PointerRoi(), Pitch());
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using convert = ConvertFunctor<1, T, TTo, RoundingMode::TowardZero>;
                const convert functor(PointerRoi(), Pitch());
                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Conversion from float32 to BFloat16 only support rounding modes "
                                                         << mpp::RoundingMode::NearestTiesToEven << " and "
                                                         << mpp::RoundingMode::TowardZero
                                                         << " but provided rounding mode is: " << aRoundingMode);
        }
    }
    else if constexpr (std::same_as<remove_vector_t<T>, float> && std::same_as<remove_vector_t<TTo>, HalfFp16>)
    {
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using convert = ConvertFunctor<1, T, TTo, RoundingMode::NearestTiesToEven>;
                const convert functor(PointerRoi(), Pitch());
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using convert = ConvertFunctor<1, T, TTo, RoundingMode::TowardZero>;
                const convert functor(PointerRoi(), Pitch());
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using convert = ConvertFunctor<1, T, TTo, RoundingMode::TowardNegativeInfinity>;
                const convert functor(PointerRoi(), Pitch());
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using convert = ConvertFunctor<1, T, TTo, RoundingMode::TowardPositiveInfinity>;
                const convert functor(PointerRoi(), Pitch());
                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(
                    aRoundingMode,
                    "Unsupported rounding mode for conversion from Float32 to Half-Float16: " << aRoundingMode);
        }
    }
    else
    {
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using convert = ConvertFunctor<1, T, TTo, RoundingMode::NearestTiesToEven>;
                const convert functor(PointerRoi(), Pitch());
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using convert = ConvertFunctor<1, T, TTo, RoundingMode::NearestTiesAwayFromZero>;
                const convert functor(PointerRoi(), Pitch());
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using convert = ConvertFunctor<1, T, TTo, RoundingMode::TowardZero>;
                const convert functor(PointerRoi(), Pitch());
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using convert = ConvertFunctor<1, T, TTo, RoundingMode::TowardNegativeInfinity>;
                const convert functor(PointerRoi(), Pitch());
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using convert = ConvertFunctor<1, T, TTo, RoundingMode::TowardPositiveInfinity>;
                const convert functor(PointerRoi(), Pitch());
                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode, int aScaleFactor) const
    requires(!std::same_as<T, TTo>) &&
            (vector_size_v<T> == vector_size_v<TTo>) && ConversionRoundScaleImplemented<T, TTo> &&
            (!std::same_as<TTo, float>) && (!std::same_as<TTo, double>) && (!std::same_as<TTo, Complex<float>>) &&
            (!std::same_as<TTo, Complex<double>>)
{
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = compute_type_convertScaleRound_for_t<T, TTo>;

    if constexpr (RealOrComplexIntVector<ComputeT>)
    {
        if (scaleFactorFloat >= 1.0)
        {
            // integer to integer conversion with intgere scaling, so no rounding:
            using ScalerT = mpp::Scale<ComputeT, false>;
            const ScalerT scaler(scaleFactorFloat);
            using convert = ConvertScaleFunctor<1, T, ComputeT, TTo, ScalerT, RoundingMode::None>;
            const convert functor(PointerRoi(), Pitch(), scaler);
            forEachPixel(aDst, functor);
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using ScalerT = mpp::Scale<ComputeT, true, mpp::RoundingMode::NearestTiesToEven>;
                    const ScalerT scaler(scaleFactorFloat);
                    using convert = ConvertScaleFunctor<1, T, ComputeT, TTo, ScalerT, RoundingMode::None>;
                    const convert functor(PointerRoi(), Pitch(), scaler);
                    forEachPixel(aDst, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using ScalerT = mpp::Scale<ComputeT, true, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    const ScalerT scaler(scaleFactorFloat);
                    using convert = ConvertScaleFunctor<1, T, ComputeT, TTo, ScalerT, RoundingMode::None>;
                    const convert functor(PointerRoi(), Pitch(), scaler);
                    forEachPixel(aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using ScalerT = mpp::Scale<ComputeT, true, mpp::RoundingMode::TowardZero>;
                    const ScalerT scaler(scaleFactorFloat);
                    using convert = ConvertScaleFunctor<1, T, ComputeT, TTo, ScalerT, RoundingMode::None>;
                    const convert functor(PointerRoi(), Pitch(), scaler);
                    forEachPixel(aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using ScalerT = mpp::Scale<ComputeT, true, mpp::RoundingMode::TowardNegativeInfinity>;
                    const ScalerT scaler(scaleFactorFloat);
                    using convert = ConvertScaleFunctor<1, T, ComputeT, TTo, ScalerT, RoundingMode::None>;
                    const convert functor(PointerRoi(), Pitch(), scaler);
                    forEachPixel(aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using ScalerT = mpp::Scale<ComputeT, true, mpp::RoundingMode::TowardPositiveInfinity>;
                    const ScalerT scaler(scaleFactorFloat);
                    using convert = ConvertScaleFunctor<1, T, ComputeT, TTo, ScalerT, RoundingMode::None>;
                    const convert functor(PointerRoi(), Pitch(), scaler);
                    forEachPixel(aDst, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }
    else
    {

        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using convert = ConvertScaleFunctor<1, T, ComputeT, TTo, ScalerT, RoundingMode::NearestTiesToEven>;
                const convert functor(PointerRoi(), Pitch(), scaler);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using convert =
                    ConvertScaleFunctor<1, T, ComputeT, TTo, ScalerT, RoundingMode::NearestTiesAwayFromZero>;
                const convert functor(PointerRoi(), Pitch(), scaler);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using convert = ConvertScaleFunctor<1, T, ComputeT, TTo, ScalerT, RoundingMode::TowardZero>;
                const convert functor(PointerRoi(), Pitch(), scaler);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using convert = ConvertScaleFunctor<1, T, ComputeT, TTo, ScalerT, RoundingMode::TowardNegativeInfinity>;
                const convert functor(PointerRoi(), Pitch(), scaler);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using convert = ConvertScaleFunctor<1, T, ComputeT, TTo, ScalerT, RoundingMode::TowardPositiveInfinity>;
                const convert functor(PointerRoi(), Pitch(), scaler);
                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }

    return aDst;
}
#pragma endregion

#pragma region Copy
/// <summary>
/// Copy image.
/// </summary>
template <PixelType T> ImageView<T> &ImageView<T>::Copy(ImageView<T> &aDst) const
{
    checkSameSize(ROI(), aDst.ROI());

    using copySrc = SrcFunctor<1, T, T, T, mpp::Copy<T, T>, RoundingMode::None>;

    const mpp::Copy<T, T> op;

    const copySrc functor(PointerRoi(), Pitch(), op);
    forEachPixel(aDst, functor);

    return aDst;
}

/// <summary>
/// Copy image with mask. Pixels with mask == 0 remain untouched in destination image.
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::CopyMasked(ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
{
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using copySrc = SrcFunctor<1, T, T, T, mpp::Copy<T, T>, RoundingMode::None>;

    const mpp::Copy<T, T> op;

    const copySrc functor(PointerRoi(), Pitch(), op);
    forEachPixel(aMask, aDst, functor);

    return aDst;
}

/// <summary>
/// Copy channel aSrcChannel to channel aDstChannel of aDst.
/// </summary>
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Copy(Channel aSrcChannel, ImageView<TTo> &aDst, Channel aDstChannel) const
    requires(vector_size_v<T> > 1) &&   //
            (vector_size_v<TTo> > 1) && //
            std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = Vector1<remove_vector_t<T>>;
    using copySrc =
        SrcSingleChannelFunctor<1, T, ComputeT, ComputeT, mpp::Copy<ComputeT, ComputeT>, RoundingMode::None>;

    const mpp::Copy<ComputeT, ComputeT> op;

    const copySrc functor(PointerRoi(), Pitch(), aSrcChannel, op);
    forEachPixelSingleChannel(aDst, aDstChannel, functor);

    return aDst;
}

/// <summary>
/// Copy this single channel image to channel aDstChannel of aDst.
/// </summary>
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Copy(ImageView<TTo> &aDst, Channel aDstChannel) const
    requires(vector_size_v<T> == 1) &&  //
            (vector_size_v<TTo> > 1) && //
            std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>
{
    checkSameSize(ROI(), aDst.ROI());

    using copySrc = SrcFunctor<1, T, T, T, mpp::Copy<T, T>, RoundingMode::None>;

    const mpp::Copy<T, T> op;

    const copySrc functor(PointerRoi(), Pitch(), op);
    forEachPixelSingleChannel(aDst, aDstChannel, functor);

    return aDst;
}

/// <summary>
/// Copy channel aSrcChannel to single channel image aDst.
/// </summary>
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Copy(Channel aSrcChannel, ImageView<TTo> &aDst) const
    requires(vector_size_v<T> > 1) &&    //
            (vector_size_v<TTo> == 1) && //
            std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = Vector1<remove_vector_t<T>>;
    using copySrc =
        SrcSingleChannelFunctor<1, T, ComputeT, ComputeT, mpp::Copy<ComputeT, ComputeT>, RoundingMode::None>;

    const mpp::Copy<ComputeT, ComputeT> op;

    const copySrc functor(PointerRoi(), Pitch(), aSrcChannel, op);
    forEachPixel(aDst, functor);

    return aDst;
}

/// <summary>
/// Copy packed image pixels to planar images.
/// </summary>
template <PixelType T>
void ImageView<T>::Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
                        ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2) const
    requires(TwoChannel<T>)
{
    checkSameSize(ROI(), aDstChannel1.ROI());
    checkSameSize(ROI(), aDstChannel2.ROI());

    using copySrc = SrcFunctor<1, T, T, T, mpp::Copy<T, T>, RoundingMode::None>;

    const mpp::Copy<T, T> op;

    const copySrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<T>(aDstChannel1, aDstChannel2, functor);
}

/// <summary>
/// Copy packed image pixels to planar images.
/// </summary>
template <PixelType T>
void ImageView<T>::Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
                        ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
                        ImageView<Vector1<remove_vector_t<T>>> &aDstChannel3) const
    requires(ThreeChannel<T>)
{
    checkSameSize(ROI(), aDstChannel1.ROI());
    checkSameSize(ROI(), aDstChannel2.ROI());
    checkSameSize(ROI(), aDstChannel3.ROI());

    using copySrc = SrcFunctor<1, T, T, T, mpp::Copy<T, T>, RoundingMode::None>;

    const mpp::Copy<T, T> op;

    const copySrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<T>(aDstChannel1, aDstChannel2, aDstChannel3, functor);
}

/// <summary>
/// Copy packed image pixels to planar images.
/// </summary>
template <PixelType T>
void ImageView<T>::Copy(ImageView<Vector1<remove_vector_t<T>>> &aDstChannel1,
                        ImageView<Vector1<remove_vector_t<T>>> &aDstChannel2,
                        ImageView<Vector1<remove_vector_t<T>>> &aDstChannel3,
                        ImageView<Vector1<remove_vector_t<T>>> &aDstChannel4) const
    requires(FourChannelNoAlpha<T>)
{
    checkSameSize(ROI(), aDstChannel1.ROI());
    checkSameSize(ROI(), aDstChannel2.ROI());
    checkSameSize(ROI(), aDstChannel3.ROI());
    checkSameSize(ROI(), aDstChannel4.ROI());

    using copySrc = SrcFunctor<1, T, T, T, mpp::Copy<T, T>, RoundingMode::None>;

    const mpp::Copy<T, T> op;

    const copySrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<T>(aDstChannel1, aDstChannel2, aDstChannel3, aDstChannel4, functor);
}

/// <summary>
/// Copy planar image pixels to packed pixel image.
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2, ImageView<T> &aDst)
    requires(TwoChannel<T>)
{
    checkSameSize(aSrcChannel1.ROI(), aDst.ROI());
    checkSameSize(aSrcChannel2.ROI(), aDst.ROI());

    using copySrc = SrcPlanar2Functor<1, T, T, T, mpp::Copy<T, T>, RoundingMode::None>;

    const mpp::Copy<T, T> op;

    const copySrc functor(aSrcChannel1.PointerRoi(), aSrcChannel1.Pitch(), aSrcChannel2.PointerRoi(),
                          aSrcChannel2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

/// <summary>
/// Copy planar image pixels to packed pixel image.
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel3, ImageView<T> &aDst)
    requires(ThreeChannel<T>)
{
    checkSameSize(aSrcChannel1.ROI(), aDst.ROI());
    checkSameSize(aSrcChannel2.ROI(), aDst.ROI());
    checkSameSize(aSrcChannel3.ROI(), aDst.ROI());

    using copySrc = SrcPlanar3Functor<1, T, T, T, mpp::Copy<T, T>, RoundingMode::None>;

    const mpp::Copy<T, T> op;

    const copySrc functor(aSrcChannel1.PointerRoi(), aSrcChannel1.Pitch(), aSrcChannel2.PointerRoi(),
                          aSrcChannel2.Pitch(), aSrcChannel3.PointerRoi(), aSrcChannel3.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

/// <summary>
/// Copy planar image pixels to packed pixel image.
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::Copy(ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel1,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel2,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel3,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrcChannel4, ImageView<T> &aDst)
    requires(FourChannelNoAlpha<T>)
{
    checkSameSize(aSrcChannel1.ROI(), aDst.ROI());
    checkSameSize(aSrcChannel2.ROI(), aDst.ROI());
    checkSameSize(aSrcChannel3.ROI(), aDst.ROI());
    checkSameSize(aSrcChannel4.ROI(), aDst.ROI());

    using copySrc = SrcPlanar4Functor<1, T, T, T, mpp::Copy<T, T>, RoundingMode::None>;

    const mpp::Copy<T, T> op;

    const copySrc functor(aSrcChannel1.PointerRoi(), aSrcChannel1.Pitch(), aSrcChannel2.PointerRoi(),
                          aSrcChannel2.Pitch(), aSrcChannel3.PointerRoi(), aSrcChannel3.Pitch(),
                          aSrcChannel4.PointerRoi(), aSrcChannel4.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}
#pragma endregion

#pragma region Dup
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Dup(ImageView<TTo> &aDst) const
    requires(vector_size_v<T> == 1) &&
            (vector_size_v<TTo> > 1) && std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>
{
    checkSameSize(ROI(), aDst.ROI());

    using dupSrc = SrcFunctor<1, T, T, TTo, mpp::Dup<T, TTo>, RoundingMode::None>;
    const mpp::Dup<T, TTo> op;
    const dupSrc functor(PointerRoi(), Pitch(), op);
    forEachPixel(aDst, functor);

    return aDst;
}
#pragma endregion

#pragma region Scale
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, RoundingMode aRoundingMode) const
    requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<T> &&
            RealOrComplexIntVector<TTo> && ScaleImplemented<T, TTo>
{
    return Scale(aDst, numeric_limits<scalefactor_t<T>>::lowest(), numeric_limits<scalefactor_t<T>>::max(),
                 numeric_limits<scalefactor_t<TTo>>::lowest(), numeric_limits<scalefactor_t<TTo>>::max(),
                 aRoundingMode);
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax) const
    requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<T> &&
            RealOrComplexFloatingVector<TTo> && ScaleImplemented<T, TTo>
{
    return Scale(aDst, numeric_limits<scalefactor_t<T>>::lowest(), numeric_limits<scalefactor_t<T>>::max(), aDstMin,
                 aDstMax);
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax,
                                    RoundingMode aRoundingMode) const
    requires(!std::same_as<T, TTo>) && (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<T> &&
            RealOrComplexIntVector<TTo> && ScaleImplemented<T, TTo>
{
    return Scale(aDst, numeric_limits<scalefactor_t<T>>::lowest(), numeric_limits<scalefactor_t<T>>::max(), aDstMin,
                 aDstMax, aRoundingMode);
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                                    RoundingMode aRoundingMode) const
    requires(!std::same_as<T, TTo>) &&
            (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<TTo> && ScaleImplemented<T, TTo>
{
    return Scale(aDst, aSrcMin, aSrcMax, numeric_limits<scalefactor_t<TTo>>::lowest(),
                 numeric_limits<scalefactor_t<TTo>>::max(), aRoundingMode);
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                                    scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax) const
    requires(!std::same_as<T, TTo>) &&
            (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexFloatingVector<TTo> && ScaleImplemented<T, TTo>
{
    checkSameSize(ROI(), aDst.ROI());

    static_assert(!use_int_division_for_scale_v<T, TTo>,
                  "No integer divison should be used for floating point destination type.");

    using ComputeT         = compute_type_scale_for_t<T, TTo>;
    using scaleType        = scalefactor_t<ComputeT>;
    const scaleType srcMin = static_cast<scaleType>(aSrcMin);
    const scaleType srcMax = static_cast<scaleType>(aSrcMax);
    const scaleType dstMin = static_cast<scaleType>(aDstMin);
    const scaleType dstMax = static_cast<scaleType>(aDstMax);
    const scaleType factor = (dstMax - dstMin) / (srcMax - srcMin);

    using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::NearestTiesToEven>;

    const scale functor(PointerRoi(), Pitch(), factor, srcMin, dstMin);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                                    scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax,
                                    RoundingMode aRoundingMode) const
    requires(!std::same_as<T, TTo>) &&
            (vector_size_v<T> == vector_size_v<TTo>) && RealOrComplexIntVector<TTo> && ScaleImplemented<T, TTo>
{
    checkSameSize(ROI(), aDst.ROI());

    // When converting sbyte with ComputeT == long64, ClangTidy thinks we mess things up, but the cast is correct
    using ComputeT         = compute_type_scale_for_t<T, TTo>;
    using scaleType        = scalefactor_t<ComputeT>;
    const scaleType srcMin = static_cast<scaleType>(aSrcMin); // NOLINT(bugprone-signed-char-misuse,cert-str34-c)
    const scaleType srcMax = static_cast<scaleType>(aSrcMax); // NOLINT(bugprone-signed-char-misuse,cert-str34-c)
    const scaleType dstMin = static_cast<scaleType>(aDstMin); // NOLINT(bugprone-signed-char-misuse,cert-str34-c)
    const scaleType dstMax = static_cast<scaleType>(aDstMax); // NOLINT(bugprone-signed-char-misuse,cert-str34-c)

    if constexpr (use_int_division_for_scale_v<T, TTo>)
    {
        const scaleType srcRange = srcMax - srcMin;
        const scaleType dstRange = dstMax - dstMin;

        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::NearestTiesToEven>;
                const scale functor(PointerRoi(), Pitch(), srcMin, dstMin, srcRange, dstRange);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::NearestTiesAwayFromZero>;
                const scale functor(PointerRoi(), Pitch(), srcMin, dstMin, srcRange, dstRange);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::TowardZero>;
                const scale functor(PointerRoi(), Pitch(), srcMin, dstMin, srcRange, dstRange);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::TowardNegativeInfinity>;
                const scale functor(PointerRoi(), Pitch(), srcMin, dstMin, srcRange, dstRange);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::TowardPositiveInfinity>;
                const scale functor(PointerRoi(), Pitch(), srcMin, dstMin, srcRange, dstRange);
                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        const scaleType factor = (dstMax - dstMin) / (srcMax - srcMin);

        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::NearestTiesToEven>;
                const scale functor(PointerRoi(), Pitch(), factor, srcMin, dstMin);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::NearestTiesAwayFromZero>;
                const scale functor(PointerRoi(), Pitch(), factor, srcMin, dstMin);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::TowardZero>;
                const scale functor(PointerRoi(), Pitch(), factor, srcMin, dstMin);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::TowardNegativeInfinity>;
                const scale functor(PointerRoi(), Pitch(), factor, srcMin, dstMin);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::TowardPositiveInfinity>;
                const scale functor(PointerRoi(), Pitch(), factor, srcMin, dstMin);
                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }

    return aDst;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
#pragma endregion

#pragma region Set
template <PixelType T> ImageView<T> &ImageView<T>::Set(const T &aConst)
{
    using setC = ConstantFunctor<1, T>;
    const setC functor(aConst);
    forEachPixel(*this, functor);

    return *this;
}

template <PixelType T> ImageView<T> &ImageView<T>::SetMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
{
    using setC = ConstantFunctor<1, T>;
    const setC functor(aConst);
    forEachPixel(aMask, *this, functor);

    return *this;
}

template <PixelType T> ImageView<T> &ImageView<T>::Set(remove_vector_t<T> aConst, Channel aChannel)
{
    for (auto &pixelIterator : *this)
    {
        pixelIterator.Value()[aChannel] = aConst;
    }

    return *this;
}
#pragma endregion

#pragma region Swap Channel
/// <summary>
/// Swap channels
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::SwapChannel(ImageView<T> &aDst) const
    requires(vector_size_v<T> == 2)
{
    checkSameSize(ROI(), aDst.ROI());

    using swapChannelSrc = SrcFunctor<1, T, T, T, mpp::SwapChannel<T>, RoundingMode::None>;

    const mpp::SwapChannel<T> op;

    const swapChannelSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}
/// <summary>
/// Swap channels (inplace)
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::SwapChannel()
    requires(vector_size_v<T> == 2)
{
    using swapChannelInplace = InplaceFunctor<1, T, T, mpp::SwapChannel<T>, RoundingMode::None>;

    const mpp::SwapChannel<T> op;

    const swapChannelInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
/// <summary>
/// Swap channels
/// </summary>
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::SwapChannel(ImageView<TTo> &aDst,
                                          const ChannelList<vector_active_size_v<TTo>> &aDstChannels) const
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

    using swapChannelSrc = SrcFunctor<1, T, T, TTo, mpp::SwapChannel<T, TTo>, RoundingMode::None>;

    const mpp::SwapChannel<T, TTo> op(aDstChannels);

    const swapChannelSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}
/// <summary>
/// Swap channels (inplace)
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::SwapChannel(const ChannelList<vector_active_size_v<T>> &aDstChannels)
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

    using swapChannelInplace = InplaceFunctor<1, T, T, mpp::SwapChannel<T, T>, RoundingMode::None>;

    const mpp::SwapChannel<T, T> op(aDstChannels);

    const swapChannelInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::SwapChannel(ImageView<TTo> &aDst,
                                          const ChannelList<vector_active_size_v<TTo>> &aDstChannels,
                                          remove_vector_t<T> aValue) const
    requires(vector_size_v<T> == 3) &&          //
            (vector_active_size_v<TTo> == 4) && //
            (!has_alpha_channel_v<TTo>) &&      //
            (!has_alpha_channel_v<T>) &&        //
            std::same_as<remove_vector_t<T>, remove_vector_t<TTo>>
{
    checkSameSize(ROI(), aDst.ROI());

    using swapChannelSrcDstAsSrc = SrcDstAsSrcFunctor<1, T, TTo, mpp::SwapChannel<T, TTo>>;

    const mpp::SwapChannel<T, TTo> op(aDstChannels, aValue);

    const swapChannelSrcDstAsSrc functor(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}
#pragma endregion

#pragma region Transpose
/// <summary>
/// Transpose image.
/// </summary>
template <PixelType T>
ImageView<T> &ImageView<T>::Transpose(ImageView<T> &aDst) const
    requires NoAlpha<T>
{
    if (SizeRoi() != Size2D(aDst.SizeRoi().y, aDst.SizeRoi().x))
    {
        throw ROIEXCEPTION(
            "Width of destination image ROI must be the same as the height of the source image ROI, and height "
            "of the destination image ROI must be the same as the width of the source image ROI. Source ROI size: "
            << SizeRoi() << " provided destination image ROI size: " << aDst.SizeRoi());
    }

    for (auto &pixelIterator : aDst)
    {
        const int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        const int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        // will be optimized away as unused in case of no alpha channel:
        pixel_basetype_t<T> alphaChannel; // NOLINT

        T &pixelOut = pixelIterator.Value();

        // load the alpha channel, if needed:
        if constexpr (has_alpha_channel_v<T>)
        {
            alphaChannel = pixelOut.w;
        }

        T res = (*this)(pixelY, pixelX); // NOLINT(readability-suspicious-call-argument) --> transpose...

        // restore alpha channel value:
        if constexpr (has_alpha_channel_v<T>)
        {
            res.w = alphaChannel;
        }

        pixelOut = res;
    }

    return aDst;
}
#pragma endregion

#pragma region FillRandom
template <PixelType T> ImageView<T> &ImageView<T>::FillRandom()
{
    using randomInplace = InplaceFunctor<1, T, T, mpp::FillRandom<T>, RoundingMode::None>;

    const mpp::FillRandom<T> op;
    const randomInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T> ImageView<T> &ImageView<T>::FillRandom(uint aSeed)
{
    using randomInplace = InplaceFunctor<1, T, T, mpp::FillRandom<T>, RoundingMode::None>;

    const mpp::FillRandom<T> op(aSeed);
    const randomInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T> ImageView<T> &ImageView<T>::FillRandomNormal(uint aSeed, double aMean, double aStd)
{
    using randomInplace = InplaceFunctor<1, T, T, mpp::FillRandomNormal<T>, RoundingMode::None>;

    const mpp::FillRandomNormal<T> op(aSeed, aMean, aStd);
    const randomInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion

} // namespace mpp::image::cpuSimple