#pragma once
#include <backends/simple_cpu/image/forEachPixel.h>
#include <backends/simple_cpu/image/forEachPixelMasked.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/arithmetic/binary_operators.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/border.h>
#include <common/image/functors/constantFunctor.h>
#include <common/image/functors/convertFunctor.h>
#include <common/image/functors/convertScaleFunctor.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/functors/inplaceConstantFunctor.h>
#include <common/image/functors/inplaceConstantScaleFunctor.h>
#include <common/image/functors/inplaceDevConstantFunctor.h>
#include <common/image/functors/inplaceDevConstantScaleFunctor.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/inplaceSrcScaleFunctor.h>
#include <common/image/functors/scaleConversionFunctor.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcConstantScaleFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
#include <common/image/functors/srcDevConstantScaleFunctor.h>
#include <common/image/functors/srcScaleFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/functors/srcSrcScaleFunctor.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace opp::image::cpuSimple
{

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Convert(ImageView<TTo> &aDst)
    requires(!std::same_as<T, TTo>) &&
            (RealOrComplexIntVector<T> || (std::same_as<complex_basetype_t<remove_vector_t<T>>, float> &&
                                           (std::same_as<complex_basetype_t<remove_vector_t<TTo>>, BFloat16> ||
                                            std::same_as<complex_basetype_t<remove_vector_t<TTo>>, HalfFp16>)))
{
    checkSameSize(ROI(), aDst.ROI());

    using convert = ConvertFunctor<1, T, TTo, RoundingMode::NearestTiesToEven>;

    const convert functor(PointerRoi(), Pitch());
    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode)
    requires(!std::same_as<T, TTo>) && RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using convert = ConvertFunctor<1, T, TTo, RoundingMode::NearestTiesToEven>;
            const convert functor(PointerRoi(), Pitch());
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using convert = ConvertFunctor<1, T, TTo, RoundingMode::NearestTiesAwayFromZero>;
            const convert functor(PointerRoi(), Pitch());
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using convert = ConvertFunctor<1, T, TTo, RoundingMode::TowardZero>;
            const convert functor(PointerRoi(), Pitch());
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using convert = ConvertFunctor<1, T, TTo, RoundingMode::TowardNegativeInfinity>;
            const convert functor(PointerRoi(), Pitch());
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using convert = ConvertFunctor<1, T, TTo, RoundingMode::TowardPositiveInfinity>;
            const convert functor(PointerRoi(), Pitch());
            forEachPixel(aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Convert(ImageView<TTo> &aDst, RoundingMode aRoundingMode, int aScaleFactor)
    requires(!std::same_as<T, TTo>) && (!std::same_as<TTo, float>) && (!std::same_as<TTo, double>) &&
            (!std::same_as<TTo, Complex<float>>) && (!std::same_as<TTo, Complex<double>>)
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using convert = ConvertScaleFunctor<1, T, TTo, RoundingMode::NearestTiesToEven>;
            const convert functor(PointerRoi(), Pitch(), scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using convert = ConvertScaleFunctor<1, T, TTo, RoundingMode::NearestTiesAwayFromZero>;
            const convert functor(PointerRoi(), Pitch(), scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using convert = ConvertScaleFunctor<1, T, TTo, RoundingMode::TowardZero>;
            const convert functor(PointerRoi(), Pitch(), scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using convert = ConvertScaleFunctor<1, T, TTo, RoundingMode::TowardNegativeInfinity>;
            const convert functor(PointerRoi(), Pitch(), scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using convert = ConvertScaleFunctor<1, T, TTo, RoundingMode::TowardPositiveInfinity>;
            const convert functor(PointerRoi(), Pitch(), scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }

    return aDst;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst)
    requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<T> && RealOrComplexIntVector<TTo>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT             = default_compute_type_for_t<T>;
    using scaleType            = scalefactor_t<default_compute_type_for_t<T>>;
    constexpr scaleType srcMin = static_cast<scaleType>(numeric_limits<T>::lowest());
    constexpr scaleType srcMax = static_cast<scaleType>(numeric_limits<T>::max());
    constexpr scaleType dstMin = static_cast<scaleType>(numeric_limits<TTo>::lowest());
    constexpr scaleType dstMax = static_cast<scaleType>(numeric_limits<TTo>::max());
    constexpr scaleType factor = (dstMax - dstMin) / (srcMax - srcMin);

    using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::NearestTiesAwayFromZero>;

    const scale functor(PointerRoi(), Pitch(), factor, srcMin, dstMin);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax)
    requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT             = default_compute_type_for_t<T>;
    using scaleType            = scalefactor_t<default_compute_type_for_t<T>>;
    constexpr scaleType srcMin = static_cast<scaleType>(numeric_limits<T>::lowest());
    constexpr scaleType srcMax = static_cast<scaleType>(numeric_limits<T>::max());
    constexpr scaleType dstMin = static_cast<scaleType>(aDstMin);
    constexpr scaleType dstMax = static_cast<scaleType>(aDstMax);
    constexpr scaleType factor = (dstMax - dstMin) / (srcMax - srcMin);

    using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::NearestTiesAwayFromZero>;

    const scale functor(PointerRoi(), Pitch(), factor, srcMin, dstMin);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax)
    requires(!std::same_as<T, TTo>) && RealOrComplexIntVector<TTo>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT             = default_compute_type_for_t<T>;
    using scaleType            = scalefactor_t<default_compute_type_for_t<T>>;
    constexpr scaleType srcMin = static_cast<scaleType>(aSrcMin);
    constexpr scaleType srcMax = static_cast<scaleType>(aSrcMax);
    constexpr scaleType dstMin = static_cast<scaleType>(numeric_limits<TTo>::lowest());
    constexpr scaleType dstMax = static_cast<scaleType>(numeric_limits<TTo>::max());
    constexpr scaleType factor = (dstMax - dstMin) / (srcMax - srcMin);

    using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::NearestTiesAwayFromZero>;

    const scale functor(PointerRoi(), Pitch(), factor, srcMin, dstMin);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
template <PixelType TTo>
ImageView<TTo> &ImageView<T>::Scale(ImageView<TTo> &aDst, scalefactor_t<T> aSrcMin, scalefactor_t<T> aSrcMax,
                                    scalefactor_t<TTo> aDstMin, scalefactor_t<TTo> aDstMax)
    requires(!std::same_as<T, TTo>)
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT             = default_compute_type_for_t<T>;
    using scaleType            = scalefactor_t<default_compute_type_for_t<T>>;
    constexpr scaleType srcMin = static_cast<scaleType>(aSrcMin);
    constexpr scaleType srcMax = static_cast<scaleType>(aSrcMax);
    constexpr scaleType dstMin = static_cast<scaleType>(aDstMin);
    constexpr scaleType dstMax = static_cast<scaleType>(aDstMax);
    constexpr scaleType factor = (dstMax - dstMin) / (srcMax - srcMin);

    using scale = ScaleConversionFunctor<1, T, ComputeT, TTo, RoundingMode::NearestTiesAwayFromZero>;

    const scale functor(PointerRoi(), Pitch(), factor, srcMin, dstMin);

    forEachPixel(aDst, functor);

    return aDst;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

template <PixelType T> ImageView<T> &ImageView<T>::Set(const T &aConst)
{
    using setC = ConstantFunctor<1, T>;
    const setC functor(aConst);
    forEachPixel(*this, functor);

    return *this;
}

template <PixelType T> ImageView<T> &ImageView<T>::Set(const T &aConst, const ImageView<Pixel8uC1> &aMask)
{
    using setC = ConstantFunctor<1, T>;
    const setC functor(aConst);
    forEachPixel(aMask, *this, functor);

    return *this;
}

} // namespace opp::image::cpuSimple