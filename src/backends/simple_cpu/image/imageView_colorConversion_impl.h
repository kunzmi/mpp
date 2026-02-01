#pragma once
#include "forEachPixel422.h"
#include <backends/simple_cpu/image/forEachPixel_impl.h>
#include <backends/simple_cpu/image/forEachPixel411_impl.h>
#include <backends/simple_cpu/image/forEachPixel420_impl.h>
#include <backends/simple_cpu/image/forEachPixel422_impl.h>
#include <backends/simple_cpu/image/forEachPixelBlock_impl.h>
#include <backends/simple_cpu/image/forEachPixelPlanar_impl.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/bfloat16.h>
#include <common/colorConversion/color_operators.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/filtering/postOperators.h>
#include <common/half_fp16.h>
#include <common/image/border.h>
#include <common/image/channel.h>
#include <common/image/channelList.h>
#include <common/image/filterArea.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/cfaToRgbFunctor.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/rgbToCfaFunctor.h>
#include <common/image/functors/src411Functor.h>
#include <common/image/functors/src420Functor.h>
#include <common/image/functors/src422C2Functor.h>
#include <common/image/functors/src422Functor.h>
#include <common/image/functors/srcFunctor.h>
#include <common/image/functors/srcPlanar2Functor.h>
#include <common/image/functors/srcPlanar3Functor.h>
#include <common/image/functors/srcPlanar4Functor.h>
#include <common/image/gotoPtr.h>
#include <common/image/matrix.h>
#include <common/image/matrix3x4.h>
#include <common/image/matrix4x4.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/morphology/operators.h>
#include <common/morphology/postOperators.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/safeCast.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vector1.h>
#include <common/vector2.h>
#include <common/vector3.h>
#include <common/vector4A.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace mpp::image::cpuSimple
{

// NOLINTBEGIN(readability-suspicious-call-argument)

template <typename T> constexpr RoundingMode GetRoundingModeColorConv()
{
    return RoundingMode::None;
}
template <RealIntVector T> constexpr RoundingMode GetRoundingModeColorConv()
{
    if constexpr (RealSignedVector<T>)
    {
        return RoundingMode::NearestTiesAwayFromZero;
    }
    return RoundingMode::NearestTiesAwayFromZeroPositive;
}

#pragma region ColorConversion

#pragma region HLS
#pragma region RGBtoHLS
template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHLS(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHLS<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHLS<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::RGBtoHLS(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHLS<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHLS<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::RGBtoHLS(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHLS<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHLS<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHLS<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::RGBtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHLS<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHLS<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHLS<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHLS<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHLS<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHLS<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHLS<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHLS(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::RGBtoHLS<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHLS<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}
#pragma endregion
#pragma region BGRtoHLS
template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHLS(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHLS<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHLS<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::BGRtoHLS(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHLS<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHLS<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::BGRtoHLS(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHLS<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHLS<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHLS<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::BGRtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHLS<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHLS<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHLS<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHLS<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHLS<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHLS<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHLS<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHLS(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::BGRtoHLS<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHLS<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion

#pragma region HLStoRGB
template <PixelType T>
ImageView<T> &ImageView<T>::HLStoRGB(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoRGB<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::HLStoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoRGB<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::HLStoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::HLStoRGB<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoRGB<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HLStoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::HLStoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoRGB<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::HLStoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoRGB<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HLStoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::HLStoRGB<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoRGB<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HLStoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HLStoRGB(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::HLStoRGB<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}
#pragma endregion
#pragma region HLStoBGR
template <PixelType T>
ImageView<T> &ImageView<T>::HLStoBGR(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoBGR<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::HLStoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoBGR<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::HLStoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::HLStoBGR<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoBGR<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HLStoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::HLStoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoBGR<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::HLStoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoBGR<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HLStoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::HLStoBGR<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoBGR<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HLStoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HLStoBGR(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hlsInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::HLStoBGR<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HLStoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const hlsInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion
#pragma endregion

#pragma region HSV
#pragma region RGBtoHSV
template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHSV(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHSV<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHSV<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::RGBtoHSV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHSV<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHSV<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::RGBtoHSV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHSV<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHSV<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHSV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::RGBtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHSV<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHSV<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHSV<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHSV<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHSV<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHSV<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHSV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHSV(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::RGBtoHSV<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoHSV<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}
#pragma endregion
#pragma region BGRtoHSV
template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHSV(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHSV<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHSV<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::BGRtoHSV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHSV<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHSV<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::BGRtoHSV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHSV<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHSV<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHSV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::BGRtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHSV<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHSV<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHSV<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHSV<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHSV<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHSV<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHSV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHSV(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::BGRtoHSV<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoHSV<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion

#pragma region HSVtoRGB
template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoRGB(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoRGB<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::HSVtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoRGB<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::HSVtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoRGB<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoRGB<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::HSVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoRGB<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoRGB<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoRGB<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoRGB<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoRGB(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::HSVtoRGB<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}
#pragma endregion
#pragma region HSVtoBGR
template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoBGR(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoBGR<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::HSVtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoBGR<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::HSVtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoBGR<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoBGR<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::HSVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoBGR<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoBGR<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoBGR<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoBGR<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoBGR(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using hsvInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::HSVtoBGR<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::HSVtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const hsvInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion
#pragma endregion

#pragma region Lab
#pragma region RGBtoLab
template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLab(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLab<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLab<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::RGBtoLab(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLab<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLab<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::RGBtoLab(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLab<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLab<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLab<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::RGBtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLab<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLab<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLab<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLab<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLab<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLab<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLab<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLab(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::RGBtoLab<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLab<ComputeT, doNormalize> op(aNormalizationFactor);

    const labInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}
#pragma endregion
#pragma region BGRtoLab
template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLab(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLab<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLab<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::BGRtoLab(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLab<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLab<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::BGRtoLab(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLab<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLab<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLab<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::BGRtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLab<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLab<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLab<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLab<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLab<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLab<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLab<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLab(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::BGRtoLab<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLab<ComputeT, doNormalize> op(aNormalizationFactor);

    const labInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion

#pragma region LabtoRGB
template <PixelType T>
ImageView<T> &ImageView<T>::LabtoRGB(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoRGB<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::LabtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoRGB<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::LabtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::LabtoRGB<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoRGB<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LabtoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::LabtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoRGB<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::LabtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoRGB<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LabtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::LabtoRGB<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoRGB<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LabtoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LabtoRGB(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LabtoRGB<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const labInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}
#pragma endregion
#pragma region LabtoBGR
template <PixelType T>
ImageView<T> &ImageView<T>::LabtoBGR(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoBGR<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::LabtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoBGR<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::LabtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::LabtoBGR<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoBGR<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LabtoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::LabtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoBGR<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::LabtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoBGR<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const labSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LabtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::LabtoBGR<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoBGR<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LabtoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LabtoBGR(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using labInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LabtoBGR<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LabtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const labInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion
#pragma endregion

#pragma region LUV
#pragma region RGBtoLUV
template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLUV(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLUV<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLUV<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::RGBtoLUV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLUV<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLUV<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::RGBtoLUV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLUV<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLUV<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLUV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::RGBtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLUV<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLUV<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLUV<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLUV<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLUV<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLUV<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLUV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLUV(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::RGBtoLUV<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::RGBtoLUV<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}
#pragma endregion
#pragma region BGRtoLUV
template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLUV(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLUV<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLUV<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::BGRtoLUV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLUV<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLUV<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::BGRtoLUV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLUV<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLUV<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLUV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::BGRtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLUV<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLUV<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLUV<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLUV<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLUV<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLUV<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLUV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLUV(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::BGRtoLUV<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::BGRtoLUV<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion

#pragma region LUVtoRGB
template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoRGB(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoRGB<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::LUVtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoRGB<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::LUVtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoRGB<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoRGB<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::LUVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoRGB<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoRGB<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoRGB<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoRGB<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoRGB(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LUVtoRGB<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoRGB<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}
#pragma endregion
#pragma region LUVtoBGR
template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoBGR(ImageView<T> &aDst, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoBGR<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::LUVtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoBGR<ComputeT, doNormalize>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::LUVtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoBGR<Pixel32fC4A, doNormalize>>,
                              GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoBGR<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::LUVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoBGR<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoBGR<ComputeT, doNormalize>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoBGR<Pixel32fC4A, doNormalize>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoBGR<Pixel32fC4A, doNormalize> op(aNormalizationFactor);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoBGR(float aNormalizationFactor)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;
    constexpr bool doNormalize = RealIntVector<T>;

    using luvInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LUVtoBGR<ComputeT, doNormalize>,
                                      GetRoundingModeColorConv<DstT>()>;

    const mpp::image::LUVtoBGR<ComputeT, doNormalize> op(aNormalizationFactor);

    const luvInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion
#pragma endregion

#pragma region ColorTwist3x3
template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(ImageView<T> &aDst, const Matrix<float> &aTwist) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Matrix<float> &aTwist) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::ColorTwist3x3<Pixel32fC4A>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<Pixel32fC4A> op(aTwist);
    const mpp::image::SetAlpha<ComputeT, mpp::image::ColorTwist3x3<Pixel32fC4A>> opAlpha(op);

    const colorTwistSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                       const Matrix<float> &aTwist)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                       remove_vector_t<T> aAlpha, const Matrix<float> &aTwist)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = Vector4A<remove_vector_t<T>>;
    using DstT     = T;
    using ComputeT = Pixel32fC4A;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT,
                                            mpp::image::SetAlphaConst<ComputeT, mpp::image::ColorTwist3x3<Pixel32fC4A>>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<Pixel32fC4A> op(aTwist);
    const mpp::image::SetAlphaConst<ComputeT, mpp::image::ColorTwist3x3<Pixel32fC4A>> opAlpha(
        static_cast<float>(aAlpha), op);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                       const Matrix<float> &aTwist)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                            mpp::image::SetAlpha<ComputeT, mpp::image::ColorTwist3x3<Pixel32fC4A>>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<Pixel32fC4A> op(aTwist);
    const mpp::image::SetAlpha<ComputeT, mpp::image::ColorTwist3x3<Pixel32fC4A>> opAlpha(op);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const Matrix<float> &aTwist)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                             GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

template <PixelType T>
ImageView<Vector2<remove_vector_t<T>>> &ImageView<T>::ColorTwistTo422(
    ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma, const Matrix<float> &aTwist,
    ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLumaChroma.ROI());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel422<T>(aDstLumaChroma, aChromaSubsamplePos,
                       aSwapLumaChroma ? Dst422C2Layout::CbYCr : Dst422C2Layout::YCbCr, functor);

    return aDstLumaChroma;
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi()), aDstChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel422<T>(aDstLuma, aDstChroma, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi()), aDstChroma2.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel422<T>(aDstLuma, aDstChroma1, aDstChroma2, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi() / 2), aDstChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel420<T>(aDstLuma, aDstChroma, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi() / 2), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi() / 2), aDstChroma2.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel420<T>(aDstLuma, aDstChroma1, aDstChroma2, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 4, HeightRoi()), aDstChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel411<T>(aDstLuma, aDstChroma, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 4, HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(WidthRoi() / 4, HeightRoi()), aDstChroma2.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel411<T>(aDstLuma, aDstChroma1, aDstChroma2, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLumaChroma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLumaChroma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLumaChroma.ROI());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel422<T>(aDstLumaChroma, aChromaSubsamplePos,
                       aSwapLumaChroma ? Dst422C2Layout::CbYCr : Dst422C2Layout::YCbCr, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi()), aDstChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel422<T>(aDstLuma, aDstChroma, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi()), aDstChroma2.SizeRoi());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel422<T>(aDstLuma, aDstChroma1, aDstChroma2, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi() / 2), aDstChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel420<T>(aDstLuma, aDstChroma, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi() / 2), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi() / 2), aDstChroma2.SizeRoi());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel420<T>(aDstLuma, aDstChroma1, aDstChroma2, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 4, aSrc0.HeightRoi()), aDstChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel411<T>(aDstLuma, aDstChroma, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 4, aSrc0.HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 4, aSrc0.HeightRoi()), aDstChroma2.SizeRoi());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel411<T>(aDstLuma, aDstChroma1, aDstChroma2, aChromaSubsamplePos, functor);
}

template <PixelType T>
ImageView<Vector3<remove_vector_t<T>>> &ImageView<T>::ColorTwistFrom422(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                                        const Matrix<float> &aTwist,
                                                                        bool aSwapLumaChroma) const
    requires std::same_as<Pixel8uC2, T> || std::same_as<Pixel16uC2, T> || std::same_as<Pixel16sC2, T> ||
             std::same_as<Pixel16fC2, T> || std::same_as<Pixel32fC2, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = Vector3<remove_vector_t<T>>;
    using DstT     = Vector3<remove_vector_t<T>>;
    using ComputeT = Pixel32fC3;

    constexpr size_t TupelSize = 1;

    if (aSwapLumaChroma)
    {
        using colorTwistSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                              Src422C2Layout::CbYCr, GetRoundingModeColorConv<DstT>()>;

        const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

        const colorTwistSrc functor(PointerRoi(), Pitch(), op);

        forEachPixel(aDst, functor);
    }
    else
    {
        using colorTwistSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                              Src422C2Layout::YCbCr, GetRoundingModeColorConv<DstT>()>;

        const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

        const colorTwistSrc functor(PointerRoi(), Pitch(), op);

        forEachPixel(aDst, functor);
    }

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     bool aSwapLumaChroma) const
    requires std::same_as<Pixel8uC2, T> || std::same_as<Pixel16uC2, T> || std::same_as<Pixel16sC2, T> ||
             std::same_as<Pixel16fC2, T> || std::same_as<Pixel32fC2, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = Vector3<remove_vector_t<T>>;
    using DstT     = Vector3<remove_vector_t<T>>;
    using ComputeT = Pixel32fC3;

    constexpr size_t TupelSize = 1;

    if (aSwapLumaChroma)
    {
        using colorTwistSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                              Src422C2Layout::CbYCr, GetRoundingModeColorConv<DstT>()>;

        const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

        const colorTwistSrc functor(PointerRoi(), Pitch(), op);

        forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
    }
    else
    {
        using colorTwistSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                              Src422C2Layout::YCbCr, GetRoundingModeColorConv<DstT>()>;

        const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

        const colorTwistSrc functor(PointerRoi(), Pitch(), op);

        forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
    }
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                              const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aSrcLuma.WidthRoi() / 2, aSrcLuma.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                              const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst.WidthRoi() / 2, aDst.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst0.WidthRoi() / 2, aDst0.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst0.WidthRoi() / 2, aDst0.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                              const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2));

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    const Size2D sizeChroma(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2);

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else if (aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                              const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2));

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    const Size2D sizeChroma(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2);

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else if (aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2));

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    const Size2D sizeChroma(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2);

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else if (aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2));

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    const Size2D sizeChroma(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2);

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else if (aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                              const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst.WidthRoi() / 4, aDst.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst.WidthRoi() / 4, aDst.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                              const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst.WidthRoi() / 4, aDst.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst.WidthRoi() / 4, aDst.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst.WidthRoi() / 4, aDst.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst0.WidthRoi() / 4, aDst0.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst0.WidthRoi() / 4, aDst0.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst0.WidthRoi() / 4, aDst0.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst0.WidthRoi() / 4, aDst0.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst0.WidthRoi() / 4, aDst0.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x3<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x3<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}
#pragma endregion

#pragma region ColorTwist3x4
template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(ImageView<T> &aDst, const Matrix3x4<float> &aTwist) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Matrix3x4<float> &aTwist) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::ColorTwist3x4<Pixel32fC4A>>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<Pixel32fC4A> op(aTwist);
    const mpp::image::SetAlpha<ComputeT, mpp::image::ColorTwist3x4<Pixel32fC4A>> opAlpha(op);

    const colorTwistSrc functor(PointerRoi(), Pitch(), opAlpha);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                       const Matrix3x4<float> &aTwist)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                       remove_vector_t<T> aAlpha, const Matrix3x4<float> &aTwist)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    using SrcT     = Vector4A<remove_vector_t<T>>;
    using DstT     = T;
    using ComputeT = Pixel32fC4A;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT,
                                            mpp::image::SetAlphaConst<ComputeT, mpp::image::ColorTwist3x4<Pixel32fC4A>>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<Pixel32fC4A> op(aTwist);
    const mpp::image::SetAlphaConst<ComputeT, mpp::image::ColorTwist3x4<Pixel32fC4A>> opAlpha(
        static_cast<float>(aAlpha), op);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                       const Matrix3x4<float> &aTwist)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                            mpp::image::SetAlpha<ComputeT, mpp::image::ColorTwist3x4<Pixel32fC4A>>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<Pixel32fC4A> op(aTwist);
    const mpp::image::SetAlpha<ComputeT, mpp::image::ColorTwist3x4<Pixel32fC4A>> opAlpha(op);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), opAlpha);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const Matrix3x4<float> &aTwist)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                             GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

template <PixelType T>
ImageView<Vector2<remove_vector_t<T>>> &ImageView<T>::ColorTwistTo422(
    ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma, const Matrix3x4<float> &aTwist,
    ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLumaChroma.ROI());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel422<T>(aDstLumaChroma, aChromaSubsamplePos,
                       aSwapLumaChroma ? Dst422C2Layout::CbYCr : Dst422C2Layout::YCbCr, functor);

    return aDstLumaChroma;
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi()), aDstChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel422<T>(aDstLuma, aDstChroma, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi()), aDstChroma2.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel422<T>(aDstLuma, aDstChroma1, aDstChroma2, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi() / 2), aDstChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel420<T>(aDstLuma, aDstChroma, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi() / 2), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi() / 2), aDstChroma2.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel420<T>(aDstLuma, aDstChroma1, aDstChroma2, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 4, HeightRoi()), aDstChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel411<T>(aDstLuma, aDstChroma, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 4, HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(WidthRoi() / 4, HeightRoi()), aDstChroma2.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel411<T>(aDstLuma, aDstChroma1, aDstChroma2, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma,
                                   const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                   bool aSwapLumaChroma)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLumaChroma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLumaChroma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLumaChroma.ROI());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel422<T>(aDstLumaChroma, aChromaSubsamplePos,
                       aSwapLumaChroma ? Dst422C2Layout::CbYCr : Dst422C2Layout::YCbCr, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi()), aDstChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel422<T>(aDstLuma, aDstChroma, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi()), aDstChroma2.SizeRoi());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel422<T>(aDstLuma, aDstChroma1, aDstChroma2, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi() / 2), aDstChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel420<T>(aDstLuma, aDstChroma, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi() / 2), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi() / 2), aDstChroma2.SizeRoi());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel420<T>(aDstLuma, aDstChroma1, aDstChroma2, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 4, aSrc0.HeightRoi()), aDstChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel411<T>(aDstLuma, aDstChroma, aChromaSubsamplePos, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 4, aSrc0.HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 4, aSrc0.HeightRoi()), aDstChroma2.SizeRoi());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel411<T>(aDstLuma, aDstChroma1, aDstChroma2, aChromaSubsamplePos, functor);
}

template <PixelType T>
ImageView<Vector3<remove_vector_t<T>>> &ImageView<T>::ColorTwistFrom422(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                                        const Matrix3x4<float> &aTwist,
                                                                        bool aSwapLumaChroma) const
    requires std::same_as<Pixel8uC2, T> || std::same_as<Pixel16uC2, T> || std::same_as<Pixel16sC2, T> ||
             std::same_as<Pixel16fC2, T> || std::same_as<Pixel32fC2, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = Vector3<remove_vector_t<T>>;
    using DstT     = Vector3<remove_vector_t<T>>;
    using ComputeT = Pixel32fC3;

    constexpr size_t TupelSize = 1;

    if (aSwapLumaChroma)
    {
        using colorTwistSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                              Src422C2Layout::CbYCr, GetRoundingModeColorConv<DstT>()>;

        const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

        const colorTwistSrc functor(PointerRoi(), Pitch(), op);

        forEachPixel(aDst, functor);
    }
    else
    {
        using colorTwistSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                              Src422C2Layout::YCbCr, GetRoundingModeColorConv<DstT>()>;

        const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

        const colorTwistSrc functor(PointerRoi(), Pitch(), op);

        forEachPixel(aDst, functor);
    }

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     bool aSwapLumaChroma) const
    requires std::same_as<Pixel8uC2, T> || std::same_as<Pixel16uC2, T> || std::same_as<Pixel16sC2, T> ||
             std::same_as<Pixel16fC2, T> || std::same_as<Pixel32fC2, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    using SrcT     = Vector3<remove_vector_t<T>>;
    using DstT     = Vector3<remove_vector_t<T>>;
    using ComputeT = Pixel32fC3;

    constexpr size_t TupelSize = 1;

    if (aSwapLumaChroma)
    {
        using colorTwistSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                              Src422C2Layout::CbYCr, GetRoundingModeColorConv<DstT>()>;

        const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

        const colorTwistSrc functor(PointerRoi(), Pitch(), op);

        forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
    }
    else
    {
        using colorTwistSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                              Src422C2Layout::YCbCr, GetRoundingModeColorConv<DstT>()>;

        const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

        const colorTwistSrc functor(PointerRoi(), Pitch(), op);

        forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
    }
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                              const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst.WidthRoi() / 2, aDst.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                              const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst.WidthRoi() / 2, aDst.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst0.WidthRoi() / 2, aDst0.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst0.WidthRoi() / 2, aDst0.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                              const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2));

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    const Size2D sizeChroma(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2);

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else if (aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                              const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2));

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    const Size2D sizeChroma(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2);

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else if (aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2));

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    const Size2D sizeChroma(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2);

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else if (aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2));

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    const Size2D sizeChroma(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2);

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else if (aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                              const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst.WidthRoi() / 4, aDst.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst.WidthRoi() / 4, aDst.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                              const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst.WidthRoi() / 4, aDst.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst.WidthRoi() / 4, aDst.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst.WidthRoi() / 4, aDst.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst0.WidthRoi() / 4, aDst0.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst0.WidthRoi() / 4, aDst0.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                            aSrcChroma.Pitch(), sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst0.WidthRoi() / 4, aDst0.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst0.WidthRoi() / 4, aDst0.HeightRoi()));
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;
    const Size2D sizeChroma(aDst0.WidthRoi() / 4, aDst0.HeightRoi());

    constexpr size_t TupelSize = 1;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::Linear, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingModeColorConv<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                            aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                            sizeChroma, op);

                forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}
#pragma endregion

#pragma region ColorTwist4x4
template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(ImageView<T> &aDst, const Matrix4x4<float> &aTwist) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist4x4<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist4x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Matrix4x4<float> &aTwist) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist4x4<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist4x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Matrix4x4<float> &aTwist)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());
    checkSameSize(aSrc0.ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist4x4<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist4x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                       const Matrix4x4<float> &aTwist)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist4x4<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist4x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const Matrix4x4<float> &aTwist)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::ColorTwist4x4<ComputeT>,
                                             GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist4x4<ComputeT> op(aTwist);

    const colorTwistInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion

#pragma region ColorTwist4x4C
template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(ImageView<T> &aDst, const Matrix4x4<float> &aTwist,
                                       const Pixel32fC4 &aConstant) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist4x4C<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist4x4C<ComputeT> op(aTwist, aConstant);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Matrix4x4<float> &aTwist,
                              const Pixel32fC4 &aConstant) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist4x4C<ComputeT>,
                                     GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist4x4C<ComputeT> op(aTwist, aConstant);

    const colorTwistSrc functor(PointerRoi(), Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
void ImageView<T>::ColorTwist(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
    ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const Matrix4x4<float> &aTwist, const Pixel32fC4 &aConstant)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());
    checkSameSize(aSrc0.ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist4x4C<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist4x4C<ComputeT> op(aTwist, aConstant);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                       const Matrix4x4<float> &aTwist, const Pixel32fC4 &aConstant)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist4x4C<ComputeT>,
                                            GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist4x4C<ComputeT> op(aTwist, aConstant);

    const colorTwistSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const Matrix4x4<float> &aTwist, const Pixel32fC4 &aConstant)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using colorTwistInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::ColorTwist4x4C<ComputeT>,
                                             GetRoundingModeColorConv<DstT>()>;

    const mpp::image::ColorTwist4x4C<ComputeT> op(aTwist, aConstant);

    const colorTwistInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion

#pragma region GammaCorrBT709
template <PixelType T>
ImageView<T> &ImageView<T>::GammaCorrBT709(ImageView<T> &aDst, float aNormFactor) const
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaBT709<ComputeT>, GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::GammaCorrBT709(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1, float aNormFactor)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 2)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaBT709<ComputeT>,
                                       GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, functor);
}

template <PixelType T>
void ImageView<T>::GammaCorrBT709(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormFactor)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaBT709<ComputeT>,
                                       GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                           aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::GammaCorrBT709(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
    ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormFactor)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 4)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());
    checkSameSize(aSrc0.ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaBT709<ComputeT>,
                                       GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                           aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::GammaCorrBT709(float aNormFactor)
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::GammaBT709<ComputeT>, GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaBT709<ComputeT> op(aNormFactor);

    const gammaInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion

#pragma region GammaInvCorrBT709
template <PixelType T>
ImageView<T> &ImageView<T>::GammaInvCorrBT709(ImageView<T> &aDst, float aNormFactor) const
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvBT709<ComputeT>,
                                GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaInvBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::GammaInvCorrBT709(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1, float aNormFactor)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 2)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvBT709<ComputeT>,
                                       GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaInvBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, functor);
}

template <PixelType T>
void ImageView<T>::GammaInvCorrBT709(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormFactor)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvBT709<ComputeT>,
                                       GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaInvBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                           aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::GammaInvCorrBT709(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
    ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormFactor)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 4)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());
    checkSameSize(aSrc0.ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvBT709<ComputeT>,
                                       GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaInvBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                           aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::GammaInvCorrBT709(float aNormFactor)
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::GammaInvBT709<ComputeT>,
                                        GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaInvBT709<ComputeT> op(aNormFactor);

    const gammaInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion

#pragma region GammaCorrsRGB
template <PixelType T>
ImageView<T> &ImageView<T>::GammaCorrsRGB(ImageView<T> &aDst, float aNormFactor) const
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammasRGB<ComputeT>, GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammasRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::GammaCorrsRGB(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst1, float aNormFactor)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 2)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammasRGB<ComputeT>,
                                       GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammasRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, functor);
}

template <PixelType T>
void ImageView<T>::GammaCorrsRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                 const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                 const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormFactor)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammasRGB<ComputeT>,
                                       GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammasRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                           aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::GammaCorrsRGB(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
    ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormFactor)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 4)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());
    checkSameSize(aSrc0.ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammasRGB<ComputeT>,
                                       GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammasRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                           aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::GammaCorrsRGB(float aNormFactor)
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::GammasRGB<ComputeT>, GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammasRGB<ComputeT> op(aNormFactor);

    const gammaInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion

#pragma region GammaInvCorrsRGB
template <PixelType T>
ImageView<T> &ImageView<T>::GammaInvCorrsRGB(ImageView<T> &aDst, float aNormFactor) const
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvsRGB<ComputeT>,
                                GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaInvsRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::GammaInvCorrsRGB(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst1, float aNormFactor)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 2)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvsRGB<ComputeT>,
                                       GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaInvsRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, functor);
}

template <PixelType T>
void ImageView<T>::GammaInvCorrsRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormFactor)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvsRGB<ComputeT>,
                                       GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaInvsRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                           aSrc2.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, functor);
}

template <PixelType T>
void ImageView<T>::GammaInvCorrsRGB(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
    ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormFactor)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 4)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());
    checkSameSize(aSrc0.ROI(), aDst3.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvsRGB<ComputeT>,
                                       GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaInvsRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                           aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), op);

    forEachPixelPlanar<DstT>(aDst0, aDst1, aDst2, aDst3, functor);
}

template <PixelType T>
ImageView<T> &ImageView<T>::GammaInvCorrsRGB(float aNormFactor)
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    using DstT     = T;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using gammaInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::GammaInvsRGB<ComputeT>, GetRoundingModeColorConv<DstT>()>;

    const mpp::image::GammaInvsRGB<ComputeT> op(aNormFactor);

    const gammaInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion

#pragma region GradientColorToGray
template <PixelType T>
ImageView<Vector1<remove_vector_t<T>>> &ImageView<T>::GradientColorToGray(ImageView<Vector1<remove_vector_t<T>>> &aDst,
                                                                          Norm aNorm) const
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_size_v<T> > 1)
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using DstT     = Vector1<remove_vector_t<T>>;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    if (aNorm == Norm::Inf)
    {
        using tograySrc =
            SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::Inf>,
                       GetRoundingModeColorConv<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::Inf> op;
        const tograySrc functor(PointerRoi(), Pitch(), op);
        forEachPixel(aDst, functor);
    }
    else if (aNorm == Norm::L1)
    {
        using tograySrc =
            SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L1>,
                       GetRoundingModeColorConv<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L1> op;
        const tograySrc functor(PointerRoi(), Pitch(), op);
        forEachPixel(aDst, functor);
    }
    else if (aNorm == Norm::L2)
    {
        using tograySrc =
            SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L2>,
                       GetRoundingModeColorConv<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L2> op;
        const tograySrc functor(PointerRoi(), Pitch(), op);
        forEachPixel(aDst, functor);
    }
    else
    {
        throw INVALIDARGUMENT(aNorm, "Unknown Norm '" << aNorm << "'. Expected either Inf, L1 or L2.");
    }

    return aDst;
}

template <PixelType T>
void ImageView<T>::GradientColorToGray(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst0, Norm aNorm)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 2)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    using SrcT     = T;
    using DstT     = Vector1<remove_vector_t<T>>;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    if (aNorm == Norm::Inf)
    {
        using tograySrc =
            SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::Inf>,
                              GetRoundingModeColorConv<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::Inf> op;
        const tograySrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), op);
        forEachPixel(aDst0, functor);
    }
    else if (aNorm == Norm::L1)
    {
        using tograySrc =
            SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L1>,
                              GetRoundingModeColorConv<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L1> op;
        const tograySrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), op);
        forEachPixel(aDst0, functor);
    }
    else if (aNorm == Norm::L2)
    {
        using tograySrc =
            SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L2>,
                              GetRoundingModeColorConv<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L2> op;
        const tograySrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), op);
        forEachPixel(aDst0, functor);
    }
    else
    {
        throw INVALIDARGUMENT(aNorm, "Unknown Norm '" << aNorm << "'. Expected either Inf, L1 or L2.");
    }
}

template <PixelType T>
void ImageView<T>::GradientColorToGray(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst0, Norm aNorm)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    using SrcT     = T;
    using DstT     = Vector1<remove_vector_t<T>>;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    if (aNorm == Norm::Inf)
    {
        using tograySrc =
            SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::Inf>,
                              GetRoundingModeColorConv<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::Inf> op;
        const tograySrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);
        forEachPixel(aDst0, functor);
    }
    else if (aNorm == Norm::L1)
    {
        using tograySrc =
            SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L1>,
                              GetRoundingModeColorConv<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L1> op;
        const tograySrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);
        forEachPixel(aDst0, functor);
    }
    else if (aNorm == Norm::L2)
    {
        using tograySrc =
            SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L2>,
                              GetRoundingModeColorConv<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L2> op;
        const tograySrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), op);
        forEachPixel(aDst0, functor);
    }
    else
    {
        throw INVALIDARGUMENT(aNorm, "Unknown Norm '" << aNorm << "'. Expected either Inf, L1 or L2.");
    }
}

template <PixelType T>
void ImageView<T>::GradientColorToGray(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst0, Norm aNorm)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 4)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    using SrcT     = T;
    using DstT     = Vector1<remove_vector_t<T>>;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    if (aNorm == Norm::Inf)
    {
        using tograySrc =
            SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::Inf>,
                              GetRoundingModeColorConv<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::Inf> op;
        const tograySrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), op);
        forEachPixel(aDst0, functor);
    }
    else if (aNorm == Norm::L1)
    {
        using tograySrc =
            SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L1>,
                              GetRoundingModeColorConv<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L1> op;
        const tograySrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), op);
        forEachPixel(aDst0, functor);
    }
    else if (aNorm == Norm::L2)
    {
        using tograySrc =
            SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L2>,
                              GetRoundingModeColorConv<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L2> op;
        const tograySrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                aSrc2.PointerRoi(), aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), op);
        forEachPixel(aDst0, functor);
    }
    else
    {
        throw INVALIDARGUMENT(aNorm, "Unknown Norm '" << aNorm << "'. Expected either Inf, L1 or L2.");
    }
}

#pragma endregion

#pragma region ColorToGray
template <PixelType T>
ImageView<Vector1<remove_vector_t<T>>> &ImageView<T>::ColorToGray(
    ImageView<Vector1<remove_vector_t<T>>> &aDst, const same_vector_size_different_type_t<T, float> &aWeights) const
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_size_v<T> > 1)
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using DstT     = Vector1<remove_vector_t<T>>;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using tograySrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorToGray<ComputeT>,
                                 GetRoundingModeColorConv<DstT>()>;
    const mpp::image::ColorToGray<ComputeT> op(aWeights);
    const tograySrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorToGray(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                               ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                               const same_vector_size_different_type_t<T, float> &aWeights)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 2)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());

    using SrcT     = T;
    using DstT     = Vector1<remove_vector_t<T>>;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using tograySrc = SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorToGray<ComputeT>,
                                        GetRoundingModeColorConv<DstT>()>;
    const mpp::image::ColorToGray<ComputeT> op(aWeights);
    const tograySrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), op);

    forEachPixel(aDst0, functor);
}

template <PixelType T>
void ImageView<T>::ColorToGray(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                               const same_vector_size_different_type_t<T, float> &aWeights)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());

    using SrcT     = T;
    using DstT     = Vector1<remove_vector_t<T>>;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using tograySrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorToGray<ComputeT>,
                                        GetRoundingModeColorConv<DstT>()>;
    const mpp::image::ColorToGray<ComputeT> op(aWeights);
    const tograySrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                            aSrc2.Pitch(), op);

    forEachPixel(aDst0, functor);
}

template <PixelType T>
void ImageView<T>::ColorToGray(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                               const same_vector_size_different_type_t<T, float> &aWeights)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 4)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());

    using SrcT     = T;
    using DstT     = Vector1<remove_vector_t<T>>;
    using ComputeT = same_vector_size_different_type_t<T, float>;

    constexpr size_t TupelSize = 1;

    using tograySrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorToGray<ComputeT>,
                                        GetRoundingModeColorConv<DstT>()>;
    const mpp::image::ColorToGray<ComputeT> op(aWeights);
    const tograySrc functor(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                            aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), op);

    forEachPixel(aDst0, functor);
}

#pragma endregion

#pragma region CFAToRGB
template <PixelType T>
ImageView<Vector3<remove_vector_t<T>>> &ImageView<T>::CFAToRGB(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                               BayerGridPosition aBayerGrid, Roi aAllowedReadRoi) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16uC1, T> || std::same_as<Pixel32uC1, T> ||
             std::same_as<Pixel16sC1, T> || std::same_as<Pixel32sC1, T> || std::same_as<Pixel16bfC1, T> ||
             std::same_as<Pixel16fC1, T> || std::same_as<Pixel32fC1, T>)
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));
    checkSameSize(ROI(), aDst.ROI());
    if (WidthRoi() % 2 != 0 || HeightRoi() % 2 != 0)
    {
        INVALIDARGUMENT(ROI, "The image ROI must have even width and height, but is: " << SizeRoi());
    }

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    using DstT   = Vector3<remove_vector_t<T>>;
    using OpT    = NOP<DstT>;
    using BCType = BorderControl<T, BorderType::Mirror, false, false, false, false>;
    const OpT op;
    const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

    if (aBayerGrid == BayerGridPosition::BGGR)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::BGGR>;
        const cfa functor(bc, op);

        forEachPixelBlock(aDst, functor);
    }
    else if (aBayerGrid == BayerGridPosition::GBRG)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::GBRG>;
        const cfa functor(bc, op);

        forEachPixelBlock(aDst, functor);
    }
    else if (aBayerGrid == BayerGridPosition::GRBG)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::GRBG>;
        const cfa functor(bc, op);

        forEachPixelBlock(aDst, functor);
    }
    else if (aBayerGrid == BayerGridPosition::RGGB)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::RGGB>;
        const cfa functor(bc, op);

        forEachPixelBlock(aDst, functor);
    }
    else
    {
        INVALIDARGUMENT(aBayerGrid, "Unknown BayerGridPosition: " << aBayerGrid);
    }

    return aDst;
}

template <PixelType T>
ImageView<Vector4<remove_vector_t<T>>> &ImageView<T>::CFAToRGB(ImageView<Vector4<remove_vector_t<T>>> &aDst,
                                                               remove_vector_t<T> aAlpha, BayerGridPosition aBayerGrid,
                                                               Roi aAllowedReadRoi) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16uC1, T> || std::same_as<Pixel32uC1, T> ||
             std::same_as<Pixel16sC1, T> || std::same_as<Pixel32sC1, T> || std::same_as<Pixel16bfC1, T> ||
             std::same_as<Pixel16fC1, T> || std::same_as<Pixel32fC1, T>)
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));
    checkSameSize(ROI(), aDst.ROI());
    if (WidthRoi() % 2 != 0 || HeightRoi() % 2 != 0)
    {
        INVALIDARGUMENT(ROI, "The image ROI must have even width and height, but is: " << SizeRoi());
    }

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    using DstT   = Vector4<remove_vector_t<T>>;
    using OpT    = SetAlphaConst<DstT, NOP<DstT>>;
    using BCType = BorderControl<T, BorderType::Mirror, false, false, false, false>;
    const OpT op(aAlpha, {});
    const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

    if (aBayerGrid == BayerGridPosition::BGGR)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::BGGR>;
        const cfa functor(bc, op);

        forEachPixelBlock(aDst, functor);
    }
    else if (aBayerGrid == BayerGridPosition::GBRG)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::GBRG>;
        const cfa functor(bc, op);

        forEachPixelBlock(aDst, functor);
    }
    else if (aBayerGrid == BayerGridPosition::GRBG)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::GRBG>;
        const cfa functor(bc, op);

        forEachPixelBlock(aDst, functor);
    }
    else if (aBayerGrid == BayerGridPosition::RGGB)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::RGGB>;
        const cfa functor(bc, op);

        forEachPixelBlock(aDst, functor);
    }
    else
    {
        INVALIDARGUMENT(aBayerGrid, "Unknown BayerGridPosition: " << aBayerGrid);
    }

    return aDst;
}

template <PixelType T>
ImageView<Vector1<remove_vector_t<T>>> &ImageView<T>::RGBToCFA(ImageView<Vector1<remove_vector_t<T>>> &aDst,
                                                               BayerGridPosition aBayerGrid) const
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<int, remove_vector_t<T>> ||
             std::same_as<uint, remove_vector_t<T>> || std::same_as<BFloat16, remove_vector_t<T>> ||
             std::same_as<HalfFp16, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(ROI(), aDst.ROI());
    if (WidthRoi() % 2 != 0 || HeightRoi() % 2 != 0)
    {
        INVALIDARGUMENT(ROI, "The image ROI must have even width and height, but is: " << SizeRoi());
    }

    using DstT   = Vector1<remove_vector_t<T>>;
    using BCType = BorderControl<T, BorderType::None, false, false, false, false>;
    const BCType bc(PointerRoi(), Pitch(), SizeRoi(), {0, 0});

    if (aBayerGrid == BayerGridPosition::BGGR)
    {
        using cfa = RGBToCFAFunctor<DstT, BCType, BayerGridPosition::BGGR>;
        const cfa functor(bc);

        forEachPixelBlock(aDst, functor);
    }
    else if (aBayerGrid == BayerGridPosition::GBRG)
    {
        using cfa = RGBToCFAFunctor<DstT, BCType, BayerGridPosition::GBRG>;
        const cfa functor(bc);

        forEachPixelBlock(aDst, functor);
    }
    else if (aBayerGrid == BayerGridPosition::GRBG)
    {
        using cfa = RGBToCFAFunctor<DstT, BCType, BayerGridPosition::GRBG>;
        const cfa functor(bc);

        forEachPixelBlock(aDst, functor);
    }
    else if (aBayerGrid == BayerGridPosition::RGGB)
    {
        using cfa = RGBToCFAFunctor<DstT, BCType, BayerGridPosition::RGGB>;
        const cfa functor(bc);

        forEachPixelBlock(aDst, functor);
    }
    else
    {
        INVALIDARGUMENT(aBayerGrid, "Unknown BayerGridPosition: " << aBayerGrid);
    }

    return aDst;
}
#pragma endregion

#pragma region LUTPalette
template <PixelType T>
void ImageView<T>::LUTToPalette(const int *aLevels, const int *aValues, int aLUTSize,
                                Vector1<remove_vector_t<T>> *aPalette, InterpolationMode aInterpolationMode)
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
             std::same_as<remove_vector_t<T>, ushort>)
{
    switch (aInterpolationMode)
    {
        case mpp::InterpolationMode::NearestNeighbor:
            LUTtoPalette(aLevels, aValues, aLUTSize, reinterpret_cast<remove_vector_t<T> *>(aPalette));
            break;
        case mpp::InterpolationMode::Linear:
            LUTtoPaletteLinear(aLevels, aValues, aLUTSize, reinterpret_cast<remove_vector_t<T> *>(aPalette));
            break;
        case mpp::InterpolationMode::CubicLagrange:
            LUTtoPaletteCubic(aLevels, aValues, aLUTSize, reinterpret_cast<remove_vector_t<T> *>(aPalette));
            break;
        default:
            throw INVALIDARGUMENT(
                aInterpolationMode,
                "Unsupported interpolation mode: "
                    << aInterpolationMode
                    << ". Only NearestNeighbor, Linear and CubicLagrange interpolation modes are supported.");
            break;
    }
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTPalette(ImageView<T> &aDst, const T *aPalette, int aBitSize) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
             std::same_as<remove_vector_t<T>, ushort>) &&
            (vector_size_v<T> == 1)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = T;
    using DstT           = T;
    using ComputeT       = T;
    using LutT           = T;

    constexpr size_t TupelSize = 1;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTPalette(const T *aPalette, int aBitSize)
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>)
{
    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = T;
    using DstT           = T;
    using ComputeT       = T;
    using LutT           = T;

    constexpr size_t TupelSize = 1;

    using lutInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                                      RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutInplace functor(op);

    forEachPixel(*this, functor);

    return *this;
}

template <PixelType T>
ImageView<Vector3<remove_vector_t<T>>> &ImageView<T>::LUTPalette(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                                 const Vector3<remove_vector_t<T>> *aPalette,
                                                                 int aBitSize) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = T;
    using DstT           = Vector3<remove_vector_t<T>>;
    using ComputeT       = T;
    using LutT           = Vector3<remove_vector_t<T>>;

    constexpr size_t TupelSize = 1;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<Vector3<remove_vector_t<T>>> &ImageView<T>::LUTPalette(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                                 const Vector4A<remove_vector_t<T>> *aPalette,
                                                                 int aBitSize) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = T;
    using DstT           = Vector3<remove_vector_t<T>>;
    using ComputeT       = T;
    using LutT           = Vector4A<remove_vector_t<T>>;

    constexpr size_t TupelSize = 1;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<Vector4A<remove_vector_t<T>>> &ImageView<T>::LUTPalette(ImageView<Vector4A<remove_vector_t<T>>> &aDst,
                                                                  const Vector4A<remove_vector_t<T>> *aPalette,
                                                                  int aBitSize) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = T;
    using DstT           = Vector4A<remove_vector_t<T>>;
    using ComputeT       = T;
    using LutT           = Vector4A<remove_vector_t<T>>;

    constexpr size_t TupelSize = 1;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<Vector4A<remove_vector_t<T>>> &ImageView<T>::LUTPalette(ImageView<Vector4A<remove_vector_t<T>>> &aDst,
                                                                  const Vector3<remove_vector_t<T>> *aPalette,
                                                                  int aBitSize) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = T;
    using DstT           = Vector4A<remove_vector_t<T>>;
    using ComputeT       = T;
    using LutT           = Vector3<remove_vector_t<T>>;

    constexpr size_t TupelSize = 1;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<Vector4<remove_vector_t<T>>> &ImageView<T>::LUTPalette(ImageView<Vector4<remove_vector_t<T>>> &aDst,
                                                                 const Vector4<remove_vector_t<T>> *aPalette,
                                                                 int aBitSize) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = T;
    using DstT           = Vector4<remove_vector_t<T>>;
    using ComputeT       = T;
    using LutT           = Vector4<remove_vector_t<T>>;

    constexpr size_t TupelSize = 1;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTPalette(ImageView<T> &aDst,
                                       const Vector1<remove_vector_t<T>> *const aPalette[vector_active_size_v<T>],
                                       int aBitSize) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
             std::same_as<remove_vector_t<T>, ushort>) &&
            (vector_active_size_v<T> >= 2)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));

    constexpr size_t TupelSize = 1;

    if constexpr (vector_active_size_v<T> == 2)
    {
        using lutSrc = SrcFunctor<TupelSize, T, T, T, mpp::image::LUTPalettePlanar2WithBounds<T>, RoundingMode::None>;

        const mpp::image::LUTPalettePlanar2WithBounds<T> op(reinterpret_cast<const remove_vector_t<T> *>(aPalette[0]),
                                                            reinterpret_cast<const remove_vector_t<T> *>(aPalette[1]),
                                                            indexBound);
        const lutSrc functor(PointerRoi(), Pitch(), op);
        forEachPixel(aDst, functor);
    }
    else if constexpr (vector_active_size_v<T> == 3)
    {
        using lutSrc = SrcFunctor<TupelSize, T, T, T, mpp::image::LUTPalettePlanar3WithBounds<T>, RoundingMode::None>;

        const mpp::image::LUTPalettePlanar3WithBounds<T> op(reinterpret_cast<const remove_vector_t<T> *>(aPalette[0]),
                                                            reinterpret_cast<const remove_vector_t<T> *>(aPalette[1]),
                                                            reinterpret_cast<const remove_vector_t<T> *>(aPalette[2]),
                                                            indexBound);
        const lutSrc functor(PointerRoi(), Pitch(), op);
        forEachPixel(aDst, functor);
    }
    else
    {
        using lutSrc = SrcFunctor<TupelSize, T, T, T, mpp::image::LUTPalettePlanar4WithBounds<T>, RoundingMode::None>;

        const mpp::image::LUTPalettePlanar4WithBounds<T> op(reinterpret_cast<const remove_vector_t<T> *>(aPalette[0]),
                                                            reinterpret_cast<const remove_vector_t<T> *>(aPalette[1]),
                                                            reinterpret_cast<const remove_vector_t<T> *>(aPalette[2]),
                                                            reinterpret_cast<const remove_vector_t<T> *>(aPalette[3]),
                                                            indexBound);
        const lutSrc functor(PointerRoi(), Pitch(), op);
        forEachPixel(aDst, functor);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTPalette(const Vector1<remove_vector_t<T>> *const aPalette[vector_active_size_v<T>],
                                       int aBitSize)
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
             std::same_as<remove_vector_t<T>, ushort>) &&
            (vector_active_size_v<T> >= 2)
{
    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));

    constexpr size_t TupelSize = 1;

    if constexpr (vector_active_size_v<T> == 2)
    {
        using lutInplace =
            InplaceFunctor<TupelSize, T, T, mpp::image::LUTPalettePlanar2WithBounds<T>, RoundingMode::None>;

        const mpp::image::LUTPalettePlanar2WithBounds<T> op(reinterpret_cast<const remove_vector_t<T> *>(aPalette[0]),
                                                            reinterpret_cast<const remove_vector_t<T> *>(aPalette[1]),
                                                            indexBound);
        const lutInplace functor(op);
        forEachPixel(*this, functor);
    }
    else if constexpr (vector_active_size_v<T> == 3)
    {
        using lutInplace =
            InplaceFunctor<TupelSize, T, T, mpp::image::LUTPalettePlanar3WithBounds<T>, RoundingMode::None>;

        const mpp::image::LUTPalettePlanar3WithBounds<T> op(reinterpret_cast<const remove_vector_t<T> *>(aPalette[0]),
                                                            reinterpret_cast<const remove_vector_t<T> *>(aPalette[1]),
                                                            reinterpret_cast<const remove_vector_t<T> *>(aPalette[2]),
                                                            indexBound);
        const lutInplace functor(op);
        forEachPixel(*this, functor);
    }
    else
    {
        using lutInplace =
            InplaceFunctor<TupelSize, T, T, mpp::image::LUTPalettePlanar4WithBounds<T>, RoundingMode::None>;

        const mpp::image::LUTPalettePlanar4WithBounds<T> op(reinterpret_cast<const remove_vector_t<T> *>(aPalette[0]),
                                                            reinterpret_cast<const remove_vector_t<T> *>(aPalette[1]),
                                                            reinterpret_cast<const remove_vector_t<T> *>(aPalette[2]),
                                                            reinterpret_cast<const remove_vector_t<T> *>(aPalette[3]),
                                                            indexBound);
        const lutInplace functor(op);
        forEachPixel(*this, functor);
    }

    return *this;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::LUTPalette(ImageView<Pixel8uC1> &aDst, const Pixel8uC1 *aPalette,
                                               int aBitSize) const
    requires(std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = T;
    using DstT           = Pixel8uC1;
    using ComputeT       = T;
    using LutT           = Pixel8uC1;

    constexpr size_t TupelSize = 1;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None, voidType, voidType, true>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC3> &ImageView<T>::LUTPalette(ImageView<Pixel8uC3> &aDst, const Pixel8uC3 *aPalette,
                                               int aBitSize) const
    requires(std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = T;
    using DstT           = Pixel8uC3;
    using ComputeT       = T;
    using LutT           = Pixel8uC3;

    constexpr size_t TupelSize = 1;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None, voidType, voidType, true>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC4> &ImageView<T>::LUTPalette(ImageView<Pixel8uC4> &aDst, const Pixel8uC4 *aPalette,
                                               int aBitSize) const
    requires(std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = T;
    using DstT           = Pixel8uC4;
    using ComputeT       = T;
    using LutT           = Pixel8uC4;

    constexpr size_t TupelSize = 1;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None, voidType, voidType, true>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC4A> &ImageView<T>::LUTPalette(ImageView<Pixel8uC4A> &aDst, const Pixel8uC4A *aPalette,
                                                int aBitSize) const
    requires(std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    const int indexBound = static_cast<int>(1u << static_cast<uint>(aBitSize));
    using SrcT           = T;
    using DstT           = Pixel8uC4A;
    using ComputeT       = T;
    using LutT           = Pixel8uC4A;

    constexpr size_t TupelSize = 1;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT>,
                              RoundingMode::None, voidType, voidType, true>;

    const mpp::image::LUTPaletteWithBounds<SrcT, LutT, DstT> op(aPalette, indexBound);

    const lutSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

#pragma endregion
#pragma region LUT
template <PixelType T>
void ImageView<T>::LUTAccelerator(const Pixel32fC1 *aLevels, int *aAccelerator, int aLUTSize, int aAcceleratorSize)
    requires(RealFloatingPoint<remove_vector_t<T>>)
{
    mpp::LUTAccelerator(reinterpret_cast<const float *>(aLevels), aLUTSize, aAccelerator, aAcceleratorSize);
}

template <typename TT> struct GetLUTComputeT
{
    using type = double;
};
template <> struct GetLUTComputeT<HalfFp16>
{
    using type = float;
};
template <> struct GetLUTComputeT<BFloat16>
{
    using type = float;
};

template <PixelType T>
ImageView<T> &ImageView<T>::LUT(ImageView<T> &aDst, const Pixel32fC1 *aLevels, const Pixel32fC1 *aValues,
                                const int *aAccelerator, int aLutSize, int aAcceleratorSize,
                                InterpolationMode aInterpolationMode) const
    requires(RealFloatingPoint<remove_vector_t<T>>) && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT        = T;
    using DstT        = T;
    using ComputeT    = T;
    using LutComputeT = typename GetLUTComputeT<remove_vector_t<T>>::type;
    using LutT        = float;

    constexpr size_t TupelSize = 1;
    if (aInterpolationMode == InterpolationMode::NearestNeighbor)
    {
        using opT            = mpp::image::LUT1Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
        using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
        using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

        const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels), reinterpret_cast<const float *>(aValues),
                                     aAccelerator, aLutSize, aAcceleratorSize);
        const opT op(lutSrc0);
        const lutSrc functor(PointerRoi(), Pitch(), op);
        forEachPixel(aDst, functor);
    }
    else if (aInterpolationMode == InterpolationMode::Linear)
    {
        using opT            = mpp::image::LUT1Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
        using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
        using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

        const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels), reinterpret_cast<const float *>(aValues),
                                     aAccelerator, aLutSize, aAcceleratorSize);
        const opT op(lutSrc0);
        const lutSrc functor(PointerRoi(), Pitch(), op);
        forEachPixel(aDst, functor);
    }
    else if (aInterpolationMode == InterpolationMode::CubicLagrange)
    {
        using opT            = mpp::image::LUT1Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
        using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
        using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

        const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels), reinterpret_cast<const float *>(aValues),
                                     aAccelerator, aLutSize, aAcceleratorSize);
        const opT op(lutSrc0);
        const lutSrc functor(PointerRoi(), Pitch(), op);
        forEachPixel(aDst, functor);
    }
    else
    {
        throw INVALIDARGUMENT(aInterpolationMode, "Unsupported interpolation mode. Only NearestNeighbor, Linear and "
                                                  "CubicLagrange are supported, but provided aInterpolationMode is "
                                                      << aInterpolationMode);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUT(ImageView<T> &aDst, const Pixel32fC1 *const aLevels[vector_active_size_v<T>],
                                const Pixel32fC1 *const aValues[vector_active_size_v<T>],
                                const int *const aAccelerator[vector_active_size_v<T>],
                                int const aLutSize[vector_active_size_v<T>],
                                int const aAcceleratorSize[vector_active_size_v<T>],
                                InterpolationMode aInterpolationMode) const
    requires(RealFloatingPoint<remove_vector_t<T>>) && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT        = T;
    using DstT        = T;
    using ComputeT    = T;
    using LutComputeT = typename GetLUTComputeT<remove_vector_t<T>>::type;
    using LutT        = float;

    constexpr size_t TupelSize = 1;

    if (aInterpolationMode != InterpolationMode::NearestNeighbor && aInterpolationMode != InterpolationMode::Linear &&
        aInterpolationMode != InterpolationMode::CubicLagrange)
    {
        throw INVALIDARGUMENT(aInterpolationMode, "Unsupported interpolation mode. Only NearestNeighbor, Linear and "
                                                  "CubicLagrange are supported, but provided aInterpolationMode is "
                                                      << aInterpolationMode);
    }

    if constexpr (vector_active_size_v<T> == 2)
    {
        if (aInterpolationMode == InterpolationMode::NearestNeighbor)
        {
            using opT            = mpp::image::LUT2Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);

            const opT op(lutSrc0, lutSrc1);
            const lutSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
        else if (aInterpolationMode == InterpolationMode::Linear)
        {
            using opT            = mpp::image::LUT2Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);

            const opT op(lutSrc0, lutSrc1);
            const lutSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
        else
        {
            using opT            = mpp::image::LUT2Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);

            const opT op(lutSrc0, lutSrc1);
            const lutSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
    }
    else if constexpr (vector_active_size_v<T> == 3)
    {
        if (aInterpolationMode == InterpolationMode::NearestNeighbor)
        {
            using opT            = mpp::image::LUT3Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);

            const opT op(lutSrc0, lutSrc1, lutSrc2);
            const lutSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
        else if (aInterpolationMode == InterpolationMode::Linear)
        {
            using opT            = mpp::image::LUT3Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);

            const opT op(lutSrc0, lutSrc1, lutSrc2);
            const lutSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
        else
        {
            using opT            = mpp::image::LUT3Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);

            const opT op(lutSrc0, lutSrc1, lutSrc2);
            const lutSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
    }
    else
    {
        if (aInterpolationMode == InterpolationMode::NearestNeighbor)
        {
            using opT            = mpp::image::LUT4Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);
            const LUTChannelSrcT lutSrc3(reinterpret_cast<const float *>(aLevels[3]),
                                         reinterpret_cast<const float *>(aValues[3]), aAccelerator[3], aLutSize[3],
                                         aAcceleratorSize[3]);

            const opT op(lutSrc0, lutSrc1, lutSrc2, lutSrc3);
            const lutSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
        else if (aInterpolationMode == InterpolationMode::Linear)
        {
            using opT            = mpp::image::LUT4Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);
            const LUTChannelSrcT lutSrc3(reinterpret_cast<const float *>(aLevels[3]),
                                         reinterpret_cast<const float *>(aValues[3]), aAccelerator[3], aLutSize[3],
                                         aAcceleratorSize[3]);

            const opT op(lutSrc0, lutSrc1, lutSrc2, lutSrc3);
            const lutSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
        else
        {
            using opT            = mpp::image::LUT4Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);
            const LUTChannelSrcT lutSrc3(reinterpret_cast<const float *>(aLevels[3]),
                                         reinterpret_cast<const float *>(aValues[3]), aAccelerator[3], aLutSize[3],
                                         aAcceleratorSize[3]);

            const opT op(lutSrc0, lutSrc1, lutSrc2, lutSrc3);
            const lutSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUT(const Pixel32fC1 *aLevels, const Pixel32fC1 *aValues, const int *aAccelerator,
                                int aLutSize, int aAcceleratorSize, InterpolationMode aInterpolationMode)
    requires(RealFloatingPoint<remove_vector_t<T>>) && (vector_active_size_v<T> == 1)
{
    using SrcT        = T;
    using DstT        = T;
    using ComputeT    = T;
    using LutComputeT = typename GetLUTComputeT<remove_vector_t<T>>::type;
    using LutT        = float;

    constexpr size_t TupelSize = 1;
    if (aInterpolationMode == InterpolationMode::NearestNeighbor)
    {
        using opT            = mpp::image::LUT1Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
        using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
        using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

        const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels), reinterpret_cast<const float *>(aValues),
                                     aAccelerator, aLutSize, aAcceleratorSize);
        const opT op(lutSrc0);
        const lutSrc functor(op);
        forEachPixel(*this, functor);
    }
    else if (aInterpolationMode == InterpolationMode::Linear)
    {
        using opT            = mpp::image::LUT1Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
        using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
        using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

        const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels), reinterpret_cast<const float *>(aValues),
                                     aAccelerator, aLutSize, aAcceleratorSize);
        const opT op(lutSrc0);
        const lutSrc functor(op);
        forEachPixel(*this, functor);
    }
    else if (aInterpolationMode == InterpolationMode::CubicLagrange)
    {
        using opT            = mpp::image::LUT1Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
        using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
        using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

        const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels), reinterpret_cast<const float *>(aValues),
                                     aAccelerator, aLutSize, aAcceleratorSize);
        const opT op(lutSrc0);
        const lutSrc functor(op);
        forEachPixel(*this, functor);
    }
    else
    {
        throw INVALIDARGUMENT(aInterpolationMode, "Unsupported interpolation mode. Only NearestNeighbor, Linear and "
                                                  "CubicLagrange are supported, but provided aInterpolationMode is "
                                                      << aInterpolationMode);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUT(const Pixel32fC1 *const aLevels[vector_active_size_v<T>],
                                const Pixel32fC1 *const aValues[vector_active_size_v<T>],
                                const int *const aAccelerator[vector_active_size_v<T>],
                                int const aLutSize[vector_active_size_v<T>],
                                int const aAcceleratorSize[vector_active_size_v<T>],
                                InterpolationMode aInterpolationMode)
    requires(RealFloatingPoint<remove_vector_t<T>>) && (vector_active_size_v<T> > 1)
{
    using SrcT        = T;
    using DstT        = T;
    using ComputeT    = T;
    using LutComputeT = typename GetLUTComputeT<remove_vector_t<T>>::type;
    using LutT        = float;

    constexpr size_t TupelSize = 1;

    if (aInterpolationMode != InterpolationMode::NearestNeighbor && aInterpolationMode != InterpolationMode::Linear &&
        aInterpolationMode != InterpolationMode::CubicLagrange)
    {
        throw INVALIDARGUMENT(aInterpolationMode, "Unsupported interpolation mode. Only NearestNeighbor, Linear and "
                                                  "CubicLagrange are supported, but provided aInterpolationMode is "
                                                      << aInterpolationMode);
    }

    if constexpr (vector_active_size_v<T> == 2)
    {
        if (aInterpolationMode == InterpolationMode::NearestNeighbor)
        {
            using opT            = mpp::image::LUT2Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);

            const opT op(lutSrc0, lutSrc1);
            const lutSrc functor(op);
            forEachPixel(*this, functor);
        }
        else if (aInterpolationMode == InterpolationMode::Linear)
        {
            using opT            = mpp::image::LUT2Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);

            const opT op(lutSrc0, lutSrc1);
            const lutSrc functor(op);
            forEachPixel(*this, functor);
        }
        else
        {
            using opT            = mpp::image::LUT2Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);

            const opT op(lutSrc0, lutSrc1);
            const lutSrc functor(op);
            forEachPixel(*this, functor);
        }
    }
    else if constexpr (vector_active_size_v<T> == 3)
    {
        if (aInterpolationMode == InterpolationMode::NearestNeighbor)
        {
            using opT            = mpp::image::LUT3Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);

            const opT op(lutSrc0, lutSrc1, lutSrc2);
            const lutSrc functor(op);
            forEachPixel(*this, functor);
        }
        else if (aInterpolationMode == InterpolationMode::Linear)
        {
            using opT            = mpp::image::LUT3Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);

            const opT op(lutSrc0, lutSrc1, lutSrc2);
            const lutSrc functor(op);
            forEachPixel(*this, functor);
        }
        else
        {
            using opT            = mpp::image::LUT3Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);

            const opT op(lutSrc0, lutSrc1, lutSrc2);
            const lutSrc functor(op);
            forEachPixel(*this, functor);
        }
    }
    else
    {
        if (aInterpolationMode == InterpolationMode::NearestNeighbor)
        {
            using opT            = mpp::image::LUT4Channel<SrcT, LutT, LutComputeT, InterpolationMode::NearestNeighbor>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);
            const LUTChannelSrcT lutSrc3(reinterpret_cast<const float *>(aLevels[3]),
                                         reinterpret_cast<const float *>(aValues[3]), aAccelerator[3], aLutSize[3],
                                         aAcceleratorSize[3]);

            const opT op(lutSrc0, lutSrc1, lutSrc2, lutSrc3);
            const lutSrc functor(op);
            forEachPixel(*this, functor);
        }
        else if (aInterpolationMode == InterpolationMode::Linear)
        {
            using opT            = mpp::image::LUT4Channel<SrcT, LutT, LutComputeT, InterpolationMode::Linear>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);
            const LUTChannelSrcT lutSrc3(reinterpret_cast<const float *>(aLevels[3]),
                                         reinterpret_cast<const float *>(aValues[3]), aAccelerator[3], aLutSize[3],
                                         aAcceleratorSize[3]);

            const opT op(lutSrc0, lutSrc1, lutSrc2, lutSrc3);
            const lutSrc functor(op);
            forEachPixel(*this, functor);
        }
        else
        {
            using opT            = mpp::image::LUT4Channel<SrcT, LutT, LutComputeT, InterpolationMode::CubicLagrange>;
            using LUTChannelSrcT = typename opT::LUTChannelSrc_type;
            using lutSrc         = InplaceFunctor<TupelSize, ComputeT, DstT, opT, RoundingMode::None>;

            const LUTChannelSrcT lutSrc0(reinterpret_cast<const float *>(aLevels[0]),
                                         reinterpret_cast<const float *>(aValues[0]), aAccelerator[0], aLutSize[0],
                                         aAcceleratorSize[0]);
            const LUTChannelSrcT lutSrc1(reinterpret_cast<const float *>(aLevels[1]),
                                         reinterpret_cast<const float *>(aValues[1]), aAccelerator[1], aLutSize[1],
                                         aAcceleratorSize[1]);
            const LUTChannelSrcT lutSrc2(reinterpret_cast<const float *>(aLevels[2]),
                                         reinterpret_cast<const float *>(aValues[2]), aAccelerator[2], aLutSize[2],
                                         aAcceleratorSize[2]);
            const LUTChannelSrcT lutSrc3(reinterpret_cast<const float *>(aLevels[3]),
                                         reinterpret_cast<const float *>(aValues[3]), aAccelerator[3], aLutSize[3],
                                         aAcceleratorSize[3]);

            const opT op(lutSrc0, lutSrc1, lutSrc2, lutSrc3);
            const lutSrc functor(op);
            forEachPixel(*this, functor);
        }
    }

    return *this;
}

#pragma endregion
#pragma region Lut3D

template <PixelType T>
ImageView<T> &ImageView<T>::LUTTrilinear(ImageView<T> &aDst, const Vector3<remove_vector_t<T>> *aLut3D,
                                         const Vector3<remove_vector_t<T>> &aMinLevel,
                                         const Vector3<remove_vector_t<T>> &aMaxLevel, const Pixel32sC3 &aLutSize) const
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> >= 3)
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = T;
    using LutT     = Vector3<remove_vector_t<T>>;

    constexpr size_t TupelSize = 1;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTTrilinear<T, LutT>, RoundingMode::None>;

    const mpp::image::LUTTrilinear<T, LutT> op(aLut3D, aMinLevel, aMaxLevel - aMinLevel, aLutSize);

    const lutSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTTrilinear(ImageView<T> &aDst, const Vector4A<remove_vector_t<T>> *aLut3D,
                                         const Vector3<remove_vector_t<T>> &aMinLevel,
                                         const Vector3<remove_vector_t<T>> &aMaxLevel, const Pixel32sC3 &aLutSize) const
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> >= 3)
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = T;
    using LutT     = Vector4A<remove_vector_t<T>>;

    constexpr size_t TupelSize = 1;

    using lutSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUTTrilinear<T, LutT>, RoundingMode::None>;

    const mpp::image::LUTTrilinear<T, LutT> op(aLut3D, aMinLevel, aMaxLevel - aMinLevel, aLutSize);

    const lutSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTTrilinear(const Vector3<remove_vector_t<T>> *aLut3D,
                                         const Vector3<remove_vector_t<T>> &aMinLevel,
                                         const Vector3<remove_vector_t<T>> &aMaxLevel, const Pixel32sC3 &aLutSize)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> >= 3)
{
    using DstT     = T;
    using ComputeT = T;
    using LutT     = Vector3<remove_vector_t<T>>;

    constexpr size_t TupelSize = 1;

    using lutSrc = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LUTTrilinear<T, LutT>, RoundingMode::None>;

    const mpp::image::LUTTrilinear<T, LutT> op(aLut3D, aMinLevel, aMaxLevel - aMinLevel, aLutSize);

    const lutSrc functor(op);

    forEachPixel(*this, functor);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTTrilinear(const Vector4A<remove_vector_t<T>> *aLut3D,
                                         const Vector3<remove_vector_t<T>> &aMinLevel,
                                         const Vector3<remove_vector_t<T>> &aMaxLevel, const Pixel32sC3 &aLutSize)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> >= 3)
{
    using DstT     = T;
    using ComputeT = T;
    using LutT     = Vector4A<remove_vector_t<T>>;

    constexpr size_t TupelSize = 1;

    using lutSrc = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LUTTrilinear<T, LutT>, RoundingMode::None>;

    const mpp::image::LUTTrilinear<T, LutT> op(aLut3D, aMinLevel, aMaxLevel - aMinLevel, aLutSize);

    const lutSrc functor(op);

    forEachPixel(*this, functor);

    return *this;
}

#pragma endregion

#pragma region CompColorKey

template <PixelType T>
ImageView<T> &ImageView<T>::CompColorKey(const ImageView<T> &aSrc2, const T &aColorKey, ImageView<T> &aDst) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = T;

    constexpr size_t TupelSize = 1;

    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::CompColorKey<SrcT>,
                                        RoundingMode::None, voidType, voidType, true>;
    const mpp::image::CompColorKey<SrcT> op(aColorKey);

    const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::CompColorKey(const ImageView<T> &aSrc2, const T &aColorKey)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = T;

    constexpr size_t TupelSize = 1;

    using compareInplaceSrc = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::CompColorKey<SrcT>,
                                                RoundingMode::None, voidType, voidType>;
    const mpp::image::CompColorKey<SrcT> op(aColorKey);

    const compareInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
    forEachPixel(*this, functor);

    return *this;
}
#pragma endregion

#pragma region ConvertSampling422
template <PixelType T>
void ImageView<T>::ConvertSampling422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                      ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, bool aSwapLumaChroma) const
    requires RealVector<T> && (vector_size_v<T> == 2)
{
    checkSameSize(*this, aDstLuma);
    checkSameSize(SizeRoi() / Vec2i(2, 1), aDstChroma.SizeRoi());

    using SrcT     = Vector3<remove_vector_t<T>>;
    using DstT     = Vector3<remove_vector_t<T>>;
    using ComputeT = Vector3<remove_vector_t<T>>;

    constexpr size_t TupelSize = 1;

    if (aSwapLumaChroma)
    {
        using nopSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::NOP<ComputeT>,
                                       Src422C2Layout::CbYCr, RoundingMode::None>;

        const mpp::image::NOP<ComputeT> op;

        const nopSrc functor(PointerRoi(), Pitch(), op);
        forEachPixel422<SrcT>(aDstLuma, aDstChroma, ChromaSubsamplePos::TopLeft, functor);
    }
    else
    {
        using nopSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::NOP<ComputeT>,
                                       Src422C2Layout::YCbCr, RoundingMode::None>;

        const mpp::image::NOP<ComputeT> op;

        const nopSrc functor(PointerRoi(), Pitch(), op);

        forEachPixel422<SrcT>(aDstLuma, aDstChroma, ChromaSubsamplePos::TopLeft, functor);
    }
}

template <PixelType T>
void ImageView<T>::ConvertSampling422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                      ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                      ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, bool aSwapLumaChroma) const
    requires RealVector<T> && (vector_size_v<T> == 2)
{
    checkSameSize(*this, aDstLuma);
    checkSameSize(SizeRoi() / Vec2i(2, 1), aDstChroma1.SizeRoi());
    checkSameSize(SizeRoi() / Vec2i(2, 1), aDstChroma2.SizeRoi());

    using SrcT     = Vector3<remove_vector_t<T>>;
    using DstT     = Vector3<remove_vector_t<T>>;
    using ComputeT = Vector3<remove_vector_t<T>>;

    constexpr size_t TupelSize = 1;

    if (aSwapLumaChroma)
    {
        using nopSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::NOP<ComputeT>,
                                       Src422C2Layout::CbYCr, RoundingMode::None>;

        const mpp::image::NOP<ComputeT> op;

        const nopSrc functor(PointerRoi(), Pitch(), op);
        forEachPixel422<SrcT>(aDstLuma, aDstChroma1, aDstChroma2, ChromaSubsamplePos::TopLeft, functor);
    }
    else
    {
        using nopSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::NOP<ComputeT>,
                                       Src422C2Layout::YCbCr, RoundingMode::None>;

        const mpp::image::NOP<ComputeT> op;

        const nopSrc functor(PointerRoi(), Pitch(), op);

        forEachPixel422<SrcT>(aDstLuma, aDstChroma1, aDstChroma2, ChromaSubsamplePos::TopLeft, functor);
    }
}

template <PixelType T>
ImageView<Vector2<remove_vector_t<T>>> &ImageView<T>::ConvertSampling422(
    ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma, ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
    ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma, bool aSwapLumaChroma)
    requires RealVector<T> && (vector_size_v<T> == 3)
{
    checkSameSize(aSrcLuma, aDstLumaChroma);
    checkSameSize(aSrcLuma.SizeRoi() / Vec2i(2, 1), aSrcChroma.SizeRoi());
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = T;

    const Size2D sizeChroma(aSrcLuma.SizeRoi().x / 2, aSrcLuma.SizeRoi().y);

    constexpr size_t TupelSize = 1;

    using nopSrc =
        Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::NOP<ComputeT>, ChromaSubsamplePos::TopLeft,
                      InterpolationMode::NearestNeighbor, false, false, RoundingMode::None>;

    const mpp::image::NOP<ComputeT> op;

    const nopSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(), aSrcChroma.Pitch(),
                         sizeChroma, op);

    forEachPixel422<T>(aDstLumaChroma, ChromaSubsamplePos::TopLeft,
                       aSwapLumaChroma ? Dst422C2Layout::CbYCr : Dst422C2Layout::YCbCr, functor);

    return aDstLumaChroma;
}

template <PixelType T>
ImageView<Vector2<remove_vector_t<T>>> &ImageView<T>::ConvertSampling422(
    ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma, ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
    ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma,
    bool aSwapLumaChroma)
    requires RealVector<T> && (vector_size_v<T> == 3)
{
    checkSameSize(aSrcLuma, aDstLumaChroma);
    checkSameSize(aSrcLuma.SizeRoi() / Vec2i(2, 1), aSrcChroma1.SizeRoi());
    checkSameSize(aSrcLuma.SizeRoi() / Vec2i(2, 1), aSrcChroma2.SizeRoi());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = T;

    const Size2D sizeChroma(aSrcLuma.SizeRoi().x / 2, aSrcLuma.SizeRoi().y);

    constexpr size_t TupelSize = 1;

    using nopSrc =
        Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::NOP<ComputeT>, ChromaSubsamplePos::TopLeft,
                      InterpolationMode::NearestNeighbor, false, true, RoundingMode::None>;

    const mpp::image::NOP<ComputeT> op;

    const nopSrc functor(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(), aSrcChroma1.Pitch(),
                         aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(), sizeChroma, op);

    forEachPixel422<T>(aDstLumaChroma, ChromaSubsamplePos::TopLeft,
                       aSwapLumaChroma ? Dst422C2Layout::CbYCr : Dst422C2Layout::YCbCr, functor);

    return aDstLumaChroma;
}

#pragma endregion
#pragma endregion
// NOLINTEND(readability-suspicious-call-argument)
} // namespace mpp::image::cpuSimple