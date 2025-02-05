#pragma once
#include <backends/simple_cpu/image/forEachPixel.h>
#include <backends/simple_cpu/image/forEachPixelMasked.h>
#include <backends/simple_cpu/image/forEachPixelPlanar.h>
#include <backends/simple_cpu/image/forEachPixelSingleChannel.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/bfloat16.h>
#include <common/complex.h>
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
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/utilities.h>
#include <common/vector1.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace opp::image::cpuSimple
{
#pragma region MinEvery
template <PixelType T>
ImageView<T> &ImageView<T>::MinEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using minEverySrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Min<ComputeT>, RoundingMode::None>;
    const opp::Min<ComputeT> op;
    const minEverySrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MinEvery(const ImageView<T> &aSrc2)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using minEveryInplaceSrc =
        InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Min<ComputeT>, RoundingMode::None>;
    const opp::Min<ComputeT> op;
    const minEveryInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
    forEachPixel(*this, functor);

    return *this;
}
#pragma endregion

#pragma region MaxEvery
template <PixelType T>
ImageView<T> &ImageView<T>::MaxEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using maxEverySrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Max<ComputeT>, RoundingMode::None>;
    const opp::Max<ComputeT> op;
    const maxEverySrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MaxEvery(const ImageView<T> &aSrc2)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using maxEveryInplaceSrc =
        InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Max<ComputeT>, RoundingMode::None>;
    const opp::Max<ComputeT> op;
    const maxEveryInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
    forEachPixel(*this, functor);

    return *this;
}
#pragma endregion

} // namespace opp::image::cpuSimple