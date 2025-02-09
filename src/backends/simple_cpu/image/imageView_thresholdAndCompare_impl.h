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

#pragma region Compare
template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::Compare(const ImageView<T> &aSrc2, CompareOp aCompare, ImageView<Pixel8uC1> &aDst)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = Pixel8uC1;
    constexpr size_t TupelSize = 1;

    switch (aCompare)
    {
        case opp::CompareOp::Less:
        {
            if constexpr (ComplexVector<SrcT>)
            {
                throw INVALIDARGUMENT(aCompare,
                                      "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
            }
            else
            {
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Lt<ComputeT>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const opp::Lt<ComputeT> op;
                const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case opp::CompareOp::LessEq:
        {
            if constexpr (ComplexVector<SrcT>)
            {
                throw INVALIDARGUMENT(aCompare,
                                      "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
            }
            else
            {
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Le<ComputeT>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const opp::Le<ComputeT> op;
                const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case opp::CompareOp::Eq:
        {
            using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Eq<ComputeT>, RoundingMode::None,
                                                voidType, voidType, true>;
            const opp::Eq<ComputeT> op;
            const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::CompareOp::Greater:
        {
            if constexpr (ComplexVector<SrcT>)
            {
                throw INVALIDARGUMENT(aCompare,
                                      "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
            }
            else
            {
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Gt<ComputeT>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const opp::Gt<ComputeT> op;
                const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case opp::CompareOp::GreaterEq:
        {
            if constexpr (ComplexVector<SrcT>)
            {
                throw INVALIDARGUMENT(aCompare,
                                      "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
            }
            else
            {
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Ge<ComputeT>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const opp::Ge<ComputeT> op;
                const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case opp::CompareOp::NEq:
        {
            using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::NEq<ComputeT>, RoundingMode::None,
                                                voidType, voidType, true>;
            const opp::NEq<ComputeT> op;
            const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aCompare, "Unknown CompareOp: " << aCompare);
    }
    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::Compare(const T &aConst, CompareOp aCompare, ImageView<Pixel8uC1> &aDst)
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = Pixel8uC1;
    constexpr size_t TupelSize = 1;

    switch (aCompare)
    {
        case opp::CompareOp::Less:
        {
            if constexpr (ComplexVector<SrcT>)
            {
                throw INVALIDARGUMENT(aCompare,
                                      "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
            }
            else
            {
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Lt<ComputeT>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const opp::Lt<ComputeT> op;
                const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case opp::CompareOp::LessEq:
        {
            if constexpr (ComplexVector<SrcT>)
            {
                throw INVALIDARGUMENT(aCompare,
                                      "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
            }
            else
            {
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Le<ComputeT>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const opp::Le<ComputeT> op;
                const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case opp::CompareOp::Eq:
        {
            using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Eq<ComputeT>,
                                                   RoundingMode::None, voidType, voidType, true>;
            const opp::Eq<ComputeT> op;
            const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::CompareOp::Greater:
        {
            if constexpr (ComplexVector<SrcT>)
            {
                throw INVALIDARGUMENT(aCompare,
                                      "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
            }
            else
            {
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Gt<ComputeT>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const opp::Gt<ComputeT> op;
                const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case opp::CompareOp::GreaterEq:
        {
            if constexpr (ComplexVector<SrcT>)
            {
                throw INVALIDARGUMENT(aCompare,
                                      "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
            }
            else
            {
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Ge<ComputeT>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const opp::Ge<ComputeT> op;
                const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case opp::CompareOp::NEq:
        {
            using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::NEq<ComputeT>,
                                                   RoundingMode::None, voidType, voidType, true>;
            const opp::NEq<ComputeT> op;
            const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
            forEachPixel(aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aCompare, "Unknown CompareOp: " << aCompare);
    }

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CompareEqEps(const ImageView<T> &aSrc2,
                                                 complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                                 ImageView<Pixel8uC1> &aDst)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = Pixel8uC1;
    constexpr size_t TupelSize = 1;

    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::EqEps<ComputeT>, RoundingMode::None,
                                        voidType, voidType, true>;
    const opp::EqEps<ComputeT> op(aEpsilon);
    const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CompareEqEps(const T &aConst, complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                                 ImageView<Pixel8uC1> &aDst)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = Pixel8uC1;
    constexpr size_t TupelSize = 1;

    using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::EqEps<ComputeT>, RoundingMode::None,
                                           voidType, voidType, true>;
    const opp::EqEps<ComputeT> op(aEpsilon);
    const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
    forEachPixel(aDst, functor);
    return aDst;
}
#pragma endregion
#pragma region Threshold
template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, CompareOp aCompare, ImageView<T> &aDst)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case opp::CompareOp::Less:
            return ThresholdLT(aThreshold, aDst);
            break;
        case opp::CompareOp::Greater:
            return ThresholdGT(aThreshold, aDst);
            break;
        default:
            throw INVALIDARGUMENT(
                aCompare,
                "CompareOp " << aCompare << " is not supported for Threshold, only Less and Greater are supported.");
            break;
    }
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold, ImageView<T> &aDst)
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Max<ComputeT>, RoundingMode::None>;
    const opp::Max<ComputeT> op;
    const thresholdSrcC functor(PointerRoi(), Pitch(), aThreshold, op);
    forEachPixel(aDst, functor);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, ImageView<T> &aDst)
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Min<ComputeT>, RoundingMode::None>;
    const opp::Min<ComputeT> op;
    const thresholdSrcC functor(PointerRoi(), Pitch(), aThreshold, op);
    forEachPixel(aDst, functor);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, CompareOp aCompare)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case opp::CompareOp::Less:
            return ThresholdLT(aThreshold);
            break;
        case opp::CompareOp::Greater:
            return ThresholdGT(aThreshold);
            break;
        default:
            throw INVALIDARGUMENT(
                aCompare,
                "CompareOp " << aCompare << " is not supported for Threshold, only Less and Greater are supported.");
            break;
    }
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold)
    requires RealVector<T>
{
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdInplaceC = InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Max<ComputeT>, RoundingMode::None>;
    const opp::Max<ComputeT> op;
    const thresholdInplaceC functor(aThreshold, op);
    forEachPixel(*this, functor);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold)
    requires RealVector<T>
{
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdInplaceC = InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Min<ComputeT>, RoundingMode::None>;
    const opp::Min<ComputeT> op;
    const thresholdInplaceC functor(aThreshold, op);
    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare, ImageView<T> &aDst)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case opp::CompareOp::Less:
            return ThresholdLT(aThreshold, aValue, aDst);
            break;
        case opp::CompareOp::Greater:
            return ThresholdGT(aThreshold, aValue, aDst);
            break;
        default:
            throw INVALIDARGUMENT(
                aCompare,
                "CompareOp " << aCompare << " is not supported for Threshold, only Less and Greater are supported.");
            break;
    }
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold, const T &aValue, ImageView<T> &aDst)
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::MaxVal<ComputeT>, RoundingMode::None>;
    const opp::MaxVal<ComputeT> op(aValue, aThreshold);
    const thresholdSrc functor(PointerRoi(), Pitch(), op);
    forEachPixel(aDst, functor);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, const T &aValue, ImageView<T> &aDst)
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::MinVal<ComputeT>, RoundingMode::None>;
    const opp::MinVal<ComputeT> op(aValue, aThreshold);
    const thresholdSrc functor(PointerRoi(), Pitch(), op);
    forEachPixel(aDst, functor);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare)
    requires RealVector<T>
{
    switch (aCompare)
    {
        case opp::CompareOp::Less:
            return ThresholdLT(aThreshold, aValue);
            break;
        case opp::CompareOp::Greater:
            return ThresholdGT(aThreshold, aValue);
            break;
        default:
            throw INVALIDARGUMENT(
                aCompare,
                "CompareOp " << aCompare << " is not supported for Threshold, only Less and Greater are supported.");
            break;
    }
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold, const T &aValue)
    requires RealVector<T>
{
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdInplace = InplaceFunctor<TupelSize, ComputeT, DstT, opp::MaxVal<ComputeT>, RoundingMode::None>;
    const opp::MaxVal<ComputeT> op(aValue, aThreshold);
    const thresholdInplace functor(op);
    forEachPixel(*this, functor);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, const T &aValue)
    requires RealVector<T>
{
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdInplace = InplaceFunctor<TupelSize, ComputeT, DstT, opp::MinVal<ComputeT>, RoundingMode::None>;
    const opp::MinVal<ComputeT> op(aValue, aThreshold);
    const thresholdInplace functor(op);
    forEachPixel(*this, functor);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT,
                                          const T &aValueGT, ImageView<T> &aDst)
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::MinValMaxVal<ComputeT>, RoundingMode::None>;
    const opp::MinValMaxVal<ComputeT> op(aValueGT, aThresholdGT, aValueLT, aThresholdLT);
    const thresholdSrc functor(PointerRoi(), Pitch(), op);
    forEachPixel(aDst, functor);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT,
                                          const T &aValueGT)
    requires RealVector<T>
{
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdInplace = InplaceFunctor<TupelSize, ComputeT, DstT, opp::MinValMaxVal<ComputeT>, RoundingMode::None>;
    const opp::MinValMaxVal<ComputeT> op(aValueGT, aThresholdGT, aValueLT, aThresholdLT);
    const thresholdInplace functor(op);
    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
} // namespace opp::image::cpuSimple