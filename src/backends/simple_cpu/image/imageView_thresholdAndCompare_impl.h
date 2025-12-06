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

#pragma region Compare
template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::Compare(const ImageView<T> &aSrc2, CompareOp aCompare,
                                            ImageView<Pixel8uC1> &aDst) const
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = Pixel8uC1;
    constexpr size_t TupelSize = 1;

    if (vector_active_size_v<SrcT> > 1 && CompareOp_IsPerChannel(aCompare))
    {
        throw INVALIDARGUMENT(
            aCompare,
            "CompareOp flag 'PerChannel' is not supported for multi channel images and single channel output.");
    }

    auto runOverAnyChannel = [&]<typename Tih>(Tih /*isAnyChannel*/) {
        constexpr bool anyChannel = Tih::value;

        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::Less:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Lt<ComputeT, anyChannel>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const mpp::Lt<ComputeT, anyChannel> op;
                    const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aDst, functor);
                }
            }
            break;
            case mpp::CompareOp::LessEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Le<ComputeT, anyChannel>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const mpp::Le<ComputeT, anyChannel> op;
                    const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aDst, functor);
                }
            }
            break;
            case mpp::CompareOp::Eq:
            {
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Eq<ComputeT, anyChannel>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const mpp::Eq<ComputeT, anyChannel> op;
                const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::CompareOp::Greater:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Gt<ComputeT, anyChannel>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const mpp::Gt<ComputeT, anyChannel> op;
                    const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aDst, functor);
                }
            }
            break;
            case mpp::CompareOp::GreaterEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Ge<ComputeT, anyChannel>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const mpp::Ge<ComputeT, anyChannel> op;
                    const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aDst, functor);
                }
            }
            break;
            case mpp::CompareOp::NEq:
            {
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::NEq<ComputeT, anyChannel>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const mpp::NEq<ComputeT, anyChannel> op;
                const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                    << aCompare << ". This function only supports binary comparisons.");
        }
    };

    if (CompareOp_IsAnyChannel(aCompare))
    {
        runOverAnyChannel(std::true_type{});
    }
    else
    {
        runOverAnyChannel(std::false_type{});
    }
    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::Compare(const T &aConst, CompareOp aCompare, ImageView<Pixel8uC1> &aDst) const
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = Pixel8uC1;
    constexpr size_t TupelSize = 1;

    if (vector_active_size_v<SrcT> > 1 && CompareOp_IsPerChannel(aCompare))
    {
        throw INVALIDARGUMENT(
            aCompare,
            "CompareOp flag 'PerChannel' is not supported for multi channel images and single channel output.");
    }

    auto runOverAnyChannel = [&]<typename Tih>(Tih /*isAnyChannel*/) {
        constexpr bool anyChannel = Tih::value;

        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::Less:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcC =
                        SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Lt<ComputeT, anyChannel>,
                                           RoundingMode::None, voidType, voidType, true>;
                    const mpp::Lt<ComputeT, anyChannel> op;
                    const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                    forEachPixel(aDst, functor);
                }
            }
            break;
            case mpp::CompareOp::LessEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcC =
                        SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Le<ComputeT, anyChannel>,
                                           RoundingMode::None, voidType, voidType, true>;
                    const mpp::Le<ComputeT, anyChannel> op;
                    const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                    forEachPixel(aDst, functor);
                }
            }
            break;
            case mpp::CompareOp::Eq:
            {
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Eq<ComputeT, anyChannel>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const mpp::Eq<ComputeT, anyChannel> op;
                const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::CompareOp::Greater:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcC =
                        SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Gt<ComputeT, anyChannel>,
                                           RoundingMode::None, voidType, voidType, true>;
                    const mpp::Gt<ComputeT, anyChannel> op;
                    const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                    forEachPixel(aDst, functor);
                }
            }
            break;
            case mpp::CompareOp::GreaterEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcC =
                        SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Ge<ComputeT, anyChannel>,
                                           RoundingMode::None, voidType, voidType, true>;
                    const mpp::Ge<ComputeT, anyChannel> op;
                    const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                    forEachPixel(aDst, functor);
                }
            }
            break;
            case mpp::CompareOp::NEq:
            {
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::NEq<ComputeT, anyChannel>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const mpp::NEq<ComputeT, anyChannel> op;
                const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                    << aCompare << ". This function only supports binary comparisons.");
        }
    };

    if (CompareOp_IsAnyChannel(aCompare))
    {
        runOverAnyChannel(std::true_type{});
    }
    else
    {
        runOverAnyChannel(std::false_type{});
    }

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::Compare(CompareOp aCompare, ImageView<Pixel8uC1> &aDst) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = Pixel8uC1;
    constexpr size_t TupelSize = 1;

    if (vector_active_size_v<SrcT> > 1 && CompareOp_IsPerChannel(aCompare))
    {
        throw INVALIDARGUMENT(
            aCompare,
            "CompareOp flag 'PerChannel' is not supported for multi channel images and single channel output.");
    }

    auto runOverAnyChannel = [&]<typename Tih>(Tih /*isAnyChannel*/) {
        constexpr bool anyChannel = Tih::value;

        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::IsFinite:
            {
                using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsFinite<ComputeT, anyChannel>,
                                              RoundingMode::None, voidType, voidType, true>;
                const mpp::IsFinite<ComputeT, anyChannel> op;
                const compareSrc functor(PointerRoi(), Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::CompareOp::IsNaN:
            {
                using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsNaN<ComputeT, anyChannel>,
                                              RoundingMode::None, voidType, voidType, true>;
                const mpp::IsNaN<ComputeT, anyChannel> op;
                const compareSrc functor(PointerRoi(), Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::CompareOp::IsInf:
            {
                using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsInf<ComputeT, anyChannel>,
                                              RoundingMode::None, voidType, voidType, true>;
                const mpp::IsInf<ComputeT, anyChannel> op;
                const compareSrc functor(PointerRoi(), Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::CompareOp::IsInfOrNaN:
            {
                using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsInfOrNaN<ComputeT, anyChannel>,
                                              RoundingMode::None, voidType, voidType, true>;
                const mpp::IsInfOrNaN<ComputeT, anyChannel> op;
                const compareSrc functor(PointerRoi(), Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::CompareOp::IsPositiveInf:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, use IsInf without sign instead.");
                }
                else
                {
                    using compareSrc =
                        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsPositiveInf<ComputeT, anyChannel>,
                                   RoundingMode::None, voidType, voidType, true>;
                    const mpp::IsPositiveInf<ComputeT, anyChannel> op;
                    const compareSrc functor(PointerRoi(), Pitch(), op);
                    forEachPixel(aDst, functor);
                }
            }
            break;
            case mpp::CompareOp::IsNegativeInf:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, use IsInf without sign instead.");
                }
                else
                {
                    using compareSrc =
                        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsNegativeInf<ComputeT, anyChannel>,
                                   RoundingMode::None, voidType, voidType, true>;
                    const mpp::IsNegativeInf<ComputeT, anyChannel> op;
                    const compareSrc functor(PointerRoi(), Pitch(), op);
                    forEachPixel(aDst, functor);
                }
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare,
                                      "Unsupported CompareOp: "
                                          << aCompare
                                          << ". This function only supports unary comparisons (IsInf, IsNaN, etc.).");
        }
    };

    if (CompareOp_IsAnyChannel(aCompare))
    {
        runOverAnyChannel(std::true_type{});
    }
    else
    {
        runOverAnyChannel(std::false_type{});
    }

    return aDst;
}
template <PixelType T>
ImageView<same_vector_size_different_type_t<T, byte>> &ImageView<T>::Compare(
    const ImageView<T> &aSrc2, CompareOp aCompare, ImageView<same_vector_size_different_type_t<T, byte>> &aDst) const
    requires(vector_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = same_vector_size_different_type_t<T, byte>;
    constexpr size_t TupelSize = 1;

    if (!CompareOp_IsPerChannel(aCompare))
    {
        throw INVALIDARGUMENT(aCompare, "CompareOp flag 'PerChannel' must be set for multi channel output.");
    }

    switch (CompareOp_NoFlags(aCompare))
    {
        case mpp::CompareOp::Less:
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
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareLt<ComputeT>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const mpp::CompareLt<ComputeT> op;
                const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case mpp::CompareOp::LessEq:
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
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareLe<ComputeT>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const mpp::CompareLe<ComputeT> op;
                const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case mpp::CompareOp::Eq:
        {
            using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareEq<ComputeT>,
                                                RoundingMode::None, voidType, voidType, true>;
            const mpp::CompareEq<ComputeT> op;
            const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::CompareOp::Greater:
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
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareGt<ComputeT>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const mpp::CompareGt<ComputeT> op;
                const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case mpp::CompareOp::GreaterEq:
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
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareGe<ComputeT>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const mpp::CompareGe<ComputeT> op;
                const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case mpp::CompareOp::NEq:
        {
            using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareNEq<ComputeT>,
                                                RoundingMode::None, voidType, voidType, true>;
            const mpp::CompareNEq<ComputeT> op;
            const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(
                aCompare, "Unsupported CompareOp: " << aCompare << ". This function only supports binary comparisons.");
    }
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, byte>> &ImageView<T>::Compare(
    const T &aConst, CompareOp aCompare, ImageView<same_vector_size_different_type_t<T, byte>> &aDst) const
    requires(vector_size_v<T> > 1)
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = same_vector_size_different_type_t<T, byte>;
    constexpr size_t TupelSize = 1;

    if (!CompareOp_IsPerChannel(aCompare))
    {
        throw INVALIDARGUMENT(aCompare, "CompareOp flag 'PerChannel' must be set for multi channel output.");
    }

    switch (CompareOp_NoFlags(aCompare))
    {
        case mpp::CompareOp::Less:
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
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareLt<ComputeT>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const mpp::CompareLt<ComputeT> op;
                const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case mpp::CompareOp::LessEq:
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
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareLe<ComputeT>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const mpp::CompareLe<ComputeT> op;
                const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case mpp::CompareOp::Eq:
        {
            using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareEq<ComputeT>,
                                                   RoundingMode::None, voidType, voidType, true>;
            const mpp::CompareEq<ComputeT> op;
            const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::CompareOp::Greater:
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
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareGt<ComputeT>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const mpp::CompareGt<ComputeT> op;
                const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case mpp::CompareOp::GreaterEq:
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
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareGe<ComputeT>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const mpp::CompareGe<ComputeT> op;
                const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case mpp::CompareOp::NEq:
        {
            using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareNEq<ComputeT>,
                                                   RoundingMode::None, voidType, voidType, true>;
            const mpp::CompareNEq<ComputeT> op;
            const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
            forEachPixel(aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(
                aCompare, "Unsupported CompareOp: " << aCompare << ". This function only supports binary comparisons.");
    }

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, byte>> &ImageView<T>::Compare(
    CompareOp aCompare, ImageView<same_vector_size_different_type_t<T, byte>> &aDst) const
    requires RealOrComplexFloatingVector<T> && (vector_size_v<T> > 1)
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = same_vector_size_different_type_t<T, byte>;
    constexpr size_t TupelSize = 1;

    if (!CompareOp_IsPerChannel(aCompare))
    {
        throw INVALIDARGUMENT(aCompare, "CompareOp flag 'PerChannel' must be set for multi channel output.");
    }

    switch (CompareOp_NoFlags(aCompare))
    {
        case mpp::CompareOp::IsFinite:
        {
            using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsFinite<ComputeT, false>,
                                          RoundingMode::None, voidType, voidType, true>;
            const mpp::IsFinite<ComputeT, false> op;
            const compareSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::CompareOp::IsNaN:
        {
            using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsNaN<ComputeT, false>,
                                          RoundingMode::None, voidType, voidType, true>;
            const mpp::IsNaN<ComputeT, false> op;
            const compareSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::CompareOp::IsInf:
        {
            using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsInf<ComputeT, false>,
                                          RoundingMode::None, voidType, voidType, true>;
            const mpp::IsInf<ComputeT, false> op;
            const compareSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::CompareOp::IsInfOrNaN:
        {
            using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsInfOrNaN<ComputeT, false>,
                                          RoundingMode::None, voidType, voidType, true>;
            const mpp::IsInfOrNaN<ComputeT, false> op;
            const compareSrc functor(PointerRoi(), Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::CompareOp::IsPositiveInf:
        {
            if constexpr (ComplexVector<SrcT>)
            {
                throw INVALIDARGUMENT(
                    aCompare, "CompareOp "
                                  << aCompare
                                  << " is not supported for complex datatypes, use IsInf without sign instead.");
            }
            else
            {
                using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsPositiveInf<ComputeT, false>,
                                              RoundingMode::None, voidType, voidType, true>;
                const mpp::IsPositiveInf<ComputeT, false> op;
                const compareSrc functor(PointerRoi(), Pitch(), op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        case mpp::CompareOp::IsNegativeInf:
        {
            if constexpr (ComplexVector<SrcT>)
            {
                throw INVALIDARGUMENT(
                    aCompare, "CompareOp "
                                  << aCompare
                                  << " is not supported for complex datatypes, use IsInf without sign instead.");
            }
            else
            {
                using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsNegativeInf<ComputeT, false>,
                                              RoundingMode::None, voidType, voidType, true>;
                const mpp::IsNegativeInf<ComputeT, false> op;
                const compareSrc functor(PointerRoi(), Pitch(), op);
                forEachPixel(aDst, functor);
            }
        }
        break;
        default:
            throw INVALIDARGUMENT(
                aCompare, "Unsupported CompareOp: "
                              << aCompare << ". This function only supports unary comparisons (IsInf, IsNaN, etc.).");
    }

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CompareEqEps(const ImageView<T> &aSrc2,
                                                 complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                                 ImageView<Pixel8uC1> &aDst) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = Pixel8uC1;
    constexpr size_t TupelSize = 1;

    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::EqEps<ComputeT>, RoundingMode::None,
                                        voidType, voidType, true>;
    const mpp::EqEps<ComputeT> op(aEpsilon);
    const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CompareEqEps(const T &aConst, complex_basetype_t<remove_vector_t<T>> aEpsilon,
                                                 ImageView<Pixel8uC1> &aDst) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = Pixel8uC1;
    constexpr size_t TupelSize = 1;

    using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::EqEps<ComputeT>, RoundingMode::None,
                                           voidType, voidType, true>;
    const mpp::EqEps<ComputeT> op(aEpsilon);
    const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
    forEachPixel(aDst, functor);
    return aDst;
}
#pragma endregion
#pragma region Threshold
template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, CompareOp aCompare, ImageView<T> &aDst) const
    requires RealVector<T>
{
    switch (aCompare)
    {
        case mpp::CompareOp::Less:
            return ThresholdLT(aThreshold, aDst);
            break;
        case mpp::CompareOp::Greater:
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
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold, ImageView<T> &aDst) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Max<ComputeT>, RoundingMode::None>;
    const mpp::Max<ComputeT> op;
    const thresholdSrcC functor(PointerRoi(), Pitch(), aThreshold, op);
    forEachPixel(aDst, functor);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, ImageView<T> &aDst) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Min<ComputeT>, RoundingMode::None>;
    const mpp::Min<ComputeT> op;
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
        case mpp::CompareOp::Less:
            return ThresholdLT(aThreshold);
            break;
        case mpp::CompareOp::Greater:
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

    using thresholdInplaceC = InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Max<ComputeT>, RoundingMode::None>;
    const mpp::Max<ComputeT> op;
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

    using thresholdInplaceC = InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Min<ComputeT>, RoundingMode::None>;
    const mpp::Min<ComputeT> op;
    const thresholdInplaceC functor(aThreshold, op);
    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Threshold(const T &aThreshold, const T &aValue, CompareOp aCompare,
                                      ImageView<T> &aDst) const
    requires RealVector<T>
{
    switch (aCompare)
    {
        case mpp::CompareOp::Less:
            return ThresholdLT(aThreshold, aValue, aDst);
            break;
        case mpp::CompareOp::Greater:
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
ImageView<T> &ImageView<T>::ThresholdLT(const T &aThreshold, const T &aValue, ImageView<T> &aDst) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::MaxVal<ComputeT>, RoundingMode::None>;
    const mpp::MaxVal<ComputeT> op(aValue, aThreshold);
    const thresholdSrc functor(PointerRoi(), Pitch(), op);
    forEachPixel(aDst, functor);
    return aDst;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdGT(const T &aThreshold, const T &aValue, ImageView<T> &aDst) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::MinVal<ComputeT>, RoundingMode::None>;
    const mpp::MinVal<ComputeT> op(aValue, aThreshold);
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
        case mpp::CompareOp::Less:
            return ThresholdLT(aThreshold, aValue);
            break;
        case mpp::CompareOp::Greater:
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

    using thresholdInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::MaxVal<ComputeT>, RoundingMode::None>;
    const mpp::MaxVal<ComputeT> op(aValue, aThreshold);
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

    using thresholdInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::MinVal<ComputeT>, RoundingMode::None>;
    const mpp::MinVal<ComputeT> op(aValue, aThreshold);
    const thresholdInplace functor(op);
    forEachPixel(*this, functor);
    return *this;
}
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdLTGT(const T &aThresholdLT, const T &aValueLT, const T &aThresholdGT,
                                          const T &aValueGT, ImageView<T> &aDst) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using thresholdSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::MinValMaxVal<ComputeT>, RoundingMode::None>;
    const mpp::MinValMaxVal<ComputeT> op(aValueGT, aThresholdGT, aValueLT, aThresholdLT);
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

    using thresholdInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::MinValMaxVal<ComputeT>, RoundingMode::None>;
    const mpp::MinValMaxVal<ComputeT> op(aValueGT, aThresholdGT, aValueLT, aThresholdLT);
    const thresholdInplace functor(op);
    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region ReplaceIf

template <typename SrcT, typename ComperatorT, typename CompareT> struct replaceIfInstantiationHelper
{
    using src_t        = SrcT;
    using comperator_t = ComperatorT;
    using compare_t    = CompareT;
};

template <typename CompareT, bool IsAnyChannel> struct replaceIfInstantiationHelper2
{
    using compare_t                      = CompareT;
    static constexpr bool is_any_channel = IsAnyChannel;
};

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(const ImageView<T> &aSrc2, CompareOp aCompare, const T &aValue,
                                      ImageView<T> &aDst) const
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = T;

    constexpr size_t TupelSize = 1;

    if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
    {
        throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                     << " is not supported: Flags CompareOp::AnyChannel and "
                                                        "CompareOp::PerChannel cannot be set at the same time.");
    }

    auto runComperator = [&]<typename Tih>(Tih /*instantiationHelper*/) {
        using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                                            mpp::ReplaceIf<SrcT, typename Tih::comperator_t, typename Tih::compare_t>,
                                            RoundingMode::None, voidType, voidType, true>;
        const mpp::ReplaceIf<SrcT, typename Tih::comperator_t, typename Tih::compare_t> op(aValue);

        const compareSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
        forEachPixel(aDst, functor);
    };

    auto runAnyChannel = [&]<typename Tih>(Tih /*instantiationHelper2*/) {
        constexpr bool anyChannel = Tih::is_any_channel;
        constexpr bool perChannel = vector_size_v<typename Tih::compare_t> > 1;

        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::Less:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareLt<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Lt<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::LessEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareLe<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Le<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::Eq:
            {
                if constexpr (perChannel)
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::CompareEq<ComputeT>, typename Tih::compare_t>{});
                }
                else
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::Eq<ComputeT, anyChannel>, typename Tih::compare_t>{});
                }
            }
            break;
            case mpp::CompareOp::Greater:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareGt<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Gt<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::GreaterEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareGe<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Ge<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::NEq:
            {
                if constexpr (perChannel)
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::CompareNEq<ComputeT>, typename Tih::compare_t>{});
                }
                else
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::NEq<ComputeT, anyChannel>, typename Tih::compare_t>{});
                }
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                    << aCompare << ". This function only supports binary comparisons.");
        }
    };

    if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
    {
        using CompareT = same_vector_size_different_type_t<SrcT, byte>;
        runAnyChannel(replaceIfInstantiationHelper2<CompareT, false>{});
    }
    else
    {
        using CompareT = Vector1<byte>;
        if (CompareOp_IsAnyChannel(aCompare))
        {
            runAnyChannel(replaceIfInstantiationHelper2<CompareT, true>{});
        }
        else
        {
            runAnyChannel(replaceIfInstantiationHelper2<CompareT, false>{});
        }
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(const T &aConst, CompareOp aCompare, const T &aValue, ImageView<T> &aDst) const
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = T;

    constexpr size_t TupelSize = 1;

    if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
    {
        throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                     << " is not supported: Flags CompareOp::AnyChannel and "
                                                        "CompareOp::PerChannel cannot be set at the same time.");
    }

    auto runComperator = [&]<typename Tih>(Tih /*instantiationHelper*/) {
        using compareSrcC =
            SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                               mpp::ReplaceIf<SrcT, typename Tih::comperator_t, typename Tih::compare_t>,
                               RoundingMode::None, voidType, voidType, true>;
        const mpp::ReplaceIf<SrcT, typename Tih::comperator_t, typename Tih::compare_t> op(aValue);

        const compareSrcC functor(PointerRoi(), Pitch(), aConst, op);
        forEachPixel(aDst, functor);
    };

    auto runAnyChannel = [&]<typename Tih>(Tih /*instantiationHelper2*/) {
        constexpr bool anyChannel = Tih::is_any_channel;
        constexpr bool perChannel = vector_size_v<typename Tih::compare_t> > 1;

        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::Less:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareLt<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Lt<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::LessEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareLe<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Le<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::Eq:
            {
                if constexpr (perChannel)
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::CompareEq<ComputeT>, typename Tih::compare_t>{});
                }
                else
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::Eq<ComputeT, anyChannel>, typename Tih::compare_t>{});
                }
            }
            break;
            case mpp::CompareOp::Greater:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareGt<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Gt<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::GreaterEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareGe<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Ge<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::NEq:
            {
                if constexpr (perChannel)
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::CompareNEq<ComputeT>, typename Tih::compare_t>{});
                }
                else
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::NEq<ComputeT, anyChannel>, typename Tih::compare_t>{});
                }
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                    << aCompare << ". This function only supports binary comparisons.");
        }
    };

    if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
    {
        using CompareT = same_vector_size_different_type_t<SrcT, byte>;
        runAnyChannel(replaceIfInstantiationHelper2<CompareT, false>{});
    }
    else
    {
        using CompareT = Vector1<byte>;
        if (CompareOp_IsAnyChannel(aCompare))
        {
            runAnyChannel(replaceIfInstantiationHelper2<CompareT, true>{});
        }
        else
        {
            runAnyChannel(replaceIfInstantiationHelper2<CompareT, false>{});
        }
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(CompareOp aCompare, const T &aValue, ImageView<T> &aDst) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = T;

    constexpr size_t TupelSize = 1;

    if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
    {
        throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                     << " is not supported: Flags CompareOp::AnyChannel and "
                                                        "CompareOp::PerChannel cannot be set at the same time.");
    }

    auto runComperator = [&]<typename Tih>(Tih /*instantiationHelper*/) {
        using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                                      mpp::ReplaceIfFP<SrcT, typename Tih::comperator_t, typename Tih::compare_t>,
                                      RoundingMode::None, voidType, voidType, true>;
        const mpp::ReplaceIfFP<SrcT, typename Tih::comperator_t, typename Tih::compare_t> op(aValue);

        const compareSrc functor(PointerRoi(), Pitch(), op);
        forEachPixel(aDst, functor);
    };

    auto runAnyChannel = [&]<typename Tih>(Tih /*instantiationHelper2*/) {
        constexpr bool anyChannel = Tih::is_any_channel;

        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::IsFinite:
            {
                runComperator(
                    replaceIfInstantiationHelper<SrcT, mpp::IsFinite<ComputeT, anyChannel>, typename Tih::compare_t>{});
            }
            break;
            case mpp::CompareOp::IsNaN:
            {
                runComperator(
                    replaceIfInstantiationHelper<SrcT, mpp::IsNaN<ComputeT, anyChannel>, typename Tih::compare_t>{});
            }
            break;
            case mpp::CompareOp::IsInf:
            {
                runComperator(
                    replaceIfInstantiationHelper<SrcT, mpp::IsInf<ComputeT, anyChannel>, typename Tih::compare_t>{});
            }
            break;
            case mpp::CompareOp::IsInfOrNaN:
            {
                runComperator(replaceIfInstantiationHelper<SrcT, mpp::IsInfOrNaN<ComputeT, anyChannel>,
                                                           typename Tih::compare_t>{});
            }
            break;
            case mpp::CompareOp::IsPositiveInf:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, use IsInf without sign instead.");
                }
                else
                {
                    runComperator(replaceIfInstantiationHelper<SrcT, mpp::IsPositiveInf<ComputeT, anyChannel>,
                                                               typename Tih::compare_t>{});
                }
            }
            break;
            case mpp::CompareOp::IsNegativeInf:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, use IsInf without sign instead.");
                }
                else
                {
                    runComperator(replaceIfInstantiationHelper<SrcT, mpp::IsNegativeInf<ComputeT, anyChannel>,
                                                               typename Tih::compare_t>{});
                }
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare,
                                      "Unsupported CompareOp: "
                                          << aCompare
                                          << ". This function only supports unary comparisons (IsInf, IsNaN, etc.).");
        }
    };

    if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
    {
        using CompareT = same_vector_size_different_type_t<SrcT, byte>;
        runAnyChannel(replaceIfInstantiationHelper2<CompareT, false>{});
    }
    else
    {
        using CompareT = Vector1<byte>;
        if (CompareOp_IsAnyChannel(aCompare))
        {
            runAnyChannel(replaceIfInstantiationHelper2<CompareT, true>{});
        }
        else
        {
            runAnyChannel(replaceIfInstantiationHelper2<CompareT, false>{});
        }
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(const ImageView<T> &aSrc2, CompareOp aCompare, const T &aValue)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT     = T;
    using DstT     = T;
    using ComputeT = T;

    constexpr size_t TupelSize = 1;

    if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
    {
        throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                     << " is not supported: Flags CompareOp::AnyChannel and "
                                                        "CompareOp::PerChannel cannot be set at the same time.");
    }

    auto runComperator = [&]<typename Tih>(Tih /*instantiationHelper*/) {
        using compareInplaceSrc =
            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::ReplaceIf<SrcT, typename Tih::comperator_t, typename Tih::compare_t>,
                              RoundingMode::None, voidType, voidType>;
        const mpp::ReplaceIf<SrcT, typename Tih::comperator_t, typename Tih::compare_t> op(aValue);

        const compareInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
        forEachPixel(*this, functor);
    };

    auto runAnyChannel = [&]<typename Tih>(Tih /*instantiationHelper2*/) {
        constexpr bool anyChannel = Tih::is_any_channel;
        constexpr bool perChannel = vector_size_v<typename Tih::compare_t> > 1;

        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::Less:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareLt<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Lt<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::LessEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareLe<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Le<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::Eq:
            {
                if constexpr (perChannel)
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::CompareEq<ComputeT>, typename Tih::compare_t>{});
                }
                else
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::Eq<ComputeT, anyChannel>, typename Tih::compare_t>{});
                }
            }
            break;
            case mpp::CompareOp::Greater:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareGt<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Gt<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::GreaterEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareGe<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Ge<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::NEq:
            {
                if constexpr (perChannel)
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::CompareNEq<ComputeT>, typename Tih::compare_t>{});
                }
                else
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::NEq<ComputeT, anyChannel>, typename Tih::compare_t>{});
                }
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                    << aCompare << ". This function only supports binary comparisons.");
        }
    };

    if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
    {
        using CompareT = same_vector_size_different_type_t<SrcT, byte>;
        runAnyChannel(replaceIfInstantiationHelper2<CompareT, false>{});
    }
    else
    {
        using CompareT = Vector1<byte>;
        if (CompareOp_IsAnyChannel(aCompare))
        {
            runAnyChannel(replaceIfInstantiationHelper2<CompareT, true>{});
        }
        else
        {
            runAnyChannel(replaceIfInstantiationHelper2<CompareT, false>{});
        }
    }
    return *this;
}

template <PixelType T> ImageView<T> &ImageView<T>::ReplaceIf(const T &aConst, CompareOp aCompare, const T &aValue)
{
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = T;

    constexpr size_t TupelSize = 1;

    if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
    {
        throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                     << " is not supported: Flags CompareOp::AnyChannel and "
                                                        "CompareOp::PerChannel cannot be set at the same time.");
    }

    auto runComperator = [&]<typename Tih>(Tih /*instantiationHelper*/) {
        using compareInplaceC =
            InplaceConstantFunctor<TupelSize, ComputeT, DstT,
                                   mpp::ReplaceIf<SrcT, typename Tih::comperator_t, typename Tih::compare_t>,
                                   RoundingMode::None, voidType, voidType>;
        const mpp::ReplaceIf<SrcT, typename Tih::comperator_t, typename Tih::compare_t> op(aValue);

        const compareInplaceC functor(aConst, op);
        forEachPixel(*this, functor);
    };

    auto runAnyChannel = [&]<typename Tih>(Tih /*instantiationHelper2*/) {
        constexpr bool anyChannel = Tih::is_any_channel;
        constexpr bool perChannel = vector_size_v<typename Tih::compare_t> > 1;

        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::Less:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareLt<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Lt<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::LessEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareLe<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Le<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::Eq:
            {
                if constexpr (perChannel)
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::CompareEq<ComputeT>, typename Tih::compare_t>{});
                }
                else
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::Eq<ComputeT, anyChannel>, typename Tih::compare_t>{});
                }
            }
            break;
            case mpp::CompareOp::Greater:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareGt<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Gt<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::GreaterEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    if constexpr (perChannel)
                    {
                        runComperator(
                            replaceIfInstantiationHelper<SrcT, mpp::CompareGe<ComputeT>, typename Tih::compare_t>{});
                    }
                    else
                    {
                        runComperator(replaceIfInstantiationHelper<SrcT, mpp::Ge<ComputeT, anyChannel>,
                                                                   typename Tih::compare_t>{});
                    }
                }
            }
            break;
            case mpp::CompareOp::NEq:
            {
                if constexpr (perChannel)
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::CompareNEq<ComputeT>, typename Tih::compare_t>{});
                }
                else
                {
                    runComperator(
                        replaceIfInstantiationHelper<SrcT, mpp::NEq<ComputeT, anyChannel>, typename Tih::compare_t>{});
                }
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                    << aCompare << ". This function only supports binary comparisons.");
        }
    };

    if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
    {
        using CompareT = same_vector_size_different_type_t<SrcT, byte>;
        runAnyChannel(replaceIfInstantiationHelper2<CompareT, false>{});
    }
    else
    {
        using CompareT = Vector1<byte>;
        if (CompareOp_IsAnyChannel(aCompare))
        {
            runAnyChannel(replaceIfInstantiationHelper2<CompareT, true>{});
        }
        else
        {
            runAnyChannel(replaceIfInstantiationHelper2<CompareT, false>{});
        }
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ReplaceIf(CompareOp aCompare, const T &aValue)
    requires RealOrComplexFloatingVector<T>
{
    using SrcT     = T;
    using DstT     = T;
    using ComputeT = T;

    constexpr size_t TupelSize = 1;

    if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
    {
        throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                     << " is not supported: Flags CompareOp::AnyChannel and "
                                                        "CompareOp::PerChannel cannot be set at the same time.");
    }

    auto runComperator = [&]<typename Tih>(Tih /*instantiationHelper*/) {
        using compareInplace =
            InplaceFunctor<TupelSize, ComputeT, DstT,
                           mpp::ReplaceIfFP<SrcT, typename Tih::comperator_t, typename Tih::compare_t>,
                           RoundingMode::None, voidType, voidType>;
        const mpp::ReplaceIfFP<SrcT, typename Tih::comperator_t, typename Tih::compare_t> op(aValue);

        const compareInplace functor(op);
        forEachPixel(*this, functor);
    };

    auto runAnyChannel = [&]<typename Tih>(Tih /*instantiationHelper2*/) {
        constexpr bool anyChannel = Tih::is_any_channel;

        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::IsFinite:
            {
                runComperator(
                    replaceIfInstantiationHelper<SrcT, mpp::IsFinite<ComputeT, anyChannel>, typename Tih::compare_t>{});
            }
            break;
            case mpp::CompareOp::IsNaN:
            {
                runComperator(
                    replaceIfInstantiationHelper<SrcT, mpp::IsNaN<ComputeT, anyChannel>, typename Tih::compare_t>{});
            }
            break;
            case mpp::CompareOp::IsInf:
            {
                runComperator(
                    replaceIfInstantiationHelper<SrcT, mpp::IsInf<ComputeT, anyChannel>, typename Tih::compare_t>{});
            }
            break;
            case mpp::CompareOp::IsInfOrNaN:
            {
                runComperator(replaceIfInstantiationHelper<SrcT, mpp::IsInfOrNaN<ComputeT, anyChannel>,
                                                           typename Tih::compare_t>{});
            }
            break;
            case mpp::CompareOp::IsPositiveInf:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, use IsInf without sign instead.");
                }
                else
                {
                    runComperator(replaceIfInstantiationHelper<SrcT, mpp::IsPositiveInf<ComputeT, anyChannel>,
                                                               typename Tih::compare_t>{});
                }
            }
            break;
            case mpp::CompareOp::IsNegativeInf:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, use IsInf without sign instead.");
                }
                else
                {
                    runComperator(replaceIfInstantiationHelper<SrcT, mpp::IsNegativeInf<ComputeT, anyChannel>,
                                                               typename Tih::compare_t>{});
                }
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare,
                                      "Unsupported CompareOp: "
                                          << aCompare
                                          << ". This function only supports unary comparisons (IsInf, IsNaN, etc.).");
        }
    };

    if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
    {
        using CompareT = same_vector_size_different_type_t<SrcT, byte>;
        runAnyChannel(replaceIfInstantiationHelper2<CompareT, false>{});
    }
    else
    {
        using CompareT = Vector1<byte>;
        if (CompareOp_IsAnyChannel(aCompare))
        {
            runAnyChannel(replaceIfInstantiationHelper2<CompareT, true>{});
        }
        else
        {
            runAnyChannel(replaceIfInstantiationHelper2<CompareT, false>{});
        }
    }
    return *this;
}
#pragma endregion
} // namespace mpp::image::cpuSimple