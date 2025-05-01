#pragma once
#include <backends/simple_cpu/image/forEachPixel.h>
#include <backends/simple_cpu/image/forEachPixelMasked.h>
#include <backends/simple_cpu/image/forEachPixelPlanar.h>
#include <backends/simple_cpu/image/forEachPixelSingleChannel.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/dataExchangeAndInit/operators.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/affineTransformation.h>
#include <common/image/border.h>
#include <common/image/channel.h>
#include <common/image/channelList.h>
#include <common/image/functors/borderControl.h>
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
#include <common/image/functors/interpolator.h>
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
#include <common/image/functors/transformer.h>
#include <common/image/functors/transformerFunctor.h>
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
#include <common/vector1.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace opp::image::cpuSimple
{

/// <summary>
/// WarpAffine.
/// </summary>
template <PixelType T>
template <RealFloatingPoint CoordT>
ImageView<T> &ImageView<T>::WarpAffine(ImageView<T> &aDst, const AffineTransformation<CoordT> &aAffine,
                                       InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->WarpAffine(aDst, aAffine, aInterpolation, aBorder, {0}, aAllowedReadRoi);
}
/// <summary>
/// WarpAffine.
/// </summary>
template <PixelType T>
template <RealFloatingPoint CoordT>
ImageView<T> &ImageView<T>::WarpAffine(ImageView<T> &aDst, const AffineTransformation<CoordT> &aAffine,
                                       InterpolationMode aInterpolation, BorderType aBorder, T aConstant,
                                       Roi aAllowedReadRoi) const
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    // aAffine maps from src to dst, but we compute from dst to src --> inverse
    const TransformerAffine<CoordT> affine(aAffine.Inverse());

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;

    using CoordTInterpol = coordinate_type_interpolation_for_t<T>;

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case opp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<T, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT, InterpolatorT, TransformerAffine<CoordT>, roundingMode>
                    functor(aBC, interpol, affine, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<default_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT, InterpolatorT, TransformerAffine<CoordT>, roundingMode>
                    functor(aBC, interpol, affine, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<default_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT, InterpolatorT, TransformerAffine<CoordT>, roundingMode>
                    functor(aBC, interpol, affine, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT =
                    Interpolator<default_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT, InterpolatorT, TransformerAffine<CoordT>, roundingMode>
                    functor(aBC, interpol, affine, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<default_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT, InterpolatorT, TransformerAffine<CoordT>, roundingMode>
                    functor(aBC, interpol, affine, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<default_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT, InterpolatorT, TransformerAffine<CoordT>, roundingMode>
                    functor(aBC, interpol, affine, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<default_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT, InterpolatorT, TransformerAffine<CoordT>, roundingMode>
                    functor(aBC, interpol, affine, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT =
                    Interpolator<default_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT, InterpolatorT, TransformerAffine<CoordT>, roundingMode>
                    functor(aBC, interpol, affine, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT =
                    Interpolator<default_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT, InterpolatorT, TransformerAffine<CoordT>, roundingMode>
                    functor(aBC, interpol, affine, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for WarpAffine.");
                break;
        }
    };

    switch (aBorder)
    {
        case opp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<T, BorderType::Replicate, true, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<T, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<T, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<T, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<T, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<T, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for WarpAffine.");
            break;
    }

    return aDst;
}

} // namespace opp::image::cpuSimple