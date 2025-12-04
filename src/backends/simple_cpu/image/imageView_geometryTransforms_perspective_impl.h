#pragma once
#include <backends/simple_cpu/image/forEachPixel.h>
#include <backends/simple_cpu/image/forEachPixelPlanar.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/border.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/functors/inplaceTransformerFunctor.h>
#include <common/image/functors/interpolator.h>
#include <common/image/functors/transformer.h>
#include <common/image/functors/transformerFunctor.h>
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
#pragma region Perspective
template <PixelType T>
ImageView<T> &ImageView<T>::WarpPerspective(ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective,
                                            InterpolationMode aInterpolation, BorderType aBorder,
                                            Roi aAllowedReadRoi) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->WarpPerspectiveBack(aDst, aPerspective.Inverse(), aInterpolation, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpPerspective(ImageView<T> &aDst, const PerspectiveTransformation<double> &aPerspective,
                                            InterpolationMode aInterpolation, T aConstant, BorderType aBorder,
                                            Roi aAllowedReadRoi) const
{
    return this->WarpPerspectiveBack(aDst, aPerspective.Inverse(), aInterpolation, aConstant, aBorder, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                   const PerspectiveTransformation<double> &aPerspective,
                                   InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aDst1, aDst2, aPerspective.Inverse(), aInterpolation, {0}, aBorder,
                                      aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                   const PerspectiveTransformation<double> &aPerspective,
                                   InterpolationMode aInterpolation, T aConstant, BorderType aBorder,
                                   Roi aAllowedReadRoi)
    requires TwoChannel<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aDst1, aDst2, aPerspective.Inverse(), aInterpolation, aConstant,
                                      aBorder, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                                   const PerspectiveTransformation<double> &aPerspective,
                                   InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aPerspective.Inverse(), aInterpolation,
                                      {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, T aConstant,
    BorderType aBorder, Roi aAllowedReadRoi)
    requires ThreeChannel<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aPerspective.Inverse(), aInterpolation,
                                      aConstant, aBorder, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
    Roi aAllowedReadRoi)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aPerspective.Inverse(),
                                      aInterpolation, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::WarpPerspective(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, T aConstant,
    BorderType aBorder, Roi aAllowedReadRoi)
    requires FourChannelNoAlpha<T>
{
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aPerspective.Inverse(),
                                      aInterpolation, aConstant, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpPerspectiveBack(ImageView<T> &aDst,
                                                const PerspectiveTransformation<double> &aPerspective,
                                                InterpolationMode aInterpolation, BorderType aBorder,
                                                Roi aAllowedReadRoi) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->WarpPerspectiveBack(aDst, aPerspective, aInterpolation, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WarpPerspectiveBack(ImageView<T> &aDst,
                                                const PerspectiveTransformation<double> &aPerspective,
                                                InterpolationMode aInterpolation, T aConstant, BorderType aBorder,
                                                Roi aAllowedReadRoi) const
{
    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
    using CoordTInterpol                = coordinate_type_interpolation_for_t<T>;
    using CoordT                        = coordinate_type_interpolation_for_t<T>; // double for Vector<double>...

    const PerspectiveTransformation<CoordT> transformTyped(aPerspective);
    const TransformerPerspective<CoordT> perspective(transformTyped);

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case mpp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<T, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, perspective, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case mpp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, perspective, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case mpp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, perspective, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case mpp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, perspective, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, perspective, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, perspective, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, perspective, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, perspective, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, perspective, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for WarpPerspective.");
                break;
        }
    };

    switch (aBorder)
    {
        case mpp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<T, BorderType::Replicate, true, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Constant:
        {
            using BCType = BorderControl<T, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Replicate:
        {
            using BCType = BorderControl<T, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Mirror:
        {
            using BCType = BorderControl<T, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<T, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Wrap:
        {
            using BCType = BorderControl<T, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::SmoothEdge:
        {
            using BCType = BorderControl<T, BorderType::SmoothEdge, true, false, false, false>;
            const BCType bc(allowedPtr, Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for WarpPerspective.");
            break;
    }

    return aDst;
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                       const PerspectiveTransformation<double> &aPerspective,
                                       InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aDst1, aDst2, aPerspective, aInterpolation, {0}, aBorder,
                                      aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                       const PerspectiveTransformation<double> &aPerspective,
                                       InterpolationMode aInterpolation, T aConstant, BorderType aBorder,
                                       Roi aAllowedReadRoi)
    requires TwoChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());

    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = aSrc1.ROI();
    }

    const Size2D minSizeAllocSrc = Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc());
    const Size2D minSizeAllocDst = Size2D::Min(aDst1.SizeAlloc(), aDst2.SizeAlloc());
    const Size2D minSizeAlloc    = Size2D::Min(minSizeAllocSrc, minSizeAllocDst);

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAlloc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
    using CoordTInterpol                = coordinate_type_interpolation_for_t<T>;
    using PixelT                        = T;
    using CoordT                        = coordinate_type_interpolation_for_t<T>; // double for Vector<double>...

    const PerspectiveTransformation<CoordT> transformTyped(aPerspective);
    const TransformerPerspective<CoordT> Perspective(transformTyped);

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case mpp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<PixelT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case mpp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case mpp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case mpp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for WarpPerspective.");
                break;
        }
    };

    switch (aBorder)
    {
        case mpp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<PixelT, BorderType::Replicate, true, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Constant:
        {
            using BCType = BorderControl<PixelT, BorderType::Constant, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset,
                            aConstant);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Replicate:
        {
            using BCType = BorderControl<PixelT, BorderType::Replicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Mirror:
        {
            using BCType = BorderControl<PixelT, BorderType::Mirror, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<PixelT, BorderType::MirrorReplicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Wrap:
        {
            using BCType = BorderControl<PixelT, BorderType::Wrap, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::SmoothEdge:
        {
            using BCType = BorderControl<PixelT, BorderType::SmoothEdge, true, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for WarpPerspective.");
            break;
    }
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                                       const PerspectiveTransformation<double> &aPerspective,
                                       InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aPerspective, aInterpolation, {0},
                                      aBorder, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, T aConstant,
    BorderType aBorder, Roi aAllowedReadRoi)
    requires ThreeChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI() || aSrc1.ROI() != aSrc3.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2 aSrc3,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.ROI(), aDst3.ROI());

    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = aSrc1.ROI();
    }

    const Size2D minSizeAllocSrc = Size2D::Min(Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc()), aSrc3.SizeAlloc());
    const Size2D minSizeAllocDst = Size2D::Min(Size2D::Min(aDst1.SizeAlloc(), aDst2.SizeAlloc()), aDst3.SizeAlloc());
    const Size2D minSizeAlloc    = Size2D::Min(minSizeAllocSrc, minSizeAllocDst);

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAlloc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr3 =
        gotoPtr(aSrc3.Pointer(), aSrc3.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
    using CoordTInterpol                = coordinate_type_interpolation_for_t<T>;
    using PixelT                        = T;
    using CoordT                        = coordinate_type_interpolation_for_t<T>; // double for Vector<double>...

    const PerspectiveTransformation<CoordT> transformTyped(aPerspective);
    const TransformerPerspective<CoordT> Perspective(transformTyped);

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case mpp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<PixelT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case mpp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case mpp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case mpp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for WarpPerspective.");
                break;
        }
    };

    switch (aBorder)
    {
        case mpp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<PixelT, BorderType::Replicate, true, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Constant:
        {
            using BCType = BorderControl<PixelT, BorderType::Constant, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset, aConstant);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Replicate:
        {
            using BCType = BorderControl<PixelT, BorderType::Replicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Mirror:
        {
            using BCType = BorderControl<PixelT, BorderType::Mirror, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<PixelT, BorderType::MirrorReplicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Wrap:
        {
            using BCType = BorderControl<PixelT, BorderType::Wrap, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::SmoothEdge:
        {
            using BCType = BorderControl<PixelT, BorderType::SmoothEdge, true, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for WarpPerspective.");
            break;
    }
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, BorderType aBorder,
    Roi aAllowedReadRoi)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::WarpPerspectiveBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aPerspective,
                                      aInterpolation, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::WarpPerspectiveBack(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1, const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, const ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
    ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
    ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
    const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation, T aConstant,
    BorderType aBorder, Roi aAllowedReadRoi)
    requires FourChannelNoAlpha<T>
{
    if (aSrc1.ROI() != aSrc2.ROI() || aSrc1.ROI() != aSrc3.ROI() || aSrc1.ROI() != aSrc4.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2 aSrc3 aSrc4,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.ROI(), aDst3.ROI());
    checkSameSize(aDst1.ROI(), aDst4.ROI());

    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = aSrc1.ROI();
    }

    const Size2D minSizeAllocSrc = Size2D::Min(
        Size2D::Min(Size2D::Min(aSrc1.SizeAlloc(), aSrc2.SizeAlloc()), aSrc3.SizeAlloc()), aSrc4.SizeAlloc());
    const Size2D minSizeAllocDst = Size2D::Min(
        Size2D::Min(Size2D::Min(aDst1.SizeAlloc(), aDst2.SizeAlloc()), aDst3.SizeAlloc()), aDst4.SizeAlloc());
    const Size2D minSizeAlloc = Size2D::Min(minSizeAllocSrc, minSizeAllocDst);

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, minSizeAlloc));

    const Vector2<int> roiOffset = aSrc1.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const Vector1<remove_vector_t<T>> *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr3 =
        gotoPtr(aSrc3.Pointer(), aSrc3.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());
    const Vector1<remove_vector_t<T>> *allowedPtr4 =
        gotoPtr(aSrc4.Pointer(), aSrc4.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
    using CoordTInterpol                = coordinate_type_interpolation_for_t<T>;
    using PixelT                        = T;
    using CoordT                        = coordinate_type_interpolation_for_t<T>; // double for Vector<double>...

    const PerspectiveTransformation<CoordT> transformTyped(aPerspective);
    const TransformerPerspective<CoordT> Perspective(transformTyped);

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case mpp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<PixelT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case mpp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case mpp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case mpp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<PixelT>, bcT, CoordTInterpol,
                                                   InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, PixelT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerPerspective<CoordT>, roundingMode>
                    functor(interpol, Perspective, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for WarpPerspective.");
                break;
        }
    };

    switch (aBorder)
    {
        case mpp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<PixelT, BorderType::Replicate, true, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Constant:
        {
            using BCType = BorderControl<PixelT, BorderType::Constant, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Replicate:
        {
            using BCType = BorderControl<PixelT, BorderType::Replicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Mirror:
        {
            using BCType = BorderControl<PixelT, BorderType::Mirror, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<PixelT, BorderType::MirrorReplicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Wrap:
        {
            using BCType = BorderControl<PixelT, BorderType::Wrap, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::SmoothEdge:
        {
            using BCType = BorderControl<PixelT, BorderType::SmoothEdge, true, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for WarpPerspective.");
            break;
    }
}

#pragma endregion

} // namespace mpp::image::cpuSimple