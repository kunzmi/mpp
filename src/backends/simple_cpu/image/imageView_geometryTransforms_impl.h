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
#include <common/image/functors/inplaceTransformerFunctor.h>
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
#include <common/vector1.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace opp::image::cpuSimple
{
// Note: the sub-templated versions for Affine/Perspective/Rotate/Resize are in the splitted _impl.h files for
// compilation speed optimization...

#pragma region Resize
template <PixelType T> ImageView<T> &ImageView<T>::Resize(ImageView<T> &aDst, InterpolationMode aInterpolation) const
{
    const Vec2d scaleFactor = Vec2d(aDst.SizeRoi()) / Vec2d(SizeRoi());
    const Vec2d shift       = Vec2d(0);

    return this->Resize(aDst, scaleFactor, shift, aInterpolation, BorderType::None, {0}, ROI());
}

template <PixelType T>
void ImageView<T>::Resize(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          InterpolationMode aInterpolation)
    requires TwoChannel<T>
{
    const Vec2d scaleFactor = Vec2d(aDst1.SizeRoi()) / Vec2d(aSrc1.SizeRoi());
    const Vec2d shift       = Vec2d(0);

    ImageView<T>::Resize(aSrc1, aSrc2, aDst1, aDst2, scaleFactor, shift, aInterpolation, BorderType::None, {0},
                         aSrc1.ROI());
}

template <PixelType T>
void ImageView<T>::Resize(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                          InterpolationMode aInterpolation)
    requires ThreeChannel<T>
{
    const Vec2d scaleFactor = Vec2d(aDst1.SizeRoi()) / Vec2d(aSrc1.SizeRoi());
    const Vec2d shift       = Vec2d(0);

    ImageView<T>::Resize(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, scaleFactor, shift, aInterpolation, BorderType::None,
                         {0}, aSrc1.ROI());
}

template <PixelType T>
void ImageView<T>::Resize(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                          InterpolationMode aInterpolation)
    requires FourChannelNoAlpha<T>
{
    const Vec2d scaleFactor = Vec2d(aDst1.SizeRoi()) / Vec2d(aSrc1.SizeRoi());
    const Vec2d shift       = Vec2d(0);

    ImageView<T>::Resize(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, scaleFactor, shift, aInterpolation,
                         BorderType::None, {0}, aSrc1.ROI());
}

template <PixelType T> Vec2d ImageView<T>::ResizeGetNPPShift(ImageView<T> &aDst) const
{
    const Vec2d scaleFactor    = Vec2d(aDst.SizeRoi()) / Vec2d(SizeRoi());
    const Vec2d invScaleFactor = 1.0 / scaleFactor;
    Vec2d shift(0); // no shift if scaling == 1

    if (scaleFactor.x > 1) // upscaling
    {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        shift.x = (0.25 - (1.0 - invScaleFactor.x) / 2.0) * scaleFactor.x;
    }
    else if (scaleFactor.x < 1) // downscaling
    {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        shift.x = -((1.0 - invScaleFactor.x) / 2.0) * scaleFactor.x;
    }

    if (scaleFactor.y > 1) // upscaling
    {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        shift.y = (0.25 - (1.0 - invScaleFactor.y) / 2.0) * scaleFactor.y;
    }
    else if (scaleFactor.y < 1) // downscaling
    {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        shift.y = -((1.0 - invScaleFactor.y) / 2.0) * scaleFactor.y;
    }

    return shift;
}
#pragma endregion

#pragma region Mirror
template <PixelType T> ImageView<T> &ImageView<T>::Mirror(ImageView<T> &aDst, MirrorAxis aAxis) const
{
    checkSameSize(ROI(), aDst.ROI());

    using BCType = BorderControl<T, BorderType::None>;
    const BCType bc(PointerRoi(), Pitch(), SizeRoi(), {0});
    using InterpolatorT = Interpolator<T, BCType, int, InterpolationMode::Undefined>;
    const InterpolatorT interpol(bc);

    switch (aAxis)
    {
        case opp::MirrorAxis::Horizontal:
        {
            const TransformerMirror<int, MirrorAxis::Horizontal> mirror(SizeRoi());
            const TransformerFunctor<1, T, int, false, InterpolatorT, TransformerMirror<int, MirrorAxis::Horizontal>,
                                     RoundingMode::None>
                functor(interpol, mirror, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::MirrorAxis::Vertical:
        {
            const TransformerMirror<int, MirrorAxis::Vertical> mirror(SizeRoi());
            const TransformerFunctor<1, T, int, false, InterpolatorT, TransformerMirror<int, MirrorAxis::Vertical>,
                                     RoundingMode::None>
                functor(interpol, mirror, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::MirrorAxis::Both:
        {
            const TransformerMirror<int, MirrorAxis::Both> mirror(SizeRoi());
            const TransformerFunctor<1, T, int, false, InterpolatorT, TransformerMirror<int, MirrorAxis::Both>,
                                     RoundingMode::None>
                functor(interpol, mirror, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aAxis, aAxis << " is not a supported mirror axis for Mirror.");
            break;
    }
    return aDst;
}

template <PixelType T> ImageView<T> &ImageView<T>::Mirror(MirrorAxis aAxis)
{
    using BCType = BorderControl<T, BorderType::None>;
    const BCType bc(PointerRoi(), Pitch(), SizeRoi(), {0});

    Size2D workROI = SizeRoi();

    switch (aAxis)
    {
        case opp::MirrorAxis::Horizontal:
        {
            const TransformerMirror<int, MirrorAxis::Horizontal> mirror(SizeRoi());
            const InplaceTransformerFunctor<T, BCType, TransformerMirror<int, MirrorAxis::Horizontal>> functor(
                bc, mirror, SizeRoi());

            workROI.y /= 2; // for uneven sizes, the center will remain unchanged

            forEachPixel(*this, workROI, functor);
        }
        break;
        case opp::MirrorAxis::Vertical:
        {
            const TransformerMirror<int, MirrorAxis::Vertical> mirror(SizeRoi());
            const InplaceTransformerFunctor<T, BCType, TransformerMirror<int, MirrorAxis::Vertical>> functor(bc, mirror,
                                                                                                             SizeRoi());

            workROI.x /= 2; // for uneven sizes, the center will remain unchanged

            forEachPixel(*this, workROI, functor);
        }
        break;
        case opp::MirrorAxis::Both:
        {
            const TransformerMirror<int, MirrorAxis::Both> mirror(SizeRoi());
            const InplaceTransformerFunctor<T, BCType, TransformerMirror<int, MirrorAxis::Both>> functor(bc, mirror,
                                                                                                         SizeRoi());

            if (workROI.x % 2 == 1)
            {
                workROI.x += 1;
                workROI.x /= 2; // for uneven sizes, the center will change!
                forEachPixel<T, InplaceTransformerFunctor<T, BCType, TransformerMirror<int, MirrorAxis::Both>>, true>(
                    *this, workROI, functor);
            }
            else
            {
                workROI.x /= 2;
                forEachPixel<T, InplaceTransformerFunctor<T, BCType, TransformerMirror<int, MirrorAxis::Both>>, false>(
                    *this, workROI, functor);
            }
        }
        break;
        default:
            throw INVALIDARGUMENT(aAxis, aAxis << " is not a supported mirror axis for Mirror.");
            break;
    }
    return *this;
}
#pragma endregion

#pragma region Remap
template <PixelType T>
ImageView<T> &ImageView<T>::Remap(ImageView<T> &aDst, const ImageView<Pixel32fC2> &aCoordinateMap,
                                  InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Remap(aDst, aCoordinateMap, aInterpolation, aBorder, {0}, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::Remap(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         BorderType aBorder, Roi aAllowedReadRoi)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Remap(aSrc1, aSrc2, aDst1, aDst2, aCoordinateMap, aInterpolation, aBorder, {0}, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::Remap(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         BorderType aBorder, Roi aAllowedReadRoi)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aCoordinateMap, aInterpolation, aBorder, {0},
                        aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::Remap(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         BorderType aBorder, Roi aAllowedReadRoi)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aCoordinateMap, aInterpolation, aBorder,
                        {0}, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Remap(ImageView<T> &aDst, const ImageView<Pixel32fC2> &aCoordinateMap,
                                  InterpolationMode aInterpolation, BorderType aBorder, T aConstant,
                                  Roi aAllowedReadRoi) const
{
    checkSameSize(aDst.SizeRoi(), aCoordinateMap.SizeRoi());

    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
    using CoordTInterpol                = float; // even for double types?
    using CoordT                        = float; // even for double types?

    const TransformerMap<CoordT> transMap(aCoordinateMap.PointerRoi(), aCoordinateMap.Pitch());

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case opp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<T, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for Remap.");
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
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
            break;
    }

    return aDst;
}

template <PixelType T>
void ImageView<T>::Remap(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         BorderType aBorder, T aConstant, Roi aAllowedReadRoi)
    requires TwoChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMap.SizeRoi());

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
    using CoordTInterpol                = float; // even for double types?
    using CoordT                        = float; // even for double types?

    const TransformerMap<CoordT> transMap(aCoordinateMap.PointerRoi(), aCoordinateMap.Pitch());

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case opp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<T, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for Remap.");
                break;
        }
    };

    switch (aBorder)
    {
        case opp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<T, BorderType::Replicate, true, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<T, BorderType::Constant, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset,
                            aConstant);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<T, BorderType::Replicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<T, BorderType::Mirror, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<T, BorderType::MirrorReplicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<T, BorderType::Wrap, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
            break;
    }
}

template <PixelType T>
void ImageView<T>::Remap(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         BorderType aBorder, T aConstant, Roi aAllowedReadRoi)
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
    checkSameSize(aDst1.SizeRoi(), aCoordinateMap.SizeRoi());

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
    using CoordTInterpol                = float; // even for double types?
    using CoordT                        = float; // even for double types?

    const TransformerMap<CoordT> transMap(aCoordinateMap.PointerRoi(), aCoordinateMap.Pitch());

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case opp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<T, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for Remap.");
                break;
        }
    };

    switch (aBorder)
    {
        case opp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<T, BorderType::Replicate, true, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<T, BorderType::Constant, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset, aConstant);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<T, BorderType::Replicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<T, BorderType::Mirror, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<T, BorderType::MirrorReplicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<T, BorderType::Wrap, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
            break;
    }
}

template <PixelType T>
void ImageView<T>::Remap(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                         const ImageView<Pixel32fC2> &aCoordinateMap, InterpolationMode aInterpolation,
                         BorderType aBorder, T aConstant, Roi aAllowedReadRoi)
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
    checkSameSize(aDst1.SizeRoi(), aCoordinateMap.SizeRoi());

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
    using CoordTInterpol                = float; // even for double types?
    using CoordT                        = float; // even for double types?

    const TransformerMap<CoordT> transMap(aCoordinateMap.PointerRoi(), aCoordinateMap.Pitch());

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case opp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<T, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for Remap.");
                break;
        }
    };

    switch (aBorder)
    {
        case opp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<T, BorderType::Replicate, true, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<T, BorderType::Constant, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<T, BorderType::Replicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<T, BorderType::Mirror, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<T, BorderType::MirrorReplicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<T, BorderType::Wrap, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
            break;
    }
}

template <PixelType T>
ImageView<T> &ImageView<T>::Remap(ImageView<T> &aDst, const ImageView<Pixel32fC1> &aCoordinateMapX,
                                  const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                                  BorderType aBorder, Roi aAllowedReadRoi) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Remap(aDst, aCoordinateMapX, aCoordinateMapY, aInterpolation, aBorder, {0}, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::Remap(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Remap(aSrc1, aSrc2, aDst1, aDst2, aCoordinateMapX, aCoordinateMapY, aInterpolation, aBorder, {0},
                        aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::Remap(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aCoordinateMapX, aCoordinateMapY, aInterpolation,
                        aBorder, {0}, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::Remap(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Remap(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aCoordinateMapX, aCoordinateMapY,
                        aInterpolation, aBorder, {0}, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Remap(ImageView<T> &aDst, const ImageView<Pixel32fC1> &aCoordinateMapX,
                                  const ImageView<Pixel32fC1> &aCoordinateMapY, InterpolationMode aInterpolation,
                                  BorderType aBorder, T aConstant, Roi aAllowedReadRoi) const
{
    checkSameSize(aDst.SizeRoi(), aCoordinateMapX.SizeRoi());
    checkSameSize(aDst.SizeRoi(), aCoordinateMapY.SizeRoi());

    if (aAllowedReadRoi == Roi())
    {
        aAllowedReadRoi = ROI();
    }
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
    using CoordTInterpol                = float; // even for double types?
    using CoordT                        = float; // even for double types?

    const TransformerMap2<CoordT> transMap(aCoordinateMapX.PointerRoi(), aCoordinateMapX.Pitch(),
                                           aCoordinateMapY.PointerRoi(), aCoordinateMapY.Pitch());

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case opp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<T, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, SizeRoi());

                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for Remap.");
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
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
            break;
    }

    return aDst;
}

template <PixelType T>
void ImageView<T>::Remap(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, BorderType aBorder, T aConstant, Roi aAllowedReadRoi)
    requires TwoChannel<T>
{
    if (aSrc1.ROI() != aSrc2.ROI())
    {
        throw INVALIDARGUMENT(
            aSrc1 aSrc2,
            "All source images must have the same ROI defined with the same size and the same starting pixel.");
    }
    checkSameSize(aDst1.ROI(), aDst2.ROI());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMapX.SizeRoi());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMapY.SizeRoi());

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
    using CoordTInterpol                = float; // even for double types?
    using CoordT                        = float; // even for double types?

    const TransformerMap2<CoordT> transMap(aCoordinateMapX.PointerRoi(), aCoordinateMapX.Pitch(),
                                           aCoordinateMapY.PointerRoi(), aCoordinateMapY.Pitch());

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case opp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<T, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for Remap.");
                break;
        }
    };

    switch (aBorder)
    {
        case opp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<T, BorderType::Replicate, true, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<T, BorderType::Constant, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset,
                            aConstant);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<T, BorderType::Replicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<T, BorderType::Mirror, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<T, BorderType::MirrorReplicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<T, BorderType::Wrap, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
            break;
    }
}

template <PixelType T>
void ImageView<T>::Remap(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, BorderType aBorder, T aConstant, Roi aAllowedReadRoi)
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
    checkSameSize(aDst1.SizeRoi(), aCoordinateMapX.SizeRoi());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMapY.SizeRoi());

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
    using CoordTInterpol                = float; // even for double types?
    using CoordT                        = float; // even for double types?

    const TransformerMap2<CoordT> transMap(aCoordinateMapX.PointerRoi(), aCoordinateMapX.Pitch(),
                                           aCoordinateMapY.PointerRoi(), aCoordinateMapY.Pitch());

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case opp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<T, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for Remap.");
                break;
        }
    };

    switch (aBorder)
    {
        case opp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<T, BorderType::Replicate, true, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<T, BorderType::Constant, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset, aConstant);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<T, BorderType::Replicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<T, BorderType::Mirror, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<T, BorderType::MirrorReplicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<T, BorderType::Wrap, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
            break;
    }
}

template <PixelType T>
void ImageView<T>::Remap(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                         ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                         ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                         const ImageView<Pixel32fC1> &aCoordinateMapX, const ImageView<Pixel32fC1> &aCoordinateMapY,
                         InterpolationMode aInterpolation, BorderType aBorder, T aConstant, Roi aAllowedReadRoi)
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
    checkSameSize(aDst1.SizeRoi(), aCoordinateMapX.SizeRoi());
    checkSameSize(aDst1.SizeRoi(), aCoordinateMapY.SizeRoi());

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
    using CoordTInterpol                = float; // even for double types?
    using CoordT                        = float; // even for double types?

    const TransformerMap2<CoordT> transMap(aCoordinateMapX.PointerRoi(), aCoordinateMapX.Pitch(),
                                           aCoordinateMapY.PointerRoi(), aCoordinateMapY.Pitch());

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case opp::InterpolationMode::NearestNeighbor:
            {
                using InterpolatorT = Interpolator<T, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            case opp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<T>, bcT, CoordTInterpol, InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                const TransformerFunctor<1, T, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                         TransformerMap2<CoordT>, roundingMode>
                    functor(interpol, transMap, aSrc1.SizeRoi());

                forEachPixelPlanar(aDst1, aDst2, aDst3, aDst4, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for Remap.");
                break;
        }
    };

    switch (aBorder)
    {
        case opp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<T, BorderType::Replicate, true, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<T, BorderType::Constant, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<T, BorderType::Replicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<T, BorderType::Mirror, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<T, BorderType::MirrorReplicate, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<T, BorderType::Wrap, false, false, false, true>;
            const BCType bc(allowedPtr1, aSrc1.Pitch(), allowedPtr2, aSrc2.Pitch(), allowedPtr3, aSrc3.Pitch(),
                            allowedPtr4, aSrc4.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
            break;
    }
}
#pragma endregion

#pragma region Copy (with border control)
template <PixelType T>
ImageView<T> &ImageView<T>::Copy(ImageView<T> &aDst, const Vector2<int> &aLowerBorderSize, BorderType aBorder,
                                 T aConstant) const
{
    constexpr RoundingMode roundingMode = RoundingMode::None;
    using CoordT                        = int;

    const TransformerShift<CoordT> shift(aLowerBorderSize);

    constexpr Vector2<int> roiOffset(0);

    switch (aBorder)
    {
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<T, BorderType::Constant, false, true, false, false>;
            const BCType bc(PointerRoi(), Pitch(), SizeRoi(), roiOffset, aConstant);

            using InterpolatorT = Interpolator<T, BCType, CoordT, InterpolationMode::Undefined>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, false, InterpolatorT, TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<T, BorderType::Replicate, false, true, false, false>;
            const BCType bc(PointerRoi(), Pitch(), SizeRoi(), roiOffset);

            using InterpolatorT = Interpolator<T, BCType, CoordT, InterpolationMode::Undefined>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, false, InterpolatorT, TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<T, BorderType::Mirror, false, true, false, false>;
            const BCType bc(PointerRoi(), Pitch(), SizeRoi(), roiOffset);

            using InterpolatorT = Interpolator<T, BCType, CoordT, InterpolationMode::Undefined>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, false, InterpolatorT, TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<T, BorderType::MirrorReplicate, false, true, false, false>;
            const BCType bc(PointerRoi(), Pitch(), SizeRoi(), roiOffset);

            using InterpolatorT = Interpolator<T, BCType, CoordT, InterpolationMode::Undefined>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, false, InterpolatorT, TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<T, BorderType::Wrap, false, true, false, false>;
            const BCType bc(PointerRoi(), Pitch(), SizeRoi(), roiOffset);

            using InterpolatorT = Interpolator<T, BCType, CoordT, InterpolationMode::Undefined>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, false, InterpolatorT, TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Copy.");
            break;
    }

    return aDst;
}
#pragma endregion

#pragma region Copy sub-pix
template <PixelType T>
ImageView<T> &ImageView<T>::Copy(ImageView<T> &aDst, const Pixel32fC2 &aDelta, InterpolationMode aInterpolation) const
{
    using CoordT         = float;
    using CoordTInterpol = CoordT;

    // for compatibility with NPP, negate the shift:
    const TransformerShift<CoordT> shift(-aDelta);

    constexpr Vector2<int> roiOffset(0);

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;

    // for interpolation at the border we will use replicate:
    using BCType = BorderControl<T, BorderType::Replicate, false, false, false, false>;
    const BCType bc(PointerRoi(), Pitch(), SizeRoi(), roiOffset);

    switch (aInterpolation)
    {
        case opp::InterpolationMode::Linear:
        {
            using InterpolatorT =
                Interpolator<geometry_compute_type_for_t<T>, BCType, CoordTInterpol, InterpolationMode::Linear>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, BCType::only_for_interpolation, InterpolatorT,
                                     TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::InterpolationMode::CubicHermiteSpline:
        {
            using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, BCType, CoordTInterpol,
                                               InterpolationMode::CubicHermiteSpline>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, BCType::only_for_interpolation, InterpolatorT,
                                     TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::InterpolationMode::CubicLagrange:
        {
            using InterpolatorT =
                Interpolator<geometry_compute_type_for_t<T>, BCType, CoordTInterpol, InterpolationMode::CubicLagrange>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, BCType::only_for_interpolation, InterpolatorT,
                                     TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::InterpolationMode::Cubic2ParamBSpline:
        {
            using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, BCType, CoordTInterpol,
                                               InterpolationMode::Cubic2ParamBSpline>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, BCType::only_for_interpolation, InterpolatorT,
                                     TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::InterpolationMode::Cubic2ParamCatmullRom:
        {
            using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, BCType, CoordTInterpol,
                                               InterpolationMode::Cubic2ParamCatmullRom>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, BCType::only_for_interpolation, InterpolatorT,
                                     TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::InterpolationMode::Cubic2ParamB05C03:
        {
            using InterpolatorT = Interpolator<geometry_compute_type_for_t<T>, BCType, CoordTInterpol,
                                               InterpolationMode::Cubic2ParamB05C03>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, BCType::only_for_interpolation, InterpolatorT,
                                     TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::InterpolationMode::Lanczos2Lobed:
        {
            using InterpolatorT =
                Interpolator<geometry_compute_type_for_t<T>, BCType, CoordTInterpol, InterpolationMode::Lanczos2Lobed>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, BCType::only_for_interpolation, InterpolatorT,
                                     TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        case opp::InterpolationMode::Lanczos3Lobed:
        {
            using InterpolatorT =
                Interpolator<geometry_compute_type_for_t<T>, BCType, CoordTInterpol, InterpolationMode::Lanczos3Lobed>;
            const InterpolatorT interpol(bc);
            const TransformerFunctor<1, T, CoordT, BCType::only_for_interpolation, InterpolatorT,
                                     TransformerShift<CoordT>, roundingMode>
                functor(interpol, shift, SizeRoi());

            forEachPixel(aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aInterpolation,
                                  aInterpolation << " is not a supported interpolation mode for CopySubPix.");
            break;
    }

    return aDst;
}
#pragma endregion
} // namespace opp::image::cpuSimple