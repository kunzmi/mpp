#include "affineBack.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/image/forEachPixelPlanar2Kernel.h>
#include <backends/cuda/image/forEachPixelPlanar3Kernel.h>
#include <backends/cuda/image/forEachPixelPlanar4Kernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/functors/interpolator.h>
#include <common/image/functors/transformer.h>
#include <common/image/functors/transformerFunctor.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{
template <typename SrcT>
void InvokeAffineBackSrc(const SrcT *aSrc1, size_t aPitchSrc1, SrcT *aDst, size_t aPitchDst,
                         const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                         BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                         const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                         const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
    using CoordTInterpol                = coordinate_type_interpolation_for_t<SrcT>;
    using CoordT                        = coordinate_type_interpolation_for_t<SrcT>; // double for Vector<double>...
    const AffineTransformation<CoordT> transformTyped(aAffine);
    const TransformerAffine<CoordT> affine(transformTyped);

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case mpp::InterpolationMode::NearestNeighbor:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                           functor);
            }
            break;
            case mpp::InterpolationMode::Linear:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                           functor);
            }
            break;
            case mpp::InterpolationMode::CubicHermiteSpline:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                           functor);
            }
            break;
            case mpp::InterpolationMode::CubicLagrange:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                           functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamBSpline:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                           functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                           functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamB05C03:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                           functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos2Lobed:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                           functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos3Lobed:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                           functor);
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
        case mpp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<SrcT, BorderType::Replicate, true, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset, aConstant);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::SmoothEdge:
        {
            using BCType = BorderControl<SrcT, BorderType::SmoothEdge, true, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for WarpAffine.");
            break;
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeAffineBackSrc<typeSrc>(                                                                        \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeSrc *aDst, size_t aPitchDst,                                      \
        const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, BorderType aBorder,             \
        const typeSrc &aConstant, const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize,         \
        const Size2D &aSizeSrc, const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT>
void InvokeAffineBackSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                         const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                         Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1,
                         Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2,
                         const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                         BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                         const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                         const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
    using CoordTInterpol                = coordinate_type_interpolation_for_t<SrcT>;
    using CoordT                        = coordinate_type_interpolation_for_t<SrcT>; // double for Vector<double>...
    const AffineTransformation<CoordT> transformTyped(aAffine);
    const TransformerAffine<CoordT> affine(transformTyped);

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case mpp::InterpolationMode::NearestNeighbor:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Linear:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::CubicHermiteSpline:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::CubicLagrange:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamBSpline:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamB05C03:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos2Lobed:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos3Lobed:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aSizeDst, aStreamCtx, functor);
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
        case mpp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<SrcT, BorderType::Replicate, true, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset,
                            aConstant);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::SmoothEdge:
        {
            using BCType = BorderControl<SrcT, BorderType::SmoothEdge, true, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for WarpAffine.");
            break;
    }
}

#pragma region Instantiate

#define InstantiateInvokeAffineBackSrcP2_For(typeSrc)                                                                  \
    template void InvokeAffineBackSrc<typeSrc>(                                                                        \
        const Vector1<remove_vector_t<typeSrc>> *aSrc1, size_t aPitchSrc1,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc2, size_t aPitchSrc2, Vector1<remove_vector_t<typeSrc>> *aDst1,   \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrc>> *aDst2, size_t aPitchDst2,                                \
        const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, BorderType aBorder,             \
        const typeSrc &aConstant, const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize,         \
        const Size2D &aSizeSrc, const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx);

#define InstantiateInvokeAffineBackSrcP2ForGeomType(type) InstantiateInvokeAffineBackSrcP2_For(Pixel##type##C2);

#pragma endregion

template <typename SrcT>
void InvokeAffineBackSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                         const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                         const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3,
                         Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1,
                         Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2,
                         Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3,
                         const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
                         BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                         const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                         const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
    using CoordTInterpol                = coordinate_type_interpolation_for_t<SrcT>;
    using CoordT                        = coordinate_type_interpolation_for_t<SrcT>; // double for Vector<double>...
    const AffineTransformation<CoordT> transformTyped(aAffine);
    const TransformerAffine<CoordT> affine(transformTyped);

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case mpp::InterpolationMode::NearestNeighbor:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Linear:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::CubicHermiteSpline:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::CubicLagrange:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamBSpline:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamB05C03:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos2Lobed:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos3Lobed:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
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
        case mpp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<SrcT, BorderType::Replicate, true, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                            aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                            aAllowedReadRoiOffset, aConstant);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                            aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                            aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                            aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                            aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::SmoothEdge:
        {
            using BCType = BorderControl<SrcT, BorderType::SmoothEdge, true, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                            aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for WarpAffine.");
            break;
    }
}

#pragma region Instantiate

#define InstantiateInvokeAffineBackSrcP3_For(typeSrc)                                                                  \
    template void InvokeAffineBackSrc<typeSrc>(                                                                        \
        const Vector1<remove_vector_t<typeSrc>> *aSrc1, size_t aPitchSrc1,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc2, size_t aPitchSrc2,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc3, size_t aPitchSrc3, Vector1<remove_vector_t<typeSrc>> *aDst1,   \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrc>> *aDst2, size_t aPitchDst2,                                \
        Vector1<remove_vector_t<typeSrc>> *aDst3, size_t aPitchDst3, const AffineTransformation<double> &aAffine,      \
        InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant,                                \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx);

#define InstantiateInvokeAffineBackSrcP3ForGeomType(type) InstantiateInvokeAffineBackSrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcT>
void InvokeAffineBackSrc(
    const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1, const Vector1<remove_vector_t<SrcT>> *aSrc2,
    size_t aPitchSrc2, const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3,
    const Vector1<remove_vector_t<SrcT>> *aSrc4, size_t aPitchSrc4, Vector1<remove_vector_t<SrcT>> *aDst1,
    size_t aPitchDst1, Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2, Vector1<remove_vector_t<SrcT>> *aDst3,
    size_t aPitchDst3, Vector1<remove_vector_t<SrcT>> *aDst4, size_t aPitchDst4,
    const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation, BorderType aBorder,
    const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize,
    const Size2D &aSizeSrc, const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
    using CoordTInterpol                = coordinate_type_interpolation_for_t<SrcT>;
    using CoordT                        = coordinate_type_interpolation_for_t<SrcT>; // double for Vector<double>...
    const AffineTransformation<CoordT> transformTyped(aAffine);
    const TransformerAffine<CoordT> affine(transformTyped);

    auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
        switch (aInterpolation)
        {
            case mpp::InterpolationMode::NearestNeighbor:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aDst3, aPitchDst3, aDst4, aPitchDst4,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Linear:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aDst3, aPitchDst3, aDst4, aPitchDst4,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::CubicHermiteSpline:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aDst3, aPitchDst3, aDst4, aPitchDst4,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::CubicLagrange:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aDst3, aPitchDst3, aDst4, aPitchDst4,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamBSpline:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aDst3, aPitchDst3, aDst4, aPitchDst4,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aDst3, aPitchDst3, aDst4, aPitchDst4,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Cubic2ParamB05C03:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aDst3, aPitchDst3, aDst4, aPitchDst4,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos2Lobed:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aDst3, aPitchDst3, aDst4, aPitchDst4,
                                                                                  aSizeDst, aStreamCtx, functor);
            }
            break;
            case mpp::InterpolationMode::Lanczos3Lobed:
            {
                constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                          InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(aBC);
                using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation, InterpolatorT,
                                                    TransformerAffine<CoordT>, roundingMode>;
                const FunctorT functor(interpol, affine, aSizeSrc);

                InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(aDst1, aPitchDst1, aDst2, aPitchDst2,
                                                                                  aDst3, aPitchDst3, aDst4, aPitchDst4,
                                                                                  aSizeDst, aStreamCtx, functor);
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
        case mpp::BorderType::None:
        {
            // for interpolation at the border we will still use replicate:
            using BCType = BorderControl<SrcT, BorderType::Replicate, true, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                            aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                            aAllowedReadRoiSize, aAllowedReadRoiOffset, aConstant);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                            aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                            aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                            aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                            aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        case mpp::BorderType::SmoothEdge:
        {
            using BCType = BorderControl<SrcT, BorderType::SmoothEdge, true, false, false, true>;
            const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                            aAllowedReadRoiSize, aAllowedReadRoiOffset);

            runOverInterpolation(bc);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for WarpAffine.");
            break;
    }
}

#pragma region Instantiate

#define InstantiateInvokeAffineBackSrcP4_For(typeSrc)                                                                  \
    template void InvokeAffineBackSrc<typeSrc>(                                                                        \
        const Vector1<remove_vector_t<typeSrc>> *aSrc1, size_t aPitchSrc1,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc2, size_t aPitchSrc2,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc3, size_t aPitchSrc3,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc4, size_t aPitchSrc4, Vector1<remove_vector_t<typeSrc>> *aDst1,   \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrc>> *aDst2, size_t aPitchDst2,                                \
        Vector1<remove_vector_t<typeSrc>> *aDst3, size_t aPitchDst3, Vector1<remove_vector_t<typeSrc>> *aDst4,         \
        size_t aPitchDst4, const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,              \
        BorderType aBorder, const typeSrc &aConstant, const Vector2<int> aAllowedReadRoiOffset,                        \
        const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,                             \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define InstantiateInvokeAffineBackSrcP4ForGeomType(type) InstantiateInvokeAffineBackSrcP4_For(Pixel##type##C4);

#pragma endregion
} // namespace mpp::image::cuda
