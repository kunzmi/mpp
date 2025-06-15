#if OPP_ENABLE_CUDA_BACKEND

#include "resize.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/image/forEachPixelPlanar2Kernel.h>
#include <backends/cuda/image/forEachPixelPlanar3Kernel.h>
#include <backends/cuda/image/forEachPixelPlanar4Kernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/functors/interpolator.h>
#include <common/image/functors/transformer.h>
#include <common/image/functors/transformerFunctor.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{
template <typename SrcT>
void InvokeResizeSrc(const SrcT *aSrc1, size_t aPitchSrc1, SrcT *aDst, size_t aPitchDst, const Vector2<double> &aScale,
                     const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder,
                     const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize,
                     const Size2D &aSizeSrc, const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
        using CoordTInterpol                = coordinate_type_interpolation_for_t<SrcT>;
        using CoordT                        = coordinate_type_interpolation_for_t<SrcT>; // double for Vector<double>...

        // the factor/shift values are given for a transformation src->dst, we thus have to invert:
        // (and add shift to recenter pixels)
        const Vector2<CoordT> invScale = static_cast<CoordT>(1) / Vector2<CoordT>(aScale);
        const Vector2<CoordT> invShift =
            Vector2<CoordT>(aShift) * invScale + ((static_cast<CoordT>(1) - invScale) / static_cast<CoordT>(2));

        const TransformerResize<CoordT> resize(invScale, invShift);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case opp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                               functor);
                }
                break;
                case opp::InterpolationMode::Linear:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT =
                        Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                               functor);
                }
                break;
                case opp::InterpolationMode::CubicHermiteSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicHermiteSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                               functor);
                }
                break;
                case opp::InterpolationMode::CubicLagrange:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicLagrange>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                               functor);
                }
                break;
                case opp::InterpolationMode::Cubic2ParamBSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamBSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                               functor);
                }
                break;
                case opp::InterpolationMode::Cubic2ParamCatmullRom:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamCatmullRom>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                               functor);
                }
                break;
                case opp::InterpolationMode::Cubic2ParamB05C03:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamB05C03>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                               functor);
                }
                break;
                case opp::InterpolationMode::Lanczos2Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos2Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                               functor);
                }
                break;
                case opp::InterpolationMode::Lanczos3Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos3Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                               functor);
                }
                break;
                case opp::InterpolationMode::Super:
                {
                    // assuming that we already checked that scaleFactors are < 1
                    Vector2<CoordT> scaleFactor = Vector2<CoordT>(aScale);

                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT =
                        Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Super>;
                    const InterpolatorT interpol(aBC, scaleFactor.x, scaleFactor.y);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                               functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aInterpolation,
                                          aInterpolation << " is not a supported interpolation mode for Resize.");
                    break;
            }
        };

        switch (aBorder)
        {
            case opp::BorderType::None:
            {
                // for interpolation at the border we will still use replicate:
                using BCType = BorderControl<SrcT, BorderType::Replicate, true, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Constant:
            {
                using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset, aConstant);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Replicate:
            {
                using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Mirror:
            {
                using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::MirrorReplicate:
            {
                using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Wrap:
            {
                using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            default:
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Resize.");
                break;
        }
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeResizeSrc<typeSrc>(                                                                            \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeSrc *aDst, size_t aPitchDst, const Vector2<double> &aScale,       \
        const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant, \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx);

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
void InvokeResizeSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                     const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                     Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1, Vector1<remove_vector_t<SrcT>> *aDst2,
                     size_t aPitchDst2, const Vector2<double> &aScale, const Vector2<double> &aShift,
                     InterpolationMode aInterpolation, BorderType aBorder, const SrcT &aConstant,
                     const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize,
                     const Size2D &aSizeSrc, const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
        using CoordTInterpol                = coordinate_type_interpolation_for_t<SrcT>;
        using CoordT                        = coordinate_type_interpolation_for_t<SrcT>; // double for Vector<double>...

        // the factor/shift values are given for a transformation src->dst, we thus have to invert:
        // (and add shift to recenter pixels)
        const Vector2<CoordT> invScale = static_cast<CoordT>(1) / Vector2<CoordT>(aScale);
        const Vector2<CoordT> invShift =
            Vector2<CoordT>(aShift) * invScale + ((static_cast<CoordT>(1) - invScale) / static_cast<CoordT>(2));

        const TransformerResize<CoordT> resize(invScale, invShift);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case opp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Linear:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT =
                        Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::CubicHermiteSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicHermiteSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::CubicLagrange:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicLagrange>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Cubic2ParamBSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamBSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Cubic2ParamCatmullRom:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamCatmullRom>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Cubic2ParamB05C03:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamB05C03>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Lanczos2Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos2Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Lanczos3Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos3Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Super:
                {
                    // assuming that we already checked that scaleFactors are < 1
                    Vector2<CoordT> scaleFactor = Vector2<CoordT>(aScale);

                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT =
                        Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Super>;
                    const InterpolatorT interpol(aBC, scaleFactor.x, scaleFactor.y);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aInterpolation,
                                          aInterpolation << " is not a supported interpolation mode for Resize.");
                    break;
            }
        };

        switch (aBorder)
        {
            case opp::BorderType::None:
            {
                // for interpolation at the border we will still use replicate:
                using BCType = BorderControl<SrcT, BorderType::Replicate, true, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Constant:
            {
                using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset,
                                aConstant);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Replicate:
            {
                using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Mirror:
            {
                using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::MirrorReplicate:
            {
                using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Wrap:
            {
                using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            default:
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Resize.");
                break;
        }
    }
}

#pragma region Instantiate

#define InstantiateP2_For(typeSrc)                                                                                     \
    template void InvokeResizeSrc<typeSrc>(                                                                            \
        const Vector1<remove_vector_t<typeSrc>> *aSrc1, size_t aPitchSrc1,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc2, size_t aPitchSrc2, Vector1<remove_vector_t<typeSrc>> *aDst1,   \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrc>> *aDst2, size_t aPitchDst2, const Vector2<double> &aScale, \
        const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant, \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx);

#define InstantiateP2_ForGeomType(type) InstantiateP2_For(Pixel##type##C2);

#pragma endregion

template <typename SrcT>
void InvokeResizeSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                     const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                     const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3,
                     Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1, Vector1<remove_vector_t<SrcT>> *aDst2,
                     size_t aPitchDst2, Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3,
                     const Vector2<double> &aScale, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                     BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                     const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                     const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
        using CoordTInterpol                = coordinate_type_interpolation_for_t<SrcT>;
        using CoordT                        = coordinate_type_interpolation_for_t<SrcT>; // double for Vector<double>...

        // the factor/shift values are given for a transformation src->dst, we thus have to invert:
        // (and add shift to recenter pixels)
        const Vector2<CoordT> invScale = static_cast<CoordT>(1) / Vector2<CoordT>(aScale);
        const Vector2<CoordT> invShift =
            Vector2<CoordT>(aShift) * invScale + ((static_cast<CoordT>(1) - invScale) / static_cast<CoordT>(2));

        const TransformerResize<CoordT> resize(invScale, invShift);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case opp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Linear:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT =
                        Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::CubicHermiteSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicHermiteSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::CubicLagrange:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicLagrange>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Cubic2ParamBSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamBSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Cubic2ParamCatmullRom:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamCatmullRom>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Cubic2ParamB05C03:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamB05C03>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Lanczos2Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos2Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Lanczos3Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos3Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Super:
                {
                    // assuming that we already checked that scaleFactors are < 1
                    Vector2<CoordT> scaleFactor = Vector2<CoordT>(aScale);

                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT =
                        Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Super>;
                    const InterpolatorT interpol(aBC, scaleFactor.x, scaleFactor.y);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aInterpolation,
                                          aInterpolation << " is not a supported interpolation mode for Resize.");
                    break;
            }
        };

        switch (aBorder)
        {
            case opp::BorderType::None:
            {
                // for interpolation at the border we will still use replicate:
                using BCType = BorderControl<SrcT, BorderType::Replicate, true, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                                aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Constant:
            {
                using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                                aAllowedReadRoiOffset, aConstant);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Replicate:
            {
                using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                                aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Mirror:
            {
                using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                                aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::MirrorReplicate:
            {
                using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                                aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Wrap:
            {
                using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aAllowedReadRoiSize,
                                aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            default:
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Resize.");
                break;
        }
    }
}

#pragma region Instantiate

#define InstantiateP3_For(typeSrc)                                                                                     \
    template void InvokeResizeSrc<typeSrc>(                                                                            \
        const Vector1<remove_vector_t<typeSrc>> *aSrc1, size_t aPitchSrc1,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc2, size_t aPitchSrc2,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc3, size_t aPitchSrc3, Vector1<remove_vector_t<typeSrc>> *aDst1,   \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrc>> *aDst2, size_t aPitchDst2,                                \
        Vector1<remove_vector_t<typeSrc>> *aDst3, size_t aPitchDst3, const Vector2<double> &aScale,                    \
        const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant, \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx);

#define InstantiateP3_ForGeomType(type) InstantiateP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcT>
void InvokeResizeSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                     const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                     const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3,
                     const Vector1<remove_vector_t<SrcT>> *aSrc4, size_t aPitchSrc4,
                     Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1, Vector1<remove_vector_t<SrcT>> *aDst2,
                     size_t aPitchDst2, Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3,
                     Vector1<remove_vector_t<SrcT>> *aDst4, size_t aPitchDst4, const Vector2<double> &aScale,
                     const Vector2<double> &aShift, InterpolationMode aInterpolation, BorderType aBorder,
                     const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize,
                     const Size2D &aSizeSrc, const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
        using CoordTInterpol                = coordinate_type_interpolation_for_t<SrcT>;
        using CoordT                        = coordinate_type_interpolation_for_t<SrcT>; // double for Vector<double>...

        // the factor/shift values are given for a transformation src->dst, we thus have to invert:
        // (and add shift to recenter pixels)
        const Vector2<CoordT> invScale = static_cast<CoordT>(1) / Vector2<CoordT>(aScale);
        const Vector2<CoordT> invShift =
            Vector2<CoordT>(aShift) * invScale + ((static_cast<CoordT>(1) - invScale) / static_cast<CoordT>(2));

        const TransformerResize<CoordT> resize(invScale, invShift);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case opp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Linear:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT =
                        Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::CubicHermiteSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicHermiteSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::CubicLagrange:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicLagrange>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Cubic2ParamBSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamBSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Cubic2ParamCatmullRom:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamCatmullRom>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Cubic2ParamB05C03:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamB05C03>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Lanczos2Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos2Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Lanczos3Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos3Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case opp::InterpolationMode::Super:
                {
                    // assuming that we already checked that scaleFactors are < 1
                    Vector2<CoordT> scaleFactor = Vector2<CoordT>(aScale);

                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT =
                        Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Super>;
                    const InterpolatorT interpol(aBC, scaleFactor.x, scaleFactor.y);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerResize<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, resize, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aInterpolation,
                                          aInterpolation << " is not a supported interpolation mode for Resize.");
                    break;
            }
        };

        switch (aBorder)
        {
            case opp::BorderType::None:
            {
                // for interpolation at the border we will still use replicate:
                using BCType = BorderControl<SrcT, BorderType::Replicate, true, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                                aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Constant:
            {
                using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                                aAllowedReadRoiSize, aAllowedReadRoiOffset, aConstant);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Replicate:
            {
                using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                                aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Mirror:
            {
                using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                                aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::MirrorReplicate:
            {
                using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                                aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            case opp::BorderType::Wrap:
            {
                using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, true>;
                const BCType bc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4,
                                aAllowedReadRoiSize, aAllowedReadRoiOffset);

                runOverInterpolation(bc);
            }
            break;
            default:
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Resize.");
                break;
        }
    }
}

#pragma region Instantiate

#define InstantiateP4_For(typeSrc)                                                                                     \
    template void InvokeResizeSrc<typeSrc>(                                                                            \
        const Vector1<remove_vector_t<typeSrc>> *aSrc1, size_t aPitchSrc1,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc2, size_t aPitchSrc2,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc3, size_t aPitchSrc3,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc4, size_t aPitchSrc4, Vector1<remove_vector_t<typeSrc>> *aDst1,   \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrc>> *aDst2, size_t aPitchDst2,                                \
        Vector1<remove_vector_t<typeSrc>> *aDst3, size_t aPitchDst3, Vector1<remove_vector_t<typeSrc>> *aDst4,         \
        size_t aPitchDst4, const Vector2<double> &aScale, const Vector2<double> &aShift,                               \
        InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant,                                \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx);

#define InstantiateP4_ForGeomType(type) InstantiateP4_For(Pixel##type##C4);

#pragma endregion
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
