#if MPP_ENABLE_CUDA_BACKEND

#include "remap.h"
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
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{
template <typename SrcT>
void InvokeRemapSrc(const SrcT *aSrc1, size_t aPitchSrc1, SrcT *aDst, size_t aPitchDst,
                    const Pixel32fC2 *aCoordinateMapPtr, size_t aCoordinateMapPitch, InterpolationMode aInterpolation,
                    BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                    const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
        using CoordTInterpol                = float; // even for double types?
        using CoordT                        = float; // even for double types?

        const TransformerMap<CoordT> transMap(aCoordinateMapPtr, aCoordinateMapPitch);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case mpp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                               functor);
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
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
                break;
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeRemapSrcFloat2_For(typeSrc)                                                                   \
    template void InvokeRemapSrc<typeSrc>(                                                                             \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeSrc *aDst, size_t aPitchDst, const Pixel32fC2 *aCoordinateMapPtr, \
        size_t aCoordinateMapPitch, InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant,    \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInstantiateInvokeRemapSrcFloat2_For(type)                                                 \
    InstantiateInvokeRemapSrcFloat2_For(Pixel##type##C1);                                                              \
    InstantiateInvokeRemapSrcFloat2_For(Pixel##type##C2);                                                              \
    InstantiateInvokeRemapSrcFloat2_For(Pixel##type##C3);                                                              \
    InstantiateInvokeRemapSrcFloat2_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInstantiateInvokeRemapSrcFloat2_For(type)                                               \
    InstantiateInvokeRemapSrcFloat2_For(Pixel##type##C1);                                                              \
    InstantiateInvokeRemapSrcFloat2_For(Pixel##type##C2);                                                              \
    InstantiateInvokeRemapSrcFloat2_For(Pixel##type##C3);                                                              \
    InstantiateInvokeRemapSrcFloat2_For(Pixel##type##C4);                                                              \
    InstantiateInvokeRemapSrcFloat2_For(Pixel##type##C4A);

#pragma endregion
template <typename SrcT>
void InvokeRemapSrc(const SrcT *aSrc1, size_t aPitchSrc1, SrcT *aDst, size_t aPitchDst,
                    const Pixel32fC1 *aCoordinateMapXPtr, size_t aCoordinateMapXPitch,
                    const Pixel32fC1 *aCoordinateMapYPtr, size_t aCoordinateMapYPitch, InterpolationMode aInterpolation,
                    BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                    const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
        using CoordTInterpol                = float; // even for double types?
        using CoordT                        = float; // even for double types?

        const TransformerMap2<CoordT> transMap(aCoordinateMapXPtr, aCoordinateMapXPitch, aCoordinateMapYPtr,
                                               aCoordinateMapYPitch);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case mpp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelKernelDefault<SrcT, tupelSize, FunctorT>(aDst, aPitchDst, aSizeDst, aStreamCtx,
                                                                               functor);
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
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
                break;
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeRemapSrc2Float_For(typeSrc)                                                                   \
    template void InvokeRemapSrc<typeSrc>(                                                                             \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeSrc *aDst, size_t aPitchDst,                                      \
        const Pixel32fC1 *aCoordinateMapXPtr, size_t aCoordinateMapXPitch, const Pixel32fC1 *aCoordinateMapYPtr,       \
        size_t aCoordinateMapYPitch, InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant,   \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInstantiateInvokeRemapSrc2Float_For(type)                                                 \
    InstantiateInvokeRemapSrc2Float_For(Pixel##type##C1);                                                              \
    InstantiateInvokeRemapSrc2Float_For(Pixel##type##C2);                                                              \
    InstantiateInvokeRemapSrc2Float_For(Pixel##type##C3);                                                              \
    InstantiateInvokeRemapSrc2Float_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInstantiateInvokeRemapSrc2Float_For(type)                                               \
    InstantiateInvokeRemapSrc2Float_For(Pixel##type##C1);                                                              \
    InstantiateInvokeRemapSrc2Float_For(Pixel##type##C2);                                                              \
    InstantiateInvokeRemapSrc2Float_For(Pixel##type##C3);                                                              \
    InstantiateInvokeRemapSrc2Float_For(Pixel##type##C4);                                                              \
    InstantiateInvokeRemapSrc2Float_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT>
void InvokeRemapSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                    const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                    Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1, Vector1<remove_vector_t<SrcT>> *aDst2,
                    size_t aPitchDst2, const Pixel32fC2 *aCoordinateMapPtr, size_t aCoordinateMapPitch,
                    InterpolationMode aInterpolation, BorderType aBorder, const SrcT &aConstant,
                    const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,
                    const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
        using CoordTInterpol                = float; // even for double types?
        using CoordT                        = float; // even for double types?

        const TransformerMap<CoordT> transMap(aCoordinateMapPtr, aCoordinateMapPitch);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case mpp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Linear:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT =
                        Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::CubicHermiteSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicHermiteSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::CubicLagrange:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicLagrange>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Cubic2ParamBSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamBSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Cubic2ParamCatmullRom:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamCatmullRom>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Cubic2ParamB05C03:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamB05C03>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Lanczos2Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos2Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Lanczos3Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos3Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
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
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
                break;
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeRemapSrcP2_Float2_For(typeSrc)                                                                \
    template void InvokeRemapSrc<typeSrc>(                                                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc1, size_t aPitchSrc1,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc2, size_t aPitchSrc2, Vector1<remove_vector_t<typeSrc>> *aDst1,   \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrc>> *aDst2, size_t aPitchDst2,                                \
        const Pixel32fC2 *aCoordinateMapPtr, size_t aCoordinateMapPitch, InterpolationMode aInterpolation,             \
        BorderType aBorder, const typeSrc &aConstant, const Vector2<int> aAllowedReadRoiOffset,                        \
        const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,                             \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define InstantiateInvokeRemapSrcP2_Float2_ForGeomType(type) InstantiateInvokeRemapSrcP2_Float2_For(Pixel##type##C2);

#pragma endregion

template <typename SrcT>
void InvokeRemapSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                    const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                    Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1, Vector1<remove_vector_t<SrcT>> *aDst2,
                    size_t aPitchDst2, const Pixel32fC1 *aCoordinateMapXPtr, size_t aCoordinateMapXPitch,
                    const Pixel32fC1 *aCoordinateMapYPtr, size_t aCoordinateMapYPitch, InterpolationMode aInterpolation,
                    BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                    const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
        using CoordTInterpol                = float; // even for double types?
        using CoordT                        = float; // even for double types?

        const TransformerMap2<CoordT> transMap(aCoordinateMapXPtr, aCoordinateMapXPitch, aCoordinateMapYPtr,
                                               aCoordinateMapYPitch);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case mpp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Linear:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT =
                        Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::CubicHermiteSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicHermiteSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::CubicLagrange:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicLagrange>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Cubic2ParamBSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamBSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Cubic2ParamCatmullRom:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamCatmullRom>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Cubic2ParamB05C03:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamB05C03>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Lanczos2Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos2Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Lanczos3Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos3Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar2KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aSizeDst, aStreamCtx, functor);
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
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
                break;
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeRemapSrcP2_2Float_For(typeSrc)                                                                \
    template void InvokeRemapSrc<typeSrc>(                                                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc1, size_t aPitchSrc1,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc2, size_t aPitchSrc2, Vector1<remove_vector_t<typeSrc>> *aDst1,   \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrc>> *aDst2, size_t aPitchDst2,                                \
        const Pixel32fC1 *aCoordinateMapXPtr, size_t aCoordinateMapXPitch, const Pixel32fC1 *aCoordinateMapYPtr,       \
        size_t aCoordinateMapYPitch, InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant,   \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx);

#define InstantiateInvokeRemapSrcP2_2Float_ForGeomType(type) InstantiateInvokeRemapSrcP2_2Float_For(Pixel##type##C2);

#pragma endregion

template <typename SrcT>
void InvokeRemapSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                    const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                    const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3,
                    Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1, Vector1<remove_vector_t<SrcT>> *aDst2,
                    size_t aPitchDst2, Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3,
                    const Pixel32fC2 *aCoordinateMapPtr, size_t aCoordinateMapPitch, InterpolationMode aInterpolation,
                    BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                    const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
        using CoordTInterpol                = float; // even for double types?
        using CoordT                        = float; // even for double types?

        const TransformerMap<CoordT> transMap(aCoordinateMapPtr, aCoordinateMapPitch);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case mpp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
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
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
                break;
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeRemapSrcP3_Float2_For(typeSrc)                                                                \
    template void InvokeRemapSrc<typeSrc>(                                                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc1, size_t aPitchSrc1,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc2, size_t aPitchSrc2,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc3, size_t aPitchSrc3, Vector1<remove_vector_t<typeSrc>> *aDst1,   \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrc>> *aDst2, size_t aPitchDst2,                                \
        Vector1<remove_vector_t<typeSrc>> *aDst3, size_t aPitchDst3, const Pixel32fC2 *aCoordinateMapPtr,              \
        size_t aCoordinateMapPitch, InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant,    \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx);

#define InstantiateInvokeRemapSrcP3_Float2_ForGeomType(type) InstantiateInvokeRemapSrcP3_Float2_For(Pixel##type##C3);

#pragma endregion

template <typename SrcT>
void InvokeRemapSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                    const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                    const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3,
                    Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1, Vector1<remove_vector_t<SrcT>> *aDst2,
                    size_t aPitchDst2, Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3,
                    const Pixel32fC1 *aCoordinateMapXPtr, size_t aCoordinateMapXPitch,
                    const Pixel32fC1 *aCoordinateMapYPtr, size_t aCoordinateMapYPitch, InterpolationMode aInterpolation,
                    BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                    const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
        using CoordTInterpol                = float; // even for double types?
        using CoordT                        = float; // even for double types?

        const TransformerMap2<CoordT> transMap(aCoordinateMapXPtr, aCoordinateMapXPitch, aCoordinateMapYPtr,
                                               aCoordinateMapYPitch);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case mpp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

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
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar3KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSizeDst, aStreamCtx, functor);
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
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
                break;
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeRemapSrcP3_2Float_For(typeSrc)                                                                \
    template void InvokeRemapSrc<typeSrc>(                                                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc1, size_t aPitchSrc1,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc2, size_t aPitchSrc2,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc3, size_t aPitchSrc3, Vector1<remove_vector_t<typeSrc>> *aDst1,   \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrc>> *aDst2, size_t aPitchDst2,                                \
        Vector1<remove_vector_t<typeSrc>> *aDst3, size_t aPitchDst3, const Pixel32fC1 *aCoordinateMapXPtr,             \
        size_t aCoordinateMapXPitch, const Pixel32fC1 *aCoordinateMapYPtr, size_t aCoordinateMapYPitch,                \
        InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant,                                \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx);

#define InstantiateInvokeRemapSrcP3_2Float_ForGeomType(type) InstantiateInvokeRemapSrcP3_2Float_For(Pixel##type##C3);

#pragma endregion

template <typename SrcT>
void InvokeRemapSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                    const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                    const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3,
                    const Vector1<remove_vector_t<SrcT>> *aSrc4, size_t aPitchSrc4,
                    Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1, //
                    Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2, //
                    Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3, //
                    Vector1<remove_vector_t<SrcT>> *aDst4, size_t aPitchDst4, const Pixel32fC2 *aCoordinateMapPtr,
                    size_t aCoordinateMapPitch, InterpolationMode aInterpolation, BorderType aBorder,
                    const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize,
                    const Size2D &aSizeSrc, const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
        using CoordTInterpol                = float; // even for double types?
        using CoordT                        = float; // even for double types?

        const TransformerMap<CoordT> transMap(aCoordinateMapPtr, aCoordinateMapPitch);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case mpp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Linear:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT =
                        Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::CubicHermiteSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicHermiteSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::CubicLagrange:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicLagrange>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Cubic2ParamBSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamBSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Cubic2ParamCatmullRom:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamCatmullRom>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Cubic2ParamB05C03:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamB05C03>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Lanczos2Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos2Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Lanczos3Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos3Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
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
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
                break;
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeRemapSrcP4_Float2_For(typeSrc)                                                                \
    template void InvokeRemapSrc<typeSrc>(                                                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc1, size_t aPitchSrc1,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc2, size_t aPitchSrc2,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc3, size_t aPitchSrc3,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc4, size_t aPitchSrc4, Vector1<remove_vector_t<typeSrc>> *aDst1,   \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrc>> *aDst2, size_t aPitchDst2,                                \
        Vector1<remove_vector_t<typeSrc>> *aDst3, size_t aPitchDst3, Vector1<remove_vector_t<typeSrc>> *aDst4,         \
        size_t aPitchDst4, const Pixel32fC2 *aCoordinateMapPtr, size_t aCoordinateMapPitch,                            \
        InterpolationMode aInterpolation, BorderType aBorder, const typeSrc &aConstant,                                \
        const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,           \
        const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx);

#define InstantiateInvokeRemapSrcP4_Float2_ForGeomType(type) InstantiateInvokeRemapSrcP4_Float2_For(Pixel##type##C4);

#pragma endregion

template <typename SrcT>
void InvokeRemapSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                    const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                    const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3,
                    const Vector1<remove_vector_t<SrcT>> *aSrc4, size_t aPitchSrc4,
                    Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1, //
                    Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2, //
                    Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3, //
                    Vector1<remove_vector_t<SrcT>> *aDst4, size_t aPitchDst4, const Pixel32fC1 *aCoordinateMapXPtr,
                    size_t aCoordinateMapXPitch, const Pixel32fC1 *aCoordinateMapYPtr, size_t aCoordinateMapYPitch,
                    InterpolationMode aInterpolation, BorderType aBorder, const SrcT &aConstant,
                    const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,
                    const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;
        using CoordTInterpol                = float; // even for double types?
        using CoordT                        = float; // even for double types?

        const TransformerMap2<CoordT> transMap(aCoordinateMapXPtr, aCoordinateMapXPitch, aCoordinateMapYPtr,
                                               aCoordinateMapYPitch);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case mpp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Linear:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT =
                        Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol, InterpolationMode::Linear>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::CubicHermiteSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicHermiteSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::CubicLagrange:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::CubicLagrange>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Cubic2ParamBSpline:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamBSpline>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Cubic2ParamCatmullRom:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamCatmullRom>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Cubic2ParamB05C03:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Cubic2ParamB05C03>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Lanczos2Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos2Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
                }
                break;
                case mpp::InterpolationMode::Lanczos3Lobed:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT        = Interpolator<geometry_compute_type_for_t<SrcT>, bcT, CoordTInterpol,
                                                              InterpolationMode::Lanczos3Lobed>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerMap2<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, transMap, aSizeSrc);

                    InvokeForEachPixelPlanar4KernelDefault<SrcT, tupelSize, FunctorT>(
                        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSizeDst,
                        aStreamCtx, functor);
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
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for Remap.");
                break;
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeRemapSrcP4_2Float_For(typeSrc)                                                                \
    template void InvokeRemapSrc<typeSrc>(                                                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc1, size_t aPitchSrc1,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc2, size_t aPitchSrc2,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc3, size_t aPitchSrc3,                                             \
        const Vector1<remove_vector_t<typeSrc>> *aSrc4, size_t aPitchSrc4, Vector1<remove_vector_t<typeSrc>> *aDst1,   \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrc>> *aDst2, size_t aPitchDst2,                                \
        Vector1<remove_vector_t<typeSrc>> *aDst3, size_t aPitchDst3, Vector1<remove_vector_t<typeSrc>> *aDst4,         \
        size_t aPitchDst4, const Pixel32fC1 *aCoordinateMapXPtr, size_t aCoordinateMapXPitch,                          \
        const Pixel32fC1 *aCoordinateMapYPtr, size_t aCoordinateMapYPitch, InterpolationMode aInterpolation,           \
        BorderType aBorder, const typeSrc &aConstant, const Vector2<int> aAllowedReadRoiOffset,                        \
        const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,                             \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define InstantiateInvokeRemapSrcP4_2Float_ForGeomType(type) InstantiateInvokeRemapSrcP4_2Float_For(Pixel##type##C4);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
