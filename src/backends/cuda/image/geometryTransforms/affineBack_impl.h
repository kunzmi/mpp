#if OPP_ENABLE_CUDA_BACKEND

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
void InvokeAffineBackSrc(const SrcT *aSrc1, size_t aPitchSrc1, SrcT *aDst, size_t aPitchDst,
                         const AffineTransformation<double> &aAffine, InterpolationMode aInterpolation,
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
        const AffineTransformation<CoordT> transformTyped(aAffine);
        const TransformerAffine<CoordT> affine(transformTyped);

        auto runOverInterpolation = [&]<typename bcT>(const bcT &aBC) {
            switch (aInterpolation)
            {
                case opp::InterpolationMode::NearestNeighbor:
                {
                    constexpr size_t tupelSize = bcT::only_for_interpolation ? 1 : TupelSize;
                    using InterpolatorT = Interpolator<SrcT, bcT, CoordTInterpol, InterpolationMode::NearestNeighbor>;
                    const InterpolatorT interpol(aBC);
                    using FunctorT = TransformerFunctor<tupelSize, SrcT, CoordT, bcT::only_for_interpolation,
                                                        InterpolatorT, TransformerAffine<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, affine, aSizeSrc);

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
                                                        InterpolatorT, TransformerAffine<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, affine, aSizeSrc);

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
                                                        InterpolatorT, TransformerAffine<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, affine, aSizeSrc);

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
                                                        InterpolatorT, TransformerAffine<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, affine, aSizeSrc);

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
                                                        InterpolatorT, TransformerAffine<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, affine, aSizeSrc);

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
                                                        InterpolatorT, TransformerAffine<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, affine, aSizeSrc);

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
                                                        InterpolatorT, TransformerAffine<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, affine, aSizeSrc);

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
                                                        InterpolatorT, TransformerAffine<CoordT>, roundingMode>;
                    const FunctorT functor(interpol, affine, aSizeSrc);

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
                                                        InterpolatorT, TransformerAffine<CoordT>, roundingMode>;
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
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for WarpAffine.");
                break;
        }
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
