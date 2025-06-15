#if OPP_ENABLE_CUDA_BACKEND

#include "copySubpix.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/dataExchangeAndInit/operators.h>
#include <common/defines.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/functors/interpolator.h>
#include <common/image/functors/transformer.h>
#include <common/image/functors/transformerFunctor.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
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
template <typename SrcT, typename DstT>
void InvokeCopySubpix(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Pixel32fC2 &aDelta,
                      InterpolationMode aInterpolation, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using CoordT         = float;
        using CoordTInterpol = CoordT;

        // for compatibility with NPP, negate the shift:
        const TransformerShift<CoordT> shift(-aDelta);

        constexpr Vector2<int> roiOffset(0);

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesToEven;

        // for interpolation at the border we will use replicate:
        using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
        const BCType bc(aSrc1, aPitchSrc1, aSize, roiOffset);

        switch (aInterpolation)
        {
            case opp::InterpolationMode::Linear:
            {
                using InterpolatorT =
                    Interpolator<geometry_compute_type_for_t<SrcT>, BCType, CoordTInterpol, InterpolationMode::Linear>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, BCType::only_for_interpolation,
                                                        InterpolatorT, TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::InterpolationMode::CubicHermiteSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<SrcT>, BCType, CoordTInterpol,
                                                   InterpolationMode::CubicHermiteSpline>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, BCType::only_for_interpolation,
                                                        InterpolatorT, TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::InterpolationMode::CubicLagrange:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<SrcT>, BCType, CoordTInterpol,
                                                   InterpolationMode::CubicLagrange>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, BCType::only_for_interpolation,
                                                        InterpolatorT, TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamBSpline:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<SrcT>, BCType, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamBSpline>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, BCType::only_for_interpolation,
                                                        InterpolatorT, TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamCatmullRom:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<SrcT>, BCType, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamCatmullRom>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, BCType::only_for_interpolation,
                                                        InterpolatorT, TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::InterpolationMode::Cubic2ParamB05C03:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<SrcT>, BCType, CoordTInterpol,
                                                   InterpolationMode::Cubic2ParamB05C03>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, BCType::only_for_interpolation,
                                                        InterpolatorT, TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::InterpolationMode::Lanczos2Lobed:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<SrcT>, BCType, CoordTInterpol,
                                                   InterpolationMode::Lanczos2Lobed>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, BCType::only_for_interpolation,
                                                        InterpolatorT, TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::InterpolationMode::Lanczos3Lobed:
            {
                using InterpolatorT = Interpolator<geometry_compute_type_for_t<SrcT>, BCType, CoordTInterpol,
                                                   InterpolationMode::Lanczos3Lobed>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, BCType::only_for_interpolation,
                                                        InterpolatorT, TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolation,
                                      aInterpolation << " is not a supported interpolation mode for CopySubPix.");
                break;
        }
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeCopySubpix<typeSrcIsTypeDst, typeSrcIsTypeDst>(                                                \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Pixel32fC2 &aDelta, InterpolationMode aInterpolation, const Size2D &aSize,                               \
        const opp::cuda::StreamCtx &aStreamCtx);

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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
