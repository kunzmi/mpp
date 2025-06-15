#if OPP_ENABLE_CUDA_BACKEND

#include "copyBorder.h"
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
void InvokeCopyBorder(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                      const Vector2<int> &aLowerBorderSize, BorderType aBorder, const SrcT &aConstant,
                      const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        constexpr RoundingMode roundingMode = RoundingMode::None;
        using CoordT                        = int;

        const TransformerShift<CoordT> shift(aLowerBorderSize);

        constexpr Vector2<int> roiOffset(0);

        switch (aBorder)
        {
            case opp::BorderType::Constant:
            {
                using BCType = BorderControl<SrcT, BorderType::Constant, false, true, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aSize, roiOffset, aConstant);

                using InterpolatorT = Interpolator<SrcT, BCType, CoordT, InterpolationMode::Undefined>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, false, InterpolatorT,
                                                        TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::BorderType::Replicate:
            {
                using BCType = BorderControl<SrcT, BorderType::Replicate, false, true, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aSize, roiOffset);

                using InterpolatorT = Interpolator<SrcT, BCType, CoordT, InterpolationMode::Undefined>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, false, InterpolatorT,
                                                        TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::BorderType::Mirror:
            {
                using BCType = BorderControl<SrcT, BorderType::Mirror, false, true, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aSize, roiOffset);

                using InterpolatorT = Interpolator<SrcT, BCType, CoordT, InterpolationMode::Undefined>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, false, InterpolatorT,
                                                        TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::BorderType::MirrorReplicate:
            {
                using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, true, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aSize, roiOffset);

                using InterpolatorT = Interpolator<SrcT, BCType, CoordT, InterpolationMode::Undefined>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, false, InterpolatorT,
                                                        TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::BorderType::Wrap:
            {
                using BCType = BorderControl<SrcT, BorderType::Wrap, false, true, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aSize, roiOffset);

                using InterpolatorT = Interpolator<SrcT, BCType, CoordT, InterpolationMode::Undefined>;
                const InterpolatorT interpol(bc);
                using TransformerT = TransformerFunctor<TupelSize, SrcT, CoordT, false, InterpolatorT,
                                                        TransformerShift<CoordT>, roundingMode>;
                const TransformerT functor(interpol, shift, aSize);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, TransformerT>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aBorder, aBorder << " is not a supported border type mode for CopyBorder.");
                break;
        }
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeCopyBorder<typeSrcIsTypeDst, typeSrcIsTypeDst>(                                                \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Vector2<int> &aLowerBorderSize, BorderType aBorder, const typeSrcIsTypeDst &aConstant,                   \
        const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

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
