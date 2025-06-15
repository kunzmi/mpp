#if OPP_ENABLE_CUDA_BACKEND

#include "alphaComp.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/image/functors/srcSrcFunctor.h>
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
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAlphaCompCSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                            size_t aPitchDst, remove_vector_t<SrcT> aAlpha1, remove_vector_t<SrcT> aAlpha2,
                            AlphaCompositionOp aAlphaOp, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        remove_vector_t<ComputeT> alpha1 = static_cast<remove_vector_t<ComputeT>>(aAlpha1);
        remove_vector_t<ComputeT> alpha2 = static_cast<remove_vector_t<ComputeT>>(aAlpha2);

        if constexpr (RealIntVector<SrcT>)
        {
            alpha1 /= static_cast<remove_vector_t<ComputeT>>(numeric_limits<remove_vector_t<SrcT>>::max());
            alpha2 /= static_cast<remove_vector_t<ComputeT>>(numeric_limits<remove_vector_t<SrcT>>::max());
        }

        constexpr RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero;

        switch (aAlphaOp)
        {
            case opp::AlphaCompositionOp::Over:
            {
                using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::Over>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
                const AlphaCompOp op(alpha1, alpha2);
                const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::AlphaCompositionOp::In:
            {
                using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::In>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
                const AlphaCompOp op(alpha1, alpha2);
                const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::AlphaCompositionOp::Out:
            {
                using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::Out>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
                const AlphaCompOp op(alpha1, alpha2);
                const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::AlphaCompositionOp::ATop:
            {
                using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::ATop>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
                const AlphaCompOp op(alpha1, alpha2);
                const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::AlphaCompositionOp::XOr:
            {
                using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::XOr>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
                const AlphaCompOp op(alpha1, alpha2);
                const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::AlphaCompositionOp::Plus:
            {
                using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::Plus>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
                const AlphaCompOp op(alpha1, alpha2);
                const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::AlphaCompositionOp::OverPremul:
            {
                using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::OverPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
                const AlphaCompOp op(alpha1, alpha2);
                const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::AlphaCompositionOp::InPremul:
            {
                using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::InPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
                const AlphaCompOp op(alpha1, alpha2);
                const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::AlphaCompositionOp::OutPremul:
            {
                using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::OutPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
                const AlphaCompOp op(alpha1, alpha2);
                const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::AlphaCompositionOp::ATopPremul:
            {
                using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::ATopPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
                const AlphaCompOp op(alpha1, alpha2);
                const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::AlphaCompositionOp::XOrPremul:
            {
                using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::XOrPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
                const AlphaCompOp op(alpha1, alpha2);
                const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::AlphaCompositionOp::PlusPremul:
            {
                using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::PlusPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
                const AlphaCompOp op(alpha1, alpha2);
                const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeAlphaCompCSrcSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, remove_vector_t<typeSrcIsTypeDst> aAlpha1,                           \
        remove_vector_t<typeSrcIsTypeDst> aAlpha2, AlphaCompositionOp aAlphaOp, const Size2D &aSize,                   \
        const opp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
