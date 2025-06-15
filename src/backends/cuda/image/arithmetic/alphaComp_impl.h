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
void InvokeAlphaCompSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                           size_t aPitchDst, AlphaCompositionOp aAlphaOp, const Size2D &aSize,
                           const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if constexpr (RealIntVector<SrcT>)
        {
            constexpr remove_vector_t<ComputeT> alphaScaleVal =
                static_cast<remove_vector_t<ComputeT>>(numeric_limits<remove_vector_t<SrcT>>::max());
            constexpr remove_vector_t<ComputeT> alphaScaleValInv =
                static_cast<remove_vector_t<ComputeT>>(1) / alphaScaleVal;

            switch (aAlphaOp)
            {
                case opp::AlphaCompositionOp::Over:
                {
                    using AlphaCompOp     = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       opp::AlphaCompositionOp::Over>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::In:
                {
                    using AlphaCompOp     = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       opp::AlphaCompositionOp::In>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::Out:
                {
                    using AlphaCompOp     = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       opp::AlphaCompositionOp::Out>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::ATop:
                {
                    using AlphaCompOp     = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       opp::AlphaCompositionOp::ATop>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::XOr:
                {
                    using AlphaCompOp     = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       opp::AlphaCompositionOp::XOr>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::Plus:
                {
                    using AlphaCompOp     = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       opp::AlphaCompositionOp::Plus>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::OverPremul:
                {
                    using AlphaCompOp     = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       opp::AlphaCompositionOp::OverPremul>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::InPremul:
                {
                    using AlphaCompOp     = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       opp::AlphaCompositionOp::InPremul>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::OutPremul:
                {
                    using AlphaCompOp     = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       opp::AlphaCompositionOp::OutPremul>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::ATopPremul:
                {
                    using AlphaCompOp     = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       opp::AlphaCompositionOp::ATopPremul>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::XOrPremul:
                {
                    using AlphaCompOp     = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       opp::AlphaCompositionOp::XOrPremul>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::PlusPremul:
                {
                    using AlphaCompOp     = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       opp::AlphaCompositionOp::PlusPremul>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
            }
        }
        else
        {
            switch (aAlphaOp)
            {
                case opp::AlphaCompositionOp::Over:
                {
                    using AlphaCompOp = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::Over>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::In:
                {
                    using AlphaCompOp = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::In>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::Out:
                {
                    using AlphaCompOp = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::Out>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::ATop:
                {
                    using AlphaCompOp = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::ATop>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::XOr:
                {
                    using AlphaCompOp = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::XOr>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::Plus:
                {
                    using AlphaCompOp = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::Plus>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::OverPremul:
                {
                    using AlphaCompOp = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::OverPremul>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::InPremul:
                {
                    using AlphaCompOp = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::InPremul>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::OutPremul:
                {
                    using AlphaCompOp = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::OutPremul>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::ATopPremul:
                {
                    using AlphaCompOp = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::ATopPremul>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::XOrPremul:
                {
                    using AlphaCompOp = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::XOrPremul>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case opp::AlphaCompositionOp::PlusPremul:
                {
                    using AlphaCompOp = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::PlusPremul>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
            }
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeAlphaCompSrcSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, AlphaCompositionOp aAlphaOp, const Size2D &aSize,                    \
        const opp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
