#if MPP_ENABLE_CUDA_BACKEND

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
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAlphaCompSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                           size_t aPitchDst, AlphaCompositionOp aAlphaOp, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if constexpr (RealIntVector<SrcT>)
        {
            constexpr remove_vector_t<ComputeT> alphaScaleVal =
                static_cast<remove_vector_t<ComputeT>>(numeric_limits<remove_vector_t<SrcT>>::max());
            constexpr remove_vector_t<ComputeT> alphaScaleValInv =
                static_cast<remove_vector_t<ComputeT>>(1) / alphaScaleVal;

            switch (aAlphaOp)
            {
                case mpp::AlphaCompositionOp::Over:
                {
                    using AlphaCompOp     = mpp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       mpp::AlphaCompositionOp::Over>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::In:
                {
                    using AlphaCompOp     = mpp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       mpp::AlphaCompositionOp::In>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::Out:
                {
                    using AlphaCompOp     = mpp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       mpp::AlphaCompositionOp::Out>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::ATop:
                {
                    using AlphaCompOp     = mpp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       mpp::AlphaCompositionOp::ATop>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::XOr:
                {
                    using AlphaCompOp     = mpp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       mpp::AlphaCompositionOp::XOr>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::Plus:
                {
                    using AlphaCompOp     = mpp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       mpp::AlphaCompositionOp::Plus>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::OverPremul:
                {
                    using AlphaCompOp     = mpp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       mpp::AlphaCompositionOp::OverPremul>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::InPremul:
                {
                    using AlphaCompOp     = mpp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       mpp::AlphaCompositionOp::InPremul>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::OutPremul:
                {
                    using AlphaCompOp     = mpp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       mpp::AlphaCompositionOp::OutPremul>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::ATopPremul:
                {
                    using AlphaCompOp     = mpp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       mpp::AlphaCompositionOp::ATopPremul>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::XOrPremul:
                {
                    using AlphaCompOp     = mpp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       mpp::AlphaCompositionOp::XOrPremul>;
                    using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::PlusPremul:
                {
                    using AlphaCompOp     = mpp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                                       mpp::AlphaCompositionOp::PlusPremul>;
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
                case mpp::AlphaCompositionOp::Over:
                {
                    using AlphaCompOp = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::Over>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::In:
                {
                    using AlphaCompOp = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::In>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::Out:
                {
                    using AlphaCompOp = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::Out>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::ATop:
                {
                    using AlphaCompOp = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::ATop>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::XOr:
                {
                    using AlphaCompOp = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::XOr>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::Plus:
                {
                    using AlphaCompOp = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::Plus>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::OverPremul:
                {
                    using AlphaCompOp = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::OverPremul>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::InPremul:
                {
                    using AlphaCompOp = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::InPremul>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::OutPremul:
                {
                    using AlphaCompOp = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::OutPremul>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::ATopPremul:
                {
                    using AlphaCompOp = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::ATopPremul>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::XOrPremul:
                {
                    using AlphaCompOp = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::XOrPremul>;
                    using alphaCompSrcSrc =
                        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                    const AlphaCompOp op;
                    const alphaCompSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaCompSrcSrc>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::AlphaCompositionOp::PlusPremul:
                {
                    using AlphaCompOp = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::PlusPremul>;
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
// using default_floating_compute_type_for_t for computeT
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeAlphaCompSrcSrc<typeSrcIsTypeDst, default_floating_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(  \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, AlphaCompositionOp aAlphaOp, const Size2D &aSize,                    \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
