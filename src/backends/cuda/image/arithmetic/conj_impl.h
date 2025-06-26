#if MPP_ENABLE_CUDA_BACKEND

#include "conj.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/unary_operators.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/srcFunctor.h>
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
void InvokeConjSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                   const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using conjSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Conj<ComputeT>, RoundingMode::None>;

        const mpp::Conj<ComputeT> op;

        const conjSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, conjSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define InstantiateInvokeConjSrc_For(typeSrcIsTypeDst)                                                                 \
    template void InvokeConjSrc<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeConjSrc(type)                                                                       \
    InstantiateInvokeConjSrc_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeConjSrc_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeConjSrc_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeConjSrc_For(Pixel##type##C4);

#pragma endregion

template <typename DstT, typename ComputeT>
void InvokeConjInplace(DstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_COMPUTE_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using conjInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::Conj<ComputeT>, RoundingMode::None>;

        const mpp::Conj<ComputeT> op;

        const conjInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, conjInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                      functor);
    }
}

#pragma region Instantiate

#define InstantiateInvokeConjInplace_For(typeSrcIsTypeDst)                                                             \
    template void InvokeConjInplace<typeSrcIsTypeDst, typeSrcIsTypeDst>(                                               \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeConjInplace(type)                                                                   \
    InstantiateInvokeConjInplace_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeConjInplace_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeConjInplace_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeConjInplace_For(Pixel##type##C4);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
