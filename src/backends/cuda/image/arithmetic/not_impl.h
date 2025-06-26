#if MPP_ENABLE_CUDA_BACKEND

#include "not.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/simd_types.h>
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
void InvokeNotSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                  const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using notSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Not<ComputeT>, RoundingMode::None>;

        const mpp::Not<ComputeT> op;

        const notSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, notSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using SrcT for computeT
#define InstantiateInvokeNotSrc_For(typeSrcIsTypeDst)                                                                  \
    template void InvokeNotSrc<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                                  \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeNotSrc(type)                                                                      \
    InstantiateInvokeNotSrc_For(Pixel##type##C1);                                                                      \
    InstantiateInvokeNotSrc_For(Pixel##type##C2);                                                                      \
    InstantiateInvokeNotSrc_For(Pixel##type##C3);                                                                      \
    InstantiateInvokeNotSrc_For(Pixel##type##C4);                                                                      \
    InstantiateInvokeNotSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename DstT>
void InvokeNotInplace(DstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using notInplace = InplaceFunctor<TupelSize, DstT, DstT, mpp::Not<DstT>, RoundingMode::None>;

        const mpp::Not<DstT> op;

        const notInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, notInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeNotInplace_For(typeDst)                                                                       \
    template void InvokeNotInplace<typeDst>(typeDst * aSrcDst, size_t aPitchSrcDst, const Size2D &aSize,               \
                                            const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeNotInplace(type)                                                                  \
    InstantiateInvokeNotInplace_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeNotInplace_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeNotInplace_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeNotInplace_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeNotInplace_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
