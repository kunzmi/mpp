#if MPP_ENABLE_CUDA_BACKEND

#include "sqr.h"
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
void InvokeSqrSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                  const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using sqrSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sqr<ComputeT>, RoundingMode::None>;

        const mpp::Sqr<ComputeT> op;

        const sqrSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, sqrSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSqrSrc_For(typeSrcIsTypeDst)                                                                  \
    template void InvokeSqrSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(      \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSqrSrc(type)                                                                        \
    InstantiateInvokeSqrSrc_For(Pixel##type##C1);                                                                      \
    InstantiateInvokeSqrSrc_For(Pixel##type##C2);                                                                      \
    InstantiateInvokeSqrSrc_For(Pixel##type##C3);                                                                      \
    InstantiateInvokeSqrSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSqrSrc(type)                                                                      \
    InstantiateInvokeSqrSrc_For(Pixel##type##C1);                                                                      \
    InstantiateInvokeSqrSrc_For(Pixel##type##C2);                                                                      \
    InstantiateInvokeSqrSrc_For(Pixel##type##C3);                                                                      \
    InstantiateInvokeSqrSrc_For(Pixel##type##C4);                                                                      \
    InstantiateInvokeSqrSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename DstT, typename ComputeT>
void InvokeSqrInplace(DstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_COMPUTE_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using sqrInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::Sqr<ComputeT>, RoundingMode::None>;

        const mpp::Sqr<ComputeT> op;

        const sqrInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, sqrInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSqrInplace_For(typeSrcIsTypeDst)                                                              \
    template void InvokeSqrInplace<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>>(                    \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSqrInplace(type)                                                                    \
    InstantiateInvokeSqrInplace_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeSqrInplace_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeSqrInplace_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeSqrInplace_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSqrInplace(type)                                                                  \
    InstantiateInvokeSqrInplace_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeSqrInplace_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeSqrInplace_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeSqrInplace_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeSqrInplace_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
