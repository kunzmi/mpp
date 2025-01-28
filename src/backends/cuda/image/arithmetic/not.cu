#if OPP_ENABLE_CUDA_BACKEND

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
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeNotSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                  const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using notSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Not<ComputeT>, RoundingMode::None>;

        Not<ComputeT> op;

        notSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, notSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using SrcT for computeT
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeNotSrc<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                                  \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8s);
ForAllChannelsWithAlpha(8u);

ForAllChannelsWithAlpha(16s);
ForAllChannelsWithAlpha(16u);

ForAllChannelsWithAlpha(32s);
ForAllChannelsWithAlpha(32u);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename DstT>
void InvokeNotInplace(DstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using notInplace = InplaceFunctor<TupelSize, DstT, DstT, opp::Not<DstT>, RoundingMode::None>;

        Not<DstT> op;

        notInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, notInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define Instantiate_For(typeDst)                                                                                       \
    template void InvokeNotInplace<typeDst>(typeDst * aSrcDst, size_t aPitchSrcDst, const Size2D &aSize,               \
                                            const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8s);
ForAllChannelsWithAlpha(8u);

ForAllChannelsWithAlpha(16s);
ForAllChannelsWithAlpha(16u);

ForAllChannelsWithAlpha(32s);
ForAllChannelsWithAlpha(32u);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
