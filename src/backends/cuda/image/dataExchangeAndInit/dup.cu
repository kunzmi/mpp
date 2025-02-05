#if OPP_ENABLE_CUDA_BACKEND

#include "dup.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/unary_operators.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
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
template <typename SrcT, typename DstT>
void InvokeDupSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                  const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using dupSrc = SrcFunctor<TupelSize, SrcT, SrcT, DstT, opp::Dup<SrcT, DstT>, RoundingMode::None>;

        Dup<SrcT, DstT> op;

        dupSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, dupSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeDupSrc<typeSrc, typeDst>(const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst,               \
                                                 size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1, Pixel##type##C2);                                                                 \
    Instantiate_For(Pixel##type##C1, Pixel##type##C3);                                                                 \
    Instantiate_For(Pixel##type##C1, Pixel##type##C4);                                                                 \
    Instantiate_For(Pixel##type##C1, Pixel##type##C4A);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1, Pixel##type##C2);                                                                 \
    Instantiate_For(Pixel##type##C1, Pixel##type##C3);                                                                 \
    Instantiate_For(Pixel##type##C1, Pixel##type##C4);

ForAllChannelsWithAlpha(8s);
ForAllChannelsWithAlpha(8u);

ForAllChannelsWithAlpha(16s);
ForAllChannelsWithAlpha(16u);

ForAllChannelsWithAlpha(32s);
ForAllChannelsWithAlpha(32u);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#undef ForAllChannelsWithAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
