#if OPP_ENABLE_CUDA_BACKEND

#include "transpose.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/transposeKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
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
template <typename SrcDstT>
void InvokeTransposeSrc(const SrcDstT *aSrc, size_t aPitchSrc, SrcDstT *aDst, size_t aPitchDst, const Size2D &aSizeDst,
                        const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcDstT> && oppEnableCudaBackend<SrcDstT>)
    {
        using SrcT = SrcDstT;
        using DstT = SrcDstT;
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        InvokeTransposeKernelDefault<SrcDstT>(aSrc, aPitchSrc, aDst, aPitchDst, aSizeDst, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeTransposeSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                 \
                                                       typeSrcIsTypeDst *aDst, size_t aPitchDst,                       \
                                                       const Size2D &aSizeDst, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

ForAllChannelsNoAlpha(8u);
ForAllChannelsNoAlpha(8s);

ForAllChannelsNoAlpha(16u);
ForAllChannelsNoAlpha(16s);

ForAllChannelsNoAlpha(32u);
ForAllChannelsNoAlpha(64u);
ForAllChannelsNoAlpha(32s);
ForAllChannelsNoAlpha(64s); // needed for integral and sqrIntegral

ForAllChannelsNoAlpha(16f);
ForAllChannelsNoAlpha(16bf);
ForAllChannelsNoAlpha(32f);
ForAllChannelsNoAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
