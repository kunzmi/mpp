#include "setChannel.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/setChannelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/constantFunctor.h>
#include <common/image/functors/devConstantFunctor.h>
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
template <typename DstT>
void InvokeSetChannelC(remove_vector_t<DstT> aConst, Channel aChannel, DstT *aDst, size_t aPitchDst,
                       const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

    InvokeSetChannelKernelDefault<DstT>(aDst, aPitchDst, aConst, aChannel, aSize, aStreamCtx);
}

#pragma region Instantiate

#define InstantiateInvokeSetChannelC_For(typeDst)                                                                      \
    template void InvokeSetChannelC<typeDst>(remove_vector_t<typeDst> aConst, Channel aChannel, typeDst * aDst,        \
                                             size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSetChannelC(type)                                                                   \
    InstantiateInvokeSetChannelC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeSetChannelC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeSetChannelC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSetChannelC(type)                                                                 \
    InstantiateInvokeSetChannelC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeSetChannelC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeSetChannelC_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeSetChannelC_For(Pixel##type##C4A);

#pragma endregion

template <typename DstT>
void InvokeSetChannelDevC(const remove_vector_t<DstT> *aConst, Channel aChannel, DstT *aDst, size_t aPitchDst,
                          const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

    InvokeSetChannelKernelDefault<DstT>(aDst, aPitchDst, aConst, aChannel, aSize, aStreamCtx);
}

#pragma region Instantiate

#define InstantiateInvokeSetChannelDevC_For(typeDst)                                                                   \
    template void InvokeSetChannelDevC<typeDst>(const remove_vector_t<typeDst> *aConst, Channel aChannel,              \
                                                typeDst *aDst, size_t aPitchDst, const Size2D &aSize,                  \
                                                const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSetChannelDevC(type)                                                                \
    InstantiateInvokeSetChannelDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeSetChannelDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeSetChannelDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeSetChannelDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSetChannelDevC(type)                                                              \
    InstantiateInvokeSetChannelDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeSetChannelDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeSetChannelDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeSetChannelDevC_For(Pixel##type##C4);                                                              \
    InstantiateInvokeSetChannelDevC_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
