#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/channelList.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/vector_typetraits.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{

template <typename SrcT, typename DstT>
void InvokeSwapChannelSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                          const ChannelList<vector_active_size_v<DstT>> &aDstChannels, const Size2D &aSize,
                          const opp::cuda::StreamCtx &aStreamCtx);

template <ThreeChannel SrcT, FourChannelNoAlpha DstT>
void InvokeSwapChannelSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                          const ChannelList<vector_active_size_v<DstT>> &aDstChannels, remove_vector_t<DstT> aValue,
                          const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
