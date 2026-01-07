#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/channelList.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/vector_typetraits.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT, typename DstT>
void InvokeSwapChannelSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                          const ChannelList<vector_active_size_v<DstT>> &aDstChannels, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

template <ThreeChannel SrcT, FourChannelNoAlpha DstT>
void InvokeSwapChannelSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                          const ChannelList<vector_active_size_v<DstT>> &aDstChannels, remove_vector_t<DstT> aValue,
                          const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeSwapChannelInplace(SrcT *aSrcDst, size_t aPitchSrcDst,
                              const ChannelList<vector_active_size_v<SrcT>> &aDstChannels, const Size2D &aSize,
                              const mpp::cuda::StreamCtx &aStreamCtx);

template <TwoChannel SrcDstT>
void InvokeSwapChannelSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

template <TwoChannel SrcDstT>
void InvokeSwapChannelInplace(SrcDstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize,
                              const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
