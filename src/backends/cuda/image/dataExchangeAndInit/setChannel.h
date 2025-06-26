#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/channel.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename DstT>
void InvokeSetChannelC(remove_vector_t<DstT> aConst, Channel aChannel, DstT *aDst, size_t aPitchDst,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename DstT>
void InvokeSetChannelDevC(const remove_vector_t<DstT> *aConst, Channel aChannel, DstT *aDst, size_t aPitchDst,
                          const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
