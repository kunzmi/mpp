#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcDstT>
void InvokeColorCompKeySrcSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, const SrcDstT *aSrc2, size_t aPitchSrc2,
                              const SrcDstT &aValue, SrcDstT *aDst, size_t aPitchDst, const Size2D &aSize,
                              const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorCompKeyInplaceSrcSrc(SrcDstT *aSrcDst, size_t aPitchSrcDst, const SrcDstT *aSrc2, size_t aPitchSrc2,
                                     const SrcDstT &aValue, const Size2D &aSize,
                                     const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
