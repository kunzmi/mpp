#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcDstT>
void InvokeLShiftSrcC(const SrcDstT *aSrc, size_t aPitchSrc, uint aConst, SrcDstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLShiftInplaceC(SrcDstT *aSrcDst, size_t aPitchSrcDst, uint aConst, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
