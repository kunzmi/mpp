#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/channel.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcT, typename DstT>
void InvokeCopyBorder(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                      const Vector2<int> &aLowerBorderSize, BorderType aBorder, const SrcT &aConstant,
                      const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);
} // namespace mpp::image::cuda
