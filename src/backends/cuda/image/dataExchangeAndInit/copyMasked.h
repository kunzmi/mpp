#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT, typename DstT>
void InvokeCopyMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                    size_t aPitchDst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
