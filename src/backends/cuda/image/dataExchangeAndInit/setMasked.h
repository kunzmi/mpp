#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename DstT>
void InvokeSetCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const DstT &aConst, DstT *aDst, size_t aPitchDst,
                    const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename DstT>
void InvokeSetDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const DstT *aConst, DstT *aDst, size_t aPitchDst,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
