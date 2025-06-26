#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/channel.h>
#include <common/image/filterArea.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/mpp_defs.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcT, typename DstT>
void InvokeBlackHat(const SrcT *aSrcFromErode, size_t aPitchSrcFromErode, const SrcT *aSrcOrig, size_t aPitchSrcOrig,
                    DstT *aDst, size_t aPitchDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                    BorderType aBorderType, const SrcT &aConstant, const Size2D &aAllowedReadRoiSize,
                    const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,
                    const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
