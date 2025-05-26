#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/channel.h>
#include <common/image/filterArea.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/opp_defs.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{
template <typename SrcT, typename DstT>
void InvokeCannyEdgeMaxSupression(const SrcT *aSrc1, size_t aPitchSrc1, const Pixel32fC1 *aSrcAngle,
                                  size_t aPitchSrcAngle, DstT *aDst, size_t aPitchDst, SrcT aLowThreshold,
                                  SrcT aHighThreshold, const Size2D &aAllowedReadRoiSize,
                                  const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,
                                  const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
