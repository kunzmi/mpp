#pragma once
#include "statisticsTypes.h"
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeQualityIndexSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2,
                              ComputeT *aTempBuffer1, ComputeT *aTempBuffer2, ComputeT *aTempBuffer3,
                              ComputeT *aTempBuffer4, ComputeT *aTempBuffer5, DstT *aDst, const Size2D &aSize,
                              const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
