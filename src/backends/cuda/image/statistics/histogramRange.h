#pragma once
#include "statisticsTypes.h"
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT>
size_t InvokeHistogramRangeGetBufferSize(const SrcT *aSrc1, size_t aPitchSrc1,
                                         const int aNumLevels[vector_active_size<SrcT>::value], const Size2D &aSize,
                                         const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeHistogramRange(const SrcT *aSrc1, size_t aPitchSrc1, void *aTempBuffer, size_t aTempBufferSize,
                          int *aHist[vector_active_size<SrcT>::value],
                          const int aNumLevels[vector_active_size<SrcT>::value],
                          const hist_range_types_for_t<SrcT> *aLevels[vector_active_size<SrcT>::value],
                          const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
