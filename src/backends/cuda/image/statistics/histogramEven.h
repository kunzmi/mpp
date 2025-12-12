#pragma once
#include "statisticsTypes.h"
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT, typename LevelT = hist_even_level_types_for_t<SrcT>>
size_t InvokeHistogramEvenGetBufferSize(const SrcT *aSrc1, size_t aPitchSrc1,
                                        const int aLevels[vector_active_size<SrcT>::value], const Size2D &aSize,
                                        const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename LevelT>
void InvokeHistogramEven(const SrcT *aSrc1, size_t aPitchSrc1, void *aTempBuffer, size_t aTempBufferSize,
                         int *aHist[vector_active_size<SrcT>::value],
                         const int aLevels[vector_active_size<SrcT>::value], const LevelT &aLowerLevel,
                         const LevelT &aUpperLevel, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
