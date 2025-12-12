#pragma once
#include "statisticsTypes.h"
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT1, typename DstT2>
void InvokeMeanStdSrc(const SrcT *aSrc1, size_t aPitchSrc1, ComputeT *aTempBuffer1, ComputeT *aTempBuffer2,
                      DstT1 *aDst1, DstT2 *aDst2, remove_vector_t<DstT1> *aDstScalar1,
                      remove_vector_t<DstT2> *aDstScalar2, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
