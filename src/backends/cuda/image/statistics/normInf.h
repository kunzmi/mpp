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
void InvokeNormInfSrc(const SrcT *aSrc1, size_t aPitchSrc1, ComputeT *aTempBuffer, DstT *aDst,
                      remove_vector_t<DstT> *aDstScalar, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
