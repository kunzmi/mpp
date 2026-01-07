#pragma once
#include "statisticsTypes.h"
#include <backends/cuda/streamCtx.h>
#include <common/image/affineTransformation.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT>
void InvokeRadialProfileSrc(const SrcT *aSrc1, size_t aPitchSrc1, int *aProfileCount,
                            same_vector_size_different_type_t<SrcT, float> *aProfileSum,
                            same_vector_size_different_type_t<SrcT, float> *aProfileSumSqr, int aProfileSize,
                            const AffineTransformation<float> &aTransformation, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
