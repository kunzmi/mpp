#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT>
void InvokeMaxIdxSrc(const SrcT *aSrc, size_t aPitchSrc, SrcT *aTempBufferMax,
                     same_vector_size_different_type_t<SrcT, int> *aTempMaxIdxX, SrcT *aDstMax,
                     same_vector_size_different_type_t<SrcT, int> *aDstMaxIdxX,
                     same_vector_size_different_type_t<SrcT, int> *aDstMaxIdxY, remove_vector_t<SrcT> *aDstScalarMax,
                     Vector3<int> *aDstScalarIdxMax, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
