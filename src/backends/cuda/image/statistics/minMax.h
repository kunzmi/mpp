#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT>
void InvokeMinMaxSrc(const SrcT *aSrc1, size_t aPitchSrc1, SrcT *aTempBuffer1, SrcT *aTempBuffer2, SrcT *aDstMin,
                     SrcT *aDstMax, remove_vector_t<SrcT> *aDstMinScalar, remove_vector_t<SrcT> *aDstMaxScalar,
                     const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
