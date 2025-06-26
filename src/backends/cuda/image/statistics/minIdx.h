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
void InvokeMinIdxSrc(const SrcT *aSrc, size_t aPitchSrc, SrcT *aTempBufferMin,
                     same_vector_size_different_type_t<SrcT, int> *aTempMinIdxX, SrcT *aDstMin,
                     same_vector_size_different_type_t<SrcT, int> *aDstMinIdxX,
                     same_vector_size_different_type_t<SrcT, int> *aDstMinIdxY, remove_vector_t<SrcT> *aDstScalarMin,
                     Vector3<int> *aDstScalarIdxMin, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
