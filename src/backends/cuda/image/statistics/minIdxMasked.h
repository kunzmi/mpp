#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT>
void InvokeMinIdxMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                           SrcT *aTempBufferMin, same_vector_size_different_type_t<SrcT, int> *aTempMinIdxX,
                           SrcT *aDstMin, same_vector_size_different_type_t<SrcT, int> *aDstMinIdxX,
                           same_vector_size_different_type_t<SrcT, int> *aDstMinIdxY,
                           remove_vector_t<SrcT> *aDstScalarMin, Vector3<int> *aDstScalarIdxMin, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
