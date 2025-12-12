#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT>
void InvokeMaxMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                        SrcT *aTempBuffer, SrcT *aDst, remove_vector_t<SrcT> *aDstScalar, const Size2D &aSize,
                        const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
