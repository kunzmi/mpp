#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{

template <typename SrcT>
void InvokeMaxIdxMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                           SrcT *aTempBufferMax, same_vector_size_different_type_t<SrcT, int> *aTempMaxIdxX,
                           SrcT *aDstMax, same_vector_size_different_type_t<SrcT, int> *aDstMaxIdxX,
                           same_vector_size_different_type_t<SrcT, int> *aDstMaxIdxY,
                           remove_vector_t<SrcT> *aDstScalarMax, Vector3<int> *aDstScalarIdxMax, const Size2D &aSize,
                           const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
