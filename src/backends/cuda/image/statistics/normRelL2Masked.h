#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include "normRelL2.h"
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeNormRelL2MaskedSrcSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                                 const SrcT *aSrc2, size_t aPitchSrc2, ComputeT *aTempBuffer1, ComputeT *aTempBuffer2,
                                 DstT *aDst, remove_vector_t<DstT> *aDstScalar, const Size2D &aSize,
                                 const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
