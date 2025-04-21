#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include "meanStd.h"
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{

template <typename SrcT, typename ComputeT, typename DstT1, typename DstT2>
void InvokeMeanStdMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                            ComputeT *aTempBuffer1, ComputeT *aTempBuffer2, ulong64 *aMaskBuffer, DstT1 *aDst1,
                            DstT2 *aDst2, remove_vector_t<DstT1> *aDstScalar1, remove_vector_t<DstT2> *aDstScalar2,
                            const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
