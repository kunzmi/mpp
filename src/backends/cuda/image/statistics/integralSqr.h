#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT, typename ComputeT = SrcT, typename DstT, typename DstSqrT>
void InvokeIntegralSqrSrc(const SrcT *aSrc, size_t aPitchSrc, DstT *aTemp, size_t aPitchTemp, DstSqrT *aTemp2,
                          size_t aPitchTemp2, DstT *aDst, size_t aPitchDst, DstSqrT *aDstSqr, size_t aPitchDstSqr,
                          const DstT &aStartValue, const DstSqrT &aStartValueSqr, const Size2D &aSizeDst,
                          const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
