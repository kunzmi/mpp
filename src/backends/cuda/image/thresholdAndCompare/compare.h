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
template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeCompareSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                         size_t aPitchDst, CompareOp aCompare, const Size2D &aSize,
                         const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeCompareSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                       CompareOp aCompare, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeCompareSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                          CompareOp aCompare, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
