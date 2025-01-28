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
template <typename SrcT, typename DstT>
void InvokeConvert(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                   const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename DstT>
void InvokeConvertRound(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, RoundingMode aRoundingMode,
                        const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename DstT>
void InvokeConvertScaleRound(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                             RoundingMode aRoundingMode, float aScaleFactor, const Size2D &aSize,
                             const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
