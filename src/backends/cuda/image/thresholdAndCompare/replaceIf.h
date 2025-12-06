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
template <typename SrcDstT>
void InvokeReplaceIfSrcSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, const SrcDstT *aSrc2, size_t aPitchSrc2,
                           const SrcDstT &aValue, SrcDstT *aDst, size_t aPitchDst, CompareOp aCompare,
                           const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeReplaceIfSrcC(const SrcDstT *aSrc, size_t aPitchSrc, const SrcDstT &aConst, const SrcDstT &aValue,
                         SrcDstT *aDst, size_t aPitchDst, CompareOp aCompare, const Size2D &aSize,
                         const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeReplaceIfSrcDevC(const SrcDstT *aSrc, size_t aPitchSrc, const SrcDstT *aConst, const SrcDstT &aValue,
                            SrcDstT *aDst, size_t aPitchDst, CompareOp aCompare, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeReplaceIfSrc(const SrcDstT *aSrc, size_t aPitchSrc, const SrcDstT &aValue, SrcDstT *aDst, size_t aPitchDst,
                        CompareOp aCompare, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeReplaceIfInplaceSrcSrc(SrcDstT *aSrcDst, size_t aPitchSrcDst, const SrcDstT *aSrc2, size_t aPitchSrc2,
                                  const SrcDstT &aValue, CompareOp aCompare, const Size2D &aSize,
                                  const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeReplaceIfInplaceSrcC(SrcDstT *aSrcDst, size_t aPitchSrcDst, const SrcDstT &aConst, const SrcDstT &aValue,
                                CompareOp aCompare, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeReplaceIfInplaceSrcDevC(SrcDstT *aSrcDst, size_t aPitchSrcDst, const SrcDstT *aConst, const SrcDstT &aValue,
                                   CompareOp aCompare, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeReplaceIfInplaceSrc(SrcDstT *aSrcDst, size_t aPitchSrcDst, const SrcDstT &aValue, CompareOp aCompare,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
