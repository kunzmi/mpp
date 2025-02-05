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
void InvokeThresholdLTSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThreshold, DstT *aDst, size_t aPitchDst,
                           const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdLTSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aThreshold, DstT *aDst, size_t aPitchDst,
                              const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdLTInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThreshold, const Size2D &aSize,
                               const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdLTInplaceDevC(DstT *aSrcDst, size_t aPitchDst, const SrcT *aThreshold, const Size2D &aSize,
                                  const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdGTSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThreshold, DstT *aDst, size_t aPitchDst,
                           const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdGTSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aThreshold, DstT *aDst, size_t aPitchDst,
                              const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdGTInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThreshold, const Size2D &aSize,
                               const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdGTInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aThreshold, const Size2D &aSize,
                                  const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdLTValSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThreshold, const SrcT &aValue,
                              DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                              const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdLTValInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThreshold, const SrcT &aValue,
                                  const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdGTValSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThreshold, const SrcT &aValue,
                              DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                              const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdGTValInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThreshold, const SrcT &aValue,
                                  const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdLTValGTValSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThresholdLT, const SrcT &aValueLT,
                                   const SrcT &aThresholdGT, const SrcT &aValueGT, DstT *aDst, size_t aPitchDst,
                                   const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeThresholdLTValGTValInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThresholdLT,
                                       const SrcT &aValueLT, const SrcT &aThresholdGT, const SrcT &aValueGT,
                                       const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
