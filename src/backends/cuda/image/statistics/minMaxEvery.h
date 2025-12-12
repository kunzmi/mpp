#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeMinEverySrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeMinEveryInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeMaxEverySrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeMaxEveryInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
