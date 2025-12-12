#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

// No SIMD as we operate inter-vector, i.e. we multiply channel 1-3 with channel 4

template <typename SrcT, typename ComputeT = default_floating_compute_type_for_t<SrcT>, typename DstT>
void InvokeAlphaPremulSrc(const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT, typename ComputeT = default_floating_compute_type_for_t<SrcDstT>>
void InvokeAlphaPremulInplace(SrcDstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize,
                              const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_floating_compute_type_for_t<SrcT>, typename DstT>
void InvokeAlphaPremulACSrc(const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst,
                            remove_vector_t<SrcT> aAlpha, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT, typename ComputeT = default_floating_compute_type_for_t<SrcDstT>>
void InvokeAlphaPremulACInplace(SrcDstT *aSrcDst, size_t aPitchSrcDst, remove_vector_t<SrcDstT> aAlpha,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
