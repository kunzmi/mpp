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

// No SIMD as we operate inter-vector, i.e. we multiply channel 1-3 with channel 4

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAlphaPremulSrc(const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT, typename ComputeT = default_compute_type_for_t<SrcDstT>>
void InvokeAlphaPremulInplace(SrcDstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize,
                              const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAlphaPremulACSrc(const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst,
                            remove_vector_t<SrcT> aAlpha, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT, typename ComputeT = default_compute_type_for_t<SrcDstT>>
void InvokeAlphaPremulACInplace(SrcDstT *aSrcDst, size_t aPitchSrcDst, remove_vector_t<SrcDstT> aAlpha,
                                const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
