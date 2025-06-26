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
template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeCompareEqEpsSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                              size_t aPitchDst, complex_basetype_t<remove_vector_t<SrcT>> aEpsilon, const Size2D &aSize,
                              const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeCompareEqEpsSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                            complex_basetype_t<remove_vector_t<SrcT>> aEpsilon, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = SrcT, typename DstT>
void InvokeCompareEqEpsSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                               complex_basetype_t<remove_vector_t<SrcT>> aEpsilon, const Size2D &aSize,
                               const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
