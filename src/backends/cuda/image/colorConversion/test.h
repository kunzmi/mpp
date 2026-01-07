#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcT, typename ComputeT = same_vector_size_different_type_t<SrcT, float>, typename DstT>
void InvokeTestSrc(const SrcT *aSrc1, size_t aPitchSrc1, Vector2<remove_vector_t<DstT>> *aDst, size_t aPitchDst,
                   const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename DstT, typename ComputeT = same_vector_size_different_type_t<DstT, float>>
void InvokeTestInplace(DstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
