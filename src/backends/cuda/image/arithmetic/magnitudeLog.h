#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeMagnitudeLogSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                           remove_vector_t<DstT> aOffset, const Size2D &aSizeSrc, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
