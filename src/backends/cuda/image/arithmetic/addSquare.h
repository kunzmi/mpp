#pragma once
#include "addSquareProductWeightedOutputType.h"
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcT, typename ComputeT = add_spw_output_for_t<SrcT>, typename DstT = add_spw_output_for_t<SrcT>>
void InvokeAddSquareInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
