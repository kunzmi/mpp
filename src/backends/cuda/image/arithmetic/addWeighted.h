#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include "addSquareProductWeightedOutputType.h"
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/vector_typetraits.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{
template <typename SrcT, typename ComputeT = add_spw_output_for_t<SrcT>, typename DstT = add_spw_output_for_t<SrcT>>
void InvokeAddWeightedSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                             size_t aPitchDst, remove_vector_t<ComputeT> aAlpha, const Size2D &aSize,
                             const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = add_spw_output_for_t<SrcT>, typename DstT = add_spw_output_for_t<SrcT>>
void InvokeAddWeightedInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                                 remove_vector_t<ComputeT> aAlpha, const Size2D &aSize,
                                 const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
