#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/mpp_defs.h>
#include <common/vector_typetraits.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT, typename ComputeT = default_floating_compute_type_for_t<SrcT>, typename DstT>
void InvokeAlphaCompCSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                            size_t aPitchDst, const remove_vector_t<SrcT> &aAlpha1,
                            const remove_vector_t<SrcT> &aAlpha2, AlphaCompositionOp aAlphaOp, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
