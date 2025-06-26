#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/channel.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/mpp_defs.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcT, typename DstT>
void InvokeCrossCorrelationCoefficient(const SrcT *aSrc1, size_t aPitchSrc1, const Pixel32fC2 *aSrcBoxFiltered,
                                       size_t aPitchSrcBoxFiltered, DstT *aDst, size_t aPitchDst, const SrcT *aTemplate,
                                       size_t aPitchTemplate, const Size2D &aSizeTemplate,
                                       const Pixel64fC1 *aMeanTemplate, BorderType aBorderType, const SrcT &aConstant,
                                       const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi,
                                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
