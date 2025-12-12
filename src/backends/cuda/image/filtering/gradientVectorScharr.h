#pragma once
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
void InvokeGradientVectorScharr(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDstX, size_t aPitchDstX, DstT *aDstY,
                                size_t aPitchDstY, DstT *aDstMag, size_t aPitchDstMag, Pixel32fC1 *aDstAngle,
                                size_t aPitchDstAngle, Pixel32fC4 *aDstCovariance, size_t aPitchDstCovariance,
                                Norm aNorm, MaskSize aMaskSize, BorderType aBorderType, const SrcT &aConstant,
                                const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
