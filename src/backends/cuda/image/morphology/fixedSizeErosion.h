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
void InvokeFixedSizeErosion(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Pixel8uC1 *aMask,
                            MaskSize aMaskSize, const Vector2<int> &aFilterCenter, BorderType aBorderType,
                            const SrcT &aConstant, const Size2D &aAllowedReadRoiSize,
                            const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
