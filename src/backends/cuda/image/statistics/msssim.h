#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT, typename DstT>
void InvokeMSSSIMSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aTempBuffer,
                        size_t aPitchTempBuffer, DstT *aTempBufferAvg, DstT *aDst, int aIteration,
                        remove_vector_t<DstT> aDynamicRange, remove_vector_t<DstT> aK1, remove_vector_t<DstT> aK2,
                        const Size2D &aAllowedReadRoiSize1, const Vector2<int> &aOffsetToActualRoi1,
                        const Size2D &aAllowedReadRoiSize2, const Vector2<int> &aOffsetToActualRoi2,
                        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
