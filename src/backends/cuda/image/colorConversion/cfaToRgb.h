#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/mpp_defs.h>

namespace mpp::image::cuda
{
template <typename SrcT>
void InvokeCfaToRgbSrc(const SrcT *aSrc1, size_t aPitchSrc1, Vector3<remove_vector_t<SrcT>> *aDst, size_t aPitchDst,
                       BayerGridPosition aBayerGrid, const Vector2<int> &aAllowedReadRoiOffset,
                       const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeCfaToRgbSrc(const SrcT *aSrc1, size_t aPitchSrc1, Vector4<remove_vector_t<SrcT>> *aDst, size_t aPitchDst,
                       remove_vector_t<SrcT> aAlpha, BayerGridPosition aBayerGrid,
                       const Vector2<int> &aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize,
                       const Size2D &aSizeSrc, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
