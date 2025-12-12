#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/channel.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcT, typename DstT>
void InvokeCopy(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                const mpp::cuda::StreamCtx &aStreamCtx);

template <SingleChannel SrcT, typename DstT>
void InvokeCopyChannel(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, Channel aDstChannel,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, SingleChannel DstT>
void InvokeCopyChannel(const SrcT *aSrc1, size_t aPitchSrc1, Channel aSrcChannel, DstT *aDst, size_t aPitchDst,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename DstT>
void InvokeCopyChannel(const SrcT *aSrc1, size_t aPitchSrc1, Channel aSrcChannel, DstT *aDst, size_t aPitchDst,
                       Channel aDstChannel, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <SingleChannel SrcT, TwoChannel DstT>
void InvokeCopyPlanar(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                      size_t aPitchDst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <SingleChannel SrcT, ThreeChannel DstT>
void InvokeCopyPlanar(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, const SrcT *aSrc3,
                      size_t aPitchSrc3, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                      const mpp::cuda::StreamCtx &aStreamCtx);

template <SingleChannel SrcT, FourChannel DstT>
void InvokeCopyPlanar(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, const SrcT *aSrc3,
                      size_t aPitchSrc3, const SrcT *aSrc4, size_t aPitchSrc4, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <TwoChannel SrcT, SingleChannel DstT>
void InvokeCopyPlanar(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst1, size_t aPitchDst1, DstT *aDst2,
                      size_t aPitchDst2, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <ThreeChannel SrcT, SingleChannel DstT>
void InvokeCopyPlanar(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst1, size_t aPitchDst1, DstT *aDst2,
                      size_t aPitchDst2, DstT *aDst3, size_t aPitchDst3, const Size2D &aSize,
                      const mpp::cuda::StreamCtx &aStreamCtx);

template <FourChannel SrcT, SingleChannel DstT>
void InvokeCopyPlanar(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst1, size_t aPitchDst1, DstT *aDst2,
                      size_t aPitchDst2, DstT *aDst3, size_t aPitchDst3, DstT *aDst4, size_t aPitchDst4,
                      const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);
} // namespace mpp::image::cuda
