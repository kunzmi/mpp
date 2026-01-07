#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcDstT>
void InvokeRGBtoLUVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeRGBtoLUVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeRGBtoLUVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeRGBtoLUVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                       Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeRGBtoLUVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeRGBtoLUVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                       Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeBGRtoLUVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeBGRtoLUVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeBGRtoLUVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeBGRtoLUVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                       Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeBGRtoLUVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeBGRtoLUVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                       Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeRGBtoLUVInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoRGBInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeBGRtoLUVInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLUVtoBGRInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
