#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/mpp_defs.h>

namespace mpp::image::cuda
{
template <typename SrcDstT>
void InvokeColorToGraySrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst,
                          size_t aPitchDst, const same_vector_size_different_type_t<SrcDstT, float> &aWeights,
                          const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorToGraySrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                          const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                          Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                          const same_vector_size_different_type_t<SrcDstT, float> &aWeights, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorToGraySrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                          const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                          const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                          Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                          const same_vector_size_different_type_t<SrcDstT, float> &aWeights, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorToGraySrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                          const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                          const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                          const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                          Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                          const same_vector_size_different_type_t<SrcDstT, float> &aWeights, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
