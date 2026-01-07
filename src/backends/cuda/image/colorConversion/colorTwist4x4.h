#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/matrix4x4.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcDstT>
void InvokeColorTwist4x4Src(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                            const Matrix4x4<float> &aTwist, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist4x4Src(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                            size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                            Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, const Matrix4x4<float> &aTwist,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist4x4Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                            Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                            Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                            Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, const Matrix4x4<float> &aTwist,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist4x4Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4, SrcDstT *aDst1,
                            size_t aPitchDst1, const Matrix4x4<float> &aTwist, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist4x4Inplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, const Matrix4x4<float> &aTwist,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
