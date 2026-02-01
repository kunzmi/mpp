#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/matrix.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcDstT>
void InvokeSampling422ConversionC2P2Src(const Vector2<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                        bool aSwapLumaChroma, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                        size_t aPitchDst1, Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeSampling422ConversionC2P3Src(const Vector2<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                        bool aSwapLumaChroma, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                        size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                        Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeSampling422ConversionP2C2Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                        const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                        Vector2<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                        bool aSwapLumaChroma, const Size2D &aSize,
                                        const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeSampling422ConversionP3C2Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                        const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                        const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                        Vector2<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                        bool aSwapLumaChroma, const Size2D &aSize,
                                        const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
