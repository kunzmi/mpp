#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcDstT>
void InvokeGammaCorrBT709Src(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                             float aNormFactor, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaCorrBT709Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                             const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                             Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                             Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2, float aNormFactor,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaCorrBT709Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                             const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                             const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                             Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                             Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                             Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormFactor,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaCorrBT709Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                             const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                             const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                             const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                             Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                             Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                             Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                             Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormFactor,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaCorrBT709Inplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormFactor, const Size2D &aSize,
                                 const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaInvCorrBT709Src(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                                float aNormFactor, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaInvCorrBT709Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2, float aNormFactor,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaInvCorrBT709Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormFactor,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaInvCorrBT709Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                                Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormFactor,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaInvCorrBT709Inplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormFactor, const Size2D &aSize,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaCorrsRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormFactor,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaCorrsRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                            Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2, float aNormFactor,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaCorrsRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                            Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                            Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormFactor,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaCorrsRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                            Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                            Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                            Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormFactor,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaCorrsRGBInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormFactor, const Size2D &aSize,
                                const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaInvCorrsRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                               float aNormFactor, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaInvCorrsRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                               const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                               Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                               Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2, float aNormFactor,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaInvCorrsRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                               const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                               const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                               Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                               Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                               Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormFactor,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaInvCorrsRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                               const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                               const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                               const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                               Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                               Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                               Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                               Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormFactor,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeGammaInvCorrsRGBInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormFactor, const Size2D &aSize,
                                   const mpp::cuda::StreamCtx &aStreamCtx);
} // namespace mpp::image::cuda
