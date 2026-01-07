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
void InvokeColorTwist3x3Src(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                            const Matrix<float> &aTwist, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                            size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, const Matrix<float> &aTwist,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                            size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                            Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, const Matrix<float> &aTwist,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                            Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                            Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, const Matrix<float> &aTwist,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                            size_t aPitchDst1, const Matrix<float> &aTwist, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                            size_t aPitchDst1, remove_vector_t<SrcDstT> aAlpha, const Matrix<float> &aTwist,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4, SrcDstT *aDst1,
                            size_t aPitchDst1, const Matrix<float> &aTwist, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Inplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, const Matrix<float> &aTwist,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to422(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector2<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to422(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to422(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to422(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector2<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to422(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to422(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to420(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to420(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to420(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to420(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to411(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to411(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to411(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src444to411(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src422to444(const Vector2<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize, bool aSwapLumaChroma,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src422to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src422to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src422to444(const Vector2<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1, SrcDstT *aDst,
                                    size_t aPitchDst, const Matrix<float> &aTwist, const Size2D &aSize,
                                    bool aSwapLumaChroma, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src422to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2, SrcDstT *aDst,
                                    size_t aPitchDst, const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src422to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst,
                                    size_t aPitchDst, const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src420to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src420to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src420to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2, SrcDstT *aDst,
                                    size_t aPitchDst, const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src420to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst,
                                    size_t aPitchDst, const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src411to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src411to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src411to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2, SrcDstT *aDst,
                                    size_t aPitchDst, const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeColorTwist3x3Src411to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst,
                                    size_t aPitchDst, const Matrix<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
