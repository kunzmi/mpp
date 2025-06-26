#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/defines.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/matrix.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcT>
void InvokePerspectiveBackSrc(const SrcT *aSrc1, size_t aPitchSrc1, SrcT *aDst, size_t aPitchDst,
                              const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation,
                              BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                              const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                              const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokePerspectiveBackSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                              const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                              Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1,
                              Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2,
                              const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation,
                              BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                              const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                              const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokePerspectiveBackSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                              const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                              const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3,
                              Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1,
                              Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2,
                              Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3,
                              const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation,
                              BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                              const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                              const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokePerspectiveBackSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1, //
                              const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2, //
                              const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3, //
                              const Vector1<remove_vector_t<SrcT>> *aSrc4, size_t aPitchSrc4,
                              Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1,
                              Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2,
                              Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3,
                              Vector1<remove_vector_t<SrcT>> *aDst4, size_t aPitchDst4,
                              const PerspectiveTransformation<double> &aPerspective, InterpolationMode aInterpolation,
                              BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                              const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                              const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
