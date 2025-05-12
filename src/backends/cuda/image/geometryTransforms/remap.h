#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{
template <typename SrcT>
void InvokeRemapSrc(const SrcT *aSrc1, size_t aPitchSrc1, SrcT *aDst, size_t aPitchDst,
                    const Pixel32fC2 *aCoordinateMapPtr, size_t aCoordinateMapPitch, InterpolationMode aInterpolation,
                    BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                    const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                    const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeRemapSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                    const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                    Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1, //
                    Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2, const Pixel32fC2 *aCoordinateMapPtr,
                    size_t aCoordinateMapPitch, InterpolationMode aInterpolation, BorderType aBorder,
                    const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize,
                    const Size2D &aSizeSrc, const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeRemapSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1, //
                    const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2, //
                    const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3, //
                    Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1,       //
                    Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2,       //
                    Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3, const Pixel32fC2 *aCoordinateMapPtr,
                    size_t aCoordinateMapPitch, InterpolationMode aInterpolation, BorderType aBorder,
                    const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize,
                    const Size2D &aSizeSrc, const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeRemapSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1, //
                    const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2, //
                    const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3, //
                    const Vector1<remove_vector_t<SrcT>> *aSrc4, size_t aPitchSrc4, //
                    Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1,       //
                    Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2,       //
                    Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3,       //
                    Vector1<remove_vector_t<SrcT>> *aDst4, size_t aPitchDst4, const Pixel32fC2 *aCoordinateMapPtr,
                    size_t aCoordinateMapPitch, InterpolationMode aInterpolation, BorderType aBorder,
                    const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize,
                    const Size2D &aSizeSrc, const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeRemapSrc(const SrcT *aSrc1, size_t aPitchSrc1, SrcT *aDst, size_t aPitchDst,
                    const Pixel32fC1 *aCoordinateMapXPtr, size_t aCoordinateMapXPitch,
                    const Pixel32fC1 *aCoordinateMapYPtr, size_t aCoordinateMapYPitch, InterpolationMode aInterpolation,
                    BorderType aBorder, const SrcT &aConstant, const Vector2<int> aAllowedReadRoiOffset,
                    const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const Size2D &aSizeDst,
                    const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeRemapSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1,
                    const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2,
                    Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1, //
                    Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2, const Pixel32fC1 *aCoordinateMapXPtr,
                    size_t aCoordinateMapXPitch, const Pixel32fC1 *aCoordinateMapYPtr, size_t aCoordinateMapYPitch,
                    InterpolationMode aInterpolation, BorderType aBorder, const SrcT &aConstant,
                    const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,
                    const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeRemapSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1, //
                    const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2, //
                    const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3, //
                    Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1,       //
                    Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2,       //
                    Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3, const Pixel32fC1 *aCoordinateMapXPtr,
                    size_t aCoordinateMapXPitch, const Pixel32fC1 *aCoordinateMapYPtr, size_t aCoordinateMapYPitch,
                    InterpolationMode aInterpolation, BorderType aBorder, const SrcT &aConstant,
                    const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,
                    const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeRemapSrc(const Vector1<remove_vector_t<SrcT>> *aSrc1, size_t aPitchSrc1, //
                    const Vector1<remove_vector_t<SrcT>> *aSrc2, size_t aPitchSrc2, //
                    const Vector1<remove_vector_t<SrcT>> *aSrc3, size_t aPitchSrc3, //
                    const Vector1<remove_vector_t<SrcT>> *aSrc4, size_t aPitchSrc4, //
                    Vector1<remove_vector_t<SrcT>> *aDst1, size_t aPitchDst1,       //
                    Vector1<remove_vector_t<SrcT>> *aDst2, size_t aPitchDst2,       //
                    Vector1<remove_vector_t<SrcT>> *aDst3, size_t aPitchDst3,       //
                    Vector1<remove_vector_t<SrcT>> *aDst4, size_t aPitchDst4, const Pixel32fC1 *aCoordinateMapXPtr,
                    size_t aCoordinateMapXPitch, const Pixel32fC1 *aCoordinateMapYPtr, size_t aCoordinateMapYPitch,
                    InterpolationMode aInterpolation, BorderType aBorder, const SrcT &aConstant,
                    const Vector2<int> aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,
                    const Size2D &aSizeDst, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
