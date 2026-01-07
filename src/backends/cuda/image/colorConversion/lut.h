#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcDstT>
void InvokeLutPaletteSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                         const SrcDstT *aPalette, int aBitSize, const Size2D &aSize,
                         const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutPaletteInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, const SrcDstT *aPalette, int aBitSize,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutPaletteSrc33(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector3<remove_vector_t<SrcDstT>> *aDst,
                           size_t aPitchDst, const Vector3<remove_vector_t<SrcDstT>> *aPalette, int aBitSize,
                           const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutPaletteSrc34A(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector3<remove_vector_t<SrcDstT>> *aDst,
                            size_t aPitchDst, const Vector4A<remove_vector_t<SrcDstT>> *aPalette, int aBitSize,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutPaletteSrc4A3(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector4A<remove_vector_t<SrcDstT>> *aDst,
                            size_t aPitchDst, const Vector3<remove_vector_t<SrcDstT>> *aPalette, int aBitSize,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutPaletteSrc4A4A(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector4A<remove_vector_t<SrcDstT>> *aDst,
                             size_t aPitchDst, const Vector4A<remove_vector_t<SrcDstT>> *aPalette, int aBitSize,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutPaletteSrc44(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector4<remove_vector_t<SrcDstT>> *aDst,
                           size_t aPitchDst, const Vector4<remove_vector_t<SrcDstT>> *aPalette, int aBitSize,
                           const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutPaletteSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                         const Vector1<remove_vector_t<SrcDstT>> *const *aPalette, int aBitSize, const Size2D &aSize,
                         const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutPaletteInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1,
                             const Vector1<remove_vector_t<SrcDstT>> *const *aPalette, int aBitSize,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutPaletteSrc16u(const SrcDstT *aSrc1, size_t aPitchSrc1, Pixel8uC1 *aDst, size_t aPitchDst,
                            const Pixel8uC1 *aPalette, int aBitSize, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutPaletteSrc16u(const SrcDstT *aSrc1, size_t aPitchSrc1, Pixel8uC3 *aDst, size_t aPitchDst,
                            const Pixel8uC3 *aPalette, int aBitSize, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutPaletteSrc16u(const SrcDstT *aSrc1, size_t aPitchSrc1, Pixel8uC4 *aDst, size_t aPitchDst,
                            const Pixel8uC4 *aPalette, int aBitSize, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutPaletteSrc16u(const SrcDstT *aSrc1, size_t aPitchSrc1, Pixel8uC4A *aDst, size_t aPitchDst,
                            const Pixel8uC4A *aPalette, int aBitSize, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, const Pixel32fC1 *aLevels,
                  const Pixel32fC1 *aValues, const int *aAccelerator, int aLutSize, int aAcceleratorSize,
                  InterpolationMode aInterpolationMode, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                  const Pixel32fC1 *const *aLevels, const Pixel32fC1 *const *aValues, const int *const *aAccelerator,
                  int const *aLutSize, int const *aAcceleratorSize, InterpolationMode aInterpolationMode,
                  const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, const Pixel32fC1 *aLevels, const Pixel32fC1 *aValues,
                      const int *aAccelerator, int aLutSize, int aAcceleratorSize, InterpolationMode aInterpolationMode,
                      const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, const Pixel32fC1 *const *aLevels,
                      const Pixel32fC1 *const *aValues, const int *const *aAccelerator, int const *aLutSize,
                      int const *aAcceleratorSize, InterpolationMode aInterpolationMode, const Size2D &aSize,
                      const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutTrilinearSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                           const Vector3<remove_vector_t<SrcDstT>> *aLut3D,
                           const Vector3<remove_vector_t<SrcDstT>> &aMinLevel,
                           const Vector3<remove_vector_t<SrcDstT>> &aMaxLevel, const Pixel32sC3 &aLutSize,
                           const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutTrilinearSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                           const Vector4A<remove_vector_t<SrcDstT>> *aLut3D,
                           const Vector3<remove_vector_t<SrcDstT>> &aMinLevel,
                           const Vector3<remove_vector_t<SrcDstT>> &aMaxLevel, const Pixel32sC3 &aLutSize,
                           const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutTrilinearInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, const Vector3<remove_vector_t<SrcDstT>> *aLut3D,
                               const Vector3<remove_vector_t<SrcDstT>> &aMinLevel,
                               const Vector3<remove_vector_t<SrcDstT>> &aMaxLevel, const Pixel32sC3 &aLutSize,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcDstT>
void InvokeLutTrilinearInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1,
                               const Vector4A<remove_vector_t<SrcDstT>> *aLut3D,
                               const Vector3<remove_vector_t<SrcDstT>> &aMinLevel,
                               const Vector3<remove_vector_t<SrcDstT>> &aMaxLevel, const Pixel32sC3 &aLutSize,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

void InvokeLutAcceleratorKernelDefault(const float *aX, int aLutSize, int *aAccelerator, int aAccerlatorSize,
                                       const mpp::cuda::StreamCtx &aStreamCtx);
template <typename LutT>
void InvokeLutToPaletteKernelDefault(const int *__restrict__ aX, const int *__restrict__ aY, int aLutSize,
                                     LutT *__restrict__ aPalette, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx);
} // namespace mpp::image::cuda
