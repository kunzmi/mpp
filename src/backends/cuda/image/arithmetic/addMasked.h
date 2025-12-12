#pragma once
#include "add.h"
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT, typename ComputeT = add_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                         const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                         const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                              const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst, double aScaleFactor,
                              const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = add_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                       const SrcT &aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                            const SrcT &aConst, DstT *aDst, size_t aPitchDst, double aScaleFactor, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = add_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                          const SrcT *aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                               const SrcT *aConst, DstT *aDst, size_t aPitchDst, double aScaleFactor,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = add_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                             const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                             const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                  const SrcT *aSrc2, size_t aPitchSrc2, double aScaleFactor, const Size2D &aSize,
                                  const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = add_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                           const SrcT &aConst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT &aConst, double aScaleFactor, const Size2D &aSize,
                                const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = add_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchDst,
                              const SrcT *aConst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchDst,
                                   const SrcT *aConst, double aScaleFactor, const Size2D &aSize,
                                   const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
