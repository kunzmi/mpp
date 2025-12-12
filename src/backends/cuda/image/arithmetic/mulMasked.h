#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

template <typename SrcT, typename ComputeT = default_ext_int_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulSrcSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                         const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                         const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_int_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulSrcSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                              const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst, double aScaleFactor,
                              const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_int_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulSrcCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                       const SrcT &aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_int_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulSrcCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                            const SrcT &aConst, DstT *aDst, size_t aPitchDst, double aScaleFactor, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_int_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulSrcDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                          const SrcT *aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_int_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulSrcDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                               const SrcT *aConst, DstT *aDst, size_t aPitchDst, double aScaleFactor,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_int_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                             const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                             const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_int_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                  const SrcT *aSrc2, size_t aPitchSrc2, double aScaleFactor, const Size2D &aSize,
                                  const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_int_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                           const SrcT &aConst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_int_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT &aConst, double aScaleFactor, const Size2D &aSize,
                                const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_int_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchDst,
                              const SrcT *aConst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_int_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchDst,
                                   const SrcT *aConst, double aScaleFactor, const Size2D &aSize,
                                   const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
