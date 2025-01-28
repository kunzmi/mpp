#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include "sub.h"
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{

template <typename SrcT, typename ComputeT = sub_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubSrcSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                         const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                         const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubSrcSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                              const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst,
                              scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                              const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = sub_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubSrcCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                       const SrcT &aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                       const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubSrcCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                            const SrcT &aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                            const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = sub_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubSrcDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                          const SrcT *aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubSrcDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                               const SrcT *aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                               const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = sub_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                             const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                             const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                  const SrcT *aSrc2, size_t aPitchSrc2, scalefactor_t<ComputeT> aScaleFactor,
                                  const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = sub_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                           const SrcT &aConst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT &aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = sub_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchDst,
                              const SrcT *aConst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchDst,
                                   const SrcT *aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                   const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = sub_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubInvInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                                const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubInvInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                     const SrcT *aSrc2, size_t aPitchSrc2, scalefactor_t<ComputeT> aScaleFactor,
                                     const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = sub_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubInvInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                              const SrcT &aConst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubInvInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                   const SrcT &aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                   const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = sub_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubInvInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchDst,
                                 const SrcT *aConst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeSubInvInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchDst,
                                      const SrcT *aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                      const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
