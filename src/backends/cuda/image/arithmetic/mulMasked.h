#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{

template <typename SrcT, typename ComputeT = default_ext_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulSrcSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                         const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                         const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulSrcSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                              const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst,
                              scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                              const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulSrcCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                       const SrcT &aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                       const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulSrcCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                            const SrcT &aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                            const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulSrcDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                          const SrcT *aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulSrcDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                               const SrcT *aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                               const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                             const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                             const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                  const SrcT *aSrc2, size_t aPitchSrc2, scalefactor_t<ComputeT> aScaleFactor,
                                  const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                           const SrcT &aConst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT &aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchDst,
                              const SrcT *aConst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_ext_compute_type_for_t<SrcT>, typename DstT>
void InvokeMulInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchDst,
                                   const SrcT *aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                   const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
