#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/opp_defs.h>

namespace opp::image::cuda
{
// Only BFloat16 and Half-Float16 have SIMD instructions, but they already have the same ComputeType by default, so we
// stick with default
template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                     size_t aPitchDst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivSrcSrcScale(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode,
                          const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                   const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivSrcCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                        scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode, const Size2D &aSize,
                        const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivSrcDevCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                           scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode, const Size2D &aSize,
                           const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                         const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode,
                              const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                       const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst,
                            scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode, const Size2D &aSize,
                            const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInplaceDevC(DstT *aSrcDst, size_t aPitchDst, const SrcT *aConst, const Size2D &aSize,
                          const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInplaceDevCScale(DstT *aSrcDst, size_t aPitchDst, const SrcT *aConst,
                               scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode,
                               const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInvInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                            const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInvInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                                 scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode,
                                 const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInvInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                          const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInvInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst,
                               scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode,
                               const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInvInplaceDevC(DstT *aSrcDst, size_t aPitchDst, const SrcT *aConst, const Size2D &aSize,
                             const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInvInplaceDevCScale(DstT *aSrcDst, size_t aPitchDst, const SrcT *aConst,
                                  scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode,
                                  const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
