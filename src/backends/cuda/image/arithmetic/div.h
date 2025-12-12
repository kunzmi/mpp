#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/mpp_defs.h>

namespace mpp::image::cuda
{
// Only BFloat16 and Half-Float16 have SIMD instructions, but they already have the same ComputeType by default, so we
// stick with default
template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                     size_t aPitchDst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivSrcSrcScale(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                   const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivSrcCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                        double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                        const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivSrcDevCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                           double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                         const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                              const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, double aScaleFactor,
                            mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInplaceDevC(DstT *aSrcDst, size_t aPitchDst, const SrcT *aConst, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInplaceDevCScale(DstT *aSrcDst, size_t aPitchDst, const SrcT *aConst, double aScaleFactor,
                               mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                               const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInvInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInvInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                                 double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                                 const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInvInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInvInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, double aScaleFactor,
                               mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                               const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInvInplaceDevC(DstT *aSrcDst, size_t aPitchDst, const SrcT *aConst, const Size2D &aSize,
                             const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeDivInvInplaceDevCScale(DstT *aSrcDst, size_t aPitchDst, const SrcT *aConst, double aScaleFactor,
                                  mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                                  const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
