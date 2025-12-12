#pragma once
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{

// activate SIMD only on vector for those types who support it (and are not already on by default (half-float16 and
// bfloat16)), i.e. set computeType to input/output type
// Note: The signed types have a SIMD intrinsic, but the NPP API does only provide unsigned types as the value range for
// signed is not sufficient for all values. We keep the SIMD here but as NPP we won't expose them in the final API.
template <typename T> struct absDiff_simd_vector_compute_type_for
{
    using type = default_compute_type_for_t<T>;
};
template <> struct absDiff_simd_vector_compute_type_for<Pixel8uC4> // vector simd
{
    using type = Pixel8uC4;
};
template <> struct absDiff_simd_vector_compute_type_for<Pixel8uC4A> // vector simd
{
    using type = Pixel8uC4A;
};
template <> struct absDiff_simd_vector_compute_type_for<Pixel8sC4> // vector simd
{
    using type = Pixel8sC4;
};
template <> struct absDiff_simd_vector_compute_type_for<Pixel8sC4A> // vector simd
{
    using type = Pixel8sC4A;
};
template <> struct absDiff_simd_vector_compute_type_for<Pixel16uC2> // vector simd
{
    using type = Pixel16uC2;
};
template <> struct absDiff_simd_vector_compute_type_for<Pixel16uC4> // vector simd
{
    using type = Pixel16uC4;
};
template <> struct absDiff_simd_vector_compute_type_for<Pixel16uC4A> // vector simd
{
    using type = Pixel16uC4A;
};
template <> struct absDiff_simd_vector_compute_type_for<Pixel16sC2> // vector simd
{
    using type = Pixel16sC2;
};
template <> struct absDiff_simd_vector_compute_type_for<Pixel16sC4> // vector simd
{
    using type = Pixel16sC4;
};
template <> struct absDiff_simd_vector_compute_type_for<Pixel16sC4A> // vector simd
{
    using type = Pixel16sC4A;
};

template <typename T>
using absDiff_simd_vector_compute_type_for_t = typename absDiff_simd_vector_compute_type_for<T>::type;

// activate SIMD also on tupels for those types who support it, used for ComputT_SIMD
template <typename T> struct absDiff_simd_tupel_compute_type_for
{
    using type = voidType; // no SIMD on Tupel by default
};

template <> struct absDiff_simd_tupel_compute_type_for<Pixel8uC1> // tupel
{
    using type = Pixel8uC1;
};
template <> struct absDiff_simd_tupel_compute_type_for<Pixel8uC2> // tupel
{
    using type = Pixel8uC2;
};
template <> struct absDiff_simd_tupel_compute_type_for<Pixel8sC1> // tupel
{
    using type = Pixel8sC1;
};
template <> struct absDiff_simd_tupel_compute_type_for<Pixel8sC2> // tupel
{
    using type = Pixel8sC2;
};
template <> struct absDiff_simd_tupel_compute_type_for<Pixel16uC1> // tupel
{
    using type = Pixel16uC1;
};
template <> struct absDiff_simd_tupel_compute_type_for<Pixel16sC1> // tupel
{
    using type = Pixel16sC1;
};

template <typename T>
using absDiff_simd_tupel_compute_type_for_t = typename absDiff_simd_tupel_compute_type_for<T>::type;

template <typename SrcT, typename ComputeT = absDiff_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAbsDiffSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                         size_t aPitchDst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = absDiff_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAbsDiffSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = absDiff_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAbsDiffSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                          const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = absDiff_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAbsDiffInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = absDiff_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAbsDiffInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = absDiff_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAbsDiffInplaceDevC(DstT *aSrcDst, size_t aPitchDst, const SrcT *aConst, const Size2D &aSize,
                              const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
