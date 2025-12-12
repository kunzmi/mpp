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
template <typename T> struct add_simd_vector_compute_type_for
{
    using type = default_compute_type_for_t<T>;
};
template <> struct add_simd_vector_compute_type_for<Pixel8uC4> // vector simd
{
    using type = Pixel8uC4;
};
template <> struct add_simd_vector_compute_type_for<Pixel8uC4A> // vector simd
{
    using type = Pixel8uC4A;
};
template <> struct add_simd_vector_compute_type_for<Pixel8sC4> // vector simd
{
    using type = Pixel8sC4;
};
template <> struct add_simd_vector_compute_type_for<Pixel8sC4A> // vector simd
{
    using type = Pixel8sC4A;
};
template <> struct add_simd_vector_compute_type_for<Pixel16uC2> // vector simd
{
    using type = Pixel16uC2;
};
template <> struct add_simd_vector_compute_type_for<Pixel16uC4> // vector simd
{
    using type = Pixel16uC4;
};
template <> struct add_simd_vector_compute_type_for<Pixel16uC4A> // vector simd
{
    using type = Pixel16uC4A;
};
template <> struct add_simd_vector_compute_type_for<Pixel16sC2> // vector simd
{
    using type = Pixel16sC2;
};
template <> struct add_simd_vector_compute_type_for<Pixel16sC4> // vector simd
{
    using type = Pixel16sC4;
};
template <> struct add_simd_vector_compute_type_for<Pixel16sC4A> // vector simd
{
    using type = Pixel16sC4A;
};
template <> struct add_simd_vector_compute_type_for<Pixel16scC1> // vector/complex simd
{
    using type = Pixel16scC1;
};
template <> struct add_simd_vector_compute_type_for<Pixel16scC2> // vector/complex simd
{
    using type = Pixel16scC2;
};
template <> struct add_simd_vector_compute_type_for<Pixel16scC3> // vector/complex simd
{
    using type = Pixel16scC3;
};
template <> struct add_simd_vector_compute_type_for<Pixel16scC4> // vector/complex simd
{
    using type = Pixel16scC4;
};

template <typename T> using add_simd_vector_compute_type_for_t = typename add_simd_vector_compute_type_for<T>::type;

// activate SIMD also on tupels for those types who support it, used for ComputT_SIMD
template <typename T> struct add_simd_tupel_compute_type_for
{
    using type = voidType; // no SIMD on Tupel by default
};

template <> struct add_simd_tupel_compute_type_for<Pixel8uC1> // tupel
{
    using type = Pixel8uC1;
};
template <> struct add_simd_tupel_compute_type_for<Pixel8uC2> // tupel
{
    using type = Pixel8uC2;
};
template <> struct add_simd_tupel_compute_type_for<Pixel8sC1> // tupel
{
    using type = Pixel8sC1;
};
template <> struct add_simd_tupel_compute_type_for<Pixel8sC2> // tupel
{
    using type = Pixel8sC2;
};
template <> struct add_simd_tupel_compute_type_for<Pixel16uC1> // tupel
{
    using type = Pixel16uC1;
};
template <> struct add_simd_tupel_compute_type_for<Pixel16sC1> // tupel
{
    using type = Pixel16sC1;
};
template <> struct add_simd_tupel_compute_type_for<Pixel16fC1> // tupel
{
    using type = Pixel16fC1;
};
template <> struct add_simd_tupel_compute_type_for<Pixel16bfC1> // tupel
{
    using type = Pixel16bfC1;
};
template <typename T> using add_simd_tupel_compute_type_for_t = typename add_simd_tupel_compute_type_for<T>::type;

template <typename SrcT, typename ComputeT = add_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                     size_t aPitchDst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcSrcScale(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, double aScaleFactor, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = add_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                   const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                        double aScaleFactor, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = add_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddSrcDevCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                           double aScaleFactor, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = add_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                         const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              double aScaleFactor, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = add_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, double aScaleFactor,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = add_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddInplaceDevC(DstT *aSrcDst, size_t aPitchDst, const SrcT *aConst, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename ComputeT = default_compute_type_for_t<SrcT>, typename DstT>
void InvokeAddInplaceDevCScale(DstT *aSrcDst, size_t aPitchDst, const SrcT *aConst, double aScaleFactor,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
