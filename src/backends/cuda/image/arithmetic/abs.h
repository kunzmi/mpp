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
// If no SIMD operation is available, we use default compute type (mostly float) as compute type to avoid overflow for
// abs(T::minValue()) that is larger than T::maxValue()!

// activate SIMD only on vector for those types who support it (and are not already on by default (half-float16 and
// bfloat16)), i.e. set computeType to input/output type
template <typename T> struct abs_simd_vector_compute_type_for
{
    using type = default_compute_type_for_t<T>;
};
template <> struct abs_simd_vector_compute_type_for<Pixel8sC4> // vector simd
{
    using type = Pixel8sC4;
};
template <> struct abs_simd_vector_compute_type_for<Pixel8sC4A> // vector simd
{
    using type = Pixel8sC4A;
};
template <> struct abs_simd_vector_compute_type_for<Pixel16sC2> // vector simd
{
    using type = Pixel16sC2;
};
template <> struct abs_simd_vector_compute_type_for<Pixel16sC4> // vector simd
{
    using type = Pixel16sC4;
};
template <> struct abs_simd_vector_compute_type_for<Pixel16sC4A> // vector simd
{
    using type = Pixel16sC4A;
};

template <typename T> using abs_simd_vector_compute_type_for_t = typename abs_simd_vector_compute_type_for<T>::type;

// activate SIMD also on tupels for those types who support it, used for ComputT_SIMD
template <typename T> struct abs_simd_tupel_compute_type_for
{
    using type = voidType; // no SIMD on Tupel by default
};

template <> struct abs_simd_tupel_compute_type_for<Pixel8sC1> // tupel
{
    using type = Pixel8sC1;
};
template <> struct abs_simd_tupel_compute_type_for<Pixel8sC2> // tupel
{
    using type = Pixel8sC2;
};
template <> struct abs_simd_tupel_compute_type_for<Pixel16sC1> // tupel
{
    using type = Pixel16sC1;
};

template <typename T> using abs_simd_tupel_compute_type_for_t = typename abs_simd_tupel_compute_type_for<T>::type;

template <typename SrcT, typename ComputeT = abs_simd_vector_compute_type_for_t<SrcT>, typename DstT>
void InvokeAbsSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                  const opp::cuda::StreamCtx &aStreamCtx);

template <typename DstT, typename ComputeT = abs_simd_vector_compute_type_for_t<DstT>>
void InvokeAbsInplace(DstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
