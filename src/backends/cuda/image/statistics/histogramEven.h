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

// level types for histogram even:
template <typename SrcT> struct hist_even_level_types_scalar_for
{
    using levelType = int;
};
template <> struct hist_even_level_types_scalar_for<uint>
{
    using levelType = ulong64;
};
template <> struct hist_even_level_types_scalar_for<int>
{
    using levelType = long64;
};

template <> struct hist_even_level_types_scalar_for<HalfFp16>
{
    using levelType = float;
};
template <> struct hist_even_level_types_scalar_for<BFloat16>
{
    using levelType = float;
};

template <> struct hist_even_level_types_scalar_for<float>
{
    using levelType = float;
};
template <> struct hist_even_level_types_scalar_for<double>
{
    using levelType = double;
};

// compute and result types for sum reduction:
template <typename SrcT> struct hist_even_level_types_for
{
    using levelType =
        same_vector_size_different_type_t<SrcT,
                                          typename hist_even_level_types_scalar_for<remove_vector_t<SrcT>>::levelType>;
};

template <typename T> using hist_even_level_types_for_t = typename hist_even_level_types_for<T>::levelType;

template <typename SrcT, typename LevelT = hist_even_level_types_for_t<SrcT>>
size_t InvokeHistogramEvenGetBufferSize(const SrcT *aSrc1, size_t aPitchSrc1,
                                        const int aLevels[vector_active_size<SrcT>::value], const Size2D &aSize,
                                        const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT, typename LevelT>
void InvokeHistogramEven(const SrcT *aSrc1, size_t aPitchSrc1, void *aTempBuffer, size_t aTempBufferSize,
                         int *aHist[vector_active_size<SrcT>::value],
                         const int aLevels[vector_active_size<SrcT>::value], const LevelT &aLowerLevel,
                         const LevelT &aUpperLevel, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
