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

// level types for histogram range:
template <typename SrcT> struct hist_range_types_scalar_for
{
    using levelType = int;
};
template <> struct hist_range_types_scalar_for<uint>
{
    using levelType = ulong64;
};
template <> struct hist_range_types_scalar_for<int>
{
    using levelType = long64;
};

template <> struct hist_range_types_scalar_for<HalfFp16>
{
    using levelType = float;
};
template <> struct hist_range_types_scalar_for<BFloat16>
{
    using levelType = float;
};

template <> struct hist_range_types_scalar_for<float>
{
    using levelType = float;
};
template <> struct hist_range_types_scalar_for<double>
{
    using levelType = double;
};

// compute and result types for histogram range:
template <typename SrcT> struct hist_range_types_for
{
    using levelType = typename hist_range_types_scalar_for<remove_vector_t<SrcT>>::levelType;
};

template <typename T> using hist_range_types_for_t = typename hist_range_types_for<T>::levelType;

template <typename SrcT>
size_t InvokeHistogramRangeGetBufferSize(const SrcT *aSrc1, size_t aPitchSrc1,
                                         const int aNumLevels[vector_active_size<SrcT>::value], const Size2D &aSize,
                                         const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeHistogramRange(const SrcT *aSrc1, size_t aPitchSrc1, void *aTempBuffer, size_t aTempBufferSize,
                          int *aHist[vector_active_size<SrcT>::value],
                          const int aNumLevels[vector_active_size<SrcT>::value],
                          const hist_range_types_for_t<SrcT> *aLevels[vector_active_size<SrcT>::value],
                          const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
