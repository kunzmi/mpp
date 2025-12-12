#include "histogramEven.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongXKernel.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <cub/device/device_histogram.cuh>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{

template <typename SrcT, typename LevelT>
size_t InvokeHistogramEvenGetBufferSize(const SrcT *aSrc1, size_t aPitchSrc1,
                                        const int aLevels[vector_active_size<SrcT>::value], const Size2D &aSize,
                                        const mpp::cuda::StreamCtx &aStreamCtx)
{
    size_t bufferSize = 0;
    LevelT lower_level(0);
    LevelT upper_level(1);
    int *hist[vector_active_size<SrcT>::value]{nullptr};

    if constexpr (std::same_as<HalfFp16, remove_vector_t<SrcT>>)
    {
        cudaSafeCall(
            (cub::DeviceHistogram::MultiHistogramEven<vector_size<SrcT>::value, vector_active_size<SrcT>::value>(
                nullptr, bufferSize, reinterpret_cast<const __half *>(aSrc1), hist, aLevels, lower_level.data(),
                upper_level.data(), aSize.x, aSize.y, aPitchSrc1, aStreamCtx.Stream)));
    }
    else if constexpr (std::same_as<BFloat16, remove_vector_t<SrcT>>)
    {
        cudaSafeCall(
            (cub::DeviceHistogram::MultiHistogramEven<vector_size<SrcT>::value, vector_active_size<SrcT>::value>(
                nullptr, bufferSize, reinterpret_cast<const __nv_bfloat16 *>(aSrc1), hist, aLevels, lower_level.data(),
                upper_level.data(), aSize.x, aSize.y, aPitchSrc1, aStreamCtx.Stream)));
    }
    else
    {
        cudaSafeCall(
            (cub::DeviceHistogram::MultiHistogramEven<vector_size<SrcT>::value, vector_active_size<SrcT>::value>(
                nullptr, bufferSize, reinterpret_cast<const remove_vector_t<SrcT> *>(aSrc1), hist, aLevels,
                lower_level.data(), upper_level.data(), aSize.x, aSize.y, aPitchSrc1, aStreamCtx.Stream)));
    }

    return bufferSize;
}

template <typename SrcT, typename LevelT>
void InvokeHistogramEven(const SrcT *aSrc1, size_t aPitchSrc1, void *aTempBuffer, size_t aTempBufferSize,
                         int *aHist[vector_active_size<SrcT>::value],
                         const int aLevels[vector_active_size<SrcT>::value], const LevelT &aLowerLevel,
                         const LevelT &aUpperLevel, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

    if constexpr (std::same_as<HalfFp16, remove_vector_t<SrcT>>)
    {
        cudaSafeCall(
            (cub::DeviceHistogram::MultiHistogramEven<vector_size<SrcT>::value, vector_active_size<SrcT>::value>(
                aTempBuffer, aTempBufferSize, reinterpret_cast<const __nv_bfloat16 *>(aSrc1), aHist, aLevels,
                aLowerLevel.data(), aUpperLevel.data(), aSize.x, aSize.y, aPitchSrc1, aStreamCtx.Stream)));
    }
    else if constexpr (std::same_as<BFloat16, remove_vector_t<SrcT>>)
    {
        cudaSafeCall(
            (cub::DeviceHistogram::MultiHistogramEven<vector_size<SrcT>::value, vector_active_size<SrcT>::value>(
                aTempBuffer, aTempBufferSize, reinterpret_cast<const __half *>(aSrc1), aHist, aLevels,
                aLowerLevel.data(), aUpperLevel.data(), aSize.x, aSize.y, aPitchSrc1, aStreamCtx.Stream)));
    }
    else
    {
        cudaSafeCall(
            (cub::DeviceHistogram::MultiHistogramEven<vector_size<SrcT>::value, vector_active_size<SrcT>::value>(
                aTempBuffer, aTempBufferSize, reinterpret_cast<const remove_vector_t<SrcT> *>(aSrc1), aHist, aLevels,
                aLowerLevel.data(), aUpperLevel.data(), aSize.x, aSize.y, aPitchSrc1, aStreamCtx.Stream)));
    }
}

#pragma region Instantiate

#define Instantiate_For(type)                                                                                          \
    template size_t InvokeHistogramEvenGetBufferSize<type, hist_even_level_types_for_t<type>>(                         \
        const type *aSrc1, size_t aPitchSrc1, const int aLevels[vector_active_size<type>::value], const Size2D &aSize, \
        const mpp::cuda::StreamCtx &aStreamCtx);                                                                       \
    template void InvokeHistogramEven<type>(                                                                           \
        const type *aSrc1, size_t aPitchSrc1, void *aTempBuffer, size_t aTempBufferSize,                               \
        int *aHist[vector_active_size<type>::value], const int aLevels[vector_active_size<type>::value],               \
        const hist_even_level_types_for_t<type> &aLowerLevel, const hist_even_level_types_for_t<type> &aUpperLevel,    \
        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);
#pragma endregion

} // namespace mpp::image::cuda
