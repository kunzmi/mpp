#if OPP_ENABLE_CUDA_BACKEND

#include "histogramRange.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongXKernel.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>
#include <cub/device/device_histogram.cuh>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{

template <typename SrcT>
size_t InvokeHistogramRangeGetBufferSize(const SrcT *aSrc1, size_t aPitchSrc1,
                                         const int aNumLevels[vector_active_size<SrcT>::value], const Size2D &aSize,
                                         const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        size_t bufferSize = 0;
        int *hist[vector_active_size<SrcT>::value]{nullptr};
        hist_range_types_for_t<SrcT> *levels[vector_active_size<SrcT>::value]{nullptr};

        if constexpr (std::same_as<HalfFp16, remove_vector_t<SrcT>>)
        {
            cudaSafeCall(
                (cub::DeviceHistogram::MultiHistogramRange<vector_size<SrcT>::value, vector_active_size<SrcT>::value>(
                    nullptr, bufferSize, reinterpret_cast<const __half *>(aSrc1), hist, aNumLevels, levels, aSize.x,
                    aSize.y, aPitchSrc1, aStreamCtx.Stream)));
        }
        else if constexpr (std::same_as<BFloat16, remove_vector_t<SrcT>>)
        {
            cudaSafeCall(
                (cub::DeviceHistogram::MultiHistogramRange<vector_size<SrcT>::value, vector_active_size<SrcT>::value>(
                    nullptr, bufferSize, reinterpret_cast<const __nv_bfloat16 *>(aSrc1), hist, aNumLevels, levels,
                    aSize.x, aSize.y, aPitchSrc1, aStreamCtx.Stream)));
        }
        else
        {
            cudaSafeCall(
                (cub::DeviceHistogram::MultiHistogramRange<vector_size<SrcT>::value, vector_active_size<SrcT>::value>(
                    nullptr, bufferSize, reinterpret_cast<const remove_vector_t<SrcT> *>(aSrc1), hist, aNumLevels,
                    levels, aSize.x, aSize.y, aPitchSrc1, aStreamCtx.Stream)));
        }

        return bufferSize;
    }
}

template <typename SrcT>
void InvokeHistogramRange(const SrcT *aSrc1, size_t aPitchSrc1, void *aTempBuffer, size_t aTempBufferSize,
                          int *aHist[vector_active_size<SrcT>::value],
                          const int aNumLevels[vector_active_size<SrcT>::value],
                          const hist_range_types_for_t<SrcT> *aLevels[vector_active_size<SrcT>::value],
                          const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        if constexpr (std::same_as<HalfFp16, remove_vector_t<SrcT>>)
        {
            cudaSafeCall(
                (cub::DeviceHistogram::MultiHistogramRange<vector_size<SrcT>::value, vector_active_size<SrcT>::value>(
                    aTempBuffer, aTempBufferSize, reinterpret_cast<const __nv_bfloat16 *>(aSrc1), aHist, aNumLevels,
                    aLevels, aSize.x, aSize.y, aPitchSrc1, aStreamCtx.Stream)));
        }
        else if constexpr (std::same_as<BFloat16, remove_vector_t<SrcT>>)
        {
            cudaSafeCall(
                (cub::DeviceHistogram::MultiHistogramRange<vector_size<SrcT>::value, vector_active_size<SrcT>::value>(
                    aTempBuffer, aTempBufferSize, reinterpret_cast<const __half *>(aSrc1), aHist, aNumLevels, aLevels,
                    aSize.x, aSize.y, aPitchSrc1, aStreamCtx.Stream)));
        }
        else
        {
            cudaSafeCall(
                (cub::DeviceHistogram::MultiHistogramRange<vector_size<SrcT>::value, vector_active_size<SrcT>::value>(
                    aTempBuffer, aTempBufferSize, reinterpret_cast<const remove_vector_t<SrcT> *>(aSrc1), aHist,
                    aNumLevels, aLevels, aSize.x, aSize.y, aPitchSrc1, aStreamCtx.Stream)));
        }
    }
}

#pragma region Instantiate

#define Instantiate_For(type)                                                                                          \
    template size_t InvokeHistogramRangeGetBufferSize<type>(                                                           \
        const type *aSrc1, size_t aPitchSrc1, const int aNumLevels[vector_active_size<type>::value],                   \
        const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);                                                  \
    template void InvokeHistogramRange<type>(                                                                          \
        const type *aSrc1, size_t aPitchSrc1, void *aTempBuffer, size_t aTempBufferSize,                               \
        int *aHist[vector_active_size<type>::value], const int aNumLevels[vector_active_size<type>::value],            \
        const hist_range_types_for_t<type> *aLevels[vector_active_size<type>::value], const Size2D &aSize,             \
        const opp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
