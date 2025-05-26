#if OPP_ENABLE_CUDA_BACKEND

#include "columnWindowSum.h"
#include <backends/cuda/image/columnWindowSumKernel.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/filtering/windowSumResultType.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{

template <typename T> struct pixel_block_size_x
{
    constexpr static int value = 1;
};
template <> struct pixel_block_size_x<Pixel8uC3>
{
    constexpr static int value = 4;
};
template <> struct pixel_block_size_x<Pixel8sC3>
{
    constexpr static int value = 4;
};

template <typename T> struct pixel_block_size_y
{
    constexpr static int value = 1;
};
template <typename T>
    requires(sizeof(remove_vector_t<T>) == 2)
struct pixel_block_size_y<T>
{
    constexpr static int value = 2;
};
template <typename T>
    requires(sizeof(remove_vector_t<T>) == 1)
struct pixel_block_size_y<T>
{
    constexpr static int value = 4;
};

template <typename SrcT>
void InvokeColumnWindowSum(const SrcT *aSrc1, size_t aPitchSrc1, window_sum_result_type_t<SrcT> *aDst, size_t aPitchDst,
                           complex_basetype_t<remove_vector_t<window_sum_result_type_t<SrcT>>> aScalingValue,
                           int aFilterSize, int aFilterCenter, BorderType aBorderType, const SrcT &aConstant,
                           const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi,
                           const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        using DstT    = window_sum_result_type_t<SrcT>;
        using FilterT = remove_vector_t<window_sum_result_type_t<SrcT>>;
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
        using ComputeT             = DstT;

        constexpr int pixelBlockSizeY = pixel_block_size_y<DstT>::value;

        switch (aBorderType)
        {
            case opp::BorderType::None:
            {
                using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeColumnWindowSumKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY, RoundingMode::None,
                                                   BCType, FilterT>(bc, aDst, aPitchDst, aScalingValue, aFilterSize,
                                                                    aFilterCenter, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::Constant:
            {
                using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi, aConstant);

                InvokeColumnWindowSumKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY, RoundingMode::None,
                                                   BCType, FilterT>(bc, aDst, aPitchDst, aScalingValue, aFilterSize,
                                                                    aFilterCenter, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::Replicate:
            {
                using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeColumnWindowSumKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY, RoundingMode::None,
                                                   BCType, FilterT>(bc, aDst, aPitchDst, aScalingValue, aFilterSize,
                                                                    aFilterCenter, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::Mirror:
            {
                using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeColumnWindowSumKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY, RoundingMode::None,
                                                   BCType, FilterT>(bc, aDst, aPitchDst, aScalingValue, aFilterSize,
                                                                    aFilterCenter, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::MirrorReplicate:
            {
                using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeColumnWindowSumKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY, RoundingMode::None,
                                                   BCType, FilterT>(bc, aDst, aPitchDst, aScalingValue, aFilterSize,
                                                                    aFilterCenter, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::Wrap:
            {
                using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeColumnWindowSumKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY, RoundingMode::None,
                                                   BCType, FilterT>(bc, aDst, aPitchDst, aScalingValue, aFilterSize,
                                                                    aFilterCenter, aSize, aStreamCtx);
            }
            break;
            default:
                throw INVALIDARGUMENT(aBorderType,
                                      aBorderType << " is not a supported border type mode for column window sum.");
                break;
        }
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeColumnWindowSum<typeSrc>(                                                                      \
        const typeSrc *aSrc1, size_t aPitchSrc1, window_sum_result_type_t<typeSrc> *aDst, size_t aPitchDst,            \
        complex_basetype_t<remove_vector_t<window_sum_result_type_t<typeSrc>>> aScalingValue, int aFilterSize,         \
        int aFilterCenter, BorderType aBorderType, const typeSrc &aConstant, const Size2D &aAllowedReadRoiSize,        \
        const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(typeSrc)                                                                                 \
    Instantiate_For(Pixel##typeSrc##C1);                                                                               \
    Instantiate_For(Pixel##typeSrc##C2);                                                                               \
    Instantiate_For(Pixel##typeSrc##C3);                                                                               \
    Instantiate_For(Pixel##typeSrc##C4);

#define ForAllChannelsWithAlpha(typeSrc)                                                                               \
    Instantiate_For(Pixel##typeSrc##C1);                                                                               \
    Instantiate_For(Pixel##typeSrc##C2);                                                                               \
    Instantiate_For(Pixel##typeSrc##C3);                                                                               \
    Instantiate_For(Pixel##typeSrc##C4);                                                                               \
    Instantiate_For(Pixel##typeSrc##C4A);

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

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
