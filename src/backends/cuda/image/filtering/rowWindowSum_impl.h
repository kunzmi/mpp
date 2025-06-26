#if MPP_ENABLE_CUDA_BACKEND

#include "rowWindowSum.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/filtering/windowSumResultType.h>
#include <backends/cuda/image/rowWindowSumKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{

template <typename SrcT>
void InvokeRowWindowSum(const SrcT *aSrc1, size_t aPitchSrc1, window_sum_result_type_t<SrcT> *aDst, size_t aPitchDst,
                        complex_basetype_t<remove_vector_t<window_sum_result_type_t<SrcT>>> aScalingValue,
                        int aFilterSize, int aFilterCenter, BorderType aBorderType, const SrcT &aConstant,
                        const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,
                        const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        using DstT    = window_sum_result_type_t<SrcT>;
        using FilterT = remove_vector_t<window_sum_result_type_t<SrcT>>;
        MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
        using ComputeT             = DstT;

        switch (aBorderType)
        {
            case mpp::BorderType::None:
            {
                using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeRowWindowSumKernelDefault<ComputeT, DstT, TupelSize, RoundingMode::None, BCType, FilterT>(
                    bc, aDst, aPitchDst, aScalingValue, aFilterSize, aFilterCenter, aSize, aStreamCtx);
            }
            break;
            case mpp::BorderType::Constant:
            {
                using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi, aConstant);

                InvokeRowWindowSumKernelDefault<ComputeT, DstT, TupelSize, RoundingMode::None, BCType, FilterT>(
                    bc, aDst, aPitchDst, aScalingValue, aFilterSize, aFilterCenter, aSize, aStreamCtx);
            }
            break;
            case mpp::BorderType::Replicate:
            {
                using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeRowWindowSumKernelDefault<ComputeT, DstT, TupelSize, RoundingMode::None, BCType, FilterT>(
                    bc, aDst, aPitchDst, aScalingValue, aFilterSize, aFilterCenter, aSize, aStreamCtx);
            }
            break;
            case mpp::BorderType::Mirror:
            {
                using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeRowWindowSumKernelDefault<ComputeT, DstT, TupelSize, RoundingMode::None, BCType, FilterT>(
                    bc, aDst, aPitchDst, aScalingValue, aFilterSize, aFilterCenter, aSize, aStreamCtx);
            }
            break;
            case mpp::BorderType::MirrorReplicate:
            {
                using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeRowWindowSumKernelDefault<ComputeT, DstT, TupelSize, RoundingMode::None, BCType, FilterT>(
                    bc, aDst, aPitchDst, aScalingValue, aFilterSize, aFilterCenter, aSize, aStreamCtx);
            }
            break;
            case mpp::BorderType::Wrap:
            {
                using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeRowWindowSumKernelDefault<ComputeT, DstT, TupelSize, RoundingMode::None, BCType, FilterT>(
                    bc, aDst, aPitchDst, aScalingValue, aFilterSize, aFilterCenter, aSize, aStreamCtx);
            }
            break;
            default:
                throw INVALIDARGUMENT(aBorderType,
                                      aBorderType << " is not a supported border type mode for row window sum.");
                break;
        }
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeRowWindowSum<typeSrc>(                                                                         \
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

#pragma endregion
} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
