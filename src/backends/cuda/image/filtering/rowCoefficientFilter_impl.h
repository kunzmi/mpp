#include "rowCoefficientFilter.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/rowCoefficientFilterKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/fixedSizeFilters.h>
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

template <typename SrcT, typename DstT, typename FilterT>
void InvokeRowCoefficientFilter(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                                const FilterT *aFilter, FilterT aScalingValueInv, int aFilterSize, int aFilterCenter,
                                BorderType aBorderType, const SrcT &aConstant, const Size2D &aAllowedReadRoiSize,
                                const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,
                                const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    using ComputeT             = filter_compute_type_for_t<SrcT>;

    switch (aBorderType)
    {
        case mpp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            InvokeRowCoefficientFilterKernelDefault<ComputeT, DstT, TupelSize, RoundingMode::NearestTiesToEven, BCType,
                                                    FilterT, false>(bc, aDst, aPitchDst, aFilter, aScalingValueInv,
                                                                    aFilterSize, aFilterCenter, aSize, aStreamCtx);
        }
        break;
        case mpp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi, aConstant);

            InvokeRowCoefficientFilterKernelDefault<ComputeT, DstT, TupelSize, RoundingMode::NearestTiesToEven, BCType,
                                                    FilterT, false>(bc, aDst, aPitchDst, aFilter, aScalingValueInv,
                                                                    aFilterSize, aFilterCenter, aSize, aStreamCtx);
        }
        break;
        case mpp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            InvokeRowCoefficientFilterKernelDefault<ComputeT, DstT, TupelSize, RoundingMode::NearestTiesToEven, BCType,
                                                    FilterT, false>(bc, aDst, aPitchDst, aFilter, aScalingValueInv,
                                                                    aFilterSize, aFilterCenter, aSize, aStreamCtx);
        }
        break;
        case mpp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            InvokeRowCoefficientFilterKernelDefault<ComputeT, DstT, TupelSize, RoundingMode::NearestTiesToEven, BCType,
                                                    FilterT, false>(bc, aDst, aPitchDst, aFilter, aScalingValueInv,
                                                                    aFilterSize, aFilterCenter, aSize, aStreamCtx);
        }
        break;
        case mpp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            InvokeRowCoefficientFilterKernelDefault<ComputeT, DstT, TupelSize, RoundingMode::NearestTiesToEven, BCType,
                                                    FilterT, false>(bc, aDst, aPitchDst, aFilter, aScalingValueInv,
                                                                    aFilterSize, aFilterCenter, aSize, aStreamCtx);
        }
        break;
        case mpp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            InvokeRowCoefficientFilterKernelDefault<ComputeT, DstT, TupelSize, RoundingMode::NearestTiesToEven, BCType,
                                                    FilterT, false>(bc, aDst, aPitchDst, aFilter, aScalingValueInv,
                                                                    aFilterSize, aFilterCenter, aSize, aStreamCtx);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for row filter.");
            break;
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIsTypeDst, typeFilter)                                                                  \
    template void InvokeRowCoefficientFilter<typeSrcIsTypeDst, typeSrcIsTypeDst, typeFilter>(                          \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const typeFilter *aFilter, typeFilter aScalingValueInv, int aFilterSize, int aFilterCenter,                    \
        BorderType aBorderType, const typeSrcIsTypeDst &aConstant, const Size2D &aAllowedReadRoiSize,                  \
        const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type, typeFilter)                                                                        \
    Instantiate_For(Pixel##type##C1, typeFilter);                                                                      \
    Instantiate_For(Pixel##type##C2, typeFilter);                                                                      \
    Instantiate_For(Pixel##type##C3, typeFilter);                                                                      \
    Instantiate_For(Pixel##type##C4, typeFilter);

#define ForAllChannelsWithAlpha(type, typeFilter)                                                                      \
    Instantiate_For(Pixel##type##C1, typeFilter);                                                                      \
    Instantiate_For(Pixel##type##C2, typeFilter);                                                                      \
    Instantiate_For(Pixel##type##C3, typeFilter);                                                                      \
    Instantiate_For(Pixel##type##C4, typeFilter);                                                                      \
    Instantiate_For(Pixel##type##C4A, typeFilter);

#pragma endregion
} // namespace mpp::image::cuda
