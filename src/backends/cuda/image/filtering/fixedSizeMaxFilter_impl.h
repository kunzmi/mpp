#if OPP_ENABLE_CUDA_BACKEND

#include "maxFilter.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/fixedSizeSeparableWindowOpKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/image/filterArea.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/statistics/postOperators.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{

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

template <typename SrcT, typename DstT>
void InvokeFixedSizeMaxFilter(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, int aFilterSize,
                              const Vector2<int> &aFilterCenter, BorderType aBorderType, const SrcT &aConstant,
                              const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi,
                              const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize           = ConfigTupelSize<"Default", sizeof(DstT)>::value;
        using ComputeT                       = SrcT;
        using WindowOpT                      = opp::Max<ComputeT>;
        using PostOpT                        = opp::Nothing<ComputeT>;
        constexpr ReductionInitValue initVal = ReductionInitValue::Min;

        PostOpT postOp;

        constexpr int pixelBlockSizeY = pixel_block_size_y<DstT>::value;

        switch (aBorderType)
        {
            case opp::BorderType::None:
            {
                using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeFixedSizeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                              RoundingMode::NearestTiesToEven, BCType, WindowOpT,
                                                              PostOpT, initVal>(
                    bc, aDst, aPitchDst, aFilterSize, aFilterCenter, WindowOpT(), postOp, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::Constant:
            {
                using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi, aConstant);

                InvokeFixedSizeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                              RoundingMode::NearestTiesToEven, BCType, WindowOpT,
                                                              PostOpT, initVal>(
                    bc, aDst, aPitchDst, aFilterSize, aFilterCenter, WindowOpT(), postOp, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::Replicate:
            {
                using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeFixedSizeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                              RoundingMode::NearestTiesToEven, BCType, WindowOpT,
                                                              PostOpT, initVal>(
                    bc, aDst, aPitchDst, aFilterSize, aFilterCenter, WindowOpT(), postOp, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::Mirror:
            {
                using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeFixedSizeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                              RoundingMode::NearestTiesToEven, BCType, WindowOpT,
                                                              PostOpT, initVal>(
                    bc, aDst, aPitchDst, aFilterSize, aFilterCenter, WindowOpT(), postOp, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::MirrorReplicate:
            {
                using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeFixedSizeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                              RoundingMode::NearestTiesToEven, BCType, WindowOpT,
                                                              PostOpT, initVal>(
                    bc, aDst, aPitchDst, aFilterSize, aFilterCenter, WindowOpT(), postOp, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::Wrap:
            {
                using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeFixedSizeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                              RoundingMode::NearestTiesToEven, BCType, WindowOpT,
                                                              PostOpT, initVal>(
                    bc, aDst, aPitchDst, aFilterSize, aFilterCenter, WindowOpT(), postOp, aSize, aStreamCtx);
            }
            break;
            default:
                throw INVALIDARGUMENT(aBorderType,
                                      aBorderType << " is not a supported border type mode for box filter.");
                break;
        }
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeFixedSizeMaxFilter<typeSrcIsTypeDst, typeSrcIsTypeDst>(                                        \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst, int aFilterSize,   \
        const Vector2<int> &aFilterCenter, BorderType aBorderType, const typeSrcIsTypeDst &aConstant,                  \
        const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,                \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

#pragma endregion
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
