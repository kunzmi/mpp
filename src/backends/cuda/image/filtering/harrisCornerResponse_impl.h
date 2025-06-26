#if OPP_ENABLE_CUDA_BACKEND

#include "harrisCornerResponse.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/fixedSizeSeparableWindowOpKernel.h>
#include <backends/cuda/image/separableWindowOpKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/filtering/postOperators.h>
#include <common/image/filterArea.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/reductionInitValues.h>
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

template <typename T> struct pixel_block_size_y
{
    constexpr static int value = 1;
};

template <typename SrcT, typename DstT>
void InvokeHarrisCornerResponse(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                                const FilterArea &aFilterArea, float aK, float aScale, BorderType aBorderType,
                                const SrcT &aConstant, const Size2D &aAllowedReadRoiSize,
                                const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,
                                const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize           = ConfigTupelSize<"Default", sizeof(DstT)>::value;
        using ComputeT                       = filter_compute_type_for_t<SrcT>;
        using WindowOpT                      = opp::Add<ComputeT>;
        using PostOpT                        = opp::HarrisCorner;
        constexpr ReductionInitValue initVal = ReductionInitValue::Zero;

        PostOpT postOp(aK, aScale);

        constexpr int pixelBlockSizeY = pixel_block_size_y<DstT>::value;

        if (aFilterArea.Size == 3 || aFilterArea.Size == 5 || aFilterArea.Size == 7 || aFilterArea.Size == 9)
        {
            switch (aBorderType)
            {
                case opp::BorderType::None:
                {
                    using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
                    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                    InvokeFixedSizeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                                  RoundingMode::NearestTiesToEven, BCType, WindowOpT,
                                                                  PostOpT, initVal>(
                        bc, aDst, aPitchDst, aFilterArea.Size.x, aFilterArea.Center, WindowOpT(), postOp, aSize,
                        aStreamCtx);
                }
                break;
                case opp::BorderType::Constant:
                {
                    using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
                    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi, aConstant);

                    InvokeFixedSizeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                                  RoundingMode::NearestTiesToEven, BCType, WindowOpT,
                                                                  PostOpT, initVal>(
                        bc, aDst, aPitchDst, aFilterArea.Size.x, aFilterArea.Center, WindowOpT(), postOp, aSize,
                        aStreamCtx);
                }
                break;
                case opp::BorderType::Replicate:
                {
                    using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
                    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                    InvokeFixedSizeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                                  RoundingMode::NearestTiesToEven, BCType, WindowOpT,
                                                                  PostOpT, initVal>(
                        bc, aDst, aPitchDst, aFilterArea.Size.x, aFilterArea.Center, WindowOpT(), postOp, aSize,
                        aStreamCtx);
                }
                break;
                case opp::BorderType::Mirror:
                {
                    using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
                    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                    InvokeFixedSizeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                                  RoundingMode::NearestTiesToEven, BCType, WindowOpT,
                                                                  PostOpT, initVal>(
                        bc, aDst, aPitchDst, aFilterArea.Size.x, aFilterArea.Center, WindowOpT(), postOp, aSize,
                        aStreamCtx);
                }
                break;
                case opp::BorderType::MirrorReplicate:
                {
                    using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
                    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                    InvokeFixedSizeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                                  RoundingMode::NearestTiesToEven, BCType, WindowOpT,
                                                                  PostOpT, initVal>(
                        bc, aDst, aPitchDst, aFilterArea.Size.x, aFilterArea.Center, WindowOpT(), postOp, aSize,
                        aStreamCtx);
                }
                break;
                case opp::BorderType::Wrap:
                {
                    using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
                    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                    InvokeFixedSizeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                                  RoundingMode::NearestTiesToEven, BCType, WindowOpT,
                                                                  PostOpT, initVal>(
                        bc, aDst, aPitchDst, aFilterArea.Size.x, aFilterArea.Center, WindowOpT(), postOp, aSize,
                        aStreamCtx);
                }
                break;
                default:
                    throw INVALIDARGUMENT(
                        aBorderType, aBorderType << " is not a supported border type mode for harris corner response.");
                    break;
            }
        }
        else
        {
            switch (aBorderType)
            {
                case opp::BorderType::None:
                {
                    using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
                    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                    InvokeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                         RoundingMode::NearestTiesToEven, BCType, WindowOpT, PostOpT,
                                                         initVal>(bc, aDst, aPitchDst, aFilterArea, WindowOpT(), postOp,
                                                                  aSize, aStreamCtx);
                }
                break;
                case opp::BorderType::Constant:
                {
                    using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
                    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi, aConstant);

                    InvokeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                         RoundingMode::NearestTiesToEven, BCType, WindowOpT, PostOpT,
                                                         initVal>(bc, aDst, aPitchDst, aFilterArea, WindowOpT(), postOp,
                                                                  aSize, aStreamCtx);
                }
                break;
                case opp::BorderType::Replicate:
                {
                    using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
                    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                    InvokeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                         RoundingMode::NearestTiesToEven, BCType, WindowOpT, PostOpT,
                                                         initVal>(bc, aDst, aPitchDst, aFilterArea, WindowOpT(), postOp,
                                                                  aSize, aStreamCtx);
                }
                break;
                case opp::BorderType::Mirror:
                {
                    using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
                    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                    InvokeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                         RoundingMode::NearestTiesToEven, BCType, WindowOpT, PostOpT,
                                                         initVal>(bc, aDst, aPitchDst, aFilterArea, WindowOpT(), postOp,
                                                                  aSize, aStreamCtx);
                }
                break;
                case opp::BorderType::MirrorReplicate:
                {
                    using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
                    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                    InvokeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                         RoundingMode::NearestTiesToEven, BCType, WindowOpT, PostOpT,
                                                         initVal>(bc, aDst, aPitchDst, aFilterArea, WindowOpT(), postOp,
                                                                  aSize, aStreamCtx);
                }
                break;
                case opp::BorderType::Wrap:
                {
                    using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
                    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                    InvokeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                         RoundingMode::NearestTiesToEven, BCType, WindowOpT, PostOpT,
                                                         initVal>(bc, aDst, aPitchDst, aFilterArea, WindowOpT(), postOp,
                                                                  aSize, aStreamCtx);
                }
                break;
                default:
                    throw INVALIDARGUMENT(
                        aBorderType, aBorderType << " is not a supported border type mode for harris corner response.");
                    break;
            }
        }
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeHarrisCornerResponse<typeSrc, typeDst>(                                                        \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst, const FilterArea &aFilterArea,       \
        float aK, float aScale, BorderType aBorderType, const typeSrc &aConstant, const Size2D &aAllowedReadRoiSize,   \
        const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type) Instantiate_For(Pixel##type##C4, Pixel##type##C1);

#pragma endregion
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
