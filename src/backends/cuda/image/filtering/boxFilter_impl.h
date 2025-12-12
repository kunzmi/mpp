#include "boxFilter.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/separableWindowOpKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/image/filterArea.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/statistics/postOperators.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
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
void InvokeBoxFilter(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const FilterArea &aFilterArea,
                     BorderType aBorderType, const SrcT &aConstant, const Size2D &aAllowedReadRoiSize,
                     const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,
                     const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

    constexpr size_t TupelSize           = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    using ComputeT                       = filter_compute_type_for_t<SrcT>;
    using WindowOpT                      = mpp::Add<ComputeT>;
    using PostOpT                        = mpp::DivPostOp<ComputeT>;
    constexpr ReductionInitValue initVal = ReductionInitValue::Zero;

    PostOpT postOp(static_cast<remove_vector_t<ComputeT>>(aFilterArea.Size.x * aFilterArea.Size.y));

    constexpr int pixelBlockSizeY = pixel_block_size_y<DstT>::value;

    switch (aBorderType)
    {
        case mpp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            InvokeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                 RoundingMode::NearestTiesToEven, BCType, WindowOpT, PostOpT, initVal>(
                bc, aDst, aPitchDst, aFilterArea, WindowOpT(), postOp, aSize, aStreamCtx);
        }
        break;
        case mpp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi, aConstant);

            InvokeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                 RoundingMode::NearestTiesToEven, BCType, WindowOpT, PostOpT, initVal>(
                bc, aDst, aPitchDst, aFilterArea, WindowOpT(), postOp, aSize, aStreamCtx);
        }
        break;
        case mpp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            InvokeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                 RoundingMode::NearestTiesToEven, BCType, WindowOpT, PostOpT, initVal>(
                bc, aDst, aPitchDst, aFilterArea, WindowOpT(), postOp, aSize, aStreamCtx);
        }
        break;
        case mpp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            InvokeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                 RoundingMode::NearestTiesToEven, BCType, WindowOpT, PostOpT, initVal>(
                bc, aDst, aPitchDst, aFilterArea, WindowOpT(), postOp, aSize, aStreamCtx);
        }
        break;
        case mpp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            InvokeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                 RoundingMode::NearestTiesToEven, BCType, WindowOpT, PostOpT, initVal>(
                bc, aDst, aPitchDst, aFilterArea, WindowOpT(), postOp, aSize, aStreamCtx);
        }
        break;
        case mpp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            InvokeSeparableWindowOpKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY,
                                                 RoundingMode::NearestTiesToEven, BCType, WindowOpT, PostOpT, initVal>(
                bc, aDst, aPitchDst, aFilterArea, WindowOpT(), postOp, aSize, aStreamCtx);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for box filter.");
            break;
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeBoxFilter<typeSrc, typeDst>(                                                                   \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst, const FilterArea &aFilterArea,       \
        BorderType aBorderType, const typeSrc &aConstant, const Size2D &aAllowedReadRoiSize,                           \
        const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(typeSrc, typeDst)                                                                        \
    Instantiate_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                           \
    Instantiate_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                           \
    Instantiate_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                           \
    Instantiate_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsWithAlpha(typeSrc, typeDst)                                                                      \
    Instantiate_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                           \
    Instantiate_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                           \
    Instantiate_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                           \
    Instantiate_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                                           \
    Instantiate_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#pragma endregion
} // namespace mpp::image::cuda
