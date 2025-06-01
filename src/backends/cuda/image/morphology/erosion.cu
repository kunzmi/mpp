#if OPP_ENABLE_CUDA_BACKEND

#include "erosion.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/morphologyKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/filterArea.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/morphology/operators.h>
#include <common/morphology/postOperators.h>
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
    constexpr static int value = 4;
};
template <typename T>
    requires(sizeof(remove_vector_t<T>) == 1)
struct pixel_block_size_y<T>
{
    constexpr static int value = 4;
};

template <typename SrcT, typename DstT>
void InvokeErosion(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Pixel8uC1 *aMask,
                   const FilterArea &aFilterArea, BorderType aBorderType, const SrcT &aConstant,
                   const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,
                   const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
        using FilterT              = Pixel8uC1;
        using MorphOp              = opp::Erode<DstT, FilterT>;
        using PostOp               = opp::NothingMorph<DstT>;

        constexpr int pixelBlockSizeX = pixel_block_size_x<DstT>::value;
        constexpr int pixelBlockSizeY = pixel_block_size_y<DstT>::value;

        const MorphOp op;
        const PostOp postOp;

        switch (aBorderType)
        {
            case opp::BorderType::None:
            {
                using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeMorphologyKernelDefault<DstT, TupelSize, FilterT, pixelBlockSizeX, pixelBlockSizeY, BCType,
                                              MorphOp, PostOp>(bc, aDst, aPitchDst, aMask, aFilterArea, aSize, op,
                                                               postOp, aStreamCtx);
            }
            break;
            case opp::BorderType::Constant:
            {
                using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi, aConstant);

                InvokeMorphologyKernelDefault<DstT, TupelSize, FilterT, pixelBlockSizeX, pixelBlockSizeY, BCType,
                                              MorphOp, PostOp>(bc, aDst, aPitchDst, aMask, aFilterArea, aSize, op,
                                                               postOp, aStreamCtx);
            }
            break;
            case opp::BorderType::Replicate:
            {
                using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeMorphologyKernelDefault<DstT, TupelSize, FilterT, pixelBlockSizeX, pixelBlockSizeY, BCType,
                                              MorphOp, PostOp>(bc, aDst, aPitchDst, aMask, aFilterArea, aSize, op,
                                                               postOp, aStreamCtx);
            }
            break;
            case opp::BorderType::Mirror:
            {
                using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeMorphologyKernelDefault<DstT, TupelSize, FilterT, pixelBlockSizeX, pixelBlockSizeY, BCType,
                                              MorphOp, PostOp>(bc, aDst, aPitchDst, aMask, aFilterArea, aSize, op,
                                                               postOp, aStreamCtx);
            }
            break;
            case opp::BorderType::MirrorReplicate:
            {
                using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeMorphologyKernelDefault<DstT, TupelSize, FilterT, pixelBlockSizeX, pixelBlockSizeY, BCType,
                                              MorphOp, PostOp>(bc, aDst, aPitchDst, aMask, aFilterArea, aSize, op,
                                                               postOp, aStreamCtx);
            }
            break;
            case opp::BorderType::Wrap:
            {
                using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeMorphologyKernelDefault<DstT, TupelSize, FilterT, pixelBlockSizeX, pixelBlockSizeY, BCType,
                                              MorphOp, PostOp>(bc, aDst, aPitchDst, aMask, aFilterArea, aSize, op,
                                                               postOp, aStreamCtx);
            }
            break;
            default:
                throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for Erosion.");
                break;
        }
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeErosion<typeSrc, typeDst>(                                                                     \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst, const Pixel8uC1 *aMask,              \
        const FilterArea &aFilterArea, BorderType aBorderType, const typeSrc &aConstant,                               \
        const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,                \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(typeSrc, typeDst)                                                                      \
    Instantiate_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                           \
    Instantiate_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                           \
    Instantiate_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                           \
    Instantiate_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                                           \
    Instantiate_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

ForAllChannelsWithAlpha(8u, 8u);
ForAllChannelsWithAlpha(8s, 8s);

ForAllChannelsWithAlpha(16u, 16u);
ForAllChannelsWithAlpha(16s, 16s);

ForAllChannelsWithAlpha(32u, 32u);
ForAllChannelsWithAlpha(32s, 32s);

ForAllChannelsWithAlpha(16f, 16f);
ForAllChannelsWithAlpha(16bf, 16bf);
ForAllChannelsWithAlpha(32f, 32f);
ForAllChannelsWithAlpha(64f, 64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
