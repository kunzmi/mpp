#if OPP_ENABLE_CUDA_BACKEND

#include "crossCorrelation.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/crossCorrelationKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
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
    constexpr static int value = 4;
};
template <typename T>
    requires(sizeof(remove_vector_t<T>) == 1)
struct pixel_block_size_y<T>
{
    constexpr static int value = 4;
};

template <typename SrcT, typename DstT>
void InvokeCrossCorrelation(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const SrcT *aTemplate,
                            size_t aPitchTemplate, const Size2D &aSizeTemplate, BorderType aBorderType,
                            const SrcT &aConstant, const Size2D &aAllowedReadRoiSize,
                            const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,
                            const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
        using ComputeT             = DstT;

        constexpr int pixelBlockSizeX = pixel_block_size_x<DstT>::value;
        constexpr int pixelBlockSizeY = pixel_block_size_y<DstT>::value;

        switch (aBorderType)
        {
            case opp::BorderType::None:
            {
                using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeCrossCorrelationKernelDefault<ComputeT, DstT, TupelSize, SrcT, pixelBlockSizeX, pixelBlockSizeY,
                                                    RoundingMode::NearestTiesToEven, BCType>(
                    bc, aDst, aPitchDst, aTemplate, aPitchTemplate, aSizeTemplate, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::Constant:
            {
                using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi, aConstant);

                InvokeCrossCorrelationKernelDefault<ComputeT, DstT, TupelSize, SrcT, pixelBlockSizeX, pixelBlockSizeY,
                                                    RoundingMode::NearestTiesToEven, BCType>(
                    bc, aDst, aPitchDst, aTemplate, aPitchTemplate, aSizeTemplate, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::Replicate:
            {
                using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeCrossCorrelationKernelDefault<ComputeT, DstT, TupelSize, SrcT, pixelBlockSizeX, pixelBlockSizeY,
                                                    RoundingMode::NearestTiesToEven, BCType>(
                    bc, aDst, aPitchDst, aTemplate, aPitchTemplate, aSizeTemplate, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::Mirror:
            {
                using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeCrossCorrelationKernelDefault<ComputeT, DstT, TupelSize, SrcT, pixelBlockSizeX, pixelBlockSizeY,
                                                    RoundingMode::NearestTiesToEven, BCType>(
                    bc, aDst, aPitchDst, aTemplate, aPitchTemplate, aSizeTemplate, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::MirrorReplicate:
            {
                using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeCrossCorrelationKernelDefault<ComputeT, DstT, TupelSize, SrcT, pixelBlockSizeX, pixelBlockSizeY,
                                                    RoundingMode::NearestTiesToEven, BCType>(
                    bc, aDst, aPitchDst, aTemplate, aPitchTemplate, aSizeTemplate, aSize, aStreamCtx);
            }
            break;
            case opp::BorderType::Wrap:
            {
                using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
                const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                InvokeCrossCorrelationKernelDefault<ComputeT, DstT, TupelSize, SrcT, pixelBlockSizeX, pixelBlockSizeY,
                                                    RoundingMode::NearestTiesToEven, BCType>(
                    bc, aDst, aPitchDst, aTemplate, aPitchTemplate, aSizeTemplate, aSize, aStreamCtx);
            }
            break;
            default:
                throw INVALIDARGUMENT(aBorderType,
                                      aBorderType << " is not a supported border type mode for CrossCorrelation.");
                break;
        }
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeCrossCorrelation<typeSrc, typeDst>(                                                            \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst, const typeSrc *aTemplate,            \
        size_t aPitchTemplate, const Size2D &aSizeTemplate, BorderType aBorderType, const typeSrc &aConstant,          \
        const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,                \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(typeSrc, typeDst) Instantiate_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
