#if MPP_ENABLE_CUDA_BACKEND

#include "prewittHoriz.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/fixedFilterKernel.h>
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
void InvokePrewittHoriz(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, MaskSize aMaskSize,
                        BorderType aBorderType, const SrcT &aConstant, const Size2D &aAllowedReadRoiSize,
                        const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,
                        const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
        using ComputeT             = filter_integer_compute_type_for_t<SrcT>;
        using FilterT              = filtertype_for_t<ComputeT>;

        constexpr int pixelBlockSizeX = pixel_block_size_x<DstT>::value;
        constexpr int pixelBlockSizeY = pixel_block_size_y<DstT>::value;

        switch (aMaskSize)
        {
            case MaskSize::Mask_3x3:
            {
                constexpr int filterSize  = 3;
                constexpr int centerPixel = 1;

                using FixedFilterKernelT = FixedFilterKernel<mpp::FixedFilter::PrewittHoriz, filterSize, FilterT>;

                switch (aBorderType)
                {
                    case mpp::BorderType::None:
                    {
                        using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
                        const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                        InvokeFixedFilterKernelDefault<ComputeT, DstT, TupelSize, filterSize, filterSize, centerPixel,
                                                       centerPixel, pixelBlockSizeX, pixelBlockSizeY,
                                                       RoundingMode::NearestTiesToEven, BCType, FixedFilterKernelT>(
                            bc, aDst, aPitchDst, aSize, aStreamCtx);
                    }
                    break;
                    case mpp::BorderType::Constant:
                    {
                        using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
                        const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi, aConstant);

                        InvokeFixedFilterKernelDefault<ComputeT, DstT, TupelSize, filterSize, filterSize, centerPixel,
                                                       centerPixel, pixelBlockSizeX, pixelBlockSizeY,
                                                       RoundingMode::NearestTiesToEven, BCType, FixedFilterKernelT>(
                            bc, aDst, aPitchDst, aSize, aStreamCtx);
                    }
                    break;
                    case mpp::BorderType::Replicate:
                    {
                        using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
                        const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                        InvokeFixedFilterKernelDefault<ComputeT, DstT, TupelSize, filterSize, filterSize, centerPixel,
                                                       centerPixel, pixelBlockSizeX, pixelBlockSizeY,
                                                       RoundingMode::NearestTiesToEven, BCType, FixedFilterKernelT>(
                            bc, aDst, aPitchDst, aSize, aStreamCtx);
                    }
                    break;
                    case mpp::BorderType::Mirror:
                    {
                        using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
                        const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                        InvokeFixedFilterKernelDefault<ComputeT, DstT, TupelSize, filterSize, filterSize, centerPixel,
                                                       centerPixel, pixelBlockSizeX, pixelBlockSizeY,
                                                       RoundingMode::NearestTiesToEven, BCType, FixedFilterKernelT>(
                            bc, aDst, aPitchDst, aSize, aStreamCtx);
                    }
                    break;
                    case mpp::BorderType::MirrorReplicate:
                    {
                        using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
                        const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                        InvokeFixedFilterKernelDefault<ComputeT, DstT, TupelSize, filterSize, filterSize, centerPixel,
                                                       centerPixel, pixelBlockSizeX, pixelBlockSizeY,
                                                       RoundingMode::NearestTiesToEven, BCType, FixedFilterKernelT>(
                            bc, aDst, aPitchDst, aSize, aStreamCtx);
                    }
                    break;
                    case mpp::BorderType::Wrap:
                    {
                        using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
                        const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

                        InvokeFixedFilterKernelDefault<ComputeT, DstT, TupelSize, filterSize, filterSize, centerPixel,
                                                       centerPixel, pixelBlockSizeX, pixelBlockSizeY,
                                                       RoundingMode::NearestTiesToEven, BCType, FixedFilterKernelT>(
                            bc, aDst, aPitchDst, aSize, aStreamCtx);
                    }
                    break;
                    default:
                        throw INVALIDARGUMENT(aBorderType,
                                              aBorderType
                                                  << " is not a supported border type mode for PrewittHoriz filter.");
                        break;
                }
            }
            break;
            default:
                throw INVALIDARGUMENT(aMaskSize, "Invalid MaskSize for PrewittHoriz filter: " << aMaskSize);
                break;
        }
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokePrewittHoriz<typeSrc, typeDst>(                                                                \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst, MaskSize aMaskSize,                  \
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
#endif // MPP_ENABLE_CUDA_BACKEND
