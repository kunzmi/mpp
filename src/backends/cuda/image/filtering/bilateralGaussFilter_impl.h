#include "bilateralGaussFilter.h"
#include <backends/cuda/image/bilateralGaussFilterKernel.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/filterArea.h>
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

template <typename T> struct pixel_block_size_y
{
    constexpr static int value = 1;
};
template <typename T>
    requires(sizeof(remove_vector_t<T>) == 2)
struct pixel_block_size_y<T>
{
    constexpr static int value = 1;
};
template <typename T>
    requires(sizeof(remove_vector_t<T>) == 1)
struct pixel_block_size_y<T>
{
    constexpr static int value = 2;
};

template <typename T> struct tupel_size
{
    constexpr static size_t value = ConfigTupelSize<"Default", sizeof(T)>::value;
};
template <typename T>
    requires(ConfigTupelSize<"Default", sizeof(T)>::value >= 4)
struct tupel_size<T>
{
    constexpr static size_t value = 2;
};

template <typename SrcT, typename DstT>
void InvokeBilateralGaussFilter(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                                const FilterArea &aFilterArea, const Pixel32fC1 *aPreCompGeomDistCoeff,
                                float aValSquareSigma, mpp::Norm aNorm, BorderType aBorderType, const SrcT &aConstant,
                                const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

    constexpr size_t TupelSize = tupel_size<DstT>::value;
    using ComputeT             = filter_compute_type_for_t<SrcT>;
    using FilterT              = filtertype_for_t<ComputeT>;

    constexpr int pixelBlockSizeX = pixel_block_size_x<DstT>::value;
    constexpr int pixelBlockSizeY = pixel_block_size_y<DstT>::value;

    switch (aBorderType)
    {
        case mpp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            if (aNorm == mpp::Norm::L1 || vector_active_size_v<SrcT> == 1)
            {
                InvokeBilateralGaussFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeX, pixelBlockSizeY,
                                                        RoundingMode::NearestTiesToEven, BCType, mpp::Norm::L1>(
                    bc, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, aStreamCtx);
            }
            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                if (aNorm == mpp::Norm::L2)
                {
                    InvokeBilateralGaussFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeX, pixelBlockSizeY,
                                                            RoundingMode::NearestTiesToEven, BCType, mpp::Norm::L2>(
                        bc, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, aStreamCtx);
                }
                else
                {
                    throw INVALIDARGUMENT(aNorm, aNorm << " is not a supported norm for Bilateral Gauss Filter.");
                }
            }
        }
        break;
        case mpp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi, aConstant);

            if (aNorm == mpp::Norm::L1 || vector_active_size_v<SrcT> == 1)
            {
                InvokeBilateralGaussFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeX, pixelBlockSizeY,
                                                        RoundingMode::NearestTiesToEven, BCType, mpp::Norm::L1>(
                    bc, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, aStreamCtx);
            }
            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                if (aNorm == mpp::Norm::L2)
                {
                    InvokeBilateralGaussFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeX, pixelBlockSizeY,
                                                            RoundingMode::NearestTiesToEven, BCType, mpp::Norm::L2>(
                        bc, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, aStreamCtx);
                }
                else
                {
                    throw INVALIDARGUMENT(aNorm, aNorm << " is not a supported norm for Bilateral Gauss Filter.");
                }
            }
        }
        break;
        case mpp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            if (aNorm == mpp::Norm::L1 || vector_active_size_v<SrcT> == 1)
            {
                InvokeBilateralGaussFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeX, pixelBlockSizeY,
                                                        RoundingMode::NearestTiesToEven, BCType, mpp::Norm::L1>(
                    bc, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, aStreamCtx);
            }
            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                if (aNorm == mpp::Norm::L2)
                {
                    InvokeBilateralGaussFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeX, pixelBlockSizeY,
                                                            RoundingMode::NearestTiesToEven, BCType, mpp::Norm::L2>(
                        bc, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, aStreamCtx);
                }
                else
                {
                    throw INVALIDARGUMENT(aNorm, aNorm << " is not a supported norm for Bilateral Gauss Filter.");
                }
            }
        }
        break;
        case mpp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            if (aNorm == mpp::Norm::L1 || vector_active_size_v<SrcT> == 1)
            {
                InvokeBilateralGaussFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeX, pixelBlockSizeY,
                                                        RoundingMode::NearestTiesToEven, BCType, mpp::Norm::L1>(
                    bc, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, aStreamCtx);
            }
            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                if (aNorm == mpp::Norm::L2)
                {
                    InvokeBilateralGaussFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeX, pixelBlockSizeY,
                                                            RoundingMode::NearestTiesToEven, BCType, mpp::Norm::L2>(
                        bc, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, aStreamCtx);
                }
                else
                {
                    throw INVALIDARGUMENT(aNorm, aNorm << " is not a supported norm for Bilateral Gauss Filter.");
                }
            }
        }
        break;
        case mpp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            if (aNorm == mpp::Norm::L1 || vector_active_size_v<SrcT> == 1)
            {
                InvokeBilateralGaussFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeX, pixelBlockSizeY,
                                                        RoundingMode::NearestTiesToEven, BCType, mpp::Norm::L1>(
                    bc, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, aStreamCtx);
            }
            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                if (aNorm == mpp::Norm::L2)
                {
                    InvokeBilateralGaussFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeX, pixelBlockSizeY,
                                                            RoundingMode::NearestTiesToEven, BCType, mpp::Norm::L2>(
                        bc, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, aStreamCtx);
                }
                else
                {
                    throw INVALIDARGUMENT(aNorm, aNorm << " is not a supported norm for Bilateral Gauss Filter.");
                }
            }
        }
        break;
        case mpp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            if (aNorm == mpp::Norm::L1 || vector_active_size_v<SrcT> == 1)
            {
                InvokeBilateralGaussFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeX, pixelBlockSizeY,
                                                        RoundingMode::NearestTiesToEven, BCType, mpp::Norm::L1>(
                    bc, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, aStreamCtx);
            }
            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                if (aNorm == mpp::Norm::L2)
                {
                    InvokeBilateralGaussFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeX, pixelBlockSizeY,
                                                            RoundingMode::NearestTiesToEven, BCType, mpp::Norm::L2>(
                        bc, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, aStreamCtx);
                }
                else
                {
                    throw INVALIDARGUMENT(aNorm, aNorm << " is not a supported norm for Bilateral Gauss Filter.");
                }
            }
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType,
                                  aBorderType << " is not a supported border type mode for Bilateral Gauss Filter.");
            break;
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeBilateralGaussFilter<typeSrc, typeDst>(                                                        \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst, const FilterArea &aFilterArea,       \
        const Pixel32fC1 *aPreCompGeomDistCoeff, float aValSquareSigma, mpp::Norm aNorm, BorderType aBorderType,       \
        const typeSrc &aConstant, const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi,           \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

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
