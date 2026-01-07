#include "fixedSizeMedianFilter.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/fixedSizeMedianFilterKernel.h>
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

template <typename SrcT, typename DstT>
void InvokeFixedSizeMedianFilter(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, int aFilterSize,
                                 const Vector2<int> &aFilterCenter, BorderType aBorderType,
                                 const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi,
                                 const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    using ComputeT             = SrcT;

    constexpr int pixelBlockSizeY = 1;

    switch (aBorderType)
    {
        case mpp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

            InvokeFixedSizeMedianFilterKernelDefault<ComputeT, DstT, TupelSize, pixelBlockSizeY, BCType>(
                bc, aDst, aPitchDst, aFilterSize, aFilterCenter, aSize, aStreamCtx);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType,
                                  aBorderType << " is not a supported border type mode for median filter.");
            break;
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeFixedSizeMedianFilter<typeSrcIsTypeDst, typeSrcIsTypeDst>(                                     \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst, int aFilterSize,   \
        const Vector2<int> &aFilterCenter, BorderType aBorderType, const Size2D &aAllowedReadRoiSize,                  \
        const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize, const StreamCtx &aStreamCtx);

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
} // namespace mpp::image::cuda
