#if OPP_ENABLE_CUDA_BACKEND

#include "cannyEdgeHysteresis.h"
#include <backends/cuda/image/cannyEdgeHysteresisKernel.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/image/filterArea.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/borderControl.h>
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

template <typename SrcT, typename DstT>
void InvokeCannyEdgeHysteresis(const SrcT *aSrc1, size_t aPitchSrc1, const Pixel32fC1 *aSrcAngle, size_t aPitchSrcAngle,
                               DstT *aDst, size_t aPitchDst, const Size2D &aAllowedReadRoiSize,
                               const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,
                               const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
        const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aOffsetToActualRoi);

        InvokeCannyEdgeHysteresisKernelDefault<DstT, TupelSize, BCType>(bc, aSrcAngle, aPitchSrcAngle, aDst, aPitchDst,
                                                                        aSize, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeCannyEdgeHysteresis<typeSrc, typeDst>(                                                         \
        const typeSrc *aSrc1, size_t aPitchSrc1, const Pixel32fC1 *aSrcAngle, size_t aPitchSrcAngle, typeDst *aDst,    \
        size_t aPitchDst, const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi,                   \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(typeSrc, typeDst) Instantiate_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);

#pragma endregion
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
