#if MPP_ENABLE_CUDA_BACKEND

#include "integral.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/dataExchangeAndInit/transpose.h>
#include <backends/cuda/image/integralXKernel.h>
#include <backends/cuda/image/integralYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
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
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeIntegralSrc(const SrcT *aSrc, size_t aPitchSrc, DstT *aTemp, size_t aPitchTemp, DstT *aDst, size_t aPitchDst,
                       const DstT &aStartValue, const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        InvokeIntegralXKernelDefault<SrcT, DstT, TupelSize>(aSrc, aPitchSrc, aDst, aPitchDst, aSizeDst, aStreamCtx);

        const Size2D sizeTrans(aSizeDst.y, aSizeDst.x);

        InvokeTransposeSrc<DstT>(aDst, aPitchDst, aTemp, aPitchTemp, sizeTrans, aStreamCtx);

        InvokeIntegralYKernelDefault<DstT, TupelSize>(aTemp, aPitchTemp, aStartValue, sizeTrans, aStreamCtx);

        InvokeTransposeSrc<DstT>(aTemp, aPitchTemp, aDst, aPitchDst, aSizeDst, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeIntegralSrc(const typeSrc *aSrc, size_t aPitchSrc, typeDst *aTemp, size_t aPitchTemp,          \
                                    typeDst *aDst, size_t aPitchDst, const typeDst &aStartValue,                       \
                                    const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(typeSrc, typeDst)                                                                        \
    Instantiate_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                           \
    Instantiate_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                           \
    Instantiate_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                           \
    Instantiate_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);
#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
