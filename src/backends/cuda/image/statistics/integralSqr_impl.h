#include "integralSqr.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/dataExchangeAndInit/transpose.h>
#include <backends/cuda/image/integralSqrXKernel.h>
#include <backends/cuda/image/integralYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
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
template <typename SrcT, typename ComputeT, typename DstT, typename DstSqrT>
void InvokeIntegralSqrSrc(const SrcT *aSrc, size_t aPitchSrc, DstT *aTemp, size_t aPitchTemp, DstSqrT *aTemp2,
                          size_t aPitchTemp2, DstT *aDst, size_t aPitchDst, DstSqrT *aDstSqr, size_t aPitchDstSqr,
                          const DstT &aStartValue, const DstSqrT &aStartValueSqr, const Size2D &aSizeDst,
                          const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    // DstSqrT is always 64 wide, so TupelSize is always 1 for DstSqrT

    InvokeIntegralSqrXKernelDefault<SrcT, DstT, DstSqrT, 1>(aSrc, aPitchSrc, aDst, aPitchDst, aDstSqr, aPitchDstSqr,
                                                            aSizeDst, aStreamCtx);

    const Size2D sizeTrans(aSizeDst.y, aSizeDst.x);

    InvokeTransposeSrc<DstT>(aDst, aPitchDst, aTemp, aPitchTemp, sizeTrans, aStreamCtx);
    InvokeTransposeSrc<DstSqrT>(aDstSqr, aPitchDstSqr, aTemp2, aPitchTemp2, sizeTrans, aStreamCtx);

    InvokeIntegralYKernelDefault<DstT, TupelSize>(aTemp, aPitchTemp, aStartValue, sizeTrans, aStreamCtx);
    InvokeIntegralYKernelDefault<DstSqrT, 1>(aTemp2, aPitchTemp2, aStartValueSqr, sizeTrans, aStreamCtx);

    InvokeTransposeSrc<DstT>(aTemp, aPitchTemp, aDst, aPitchDst, aSizeDst, aStreamCtx);
    InvokeTransposeSrc<DstSqrT>(aTemp2, aPitchTemp2, aDstSqr, aPitchDstSqr, aSizeDst, aStreamCtx);
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst, typeDstSqr)                                                                  \
    template void InvokeIntegralSqrSrc(const typeSrc *aSrc, size_t aPitchSrc, typeDst *aTemp1, size_t aPitchTemp1,     \
                                       typeDstSqr *aTemp2, size_t aPitchTemp2, typeDst *aDst, size_t aPitchDst,        \
                                       typeDstSqr *aDstSqr, size_t aPitchDstSqr, const typeDst &aStartValue,           \
                                       const typeDstSqr &aStartValueSqr, const Size2D &aSizeDst,                       \
                                       const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(typeSrc, typeDst, typeDstSqr)                                                            \
    Instantiate_For(Pixel##typeSrc##C1, Pixel##typeDst##C1, Pixel##typeDstSqr##C1);                                    \
    Instantiate_For(Pixel##typeSrc##C2, Pixel##typeDst##C2, Pixel##typeDstSqr##C2);                                    \
    Instantiate_For(Pixel##typeSrc##C3, Pixel##typeDst##C3, Pixel##typeDstSqr##C3);                                    \
    Instantiate_For(Pixel##typeSrc##C4, Pixel##typeDst##C4, Pixel##typeDstSqr##C4);

#pragma endregion

} // namespace mpp::image::cuda
