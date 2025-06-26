#if MPP_ENABLE_CUDA_BACKEND

#include "meanStdMasked.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reduction2MaskedCountingAlongXKernel.h>
#include <backends/cuda/image/reduction2MaskedCountingAlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcReduction2Functor.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/statistics/operators.h>
#include <common/statistics/postOperators.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT1, typename DstT2>
void InvokeMeanStdMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                            ComputeT *aTempBuffer1, ComputeT *aTempBuffer2, ulong64 *aMaskBuffer, DstT1 *aDst1,
                            DstT2 *aDst2, remove_vector_t<DstT1> *aDstScalar1, remove_vector_t<DstT2> *aDstScalar2,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        {
            using DstT = DstT1;
            MPP_CUDA_REGISTER_TEMPALTE;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using sumSumSqrSrc = SrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::Sum<SrcT, ComputeT>,
                                                  mpp::SumSqr<SrcT, ComputeT>>;

        const mpp::Sum<SrcT, ComputeT> op1;
        const mpp::SumSqr<SrcT, ComputeT> op2;

        const sumSumSqrSrc functor(aSrc, aPitchSrc, op1, op2);

        InvokeReduction2MaskedCountingAlongXKernelDefault<SrcT, ComputeT, ComputeT, TupelSize, sumSumSqrSrc,
                                                          mpp::Sum<ComputeT, ComputeT>, mpp::Sum<ComputeT, ComputeT>,
                                                          ReductionInitValue::Zero, ReductionInitValue::Zero>(
            aMask, aPitchMask, aSrc, aTempBuffer1, aTempBuffer2, aMaskBuffer, aSize, aStreamCtx, functor);

        InvokeReduction2MaskedCountingAlongYKernelDefault<
            ComputeT, ComputeT, DstT1, DstT2, mpp::Sum<DstT1, DstT1>, mpp::Sum<DstT1, DstT1>, ReductionInitValue::Zero,
            ReductionInitValue::Zero, mpp::DivPostOp<DstT1>, mpp::StdDeviation<DstT2>, mpp::DivScalar<DstT1>,
            mpp::StdDeviation<DstT2>>(aMaskBuffer, aTempBuffer1, aTempBuffer2, aDst1, aDst2, aDstScalar1, aDstScalar2,
                                      aSize.y, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeMeanStdMaskedSrc<typeSrc, meanStd_types_for_ct<typeSrc>, meanStd_types_for_rt1<typeSrc>,       \
                                         meanStd_types_for_rt2<typeSrc>>(                                              \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrc *aSrc, size_t aPitchSrc1,                             \
        meanStd_types_for_ct<typeSrc> *aTemp1, meanStd_types_for_ct<typeSrc> *aTemp2, ulong64 *aMaskBuffer,            \
        meanStd_types_for_rt1<typeSrc> *aDst1, meanStd_types_for_rt2<typeSrc> *aDst2,                                  \
        remove_vector_t<meanStd_types_for_rt1<typeSrc>> *aDstScalar1,                                                  \
        remove_vector_t<meanStd_types_for_rt2<typeSrc>> *aDstScalar2, const Size2D &aSize,                             \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(typeIn)                                                                                  \
    Instantiate_For(Pixel##typeIn##C1);                                                                                \
    Instantiate_For(Pixel##typeIn##C2);                                                                                \
    Instantiate_For(Pixel##typeIn##C3);                                                                                \
    Instantiate_For(Pixel##typeIn##C4);

#define ForAllChannelsWithAlpha(typeIn)                                                                                \
    Instantiate_For(Pixel##typeIn##C1);                                                                                \
    Instantiate_For(Pixel##typeIn##C2);                                                                                \
    Instantiate_For(Pixel##typeIn##C3);                                                                                \
    Instantiate_For(Pixel##typeIn##C4);                                                                                \
    Instantiate_For(Pixel##typeIn##C4A);
#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
