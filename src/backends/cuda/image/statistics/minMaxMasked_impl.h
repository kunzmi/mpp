#if OPP_ENABLE_CUDA_BACKEND

#include "minMaxMasked.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reduction2AlongYKernel.h>
#include <backends/cuda/image/reduction2MaskedAlongXKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcReduction2Functor.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/statistics/operators.h>
#include <common/statistics/postOperators.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{
template <typename SrcT>
void InvokeMinMaxMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                           SrcT *aTempBuffer1, SrcT *aTempBuffer2, SrcT *aDstMin, SrcT *aDstMax,
                           remove_vector_t<SrcT> *aDstMinScalar, remove_vector_t<SrcT> *aDstMaxScalar,
                           const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using minMaxSrc = SrcReduction2Functor<TupelSize, SrcT, SrcT, SrcT, opp::MinRed<SrcT>, opp::MaxRed<SrcT>>;

        const opp::MinRed<SrcT> op1;
        const opp::MaxRed<SrcT> op2;

        const minMaxSrc functor(aSrc, aPitchSrc, op1, op2);

        InvokeReduction2MaskedAlongXKernelDefault<SrcT, SrcT, SrcT, TupelSize, minMaxSrc, opp::MinRed<SrcT>,
                                                  opp::MaxRed<SrcT>, ReductionInitValue::Max, ReductionInitValue::Min>(
            aMask, aPitchMask, aSrc, aTempBuffer1, aTempBuffer2, aSize, aStreamCtx, functor);

        const opp::Nothing<SrcT> postOp1;
        const opp::Nothing<SrcT> postOp2;
        const opp::MinScalar<SrcT> postOpScalar1;
        const opp::MaxScalar<SrcT> postOpScalar2;

        InvokeReduction2AlongYKernelDefault<SrcT, SrcT, SrcT, SrcT, opp::MinRed<SrcT>, opp::MaxRed<SrcT>,
                                            ReductionInitValue::Max, ReductionInitValue::Min, opp::Nothing<SrcT>,
                                            opp::Nothing<SrcT>, opp::MinScalar<SrcT>, opp::MaxScalar<SrcT>>(
            aTempBuffer1, aTempBuffer2, aDstMin, aDstMax, aDstMinScalar, aDstMaxScalar, aSize.y, postOp1, postOp2,
            postOpScalar1, postOpScalar2, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(type)                                                                                          \
    template void InvokeMinMaxMaskedSrc<type>(                                                                         \
        const Pixel8uC1 *aMask, size_t aPitchMask, const type *aSrc, size_t aPitchSrc1, type *aTemp1, type *aTemp2,    \
        type *aDstMin, type *aDstMax, remove_vector_t<type> *aDstMinScalar, remove_vector_t<type> *aDstMaxScalar,      \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
