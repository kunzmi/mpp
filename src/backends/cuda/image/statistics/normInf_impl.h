#if OPP_ENABLE_CUDA_BACKEND

#include "normInf.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongXKernel.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcReductionFunctor.h>
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
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeNormInfSrc(const SrcT *aSrc, size_t aPitchSrc, ComputeT *aTempBuffer, DstT *aDst,
                      remove_vector_t<DstT> *aDstScalar, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using normInfSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, opp::NormInf<SrcT, ComputeT>>;

        const opp::NormInf<SrcT, ComputeT> op;

        const normInfSrc functor(aSrc, aPitchSrc, op);

        InvokeReductionAlongXKernelDefault<SrcT, ComputeT, TupelSize, normInfSrc, opp::MaxRed<ComputeT>,
                                           ReductionInitValue::Zero>(aSrc, aTempBuffer, aSize, aStreamCtx, functor);

        const opp::Nothing<DstT> postOp;
        const opp::MaxScalar<DstT> postOpScalar;

        InvokeReductionAlongYKernelDefault<ComputeT, DstT, opp::MaxRed<DstT>, ReductionInitValue::Zero,
                                           opp::Nothing<DstT>, opp::MaxScalar<DstT>>(
            aTempBuffer, aDst, aDstScalar, aSize.y, postOp, postOpScalar, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeNormInfSrc<typeSrc, normInf_types_for_ct<typeSrc>, normInf_types_for_rt<typeSrc>>(             \
        const typeSrc *aSrc, size_t aPitchSrc1, normInf_types_for_ct<typeSrc> *aTemp,                                  \
        normInf_types_for_rt<typeSrc> *aDst, remove_vector_t<normInf_types_for_rt<typeSrc>> *aDstScalar,               \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(typeIn)                                                                                \
    Instantiate_For(Pixel##typeIn##C1);                                                                                \
    Instantiate_For(Pixel##typeIn##C2);                                                                                \
    Instantiate_For(Pixel##typeIn##C3);                                                                                \
    Instantiate_For(Pixel##typeIn##C4);                                                                                \
    Instantiate_For(Pixel##typeIn##C4A);
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
