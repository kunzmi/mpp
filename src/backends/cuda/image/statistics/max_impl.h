#if OPP_ENABLE_CUDA_BACKEND

#include "max.h"
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
template <typename SrcT>
void InvokeMaxSrc(const SrcT *aSrc, size_t aPitchSrc, SrcT *aTempBuffer, SrcT *aDst, remove_vector_t<SrcT> *aDstScalar,
                  const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using maxSrc = SrcReductionFunctor<TupelSize, SrcT, SrcT, opp::MaxRed<SrcT>>;

        const opp::MaxRed<SrcT> op;

        const maxSrc functor(aSrc, aPitchSrc, op);

        InvokeReductionAlongXKernelDefault<SrcT, SrcT, TupelSize, maxSrc, opp::MaxRed<SrcT>, ReductionInitValue::Min>(
            aSrc, aTempBuffer, aSize, aStreamCtx, functor);

        const opp::Nothing<SrcT> postOp;
        const opp::MaxScalar<SrcT> postOpScalar;

        InvokeReductionAlongYKernelDefault<SrcT, SrcT, opp::MaxRed<SrcT>, ReductionInitValue::Min, opp::Nothing<SrcT>,
                                           opp::MaxScalar<SrcT>>(aTempBuffer, aDst, aDstScalar, aSize.y, postOp,
                                                                 postOpScalar, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(type)                                                                                          \
    template void InvokeMaxSrc<type>(const type *aSrc, size_t aPitchSrc1, type *aTemp, type *aDst,                     \
                                     remove_vector_t<type> *aDstScalar, const Size2D &aSize,                           \
                                     const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
