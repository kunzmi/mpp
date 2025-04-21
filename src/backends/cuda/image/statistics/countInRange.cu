#if OPP_ENABLE_CUDA_BACKEND

#include "countInRange.h"
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
void InvokeCountInRangeSrc(const SrcT *aSrc, size_t aPitchSrc, ComputeT *aTempBuffer, DstT *aDst,
                           remove_vector_t<DstT> *aDstScalar, const SrcT &aLowerLimit, const SrcT &aUpperLimit,
                           const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using countInRangeSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, opp::CountInRange<SrcT>>;

        const opp::CountInRange<SrcT> op(aLowerLimit, aUpperLimit);

        const countInRangeSrc functor(aSrc, aPitchSrc, op);

        InvokeReductionAlongXKernelDefault<SrcT, ComputeT, TupelSize, countInRangeSrc, opp::Sum<ComputeT, ComputeT>,
                                           ReductionInitValue::Zero>(aSrc, aTempBuffer, aSize, aStreamCtx, functor);

        const opp::Nothing<DstT> postOp;
        const opp::SumScalar<DstT> postOpScalar;

        InvokeReductionAlongYKernelDefault<ComputeT, DstT, opp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                           opp::Nothing<DstT>, opp::SumScalar<DstT>>(
            aTempBuffer, aDst, aDstScalar, aSize.y, postOp, postOpScalar, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeTemp, typeDst)                                                                    \
    template void InvokeCountInRangeSrc<typeSrc, typeTemp, typeDst>(                                                   \
        const typeSrc *aSrc, size_t aPitchSrc1, typeTemp *aTemp, typeDst *aDst, remove_vector_t<typeDst> *aDstScalar,  \
        const typeSrc &aLowerLimit, const typeSrc &aUpperLimit, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(typeIn, typeCompute, typeDst)                                                          \
    Instantiate_For(Pixel##typeIn##C1, Pixel##typeCompute##C1, Pixel##typeDst##C1);                                    \
    Instantiate_For(Pixel##typeIn##C2, Pixel##typeCompute##C2, Pixel##typeDst##C2);                                    \
    Instantiate_For(Pixel##typeIn##C3, Pixel##typeCompute##C3, Pixel##typeDst##C3);                                    \
    Instantiate_For(Pixel##typeIn##C4, Pixel##typeCompute##C4, Pixel##typeDst##C4);                                    \
    Instantiate_For(Pixel##typeIn##C4A, Pixel##typeCompute##C4A, Pixel##typeDst##C4A);

ForAllChannelsWithAlpha(8u, 64u, 64u);
ForAllChannelsWithAlpha(8s, 64u, 64u);

ForAllChannelsWithAlpha(16u, 64u, 64u);
ForAllChannelsWithAlpha(16s, 64u, 64u);

ForAllChannelsWithAlpha(32u, 64u, 64u);
ForAllChannelsWithAlpha(32s, 64u, 64u);

ForAllChannelsWithAlpha(16f, 64u, 64u);
ForAllChannelsWithAlpha(16bf, 64u, 64u);
ForAllChannelsWithAlpha(32f, 64u, 64u);
ForAllChannelsWithAlpha(64f, 64u, 64u);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
