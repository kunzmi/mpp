#if OPP_ENABLE_CUDA_BACKEND

#include "averageRelativeError.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongXKernel.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcSrcReductionFunctor.h>
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
void InvokeAverageRelativeErrorSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2,
                                      ComputeT *aTempBuffer, DstT *aDst, remove_vector_t<DstT> *aDstScalar,
                                      const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using avgErrorSrcSrc =
            SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, opp::AverageRelativeError<SrcT, ComputeT>>;

        const opp::AverageRelativeError<SrcT, ComputeT> op;

        const avgErrorSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeReductionAlongXKernelDefault<SrcT, ComputeT, TupelSize, avgErrorSrcSrc, opp::Sum<ComputeT, ComputeT>,
                                           ReductionInitValue::Zero>(aSrc1, aTempBuffer, aSize, aStreamCtx, functor);

        const opp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(aSize.TotalSize()));

        const opp::DivScalar<DstT> postOpScalar(static_cast<remove_vector_t<DstT>>(aSize.TotalSize()));

        InvokeReductionAlongYKernelDefault<ComputeT, DstT, opp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                           opp::DivPostOp<DstT>, opp::DivScalar<DstT>>(
            aTempBuffer, aDst, aDstScalar, aSize.y, postOp, postOpScalar, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeAverageRelativeErrorSrcSrc<typeSrc, averageRelativeError_types_for_ct<typeSrc>,                \
                                                   averageRelativeError_types_for_rt<typeSrc>>(                        \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2,                              \
        averageRelativeError_types_for_ct<typeSrc> *aTemp, averageRelativeError_types_for_rt<typeSrc> *aDst,           \
        remove_vector_t<averageRelativeError_types_for_rt<typeSrc>> *aDstScalar, const Size2D &aSize,                  \
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

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
