#if OPP_ENABLE_CUDA_BACKEND

#include "normDiffL2.h"
#include "normDiffL2Masked.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
#include <backends/cuda/image/reductionMaskedAlongXKernel.h>
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
void InvokeNormDiffL2MaskedSrcSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                                  const SrcT *aSrc2, size_t aPitchSrc2, ComputeT *aTempBuffer, DstT *aDst,
                                  remove_vector_t<DstT> *aDstScalar, const Size2D &aSize,
                                  const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, opp::NormDiffL2<SrcT, ComputeT>>;

        const opp::NormDiffL2<SrcT, ComputeT> op;

        const normL2SrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeReductionMaskedAlongXKernelDefault<SrcT, ComputeT, TupelSize, normL2SrcSrc, opp::Sum<ComputeT, ComputeT>,
                                                 ReductionInitValue::Zero>(aMask, aPitchMask, aSrc1, aTempBuffer, aSize,
                                                                           aStreamCtx, functor);

        const opp::SqrtPostOp<DstT> postOp;

        const opp::SumThenSqrtScalar<DstT> postOpScalar;

        InvokeReductionAlongYKernelDefault<ComputeT, DstT, opp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                           opp::SqrtPostOp<DstT>, opp::SumThenSqrtScalar<DstT>>(
            aTempBuffer, aDst, aDstScalar, aSize.y, postOp, postOpScalar, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void                                                                                                      \
    InvokeNormDiffL2MaskedSrcSrc<typeSrc, normDiffL2_types_for_ct<typeSrc>, normDiffL2_types_for_rt<typeSrc>>(         \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2,      \
        size_t aPitchSrc2, normDiffL2_types_for_ct<typeSrc> *aTemp, normDiffL2_types_for_rt<typeSrc> *aDst,            \
        remove_vector_t<normDiffL2_types_for_rt<typeSrc>> *aDstScalar, const Size2D &aSize,                            \
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

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
