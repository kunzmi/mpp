#if OPP_ENABLE_CUDA_BACKEND

#include "normRelL1.h"
#include "normRelL1Masked.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reduction2AlongYKernel.h>
#include <backends/cuda/image/reduction2MaskedAlongXKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcSrcReduction2Functor.h>
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
void InvokeNormRelL1MaskedSrcSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                                 const SrcT *aSrc2, size_t aPitchSrc2, ComputeT *aTempBuffer1, ComputeT *aTempBuffer2,
                                 DstT *aDst, remove_vector_t<DstT> *aDstScalar, const Size2D &aSize,
                                 const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using normL1SrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT,
                                                     opp::NormRelL1<SrcT, ComputeT>, opp::NormL1<SrcT, ComputeT>>;

        const opp::NormRelL1<SrcT, ComputeT> op1;
        const opp::NormL1<SrcT, ComputeT> op2;

        const normL1SrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op1, op2);

        InvokeReduction2MaskedAlongXKernelDefault<SrcT, ComputeT, ComputeT, TupelSize, normL1SrcSrc,
                                                  opp::Sum<ComputeT, ComputeT>, opp::Sum<ComputeT, ComputeT>,
                                                  ReductionInitValue::Zero, ReductionInitValue::Zero>(
            aMask, aPitchMask, aSrc1, aTempBuffer1, aTempBuffer2, aSize, aStreamCtx, functor);

        const opp::Nothing<DstT> postOp1;
        const opp::NormRelL1Post<DstT> postOp2;

        const opp::NothingScalar<DstT> postOpScalar1;
        const opp::NormRelL1Post<DstT> postOpScalar2;

        // ignore output on Dst1, the postOp only creates a meaning output value on Dst2:
        InvokeReduction2AlongYKernelDefault<ComputeT, ComputeT, DstT, DstT, opp::Sum<DstT, DstT>, opp::Sum<DstT, DstT>,
                                            ReductionInitValue::Zero, ReductionInitValue::Zero, opp::Nothing<DstT>,
                                            opp::NormRelL1Post<DstT>, opp::NothingScalar<DstT>,
                                            opp::NormRelL1Post<DstT>>(
            aTempBuffer1, aTempBuffer2, reinterpret_cast<DstT *>(aTempBuffer1), aDst,
            reinterpret_cast<remove_vector_t<DstT> *>(aTempBuffer1), aDstScalar, aSize.y, postOp1, postOp2,
            postOpScalar1, postOpScalar2, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void                                                                                                      \
    InvokeNormRelL1MaskedSrcSrc<typeSrc, normRelL1_types_for_ct<typeSrc>, normRelL1_types_for_rt<typeSrc>>(            \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2,      \
        size_t aPitchSrc2, normRelL1_types_for_ct<typeSrc> *aTemp1, normRelL1_types_for_ct<typeSrc> *aTemp2,           \
        normRelL1_types_for_rt<typeSrc> *aDst, remove_vector_t<normRelL1_types_for_rt<typeSrc>> *aDstScalar,           \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

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
