#if OPP_ENABLE_CUDA_BACKEND

#include "qualityIndex.h" //for datatypes
#include "ssim.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reduction5AlongXKernel.h>
#include <backends/cuda/image/reduction5AlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcSrcReduction5Functor.h>
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
void InvokeSSIMSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2,
                      ComputeT *aTempBuffer1, ComputeT *aTempBuffer2, ComputeT *aTempBuffer3, ComputeT *aTempBuffer4,
                      ComputeT *aTempBuffer5, DstT *aDst, remove_vector_t<DstT> aDynamicRange,
                      remove_vector_t<DstT> aK1, remove_vector_t<DstT> aK2, const Size2D &aSize,
                      const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using ssimSrcSrc = SrcSrcReduction5Functor<TupelSize, SrcT, ComputeT, ComputeT, ComputeT, ComputeT, ComputeT,
                                                   opp::Sum1Or2<SrcT, ComputeT, 1>, opp::SumSqr1Or2<SrcT, ComputeT, 1>,
                                                   opp::Sum1Or2<SrcT, ComputeT, 2>, opp::SumSqr1Or2<SrcT, ComputeT, 2>,
                                                   opp::DotProduct<SrcT, ComputeT>>;

        const opp::Sum1Or2<SrcT, ComputeT, 1> opSum1;
        const opp::SumSqr1Or2<SrcT, ComputeT, 1> opSumSqr1;
        const opp::Sum1Or2<SrcT, ComputeT, 2> opSum2;
        const opp::SumSqr1Or2<SrcT, ComputeT, 2> opSumSqr2;
        const opp::DotProduct<SrcT, ComputeT> opDotProduct;

        const ssimSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, opSum1, opSumSqr1, opSum2, opSumSqr2,
                                 opDotProduct);

        InvokeReduction5AlongXKernelDefault<
            SrcT, ComputeT, ComputeT, ComputeT, ComputeT, ComputeT, TupelSize, ssimSrcSrc, opp::Sum<ComputeT, ComputeT>,
            opp::Sum<ComputeT, ComputeT>, opp::Sum<ComputeT, ComputeT>, opp::Sum<ComputeT, ComputeT>,
            opp::Sum<ComputeT, ComputeT>, ReductionInitValue::Zero, ReductionInitValue::Zero, ReductionInitValue::Zero,
            ReductionInitValue::Zero, ReductionInitValue::Zero>(aSrc1, aTempBuffer1, aTempBuffer2, aTempBuffer3,
                                                                aTempBuffer4, aTempBuffer5, aSize, aStreamCtx, functor);

        const opp::SSIM<DstT> postOp(static_cast<remove_vector_t<DstT>>(aSize.TotalSize()), aDynamicRange, aK1, aK2);

        using ComputeT2 = qualityIndex_types_for_ct2<SrcT>;
        InvokeReduction5AlongYKernelDefault<
            ComputeT, ComputeT, ComputeT, ComputeT, ComputeT, ComputeT2, ComputeT2, ComputeT2, ComputeT2, ComputeT2,
            DstT, opp::Sum<ComputeT2, ComputeT2>, opp::Sum<ComputeT2, ComputeT2>, opp::Sum<ComputeT2, ComputeT2>,
            opp::Sum<ComputeT2, ComputeT2>, opp::Sum<ComputeT2, ComputeT2>, ReductionInitValue::Zero,
            ReductionInitValue::Zero, ReductionInitValue::Zero, ReductionInitValue::Zero, ReductionInitValue::Zero,
            opp::SSIM<DstT>>(aTempBuffer1, aTempBuffer2, aTempBuffer3, aTempBuffer4, aTempBuffer5, aDst, aSize.y,
                             postOp, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeSSIMSrcSrc<typeSrc, qualityIndex_types_for_ct1<typeSrc>, qualityIndex_types_for_rt<typeSrc>>(  \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2,                              \
        qualityIndex_types_for_ct1<typeSrc> *aTemp1, qualityIndex_types_for_ct1<typeSrc> *aTemp2,                      \
        qualityIndex_types_for_ct1<typeSrc> *aTemp3, qualityIndex_types_for_ct1<typeSrc> *aTemp4,                      \
        qualityIndex_types_for_ct1<typeSrc> *aTemp5, qualityIndex_types_for_rt<typeSrc> *aDst,                         \
        remove_vector_t<qualityIndex_types_for_rt<typeSrc>> aDynamicRange,                                             \
        remove_vector_t<qualityIndex_types_for_rt<typeSrc>> aK1,                                                       \
        remove_vector_t<qualityIndex_types_for_rt<typeSrc>> aK2, const Size2D &aSize, const StreamCtx &aStreamCtx);

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
    Instantiate_For(Pixel##typeIn##C4A); /**/

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
