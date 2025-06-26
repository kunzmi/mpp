#if MPP_ENABLE_CUDA_BACKEND

#include "qualityIndex.h"
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
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeQualityIndexSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2,
                              ComputeT *aTempBuffer1, ComputeT *aTempBuffer2, ComputeT *aTempBuffer3,
                              ComputeT *aTempBuffer4, ComputeT *aTempBuffer5, DstT *aDst, const Size2D &aSize,
                              const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using qualityIndexSrcSrc =
            SrcSrcReduction5Functor<TupelSize, SrcT, ComputeT, ComputeT, ComputeT, ComputeT, ComputeT,
                                    mpp::Sum1Or2<SrcT, ComputeT, 1>, mpp::SumSqr1Or2<SrcT, ComputeT, 1>,
                                    mpp::Sum1Or2<SrcT, ComputeT, 2>, mpp::SumSqr1Or2<SrcT, ComputeT, 2>,
                                    mpp::DotProduct<SrcT, ComputeT>>;

        const mpp::Sum1Or2<SrcT, ComputeT, 1> opSum1;
        const mpp::SumSqr1Or2<SrcT, ComputeT, 1> opSumSqr1;
        const mpp::Sum1Or2<SrcT, ComputeT, 2> opSum2;
        const mpp::SumSqr1Or2<SrcT, ComputeT, 2> opSumSqr2;
        const mpp::DotProduct<SrcT, ComputeT> opDotProduct;

        const qualityIndexSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, opSum1, opSumSqr1, opSum2, opSumSqr2,
                                         opDotProduct);

        InvokeReduction5AlongXKernelDefault<
            SrcT, ComputeT, ComputeT, ComputeT, ComputeT, ComputeT, TupelSize, qualityIndexSrcSrc,
            mpp::Sum<ComputeT, ComputeT>, mpp::Sum<ComputeT, ComputeT>, mpp::Sum<ComputeT, ComputeT>,
            mpp::Sum<ComputeT, ComputeT>, mpp::Sum<ComputeT, ComputeT>, ReductionInitValue::Zero,
            ReductionInitValue::Zero, ReductionInitValue::Zero, ReductionInitValue::Zero, ReductionInitValue::Zero>(
            aSrc1, aTempBuffer1, aTempBuffer2, aTempBuffer3, aTempBuffer4, aTempBuffer5, aSize, aStreamCtx, functor);

        const mpp::QualityIndex<DstT> postOp(static_cast<remove_vector_t<DstT>>(aSize.TotalSize()));

        using ComputeT2 = qualityIndex_types_for_ct2<SrcT>;
        InvokeReduction5AlongYKernelDefault<
            ComputeT, ComputeT, ComputeT, ComputeT, ComputeT, ComputeT2, ComputeT2, ComputeT2, ComputeT2, ComputeT2,
            DstT, mpp::Sum<ComputeT2, ComputeT2>, mpp::Sum<ComputeT2, ComputeT2>, mpp::Sum<ComputeT2, ComputeT2>,
            mpp::Sum<ComputeT2, ComputeT2>, mpp::Sum<ComputeT2, ComputeT2>, ReductionInitValue::Zero,
            ReductionInitValue::Zero, ReductionInitValue::Zero, ReductionInitValue::Zero, ReductionInitValue::Zero,
            mpp::QualityIndex<DstT>>(aTempBuffer1, aTempBuffer2, aTempBuffer3, aTempBuffer4, aTempBuffer5, aDst,
                                     aSize.y, postOp, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void                                                                                                      \
    InvokeQualityIndexSrcSrc<typeSrc, qualityIndex_types_for_ct1<typeSrc>, qualityIndex_types_for_rt<typeSrc>>(        \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2,                              \
        qualityIndex_types_for_ct1<typeSrc> *aTemp1, qualityIndex_types_for_ct1<typeSrc> *aTemp2,                      \
        qualityIndex_types_for_ct1<typeSrc> *aTemp3, qualityIndex_types_for_ct1<typeSrc> *aTemp4,                      \
        qualityIndex_types_for_ct1<typeSrc> *aTemp5, qualityIndex_types_for_rt<typeSrc> *aDst, const Size2D &aSize,    \
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
