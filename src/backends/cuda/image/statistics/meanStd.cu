#if OPP_ENABLE_CUDA_BACKEND

#include "meanStd.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reduction2AlongXKernel.h>
#include <backends/cuda/image/reduction2AlongYKernel.h>
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
template <typename SrcT, typename ComputeT, typename DstT1, typename DstT2>
void InvokeMeanStdSrc(const SrcT *aSrc, size_t aPitchSrc, ComputeT *aTempBuffer1, ComputeT *aTempBuffer2, DstT1 *aDst1,
                      DstT2 *aDst2, remove_vector_t<DstT1> *aDstScalar1, remove_vector_t<DstT2> *aDstScalar2,
                      const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        {
            using DstT = DstT1;
            OPP_CUDA_REGISTER_TEMPALTE;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using sumSumSqrSrc = SrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, opp::Sum<SrcT, ComputeT>,
                                                  opp::SumSqr<SrcT, ComputeT>>;

        const opp::Sum<SrcT, ComputeT> op1;
        const opp::SumSqr<SrcT, ComputeT> op2;

        const sumSumSqrSrc functor(aSrc, aPitchSrc, op1, op2);

        InvokeReduction2AlongXKernelDefault<SrcT, ComputeT, ComputeT, TupelSize, sumSumSqrSrc,
                                            opp::Sum<ComputeT, ComputeT>, opp::Sum<ComputeT, ComputeT>,
                                            ReductionInitValue::Zero, ReductionInitValue::Zero>(
            aSrc, aTempBuffer1, aTempBuffer2, aSize, aStreamCtx, functor);

        const opp::DivPostOp<DstT1> postOp1(static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(aSize.TotalSize()));
        const opp::StdDeviation<DstT2> postOp2(static_cast<remove_vector_t<DstT2>>(aSize.TotalSize()));
        const opp::DivScalar<DstT1> postOpScalar1(
            static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(aSize.TotalSize()));
        const opp::StdDeviation<DstT2> postOpScalar2((static_cast<remove_vector_t<DstT2>>(aSize.TotalSize())));

        InvokeReduction2AlongYKernelDefault<ComputeT, ComputeT, DstT1, DstT2, opp::Sum<DstT1, DstT1>,
                                            opp::Sum<DstT1, DstT1>, ReductionInitValue::Zero, ReductionInitValue::Zero,
                                            opp::DivPostOp<DstT1>, opp::StdDeviation<DstT2>, opp::DivScalar<DstT1>,
                                            opp::StdDeviation<DstT2>>(
            aTempBuffer1, aTempBuffer2, aDst1, aDst2, aDstScalar1, aDstScalar2, aSize.y, postOp1, postOp2,
            postOpScalar1, postOpScalar2, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeMeanStdSrc<typeSrc, meanStd_types_for_ct<typeSrc>, meanStd_types_for_rt1<typeSrc>,             \
                                   meanStd_types_for_rt2<typeSrc>>(                                                    \
        const typeSrc *aSrc, size_t aPitchSrc1, meanStd_types_for_ct<typeSrc> *aTemp1,                                 \
        meanStd_types_for_ct<typeSrc> *aTemp2, meanStd_types_for_rt1<typeSrc> *aDst1,                                  \
        meanStd_types_for_rt2<typeSrc> *aDst2, remove_vector_t<meanStd_types_for_rt1<typeSrc>> *aDstScalar1,           \
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
