#include "meanStd.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reduction2AlongXKernel.h>
#include <backends/cuda/image/reduction2AlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcReduction2Functor.h>
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
template <typename SrcT, typename ComputeT, typename DstT1, typename DstT2>
void InvokeMeanStdSrc(const SrcT *aSrc, size_t aPitchSrc, ComputeT *aTempBuffer1, ComputeT *aTempBuffer2, DstT1 *aDst1,
                      DstT2 *aDst2, remove_vector_t<DstT1> *aDstScalar1, remove_vector_t<DstT2> *aDstScalar2,
                      const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    {
        using DstT = DstT1;
        MPP_CUDA_REGISTER_TEMPALTE;
    }

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    using sumSumSqrSrc = SrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::Sum<SrcT, ComputeT>,
                                              mpp::SumSqr<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op1;
    const mpp::SumSqr<SrcT, ComputeT> op2;

    const sumSumSqrSrc functor(aSrc, aPitchSrc, op1, op2);

    InvokeReduction2AlongXKernelDefault<SrcT, ComputeT, ComputeT, TupelSize, sumSumSqrSrc, mpp::Sum<ComputeT, ComputeT>,
                                        mpp::Sum<ComputeT, ComputeT>, ReductionInitValue::Zero,
                                        ReductionInitValue::Zero>(aSrc, aTempBuffer1, aTempBuffer2, aSize, aStreamCtx,
                                                                  functor);

    const mpp::DivPostOp<DstT1> postOp1(static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(aSize.TotalSize()));
    const mpp::StdDeviation<DstT2> postOp2(static_cast<remove_vector_t<DstT2>>(aSize.TotalSize()));
    const mpp::DivScalar<DstT1> postOpScalar1(
        static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(aSize.TotalSize()));
    const mpp::StdDeviation<DstT2> postOpScalar2((static_cast<remove_vector_t<DstT2>>(aSize.TotalSize())));

    InvokeReduction2AlongYKernelDefault<ComputeT, ComputeT, DstT1, DstT2, mpp::Sum<DstT1, DstT1>,
                                        mpp::Sum<DstT1, DstT1>, ReductionInitValue::Zero, ReductionInitValue::Zero,
                                        mpp::DivPostOp<DstT1>, mpp::StdDeviation<DstT2>, mpp::DivScalar<DstT1>,
                                        mpp::StdDeviation<DstT2>>(aTempBuffer1, aTempBuffer2, aDst1, aDst2, aDstScalar1,
                                                                  aDstScalar2, aSize.y, postOp1, postOp2, postOpScalar1,
                                                                  postOpScalar2, aStreamCtx);
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
#pragma endregion

} // namespace mpp::image::cuda
