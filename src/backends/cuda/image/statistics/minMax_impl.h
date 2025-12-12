#include "minMax.h"
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
template <typename SrcT>
void InvokeMinMaxSrc(const SrcT *aSrc, size_t aPitchSrc, SrcT *aTempBuffer1, SrcT *aTempBuffer2, SrcT *aDstMin,
                     SrcT *aDstMax, remove_vector_t<SrcT> *aDstMinScalar, remove_vector_t<SrcT> *aDstMaxScalar,
                     const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    using minMaxSrc = SrcReduction2Functor<TupelSize, SrcT, SrcT, SrcT, mpp::MinRed<SrcT>, mpp::MaxRed<SrcT>>;

    const mpp::MinRed<SrcT> op1;
    const mpp::MaxRed<SrcT> op2;

    const minMaxSrc functor(aSrc, aPitchSrc, op1, op2);

    InvokeReduction2AlongXKernelDefault<SrcT, SrcT, SrcT, TupelSize, minMaxSrc, mpp::MinRed<SrcT>, mpp::MaxRed<SrcT>,
                                        ReductionInitValue::Max, ReductionInitValue::Min>(
        aSrc, aTempBuffer1, aTempBuffer2, aSize, aStreamCtx, functor);

    const mpp::Nothing<SrcT> postOp1;
    const mpp::Nothing<SrcT> postOp2;
    const mpp::MinScalar<SrcT> postOpScalar1;
    const mpp::MaxScalar<SrcT> postOpScalar2;

    InvokeReduction2AlongYKernelDefault<SrcT, SrcT, SrcT, SrcT, mpp::MinRed<SrcT>, mpp::MaxRed<SrcT>,
                                        ReductionInitValue::Max, ReductionInitValue::Min, mpp::Nothing<SrcT>,
                                        mpp::Nothing<SrcT>, mpp::MinScalar<SrcT>, mpp::MaxScalar<SrcT>>(
        aTempBuffer1, aTempBuffer2, aDstMin, aDstMax, aDstMinScalar, aDstMaxScalar, aSize.y, postOp1, postOp2,
        postOpScalar1, postOpScalar2, aStreamCtx);
}

#pragma region Instantiate

#define Instantiate_For(type)                                                                                          \
    template void InvokeMinMaxSrc<type>(const type *aSrc, size_t aPitchSrc1, type *aTemp1, type *aTemp2,               \
                                        type *aDstMin, type *aDstMax, remove_vector_t<type> *aDstMinScalar,            \
                                        remove_vector_t<type> *aDstMaxScalar, const Size2D &aSize,                     \
                                        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);
#pragma endregion

} // namespace mpp::image::cuda
