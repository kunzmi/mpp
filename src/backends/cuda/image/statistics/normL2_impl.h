#include "normL2.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongXKernel.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcReductionFunctor.h>
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
void InvokeNormL2Src(const SrcT *aSrc, size_t aPitchSrc, ComputeT *aTempBuffer, DstT *aDst,
                     remove_vector_t<DstT> *aDstScalar, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    using normL2Src = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormL2<SrcT, ComputeT>>;

    const mpp::NormL2<SrcT, ComputeT> op;

    const normL2Src functor(aSrc, aPitchSrc, op);

    InvokeReductionAlongXKernelDefault<SrcT, ComputeT, TupelSize, normL2Src, mpp::Sum<ComputeT, ComputeT>,
                                       ReductionInitValue::Zero>(aSrc, aTempBuffer, aSize, aStreamCtx, functor);

    const mpp::SqrtPostOp<DstT> postOp;
    const mpp::SumThenSqrtScalar<DstT> postOpScalar;

    InvokeReductionAlongYKernelDefault<ComputeT, DstT, mpp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                       mpp::SqrtPostOp<DstT>, mpp::SumThenSqrtScalar<DstT>>(
        aTempBuffer, aDst, aDstScalar, aSize.y, postOp, postOpScalar, aStreamCtx);
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeNormL2Src<typeSrc, normL2_types_for_ct<typeSrc>, normL2_types_for_rt<typeSrc>>(                \
        const typeSrc *aSrc, size_t aPitchSrc1, normL2_types_for_ct<typeSrc> *aTemp,                                   \
        normL2_types_for_rt<typeSrc> *aDst, remove_vector_t<normL2_types_for_rt<typeSrc>> *aDstScalar,                 \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(typeIn)                                                                                \
    Instantiate_For(Pixel##typeIn##C1);                                                                                \
    Instantiate_For(Pixel##typeIn##C2);                                                                                \
    Instantiate_For(Pixel##typeIn##C3);                                                                                \
    Instantiate_For(Pixel##typeIn##C4);                                                                                \
    Instantiate_For(Pixel##typeIn##C4A);
#pragma endregion

} // namespace mpp::image::cuda
