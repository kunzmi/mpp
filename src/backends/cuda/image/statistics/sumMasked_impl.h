#include "sum.h"
#include "sumMasked.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
#include <backends/cuda/image/reductionMaskedAlongXKernel.h>
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
void InvokeSumMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                        ComputeT *aTempBuffer, DstT *aDst, remove_vector_t<DstT> *aDstScalar, const Size2D &aSize,
                        const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(aSrc, aPitchSrc, op);

    InvokeReductionMaskedAlongXKernelDefault<SrcT, ComputeT, TupelSize, sumSrc, mpp::Sum<ComputeT, ComputeT>,
                                             ReductionInitValue::Zero>(aMask, aPitchMask, aSrc, aTempBuffer, aSize,
                                                                       aStreamCtx, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    InvokeReductionAlongYKernelDefault<ComputeT, DstT, mpp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                       mpp::Nothing<DstT>, mpp::SumScalar<DstT>>(aTempBuffer, aDst, aDstScalar, aSize.y,
                                                                                 postOp, postOpScalar, aStreamCtx);
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, variant)                                                                              \
    template void InvokeSumMaskedSrc<typeSrc, sum_types_for_ct<typeSrc, variant>, sum_types_for_rt<typeSrc, variant>>( \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrc *aSrc, size_t aPitchSrc1,                             \
        sum_types_for_ct<typeSrc, variant> *aTemp, sum_types_for_rt<typeSrc, variant> *aDst,                           \
        remove_vector_t<sum_types_for_rt<typeSrc, variant>> *aDstScalar, const Size2D &aSize,                          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(typeIn, variant)                                                                         \
    Instantiate_For(Pixel##typeIn##C1, variant);                                                                       \
    Instantiate_For(Pixel##typeIn##C2, variant);                                                                       \
    Instantiate_For(Pixel##typeIn##C3, variant);                                                                       \
    Instantiate_For(Pixel##typeIn##C4, variant);

#define ForAllChannelsWithAlpha(typeIn, variant)                                                                       \
    Instantiate_For(Pixel##typeIn##C1, variant);                                                                       \
    Instantiate_For(Pixel##typeIn##C2, variant);                                                                       \
    Instantiate_For(Pixel##typeIn##C3, variant);                                                                       \
    Instantiate_For(Pixel##typeIn##C4, variant);                                                                       \
    Instantiate_For(Pixel##typeIn##C4A, variant);

#pragma endregion

} // namespace mpp::image::cuda
