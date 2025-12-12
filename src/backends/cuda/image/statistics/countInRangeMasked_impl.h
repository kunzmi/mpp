#include "countInRangeMasked.h"
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
void InvokeCountInRangeMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                                 ComputeT *aTempBuffer, DstT *aDst, remove_vector_t<DstT> *aDstScalar,
                                 const SrcT &aLowerLimit, const SrcT &aUpperLimit, const Size2D &aSize,
                                 const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    using countInRangeSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::CountInRange<SrcT>>;

    const mpp::CountInRange<SrcT> op(aLowerLimit, aUpperLimit);

    const countInRangeSrc functor(aSrc, aPitchSrc, op);

    InvokeReductionMaskedAlongXKernelDefault<SrcT, ComputeT, TupelSize, countInRangeSrc, mpp::Sum<ComputeT, ComputeT>,
                                             ReductionInitValue::Zero>(aMask, aPitchMask, aSrc, aTempBuffer, aSize,
                                                                       aStreamCtx, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    InvokeReductionAlongYKernelDefault<ComputeT, DstT, mpp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                       mpp::Nothing<DstT>, mpp::SumScalar<DstT>>(aTempBuffer, aDst, aDstScalar, aSize.y,
                                                                                 postOp, postOpScalar, aStreamCtx);
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeTemp, typeDst)                                                                    \
    template void InvokeCountInRangeMaskedSrc<typeSrc, typeTemp, typeDst>(                                             \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrc *aSrc, size_t aPitchSrc1, typeTemp *aTemp,            \
        typeDst *aDst, remove_vector_t<typeDst> *aDstScalar, const typeSrc &aLowerLimit, const typeSrc &aUpperLimit,   \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(typeIn, typeCompute, typeDst)                                                          \
    Instantiate_For(Pixel##typeIn##C1, Pixel##typeCompute##C1, Pixel##typeDst##C1);                                    \
    Instantiate_For(Pixel##typeIn##C2, Pixel##typeCompute##C2, Pixel##typeDst##C2);                                    \
    Instantiate_For(Pixel##typeIn##C3, Pixel##typeCompute##C3, Pixel##typeDst##C3);                                    \
    Instantiate_For(Pixel##typeIn##C4, Pixel##typeCompute##C4, Pixel##typeDst##C4);                                    \
    Instantiate_For(Pixel##typeIn##C4A, Pixel##typeCompute##C4A, Pixel##typeDst##C4A);
#pragma endregion

} // namespace mpp::image::cuda
