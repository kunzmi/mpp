#include "mean.h"
#include "meanMasked.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionMaskedCountingAlongXKernel.h>
#include <backends/cuda/image/reductionMaskedCountingAlongYKernel.h>
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
void InvokeMeanMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                         ComputeT *aTempBuffer, ulong64 *aMaskBuffer, DstT *aDst, remove_vector_t<DstT> *aDstScalar,
                         const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(aSrc, aPitchSrc, op);

    InvokeReductionMaskedCountingAlongXKernelDefault<SrcT, ComputeT, TupelSize, sumSrc, mpp::Sum<ComputeT, ComputeT>,
                                                     ReductionInitValue::Zero>(aMask, aPitchMask, aSrc, aTempBuffer,
                                                                               aMaskBuffer, aSize, aStreamCtx, functor);

    InvokeReductionMaskedCountingAlongYKernelDefault<ComputeT, DstT, mpp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                                     mpp::DivPostOp<DstT>, mpp::DivScalar<DstT>>(
        aMaskBuffer, aTempBuffer, aDst, aDstScalar, aSize.y, aStreamCtx);
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeMeanMaskedSrc<typeSrc, mean_types_for_ct<typeSrc>, mean_types_for_rt<typeSrc>>(                \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrc *aSrc, size_t aPitchSrc1,                             \
        mean_types_for_ct<typeSrc> *aTemp, ulong64 *aMaskBuffer, mean_types_for_rt<typeSrc> *aDst,                     \
        remove_vector_t<mean_types_for_rt<typeSrc>> *aDstScalar, const Size2D &aSize, const StreamCtx &aStreamCtx);

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
