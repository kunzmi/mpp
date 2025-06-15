#if OPP_ENABLE_CUDA_BACKEND

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
void InvokeMeanMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                         ComputeT *aTempBuffer, ulong64 *aMaskBuffer, DstT *aDst, remove_vector_t<DstT> *aDstScalar,
                         const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, opp::Sum<SrcT, ComputeT>>;

        const opp::Sum<SrcT, ComputeT> op;

        const sumSrc functor(aSrc, aPitchSrc, op);

        InvokeReductionMaskedCountingAlongXKernelDefault<SrcT, ComputeT, TupelSize, sumSrc,
                                                         opp::Sum<ComputeT, ComputeT>, ReductionInitValue::Zero>(
            aMask, aPitchMask, aSrc, aTempBuffer, aMaskBuffer, aSize, aStreamCtx, functor);

        InvokeReductionMaskedCountingAlongYKernelDefault<ComputeT, DstT, opp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                                         opp::DivPostOp<DstT>, opp::DivScalar<DstT>>(
            aMaskBuffer, aTempBuffer, aDst, aDstScalar, aSize.y, aStreamCtx);
    }
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
