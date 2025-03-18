#if OPP_ENABLE_CUDA_BACKEND

#include "sum.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongXKernel.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/srcReductionFunctor.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/statistics/operators.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSumSrc(const SrcT *aSrc, size_t aPitchSrc, ComputeT *aTempBuffer, DstT *aDst, const Size2D &aSize,
                  const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        // OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
        using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, opp::Sum<SrcT, ComputeT>>;

        const opp::Sum<SrcT, ComputeT> op;

        const sumSrc functor(aSrc, aPitchSrc, op);

        InvokeReductionAlongXKernelDefault<SrcT, ComputeT, TupelSize, sumSrc>(aSrc, aTempBuffer, aSize, aStreamCtx,
                                                                              functor);

        InvokeReductionAlongYKernelDefault<ComputeT, DstT>(aTempBuffer, aDst, aSize.y, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeTemp, typeDst)                                                                    \
    template void InvokeSumSrc<typeSrc, typeTemp, typeDst>(const typeSrc *aSrc, size_t aPitchSrc1, typeTemp *aTemp,    \
                                                           typeDst *aDst, const Size2D &aSize,                         \
                                                           const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1, Pixel32fC1, Pixel64fC1);                                                          \
    Instantiate_For(Pixel##type##C2, Pixel32fC2, Pixel64fC2);                                                          \
    Instantiate_For(Pixel##type##C3, Pixel32fC3, Pixel64fC3);                                                          \
    Instantiate_For(Pixel##type##C4, Pixel32fC4, Pixel64fC4);                                                          \
    Instantiate_For(Pixel##type##C4A, Pixel32fC4A, Pixel64fC4A);

ForAllChannelsWithAlpha(8u);
// ForAllChannelsWithAlpha(8s);
//
// ForAllChannelsWithAlpha(16u);
// ForAllChannelsWithAlpha(16s);
//
// ForAllChannelsWithAlpha(32u);
// ForAllChannelsWithAlpha(32s);
//
// ForAllChannelsWithAlpha(16f);
// ForAllChannelsWithAlpha(16bf);
// ForAllChannelsWithAlpha(32f);
// ForAllChannelsWithAlpha(64f);
//
// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
// ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
