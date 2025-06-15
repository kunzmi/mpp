#if OPP_ENABLE_CUDA_BACKEND

#include "minMasked.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
#include <backends/cuda/image/reductionMaskedAlongXKernel.h>
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
template <typename SrcT>
void InvokeMinMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                        SrcT *aTempBuffer, SrcT *aDst, remove_vector_t<SrcT> *aDstScalar, const Size2D &aSize,
                        const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using minSrc = SrcReductionFunctor<TupelSize, SrcT, SrcT, opp::MinRed<SrcT>>;

        const opp::MinRed<SrcT> op;

        const minSrc functor(aSrc, aPitchSrc, op);

        InvokeReductionMaskedAlongXKernelDefault<SrcT, SrcT, TupelSize, minSrc, opp::MinRed<SrcT>,
                                                 ReductionInitValue::Max>(aMask, aPitchMask, aSrc, aTempBuffer, aSize,
                                                                          aStreamCtx, functor);

        const opp::Nothing<SrcT> postOp;
        const opp::MinScalar<SrcT> postOpScalar;

        InvokeReductionAlongYKernelDefault<SrcT, SrcT, opp::MinRed<SrcT>, ReductionInitValue::Max, opp::Nothing<SrcT>,
                                           opp::MinScalar<SrcT>>(aTempBuffer, aDst, aDstScalar, aSize.y, postOp,
                                                                 postOpScalar, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(type)                                                                                          \
    template void InvokeMinMaskedSrc<type>(                                                                            \
        const Pixel8uC1 *aMask, size_t aPitchMask, const type *aSrc, size_t aPitchSrc1, type *aTemp, type *aDst,       \
        remove_vector_t<type> *aDstScalar, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
