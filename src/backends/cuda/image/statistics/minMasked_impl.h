#if MPP_ENABLE_CUDA_BACKEND

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
void InvokeMinMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                        SrcT *aTempBuffer, SrcT *aDst, remove_vector_t<SrcT> *aDstScalar, const Size2D &aSize,
                        const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using minSrc = SrcReductionFunctor<TupelSize, SrcT, SrcT, mpp::MinRed<SrcT>>;

        const mpp::MinRed<SrcT> op;

        const minSrc functor(aSrc, aPitchSrc, op);

        InvokeReductionMaskedAlongXKernelDefault<SrcT, SrcT, TupelSize, minSrc, mpp::MinRed<SrcT>,
                                                 ReductionInitValue::Max>(aMask, aPitchMask, aSrc, aTempBuffer, aSize,
                                                                          aStreamCtx, functor);

        const mpp::Nothing<SrcT> postOp;
        const mpp::MinScalar<SrcT> postOpScalar;

        InvokeReductionAlongYKernelDefault<SrcT, SrcT, mpp::MinRed<SrcT>, ReductionInitValue::Max, mpp::Nothing<SrcT>,
                                           mpp::MinScalar<SrcT>>(aTempBuffer, aDst, aDstScalar, aSize.y, postOp,
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

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
