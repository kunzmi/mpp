#if MPP_ENABLE_CUDA_BACKEND

#include "mean.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongXKernel.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
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
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMeanSrc(const SrcT *aSrc, size_t aPitchSrc, ComputeT *aTempBuffer, DstT *aDst,
                   remove_vector_t<DstT> *aDstScalar, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

        const mpp::Sum<SrcT, ComputeT> op;

        const sumSrc functor(aSrc, aPitchSrc, op);

        InvokeReductionAlongXKernelDefault<SrcT, ComputeT, TupelSize, sumSrc, mpp::Sum<ComputeT, ComputeT>,
                                           ReductionInitValue::Zero>(aSrc, aTempBuffer, aSize, aStreamCtx, functor);

        const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(aSize.TotalSize()));
        const mpp::DivScalar<DstT> postOpScalar(
            static_cast<complex_basetype_t<remove_vector_t<DstT>>>(aSize.TotalSize()));

        InvokeReductionAlongYKernelDefault<ComputeT, DstT, mpp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                           mpp::DivPostOp<DstT>, mpp::DivScalar<DstT>>(
            aTempBuffer, aDst, aDstScalar, aSize.y, postOp, postOpScalar, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeMeanSrc<typeSrc, mean_types_for_ct<typeSrc>, mean_types_for_rt<typeSrc>>(                      \
        const typeSrc *aSrc, size_t aPitchSrc1, mean_types_for_ct<typeSrc> *aTemp, mean_types_for_rt<typeSrc> *aDst,   \
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
#endif // MPP_ENABLE_CUDA_BACKEND
