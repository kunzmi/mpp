#if MPP_ENABLE_CUDA_BACKEND

#include "mse.h"
#include "psnr.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongXKernel.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcSrcReductionFunctor.h>
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
void InvokePSNRSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2,
                      remove_vector_t<DstT> aValueRange, ComputeT *aTempBuffer, DstT *aDst,
                      remove_vector_t<DstT> *aDstScalar, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

        const mpp::NormDiffL2<SrcT, ComputeT> op;

        const normL2SrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeReductionAlongXKernelDefault<SrcT, ComputeT, TupelSize, normL2SrcSrc, mpp::Sum<ComputeT, ComputeT>,
                                           ReductionInitValue::Zero>(aSrc1, aTempBuffer, aSize, aStreamCtx, functor);

        const mpp::PSNR<DstT> postOp(static_cast<remove_vector_t<DstT>>(aSize.TotalSize()), aValueRange);

        const mpp::PSNRScalar<DstT> postOpScalar(static_cast<remove_vector_t<DstT>>(aSize.TotalSize()), aValueRange);

        InvokeReductionAlongYKernelDefault<ComputeT, DstT, mpp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                           mpp::PSNR<DstT>, mpp::PSNRScalar<DstT>>(
            aTempBuffer, aDst, aDstScalar, aSize.y, postOp, postOpScalar, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokePSNRSrcSrc<typeSrc, mse_types_for_ct<typeSrc>, mse_types_for_rt<typeSrc>>(                     \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2,                              \
        remove_vector_t<mse_types_for_rt<typeSrc>> aValueRange, mse_types_for_ct<typeSrc> *aTemp,                      \
        mse_types_for_rt<typeSrc> *aDst, remove_vector_t<mse_types_for_rt<typeSrc>> *aDstScalar, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

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
