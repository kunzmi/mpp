#if MPP_ENABLE_CUDA_BACKEND

#include "normDiffL1.h"
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
void InvokeNormDiffL1SrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2,
                            ComputeT *aTempBuffer, DstT *aDst, remove_vector_t<DstT> *aDstScalar, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcT> && mppEnableCudaBackend<SrcT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        using normL1SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL1<SrcT, ComputeT>>;

        const mpp::NormDiffL1<SrcT, ComputeT> op;

        const normL1SrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeReductionAlongXKernelDefault<SrcT, ComputeT, TupelSize, normL1SrcSrc, mpp::Sum<ComputeT, ComputeT>,
                                           ReductionInitValue::Zero>(aSrc1, aTempBuffer, aSize, aStreamCtx, functor);

        const mpp::Nothing<DstT> postOp;

        const mpp::SumScalar<DstT> postOpScalar;

        InvokeReductionAlongYKernelDefault<ComputeT, DstT, mpp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                           mpp::Nothing<DstT>, mpp::SumScalar<DstT>>(
            aTempBuffer, aDst, aDstScalar, aSize.y, postOp, postOpScalar, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeNormDiffL1SrcSrc<typeSrc, normDiffL1_types_for_ct<typeSrc>, normDiffL1_types_for_rt<typeSrc>>( \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2,                              \
        normDiffL1_types_for_ct<typeSrc> *aTemp, normDiffL1_types_for_rt<typeSrc> *aDst,                               \
        remove_vector_t<normDiffL1_types_for_rt<typeSrc>> *aDstScalar, const Size2D &aSize,                            \
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
