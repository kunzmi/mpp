#include "normInf.h"
#include "normInfMasked.h"
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
void InvokeNormInfMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                            ComputeT *aTempBuffer, DstT *aDst, remove_vector_t<DstT> *aDstScalar, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    using normInfSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormInf<SrcT, ComputeT>>;

    const mpp::NormInf<SrcT, ComputeT> op;

    const normInfSrc functor(aSrc, aPitchSrc, op);

    InvokeReductionMaskedAlongXKernelDefault<SrcT, ComputeT, TupelSize, normInfSrc, mpp::MaxRed<ComputeT>,
                                             ReductionInitValue::Zero>(aMask, aPitchMask, aSrc, aTempBuffer, aSize,
                                                                       aStreamCtx, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::MaxScalar<DstT> postOpScalar;

    InvokeReductionAlongYKernelDefault<ComputeT, DstT, mpp::MaxRed<DstT>, ReductionInitValue::Zero, mpp::Nothing<DstT>,
                                       mpp::MaxScalar<DstT>>(aTempBuffer, aDst, aDstScalar, aSize.y, postOp,
                                                             postOpScalar, aStreamCtx);
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeNormInfMaskedSrc<typeSrc, normInf_types_for_ct<typeSrc>, normInf_types_for_rt<typeSrc>>(       \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrc *aSrc, size_t aPitchSrc1,                             \
        normInf_types_for_ct<typeSrc> *aTemp, normInf_types_for_rt<typeSrc> *aDst,                                     \
        remove_vector_t<normInf_types_for_rt<typeSrc>> *aDstScalar, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(typeIn)                                                                                \
    Instantiate_For(Pixel##typeIn##C1);                                                                                \
    Instantiate_For(Pixel##typeIn##C2);                                                                                \
    Instantiate_For(Pixel##typeIn##C3);                                                                                \
    Instantiate_For(Pixel##typeIn##C4);                                                                                \
    Instantiate_For(Pixel##typeIn##C4A);
#pragma endregion

} // namespace mpp::image::cuda
