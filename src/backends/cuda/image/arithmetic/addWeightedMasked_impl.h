#include "addSquareProductWeightedOutputType.h"
#include "addWeightedMasked.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelMaskedKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/ternary_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddWeightedSrcSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                                 const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst,
                                 const remove_vector_t<ComputeT> &aAlpha, const Size2D &aSize,
                                 const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using addWeightedSrcSrc =
        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::AddWeighted<ComputeT>, RoundingMode::None>;

    const mpp::AddWeighted<ComputeT> op(aAlpha);

    const addWeightedSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

    InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addWeightedSrcSrc>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                              aStreamCtx, functor);
}

#pragma region Instantiate
// using add_spw_output_for_t for computeT and DstT
#define InstantiateInvokeAddWeightedSrcSrcMask_For(typeSrc)                                                            \
    template void InvokeAddWeightedSrcSrcMask<typeSrc, add_spw_output_for_t<typeSrc>, add_spw_output_for_t<typeSrc>>(  \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2,      \
        size_t aPitchSrc2, add_spw_output_for_t<typeSrc> *aDst, size_t aPitchDst,                                      \
        const remove_vector_t<add_spw_output_for_t<typeSrc>> &aAlpha, const Size2D &aSize,                             \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddWeightedSrcSrcMask(type)                                                         \
    InstantiateInvokeAddWeightedSrcSrcMask_For(Pixel##type##C1);                                                       \
    InstantiateInvokeAddWeightedSrcSrcMask_For(Pixel##type##C2);                                                       \
    InstantiateInvokeAddWeightedSrcSrcMask_For(Pixel##type##C3);                                                       \
    InstantiateInvokeAddWeightedSrcSrcMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(type)                                                       \
    InstantiateInvokeAddWeightedSrcSrcMask_For(Pixel##type##C1);                                                       \
    InstantiateInvokeAddWeightedSrcSrcMask_For(Pixel##type##C2);                                                       \
    InstantiateInvokeAddWeightedSrcSrcMask_For(Pixel##type##C3);                                                       \
    InstantiateInvokeAddWeightedSrcSrcMask_For(Pixel##type##C4);                                                       \
    InstantiateInvokeAddWeightedSrcSrcMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddWeightedInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                     const SrcT *aSrc2, size_t aPitchSrc2, const remove_vector_t<ComputeT> &aAlpha,
                                     const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using addWeightedInplaceSrc =
        InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::AddWeighted<ComputeT>, RoundingMode::None>;

    const mpp::AddWeighted<ComputeT> op(aAlpha);

    const addWeightedInplaceSrc functor(aSrc2, aPitchSrc2, op);

    InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addWeightedInplaceSrc>(
        aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
// using add_spw_output_for_t for computeT and DstT
#define InstantiateInvokeAddWeightedInplaceSrcMask_For(typeSrc)                                                        \
    template void                                                                                                      \
    InvokeAddWeightedInplaceSrcMask<typeSrc, add_spw_output_for_t<typeSrc>, add_spw_output_for_t<typeSrc>>(            \
        const Pixel8uC1 *aMask, size_t aPitchMask, add_spw_output_for_t<typeSrc> *aSrcDst, size_t aPitchSrcDst,        \
        const typeSrc *aSrc2, size_t aPitchSrc2, const remove_vector_t<add_spw_output_for_t<typeSrc>> &aAlpha,         \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrcMask(type)                                                     \
    InstantiateInvokeAddWeightedInplaceSrcMask_For(Pixel##type##C1);                                                   \
    InstantiateInvokeAddWeightedInplaceSrcMask_For(Pixel##type##C2);                                                   \
    InstantiateInvokeAddWeightedInplaceSrcMask_For(Pixel##type##C3);                                                   \
    InstantiateInvokeAddWeightedInplaceSrcMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(type)                                                   \
    InstantiateInvokeAddWeightedInplaceSrcMask_For(Pixel##type##C1);                                                   \
    InstantiateInvokeAddWeightedInplaceSrcMask_For(Pixel##type##C2);                                                   \
    InstantiateInvokeAddWeightedInplaceSrcMask_For(Pixel##type##C3);                                                   \
    InstantiateInvokeAddWeightedInplaceSrcMask_For(Pixel##type##C4);                                                   \
    InstantiateInvokeAddWeightedInplaceSrcMask_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
