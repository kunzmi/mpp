#if OPP_ENABLE_CUDA_BACKEND

#include "addSquareProductWeightedOutputType.h"
#include "addWeighted.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/ternary_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddWeightedSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                             size_t aPitchDst, remove_vector_t<ComputeT> aAlpha, const Size2D &aSize,
                             const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addWeightedSrcSrc =
            SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AddWeighted<ComputeT>, RoundingMode::None>;

        const opp::AddWeighted<ComputeT> op(aAlpha);

        const addWeightedSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, addWeightedSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                            functor);
    }
}

#pragma region Instantiate
// using add_spw_output_for_t for computeT and DstT
#define InstantiateInvokeAddWeightedSrcSrc_For(typeSrc)                                                                \
    template void InvokeAddWeightedSrcSrc<typeSrc, add_spw_output_for_t<typeSrc>, add_spw_output_for_t<typeSrc>>(      \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2,                              \
        add_spw_output_for_t<typeSrc> *aDst, size_t aPitchDst, remove_vector_t<add_spw_output_for_t<typeSrc>> aAlpha,  \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddWeightedSrcSrc(type)                                                             \
    InstantiateInvokeAddWeightedSrcSrc_For(Pixel##type##C1);                                                           \
    InstantiateInvokeAddWeightedSrcSrc_For(Pixel##type##C2);                                                           \
    InstantiateInvokeAddWeightedSrcSrc_For(Pixel##type##C3);                                                           \
    InstantiateInvokeAddWeightedSrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddWeightedSrcSrc(type)                                                           \
    InstantiateInvokeAddWeightedSrcSrc_For(Pixel##type##C1);                                                           \
    InstantiateInvokeAddWeightedSrcSrc_For(Pixel##type##C2);                                                           \
    InstantiateInvokeAddWeightedSrcSrc_For(Pixel##type##C3);                                                           \
    InstantiateInvokeAddWeightedSrcSrc_For(Pixel##type##C4);                                                           \
    InstantiateInvokeAddWeightedSrcSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddWeightedInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                                 remove_vector_t<ComputeT> aAlpha, const Size2D &aSize,
                                 const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addWeightedInplaceSrc =
            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AddWeighted<ComputeT>, RoundingMode::None>;

        const opp::AddWeighted<ComputeT> op(aAlpha);

        const addWeightedInplaceSrc functor(aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, addWeightedInplaceSrc>(aSrcDst, aPitchSrcDst, aSize,
                                                                                aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using add_spw_output_for_t for computeT and DstT
#define InstantiateInvokeAddWeightedInplaceSrc_For(typeSrc)                                                            \
    template void InvokeAddWeightedInplaceSrc<typeSrc, add_spw_output_for_t<typeSrc>, add_spw_output_for_t<typeSrc>>(  \
        add_spw_output_for_t<typeSrc> * aSrcDst, size_t aPitchSrcDst, const typeSrc *aSrc2, size_t aPitchSrc2,         \
        remove_vector_t<add_spw_output_for_t<typeSrc>> aAlpha, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrc(type)                                                         \
    InstantiateInvokeAddWeightedInplaceSrc_For(Pixel##type##C1);                                                       \
    InstantiateInvokeAddWeightedInplaceSrc_For(Pixel##type##C2);                                                       \
    InstantiateInvokeAddWeightedInplaceSrc_For(Pixel##type##C3);                                                       \
    InstantiateInvokeAddWeightedInplaceSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrc(type)                                                       \
    InstantiateInvokeAddWeightedInplaceSrc_For(Pixel##type##C1);                                                       \
    InstantiateInvokeAddWeightedInplaceSrc_For(Pixel##type##C2);                                                       \
    InstantiateInvokeAddWeightedInplaceSrc_For(Pixel##type##C3);                                                       \
    InstantiateInvokeAddWeightedInplaceSrc_For(Pixel##type##C4);                                                       \
    InstantiateInvokeAddWeightedInplaceSrc_For(Pixel##type##C4A);

#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
