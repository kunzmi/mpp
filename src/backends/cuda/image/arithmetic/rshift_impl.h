#if OPP_ENABLE_CUDA_BACKEND

#include "rshift.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/binary_operators.h>
#include <backends/cuda/simd_operators/simd_types.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/srcFunctor.h>
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
template <typename SrcDstT>
void InvokeRShiftSrcC(const SrcDstT *aSrc, size_t aPitchSrc, uint aConst, SrcDstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcDstT> && oppEnableCudaBackend<SrcDstT>)
    {
        using DstT = SrcDstT;
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcDstT)>::value;

        using rshiftSrcC = SrcFunctor<TupelSize, SrcDstT, SrcDstT, SrcDstT, opp::RShift<SrcDstT>, RoundingMode::None>;

        const opp::RShift<SrcDstT> op(aConst);

        const rshiftSrcC functor(aSrc, aPitchSrc, op);

        InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, rshiftSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeRShiftSrcC_For(typeSrcIsTypeDst)                                                              \
    template void InvokeRShiftSrcC<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, uint aConst,      \
                                                     typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,    \
                                                     const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeRShiftSrcC(type)                                                                    \
    InstantiateInvokeRShiftSrcC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeRShiftSrcC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeRShiftSrcC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeRShiftSrcC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeRShiftSrcC(type)                                                                  \
    InstantiateInvokeRShiftSrcC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeRShiftSrcC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeRShiftSrcC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeRShiftSrcC_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeRShiftSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRShiftInplaceC(SrcDstT *aSrcDst, size_t aPitchSrcDst, uint aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcDstT> && oppEnableCudaBackend<SrcDstT>)
    {
        using DstT = SrcDstT;
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcDstT)>::value;

        using rshiftInplaceC = InplaceFunctor<TupelSize, SrcDstT, SrcDstT, opp::RShift<SrcDstT>, RoundingMode::None>;

        const opp::RShift<SrcDstT> op(aConst);

        const rshiftInplaceC functor(op);

        InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, rshiftInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeRShiftInplaceC_For(typeSrcIsTypeDst)                                                          \
    template void InvokeRShiftInplaceC<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, uint aConst, \
                                                         const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeRShiftInplaceC(type)                                                                \
    InstantiateInvokeRShiftInplaceC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeRShiftInplaceC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeRShiftInplaceC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeRShiftInplaceC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeRShiftInplaceC(type)                                                              \
    InstantiateInvokeRShiftInplaceC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeRShiftInplaceC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeRShiftInplaceC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeRShiftInplaceC_For(Pixel##type##C4);                                                              \
    InstantiateInvokeRShiftInplaceC_For(Pixel##type##C4A);

#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
