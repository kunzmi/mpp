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

        RShift<SrcDstT> op(aConst);

        rshiftSrcC functor(aSrc, aPitchSrc, op);

        InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, rshiftSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeRShiftSrcC<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, uint aConst,      \
                                                     typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,    \
                                                     const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
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

        RShift<SrcDstT> op(aConst);

        rshiftInplaceC functor(op);

        InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, rshiftInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
    }
}

#pragma region Instantiate
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeRShiftInplaceC<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, uint aConst, \
                                                         const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
