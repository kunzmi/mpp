#if OPP_ENABLE_CUDA_BACKEND

#include "lshift.h"
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
void InvokeLShiftSrcC(const SrcDstT *aSrc, size_t aPitchSrc, uint aConst, SrcDstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcDstT> && oppEnableCudaBackend<SrcDstT>)
    {
        using DstT = SrcDstT;
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcDstT)>::value;

        using lshiftSrcC = SrcFunctor<TupelSize, SrcDstT, SrcDstT, SrcDstT, opp::LShift<SrcDstT>, RoundingMode::None>;

        const opp::LShift<SrcDstT> op(aConst);

        const lshiftSrcC functor(aSrc, aPitchSrc, op);

        InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, lshiftSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeLShiftSrcC<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, uint aConst,      \
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
void InvokeLShiftInplaceC(SrcDstT *aSrcDst, size_t aPitchSrcDst, uint aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcDstT> && oppEnableCudaBackend<SrcDstT>)
    {
        using DstT = SrcDstT;
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcDstT)>::value;

        using lshiftInplaceC = InplaceFunctor<TupelSize, SrcDstT, SrcDstT, opp::LShift<SrcDstT>, RoundingMode::None>;

        const opp::LShift<SrcDstT> op(aConst);

        const lshiftInplaceC functor(op);

        InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, lshiftInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
    }
}

#pragma region Instantiate
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeLShiftInplaceC<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, uint aConst, \
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
