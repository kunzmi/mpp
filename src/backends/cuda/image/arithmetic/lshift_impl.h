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
template <typename SrcDstT>
void InvokeLShiftSrcC(const SrcDstT *aSrc, size_t aPitchSrc, uint aConst, SrcDstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    using DstT = SrcDstT;
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcDstT)>::value;

    using lshiftSrcC = SrcFunctor<TupelSize, SrcDstT, SrcDstT, SrcDstT, mpp::LShift<SrcDstT>, RoundingMode::None>;

    const mpp::LShift<SrcDstT> op(aConst);

    const lshiftSrcC functor(aSrc, aPitchSrc, op);

    InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, lshiftSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLShiftSrcC_For(typeSrcIsTypeDst)                                                              \
    template void InvokeLShiftSrcC<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, uint aConst,      \
                                                     typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,    \
                                                     const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeLShiftSrcC(type)                                                                    \
    InstantiateInvokeLShiftSrcC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeLShiftSrcC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeLShiftSrcC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeLShiftSrcC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeLShiftSrcC(type)                                                                  \
    InstantiateInvokeLShiftSrcC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeLShiftSrcC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeLShiftSrcC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeLShiftSrcC_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeLShiftSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLShiftInplaceC(SrcDstT *aSrcDst, size_t aPitchSrcDst, uint aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    using DstT = SrcDstT;
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcDstT)>::value;

    using lshiftInplaceC = InplaceFunctor<TupelSize, SrcDstT, SrcDstT, mpp::LShift<SrcDstT>, RoundingMode::None>;

    const mpp::LShift<SrcDstT> op(aConst);

    const lshiftInplaceC functor(op);

    InvokeForEachPixelKernelDefault<SrcDstT, TupelSize, lshiftInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                        functor);
}

#pragma region Instantiate
#define InstantiateInvokeLShiftInplaceC_For(typeSrcIsTypeDst)                                                          \
    template void InvokeLShiftInplaceC<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, uint aConst, \
                                                         const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeLShiftInplaceC(type)                                                                \
    InstantiateInvokeLShiftInplaceC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeLShiftInplaceC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeLShiftInplaceC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeLShiftInplaceC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeLShiftInplaceC(type)                                                              \
    InstantiateInvokeLShiftInplaceC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeLShiftInplaceC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeLShiftInplaceC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeLShiftInplaceC_For(Pixel##type##C4);                                                              \
    InstantiateInvokeLShiftInplaceC_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
