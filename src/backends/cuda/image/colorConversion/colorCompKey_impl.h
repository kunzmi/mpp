#include "colorCompKey.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/colorConversion/color_operators.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/functors/inplaceConstantFunctor.h>
#include <common/image/functors/inplaceDevConstantFunctor.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
#include <common/image/functors/srcFunctor.h>
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

template <typename SrcDstT>
void InvokeColorCompKeySrcSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, const SrcDstT *aSrc2, size_t aPitchSrc2,
                              const SrcDstT &aValue, SrcDstT *aDst, size_t aPitchDst, const Size2D &aSize,
                              const StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = SrcDstT;

    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, CompColorKey<SrcT>, RoundingMode::None,
                                        voidType, voidType, true>;
    const CompColorKey<SrcT> op(aValue);

    const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate

#define InstantiateInvokeColorCompKeySrcSrc_For(typeSrcDst)                                                            \
    template void InvokeColorCompKeySrcSrc<typeSrcDst>(const typeSrcDst *aSrc1, size_t aPitchSrc1,                     \
                                                       const typeSrcDst *aSrc2, size_t aPitchSrc2,                     \
                                                       const typeSrcDst &aValue, typeSrcDst *aDst, size_t aPitchDst,   \
                                                       const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeColorCompKeySrcSrc(type)                                                            \
    InstantiateInvokeColorCompKeySrcSrc_For(Pixel##type##C1);                                                          \
    InstantiateInvokeColorCompKeySrcSrc_For(Pixel##type##C2);                                                          \
    InstantiateInvokeColorCompKeySrcSrc_For(Pixel##type##C3);                                                          \
    InstantiateInvokeColorCompKeySrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeColorCompKeySrcSrc(type)                                                          \
    InstantiateInvokeColorCompKeySrcSrc_For(Pixel##type##C1);                                                          \
    InstantiateInvokeColorCompKeySrcSrc_For(Pixel##type##C2);                                                          \
    InstantiateInvokeColorCompKeySrcSrc_For(Pixel##type##C3);                                                          \
    InstantiateInvokeColorCompKeySrcSrc_For(Pixel##type##C4);                                                          \
    InstantiateInvokeColorCompKeySrcSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorCompKeyInplaceSrcSrc(SrcDstT *aSrcDst, size_t aPitchSrcDst, const SrcDstT *aSrc2, size_t aPitchSrc2,
                                     const SrcDstT &aValue, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = SrcDstT;

    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    using compareInplaceSrc =
        InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, CompColorKey<SrcT>, RoundingMode::None, voidType, voidType>;
    const CompColorKey<SrcT> op(aValue);

    const compareInplaceSrc functor(aSrc2, aPitchSrc2, op);
    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                        functor);
}

#pragma region Instantiate

#define InstantiateInvokeColorCompKeyInplaceSrcSrc_For(typeSrcDst)                                                     \
    template void InvokeColorCompKeyInplaceSrcSrc<typeSrcDst>(                                                         \
        typeSrcDst * aSrcDst, size_t aPitchSrcDst, const typeSrcDst *aSrc2, size_t aPitchSrc2,                         \
        const typeSrcDst &aValue, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeColorCompKeyInplaceSrcSrc(type)                                                     \
    InstantiateInvokeColorCompKeyInplaceSrcSrc_For(Pixel##type##C1);                                                   \
    InstantiateInvokeColorCompKeyInplaceSrcSrc_For(Pixel##type##C2);                                                   \
    InstantiateInvokeColorCompKeyInplaceSrcSrc_For(Pixel##type##C3);                                                   \
    InstantiateInvokeColorCompKeyInplaceSrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeColorCompKeyInplaceSrcSrc(type)                                                   \
    InstantiateInvokeColorCompKeyInplaceSrcSrc_For(Pixel##type##C1);                                                   \
    InstantiateInvokeColorCompKeyInplaceSrcSrc_For(Pixel##type##C2);                                                   \
    InstantiateInvokeColorCompKeyInplaceSrcSrc_For(Pixel##type##C3);                                                   \
    InstantiateInvokeColorCompKeyInplaceSrcSrc_For(Pixel##type##C4);                                                   \
    InstantiateInvokeColorCompKeyInplaceSrcSrc_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
