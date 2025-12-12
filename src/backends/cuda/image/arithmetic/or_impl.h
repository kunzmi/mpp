#include "or.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/binary_operators.h>
#include <backends/cuda/simd_operators/simd_types.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceConstantFunctor.h>
#include <common/image/functors/inplaceDevConstantFunctor.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
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
void InvokeOrSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                    size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using orSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Or<ComputeT>, RoundingMode::None>;

    const mpp::Or<ComputeT> op;

    const orSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, orSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate

#define InstantiateInvokeOrSrcSrc_For(typeSrcIsTypeDst)                                                                \
    template void InvokeOrSrcSrc<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                                \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeOrSrcSrc(type)                                                                      \
    InstantiateInvokeOrSrcSrc_For(Pixel##type##C1);                                                                    \
    InstantiateInvokeOrSrcSrc_For(Pixel##type##C2);                                                                    \
    InstantiateInvokeOrSrcSrc_For(Pixel##type##C3);                                                                    \
    InstantiateInvokeOrSrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeOrSrcSrc(type)                                                                    \
    InstantiateInvokeOrSrcSrc_For(Pixel##type##C1);                                                                    \
    InstantiateInvokeOrSrcSrc_For(Pixel##type##C2);                                                                    \
    InstantiateInvokeOrSrcSrc_For(Pixel##type##C3);                                                                    \
    InstantiateInvokeOrSrcSrc_For(Pixel##type##C4);                                                                    \
    InstantiateInvokeOrSrcSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeOrSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                  const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using orSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Or<ComputeT>, RoundingMode::None>;

    const mpp::Or<ComputeT> op;

    const orSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, orSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeOrSrcC_For(typeSrcIsTypeDst)                                                                  \
    template void InvokeOrSrcC<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                                  \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeOrSrcC(type)                                                                        \
    InstantiateInvokeOrSrcC_For(Pixel##type##C1);                                                                      \
    InstantiateInvokeOrSrcC_For(Pixel##type##C2);                                                                      \
    InstantiateInvokeOrSrcC_For(Pixel##type##C3);                                                                      \
    InstantiateInvokeOrSrcC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeOrSrcC(type)                                                                      \
    InstantiateInvokeOrSrcC_For(Pixel##type##C1);                                                                      \
    InstantiateInvokeOrSrcC_For(Pixel##type##C2);                                                                      \
    InstantiateInvokeOrSrcC_For(Pixel##type##C3);                                                                      \
    InstantiateInvokeOrSrcC_For(Pixel##type##C4);                                                                      \
    InstantiateInvokeOrSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeOrSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                     const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using orSrcDevC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Or<ComputeT>, RoundingMode::None>;

    const mpp::Or<ComputeT> op;

    const orSrcDevC functor(aSrc, aPitchSrc, aConst, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, orSrcDevC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate

#define InstantiateInvokeOrSrcDevC_For(typeSrcIsTypeDst)                                                               \
    template void InvokeOrSrcDevC<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                               \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeOrSrcDevC(type)                                                                     \
    InstantiateInvokeOrSrcDevC_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeOrSrcDevC_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeOrSrcDevC_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeOrSrcDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeOrSrcDevC(type)                                                                   \
    InstantiateInvokeOrSrcDevC_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeOrSrcDevC_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeOrSrcDevC_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeOrSrcDevC_For(Pixel##type##C4);                                                                   \
    InstantiateInvokeOrSrcDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeOrInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                        const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using orInplaceSrc = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Or<ComputeT>, RoundingMode::None>;

    const mpp::Or<ComputeT> op;

    const orInplaceSrc functor(aSrc2, aPitchSrc2, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, orInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeOrInplaceSrc_For(typeSrcIsTypeDst)                                                            \
    template void InvokeOrInplaceSrc<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                            \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeOrInplaceSrc(type)                                                                  \
    InstantiateInvokeOrInplaceSrc_For(Pixel##type##C1);                                                                \
    InstantiateInvokeOrInplaceSrc_For(Pixel##type##C2);                                                                \
    InstantiateInvokeOrInplaceSrc_For(Pixel##type##C3);                                                                \
    InstantiateInvokeOrInplaceSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeOrInplaceSrc(type)                                                                \
    InstantiateInvokeOrInplaceSrc_For(Pixel##type##C1);                                                                \
    InstantiateInvokeOrInplaceSrc_For(Pixel##type##C2);                                                                \
    InstantiateInvokeOrInplaceSrc_For(Pixel##type##C3);                                                                \
    InstantiateInvokeOrInplaceSrc_For(Pixel##type##C4);                                                                \
    InstantiateInvokeOrInplaceSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeOrInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                      const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using orInplaceC = InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Or<ComputeT>, RoundingMode::None>;

    const mpp::Or<ComputeT> op;

    const orInplaceC functor(static_cast<ComputeT>(aConst), op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, orInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeOrInplaceC_For(typeSrcIsTypeDst)                                                              \
    template void InvokeOrInplaceC<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                              \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeOrInplaceC(type)                                                                    \
    InstantiateInvokeOrInplaceC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeOrInplaceC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeOrInplaceC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeOrInplaceC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeOrInplaceC(type)                                                                  \
    InstantiateInvokeOrInplaceC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeOrInplaceC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeOrInplaceC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeOrInplaceC_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeOrInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeOrInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using orInplaceDevC = InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::Or<ComputeT>, RoundingMode::None>;

    const mpp::Or<ComputeT> op;

    const orInplaceDevC functor(aConst, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, orInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeOrInplaceDevC_For(typeSrcIsTypeDst)                                                           \
    template void InvokeOrInplaceDevC<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                           \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeOrInplaceDevC(type)                                                                 \
    InstantiateInvokeOrInplaceDevC_For(Pixel##type##C1);                                                               \
    InstantiateInvokeOrInplaceDevC_For(Pixel##type##C2);                                                               \
    InstantiateInvokeOrInplaceDevC_For(Pixel##type##C3);                                                               \
    InstantiateInvokeOrInplaceDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeOrInplaceDevC(type)                                                               \
    InstantiateInvokeOrInplaceDevC_For(Pixel##type##C1);                                                               \
    InstantiateInvokeOrInplaceDevC_For(Pixel##type##C2);                                                               \
    InstantiateInvokeOrInplaceDevC_For(Pixel##type##C3);                                                               \
    InstantiateInvokeOrInplaceDevC_For(Pixel##type##C4);                                                               \
    InstantiateInvokeOrInplaceDevC_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
