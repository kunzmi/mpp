#if MPP_ENABLE_CUDA_BACKEND

#include "xor.h"
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
#include <common/image/pixelTypeEnabler.h>
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
void InvokeXorSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                     size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using xorSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Xor<ComputeT>, RoundingMode::None>;

        const mpp::Xor<ComputeT> op;

        const xorSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, xorSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define InstantiateInvokeXorSrcSrc_For(typeSrcIsTypeDst)                                                               \
    template void InvokeXorSrcSrc<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                               \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeXorSrcSrc(type)                                                                     \
    InstantiateInvokeXorSrcSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeXorSrcSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeXorSrcSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeXorSrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeXorSrcSrc(type)                                                                   \
    InstantiateInvokeXorSrcSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeXorSrcSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeXorSrcSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeXorSrcSrc_For(Pixel##type##C4);                                                                   \
    InstantiateInvokeXorSrcSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeXorSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                   const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using xorSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Xor<ComputeT>, RoundingMode::None>;

        const mpp::Xor<ComputeT> op;

        const xorSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, xorSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeXorSrcC_For(typeSrcIsTypeDst)                                                                 \
    template void InvokeXorSrcC<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                                 \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeXorSrcC(type)                                                                       \
    InstantiateInvokeXorSrcC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeXorSrcC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeXorSrcC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeXorSrcC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeXorSrcC(type)                                                                     \
    InstantiateInvokeXorSrcC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeXorSrcC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeXorSrcC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeXorSrcC_For(Pixel##type##C4);                                                                     \
    InstantiateInvokeXorSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeXorSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using xorSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Xor<ComputeT>, RoundingMode::None>;

        const mpp::Xor<ComputeT> op;

        const xorSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, xorSrcDevC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define InstantiateInvokeXorSrcDevC_For(typeSrcIsTypeDst)                                                              \
    template void InvokeXorSrcDevC<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                              \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeXorSrcDevC(type)                                                                    \
    InstantiateInvokeXorSrcDevC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeXorSrcDevC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeXorSrcDevC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeXorSrcDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeXorSrcDevC(type)                                                                  \
    InstantiateInvokeXorSrcDevC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeXorSrcDevC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeXorSrcDevC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeXorSrcDevC_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeXorSrcDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeXorInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using xorInplaceSrc =
            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Xor<ComputeT>, RoundingMode::None>;

        const mpp::Xor<ComputeT> op;

        const xorInplaceSrc functor(aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, xorInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                        functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeXorInplaceSrc_For(typeSrcIsTypeDst)                                                           \
    template void InvokeXorInplaceSrc<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                           \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeXorInplaceSrc(type)                                                                 \
    InstantiateInvokeXorInplaceSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeXorInplaceSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeXorInplaceSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeXorInplaceSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeXorInplaceSrc(type)                                                               \
    InstantiateInvokeXorInplaceSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeXorInplaceSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeXorInplaceSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeXorInplaceSrc_For(Pixel##type##C4);                                                               \
    InstantiateInvokeXorInplaceSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeXorInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                       const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using xorInplaceC = InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Xor<ComputeT>, RoundingMode::None>;

        const mpp::Xor<ComputeT> op;

        const xorInplaceC functor(static_cast<ComputeT>(aConst), op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, xorInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                      functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeXorInplaceC_For(typeSrcIsTypeDst)                                                             \
    template void InvokeXorInplaceC<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                             \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeXorInplaceC(type)                                                                   \
    InstantiateInvokeXorInplaceC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeXorInplaceC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeXorInplaceC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeXorInplaceC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeXorInplaceC(type)                                                                 \
    InstantiateInvokeXorInplaceC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeXorInplaceC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeXorInplaceC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeXorInplaceC_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeXorInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeXorInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using xorInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::Xor<ComputeT>, RoundingMode::None>;

        const mpp::Xor<ComputeT> op;

        const xorInplaceDevC functor(aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, xorInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeXorInplaceDevC_For(typeSrcIsTypeDst)                                                          \
    template void InvokeXorInplaceDevC<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                          \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeXorInplaceDevC(type)                                                                \
    InstantiateInvokeXorInplaceDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeXorInplaceDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeXorInplaceDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeXorInplaceDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeXorInplaceDevC(type)                                                              \
    InstantiateInvokeXorInplaceDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeXorInplaceDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeXorInplaceDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeXorInplaceDevC_For(Pixel##type##C4);                                                              \
    InstantiateInvokeXorInplaceDevC_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
