#if MPP_ENABLE_CUDA_BACKEND

#include "and.h"
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
void InvokeAndSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                     size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using andSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::And<ComputeT>, RoundingMode::None>;

        const mpp::And<ComputeT> op;

        const andSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, andSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define InstantiateInvokeAndSrcSrc_For(typeSrcIsTypeDst)                                                               \
    template void InvokeAndSrcSrc<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                               \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAndSrcSrc(type)                                                                     \
    InstantiateInvokeAndSrcSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeAndSrcSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeAndSrcSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeAndSrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAndSrcSrc(type)                                                                   \
    InstantiateInvokeAndSrcSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeAndSrcSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeAndSrcSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeAndSrcSrc_For(Pixel##type##C4);                                                                   \
    InstantiateInvokeAndSrcSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAndSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                   const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using andSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::And<ComputeT>, RoundingMode::None>;

        const mpp::And<ComputeT> op;

        const andSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, andSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeAndSrcC_For(typeSrcIsTypeDst)                                                                 \
    template void InvokeAndSrcC<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                                 \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAndSrcC(type)                                                                       \
    InstantiateInvokeAndSrcC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeAndSrcC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeAndSrcC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeAndSrcC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAndSrcC(type)                                                                     \
    InstantiateInvokeAndSrcC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeAndSrcC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeAndSrcC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeAndSrcC_For(Pixel##type##C4);                                                                     \
    InstantiateInvokeAndSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAndSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using andSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::And<ComputeT>, RoundingMode::None>;

        const mpp::And<ComputeT> op;

        const andSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, andSrcDevC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define InstantiateInvokeAndSrcDevC_For(typeSrcIsTypeDst)                                                              \
    template void InvokeAndSrcDevC<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                              \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAndSrcDevC(type)                                                                    \
    InstantiateInvokeAndSrcDevC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeAndSrcDevC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeAndSrcDevC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeAndSrcDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAndSrcDevC(type)                                                                  \
    InstantiateInvokeAndSrcDevC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeAndSrcDevC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeAndSrcDevC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeAndSrcDevC_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeAndSrcDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAndInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using andInplaceSrc =
            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::And<ComputeT>, RoundingMode::None>;

        const mpp::And<ComputeT> op;

        const andInplaceSrc functor(aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, andInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                        functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeAndInplaceSrc_For(typeSrcIsTypeDst)                                                           \
    template void InvokeAndInplaceSrc<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                           \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAndInplaceSrc(type)                                                                 \
    InstantiateInvokeAndInplaceSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeAndInplaceSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeAndInplaceSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeAndInplaceSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAndInplaceSrc(type)                                                               \
    InstantiateInvokeAndInplaceSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeAndInplaceSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeAndInplaceSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeAndInplaceSrc_For(Pixel##type##C4);                                                               \
    InstantiateInvokeAndInplaceSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAndInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                       const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using andInplaceC = InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::And<ComputeT>, RoundingMode::None>;

        const mpp::And<ComputeT> op;

        const andInplaceC functor(static_cast<ComputeT>(aConst), op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, andInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                      functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeAndInplaceC_For(typeSrcIsTypeDst)                                                             \
    template void InvokeAndInplaceC<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                             \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAndInplaceC(type)                                                                   \
    InstantiateInvokeAndInplaceC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeAndInplaceC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeAndInplaceC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeAndInplaceC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAndInplaceC(type)                                                                 \
    InstantiateInvokeAndInplaceC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeAndInplaceC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeAndInplaceC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeAndInplaceC_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeAndInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAndInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using andInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::And<ComputeT>, RoundingMode::None>;

        const mpp::And<ComputeT> op;

        const andInplaceDevC functor(aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, andInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeAndInplaceDevC_For(typeSrcIsTypeDst)                                                          \
    template void InvokeAndInplaceDevC<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                          \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAndInplaceDevC(type)                                                                \
    InstantiateInvokeAndInplaceDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeAndInplaceDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeAndInplaceDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeAndInplaceDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAndInplaceDevC(type)                                                              \
    InstantiateInvokeAndInplaceDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeAndInplaceDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeAndInplaceDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeAndInplaceDevC_For(Pixel##type##C4);                                                              \
    InstantiateInvokeAndInplaceDevC_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
