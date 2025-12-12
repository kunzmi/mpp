#include "threshold.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceConstantFunctor.h>
#include <common/image/functors/inplaceDevConstantFunctor.h>
#include <common/image/functors/inplaceFunctor.h>
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
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThreshold, DstT *aDst, size_t aPitchDst,
                           const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Max<ComputeT>, RoundingMode::None>;

    const mpp::Max<ComputeT> op;

    const thresholdSrcC functor(aSrc, aPitchSrc, aThreshold, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdLTSrcC_For(typeSrcIstypeDst)                                                         \
    template void InvokeThresholdLTSrcC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                         \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst &aThreshold, typeSrcIstypeDst *aDst,  \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdLTSrcC(type)                                                             \
    InstantiateInvokeThresholdLTSrcC_For(Pixel##type##C1);                                                             \
    InstantiateInvokeThresholdLTSrcC_For(Pixel##type##C2);                                                             \
    InstantiateInvokeThresholdLTSrcC_For(Pixel##type##C3);                                                             \
    InstantiateInvokeThresholdLTSrcC_For(Pixel##type##C4);                                                             \
    InstantiateInvokeThresholdLTSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aThreshold, DstT *aDst, size_t aPitchDst,
                              const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdSrcC =
        SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Max<ComputeT>, RoundingMode::None>;

    const mpp::Max<ComputeT> op;

    const thresholdSrcC functor(aSrc, aPitchSrc, aThreshold, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdLTSrcDevC_For(typeSrcIstypeDst)                                                      \
    template void InvokeThresholdLTSrcDevC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                      \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst *aThreshold, typeSrcIstypeDst *aDst,  \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdLTSrcDevC(type)                                                          \
    InstantiateInvokeThresholdLTSrcDevC_For(Pixel##type##C1);                                                          \
    InstantiateInvokeThresholdLTSrcDevC_For(Pixel##type##C2);                                                          \
    InstantiateInvokeThresholdLTSrcDevC_For(Pixel##type##C3);                                                          \
    InstantiateInvokeThresholdLTSrcDevC_For(Pixel##type##C4);                                                          \
    InstantiateInvokeThresholdLTSrcDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThreshold, const Size2D &aSize,
                               const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdInplaceC = InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Max<ComputeT>, RoundingMode::None>;

    const mpp::Max<ComputeT> op;

    const thresholdInplaceC functor(aThreshold, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                        functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdLTInplaceC_For(typeSrcIstypeDst)                                                     \
    template void InvokeThresholdLTInplaceC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                     \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst &aThreshold, const Size2D &aSize,      \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdLTInplaceC(type)                                                         \
    InstantiateInvokeThresholdLTInplaceC_For(Pixel##type##C1);                                                         \
    InstantiateInvokeThresholdLTInplaceC_For(Pixel##type##C2);                                                         \
    InstantiateInvokeThresholdLTInplaceC_For(Pixel##type##C3);                                                         \
    InstantiateInvokeThresholdLTInplaceC_For(Pixel##type##C4);                                                         \
    InstantiateInvokeThresholdLTInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aThreshold, const Size2D &aSize,
                                  const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdInplaceC =
        InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::Max<ComputeT>, RoundingMode::None>;

    const mpp::Max<ComputeT> op;

    const thresholdInplaceC functor(aThreshold, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                        functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdLTInplaceDevC_For(typeSrcIstypeDst)                                                  \
    template void InvokeThresholdLTInplaceDevC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                  \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst *aThreshold, const Size2D &aSize,      \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdLTInplaceDevC(type)                                                      \
    InstantiateInvokeThresholdLTInplaceDevC_For(Pixel##type##C1);                                                      \
    InstantiateInvokeThresholdLTInplaceDevC_For(Pixel##type##C2);                                                      \
    InstantiateInvokeThresholdLTInplaceDevC_For(Pixel##type##C3);                                                      \
    InstantiateInvokeThresholdLTInplaceDevC_For(Pixel##type##C4);                                                      \
    InstantiateInvokeThresholdLTInplaceDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdGTSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThreshold, DstT *aDst, size_t aPitchDst,
                           const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Min<ComputeT>, RoundingMode::None>;

    const mpp::Min<ComputeT> op;

    const thresholdSrcC functor(aSrc, aPitchSrc, aThreshold, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdGTSrcC_For(typeSrcIstypeDst)                                                         \
    template void InvokeThresholdGTSrcC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                         \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst &aThreshold, typeSrcIstypeDst *aDst,  \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdGTSrcC(type)                                                             \
    InstantiateInvokeThresholdGTSrcC_For(Pixel##type##C1);                                                             \
    InstantiateInvokeThresholdGTSrcC_For(Pixel##type##C2);                                                             \
    InstantiateInvokeThresholdGTSrcC_For(Pixel##type##C3);                                                             \
    InstantiateInvokeThresholdGTSrcC_For(Pixel##type##C4);                                                             \
    InstantiateInvokeThresholdGTSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdGTSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aThreshold, DstT *aDst, size_t aPitchDst,
                              const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdSrcC =
        SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Min<ComputeT>, RoundingMode::None>;

    const mpp::Min<ComputeT> op;

    const thresholdSrcC functor(aSrc, aPitchSrc, aThreshold, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdGTSrcDevC_For(typeSrcIstypeDst)                                                      \
    template void InvokeThresholdGTSrcDevC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                      \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst *aThreshold, typeSrcIstypeDst *aDst,  \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdGTSrcDevC(type)                                                          \
    InstantiateInvokeThresholdGTSrcDevC_For(Pixel##type##C1);                                                          \
    InstantiateInvokeThresholdGTSrcDevC_For(Pixel##type##C2);                                                          \
    InstantiateInvokeThresholdGTSrcDevC_For(Pixel##type##C3);                                                          \
    InstantiateInvokeThresholdGTSrcDevC_For(Pixel##type##C4);                                                          \
    InstantiateInvokeThresholdGTSrcDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdGTInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThreshold, const Size2D &aSize,
                               const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdInplaceC = InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Min<ComputeT>, RoundingMode::None>;

    const mpp::Min<ComputeT> op;

    const thresholdInplaceC functor(aThreshold, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                        functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdGTInplaceC_For(typeSrcIstypeDst)                                                     \
    template void InvokeThresholdGTInplaceC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                     \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst &aThreshold, const Size2D &aSize,      \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdGTInplaceC(type)                                                         \
    InstantiateInvokeThresholdGTInplaceC_For(Pixel##type##C1);                                                         \
    InstantiateInvokeThresholdGTInplaceC_For(Pixel##type##C2);                                                         \
    InstantiateInvokeThresholdGTInplaceC_For(Pixel##type##C3);                                                         \
    InstantiateInvokeThresholdGTInplaceC_For(Pixel##type##C4);                                                         \
    InstantiateInvokeThresholdGTInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdGTInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aThreshold, const Size2D &aSize,
                                  const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdInplaceC =
        InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::Min<ComputeT>, RoundingMode::None>;

    const mpp::Min<ComputeT> op;

    const thresholdInplaceC functor(aThreshold, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                        functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdGTInplaceDevC_For(typeSrcIstypeDst)                                                  \
    template void InvokeThresholdGTInplaceDevC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                  \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst *aThreshold, const Size2D &aSize,      \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdGTInplaceDevC(type)                                                      \
    InstantiateInvokeThresholdGTInplaceDevC_For(Pixel##type##C1);                                                      \
    InstantiateInvokeThresholdGTInplaceDevC_For(Pixel##type##C2);                                                      \
    InstantiateInvokeThresholdGTInplaceDevC_For(Pixel##type##C3);                                                      \
    InstantiateInvokeThresholdGTInplaceDevC_For(Pixel##type##C4);                                                      \
    InstantiateInvokeThresholdGTInplaceDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTValSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThreshold, const SrcT &aValue,
                              DstT *aDst, size_t aPitchDst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::MaxVal<ComputeT>, RoundingMode::None>;

    const mpp::MaxVal<ComputeT> op(aValue, aThreshold);

    const thresholdSrc functor(aSrc, aPitchSrc, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdLTValSrcC_For(typeSrcIstypeDst)                                                      \
    template void InvokeThresholdLTValSrcC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                      \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst &aThreshold,                          \
        const typeSrcIstypeDst &aValue, typeSrcIstypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdLTValSrcC(type)                                                          \
    InstantiateInvokeThresholdLTValSrcC_For(Pixel##type##C1);                                                          \
    InstantiateInvokeThresholdLTValSrcC_For(Pixel##type##C2);                                                          \
    InstantiateInvokeThresholdLTValSrcC_For(Pixel##type##C3);                                                          \
    InstantiateInvokeThresholdLTValSrcC_For(Pixel##type##C4);                                                          \
    InstantiateInvokeThresholdLTValSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTValInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThreshold, const SrcT &aValue,
                                  const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::MaxVal<ComputeT>, RoundingMode::None>;

    const mpp::MaxVal<ComputeT> op(aValue, aThreshold);

    const thresholdInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                       functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdLTValInplaceC_For(typeSrcIstypeDst)                                                  \
    template void InvokeThresholdLTValInplaceC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                  \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst &aThreshold,                           \
        const typeSrcIstypeDst &aValue, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdLTValInplaceC(type)                                                      \
    InstantiateInvokeThresholdLTValInplaceC_For(Pixel##type##C1);                                                      \
    InstantiateInvokeThresholdLTValInplaceC_For(Pixel##type##C2);                                                      \
    InstantiateInvokeThresholdLTValInplaceC_For(Pixel##type##C3);                                                      \
    InstantiateInvokeThresholdLTValInplaceC_For(Pixel##type##C4);                                                      \
    InstantiateInvokeThresholdLTValInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdGTValSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThreshold, const SrcT &aValue,
                              DstT *aDst, size_t aPitchDst, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::MinVal<ComputeT>, RoundingMode::None>;

    const mpp::MinVal<ComputeT> op(aValue, aThreshold);

    const thresholdSrc functor(aSrc, aPitchSrc, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdGTValSrcC_For(typeSrcIstypeDst)                                                      \
    template void InvokeThresholdGTValSrcC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                      \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst &aThreshold,                          \
        const typeSrcIstypeDst &aValue, typeSrcIstypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdGTValSrcC(type)                                                          \
    InstantiateInvokeThresholdGTValSrcC_For(Pixel##type##C1);                                                          \
    InstantiateInvokeThresholdGTValSrcC_For(Pixel##type##C2);                                                          \
    InstantiateInvokeThresholdGTValSrcC_For(Pixel##type##C3);                                                          \
    InstantiateInvokeThresholdGTValSrcC_For(Pixel##type##C4);                                                          \
    InstantiateInvokeThresholdGTValSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdGTValInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThreshold, const SrcT &aValue,
                                  const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::MinVal<ComputeT>, RoundingMode::None>;

    const mpp::MinVal<ComputeT> op(aValue, aThreshold);

    const thresholdInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                       functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdGTValInplaceC_For(typeSrcIstypeDst)                                                  \
    template void InvokeThresholdGTValInplaceC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                  \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst &aThreshold,                           \
        const typeSrcIstypeDst &aValue, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdGTValInplaceC(type)                                                      \
    InstantiateInvokeThresholdGTValInplaceC_For(Pixel##type##C1);                                                      \
    InstantiateInvokeThresholdGTValInplaceC_For(Pixel##type##C2);                                                      \
    InstantiateInvokeThresholdGTValInplaceC_For(Pixel##type##C3);                                                      \
    InstantiateInvokeThresholdGTValInplaceC_For(Pixel##type##C4);                                                      \
    InstantiateInvokeThresholdGTValInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTValGTValSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThresholdLT, const SrcT &aValueLT,
                                   const SrcT &aThresholdGT, const SrcT &aValueGT, DstT *aDst, size_t aPitchDst,
                                   const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::MinValMaxVal<ComputeT>, RoundingMode::None>;

    const mpp::MinValMaxVal<ComputeT> op(aValueGT, aThresholdGT, aValueLT, aThresholdLT);

    const thresholdSrc functor(aSrc, aPitchSrc, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdLTValGTValSrcC_For(typeSrcIstypeDst)                                                 \
    template void InvokeThresholdLTValGTValSrcC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                 \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst &aThresholdLT,                        \
        const typeSrcIstypeDst &aValueLT, const typeSrcIstypeDst &aThresholdGT, const typeSrcIstypeDst &aValueGT,      \
        typeSrcIstypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdLTValGTValSrcC(type)                                                     \
    InstantiateInvokeThresholdLTValGTValSrcC_For(Pixel##type##C1);                                                     \
    InstantiateInvokeThresholdLTValGTValSrcC_For(Pixel##type##C2);                                                     \
    InstantiateInvokeThresholdLTValGTValSrcC_For(Pixel##type##C3);                                                     \
    InstantiateInvokeThresholdLTValGTValSrcC_For(Pixel##type##C4);                                                     \
    InstantiateInvokeThresholdLTValGTValSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTValGTValInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThresholdLT,
                                       const SrcT &aValueLT, const SrcT &aThresholdGT, const SrcT &aValueGT,
                                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using thresholdInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::MinValMaxVal<ComputeT>, RoundingMode::None>;

    const mpp::MinValMaxVal<ComputeT> op(aValueGT, aThresholdGT, aValueLT, aThresholdLT);

    const thresholdInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                       functor);
}

#pragma region Instantiate

#define InstantiateInvokeThresholdLTValGTValInplaceC_For(typeSrcIstypeDst)                                             \
    template void InvokeThresholdLTValGTValInplaceC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(             \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst &aThresholdLT,                         \
        const typeSrcIstypeDst &aValueLT, const typeSrcIstypeDst &aThresholdGT, const typeSrcIstypeDst &aValueGT,      \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeThresholdLTValGTValInplaceC(type)                                                 \
    InstantiateInvokeThresholdLTValGTValInplaceC_For(Pixel##type##C1);                                                 \
    InstantiateInvokeThresholdLTValGTValInplaceC_For(Pixel##type##C2);                                                 \
    InstantiateInvokeThresholdLTValGTValInplaceC_For(Pixel##type##C3);                                                 \
    InstantiateInvokeThresholdLTValGTValInplaceC_For(Pixel##type##C4);                                                 \
    InstantiateInvokeThresholdLTValGTValInplaceC_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
