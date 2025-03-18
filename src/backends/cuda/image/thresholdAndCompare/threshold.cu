#if OPP_ENABLE_CUDA_BACKEND

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
void InvokeThresholdLTSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThreshold, DstT *aDst, size_t aPitchDst,
                           const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdSrcC =
            SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Max<ComputeT>, RoundingMode::None>;

        const opp::Max<ComputeT> op;

        const thresholdSrcC functor(aSrc, aPitchSrc, aThreshold, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdLTSrcC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                         \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst &aThreshold, typeSrcIstypeDst *aDst,  \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aThreshold, DstT *aDst, size_t aPitchDst,
                              const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdSrcC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Max<ComputeT>, RoundingMode::None>;

        const opp::Max<ComputeT> op;

        const thresholdSrcC functor(aSrc, aPitchSrc, aThreshold, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdLTSrcDevC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                      \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst *aThreshold, typeSrcIstypeDst *aDst,  \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThreshold, const Size2D &aSize,
                               const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdInplaceC =
            InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Max<ComputeT>, RoundingMode::None>;

        const opp::Max<ComputeT> op;

        const thresholdInplaceC functor(aThreshold, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdLTInplaceC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                     \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst &aThreshold, const Size2D &aSize,      \
        const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aThreshold, const Size2D &aSize,
                                  const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdInplaceC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, opp::Max<ComputeT>, RoundingMode::None>;

        const opp::Max<ComputeT> op;

        const thresholdInplaceC functor(aThreshold, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdLTInplaceDevC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                  \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst *aThreshold, const Size2D &aSize,      \
        const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdGTSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThreshold, DstT *aDst, size_t aPitchDst,
                           const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdSrcC =
            SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Min<ComputeT>, RoundingMode::None>;

        const opp::Min<ComputeT> op;

        const thresholdSrcC functor(aSrc, aPitchSrc, aThreshold, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdGTSrcC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                         \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst &aThreshold, typeSrcIstypeDst *aDst,  \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdGTSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aThreshold, DstT *aDst, size_t aPitchDst,
                              const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdSrcC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Min<ComputeT>, RoundingMode::None>;

        const opp::Min<ComputeT> op;

        const thresholdSrcC functor(aSrc, aPitchSrc, aThreshold, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdGTSrcDevC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                      \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst *aThreshold, typeSrcIstypeDst *aDst,  \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdGTInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThreshold, const Size2D &aSize,
                               const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdInplaceC =
            InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Min<ComputeT>, RoundingMode::None>;

        const opp::Min<ComputeT> op;

        const thresholdInplaceC functor(aThreshold, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdGTInplaceC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                     \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst &aThreshold, const Size2D &aSize,      \
        const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdGTInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aThreshold, const Size2D &aSize,
                                  const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdInplaceC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, opp::Min<ComputeT>, RoundingMode::None>;

        const opp::Min<ComputeT> op;

        const thresholdInplaceC functor(aThreshold, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdGTInplaceDevC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                  \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst *aThreshold, const Size2D &aSize,      \
        const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTValSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThreshold, const SrcT &aValue,
                              DstT *aDst, size_t aPitchDst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::MaxVal<ComputeT>, RoundingMode::None>;

        const opp::MaxVal<ComputeT> op(aValue, aThreshold);

        const thresholdSrc functor(aSrc, aPitchSrc, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdLTValSrcC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                      \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst &aThreshold,                          \
        const typeSrcIstypeDst &aValue, typeSrcIstypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTValInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThreshold, const SrcT &aValue,
                                  const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdInplace = InplaceFunctor<TupelSize, ComputeT, DstT, opp::MaxVal<ComputeT>, RoundingMode::None>;

        const opp::MaxVal<ComputeT> op(aValue, aThreshold);

        const thresholdInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                           functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdLTValInplaceC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                  \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst &aThreshold,                           \
        const typeSrcIstypeDst &aValue, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdGTValSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThreshold, const SrcT &aValue,
                              DstT *aDst, size_t aPitchDst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::MinVal<ComputeT>, RoundingMode::None>;

        const opp::MinVal<ComputeT> op(aValue, aThreshold);

        const thresholdSrc functor(aSrc, aPitchSrc, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdGTValSrcC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                      \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst &aThreshold,                          \
        const typeSrcIstypeDst &aValue, typeSrcIstypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdGTValInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThreshold, const SrcT &aValue,
                                  const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdInplace = InplaceFunctor<TupelSize, ComputeT, DstT, opp::MinVal<ComputeT>, RoundingMode::None>;

        const opp::MinVal<ComputeT> op(aValue, aThreshold);

        const thresholdInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                           functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdGTValInplaceC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                  \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst &aThreshold,                           \
        const typeSrcIstypeDst &aValue, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTValGTValSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aThresholdLT, const SrcT &aValueLT,
                                   const SrcT &aThresholdGT, const SrcT &aValueGT, DstT *aDst, size_t aPitchDst,
                                   const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdSrc =
            SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::MinValMaxVal<ComputeT>, RoundingMode::None>;

        const opp::MinValMaxVal<ComputeT> op(aValueGT, aThresholdGT, aValueLT, aThresholdLT);

        const thresholdSrc functor(aSrc, aPitchSrc, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdLTValGTValSrcC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                 \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst &aThresholdLT,                        \
        const typeSrcIstypeDst &aValueLT, const typeSrcIstypeDst &aThresholdGT, const typeSrcIstypeDst &aValueGT,      \
        typeSrcIstypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeThresholdLTValGTValInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aThresholdLT,
                                       const SrcT &aValueLT, const SrcT &aThresholdGT, const SrcT &aValueGT,
                                       const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using thresholdInplace =
            InplaceFunctor<TupelSize, ComputeT, DstT, opp::MinValMaxVal<ComputeT>, RoundingMode::None>;

        const opp::MinValMaxVal<ComputeT> op(aValueGT, aThresholdGT, aValueLT, aThresholdLT);

        const thresholdInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, thresholdInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                           functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeThresholdLTValGTValInplaceC<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(             \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst &aThresholdLT,                         \
        const typeSrcIstypeDst &aValueLT, const typeSrcIstypeDst &aThresholdGT, const typeSrcIstypeDst &aValueGT,      \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
