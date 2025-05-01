#if OPP_ENABLE_CUDA_BACKEND

#include "copy.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/image/forEachPixelPlanar2Kernel.h>
#include <backends/cuda/image/forEachPixelPlanar3Kernel.h>
#include <backends/cuda/image/forEachPixelPlanar4Kernel.h>
#include <backends/cuda/image/forEachPixelSingleChannelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/dataExchangeAndInit/operators.h>
#include <common/defines.h>
#include <common/image/channel.h>
#include <common/image/functors/srcFunctor.h>
#include <common/image/functors/srcPlanar2Functor.h>
#include <common/image/functors/srcPlanar3Functor.h>
#include <common/image/functors/srcPlanar4Functor.h>
#include <common/image/functors/srcSingleChannelFunctor.h>
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
template <typename SrcT, typename DstT>
void InvokeCopy(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using copySrc = SrcFunctor<TupelSize, SrcT, SrcT, DstT, opp::Copy<SrcT, DstT>, RoundingMode::None>;

        const opp::Copy<SrcT, DstT> op;

        const copySrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, copySrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeCopy<typeSrcIsTypeDst, typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,     \
                                                                 typeSrcIsTypeDst *aDst, size_t aPitchDst,             \
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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <SingleChannel SrcT, typename DstT>
void InvokeCopyChannel(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, Channel aDstChannel,
                       const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = 1;

        using copySrc = SrcFunctor<TupelSize, SrcT, SrcT, SrcT, opp::Copy<SrcT, SrcT>, RoundingMode::None>;

        const opp::Copy<SrcT, SrcT> op;

        const copySrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelSingleChannelKernelDefault<DstT, copySrc>(aDst, aPitchDst, aDstChannel, aSize, aStreamCtx,
                                                                    functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeCopyChannel<typeSrc, typeDst>(const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst,          \
                                                      size_t aPitchDst, Channel aDstChannel, const Size2D &aSize,      \
                                                      const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1, Pixel##type##C2);                                                                 \
    Instantiate_For(Pixel##type##C1, Pixel##type##C3);                                                                 \
    Instantiate_For(Pixel##type##C1, Pixel##type##C4);

ForAllChannelsNoAlpha(8u);
ForAllChannelsNoAlpha(8s);

ForAllChannelsNoAlpha(16u);
ForAllChannelsNoAlpha(16s);

ForAllChannelsNoAlpha(32u);
ForAllChannelsNoAlpha(32s);

ForAllChannelsNoAlpha(16f);
ForAllChannelsNoAlpha(16bf);
ForAllChannelsNoAlpha(32f);
ForAllChannelsNoAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, SingleChannel DstT>
void InvokeCopyChannel(const SrcT *aSrc1, size_t aPitchSrc1, Channel aSrcChannel, DstT *aDst, size_t aPitchDst,
                       const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using copySrc = SrcSingleChannelFunctor<TupelSize, SrcT, DstT, DstT, opp::Copy<DstT, DstT>, RoundingMode::None>;

        const opp::Copy<DstT, DstT> op;

        const copySrc functor(aSrc1, aPitchSrc1, aSrcChannel, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, copySrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeCopyChannel<typeSrc, typeDst>(const typeSrc *aSrc1, size_t aPitchSrc1, Channel aSrcChannel,    \
                                                      typeDst *aDst, size_t aPitchDst, const Size2D &aSize,            \
                                                      const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C2, Pixel##type##C1);                                                                 \
    Instantiate_For(Pixel##type##C3, Pixel##type##C1);                                                                 \
    Instantiate_For(Pixel##type##C4, Pixel##type##C1);

ForAllChannelsNoAlpha(8u);
ForAllChannelsNoAlpha(8s);

ForAllChannelsNoAlpha(16u);
ForAllChannelsNoAlpha(16s);

ForAllChannelsNoAlpha(32u);
ForAllChannelsNoAlpha(32s);

ForAllChannelsNoAlpha(16f);
ForAllChannelsNoAlpha(16bf);
ForAllChannelsNoAlpha(32f);
ForAllChannelsNoAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename DstT>
void InvokeCopyChannel(const SrcT *aSrc1, size_t aPitchSrc1, Channel aSrcChannel, DstT *aDst, size_t aPitchDst,
                       Channel aDstChannel, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        using ComputeT = Vector1<remove_vector_t<SrcT>>;

        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = 1;

        using copySrc = SrcSingleChannelFunctor<TupelSize, SrcT, ComputeT, ComputeT, opp::Copy<ComputeT, ComputeT>,
                                                RoundingMode::None>;

        const opp::Copy<ComputeT, ComputeT> op;

        const copySrc functor(aSrc1, aPitchSrc1, aSrcChannel, op);

        InvokeForEachPixelSingleChannelKernelDefault<DstT, copySrc>(aDst, aPitchDst, aDstChannel, aSize, aStreamCtx,
                                                                    functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeCopyChannel<typeSrc, typeDst>(const typeSrc *aSrc1, size_t aPitchSrc1, Channel aSrcChannel,    \
                                                      typeDst *aDst, size_t aPitchDst, Channel aDstChannel,            \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C2, Pixel##type##C2);                                                                 \
    Instantiate_For(Pixel##type##C2, Pixel##type##C3);                                                                 \
    Instantiate_For(Pixel##type##C2, Pixel##type##C4);                                                                 \
    Instantiate_For(Pixel##type##C3, Pixel##type##C2);                                                                 \
    Instantiate_For(Pixel##type##C3, Pixel##type##C3);                                                                 \
    Instantiate_For(Pixel##type##C3, Pixel##type##C4);                                                                 \
    Instantiate_For(Pixel##type##C4, Pixel##type##C2);                                                                 \
    Instantiate_For(Pixel##type##C4, Pixel##type##C3);                                                                 \
    Instantiate_For(Pixel##type##C4, Pixel##type##C4);

ForAllChannelsNoAlpha(8u);
ForAllChannelsNoAlpha(8s);

ForAllChannelsNoAlpha(16u);
ForAllChannelsNoAlpha(16s);

ForAllChannelsNoAlpha(32u);
ForAllChannelsNoAlpha(32s);

ForAllChannelsNoAlpha(16f);
ForAllChannelsNoAlpha(16bf);
ForAllChannelsNoAlpha(32f);
ForAllChannelsNoAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#pragma endregion

template <SingleChannel SrcT, TwoChannel DstT>
void InvokeCopyPlanar(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                      size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using copySrc = SrcPlanar2Functor<TupelSize, SrcT, DstT, DstT, opp::Copy<DstT, DstT>, RoundingMode::None>;

        const opp::Copy<DstT, DstT> op;

        const copySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, copySrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeCopyPlanar<typeSrc, typeDst>(const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2,    \
                                                     size_t aPitchSrc2, typeDst *aDst, size_t aPitchDst,               \
                                                     const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type) Instantiate_For(Pixel##type##C1, Pixel##type##C2);

ForAllChannelsNoAlpha(8u);
ForAllChannelsNoAlpha(8s);

ForAllChannelsNoAlpha(16u);
ForAllChannelsNoAlpha(16s);

ForAllChannelsNoAlpha(32u);
ForAllChannelsNoAlpha(32s);

ForAllChannelsNoAlpha(16f);
ForAllChannelsNoAlpha(16bf);
ForAllChannelsNoAlpha(32f);
ForAllChannelsNoAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#pragma endregion

template <SingleChannel SrcT, ThreeChannel DstT>
void InvokeCopyPlanar(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, const SrcT *aSrc3,
                      size_t aPitchSrc3, DstT *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using copySrc = SrcPlanar3Functor<TupelSize, SrcT, DstT, DstT, opp::Copy<DstT, DstT>, RoundingMode::None>;

        const opp::Copy<DstT, DstT> op;

        const copySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, copySrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeCopyPlanar<typeSrc, typeDst>(                                                                  \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2, const typeSrc *aSrc3,        \
        size_t aPitchSrc3, typeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type) Instantiate_For(Pixel##type##C1, Pixel##type##C3);

ForAllChannelsNoAlpha(8u);
ForAllChannelsNoAlpha(8s);

ForAllChannelsNoAlpha(16u);
ForAllChannelsNoAlpha(16s);

ForAllChannelsNoAlpha(32u);
ForAllChannelsNoAlpha(32s);

ForAllChannelsNoAlpha(16f);
ForAllChannelsNoAlpha(16bf);
ForAllChannelsNoAlpha(32f);
ForAllChannelsNoAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#pragma endregion

template <SingleChannel SrcT, FourChannel DstT>
void InvokeCopyPlanar(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, const SrcT *aSrc3,
                      size_t aPitchSrc3, const SrcT *aSrc4, size_t aPitchSrc4, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using copySrc = SrcPlanar4Functor<TupelSize, SrcT, DstT, DstT, opp::Copy<DstT, DstT>, RoundingMode::None>;

        const opp::Copy<DstT, DstT> op;

        const copySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, copySrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeCopyPlanar<typeSrc, typeDst>(                                                                  \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2, const typeSrc *aSrc3,        \
        size_t aPitchSrc3, const typeSrc *aSrc4, size_t aPitchSrc4, typeDst *aDst, size_t aPitchDst,                   \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type) Instantiate_For(Pixel##type##C1, Pixel##type##C4);

ForAllChannelsNoAlpha(8u);
ForAllChannelsNoAlpha(8s);

ForAllChannelsNoAlpha(16u);
ForAllChannelsNoAlpha(16s);

ForAllChannelsNoAlpha(32u);
ForAllChannelsNoAlpha(32s);

ForAllChannelsNoAlpha(16f);
ForAllChannelsNoAlpha(16bf);
ForAllChannelsNoAlpha(32f);
ForAllChannelsNoAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#pragma endregion

template <TwoChannel SrcT, SingleChannel DstT>
void InvokeCopyPlanar(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst1, size_t aPitchDst1, DstT *aDst2,
                      size_t aPitchDst2, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using copySrc = SrcFunctor<TupelSize, SrcT, SrcT, SrcT, opp::Copy<SrcT, SrcT>, RoundingMode::None>;

        const opp::Copy<SrcT, SrcT> op;

        const copySrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelPlanar2KernelDefault<SrcT, TupelSize, copySrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                         aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeCopyPlanar<typeSrc, typeDst>(const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst1,          \
                                                     size_t aPitchDst1, typeDst *aDst2, size_t aPitchDst2,             \
                                                     const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type) Instantiate_For(Pixel##type##C2, Pixel##type##C1);

ForAllChannelsNoAlpha(8u);
ForAllChannelsNoAlpha(8s);

ForAllChannelsNoAlpha(16u);
ForAllChannelsNoAlpha(16s);

ForAllChannelsNoAlpha(32u);
ForAllChannelsNoAlpha(32s);

ForAllChannelsNoAlpha(16f);
ForAllChannelsNoAlpha(16bf);
ForAllChannelsNoAlpha(32f);
ForAllChannelsNoAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#pragma endregion

template <ThreeChannel SrcT, SingleChannel DstT>
void InvokeCopyPlanar(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst1, size_t aPitchDst1, DstT *aDst2,
                      size_t aPitchDst2, DstT *aDst3, size_t aPitchDst3, const Size2D &aSize,
                      const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = 1;

        using copySrc = SrcFunctor<TupelSize, SrcT, SrcT, SrcT, opp::Copy<SrcT, SrcT>, RoundingMode::None>;

        const opp::Copy<SrcT, SrcT> op;

        const copySrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelPlanar3KernelDefault<SrcT, TupelSize, copySrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                         aPitchDst3, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeCopyPlanar<typeSrc, typeDst>(                                                                  \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst1, size_t aPitchDst1, typeDst *aDst2, size_t aPitchDst2, \
        typeDst *aDst3, size_t aPitchDst3, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type) Instantiate_For(Pixel##type##C3, Pixel##type##C1);

ForAllChannelsNoAlpha(8u);
ForAllChannelsNoAlpha(8s);

ForAllChannelsNoAlpha(16u);
ForAllChannelsNoAlpha(16s);

ForAllChannelsNoAlpha(32u);
ForAllChannelsNoAlpha(32s);

ForAllChannelsNoAlpha(16f);
ForAllChannelsNoAlpha(16bf);
ForAllChannelsNoAlpha(32f);
ForAllChannelsNoAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#pragma endregion

template <FourChannel SrcT, SingleChannel DstT>
void InvokeCopyPlanar(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst1, size_t aPitchDst1, DstT *aDst2,
                      size_t aPitchDst2, DstT *aDst3, size_t aPitchDst3, DstT *aDst4, size_t aPitchDst4,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using copySrc = SrcFunctor<TupelSize, SrcT, SrcT, SrcT, opp::Copy<SrcT, SrcT>, RoundingMode::None>;

        const opp::Copy<SrcT, SrcT> op;

        const copySrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelPlanar4KernelDefault<SrcT, TupelSize, copySrc>(
            aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeCopyPlanar<typeSrc, typeDst>(                                                                  \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst1, size_t aPitchDst1, typeDst *aDst2, size_t aPitchDst2, \
        typeDst *aDst3, size_t aPitchDst3, typeDst *aDst4, size_t aPitchDst4, const Size2D &aSize,                     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type) Instantiate_For(Pixel##type##C4, Pixel##type##C1);

ForAllChannelsNoAlpha(8u);
ForAllChannelsNoAlpha(8s);

ForAllChannelsNoAlpha(16u);
ForAllChannelsNoAlpha(16s);

ForAllChannelsNoAlpha(32u);
ForAllChannelsNoAlpha(32s);

ForAllChannelsNoAlpha(16f);
ForAllChannelsNoAlpha(16bf);
ForAllChannelsNoAlpha(32f);
ForAllChannelsNoAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#pragma endregion
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
