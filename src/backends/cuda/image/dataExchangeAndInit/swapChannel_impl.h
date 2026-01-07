#include "swapChannel.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/dataExchangeAndInit/operators.h>
#include <common/defines.h>
#include <common/image/channel.h>
#include <common/image/channelList.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/srcDstAsSrcFunctor.h>
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
template <typename SrcT, typename DstT>
void InvokeSwapChannelSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                          const ChannelList<vector_active_size_v<DstT>> &aDstChannels, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

    // in case that either input or ourput type is three-channel, we can't use tupels:
    if constexpr (vector_size_v<SrcT> == 3 || vector_size_v<DstT> == 3)
    {
        constexpr size_t TupelSize = 1;

        using swapChannelSrc =
            SrcFunctor<TupelSize, SrcT, SrcT, DstT, mpp::SwapChannel<SrcT, DstT>, RoundingMode::None>;

        const mpp::SwapChannel<SrcT, DstT> op(aDstChannels);

        const swapChannelSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, swapChannelSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else
    {
        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using swapChannelSrc =
            SrcFunctor<TupelSize, SrcT, SrcT, DstT, mpp::SwapChannel<SrcT, DstT>, RoundingMode::None>;

        const mpp::SwapChannel<SrcT, DstT> op(aDstChannels);

        const swapChannelSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, swapChannelSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define InstantiateInvokeSwapChannelSrc_For(typeSrc, typeDst)                                                          \
    template void InvokeSwapChannelSrc<typeSrc, typeDst>(                                                              \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst,                                      \
        const ChannelList<vector_active_size_v<typeDst>> &aDstChannels, const Size2D &aSize,                           \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSwapChannelSrc(type)                                                                \
    InstantiateInvokeSwapChannelSrc_For(Pixel##type##C3, Pixel##type##C3);                                             \
    InstantiateInvokeSwapChannelSrc_For(Pixel##type##C3, Pixel##type##C4);                                             \
    InstantiateInvokeSwapChannelSrc_For(Pixel##type##C4, Pixel##type##C3);                                             \
    InstantiateInvokeSwapChannelSrc_For(Pixel##type##C4, Pixel##type##C4);
#pragma endregion

template <ThreeChannel SrcT, FourChannelNoAlpha DstT>
void InvokeSwapChannelSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                          const ChannelList<vector_active_size_v<DstT>> &aDstChannels, remove_vector_t<DstT> aValue,
                          const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

    constexpr size_t TupelSize = 1; // ConfigTupelSize<"Default", sizeof(DstT)>::value;

    bool allChannelsSet = true;
    for (size_t i = 0; i < vector_active_size_v<DstT>; i++)
    {
        allChannelsSet &= aDstChannels.data()[i].template IsInRange<DstT>();
    }

    if (allChannelsSet)
    {
        using swapChannelSrc =
            SrcFunctor<TupelSize, SrcT, SrcT, DstT, mpp::SwapChannel<SrcT, DstT>, RoundingMode::None>;

        const mpp::SwapChannel<SrcT, DstT> op(aDstChannels, aValue);

        const swapChannelSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, swapChannelSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else
    {
        using swapChannelSrcDstAsSrc = SrcDstAsSrcFunctor<TupelSize, SrcT, DstT, mpp::SwapChannel<SrcT, DstT>>;

        const mpp::SwapChannel<SrcT, DstT> op(aDstChannels, aValue);

        const swapChannelSrcDstAsSrc functor(aSrc1, aPitchSrc1, aDst, aPitchDst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, swapChannelSrcDstAsSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                 functor);
    }
}

#pragma region Instantiate

#define InstantiateInvokeSwapChannelSrc34_For(typeSrc, typeDst)                                                        \
    template void InvokeSwapChannelSrc<typeSrc, typeDst>(                                                              \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst,                                      \
        const ChannelList<vector_active_size_v<typeDst>> &aDstChannels, remove_vector_t<typeDst> aValue,               \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSwapChannelSrc34(type)                                                              \
    InstantiateInvokeSwapChannelSrc34_For(Pixel##type##C3, Pixel##type##C4);
#pragma endregion

template <typename SrcT>
void InvokeSwapChannelInplace(SrcT *aSrcDst, size_t aPitchSrcDst,
                              const ChannelList<vector_active_size_v<SrcT>> &aDstChannels, const Size2D &aSize,
                              const mpp::cuda::StreamCtx &aStreamCtx)
{
    using DstT = SrcT;
    MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using swapChannelInplace = InplaceFunctor<TupelSize, SrcT, DstT, mpp::SwapChannel<SrcT, DstT>, RoundingMode::None>;

    const mpp::SwapChannel<SrcT, DstT> op(aDstChannels);

    const swapChannelInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, swapChannelInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
}

#pragma region Instantiate

#define InstantiateInvokeSwapChannelInplace_For(typeSrc)                                                               \
    template void InvokeSwapChannelInplace<typeSrc>(typeSrc * aSrcDst, size_t aPitchSrcDst,                            \
                                                    const ChannelList<vector_active_size_v<typeSrc>> &aDstChannels,    \
                                                    const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSwapChannelInplace(type)                                                            \
    InstantiateInvokeSwapChannelInplace_For(Pixel##type##C3);                                                          \
    InstantiateInvokeSwapChannelInplace_For(Pixel##type##C4);
#pragma endregion

template <TwoChannel SrcDstT>
void InvokeSwapChannelSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT = SrcDstT;
    using DstT = SrcDstT;
    MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using swapChannelSrc = SrcFunctor<TupelSize, SrcT, SrcT, DstT, mpp::SwapChannel<SrcDstT>, RoundingMode::None>;

    const mpp::SwapChannel<SrcDstT> op;

    const swapChannelSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, swapChannelSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate

#define InstantiateInvokeSwapChannelSrc2_For(typeSrcDst)                                                               \
    template void InvokeSwapChannelSrc<typeSrcDst>(const typeSrcDst *aSrc1, size_t aPitchSrc1, typeSrcDst *aDst,       \
                                                   size_t aPitchDst, const Size2D &aSize,                              \
                                                   const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSwapChannelSrc2(type) InstantiateInvokeSwapChannelSrc2_For(Pixel##type##C2);
#pragma endregion

template <TwoChannel SrcDstT>
void InvokeSwapChannelInplace(SrcDstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize,
                              const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT = SrcDstT;
    using DstT = SrcDstT;
    MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using swapChannelInplace = InplaceFunctor<TupelSize, SrcT, DstT, mpp::SwapChannel<SrcDstT>, RoundingMode::None>;

    const mpp::SwapChannel<SrcDstT> op;

    const swapChannelInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, swapChannelInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
}

#pragma region Instantiate

#define InstantiateInvokeSwapChannelInplace2_For(typeSrc)                                                              \
    template void InvokeSwapChannelInplace<typeSrc>(typeSrc * aSrcDst, size_t aPitchSrcDst, const Size2D &aSize,       \
                                                    const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSwapChannelInplace2(type) InstantiateInvokeSwapChannelInplace2_For(Pixel##type##C2);
#pragma endregion

} // namespace mpp::image::cuda
