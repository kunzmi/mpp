#if OPP_ENABLE_CUDA_BACKEND

#include "swapChannel.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/unary_operators.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/channel.h>
#include <common/image/channelList.h>
#include <common/image/functors/srcDstAsSrcFunctor.h>
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
template <typename SrcT, typename DstT>
void InvokeSwapChannelSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                          const ChannelList<vector_active_size_v<DstT>> &aDstChannels, const Size2D &aSize,
                          const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using swapChannelSrc =
            SrcFunctor<TupelSize, SrcT, SrcT, DstT, opp::SwapChannel<SrcT, DstT>, RoundingMode::None>;

        SwapChannel<SrcT, DstT> op(aDstChannels);

        swapChannelSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, swapChannelSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeSwapChannelSrc<typeSrc, typeDst>(                                                              \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst,                                      \
        const ChannelList<vector_active_size_v<typeDst>> &aDstChannels, const Size2D &aSize,                           \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C3, Pixel##type##C3);                                                                 \
    Instantiate_For(Pixel##type##C4, Pixel##type##C4);                                                                 \
    Instantiate_For(Pixel##type##C4, Pixel##type##C3);                                                                 \
    Instantiate_For(Pixel##type##C4A, Pixel##type##C4A);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C3, Pixel##type##C3);                                                                 \
    Instantiate_For(Pixel##type##C4, Pixel##type##C4);                                                                 \
    Instantiate_For(Pixel##type##C4, Pixel##type##C3);

ForAllChannelsWithAlpha(8s);
ForAllChannelsWithAlpha(8u);

ForAllChannelsWithAlpha(16s);
ForAllChannelsWithAlpha(16u);

ForAllChannelsWithAlpha(32s);
ForAllChannelsWithAlpha(32u);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#undef ForAllChannelsWithAlpha
#pragma endregion

template <ThreeChannel SrcT, FourChannelNoAlpha DstT>
void InvokeSwapChannelSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                          const ChannelList<vector_active_size_v<DstT>> &aDstChannels, remove_vector_t<DstT> aValue,
                          const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = 1; // ConfigTupelSize<"Default", sizeof(DstT)>::value;

        bool allChannelsSet = true;
        for (size_t i = 0; i < vector_active_size_v<DstT>; i++)
        {
            allChannelsSet &= aDstChannels.data()[i].template IsInRange<DstT>();
        }

        if (allChannelsSet)
        {
            using swapChannelSrc =
                SrcFunctor<TupelSize, SrcT, SrcT, DstT, opp::SwapChannel<SrcT, DstT>, RoundingMode::None>;

            SwapChannel<SrcT, DstT> op(aDstChannels, aValue);

            swapChannelSrc functor(aSrc1, aPitchSrc1, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, swapChannelSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                             functor);
        }
        else
        {
            using swapChannelSrcDstAsSrc = SrcDstAsSrcFunctor<TupelSize, SrcT, DstT, opp::SwapChannel<SrcT, DstT>>;

            SwapChannel<SrcT, DstT> op(aDstChannels, aValue);

            swapChannelSrcDstAsSrc functor(aSrc1, aPitchSrc1, aDst, aPitchDst, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, swapChannelSrcDstAsSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                     functor);
        }
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeSwapChannelSrc<typeSrc, typeDst>(                                                              \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst,                                      \
        const ChannelList<vector_active_size_v<typeDst>> &aDstChannels, remove_vector_t<typeDst> aValue,               \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type) Instantiate_For(Pixel##type##C3, Pixel##type##C4);

#define ForAllChannelsNoAlpha(type) Instantiate_For(Pixel##type##C3, Pixel##type##C4);

ForAllChannelsWithAlpha(8s);
ForAllChannelsWithAlpha(8u);

ForAllChannelsWithAlpha(16s);
ForAllChannelsWithAlpha(16u);

ForAllChannelsWithAlpha(32s);
ForAllChannelsWithAlpha(32u);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#undef ForAllChannelsWithAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
