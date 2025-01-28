#if OPP_ENABLE_CUDA_BACKEND

#include "setMasked.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelMaskedKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/constantFunctor.h>
#include <common/image/functors/devConstantFunctor.h>
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
template <typename DstT>
void InvokeSetCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const DstT &aConst, DstT *aDst, size_t aPitchDst,
                    const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using setC = ConstantFunctor<TupelSize, DstT>;

        setC functor(aConst);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, setC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                     aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define Instantiate_For(typeDst)                                                                                       \
    template void InvokeSetCMask<typeDst>(const Pixel8uC1 *aMask, size_t aPitchMask, const typeDst &aConst,            \
                                          typeDst *aDst, size_t aPitchDst, const Size2D &aSize,                        \
                                          const StreamCtx &aStreamCtx);

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

template <typename DstT>
void InvokeSetDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const DstT *aConst, DstT *aDst, size_t aPitchDst,
                       const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using setDevC = DevConstantFunctor<TupelSize, DstT>;

        setDevC functor(aConst);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, setDevC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                        aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define Instantiate_For(typeDst)                                                                                       \
    template void InvokeSetDevCMask<typeDst>(const Pixel8uC1 *aMask, size_t aPitchMask, const typeDst *aConst,         \
                                             typeDst *aDst, size_t aPitchDst, const Size2D &aSize,                     \
                                             const StreamCtx &aStreamCtx);

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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
