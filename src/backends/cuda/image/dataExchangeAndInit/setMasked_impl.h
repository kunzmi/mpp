#if MPP_ENABLE_CUDA_BACKEND

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
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{
template <typename DstT>
void InvokeSetCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const DstT &aConst, DstT *aDst, size_t aPitchDst,
                    const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using setC = ConstantFunctor<TupelSize, DstT>;

        const setC functor(aConst);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, setC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                     aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeSetCMask_For(typeDst)                                                                         \
    template void InvokeSetCMask<typeDst>(const Pixel8uC1 *aMask, size_t aPitchMask, const typeDst &aConst,            \
                                          typeDst *aDst, size_t aPitchDst, const Size2D &aSize,                        \
                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSetCMask(type)                                                                      \
    InstantiateInvokeSetCMask_For(Pixel##type##C1);                                                                    \
    InstantiateInvokeSetCMask_For(Pixel##type##C2);                                                                    \
    InstantiateInvokeSetCMask_For(Pixel##type##C3);                                                                    \
    InstantiateInvokeSetCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSetCMask(type)                                                                    \
    InstantiateInvokeSetCMask_For(Pixel##type##C1);                                                                    \
    InstantiateInvokeSetCMask_For(Pixel##type##C2);                                                                    \
    InstantiateInvokeSetCMask_For(Pixel##type##C3);                                                                    \
    InstantiateInvokeSetCMask_For(Pixel##type##C4);                                                                    \
    InstantiateInvokeSetCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename DstT>
void InvokeSetDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const DstT *aConst, DstT *aDst, size_t aPitchDst,
                       const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using setDevC = DevConstantFunctor<TupelSize, DstT>;

        const setDevC functor(aConst);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, setDevC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                        aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeSetDevCMask_For(typeDst)                                                                      \
    template void InvokeSetDevCMask<typeDst>(const Pixel8uC1 *aMask, size_t aPitchMask, const typeDst *aConst,         \
                                             typeDst *aDst, size_t aPitchDst, const Size2D &aSize,                     \
                                             const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSetDevCMask(type)                                                                   \
    InstantiateInvokeSetDevCMask_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeSetDevCMask_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeSetDevCMask_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeSetDevCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSetDevCMask(type)                                                                 \
    InstantiateInvokeSetDevCMask_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeSetDevCMask_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeSetDevCMask_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeSetDevCMask_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeSetDevCMask_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
