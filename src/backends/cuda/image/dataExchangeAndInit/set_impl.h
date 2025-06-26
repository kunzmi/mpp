#if MPP_ENABLE_CUDA_BACKEND

#include "set.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
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
void InvokeSetC(const DstT &aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using setC = ConstantFunctor<TupelSize, DstT>;

        const setC functor(aConst);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, setC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define InstantiateInvokeSetC_For(typeDst)                                                                             \
    template void InvokeSetC<typeDst>(const typeDst &aConst, typeDst *aDst, size_t aPitchDst, const Size2D &aSize,     \
                                      const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSetC(type)                                                                          \
    InstantiateInvokeSetC_For(Pixel##type##C1);                                                                        \
    InstantiateInvokeSetC_For(Pixel##type##C2);                                                                        \
    InstantiateInvokeSetC_For(Pixel##type##C3);                                                                        \
    InstantiateInvokeSetC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSetC(type)                                                                        \
    InstantiateInvokeSetC_For(Pixel##type##C1);                                                                        \
    InstantiateInvokeSetC_For(Pixel##type##C2);                                                                        \
    InstantiateInvokeSetC_For(Pixel##type##C3);                                                                        \
    InstantiateInvokeSetC_For(Pixel##type##C4);                                                                        \
    InstantiateInvokeSetC_For(Pixel##type##C4A);

#pragma endregion

template <typename DstT>
void InvokeSetDevC(const DstT *aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using setDevC = DevConstantFunctor<TupelSize, DstT>;

        const setDevC functor(aConst);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, setDevC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define InstantiateInvokeSetDevC_For(typeDst)                                                                          \
    template void InvokeSetDevC<typeDst>(const typeDst *aConst, typeDst *aDst, size_t aPitchDst, const Size2D &aSize,  \
                                         const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSetDevC(type)                                                                       \
    InstantiateInvokeSetDevC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeSetDevC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeSetDevC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeSetDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSetDevC(type)                                                                     \
    InstantiateInvokeSetDevC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeSetDevC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeSetDevC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeSetDevC_For(Pixel##type##C4);                                                                     \
    InstantiateInvokeSetDevC_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
