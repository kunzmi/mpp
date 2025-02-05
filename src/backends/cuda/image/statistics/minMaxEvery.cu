#if OPP_ENABLE_CUDA_BACKEND

#include "minMaxEvery.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceSrcFunctor.h>
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
void InvokeMinEverySrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using minEverySrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Min<ComputeT>, RoundingMode::None>;

        const opp::Min<ComputeT> op;

        const minEverySrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, minEverySrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeMinEverySrcSrc<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                          \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst *aSrc2, size_t aPitchSrc2,            \
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
void InvokeMinEveryInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using minEveryInplaceSrc =
            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Min<ComputeT>, RoundingMode::None>;

        const opp::Min<ComputeT> op;

        const minEveryInplaceSrc functor(aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, minEveryInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                             functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeMinEveryInplaceSrc<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                      \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst *aSrc2, size_t aPitchSrc2,             \
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

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMaxEverySrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using maxEverySrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Max<ComputeT>, RoundingMode::None>;

        const opp::Max<ComputeT> op;

        const maxEverySrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, maxEverySrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeMaxEverySrcSrc<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                          \
        const typeSrcIstypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIstypeDst *aSrc2, size_t aPitchSrc2,            \
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
void InvokeMaxEveryInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using maxEveryInplaceSrc =
            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Max<ComputeT>, RoundingMode::None>;

        const opp::Max<ComputeT> op;

        const maxEveryInplaceSrc functor(aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, maxEveryInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                             functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrcIstypeDst)                                                                              \
    template void InvokeMaxEveryInplaceSrc<typeSrcIstypeDst, typeSrcIstypeDst, typeSrcIstypeDst>(                      \
        typeSrcIstypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIstypeDst *aSrc2, size_t aPitchSrc2,             \
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
