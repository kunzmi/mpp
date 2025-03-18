#if OPP_ENABLE_CUDA_BACKEND

#include "compareEqEps.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
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
void InvokeCompareEqEpsSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                              size_t aPitchDst, complex_basetype_t<remove_vector_t<SrcT>> aEpsilon, const Size2D &aSize,
                              const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = vector_size_v<SrcT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::EqEps<ComputeT>, RoundingMode::None,
                                            voidType, voidType, true>;
        const opp::EqEps<ComputeT> op(aEpsilon);
        const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeCompareEqEpsSrcSrc<typeSrc, typeSrc, Vector1<byte>>(                                           \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2, Vector1<byte> *aDst,         \
        size_t aPitchDst, complex_basetype_t<remove_vector_t<typeSrc>> aEpsilon, const Size2D &aSize,                  \
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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeCompareEqEpsSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                            complex_basetype_t<remove_vector_t<SrcT>> aEpsilon, const Size2D &aSize,
                            const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = vector_size_v<SrcT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(DstT)>::value;
        using compareSrcC          = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::EqEps<ComputeT>,
                                                        RoundingMode::None, voidType, voidType, true>;
        const opp::EqEps<ComputeT> op(aEpsilon);
        const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeCompareEqEpsSrcC<typeSrc, typeSrc, Vector1<byte>>(                                             \
        const typeSrc *aSrc, size_t aPitchSrc, const typeSrc &aConst, Vector1<byte> *aDst, size_t aPitchDst,           \
        complex_basetype_t<remove_vector_t<typeSrc>> aEpsilon, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeCompareEqEpsSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                               complex_basetype_t<remove_vector_t<SrcT>> aEpsilon, const Size2D &aSize,
                               const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = vector_size_v<SrcT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(DstT)>::value;
        using compareSrcC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::EqEps<ComputeT>, RoundingMode::None, true>;
        const opp::EqEps<ComputeT> op(aEpsilon);
        const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeCompareEqEpsSrcDevC<typeSrc, typeSrc, Vector1<byte>>(                                          \
        const typeSrc *aSrc, size_t aPitchSrc, const typeSrc *aConst, Vector1<byte> *aDst, size_t aPitchDst,           \
        complex_basetype_t<remove_vector_t<typeSrc>> aEpsilon, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
