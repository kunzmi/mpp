#include "gammaCorr.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/image/forEachPixelPlanar2Kernel.h>
#include <backends/cuda/image/forEachPixelPlanar3Kernel.h>
#include <backends/cuda/image/forEachPixelPlanar4Kernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/colorConversion/color_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/srcFunctor.h>
#include <common/image/functors/srcPlanar2Functor.h>
#include <common/image/functors/srcPlanar3Functor.h>
#include <common/image/functors/srcPlanar4Functor.h>
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
namespace
{
template <typename T> constexpr RoundingMode GetRoundingMode()
{
    return RoundingMode::None;
}
template <RealIntVector T> constexpr RoundingMode GetRoundingMode()
{
    if constexpr (RealSignedVector<T>)
    {
        return RoundingMode::NearestTiesAwayFromZero;
    }
    return RoundingMode::NearestTiesAwayFromZeroPositive;
}
} // namespace

template <typename SrcDstT>
void InvokeGammaCorrBT709Src(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                             float aNormFactor, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using gammaSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaBT709<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammaBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, gammaSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaCorrBT709Src_For(typeSrcIsTypeDst)                                                       \
    template void InvokeGammaCorrBT709Src<typeSrcIsTypeDst>(                                                           \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormFactor, \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaCorrBT709Src(type)                                                           \
    InstantiateInvokeGammaCorrBT709Src_For(Pixel##type##C1);                                                           \
    InstantiateInvokeGammaCorrBT709Src_For(Pixel##type##C2);                                                           \
    InstantiateInvokeGammaCorrBT709Src_For(Pixel##type##C3);                                                           \
    InstantiateInvokeGammaCorrBT709Src_For(Pixel##type##C4);                                                           \
    InstantiateInvokeGammaCorrBT709Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeGammaCorrBT709Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                             const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                             Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                             Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2, float aNormFactor,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using gammaSrc =
        SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaBT709<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammaBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

    InvokeForEachPixelPlanar2KernelDefault<DstT, TupelSize, gammaSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                      aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaCorrBT709P2SrcP2_For(typeSrcIsTypeDst)                                                   \
    template void InvokeGammaCorrBT709Src<typeSrcIsTypeDst>(                                                           \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2, float aNormFactor, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaCorrBT709P2SrcP2(type)                                                       \
    InstantiateInvokeGammaCorrBT709P2SrcP2_For(Pixel##type##C2);
#pragma endregion

template <typename SrcDstT>
void InvokeGammaCorrBT709Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                             const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                             const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                             Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                             Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                             Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormFactor,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using gammaSrc =
        SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaBT709<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammaBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, gammaSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                      aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaCorrBT709P3SrcP3_For(typeSrcIsTypeDst)                                                   \
    template void InvokeGammaCorrBT709Src<typeSrcIsTypeDst>(                                                           \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormFactor, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaCorrBT709P3SrcP3(type)                                                       \
    InstantiateInvokeGammaCorrBT709P3SrcP3_For(Pixel##type##C4A);
#pragma endregion

template <typename SrcDstT>
void InvokeGammaCorrBT709Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                             const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                             const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                             const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                             Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                             Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                             Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                             Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormFactor,
                             const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using gammaSrc =
        SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaBT709<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammaBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, op);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, gammaSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaCorrBT709P4SrcP4_For(typeSrcIsTypeDst)                                                   \
    template void InvokeGammaCorrBT709Src<typeSrcIsTypeDst>(                                                           \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormFactor, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaCorrBT709P4SrcP4(type)                                                       \
    InstantiateInvokeGammaCorrBT709P4SrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeGammaCorrBT709Inplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormFactor, const Size2D &aSize,
                                 const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using gammaInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::GammaBT709<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammaBT709<ComputeT> op(aNormFactor);

    const gammaInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, gammaInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaCorrBT709Inplace_For(typeSrcIsTypeDst)                                                   \
    template void InvokeGammaCorrBT709Inplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,     \
                                                                float aNormFactor, const Size2D &aSize,                \
                                                                const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaCorrBT709Inplace(type)                                                       \
    InstantiateInvokeGammaCorrBT709Inplace_For(Pixel##type##C1);                                                       \
    InstantiateInvokeGammaCorrBT709Inplace_For(Pixel##type##C2);                                                       \
    InstantiateInvokeGammaCorrBT709Inplace_For(Pixel##type##C3);                                                       \
    InstantiateInvokeGammaCorrBT709Inplace_For(Pixel##type##C4);                                                       \
    InstantiateInvokeGammaCorrBT709Inplace_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeGammaInvCorrBT709Src(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                                float aNormFactor, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using gammaSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvBT709<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammaInvBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, gammaSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaInvCorrBT709Src_For(typeSrcIsTypeDst)                                                    \
    template void InvokeGammaInvCorrBT709Src<typeSrcIsTypeDst>(                                                        \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormFactor, \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaInvCorrBT709Src(type)                                                        \
    InstantiateInvokeGammaInvCorrBT709Src_For(Pixel##type##C1);                                                        \
    InstantiateInvokeGammaInvCorrBT709Src_For(Pixel##type##C2);                                                        \
    InstantiateInvokeGammaInvCorrBT709Src_For(Pixel##type##C3);                                                        \
    InstantiateInvokeGammaInvCorrBT709Src_For(Pixel##type##C4);                                                        \
    InstantiateInvokeGammaInvCorrBT709Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeGammaInvCorrBT709Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2, float aNormFactor,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using gammaSrc = SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvBT709<ComputeT>,
                                       GetRoundingMode<DstT>()>;

    const mpp::image::GammaInvBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

    InvokeForEachPixelPlanar2KernelDefault<DstT, TupelSize, gammaSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                      aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaInvCorrBT709P2SrcP2_For(typeSrcIsTypeDst)                                                \
    template void InvokeGammaInvCorrBT709Src<typeSrcIsTypeDst>(                                                        \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2, float aNormFactor, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaInvCorrBT709P2SrcP2(type)                                                    \
    InstantiateInvokeGammaInvCorrBT709P2SrcP2_For(Pixel##type##C2);
#pragma endregion

template <typename SrcDstT>
void InvokeGammaInvCorrBT709Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormFactor,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using gammaSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvBT709<ComputeT>,
                                       GetRoundingMode<DstT>()>;

    const mpp::image::GammaInvBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, gammaSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                      aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaInvCorrBT709P3SrcP3_For(typeSrcIsTypeDst)                                                \
    template void InvokeGammaInvCorrBT709Src<typeSrcIsTypeDst>(                                                        \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormFactor, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaInvCorrBT709P3SrcP3(type)                                                    \
    InstantiateInvokeGammaInvCorrBT709P3SrcP3_For(Pixel##type##C4A);
#pragma endregion

template <typename SrcDstT>
void InvokeGammaInvCorrBT709Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                                Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormFactor,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using gammaSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvBT709<ComputeT>,
                                       GetRoundingMode<DstT>()>;

    const mpp::image::GammaInvBT709<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, op);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, gammaSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaInvCorrBT709P4SrcP4_For(typeSrcIsTypeDst)                                                \
    template void InvokeGammaInvCorrBT709Src<typeSrcIsTypeDst>(                                                        \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormFactor, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaInvCorrBT709P4SrcP4(type)                                                    \
    InstantiateInvokeGammaInvCorrBT709P4SrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeGammaInvCorrBT709Inplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormFactor, const Size2D &aSize,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using gammaInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::GammaInvBT709<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammaInvBT709<ComputeT> op(aNormFactor);

    const gammaInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, gammaInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaInvCorrBT709Inplace_For(typeSrcIsTypeDst)                                                \
    template void InvokeGammaInvCorrBT709Inplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,  \
                                                                   float aNormFactor, const Size2D &aSize,             \
                                                                   const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaInvCorrBT709Inplace(type)                                                    \
    InstantiateInvokeGammaInvCorrBT709Inplace_For(Pixel##type##C1);                                                    \
    InstantiateInvokeGammaInvCorrBT709Inplace_For(Pixel##type##C2);                                                    \
    InstantiateInvokeGammaInvCorrBT709Inplace_For(Pixel##type##C3);                                                    \
    InstantiateInvokeGammaInvCorrBT709Inplace_For(Pixel##type##C4);                                                    \
    InstantiateInvokeGammaInvCorrBT709Inplace_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeGammaCorrsRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormFactor,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using gammaSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammasRGB<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammasRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, gammaSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaCorrsRGBSrc_For(typeSrcIsTypeDst)                                                        \
    template void InvokeGammaCorrsRGBSrc<typeSrcIsTypeDst>(                                                            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormFactor, \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaCorrsRGBSrc(type)                                                            \
    InstantiateInvokeGammaCorrsRGBSrc_For(Pixel##type##C1);                                                            \
    InstantiateInvokeGammaCorrsRGBSrc_For(Pixel##type##C2);                                                            \
    InstantiateInvokeGammaCorrsRGBSrc_For(Pixel##type##C3);                                                            \
    InstantiateInvokeGammaCorrsRGBSrc_For(Pixel##type##C4);                                                            \
    InstantiateInvokeGammaCorrsRGBSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeGammaCorrsRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                            Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2, float aNormFactor,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using gammaSrc =
        SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammasRGB<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammasRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

    InvokeForEachPixelPlanar2KernelDefault<DstT, TupelSize, gammaSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                      aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaCorrsRGBP2SrcP2_For(typeSrcIsTypeDst)                                                    \
    template void InvokeGammaCorrsRGBSrc<typeSrcIsTypeDst>(                                                            \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2, float aNormFactor, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaCorrsRGBP2SrcP2(type)                                                        \
    InstantiateInvokeGammaCorrsRGBP2SrcP2_For(Pixel##type##C2);
#pragma endregion

template <typename SrcDstT>
void InvokeGammaCorrsRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                            Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                            Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormFactor,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using gammaSrc =
        SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammasRGB<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammasRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, gammaSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                      aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaCorrsRGBP3SrcP3_For(typeSrcIsTypeDst)                                                    \
    template void InvokeGammaCorrsRGBSrc<typeSrcIsTypeDst>(                                                            \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormFactor, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaCorrsRGBP3SrcP3(type)                                                        \
    InstantiateInvokeGammaCorrsRGBP3SrcP3_For(Pixel##type##C4A);
#pragma endregion

template <typename SrcDstT>
void InvokeGammaCorrsRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                            Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                            Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                            Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormFactor,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using gammaSrc =
        SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammasRGB<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammasRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, op);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, gammaSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaCorrsRGBP4SrcP4_For(typeSrcIsTypeDst)                                                    \
    template void InvokeGammaCorrsRGBSrc<typeSrcIsTypeDst>(                                                            \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormFactor, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaCorrsRGBP4SrcP4(type)                                                        \
    InstantiateInvokeGammaCorrsRGBP4SrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeGammaCorrsRGBInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormFactor, const Size2D &aSize,
                                const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using gammaInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::GammasRGB<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammasRGB<ComputeT> op(aNormFactor);

    const gammaInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, gammaInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaCorrsRGBInplace_For(typeSrcIsTypeDst)                                                    \
    template void InvokeGammaCorrsRGBInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,      \
                                                               float aNormFactor, const Size2D &aSize,                 \
                                                               const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaCorrsRGBInplace(type)                                                        \
    InstantiateInvokeGammaCorrsRGBInplace_For(Pixel##type##C1);                                                        \
    InstantiateInvokeGammaCorrsRGBInplace_For(Pixel##type##C2);                                                        \
    InstantiateInvokeGammaCorrsRGBInplace_For(Pixel##type##C3);                                                        \
    InstantiateInvokeGammaCorrsRGBInplace_For(Pixel##type##C4);                                                        \
    InstantiateInvokeGammaCorrsRGBInplace_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeGammaInvCorrsRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                               float aNormFactor, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using gammaSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvsRGB<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammaInvsRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, gammaSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaInvCorrsRGBSrc_For(typeSrcIsTypeDst)                                                     \
    template void InvokeGammaInvCorrsRGBSrc<typeSrcIsTypeDst>(                                                         \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormFactor, \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaInvCorrsRGBSrc(type)                                                         \
    InstantiateInvokeGammaInvCorrsRGBSrc_For(Pixel##type##C1);                                                         \
    InstantiateInvokeGammaInvCorrsRGBSrc_For(Pixel##type##C2);                                                         \
    InstantiateInvokeGammaInvCorrsRGBSrc_For(Pixel##type##C3);                                                         \
    InstantiateInvokeGammaInvCorrsRGBSrc_For(Pixel##type##C4);                                                         \
    InstantiateInvokeGammaInvCorrsRGBSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeGammaInvCorrsRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                               const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                               Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                               Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2, float aNormFactor,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using gammaSrc =
        SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvsRGB<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammaInvsRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

    InvokeForEachPixelPlanar2KernelDefault<DstT, TupelSize, gammaSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                      aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaInvCorrsRGBP2SrcP2_For(typeSrcIsTypeDst)                                                 \
    template void InvokeGammaInvCorrsRGBSrc<typeSrcIsTypeDst>(                                                         \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2, float aNormFactor, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaInvCorrsRGBP2SrcP2(type)                                                     \
    InstantiateInvokeGammaInvCorrsRGBP2SrcP2_For(Pixel##type##C2);
#pragma endregion

template <typename SrcDstT>
void InvokeGammaInvCorrsRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                               const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                               const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                               Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                               Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                               Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormFactor,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using gammaSrc =
        SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvsRGB<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammaInvsRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, gammaSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                      aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaInvCorrsRGBP3SrcP3_For(typeSrcIsTypeDst)                                                 \
    template void InvokeGammaInvCorrsRGBSrc<typeSrcIsTypeDst>(                                                         \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormFactor, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaInvCorrsRGBP3SrcP3(type)                                                     \
    InstantiateInvokeGammaInvCorrsRGBP3SrcP3_For(Pixel##type##C4A);
#pragma endregion

template <typename SrcDstT>
void InvokeGammaInvCorrsRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                               const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                               const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                               const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                               Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                               Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                               Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                               Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormFactor,
                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using gammaSrc =
        SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::GammaInvsRGB<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammaInvsRGB<ComputeT> op(aNormFactor);

    const gammaSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, op);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, gammaSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaInvCorrsRGBP4SrcP4_For(typeSrcIsTypeDst)                                                 \
    template void InvokeGammaInvCorrsRGBSrc<typeSrcIsTypeDst>(                                                         \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormFactor, const Size2D &aSize,  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaInvCorrsRGBP4SrcP4(type)                                                     \
    InstantiateInvokeGammaInvCorrsRGBP4SrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeGammaInvCorrsRGBInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormFactor, const Size2D &aSize,
                                   const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using gammaInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::GammaInvsRGB<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::GammaInvsRGB<ComputeT> op(aNormFactor);

    const gammaInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, gammaInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeGammaInvCorrsRGBInplace_For(typeSrcIsTypeDst)                                                 \
    template void InvokeGammaInvCorrsRGBInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,   \
                                                                  float aNormFactor, const Size2D &aSize,              \
                                                                  const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeGammaInvCorrsRGBInplace(type)                                                     \
    InstantiateInvokeGammaInvCorrsRGBInplace_For(Pixel##type##C1);                                                     \
    InstantiateInvokeGammaInvCorrsRGBInplace_For(Pixel##type##C2);                                                     \
    InstantiateInvokeGammaInvCorrsRGBInplace_For(Pixel##type##C3);                                                     \
    InstantiateInvokeGammaInvCorrsRGBInplace_For(Pixel##type##C4);                                                     \
    InstantiateInvokeGammaInvCorrsRGBInplace_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
