#include "hls.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/image/forEachPixelPlanar3Kernel.h>
#include <backends/cuda/image/forEachPixelPlanar4Kernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/colorConversion/color_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/srcFunctor.h>
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
    return RoundingMode::NearestTiesAwayFromZeroPositive;
}
} // namespace

#pragma region RGBtoHLS

template <typename SrcDstT>
void InvokeRGBtoHLSSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHLS<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHLS<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHLSSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeRGBtoHLSSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHLSSrc(type)                                                                 \
    InstantiateInvokeRGBtoHLSSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeRGBtoHLSSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoHLSSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHLS<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHLS<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHLSP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoHLSSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHLSP3Src(type)                                                               \
    InstantiateInvokeRGBtoHLSP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeRGBtoHLSP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoHLSSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHLS<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHLS<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHLS<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, hlsSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHLSP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoHLSSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHLSP4Src(type) InstantiateInvokeRGBtoHLSP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoHLSSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                       Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHLS<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHLS<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHLSP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeRGBtoHLSSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHLSP3SrcP3(type) InstantiateInvokeRGBtoHLSP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoHLSSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHLS<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHLS<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHLSSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoHLSSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHLSSrcP3(type)                                                               \
    InstantiateInvokeRGBtoHLSSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeRGBtoHLSSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoHLSSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHLS<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHLS<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHLS<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHLSSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoHLSSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHLSSrcP4(type) InstantiateInvokeRGBtoHLSSrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoHLSInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::RGBtoHLS<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHLS<ComputeT, doNormalize> op(aNormVal);

    const hlsInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHLSInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeRGBtoHLSInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHLSInplace(type)                                                             \
    InstantiateInvokeRGBtoHLSInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeRGBtoHLSInplace_For(Pixel##type##C4A);

#pragma endregion
#pragma endregion

#pragma region HLStoRGB

template <typename SrcDstT>
void InvokeHLStoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoRGB<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::HLStoRGB<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoRGBSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeHLStoRGBSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoRGBSrc(type)                                                                 \
    InstantiateInvokeHLStoRGBSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeHLStoRGBSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeHLStoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoRGB<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::HLStoRGB<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoRGBP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHLStoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoRGBP3Src(type)                                                               \
    InstantiateInvokeHLStoRGBP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeHLStoRGBP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeHLStoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::HLStoRGB<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::HLStoRGB<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HLStoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, hlsSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoRGBP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHLStoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoRGBP4Src(type) InstantiateInvokeHLStoRGBP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeHLStoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                       Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoRGB<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::HLStoRGB<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoRGBP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeHLStoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoRGBP3SrcP3(type) InstantiateInvokeHLStoRGBP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeHLStoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoRGB<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::HLStoRGB<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoRGBSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHLStoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoRGBSrcP3(type)                                                               \
    InstantiateInvokeHLStoRGBSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeHLStoRGBSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeHLStoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::HLStoRGB<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::HLStoRGB<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HLStoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoRGBSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHLStoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoRGBSrcP4(type) InstantiateInvokeHLStoRGBSrcP4_For(Pixel##type##C4);

#pragma endregion
template <typename SrcDstT>
void InvokeHLStoRGBInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::HLStoRGB<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::HLStoRGB<ComputeT, doNormalize> op(aNormVal);

    const hlsInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoRGBInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeHLStoRGBInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoRGBInplace(type)                                                             \
    InstantiateInvokeHLStoRGBInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeHLStoRGBInplace_For(Pixel##type##C4A);

#pragma endregion

#pragma endregion

#pragma region BGRtoHLS

template <typename SrcDstT>
void InvokeBGRtoHLSSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHLS<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHLS<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHLSSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeBGRtoHLSSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHLSSrc(type)                                                                 \
    InstantiateInvokeBGRtoHLSSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeBGRtoHLSSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoHLSSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHLS<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHLS<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHLSP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoHLSSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHLSP3Src(type)                                                               \
    InstantiateInvokeBGRtoHLSP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeBGRtoHLSP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoHLSSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHLS<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHLS<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHLS<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, hlsSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHLSP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoHLSSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHLSP4Src(type) InstantiateInvokeBGRtoHLSP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoHLSSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                       Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHLS<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHLS<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHLSP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeBGRtoHLSSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHLSP3SrcP3(type) InstantiateInvokeBGRtoHLSP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoHLSSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHLS<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHLS<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHLSSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoHLSSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHLSSrcP3(type)                                                               \
    InstantiateInvokeBGRtoHLSSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeBGRtoHLSSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoHLSSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHLS<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHLS<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHLS<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHLSSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoHLSSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHLSSrcP4(type) InstantiateInvokeBGRtoHLSSrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoHLSInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::BGRtoHLS<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHLS<ComputeT, doNormalize> op(aNormVal);

    const hlsInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHLSInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeBGRtoHLSInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHLSInplace(type)                                                             \
    InstantiateInvokeBGRtoHLSInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeBGRtoHLSInplace_For(Pixel##type##C4A);

#pragma endregion
#pragma endregion

#pragma region HLStoBGR

template <typename SrcDstT>
void InvokeHLStoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoBGR<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::HLStoBGR<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoBGRSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeHLStoBGRSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoBGRSrc(type)                                                                 \
    InstantiateInvokeHLStoBGRSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeHLStoBGRSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeHLStoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoBGR<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::HLStoBGR<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoBGRP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHLStoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoBGRP3Src(type)                                                               \
    InstantiateInvokeHLStoBGRP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeHLStoBGRP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeHLStoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                       size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::HLStoBGR<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::HLStoBGR<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HLStoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, hlsSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoBGRP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHLStoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoBGRP4Src(type) InstantiateInvokeHLStoBGRP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeHLStoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                       Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                       Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoBGR<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::HLStoBGR<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoBGRP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeHLStoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoBGRP3SrcP3(type) InstantiateInvokeHLStoBGRP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeHLStoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HLStoBGR<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::HLStoBGR<ComputeT, doNormalize> op(aNormVal);

    const hlsSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoBGRSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHLStoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoBGRSrcP3(type)                                                               \
    InstantiateInvokeHLStoBGRSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeHLStoBGRSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeHLStoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                       const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4, SrcDstT *aDst1,
                       size_t aPitchDst1, float aNormVal, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::HLStoBGR<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::HLStoBGR<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HLStoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hlsSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoBGRSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHLStoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoBGRSrcP4(type) InstantiateInvokeHLStoBGRSrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeHLStoBGRInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hlsInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::HLStoBGR<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::HLStoBGR<ComputeT, doNormalize> op(aNormVal);

    const hlsInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hlsInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHLStoBGRInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeHLStoBGRInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHLStoBGRInplace(type)                                                             \
    InstantiateInvokeHLStoBGRInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeHLStoBGRInplace_For(Pixel##type##C4A);

#pragma endregion
#pragma endregion
} // namespace mpp::image::cuda
