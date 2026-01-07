#include "hsv.h"
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

#pragma region RGBtoHSV

template <typename SrcDstT>
void InvokeRGBtoHSVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHSV<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHSV<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHSVSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeRGBtoHSVSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHSVSrc(type)                                                                 \
    InstantiateInvokeRGBtoHSVSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeRGBtoHSVSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoHSVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHSV<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHSV<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHSVP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoHSVSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHSVP3Src(type)                                                               \
    InstantiateInvokeRGBtoHSVP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeRGBtoHSVP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoHSVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHSV<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHSV<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHSV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, hsvSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHSVP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoHSVSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHSVP4Src(type) InstantiateInvokeRGBtoHSVP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoHSVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHSV<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHSV<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHSVP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeRGBtoHSVSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHSVP3SrcP3(type) InstantiateInvokeRGBtoHSVP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoHSVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoHSV<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHSV<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHSVSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoHSVSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHSVSrcP3(type)                                                               \
    InstantiateInvokeRGBtoHSVSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeRGBtoHSVSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoHSVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using hsvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHSV<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHSV<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoHSV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHSVSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoHSVSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHSVSrcP4(type) InstantiateInvokeRGBtoHSVSrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoHSVInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hsvInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::RGBtoHSV<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoHSV<ComputeT, doNormalize> op(aNormVal);

    const hsvInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoHSVInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeRGBtoHSVInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoHSVInplace(type)                                                             \
    InstantiateInvokeRGBtoHSVInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeRGBtoHSVInplace_For(Pixel##type##C4A);

#pragma endregion
#pragma endregion

#pragma region HSVtoRGB

template <typename SrcDstT>
void InvokeHSVtoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoRGB<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoRGB<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoRGBSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeHSVtoRGBSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoRGBSrc(type)                                                                 \
    InstantiateInvokeHSVtoRGBSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeHSVtoRGBSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeHSVtoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoRGB<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoRGB<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoRGBP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHSVtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoRGBP3Src(type)                                                               \
    InstantiateInvokeHSVtoRGBP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeHSVtoRGBP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeHSVtoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoRGB<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoRGB<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, hsvSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoRGBP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHSVtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoRGBP4Src(type) InstantiateInvokeHSVtoRGBP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeHSVtoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoRGB<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoRGB<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoRGBP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeHSVtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoRGBP3SrcP3(type) InstantiateInvokeHSVtoRGBP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeHSVtoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoRGB<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoRGB<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoRGBSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHSVtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoRGBSrcP3(type)                                                               \
    InstantiateInvokeHSVtoRGBSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeHSVtoRGBSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeHSVtoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using hsvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoRGB<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoRGB<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoRGBSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHSVtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoRGBSrcP4(type) InstantiateInvokeHSVtoRGBSrcP4_For(Pixel##type##C4);

#pragma endregion
template <typename SrcDstT>
void InvokeHSVtoRGBInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hsvInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::HSVtoRGB<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoRGB<ComputeT, doNormalize> op(aNormVal);

    const hsvInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoRGBInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeHSVtoRGBInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoRGBInplace(type)                                                             \
    InstantiateInvokeHSVtoRGBInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeHSVtoRGBInplace_For(Pixel##type##C4A);

#pragma endregion

#pragma endregion

#pragma region BGRtoHSV

template <typename SrcDstT>
void InvokeBGRtoHSVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHSV<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHSV<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHSVSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeBGRtoHSVSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHSVSrc(type)                                                                 \
    InstantiateInvokeBGRtoHSVSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeBGRtoHSVSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoHSVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHSV<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHSV<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHSVP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoHSVSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHSVP3Src(type)                                                               \
    InstantiateInvokeBGRtoHSVP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeBGRtoHSVP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoHSVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHSV<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHSV<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHSV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, hsvSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHSVP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoHSVSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHSVP4Src(type) InstantiateInvokeBGRtoHSVP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoHSVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHSV<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHSV<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHSVP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeBGRtoHSVSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHSVP3SrcP3(type) InstantiateInvokeBGRtoHSVP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoHSVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoHSV<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHSV<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHSVSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoHSVSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHSVSrcP3(type)                                                               \
    InstantiateInvokeBGRtoHSVSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeBGRtoHSVSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoHSVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using hsvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHSV<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHSV<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoHSV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHSVSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoHSVSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHSVSrcP4(type) InstantiateInvokeBGRtoHSVSrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoHSVInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hsvInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::BGRtoHSV<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoHSV<ComputeT, doNormalize> op(aNormVal);

    const hsvInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoHSVInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeBGRtoHSVInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoHSVInplace(type)                                                             \
    InstantiateInvokeBGRtoHSVInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeBGRtoHSVInplace_For(Pixel##type##C4A);

#pragma endregion
#pragma endregion

#pragma region HSVtoBGR

template <typename SrcDstT>
void InvokeHSVtoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoBGR<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoBGR<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoBGRSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeHSVtoBGRSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoBGRSrc(type)                                                                 \
    InstantiateInvokeHSVtoBGRSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeHSVtoBGRSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeHSVtoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoBGR<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoBGR<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoBGRP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHSVtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoBGRP3Src(type)                                                               \
    InstantiateInvokeHSVtoBGRP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeHSVtoBGRP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeHSVtoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using hsvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoBGR<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoBGR<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, hsvSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoBGRP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHSVtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoBGRP4Src(type) InstantiateInvokeHSVtoBGRP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeHSVtoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoBGR<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoBGR<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoBGRP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeHSVtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoBGRP3SrcP3(type) InstantiateInvokeHSVtoBGRP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeHSVtoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using hsvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::HSVtoBGR<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoBGR<ComputeT, doNormalize> op(aNormVal);

    const hsvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoBGRSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHSVtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoBGRSrcP3(type)                                                               \
    InstantiateInvokeHSVtoBGRSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeHSVtoBGRSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeHSVtoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using hsvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoBGR<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoBGR<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::HSVtoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const hsvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoBGRSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeHSVtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoBGRSrcP4(type) InstantiateInvokeHSVtoBGRSrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeHSVtoBGRInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using hsvInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::HSVtoBGR<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::HSVtoBGR<ComputeT, doNormalize> op(aNormVal);

    const hsvInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, hsvInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeHSVtoBGRInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeHSVtoBGRInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeHSVtoBGRInplace(type)                                                             \
    InstantiateInvokeHSVtoBGRInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeHSVtoBGRInplace_For(Pixel##type##C4A);

#pragma endregion
#pragma endregion
} // namespace mpp::image::cuda
