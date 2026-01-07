#include "luv.h"
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

#pragma region RGBtoLUV

template <typename SrcDstT>
void InvokeRGBtoLUVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLUV<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLUV<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLUVSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeRGBtoLUVSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLUVSrc(type)                                                                 \
    InstantiateInvokeRGBtoLUVSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeRGBtoLUVSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoLUVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLUV<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLUV<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLUVP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoLUVSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLUVP3Src(type)                                                               \
    InstantiateInvokeRGBtoLUVP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeRGBtoLUVP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoLUVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLUV<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLUV<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLUV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, luvSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLUVP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoLUVSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLUVP4Src(type) InstantiateInvokeRGBtoLUVP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoLUVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLUV<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLUV<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLUVP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeRGBtoLUVSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLUVP3SrcP3(type) InstantiateInvokeRGBtoLUVP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoLUVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLUV<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLUV<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLUVSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoLUVSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLUVSrcP3(type)                                                               \
    InstantiateInvokeRGBtoLUVSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeRGBtoLUVSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoLUVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using luvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLUV<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLUV<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLUV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLUVSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoLUVSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLUVSrcP4(type) InstantiateInvokeRGBtoLUVSrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoLUVInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using luvInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::RGBtoLUV<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLUV<ComputeT, doNormalize> op(aNormVal);

    const luvInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLUVInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeRGBtoLUVInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLUVInplace(type)                                                             \
    InstantiateInvokeRGBtoLUVInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeRGBtoLUVInplace_For(Pixel##type##C4A);

#pragma endregion
#pragma endregion

#pragma region LUVtoRGB

template <typename SrcDstT>
void InvokeLUVtoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoRGB<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoRGB<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoRGBSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeLUVtoRGBSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoRGBSrc(type)                                                                 \
    InstantiateInvokeLUVtoRGBSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeLUVtoRGBSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLUVtoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoRGB<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoRGB<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoRGBP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLUVtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoRGBP3Src(type)                                                               \
    InstantiateInvokeLUVtoRGBP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeLUVtoRGBP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLUVtoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoRGB<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoRGB<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, luvSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoRGBP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLUVtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoRGBP4Src(type) InstantiateInvokeLUVtoRGBP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeLUVtoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoRGB<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoRGB<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoRGBP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeLUVtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoRGBP3SrcP3(type) InstantiateInvokeLUVtoRGBP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeLUVtoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoRGB<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoRGB<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoRGBSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLUVtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoRGBSrcP3(type)                                                               \
    InstantiateInvokeLUVtoRGBSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeLUVtoRGBSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLUVtoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using luvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoRGB<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoRGB<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoRGBSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLUVtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoRGBSrcP4(type) InstantiateInvokeLUVtoRGBSrcP4_For(Pixel##type##C4);

#pragma endregion
template <typename SrcDstT>
void InvokeLUVtoRGBInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using luvInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LUVtoRGB<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoRGB<ComputeT, doNormalize> op(aNormVal);

    const luvInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoRGBInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeLUVtoRGBInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoRGBInplace(type)                                                             \
    InstantiateInvokeLUVtoRGBInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeLUVtoRGBInplace_For(Pixel##type##C4A);

#pragma endregion

#pragma endregion

#pragma region BGRtoLUV

template <typename SrcDstT>
void InvokeBGRtoLUVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLUV<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLUV<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLUVSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeBGRtoLUVSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLUVSrc(type)                                                                 \
    InstantiateInvokeBGRtoLUVSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeBGRtoLUVSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoLUVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLUV<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLUV<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLUVP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoLUVSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLUVP3Src(type)                                                               \
    InstantiateInvokeBGRtoLUVP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeBGRtoLUVP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoLUVSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLUV<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLUV<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLUV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, luvSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLUVP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoLUVSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLUVP4Src(type) InstantiateInvokeBGRtoLUVP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoLUVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLUV<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLUV<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLUVP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeBGRtoLUVSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLUVP3SrcP3(type) InstantiateInvokeBGRtoLUVP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoLUVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLUV<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLUV<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLUVSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoLUVSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLUVSrcP3(type)                                                               \
    InstantiateInvokeBGRtoLUVSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeBGRtoLUVSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoLUVSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using luvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLUV<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLUV<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLUV<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLUVSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoLUVSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLUVSrcP4(type) InstantiateInvokeBGRtoLUVSrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoLUVInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using luvInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::BGRtoLUV<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLUV<ComputeT, doNormalize> op(aNormVal);

    const luvInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLUVInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeBGRtoLUVInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLUVInplace(type)                                                             \
    InstantiateInvokeBGRtoLUVInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeBGRtoLUVInplace_For(Pixel##type##C4A);

#pragma endregion
#pragma endregion

#pragma region LUVtoBGR

template <typename SrcDstT>
void InvokeLUVtoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoBGR<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoBGR<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoBGRSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeLUVtoBGRSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoBGRSrc(type)                                                                 \
    InstantiateInvokeLUVtoBGRSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeLUVtoBGRSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLUVtoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoBGR<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoBGR<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoBGRP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLUVtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoBGRP3Src(type)                                                               \
    InstantiateInvokeLUVtoBGRP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeLUVtoBGRP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLUVtoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using luvSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoBGR<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoBGR<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, luvSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoBGRP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLUVtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoBGRP4Src(type) InstantiateInvokeLUVtoBGRP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeLUVtoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoBGR<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoBGR<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoBGRP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeLUVtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoBGRP3SrcP3(type) InstantiateInvokeLUVtoBGRP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeLUVtoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using luvSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LUVtoBGR<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoBGR<ComputeT, doNormalize> op(aNormVal);

    const luvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoBGRSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLUVtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoBGRSrcP3(type)                                                               \
    InstantiateInvokeLUVtoBGRSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeLUVtoBGRSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLUVtoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using luvSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoBGR<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoBGR<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LUVtoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const luvSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoBGRSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLUVtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoBGRSrcP4(type) InstantiateInvokeLUVtoBGRSrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeLUVtoBGRInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using luvInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LUVtoBGR<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::LUVtoBGR<ComputeT, doNormalize> op(aNormVal);

    const luvInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, luvInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLUVtoBGRInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeLUVtoBGRInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLUVtoBGRInplace(type)                                                             \
    InstantiateInvokeLUVtoBGRInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeLUVtoBGRInplace_For(Pixel##type##C4A);

#pragma endregion
#pragma endregion
} // namespace mpp::image::cuda
