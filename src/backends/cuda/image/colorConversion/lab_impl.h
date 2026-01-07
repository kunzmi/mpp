#include "lab.h"
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

#pragma region RGBtoLab

template <typename SrcDstT>
void InvokeRGBtoLabSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLab<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLab<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLabSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeRGBtoLabSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLabSrc(type)                                                                 \
    InstantiateInvokeRGBtoLabSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeRGBtoLabSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoLabSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLab<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLab<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLabP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoLabSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLabP3Src(type)                                                               \
    InstantiateInvokeRGBtoLabP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeRGBtoLabP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoLabSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLab<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLab<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLab<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, labSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLabP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoLabSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLabP4Src(type) InstantiateInvokeRGBtoLabP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoLabSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLab<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLab<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLabP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeRGBtoLabSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLabP3SrcP3(type) InstantiateInvokeRGBtoLabP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoLabSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::RGBtoLab<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLab<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLabSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoLabSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLabSrcP3(type)                                                               \
    InstantiateInvokeRGBtoLabSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeRGBtoLabSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoLabSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using labSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLab<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLab<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::RGBtoLab<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLabSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeRGBtoLabSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLabSrcP4(type) InstantiateInvokeRGBtoLabSrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeRGBtoLabInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using labInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::RGBtoLab<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::RGBtoLab<ComputeT, doNormalize> op(aNormVal);

    const labInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeRGBtoLabInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeRGBtoLabInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRGBtoLabInplace(type)                                                             \
    InstantiateInvokeRGBtoLabInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeRGBtoLabInplace_For(Pixel##type##C4A);

#pragma endregion
#pragma endregion

#pragma region LabtoRGB

template <typename SrcDstT>
void InvokeLabtoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoRGB<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::LabtoRGB<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoRGBSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeLabtoRGBSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoRGBSrc(type)                                                                 \
    InstantiateInvokeLabtoRGBSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeLabtoRGBSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLabtoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoRGB<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::LabtoRGB<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoRGBP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLabtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoRGBP3Src(type)                                                               \
    InstantiateInvokeLabtoRGBP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeLabtoRGBP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLabtoRGBSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::LabtoRGB<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::LabtoRGB<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LabtoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, labSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoRGBP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLabtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoRGBP4Src(type) InstantiateInvokeLabtoRGBP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeLabtoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoRGB<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::LabtoRGB<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoRGBP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeLabtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoRGBP3SrcP3(type) InstantiateInvokeLabtoRGBP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeLabtoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoRGB<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::LabtoRGB<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoRGBSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLabtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoRGBSrcP3(type)                                                               \
    InstantiateInvokeLabtoRGBSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeLabtoRGBSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLabtoRGBSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using labSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::LabtoRGB<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::LabtoRGB<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LabtoRGB<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoRGBSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLabtoRGBSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoRGBSrcP4(type) InstantiateInvokeLabtoRGBSrcP4_For(Pixel##type##C4);

#pragma endregion
template <typename SrcDstT>
void InvokeLabtoRGBInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using labInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LabtoRGB<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::LabtoRGB<ComputeT, doNormalize> op(aNormVal);

    const labInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoRGBInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeLabtoRGBInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoRGBInplace(type)                                                             \
    InstantiateInvokeLabtoRGBInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeLabtoRGBInplace_For(Pixel##type##C4A);

#pragma endregion

#pragma endregion

#pragma region BGRtoLab

template <typename SrcDstT>
void InvokeBGRtoLabSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLab<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLab<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLabSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeBGRtoLabSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLabSrc(type)                                                                 \
    InstantiateInvokeBGRtoLabSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeBGRtoLabSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoLabSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLab<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLab<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLabP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoLabSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLabP3Src(type)                                                               \
    InstantiateInvokeBGRtoLabP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeBGRtoLabP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoLabSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLab<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLab<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLab<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, labSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLabP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoLabSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLabP4Src(type) InstantiateInvokeBGRtoLabP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoLabSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLab<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLab<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLabP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeBGRtoLabSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLabP3SrcP3(type) InstantiateInvokeBGRtoLabP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoLabSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::BGRtoLab<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLab<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLabSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoLabSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLabSrcP3(type)                                                               \
    InstantiateInvokeBGRtoLabSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeBGRtoLabSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoLabSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using labSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLab<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLab<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::BGRtoLab<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLabSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeBGRtoLabSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLabSrcP4(type) InstantiateInvokeBGRtoLabSrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeBGRtoLabInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using labInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::BGRtoLab<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::BGRtoLab<ComputeT, doNormalize> op(aNormVal);

    const labInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeBGRtoLabInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeBGRtoLabInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeBGRtoLabInplace(type)                                                             \
    InstantiateInvokeBGRtoLabInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeBGRtoLabInplace_For(Pixel##type##C4A);

#pragma endregion
#pragma endregion

#pragma region LabtoBGR

template <typename SrcDstT>
void InvokeLabtoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst, float aNormVal,
                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoBGR<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::LabtoBGR<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoBGRSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeLabtoBGRSrc<typeSrcIsTypeDst>(const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                \
                                                      typeSrcIsTypeDst *aDst, size_t aPitchDst, float aNormVal,        \
                                                      const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoBGRSrc(type)                                                                 \
    InstantiateInvokeLabtoBGRSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeLabtoBGRSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLabtoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoBGR<ComputeT, doNormalize>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::LabtoBGR<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoBGRP3Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLabtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoBGRP3Src(type)                                                               \
    InstantiateInvokeLabtoBGRP3Src_For(Pixel##type##C3);                                                               \
    InstantiateInvokeLabtoBGRP3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLabtoBGRSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
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

    using labSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                              mpp::image::SetAlpha<ComputeT, mpp::image::LabtoBGR<Pixel32fC4A, doNormalize>>,
                              GetRoundingMode<DstT>()>;

    const mpp::image::LabtoBGR<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LabtoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, labSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoBGRP4Src_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLabtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoBGRP4Src(type) InstantiateInvokeLabtoBGRP4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeLabtoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoBGR<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::LabtoBGR<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                    aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoBGRP3SrcP3_For(typeSrcIsTypeDst)                                                         \
    template void InvokeLabtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, float aNormVal, const Size2D &aSize,     \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoBGRP3SrcP3(type) InstantiateInvokeLabtoBGRP3SrcP3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeLabtoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using labSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::LabtoBGR<ComputeT, doNormalize>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::LabtoBGR<ComputeT, doNormalize> op(aNormVal);

    const labSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoBGRSrcP3_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLabtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoBGRSrcP3(type)                                                               \
    InstantiateInvokeLabtoBGRSrcP3_For(Pixel##type##C3);                                                               \
    InstantiateInvokeLabtoBGRSrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeLabtoBGRSrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
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

    using labSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::SetAlpha<ComputeT, mpp::image::LabtoBGR<Pixel32fC4A, doNormalize>>,
                                     GetRoundingMode<DstT>()>;

    const mpp::image::LabtoBGR<Pixel32fC4A, doNormalize> op(aNormVal);
    const mpp::image::SetAlpha<ComputeT, mpp::image::LabtoBGR<Pixel32fC4A, doNormalize>> opAlpha(op);

    const labSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoBGRSrcP4_For(typeSrcIsTypeDst)                                                           \
    template void InvokeLabtoBGRSrc<typeSrcIsTypeDst>(                                                                 \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, float aNormVal, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoBGRSrcP4(type) InstantiateInvokeLabtoBGRSrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeLabtoBGRInplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, float aNormVal, const Size2D &aSize,
                           const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    constexpr bool doNormalize = RealIntVector<SrcDstT>;

    using labInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::LabtoBGR<ComputeT, doNormalize>, GetRoundingMode<DstT>()>;

    const mpp::image::LabtoBGR<ComputeT, doNormalize> op(aNormVal);

    const labInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, labInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeLabtoBGRInplace_For(typeSrcIsTypeDst)                                                         \
    template void InvokeLabtoBGRInplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,           \
                                                          float aNormVal, const Size2D &aSize,                         \
                                                          const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeLabtoBGRInplace(type)                                                             \
    InstantiateInvokeLabtoBGRInplace_For(Pixel##type##C3);                                                             \
    InstantiateInvokeLabtoBGRInplace_For(Pixel##type##C4A);

#pragma endregion
#pragma endregion
} // namespace mpp::image::cuda
