#include "colorTwist3x4.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixel411Planar2Kernel.h>
#include <backends/cuda/image/forEachPixel411Planar3Kernel.h>
#include <backends/cuda/image/forEachPixel420Planar2Kernel.h>
#include <backends/cuda/image/forEachPixel420Planar3Kernel.h>
#include <backends/cuda/image/forEachPixel422C2Kernel.h>
#include <backends/cuda/image/forEachPixel422Planar2Kernel.h>
#include <backends/cuda/image/forEachPixel422Planar3Kernel.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/image/forEachPixelPlanar3Kernel.h>
#include <backends/cuda/image/forEachPixelPlanar4Kernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/colorConversion/color_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/src411Functor.h>
#include <common/image/functors/src420Functor.h>
#include <common/image/functors/src422C2Functor.h>
#include <common/image/functors/src422Functor.h>
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
    if constexpr (RealSignedVector<T>)
    {
        return RoundingMode::NearestTiesAwayFromZero;
    }
    return RoundingMode::NearestTiesAwayFromZeroPositive;
}
} // namespace

template <typename SrcDstT>
void InvokeColorTwist3x4Src(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                            const Matrix3x4<float> &aTwist, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using colorTwistSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src_For(typeSrcIsTypeDst)                                                        \
    template void InvokeColorTwist3x4Src<typeSrcIsTypeDst>(                                                            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Matrix3x4<float> &aTwist, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src(type)                                                            \
    InstantiateInvokeColorTwist3x4Src_For(Pixel##type##C3);                                                            \
    InstantiateInvokeColorTwist3x4Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                            size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize =
        vector_size_v<SrcDstT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                           aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4P3Src_For(typeSrcIsTypeDst)                                                      \
    template void InvokeColorTwist3x4Src<typeSrcIsTypeDst>(                                                            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4P3Src(type)                                                          \
    InstantiateInvokeColorTwist3x4P3Src_For(Pixel##type##C3);                                                          \
    InstantiateInvokeColorTwist3x4P3Src_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                            size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                            Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, const Matrix3x4<float> &aTwist,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize =
        vector_size_v<SrcDstT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                   mpp::image::SetAlpha<ComputeT, mpp::image::ColorTwist3x4<Pixel32fC4A>>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<Pixel32fC4A> op(aTwist);
    const mpp::image::SetAlpha<ComputeT, mpp::image::ColorTwist3x4<Pixel32fC4A>> opAlpha(op);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, opAlpha);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, colorTwistSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4P4Src_For(typeSrcIsTypeDst)                                                      \
    template void InvokeColorTwist3x4Src<typeSrcIsTypeDst>(                                                            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4P4Src(type) InstantiateInvokeColorTwist3x4P4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                            Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                            Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3,
                                                                           aPitchDst3, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4P3SrcP3_For(typeSrcIsTypeDst)                                                    \
    template void InvokeColorTwist3x4Src<typeSrcIsTypeDst>(                                                            \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4P3SrcP3(type)                                                        \
    InstantiateInvokeColorTwist3x4P3SrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                            size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4SrcP3_For(typeSrcIsTypeDst)                                                      \
    template void InvokeColorTwist3x4Src<typeSrcIsTypeDst>(                                                            \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4SrcP3(type)                                                          \
    InstantiateInvokeColorTwist3x4SrcP3_For(Pixel##type##C3);                                                          \
    InstantiateInvokeColorTwist3x4SrcP3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                            size_t aPitchDst1, remove_vector_t<SrcDstT> aAlpha, const Matrix3x4<float> &aTwist,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = Vector4A<remove_vector_t<SrcDstT>>;
    using DstT     = SrcDstT;
    using ComputeT = Pixel32fC4A;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT,
                                            mpp::image::SetAlphaConst<ComputeT, mpp::image::ColorTwist3x4<Pixel32fC4A>>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<Pixel32fC4A> op(aTwist);
    const mpp::image::SetAlphaConst<ComputeT, mpp::image::ColorTwist3x4<Pixel32fC4A>> opAlpha(
        static_cast<float>(aAlpha), op);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4C4SrcP3_For(typeSrcIsTypeDst)                                                    \
    template void InvokeColorTwist3x4Src<typeSrcIsTypeDst>(                                                            \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, remove_vector_t<typeSrcIsTypeDst> aAlpha, const Matrix3x4<float> &aTwist,                   \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4C4SrcP3(type)                                                        \
    InstantiateInvokeColorTwist3x4C4SrcP3_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4, SrcDstT *aDst1,
                            size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using colorTwistSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT,
                                            mpp::image::SetAlpha<ComputeT, mpp::image::ColorTwist3x4<Pixel32fC4A>>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<Pixel32fC4A> op(aTwist);
    const mpp::image::SetAlpha<ComputeT, mpp::image::ColorTwist3x4<Pixel32fC4A>> opAlpha(op);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, opAlpha);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4SrcP4_For(typeSrcIsTypeDst)                                                      \
    template void InvokeColorTwist3x4Src<typeSrcIsTypeDst>(                                                            \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4SrcP4(type) InstantiateInvokeColorTwist3x4SrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Inplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, const Matrix3x4<float> &aTwist,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using colorTwistInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                        functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Inplace_For(typeSrcIsTypeDst)                                                    \
    template void InvokeColorTwist3x4Inplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,      \
                                                               const Matrix3x4<float> &aTwist, const Size2D &aSize,    \
                                                               const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Inplace(type)                                                        \
    InstantiateInvokeColorTwist3x4Inplace_For(Pixel##type##C3);                                                        \
    InstantiateInvokeColorTwist3x4Inplace_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to422(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector2<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize =
        vector_size_v<SrcDstT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(Vector2<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixel422C2KernelDefault<DstT, colorTwistSrc>(
        aDst1, aPitchDst1, aSize, aChromaSubsamplePos, aSwapLumaChroma ? Dst422C2Layout::CbYCr : Dst422C2Layout::YCbCr,
        aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src444to422C2_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src444to422<typeSrcIsTypeDst>(                                                    \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector2<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,                                        \
        ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src444to422C2(type)                                                  \
    InstantiateInvokeColorTwist3x4Src444to422C2_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src444to422C2_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to422(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector2<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector2<remove_vector_t<DstT>>)>::value;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixel422C2KernelDefault<DstT, colorTwistSrc>(
        aDst1, aPitchDst1, aSize, aChromaSubsamplePos, aSwapLumaChroma ? Dst422C2Layout::CbYCr : Dst422C2Layout::YCbCr,
        aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4SrcP3444to422C2_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src444to422<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector2<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma,                             \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4SrcP3444to422C2(type)                                                \
    InstantiateInvokeColorTwist3x4SrcP3444to422C2_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to422(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize =
        vector_size_v<SrcDstT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(Vector2<remove_vector_t<DstT>>)>::value;

    using colorTwistSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixel422Planar2KernelDefault<DstT, colorTwistSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                   aChromaSubsamplePos, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src444to422P2_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src444to422<typeSrcIsTypeDst>(                                                    \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector2<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        const Matrix3x4<float> &aTwist, const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos,                   \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src444to422P2(type)                                                  \
    InstantiateInvokeColorTwist3x4Src444to422P2_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src444to422P2_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to422(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize =
        vector_size_v<SrcDstT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<DstT>>)>::value;

    using colorTwistSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixel422Planar3KernelDefault<DstT, colorTwistSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aChromaSubsamplePos, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src444to422P3_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src444to422<typeSrcIsTypeDst>(                                                    \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src444to422P3(type)                                                  \
    InstantiateInvokeColorTwist3x4Src444to422P3_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src444to422P3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to422(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector2<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixel422Planar2KernelDefault<DstT, colorTwistSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                   aChromaSubsamplePos, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4SrcP3444to422P2_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src444to422<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector2<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4SrcP3444to422P2(type)                                                \
    InstantiateInvokeColorTwist3x4SrcP3444to422P2_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to422(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixel422Planar3KernelDefault<DstT, colorTwistSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aChromaSubsamplePos, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4SrcP3444to422P3_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src444to422<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4SrcP3444to422P3(type)                                                \
    InstantiateInvokeColorTwist3x4SrcP3444to422P3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to420(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize =
        vector_size_v<SrcDstT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(Vector2<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixel420Planar2KernelDefault<DstT, colorTwistSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                   aChromaSubsamplePos, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src444to420P2_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src444to420<typeSrcIsTypeDst>(                                                    \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector2<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        const Matrix3x4<float> &aTwist, const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos,                   \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src444to420P2(type)                                                  \
    InstantiateInvokeColorTwist3x4Src444to420P2_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src444to420P2_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to420(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize =
        vector_size_v<SrcDstT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixel420Planar3KernelDefault<DstT, colorTwistSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aChromaSubsamplePos, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src444to420P3_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src444to420<typeSrcIsTypeDst>(                                                    \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src444to420P3(type)                                                  \
    InstantiateInvokeColorTwist3x4Src444to420P3_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src444to420P3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to420(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector2<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixel420Planar2KernelDefault<DstT, colorTwistSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                   aChromaSubsamplePos, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4SrcP3444to420P2_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src444to420<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector2<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4SrcP3444to420P2(type)                                                \
    InstantiateInvokeColorTwist3x4SrcP3444to420P2_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to420(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixel420Planar3KernelDefault<DstT, colorTwistSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aChromaSubsamplePos, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4SrcP3444to420P3_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src444to420<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4SrcP3444to420P3(type)                                                \
    InstantiateInvokeColorTwist3x4SrcP3444to420P3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to411(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize =
        vector_size_v<SrcDstT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(Vector2<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixel411Planar2KernelDefault<DstT, colorTwistSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                   aChromaSubsamplePos, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src444to411P2_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src444to411<typeSrcIsTypeDst>(                                                    \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector2<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        const Matrix3x4<float> &aTwist, const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos,                   \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src444to411P2(type)                                                  \
    InstantiateInvokeColorTwist3x4Src444to411P2_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src444to411P2_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to411(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                    size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize =
        vector_size_v<SrcDstT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixel411Planar3KernelDefault<DstT, colorTwistSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aChromaSubsamplePos, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src444to411P3_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src444to411<typeSrcIsTypeDst>(                                                    \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src444to411P3(type)                                                  \
    InstantiateInvokeColorTwist3x4Src444to411P3_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src444to411P3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to411(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector2<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixel411Planar2KernelDefault<DstT, colorTwistSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                   aChromaSubsamplePos, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4SrcP3444to411P2_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src444to411<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector2<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4SrcP3444to411P2(type)                                                \
    InstantiateInvokeColorTwist3x4SrcP3444to411P2_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src444to411(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc = SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);

    InvokeForEachPixel411Planar3KernelDefault<DstT, colorTwistSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aChromaSubsamplePos, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4SrcP3444to411P3_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src444to411<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4SrcP3444to411P3(type)                                                \
    InstantiateInvokeColorTwist3x4SrcP3444to411P3_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src422to444(const Vector2<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1, SrcDstT *aDst1,
                                    size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    bool aSwapLumaChroma, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (aSwapLumaChroma)
    {
        using colorTwistSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                              Src422C2Layout::CbYCr, GetRoundingMode<DstT>()>;

        const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

        const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
    }
    else
    {
        using colorTwistSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                              Src422C2Layout::YCbCr, GetRoundingMode<DstT>()>;

        const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

        const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src422C2to444_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src422to444<typeSrcIsTypeDst>(                                                    \
        const Vector2<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize, bool aSwapLumaChroma,                  \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src422C2to444(type)                                                  \
    InstantiateInvokeColorTwist3x4Src422C2to444_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src422C2to444_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src422to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2, SrcDstT *aDst1,
                                    size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 2, aSize.y);

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src422Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Left,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src422Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Center,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src422P2to444_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src422to444<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector2<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,                                        \
        ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src422P2to444(type)                                                  \
    InstantiateInvokeColorTwist3x4Src422P2to444_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src422P2to444_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src422to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                                    size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 2, aSize.y);

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src422Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Left,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src422Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Center,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src422P3to444_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src422to444<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,                                        \
        ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src422P3to444(type)                                                  \
    InstantiateInvokeColorTwist3x4Src422P3to444_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src422P3to444_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src422to444(const Vector2<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize, bool aSwapLumaChroma,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    if (aSwapLumaChroma)
    {
        using colorTwistSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                              Src422C2Layout::CbYCr, GetRoundingMode<DstT>()>;

        const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

        const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
            aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
    }
    else
    {
        using colorTwistSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                              Src422C2Layout::YCbCr, GetRoundingMode<DstT>()>;

        const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

        const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
            aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src422C2to444P3_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src422to444<typeSrcIsTypeDst>(                                                    \
        const Vector2<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, bool aSwapLumaChroma, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src422C2to444P3(type)                                                \
    InstantiateInvokeColorTwist3x4Src422C2to444P3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src422to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 2, aSize.y);

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src422Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Left,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src422Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Center,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src422P2to444P3_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src422to444<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector2<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,             \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src422P2to444P3(type)                                                \
    InstantiateInvokeColorTwist3x4Src422P2to444P3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src422to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 2, aSize.y);

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src422Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Left,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src422Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Center,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src422P3to444P3_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src422to444<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,             \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src422P3to444P3(type)                                                \
    InstantiateInvokeColorTwist3x4Src422P3to444P3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src420to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2, SrcDstT *aDst1,
                                    size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 2, aSize.y / 2);

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src420Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Left,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else if (aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src420Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::TopLeft,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src420Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Center,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src420P2to444_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src420to444<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector2<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,                                        \
        ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src420P2to444(type)                                                  \
    InstantiateInvokeColorTwist3x4Src420P2to444_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src420P2to444_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src420to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                                    size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 2, aSize.y / 2);

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src420Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Left,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else if (aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src420Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::TopLeft,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src420Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Center,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src420P3to444_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src420to444<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,                                        \
        ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src420P3to444(type)                                                  \
    InstantiateInvokeColorTwist3x4Src420P3to444_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src420P3to444_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src420to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 2, aSize.y / 2);

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src420Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Left,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else if (aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src420Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::TopLeft,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src420Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Center,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src420P2to444P3_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src420to444<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector2<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,             \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src420P2to444P3(type)                                                \
    InstantiateInvokeColorTwist3x4Src420P2to444P3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src420to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 2, aSize.y / 2);

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src420Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Left,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else if (aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src420Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::TopLeft,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::TopLeft, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src420Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Center,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src420Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src420P3to444P3_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src420to444<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,             \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src420P3to444P3(type)                                                \
    InstantiateInvokeColorTwist3x4Src420P3to444P3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src411to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2, SrcDstT *aDst1,
                                    size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 4, aSize.y);

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src411Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Left,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src411Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Center,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src411P2to444_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src411to444<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector2<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,                                        \
        ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src411P2to444(type)                                                  \
    InstantiateInvokeColorTwist3x4Src411P2to444_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src411P2to444_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src411to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3, SrcDstT *aDst1,
                                    size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 4, aSize.y);

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src411Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Left,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src411Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Center,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src411P3to444_For(typeSrcIsTypeDst)                                              \
    template void InvokeColorTwist3x4Src411to444<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, const Matrix3x4<float> &aTwist, const Size2D &aSize,                                        \
        ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src411P3to444(type)                                                  \
    InstantiateInvokeColorTwist3x4Src411P3to444_For(Pixel##type##C3);                                                  \
    InstantiateInvokeColorTwist3x4Src411P3to444_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src411to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 4, aSize.y);

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src411Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Left,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src411Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Center,
                                                    InterpolationMode::Linear, false, false, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, false,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src411P2to444P3_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src411to444<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector2<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,             \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src411P2to444P3(type)                                                \
    InstantiateInvokeColorTwist3x4Src411P2to444P3_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist3x4Src411to444(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                    const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                    Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                    const Matrix3x4<float> &aTwist, const Size2D &aSize,
                                    ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 4, aSize.y);

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    if (aChromaSubsamplePos == ChromaSubsamplePos::Left || aChromaSubsamplePos == ChromaSubsamplePos::TopLeft)
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src411Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Left,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Left, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
    else // Undefined or Center
    {
        switch (aInterpolationMode)
        {
            case InterpolationMode::NearestNeighbor:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::NearestNeighbor, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::Linear:
            {
                using colorTwistSrc = Src411Functor<TupelSize, SrcT, ComputeT, DstT,
                                                    mpp::image::ColorTwist3x4<ComputeT>, ChromaSubsamplePos::Center,
                                                    InterpolationMode::Linear, false, true, GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            case InterpolationMode::CubicHermiteSpline:
            {
                using colorTwistSrc =
                    Src411Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>,
                                  ChromaSubsamplePos::Center, InterpolationMode::CubicHermiteSpline, false, true,
                                  GetRoundingMode<DstT>()>;

                const mpp::image::ColorTwist3x4<ComputeT> op(aTwist);

                const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

                InvokeForEachPixelPlanar3KernelDefault<DstT, TupelSize, colorTwistSrc>(
                    aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize, aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aInterpolationMode,
                                      aInterpolationMode
                                          << " is not a supported interpolation mode for chroma upsampling. "
                                             "Implemented are: NearestNeighbor, Linear and CubicHermiteSpline.");
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist3x4Src411P3to444P3_For(typeSrcIsTypeDst)                                            \
    template void InvokeColorTwist3x4Src411to444<typeSrcIsTypeDst>(                                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Matrix3x4<float> &aTwist,          \
        const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,             \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist3x4Src411P3to444P3(type)                                                \
    InstantiateInvokeColorTwist3x4Src411P3to444P3_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
