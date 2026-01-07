#include "colorTwist4x4.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/image/forEachPixelPlanar4Kernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/colorConversion/color_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/srcFunctor.h>
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

template <typename SrcDstT>
void InvokeColorTwist4x4Src(const SrcDstT *aSrc1, size_t aPitchSrc1, SrcDstT *aDst, size_t aPitchDst,
                            const Matrix4x4<float> &aTwist, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using colorTwistSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist4x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist4x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist4x4Src_For(typeSrcIsTypeDst)                                                        \
    template void InvokeColorTwist4x4Src<typeSrcIsTypeDst>(                                                            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Matrix4x4<float> &aTwist, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist4x4Src(type) InstantiateInvokeColorTwist4x4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist4x4Src(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                            size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                            Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, const Matrix4x4<float> &aTwist,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist4x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist4x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, colorTwistSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist4x4P4Src_For(typeSrcIsTypeDst)                                                      \
    template void InvokeColorTwist4x4Src<typeSrcIsTypeDst>(                                                            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1,           \
        size_t aPitchDst1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                       \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, const Matrix4x4<float> &aTwist,          \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist4x4P4Src(type) InstantiateInvokeColorTwist4x4P4Src_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist4x4Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                            Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                            Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                            Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                            Vector1<remove_vector_t<SrcDstT>> *aDst4, size_t aPitchDst4, const Matrix4x4<float> &aTwist,
                            const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using colorTwistSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist4x4<ComputeT>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist4x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, op);

    InvokeForEachPixelPlanar4KernelDefault<DstT, TupelSize, colorTwistSrc>(
        aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aDst4, aPitchDst4, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist4x4P4SrcP4_For(typeSrcIsTypeDst)                                                    \
    template void InvokeColorTwist4x4Src<typeSrcIsTypeDst>(                                                            \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst4, size_t aPitchDst4, const Matrix4x4<float> &aTwist,          \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist4x4P4SrcP4(type)                                                        \
    InstantiateInvokeColorTwist4x4P4SrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist4x4Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                            const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4, SrcDstT *aDst1,
                            size_t aPitchDst1, const Matrix4x4<float> &aTwist, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using colorTwistSrc = SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist4x4<ComputeT>,
                                            GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist4x4<ComputeT> op(aTwist);

    const colorTwistSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistSrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist4x4SrcP4_For(typeSrcIsTypeDst)                                                      \
    template void InvokeColorTwist4x4Src<typeSrcIsTypeDst>(                                                            \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4, typeSrcIsTypeDst *aDst1,           \
        size_t aPitchDst1, const Matrix4x4<float> &aTwist, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist4x4SrcP4(type) InstantiateInvokeColorTwist4x4SrcP4_For(Pixel##type##C4);

#pragma endregion

template <typename SrcDstT>
void InvokeColorTwist4x4Inplace(SrcDstT *aSrcDst1, size_t aPitchSrcDst1, const Matrix4x4<float> &aTwist,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using colorTwistInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::ColorTwist4x4<ComputeT>, GetRoundingMode<DstT>()>;

    const mpp::image::ColorTwist4x4<ComputeT> op(aTwist);

    const colorTwistInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, colorTwistInplace>(aSrcDst1, aPitchSrcDst1, aSize, aStreamCtx,
                                                                        functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorTwist4x4Inplace_For(typeSrcIsTypeDst)                                                    \
    template void InvokeColorTwist4x4Inplace<typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst1, size_t aPitchSrcDst1,      \
                                                               const Matrix4x4<float> &aTwist, const Size2D &aSize,    \
                                                               const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorTwist4x4Inplace(type)                                                        \
    InstantiateInvokeColorTwist4x4Inplace_For(Pixel##type##C4);

#pragma endregion

} // namespace mpp::image::cuda
