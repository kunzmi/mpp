#include "colorToGray.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/colorConversion/color_operators.h>
#include <common/defines.h>
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
void InvokeColorToGraySrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst,
                          size_t aPitchDst, const same_vector_size_different_type_t<SrcDstT, float> &aWeights,
                          const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = Vector1<remove_vector_t<SrcDstT>>;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize =
        vector_size_v<SrcDstT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using tograySrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorToGray<ComputeT>, GetRoundingMode<DstT>()>;
    const mpp::image::ColorToGray<ComputeT> op(aWeights);
    const tograySrc functor(aSrc1, aPitchSrc1, op);
    InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorToGraySrc_For(typeSrcIsTypeDst)                                                          \
    template void InvokeColorToGraySrc<typeSrcIsTypeDst>(                                                              \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst,            \
        size_t aPitchDst, const same_vector_size_different_type_t<typeSrcIsTypeDst, float> &aWeights,                  \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorToGraySrc(type)                                                              \
    InstantiateInvokeColorToGraySrc_For(Pixel##type##C2);                                                              \
    InstantiateInvokeColorToGraySrc_For(Pixel##type##C3);                                                              \
    InstantiateInvokeColorToGraySrc_For(Pixel##type##C4);                                                              \
    InstantiateInvokeColorToGraySrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorToGraySrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                          const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                          Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                          const same_vector_size_different_type_t<SrcDstT, float> &aWeights, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = Vector1<remove_vector_t<SrcDstT>>;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using tograySrc =
        SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorToGray<ComputeT>, GetRoundingMode<DstT>()>;
    const mpp::image::ColorToGray<ComputeT> op(aWeights);
    const tograySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
    InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorToGraySrcP2_For(typeSrcIsTypeDst)                                                        \
    template void InvokeColorToGraySrc<typeSrcIsTypeDst>(                                                              \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        const same_vector_size_different_type_t<typeSrcIsTypeDst, float> &aWeights, const Size2D &aSize,               \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorToGraySrcP2(type) InstantiateInvokeColorToGraySrcP2_For(Pixel##type##C2);
#pragma endregion

template <typename SrcDstT>
void InvokeColorToGraySrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                          const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                          const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                          Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                          const same_vector_size_different_type_t<SrcDstT, float> &aWeights, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = Vector1<remove_vector_t<SrcDstT>>;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using tograySrc =
        SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorToGray<ComputeT>, GetRoundingMode<DstT>()>;
    const mpp::image::ColorToGray<ComputeT> op(aWeights);
    const tograySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);
    InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorToGraySrcP3_For(typeSrcIsTypeDst)                                                        \
    template void InvokeColorToGraySrc<typeSrcIsTypeDst>(                                                              \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        const same_vector_size_different_type_t<typeSrcIsTypeDst, float> &aWeights, const Size2D &aSize,               \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorToGraySrcP3(type) InstantiateInvokeColorToGraySrcP3_For(Pixel##type##C4A);
#pragma endregion

template <typename SrcDstT>
void InvokeColorToGraySrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                          const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                          const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                          const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                          Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                          const same_vector_size_different_type_t<SrcDstT, float> &aWeights, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = Vector1<remove_vector_t<SrcDstT>>;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    using tograySrc =
        SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorToGray<ComputeT>, GetRoundingMode<DstT>()>;
    const mpp::image::ColorToGray<ComputeT> op(aWeights);
    const tograySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, op);
    InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeColorToGraySrcP4_For(typeSrcIsTypeDst)                                                        \
    template void InvokeColorToGraySrc<typeSrcIsTypeDst>(                                                              \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        const same_vector_size_different_type_t<typeSrcIsTypeDst, float> &aWeights, const Size2D &aSize,               \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorToGraySrcP4(type) InstantiateInvokeColorToGraySrcP4_For(Pixel##type##C4);

#pragma endregion

} // namespace mpp::image::cuda
