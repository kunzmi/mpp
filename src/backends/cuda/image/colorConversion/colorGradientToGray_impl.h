#include "colorGradientToGray.h"
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
void InvokeColorGradientToGraySrc(const SrcDstT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcDstT>> *aDst,
                                  size_t aPitchDst, Norm aNorm, const Size2D &aSize,
                                  const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = Vector1<remove_vector_t<SrcDstT>>;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize =
        vector_size_v<SrcDstT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    if (aNorm == Norm::Inf)
    {
        using tograySrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::ColorGradientToGray<ComputeT, Norm::Inf>, GetRoundingMode<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::Inf> op;
        const tograySrc functor(aSrc1, aPitchSrc1, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else if (aNorm == Norm::L1)
    {
        using tograySrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::ColorGradientToGray<ComputeT, Norm::L1>, GetRoundingMode<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L1> op;
        const tograySrc functor(aSrc1, aPitchSrc1, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else if (aNorm == Norm::L2)
    {
        using tograySrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                                     mpp::image::ColorGradientToGray<ComputeT, Norm::L2>, GetRoundingMode<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L2> op;
        const tograySrc functor(aSrc1, aPitchSrc1, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else
    {
        throw INVALIDARGUMENT(aNorm, "Unknown Norm '" << aNorm << "'. Expected either Inf, L1 or L2.");
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorGradientToGraySrc_For(typeSrcIsTypeDst)                                                  \
    template void InvokeColorGradientToGraySrc<typeSrcIsTypeDst>(                                                      \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst,            \
        size_t aPitchDst, Norm aNorm, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorGradientToGraySrc(type)                                                      \
    InstantiateInvokeColorGradientToGraySrc_For(Pixel##type##C2);                                                      \
    InstantiateInvokeColorGradientToGraySrc_For(Pixel##type##C3);                                                      \
    InstantiateInvokeColorGradientToGraySrc_For(Pixel##type##C4);                                                      \
    InstantiateInvokeColorGradientToGraySrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeColorGradientToGraySrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                  const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                  Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1, Norm aNorm,
                                  const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = Vector1<remove_vector_t<SrcDstT>>;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    if (aNorm == Norm::Inf)
    {
        using tograySrc =
            SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::Inf>,
                              GetRoundingMode<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::Inf> op;
        const tograySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
    }
    else if (aNorm == Norm::L1)
    {
        using tograySrc =
            SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L1>,
                              GetRoundingMode<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L1> op;
        const tograySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
    }
    else if (aNorm == Norm::L2)
    {
        using tograySrc =
            SrcPlanar2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L2>,
                              GetRoundingMode<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L2> op;
        const tograySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
    }
    else
    {
        throw INVALIDARGUMENT(aNorm, "Unknown Norm '" << aNorm << "'. Expected either Inf, L1 or L2.");
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorGradientToGraySrcP2_For(typeSrcIsTypeDst)                                                \
    template void InvokeColorGradientToGraySrc<typeSrcIsTypeDst>(                                                      \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1, Norm aNorm, const Size2D &aSize,         \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP2(type)                                                    \
    InstantiateInvokeColorGradientToGraySrcP2_For(Pixel##type##C2);
#pragma endregion

template <typename SrcDstT>
void InvokeColorGradientToGraySrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                  const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                  const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                  Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1, Norm aNorm,
                                  const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = Vector1<remove_vector_t<SrcDstT>>;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    if (aNorm == Norm::Inf)
    {
        using tograySrc =
            SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::Inf>,
                              GetRoundingMode<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::Inf> op;
        const tograySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
    }
    else if (aNorm == Norm::L1)
    {
        using tograySrc =
            SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L1>,
                              GetRoundingMode<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L1> op;
        const tograySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
    }
    else if (aNorm == Norm::L2)
    {
        using tograySrc =
            SrcPlanar3Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L2>,
                              GetRoundingMode<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L2> op;
        const tograySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
    }
    else
    {
        throw INVALIDARGUMENT(aNorm, "Unknown Norm '" << aNorm << "'. Expected either Inf, L1 or L2.");
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorGradientToGraySrcP3_For(typeSrcIsTypeDst)                                                \
    template void InvokeColorGradientToGraySrc<typeSrcIsTypeDst>(                                                      \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1, Norm aNorm, const Size2D &aSize,         \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP3(type)                                                    \
    InstantiateInvokeColorGradientToGraySrcP3_For(Pixel##type##C4A);
#pragma endregion

template <typename SrcDstT>
void InvokeColorGradientToGraySrc(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                  const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                  const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                  const Vector1<remove_vector_t<SrcDstT>> *aSrc4, size_t aPitchSrc4,
                                  Vector1<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1, Norm aNorm,
                                  const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = Vector1<remove_vector_t<SrcDstT>>;
    using ComputeT = same_vector_size_different_type_t<SrcDstT, float>;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(Vector1<remove_vector_t<SrcDstT>>)>::value;

    if (aNorm == Norm::Inf)
    {
        using tograySrc =
            SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::Inf>,
                              GetRoundingMode<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::Inf> op;
        const tograySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
    }
    else if (aNorm == Norm::L1)
    {
        using tograySrc =
            SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L1>,
                              GetRoundingMode<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L1> op;
        const tograySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
    }
    else if (aNorm == Norm::L2)
    {
        using tograySrc =
            SrcPlanar4Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorGradientToGray<ComputeT, Norm::L2>,
                              GetRoundingMode<DstT>()>;
        const mpp::image::ColorGradientToGray<ComputeT, Norm::L2> op;
        const tograySrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, aSrc4, aPitchSrc4, op);
        InvokeForEachPixelKernelDefault<DstT, TupelSize, tograySrc>(aDst1, aPitchDst1, aSize, aStreamCtx, functor);
    }
    else
    {
        throw INVALIDARGUMENT(aNorm, "Unknown Norm '" << aNorm << "'. Expected either Inf, L1 or L2.");
    }
}

#pragma region Instantiate
#define InstantiateInvokeColorGradientToGraySrcP4_For(typeSrcIsTypeDst)                                                \
    template void InvokeColorGradientToGraySrc<typeSrcIsTypeDst>(                                                      \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc4, size_t aPitchSrc4,                                    \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1, Norm aNorm, const Size2D &aSize,         \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeColorGradientToGraySrcP4(type)                                                    \
    InstantiateInvokeColorGradientToGraySrcP4_For(Pixel##type##C4);

#pragma endregion

} // namespace mpp::image::cuda
