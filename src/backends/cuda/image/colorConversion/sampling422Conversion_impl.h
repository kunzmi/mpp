#include "sampling422Conversion.h"
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

template <typename SrcDstT>
void InvokeSampling422ConversionC2P2Src(const Vector2<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                        bool aSwapLumaChroma, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                        size_t aPitchDst1, Vector2<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = SrcDstT;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = 1;

    if (aSwapLumaChroma)
    {
        using nopSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::NOP<ComputeT>,
                                       Src422C2Layout::CbYCr, RoundingMode::None>;

        const mpp::image::NOP<ComputeT> op;

        const nopSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixel422Planar2KernelDefault<DstT, nopSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                ChromaSubsamplePos::TopLeft, aStreamCtx, functor);
    }
    else
    {
        using nopSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::NOP<ComputeT>,
                                       Src422C2Layout::YCbCr, RoundingMode::None>;

        const mpp::image::NOP<ComputeT> op;

        const nopSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixel422Planar2KernelDefault<DstT, nopSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aSize,
                                                                ChromaSubsamplePos::TopLeft, aStreamCtx, functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeSampling422ConversionC2P2Src_For(typeSrcIsTypeDst)                                            \
    template void InvokeSampling422ConversionC2P2Src<typeSrcIsTypeDst>(                                                \
        const Vector2<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1, bool aSwapLumaChroma,              \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector2<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2, const Size2D &aSize,                     \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeSampling422ConversionC2P2Src(type)                                                \
    InstantiateInvokeSampling422ConversionC2P2Src_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeSampling422ConversionC2P3Src(const Vector2<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                        bool aSwapLumaChroma, Vector1<remove_vector_t<SrcDstT>> *aDst1,
                                        size_t aPitchDst1, Vector1<remove_vector_t<SrcDstT>> *aDst2, size_t aPitchDst2,
                                        Vector1<remove_vector_t<SrcDstT>> *aDst3, size_t aPitchDst3,
                                        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = SrcDstT;
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = 1;

    if (aSwapLumaChroma)
    {
        using nopSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::NOP<ComputeT>,
                                       Src422C2Layout::CbYCr, RoundingMode::None>;

        const mpp::image::NOP<ComputeT> op;

        const nopSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixel422Planar3KernelDefault<DstT, nopSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3,
                                                                aSize, ChromaSubsamplePos::TopLeft, aStreamCtx,
                                                                functor);
    }
    else
    {
        using nopSrc = Src422C2Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::NOP<ComputeT>,
                                       Src422C2Layout::YCbCr, RoundingMode::None>;

        const mpp::image::NOP<ComputeT> op;

        const nopSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixel422Planar3KernelDefault<DstT, nopSrc>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3,
                                                                aSize, ChromaSubsamplePos::TopLeft, aStreamCtx,
                                                                functor);
    }
}

#pragma region Instantiate
#define InstantiateInvokeSampling422ConversionC2P3Src_For(typeSrcIsTypeDst)                                            \
    template void InvokeSampling422ConversionC2P3Src<typeSrcIsTypeDst>(                                                \
        const Vector2<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1, bool aSwapLumaChroma,              \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst2, size_t aPitchDst2,                                          \
        Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst3, size_t aPitchDst3, const Size2D &aSize,                     \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeSampling422ConversionC2P3Src(type)                                                \
    InstantiateInvokeSampling422ConversionC2P3Src_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeSampling422ConversionP2C2Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                        const Vector2<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                        Vector2<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                        bool aSwapLumaChroma, const Size2D &aSize,
                                        const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = SrcDstT;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 2, aSize.y);

    constexpr size_t TupelSize = 1;

    using nopSrc =
        Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::NOP<ComputeT>, ChromaSubsamplePos::TopLeft,
                      InterpolationMode::NearestNeighbor, false, false, RoundingMode::None>;

    const mpp::image::NOP<ComputeT> op;

    const nopSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, sizeChroma, op);

    InvokeForEachPixel422C2KernelDefault<DstT, nopSrc>(aDst1, aPitchDst1, aSize, ChromaSubsamplePos::TopLeft,
                                                       aSwapLumaChroma ? Dst422C2Layout::CbYCr : Dst422C2Layout::YCbCr,
                                                       aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeSampling422ConversionP2C2Src_For(typeSrcIsTypeDst)                                            \
    template void InvokeSampling422ConversionP2C2Src<typeSrcIsTypeDst>(                                                \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector2<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        Vector2<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1, bool aSwapLumaChroma,                    \
        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeSampling422ConversionP2C2Src(type)                                                \
    InstantiateInvokeSampling422ConversionP2C2Src_For(Pixel##type##C3);

#pragma endregion

template <typename SrcDstT>
void InvokeSampling422ConversionP3C2Src(const Vector1<remove_vector_t<SrcDstT>> *aSrc1, size_t aPitchSrc1,
                                        const Vector1<remove_vector_t<SrcDstT>> *aSrc2, size_t aPitchSrc2,
                                        const Vector1<remove_vector_t<SrcDstT>> *aSrc3, size_t aPitchSrc3,
                                        Vector2<remove_vector_t<SrcDstT>> *aDst1, size_t aPitchDst1,
                                        bool aSwapLumaChroma, const Size2D &aSize,
                                        const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT     = SrcDstT;
    using DstT     = SrcDstT;
    using ComputeT = SrcDstT;
    MPP_CUDA_REGISTER_TEMPALTE;
    const Size2D sizeChroma(aSize.x / 2, aSize.y);

    constexpr size_t TupelSize = 1;

    using nopSrc =
        Src422Functor<TupelSize, SrcT, ComputeT, DstT, mpp::image::NOP<ComputeT>, ChromaSubsamplePos::TopLeft,
                      InterpolationMode::NearestNeighbor, false, true, RoundingMode::None>;

    const mpp::image::NOP<ComputeT> op;

    const nopSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSrc3, aPitchSrc3, sizeChroma, op);

    InvokeForEachPixel422C2KernelDefault<DstT, nopSrc>(aDst1, aPitchDst1, aSize, ChromaSubsamplePos::TopLeft,
                                                       aSwapLumaChroma ? Dst422C2Layout::CbYCr : Dst422C2Layout::YCbCr,
                                                       aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeSampling422ConversionP3C2Src_For(typeSrcIsTypeDst)                                            \
    template void InvokeSampling422ConversionP3C2Src<typeSrcIsTypeDst>(                                                \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc1, size_t aPitchSrc1,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc2, size_t aPitchSrc2,                                    \
        const Vector1<remove_vector_t<typeSrcIsTypeDst>> *aSrc3, size_t aPitchSrc3,                                    \
        Vector2<remove_vector_t<typeSrcIsTypeDst>> *aDst1, size_t aPitchDst1, bool aSwapLumaChroma,                    \
        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeSampling422ConversionP3C2Src(type)                                                \
    InstantiateInvokeSampling422ConversionP3C2Src_For(Pixel##type##C3);

#pragma endregion

#pragma endregion

} // namespace mpp::image::cuda
