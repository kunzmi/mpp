#if MPP_ENABLE_CUDA_BACKEND

#include "div.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/binary_operators.h>
#include <backends/cuda/simd_operators/simd_types.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceConstantFunctor.h>
#include <common/image/functors/inplaceConstantScaleFunctor.h>
#include <common/image/functors/inplaceDevConstantFunctor.h>
#include <common/image/functors/inplaceDevConstantScaleFunctor.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/inplaceSrcScaleFunctor.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcConstantScaleFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
#include <common/image/functors/srcDevConstantScaleFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/functors/srcSrcScaleFunctor.h>
#include <common/image/pixelTypeEnabler.h>
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
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                     size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Div<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divSrcSrcSIMD = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                RoundingMode::None, ComputeT, simdOP_t>;

            const mpp::Div<ComputeT, DstT> op;
            const simdOP_t opSIMD;

            const divSrcSrcSIMD functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcSIMD>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                            functor);
        }
        else
        {
            using divSrcSrc =
                SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, RoundingMode::None>;

            const mpp::Div<ComputeT, DstT> op;

            const divSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivSrcSrc_For(typeSrcIsTypeDst)                                                               \
    template void InvokeDivSrcSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivSrcSrc(type)                                                                     \
    InstantiateInvokeDivSrcSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeDivSrcSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeDivSrcSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeDivSrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivSrcSrc(type)                                                                   \
    InstantiateInvokeDivSrcSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeDivSrcSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeDivSrcSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeDivSrcSrc_For(Pixel##type##C4);                                                                   \
    InstantiateInvokeDivSrcSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcSrcScale(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if constexpr (RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                              ScalerT, RoundingMode::NearestTiesToEven>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                     functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                              ScalerT, RoundingMode::NearestTiesAwayFromZero>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                     functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                              ScalerT, RoundingMode::TowardZero>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                     functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                              ScalerT, RoundingMode::TowardNegativeInfinity>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                     functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                              ScalerT, RoundingMode::TowardPositiveInfinity>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                     functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
        else
        {
            if (aScaleFactor < 1)
            {
                if (aRoundingMode != RoundingMode::NearestTiesToEven)
                {
                    throw INVALIDARGUMENT(aRoundingMode,
                                          "Unsupported rounding mode: "
                                              << aRoundingMode << ". Only rounding mode "
                                              << RoundingMode::NearestTiesToEven
                                              << " is supported for this source data type and scaling factor < 1.");
                }
                else
                {
                    using ScalerT = Scale<ComputeT, true>;
                    const ScalerT scaler(aScaleFactor);
                    using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                              ScalerT, RoundingMode::None>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                     functor);
                }
            }
            else
            {
                switch (aRoundingMode)
                {
                    case mpp::RoundingMode::NearestTiesToEven:
                    {
                        using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                        using divSrcSrcScale = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const OpT op(aScaleFactor);
                        const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize,
                                                                                         aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::NearestTiesAwayFromZero:
                    {
                        using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                        using divSrcSrcScale = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const OpT op(aScaleFactor);
                        const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize,
                                                                                         aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardZero:
                    {
                        using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                        using divSrcSrcScale = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const OpT op(aScaleFactor);
                        const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize,
                                                                                         aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardNegativeInfinity:
                    {
                        using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                        using divSrcSrcScale = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const OpT op(aScaleFactor);
                        const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize,
                                                                                         aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardPositiveInfinity:
                    {
                        using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                        using divSrcSrcScale = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const OpT op(aScaleFactor);
                        const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize,
                                                                                         aStreamCtx, functor);
                    }
                    break;
                    default:
                        throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
                }
            }
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeDivSrcSrcScale_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeDivSrcSrcScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, double aScaleFactor, mpp::RoundingMode aRoundingMode,                \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivSrcSrcScale(type)                                                                \
    InstantiateInvokeDivSrcSrcScale_For(Pixel##type##C1);                                                              \
    InstantiateInvokeDivSrcSrcScale_For(Pixel##type##C2);                                                              \
    InstantiateInvokeDivSrcSrcScale_For(Pixel##type##C3);                                                              \
    InstantiateInvokeDivSrcSrcScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivSrcSrcScale(type)                                                              \
    InstantiateInvokeDivSrcSrcScale_For(Pixel##type##C1);                                                              \
    InstantiateInvokeDivSrcSrcScale_For(Pixel##type##C2);                                                              \
    InstantiateInvokeDivSrcSrcScale_For(Pixel##type##C3);                                                              \
    InstantiateInvokeDivSrcSrcScale_For(Pixel##type##C4);                                                              \
    InstantiateInvokeDivSrcSrcScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                   const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Div<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divSrcCSIMD = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                   RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            const mpp::Div<ComputeT, DstT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            const divSrcCSIMD functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCSIMD>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using divSrcC =
                SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, RoundingMode::None>;

            const mpp::Div<ComputeT, DstT> op;

            const divSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivSrcC_For(typeSrcIsTypeDst)                                                                 \
    template void InvokeDivSrcC<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivSrcC(type)                                                                       \
    InstantiateInvokeDivSrcC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeDivSrcC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeDivSrcC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeDivSrcC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivSrcC(type)                                                                     \
    InstantiateInvokeDivSrcC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeDivSrcC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeDivSrcC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeDivSrcC_For(Pixel##type##C4);                                                                     \
    InstantiateInvokeDivSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                        double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                        const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if constexpr (RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using divSrcCScale =
                        SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                RoundingMode::NearestTiesToEven>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                   functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using divSrcCScale =
                        SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                RoundingMode::NearestTiesAwayFromZero>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                   functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using divSrcCScale =
                        SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                RoundingMode::TowardZero>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                   functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using divSrcCScale =
                        SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                RoundingMode::TowardNegativeInfinity>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                   functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using divSrcCScale =
                        SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                RoundingMode::TowardPositiveInfinity>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                   functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
        else
        {
            if (aScaleFactor < 1)
            {
                if (aRoundingMode != RoundingMode::NearestTiesToEven)
                {
                    throw INVALIDARGUMENT(aRoundingMode,
                                          "Unsupported rounding mode: "
                                              << aRoundingMode << ". Only rounding mode "
                                              << RoundingMode::NearestTiesToEven
                                              << " is supported for this source data type and scaling factor < 1.");
                }
                else
                {
                    using ScalerT = Scale<ComputeT, true>;
                    const ScalerT scaler(aScaleFactor);
                    using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                                 mpp::Div<ComputeT, DstT>, ScalerT, RoundingMode::None>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                   functor);
                }
            }
            else
            {
                switch (aRoundingMode)
                {
                    case mpp::RoundingMode::NearestTiesToEven:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                        const OpT op(aScaleFactor);
                        using divSrcCScale =
                            SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;

                        const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize,
                                                                                       aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::NearestTiesAwayFromZero:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                        const OpT op(aScaleFactor);
                        using divSrcCScale =
                            SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;

                        const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize,
                                                                                       aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardZero:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                        const OpT op(aScaleFactor);
                        using divSrcCScale =
                            SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;

                        const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize,
                                                                                       aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardNegativeInfinity:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                        const OpT op(aScaleFactor);
                        using divSrcCScale =
                            SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;

                        const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize,
                                                                                       aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardPositiveInfinity:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                        const OpT op(aScaleFactor);
                        using divSrcCScale =
                            SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;

                        const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize,
                                                                                       aStreamCtx, functor);
                    }
                    break;
                    default:
                        throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
                }
            }
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeDivSrcCScale_For(typeSrcIsTypeDst)                                                            \
    template void                                                                                                      \
    InvokeDivSrcCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(              \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize,                   \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivSrcCScale(type)                                                                  \
    InstantiateInvokeDivSrcCScale_For(Pixel##type##C1);                                                                \
    InstantiateInvokeDivSrcCScale_For(Pixel##type##C2);                                                                \
    InstantiateInvokeDivSrcCScale_For(Pixel##type##C3);                                                                \
    InstantiateInvokeDivSrcCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivSrcCScale(type)                                                                \
    InstantiateInvokeDivSrcCScale_For(Pixel##type##C1);                                                                \
    InstantiateInvokeDivSrcCScale_For(Pixel##type##C2);                                                                \
    InstantiateInvokeDivSrcCScale_For(Pixel##type##C3);                                                                \
    InstantiateInvokeDivSrcCScale_For(Pixel##type##C4);                                                                \
    InstantiateInvokeDivSrcCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using divSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, RoundingMode::None>;

        const mpp::Div<ComputeT, DstT> op;

        const divSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivSrcDevC_For(typeSrcIsTypeDst)                                                              \
    template void InvokeDivSrcDevC<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(  \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivSrcDevC(type)                                                                    \
    InstantiateInvokeDivSrcDevC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeDivSrcDevC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeDivSrcDevC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeDivSrcDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivSrcDevC(type)                                                                  \
    InstantiateInvokeDivSrcDevC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeDivSrcDevC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeDivSrcDevC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeDivSrcDevC_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeDivSrcDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcDevCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                           double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                           const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if constexpr (RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using divSrcDevCScale =
                        SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                   RoundingMode::NearestTiesToEven>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using divSrcDevCScale =
                        SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                   RoundingMode::NearestTiesAwayFromZero>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using divSrcDevCScale =
                        SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                   RoundingMode::TowardZero>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using divSrcDevCScale =
                        SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                   RoundingMode::TowardNegativeInfinity>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using divSrcDevCScale =
                        SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                   RoundingMode::TowardPositiveInfinity>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
        else
        {
            if (aScaleFactor < 1)
            {
                if (aRoundingMode != RoundingMode::NearestTiesToEven)
                {
                    throw INVALIDARGUMENT(aRoundingMode,
                                          "Unsupported rounding mode: "
                                              << aRoundingMode << ". Only rounding mode "
                                              << RoundingMode::NearestTiesToEven
                                              << " is supported for this source data type and scaling factor < 1.");
                }
                else
                {
                    using ScalerT = Scale<ComputeT, true>;
                    const ScalerT scaler(aScaleFactor);
                    using divSrcDevCScale =
                        SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                   RoundingMode::None>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                }
            }
            else
            {
                switch (aRoundingMode)
                {
                    case mpp::RoundingMode::NearestTiesToEven:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                        const OpT op(aScaleFactor);
                        using divSrcDevCScale =
                            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize,
                                                                                          aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::NearestTiesAwayFromZero:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                        const OpT op(aScaleFactor);
                        using divSrcDevCScale =
                            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize,
                                                                                          aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardZero:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                        const OpT op(aScaleFactor);
                        using divSrcDevCScale =
                            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize,
                                                                                          aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardNegativeInfinity:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                        const OpT op(aScaleFactor);
                        using divSrcDevCScale =
                            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize,
                                                                                          aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardPositiveInfinity:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                        const OpT op(aScaleFactor);
                        using divSrcDevCScale =
                            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize,
                                                                                          aStreamCtx, functor);
                    }
                    break;
                    default:
                        throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
                }
            }
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeDivSrcDevCScale_For(typeSrcIsTypeDst)                                                         \
    template void                                                                                                      \
    InvokeDivSrcDevCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize,                   \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivSrcDevCScale(type)                                                               \
    InstantiateInvokeDivSrcDevCScale_For(Pixel##type##C1);                                                             \
    InstantiateInvokeDivSrcDevCScale_For(Pixel##type##C2);                                                             \
    InstantiateInvokeDivSrcDevCScale_For(Pixel##type##C3);                                                             \
    InstantiateInvokeDivSrcDevCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivSrcDevCScale(type)                                                             \
    InstantiateInvokeDivSrcDevCScale_For(Pixel##type##C1);                                                             \
    InstantiateInvokeDivSrcDevCScale_For(Pixel##type##C2);                                                             \
    InstantiateInvokeDivSrcDevCScale_For(Pixel##type##C3);                                                             \
    InstantiateInvokeDivSrcDevCScale_For(Pixel##type##C4);                                                             \
    InstantiateInvokeDivSrcDevCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Div<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                        RoundingMode::None, ComputeT, simdOP_t>;

            const mpp::Div<ComputeT, DstT> op;
            const simdOP_t opSIMD;

            const divInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcSIMD>(aSrcDst, aPitchSrcDst, aSize,
                                                                                aStreamCtx, functor);
        }
        else
        {
            using divInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, RoundingMode::None>;

            const mpp::Div<ComputeT, DstT> op;

            const divInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivInplaceSrc_For(typeSrcIsTypeDst)                                                           \
    template void                                                                                                      \
    InvokeDivInplaceSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(             \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInplaceSrc(type)                                                                 \
    InstantiateInvokeDivInplaceSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeDivInplaceSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeDivInplaceSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeDivInplaceSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInplaceSrc(type)                                                               \
    InstantiateInvokeDivInplaceSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeDivInplaceSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeDivInplaceSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeDivInplaceSrc_For(Pixel##type##C4);                                                               \
    InstantiateInvokeDivInplaceSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                              const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if constexpr (RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using divInplaceSrcScale =
                        InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                               RoundingMode::NearestTiesToEven>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                         aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using divInplaceSrcScale =
                        InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                               RoundingMode::NearestTiesAwayFromZero>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                         aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using divInplaceSrcScale =
                        InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                               RoundingMode::TowardZero>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                         aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using divInplaceSrcScale =
                        InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                               RoundingMode::TowardNegativeInfinity>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                         aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using divInplaceSrcScale =
                        InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                               RoundingMode::TowardPositiveInfinity>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                         aStreamCtx, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
        else
        {
            if (aScaleFactor < 1)
            {
                if (aRoundingMode != RoundingMode::NearestTiesToEven)
                {
                    throw INVALIDARGUMENT(aRoundingMode,
                                          "Unsupported rounding mode: "
                                              << aRoundingMode << ". Only rounding mode "
                                              << RoundingMode::NearestTiesToEven
                                              << " is supported for this source data type and scaling factor < 1.");
                }
                else
                {
                    using ScalerT = Scale<ComputeT, true>;
                    const ScalerT scaler(aScaleFactor);
                    using divInplaceSrcScale =
                        InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                               RoundingMode::None>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                         aStreamCtx, functor);
                }
            }
            else
            {
                switch (aRoundingMode)
                {
                    case mpp::RoundingMode::NearestTiesToEven:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                        const OpT op(aScaleFactor);
                        using divInplaceSrcScale =
                            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::NearestTiesAwayFromZero:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                        const OpT op(aScaleFactor);
                        using divInplaceSrcScale =
                            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardZero:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                        const OpT op(aScaleFactor);
                        using divInplaceSrcScale =
                            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardNegativeInfinity:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                        const OpT op(aScaleFactor);
                        using divInplaceSrcScale =
                            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardPositiveInfinity:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                        const OpT op(aScaleFactor);
                        using divInplaceSrcScale =
                            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    default:
                        throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
                }
            }
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeDivInplaceSrcScale_For(typeSrcIsTypeDst)                                                      \
    template void                                                                                                      \
    InvokeDivInplaceSrcScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInplaceSrcScale(type)                                                            \
    InstantiateInvokeDivInplaceSrcScale_For(Pixel##type##C1);                                                          \
    InstantiateInvokeDivInplaceSrcScale_For(Pixel##type##C2);                                                          \
    InstantiateInvokeDivInplaceSrcScale_For(Pixel##type##C3);                                                          \
    InstantiateInvokeDivInplaceSrcScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInplaceSrcScale(type)                                                          \
    InstantiateInvokeDivInplaceSrcScale_For(Pixel##type##C1);                                                          \
    InstantiateInvokeDivInplaceSrcScale_For(Pixel##type##C2);                                                          \
    InstantiateInvokeDivInplaceSrcScale_For(Pixel##type##C3);                                                          \
    InstantiateInvokeDivInplaceSrcScale_For(Pixel##type##C4);                                                          \
    InstantiateInvokeDivInplaceSrcScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                       const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Div<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divInplaceCSIMD = InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                           RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            const mpp::Div<ComputeT, DstT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            const divInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
        }
        else
        {
            using divInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, RoundingMode::None>;

            const mpp::Div<ComputeT, DstT> op;

            const divInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                          functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivInplaceC_For(typeSrcIsTypeDst)                                                             \
    template void InvokeDivInplaceC<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInplaceC(type)                                                                   \
    InstantiateInvokeDivInplaceC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeDivInplaceC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeDivInplaceC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeDivInplaceC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInplaceC(type)                                                                 \
    InstantiateInvokeDivInplaceC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeDivInplaceC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeDivInplaceC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeDivInplaceC_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeDivInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, double aScaleFactor,
                            mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;
        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if constexpr (RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using divInplaceCScale =
                        InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                    RoundingMode::NearestTiesToEven>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                       aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using divInplaceCScale =
                        InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                    RoundingMode::NearestTiesAwayFromZero>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                       aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using divInplaceCScale =
                        InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                    RoundingMode::TowardZero>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                       aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using divInplaceCScale =
                        InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                    RoundingMode::TowardNegativeInfinity>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                       aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using divInplaceCScale =
                        InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                    RoundingMode::TowardPositiveInfinity>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                       aStreamCtx, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
        else
        {
            if (aScaleFactor < 1)
            {
                if (aRoundingMode != RoundingMode::NearestTiesToEven)
                {
                    throw INVALIDARGUMENT(aRoundingMode,
                                          "Unsupported rounding mode: "
                                              << aRoundingMode << ". Only rounding mode "
                                              << RoundingMode::NearestTiesToEven
                                              << " is supported for this source data type and scaling factor < 1.");
                }
                else
                {
                    using ScalerT = Scale<ComputeT, true>;
                    const ScalerT scaler(aScaleFactor);
                    using divInplaceCScale =
                        InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                    RoundingMode::None>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                       aStreamCtx, functor);
                }
            }
            else
            {
                switch (aRoundingMode)
                {
                    case mpp::RoundingMode::NearestTiesToEven:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                        const OpT op(aScaleFactor);
                        using divInplaceCScale =
                            InplaceConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                           aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::NearestTiesAwayFromZero:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                        const OpT op(aScaleFactor);
                        using divInplaceCScale =
                            InplaceConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                           aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardZero:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                        const OpT op(aScaleFactor);
                        using divInplaceCScale =
                            InplaceConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                           aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardNegativeInfinity:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                        const OpT op(aScaleFactor);
                        using divInplaceCScale =
                            InplaceConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                           aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardPositiveInfinity:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                        const OpT op(aScaleFactor);
                        using divInplaceCScale =
                            InplaceConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                           aStreamCtx, functor);
                    }
                    break;
                    default:
                        throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
                }
            }
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeDivInplaceCScale_For(typeSrcIsTypeDst)                                                        \
    template void                                                                                                      \
    InvokeDivInplaceCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, double aScaleFactor,          \
        mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInplaceCScale(type)                                                              \
    InstantiateInvokeDivInplaceCScale_For(Pixel##type##C1);                                                            \
    InstantiateInvokeDivInplaceCScale_For(Pixel##type##C2);                                                            \
    InstantiateInvokeDivInplaceCScale_For(Pixel##type##C3);                                                            \
    InstantiateInvokeDivInplaceCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInplaceCScale(type)                                                            \
    InstantiateInvokeDivInplaceCScale_For(Pixel##type##C1);                                                            \
    InstantiateInvokeDivInplaceCScale_For(Pixel##type##C2);                                                            \
    InstantiateInvokeDivInplaceCScale_For(Pixel##type##C3);                                                            \
    InstantiateInvokeDivInplaceCScale_For(Pixel##type##C4);                                                            \
    InstantiateInvokeDivInplaceCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using divInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, RoundingMode::None>;

        const mpp::Div<ComputeT, DstT> op;

        const divInplaceDevC functor(aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivInplaceDevC_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeDivInplaceDevC<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInplaceDevC(type)                                                                \
    InstantiateInvokeDivInplaceDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeDivInplaceDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeDivInplaceDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeDivInplaceDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInplaceDevC(type)                                                              \
    InstantiateInvokeDivInplaceDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeDivInplaceDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeDivInplaceDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeDivInplaceDevC_For(Pixel##type##C4);                                                              \
    InstantiateInvokeDivInplaceDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceDevCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, double aScaleFactor,
                               mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if constexpr (RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using divInplaceDevCScale =
                        InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                       RoundingMode::NearestTiesToEven>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceDevCScale functor(aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                          aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using divInplaceDevCScale =
                        InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                       RoundingMode::NearestTiesAwayFromZero>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceDevCScale functor(aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                          aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using divInplaceDevCScale =
                        InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                       RoundingMode::TowardZero>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceDevCScale functor(aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                          aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using divInplaceDevCScale =
                        InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                       RoundingMode::TowardNegativeInfinity>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceDevCScale functor(aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                          aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using divInplaceDevCScale =
                        InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                       RoundingMode::TowardPositiveInfinity>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceDevCScale functor(aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                          aStreamCtx, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
        else
        {
            if (aScaleFactor < 1)
            {
                if (aRoundingMode != RoundingMode::NearestTiesToEven)
                {
                    throw INVALIDARGUMENT(aRoundingMode,
                                          "Unsupported rounding mode: "
                                              << aRoundingMode << ". Only rounding mode "
                                              << RoundingMode::NearestTiesToEven
                                              << " is supported for this source data type and scaling factor < 1.");
                }
                else
                {
                    using ScalerT = Scale<ComputeT, true>;
                    const ScalerT scaler(aScaleFactor);
                    using divInplaceDevCScale =
                        InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, ScalerT,
                                                       RoundingMode::None>;
                    const mpp::Div<ComputeT, DstT> op;
                    const divInplaceDevCScale functor(aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                          aStreamCtx, functor);
                }
            }
            else
            {
                switch (aRoundingMode)
                {
                    case mpp::RoundingMode::NearestTiesToEven:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                        const OpT op(aScaleFactor);
                        using divInplaceDevCScale =
                            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceDevCScale functor(aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::NearestTiesAwayFromZero:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                        const OpT op(aScaleFactor);
                        using divInplaceDevCScale =
                            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceDevCScale functor(aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardZero:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                        const OpT op(aScaleFactor);
                        using divInplaceDevCScale =
                            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceDevCScale functor(aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardNegativeInfinity:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                        const OpT op(aScaleFactor);
                        using divInplaceDevCScale =
                            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceDevCScale functor(aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardPositiveInfinity:
                    {
                        using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                        const OpT op(aScaleFactor);
                        using divInplaceDevCScale =
                            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceDevCScale functor(aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    default:
                        throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
                }
            }
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeDivInplaceDevCScale_For(typeSrcIsTypeDst)                                                     \
    template void                                                                                                      \
    InvokeDivInplaceDevCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, double aScaleFactor,          \
        mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInplaceDevCScale(type)                                                           \
    InstantiateInvokeDivInplaceDevCScale_For(Pixel##type##C1);                                                         \
    InstantiateInvokeDivInplaceDevCScale_For(Pixel##type##C2);                                                         \
    InstantiateInvokeDivInplaceDevCScale_For(Pixel##type##C3);                                                         \
    InstantiateInvokeDivInplaceDevCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInplaceDevCScale(type)                                                         \
    InstantiateInvokeDivInplaceDevCScale_For(Pixel##type##C1);                                                         \
    InstantiateInvokeDivInplaceDevCScale_For(Pixel##type##C2);                                                         \
    InstantiateInvokeDivInplaceDevCScale_For(Pixel##type##C3);                                                         \
    InstantiateInvokeDivInplaceDevCScale_For(Pixel##type##C4);                                                         \
    InstantiateInvokeDivInplaceDevCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                            const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::DivInv<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                                        RoundingMode::None, ComputeT, simdOP_t>;

            const mpp::DivInv<ComputeT, DstT> op;
            const simdOP_t opSIMD;

            const divInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcSIMD>(aSrcDst, aPitchSrcDst, aSize,
                                                                                aStreamCtx, functor);
        }
        else
        {
            using divInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, RoundingMode::None>;

            const mpp::DivInv<ComputeT, DstT> op;

            const divInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivInvInplaceSrc_For(typeSrcIsTypeDst)                                                        \
    template void                                                                                                      \
    InvokeDivInvInplaceSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInvInplaceSrc(type)                                                              \
    InstantiateInvokeDivInvInplaceSrc_For(Pixel##type##C1);                                                            \
    InstantiateInvokeDivInvInplaceSrc_For(Pixel##type##C2);                                                            \
    InstantiateInvokeDivInvInplaceSrc_For(Pixel##type##C3);                                                            \
    InstantiateInvokeDivInvInplaceSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInvInplaceSrc(type)                                                            \
    InstantiateInvokeDivInvInplaceSrc_For(Pixel##type##C1);                                                            \
    InstantiateInvokeDivInvInplaceSrc_For(Pixel##type##C2);                                                            \
    InstantiateInvokeDivInvInplaceSrc_For(Pixel##type##C3);                                                            \
    InstantiateInvokeDivInvInplaceSrc_For(Pixel##type##C4);                                                            \
    InstantiateInvokeDivInvInplaceSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                                 double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize,
                                 const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if constexpr (RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using divInplaceSrcScale =
                        InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                               RoundingMode::NearestTiesToEven>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                         aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using divInplaceSrcScale =
                        InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                               RoundingMode::NearestTiesAwayFromZero>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                         aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using divInplaceSrcScale =
                        InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                               RoundingMode::TowardZero>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                         aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using divInplaceSrcScale =
                        InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                               RoundingMode::TowardNegativeInfinity>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                         aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using divInplaceSrcScale =
                        InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                               RoundingMode::TowardPositiveInfinity>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                         aStreamCtx, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
        else
        {
            if (aScaleFactor < 1)
            {
                if (aRoundingMode != RoundingMode::NearestTiesToEven)
                {
                    throw INVALIDARGUMENT(aRoundingMode,
                                          "Unsupported rounding mode: "
                                              << aRoundingMode << ". Only rounding mode "
                                              << RoundingMode::NearestTiesToEven
                                              << " is supported for this source data type and scaling factor < 1.");
                }
                else
                {
                    using ScalerT = Scale<ComputeT, true>;
                    const ScalerT scaler(aScaleFactor);
                    using divInplaceSrcScale =
                        InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                               RoundingMode::None>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                         aStreamCtx, functor);
                }
            }
            else
            {
                switch (aRoundingMode)
                {
                    case mpp::RoundingMode::NearestTiesToEven:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                        const OpT op(aScaleFactor);
                        using divInplaceSrcScale =
                            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::NearestTiesAwayFromZero:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                        const OpT op(aScaleFactor);
                        using divInplaceSrcScale =
                            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardZero:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                        const OpT op(aScaleFactor);
                        using divInplaceSrcScale =
                            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardNegativeInfinity:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                        const OpT op(aScaleFactor);
                        using divInplaceSrcScale =
                            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardPositiveInfinity:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                        const OpT op(aScaleFactor);
                        using divInplaceSrcScale =
                            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    default:
                        throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
                }
            }
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeDivInvInplaceSrcScale_For(typeSrcIsTypeDst)                                                   \
    template void                                                                                                      \
    InvokeDivInvInplaceSrcScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        double aScaleFactor, mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInvInplaceSrcScale(type)                                                         \
    InstantiateInvokeDivInvInplaceSrcScale_For(Pixel##type##C1);                                                       \
    InstantiateInvokeDivInvInplaceSrcScale_For(Pixel##type##C2);                                                       \
    InstantiateInvokeDivInvInplaceSrcScale_For(Pixel##type##C3);                                                       \
    InstantiateInvokeDivInvInplaceSrcScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScale(type)                                                       \
    InstantiateInvokeDivInvInplaceSrcScale_For(Pixel##type##C1);                                                       \
    InstantiateInvokeDivInvInplaceSrcScale_For(Pixel##type##C2);                                                       \
    InstantiateInvokeDivInvInplaceSrcScale_For(Pixel##type##C3);                                                       \
    InstantiateInvokeDivInvInplaceSrcScale_For(Pixel##type##C4);                                                       \
    InstantiateInvokeDivInvInplaceSrcScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::DivInv<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divInplaceCSIMD = InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                                           RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            const mpp::DivInv<ComputeT, DstT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            const divInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
        }
        else
        {
            using divInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, RoundingMode::None>;

            const mpp::DivInv<ComputeT, DstT> op;

            const divInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                          functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivInvInplaceC_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeDivInvInplaceC<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInvInplaceC(type)                                                                \
    InstantiateInvokeDivInvInplaceC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeDivInvInplaceC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeDivInvInplaceC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeDivInvInplaceC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInvInplaceC(type)                                                              \
    InstantiateInvokeDivInvInplaceC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeDivInvInplaceC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeDivInvInplaceC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeDivInvInplaceC_For(Pixel##type##C4);                                                              \
    InstantiateInvokeDivInvInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, double aScaleFactor,
                               mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if constexpr (RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using divInplaceCScale =
                        InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                                    RoundingMode::NearestTiesToEven>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                       aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using divInplaceCScale =
                        InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                                    RoundingMode::NearestTiesAwayFromZero>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                       aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using divInplaceCScale =
                        InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                                    RoundingMode::TowardZero>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                       aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using divInplaceCScale =
                        InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                                    RoundingMode::TowardNegativeInfinity>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                       aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using divInplaceCScale =
                        InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                                    RoundingMode::TowardPositiveInfinity>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                       aStreamCtx, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
        else
        {
            if (aScaleFactor < 1)
            {
                if (aRoundingMode != RoundingMode::NearestTiesToEven)
                {
                    throw INVALIDARGUMENT(aRoundingMode,
                                          "Unsupported rounding mode: "
                                              << aRoundingMode << ". Only rounding mode "
                                              << RoundingMode::NearestTiesToEven
                                              << " is supported for this source data type and scaling factor < 1.");
                }
                else
                {
                    using ScalerT = Scale<ComputeT, true>;
                    const ScalerT scaler(aScaleFactor);
                    using divInplaceCScale =
                        InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                                    RoundingMode::None>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                       aStreamCtx, functor);
                }
            }
            else
            {
                switch (aRoundingMode)
                {
                    case mpp::RoundingMode::NearestTiesToEven:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                        const OpT op(aScaleFactor);
                        using divInplaceCScale =
                            InplaceConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                           aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::NearestTiesAwayFromZero:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                        const OpT op(aScaleFactor);
                        using divInplaceCScale =
                            InplaceConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                           aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardZero:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                        const OpT op(aScaleFactor);
                        using divInplaceCScale =
                            InplaceConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                           aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardNegativeInfinity:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                        const OpT op(aScaleFactor);
                        using divInplaceCScale =
                            InplaceConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                           aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardPositiveInfinity:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                        const OpT op(aScaleFactor);
                        using divInplaceCScale =
                            InplaceConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                           aStreamCtx, functor);
                    }
                    break;
                    default:
                        throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
                }
            }
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeDivInvInplaceCScale_For(typeSrcIsTypeDst)                                                     \
    template void                                                                                                      \
    InvokeDivInvInplaceCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, double aScaleFactor,          \
        mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInvInplaceCScale(type)                                                           \
    InstantiateInvokeDivInvInplaceCScale_For(Pixel##type##C1);                                                         \
    InstantiateInvokeDivInvInplaceCScale_For(Pixel##type##C2);                                                         \
    InstantiateInvokeDivInvInplaceCScale_For(Pixel##type##C3);                                                         \
    InstantiateInvokeDivInvInplaceCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInvInplaceCScale(type)                                                         \
    InstantiateInvokeDivInvInplaceCScale_For(Pixel##type##C1);                                                         \
    InstantiateInvokeDivInvInplaceCScale_For(Pixel##type##C2);                                                         \
    InstantiateInvokeDivInvInplaceCScale_For(Pixel##type##C3);                                                         \
    InstantiateInvokeDivInvInplaceCScale_For(Pixel##type##C4);                                                         \
    InstantiateInvokeDivInvInplaceCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                             const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using divInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, RoundingMode::None>;

        const mpp::DivInv<ComputeT, DstT> op;

        const divInplaceDevC functor(aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivInvInplaceDevC_For(typeSrcIsTypeDst)                                                       \
    template void                                                                                                      \
    InvokeDivInvInplaceDevC<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(         \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInvInplaceDevC(type)                                                             \
    InstantiateInvokeDivInvInplaceDevC_For(Pixel##type##C1);                                                           \
    InstantiateInvokeDivInvInplaceDevC_For(Pixel##type##C2);                                                           \
    InstantiateInvokeDivInvInplaceDevC_For(Pixel##type##C3);                                                           \
    InstantiateInvokeDivInvInplaceDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInvInplaceDevC(type)                                                           \
    InstantiateInvokeDivInvInplaceDevC_For(Pixel##type##C1);                                                           \
    InstantiateInvokeDivInvInplaceDevC_For(Pixel##type##C2);                                                           \
    InstantiateInvokeDivInvInplaceDevC_For(Pixel##type##C3);                                                           \
    InstantiateInvokeDivInvInplaceDevC_For(Pixel##type##C4);                                                           \
    InstantiateInvokeDivInvInplaceDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceDevCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, double aScaleFactor,
                                  mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if constexpr (RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using divInplaceDevCScale =
                        InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                                       RoundingMode::NearestTiesToEven>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceDevCScale functor(aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                          aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using divInplaceDevCScale =
                        InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                                       RoundingMode::NearestTiesAwayFromZero>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceDevCScale functor(aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                          aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using divInplaceDevCScale =
                        InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                                       RoundingMode::TowardZero>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceDevCScale functor(aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                          aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using divInplaceDevCScale =
                        InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                                       RoundingMode::TowardNegativeInfinity>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceDevCScale functor(aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                          aStreamCtx, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using divInplaceDevCScale =
                        InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                                       RoundingMode::TowardPositiveInfinity>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceDevCScale functor(aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                          aStreamCtx, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
        else
        {
            if (aScaleFactor < 1)
            {
                if (aRoundingMode != RoundingMode::NearestTiesToEven)
                {
                    throw INVALIDARGUMENT(aRoundingMode,
                                          "Unsupported rounding mode: "
                                              << aRoundingMode << ". Only rounding mode "
                                              << RoundingMode::NearestTiesToEven
                                              << " is supported for this source data type and scaling factor < 1.");
                }
                else
                {
                    using ScalerT = Scale<ComputeT, true>;
                    const ScalerT scaler(aScaleFactor);
                    using divInplaceDevCScale =
                        InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, ScalerT,
                                                       RoundingMode::None>;
                    const mpp::DivInv<ComputeT, DstT> op;
                    const divInplaceDevCScale functor(aConst, op, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                          aStreamCtx, functor);
                }
            }
            else
            {
                switch (aRoundingMode)
                {
                    case mpp::RoundingMode::NearestTiesToEven:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                        const OpT op(aScaleFactor);
                        using divInplaceDevCScale =
                            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceDevCScale functor(aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::NearestTiesAwayFromZero:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                        const OpT op(aScaleFactor);
                        using divInplaceDevCScale =
                            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceDevCScale functor(aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardZero:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                        const OpT op(aScaleFactor);
                        using divInplaceDevCScale =
                            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceDevCScale functor(aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardNegativeInfinity:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                        const OpT op(aScaleFactor);
                        using divInplaceDevCScale =
                            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceDevCScale functor(aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    case mpp::RoundingMode::TowardPositiveInfinity:
                    {
                        using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                        const OpT op(aScaleFactor);
                        using divInplaceDevCScale =
                            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, OpT, RoundingMode::None>;
                        const divInplaceDevCScale functor(aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                            aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
                    }
                    break;
                    default:
                        throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
                }
            }
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeDivInvInplaceDevCScale_For(typeSrcIsTypeDst)                                                  \
    template void                                                                                                      \
    InvokeDivInvInplaceDevCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, double aScaleFactor,          \
        mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInvInplaceDevCScale(type)                                                        \
    InstantiateInvokeDivInvInplaceDevCScale_For(Pixel##type##C1);                                                      \
    InstantiateInvokeDivInvInplaceDevCScale_For(Pixel##type##C2);                                                      \
    InstantiateInvokeDivInvInplaceDevCScale_For(Pixel##type##C3);                                                      \
    InstantiateInvokeDivInvInplaceDevCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScale(type)                                                      \
    InstantiateInvokeDivInvInplaceDevCScale_For(Pixel##type##C1);                                                      \
    InstantiateInvokeDivInvInplaceDevCScale_For(Pixel##type##C2);                                                      \
    InstantiateInvokeDivInvInplaceDevCScale_For(Pixel##type##C3);                                                      \
    InstantiateInvokeDivInvInplaceDevCScale_For(Pixel##type##C4);                                                      \
    InstantiateInvokeDivInvInplaceDevCScale_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
