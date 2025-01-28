#if OPP_ENABLE_CUDA_BACKEND

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
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                     size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Div<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divSrcSrcSIMD = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>, RoundingMode::None,
                                                ComputeT, simdOP_t>;

            Div<ComputeT> op;
            simdOP_t opSIMD;

            divSrcSrcSIMD functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcSIMD>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                            functor);
        }
        else
        {
            using divSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>, RoundingMode::None>;

            Div<ComputeT> op;

            divSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeDivSrcSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

// ForAllChannelsWithAlpha(8u);
// ForAllChannelsWithAlpha(8s);

// ForAllChannelsWithAlpha(16u);
// ForAllChannelsWithAlpha(16s);

// ForAllChannelsWithAlpha(32u);
// ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcSrcScale(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode,
                          const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                          RoundingMode::NearestTiesToEven>;
                Div<ComputeT> op;
                divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                 functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                Div<ComputeT> op;
                divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                 functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divSrcSrcScale =
                    SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>, RoundingMode::TowardZero>;
                Div<ComputeT> op;
                divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                 functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                          RoundingMode::TowardNegativeInfinity>;
                Div<ComputeT> op;
                divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                 functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                          RoundingMode::TowardPositiveInfinity>;
                Div<ComputeT> op;
                divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                 functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivSrcSrcScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst,                                                                      \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, opp::RoundingMode aRoundingMode,     \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                   const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Div<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divSrcCSIMD = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                   RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            Div<ComputeT> op;
            simdOP_t opSIMD;
            Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            divSrcCSIMD functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCSIMD>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using divSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>, RoundingMode::None>;

            Div<ComputeT> op;

            divSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeDivSrcC<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

// ForAllChannelsWithAlpha(8u);
// ForAllChannelsWithAlpha(8s);

// ForAllChannelsWithAlpha(16u);
// ForAllChannelsWithAlpha(16s);

// ForAllChannelsWithAlpha(32u);
// ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                        scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode, const Size2D &aSize,
                        const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                             RoundingMode::NearestTiesToEven>;
                Div<ComputeT> op;
                divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                             RoundingMode::NearestTiesAwayFromZero>;
                Div<ComputeT> op;
                divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                             RoundingMode::TowardZero>;
                Div<ComputeT> op;
                divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                             RoundingMode::TowardNegativeInfinity>;
                Div<ComputeT> op;
                divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                             RoundingMode::TowardPositiveInfinity>;
                Div<ComputeT> op;
                divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                               functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivSrcCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(              \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,                    \
        opp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using divSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>, RoundingMode::None>;

        Div<ComputeT> op;

        divSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeDivSrcDevC<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(  \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

// ForAllChannelsWithAlpha(8u);
// ForAllChannelsWithAlpha(8s);

// ForAllChannelsWithAlpha(16u);
// ForAllChannelsWithAlpha(16s);

// ForAllChannelsWithAlpha(32u);
// ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcDevCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                           scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode, const Size2D &aSize,
                           const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                                   RoundingMode::NearestTiesToEven>;
                Div<ComputeT> op;
                divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                                   RoundingMode::NearestTiesAwayFromZero>;
                Div<ComputeT> op;
                divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                                   RoundingMode::TowardZero>;
                Div<ComputeT> op;
                divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                                   RoundingMode::TowardNegativeInfinity>;
                Div<ComputeT> op;
                divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                                   RoundingMode::TowardPositiveInfinity>;
                Div<ComputeT> op;
                divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divSrcDevCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivSrcDevCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,                    \
        opp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Div<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                        RoundingMode::None, ComputeT, simdOP_t>;

            Div<ComputeT> op;
            simdOP_t opSIMD;

            divInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcSIMD>(aSrcDst, aPitchSrcDst, aSize,
                                                                                aStreamCtx, functor);
        }
        else
        {
            using divInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>, RoundingMode::None>;

            Div<ComputeT> op;

            divInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInplaceSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(             \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

// ForAllChannelsWithAlpha(8u);
// ForAllChannelsWithAlpha(8s);

// ForAllChannelsWithAlpha(16u);
// ForAllChannelsWithAlpha(16s);

// ForAllChannelsWithAlpha(32u);
// ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode,
                              const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                                  RoundingMode::NearestTiesToEven>;
                Div<ComputeT> op;
                divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                     aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                                  RoundingMode::NearestTiesAwayFromZero>;
                Div<ComputeT> op;
                divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                     aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                                  RoundingMode::TowardZero>;
                Div<ComputeT> op;
                divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                     aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                                  RoundingMode::TowardNegativeInfinity>;
                Div<ComputeT> op;
                divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                     aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT>,
                                                                  RoundingMode::TowardPositiveInfinity>;
                Div<ComputeT> op;
                divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                     aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInplaceSrcScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, opp::RoundingMode aRoundingMode,     \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                       const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Div<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divInplaceCSIMD = InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>,
                                                           RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            Div<ComputeT> op;
            simdOP_t opSIMD;
            Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            divInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
        }
        else
        {
            using divInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>, RoundingMode::None>;

            Div<ComputeT> op;

            divInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                          functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeDivInplaceC<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

// ForAllChannelsWithAlpha(8u);
// ForAllChannelsWithAlpha(8s);

// ForAllChannelsWithAlpha(16u);
// ForAllChannelsWithAlpha(16s);

// ForAllChannelsWithAlpha(32u);
// ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst,
                            scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode, const Size2D &aSize,
                            const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;
        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>,
                                                                     RoundingMode::NearestTiesToEven>;
                Div<ComputeT> op;
                divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                   aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>,
                                                                     RoundingMode::NearestTiesAwayFromZero>;
                Div<ComputeT> op;
                divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                   aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>,
                                                                     RoundingMode::TowardZero>;
                Div<ComputeT> op;
                divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                   aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>,
                                                                     RoundingMode::TowardNegativeInfinity>;
                Div<ComputeT> op;
                divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                   aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>,
                                                                     RoundingMode::TowardPositiveInfinity>;
                Div<ComputeT> op;
                divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                   aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInplaceCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst,                               \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, opp::RoundingMode aRoundingMode,     \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using divInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>, RoundingMode::None>;

        Div<ComputeT> op;

        divInplaceDevC functor(aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInplaceDevC<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

// ForAllChannelsWithAlpha(8u);
// ForAllChannelsWithAlpha(8s);

// ForAllChannelsWithAlpha(16u);
// ForAllChannelsWithAlpha(16s);

// ForAllChannelsWithAlpha(32u);
// ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceDevCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst,
                               scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode,
                               const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>,
                                                   RoundingMode::NearestTiesToEven>;
                Div<ComputeT> op;
                divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                      aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>,
                                                   RoundingMode::NearestTiesAwayFromZero>;
                Div<ComputeT> op;
                divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                      aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>,
                                                   RoundingMode::TowardZero>;
                Div<ComputeT> op;
                divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                      aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>,
                                                   RoundingMode::TowardNegativeInfinity>;
                Div<ComputeT> op;
                divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                      aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT>,
                                                   RoundingMode::TowardPositiveInfinity>;
                Div<ComputeT> op;
                divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                      aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInplaceDevCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst,                               \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, opp::RoundingMode aRoundingMode,     \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                            const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::DivInv<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT>,
                                                        RoundingMode::None, ComputeT, simdOP_t>;

            DivInv<ComputeT> op;
            simdOP_t opSIMD;

            divInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcSIMD>(aSrcDst, aPitchSrcDst, aSize,
                                                                                aStreamCtx, functor);
        }
        else
        {
            using divInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT>, RoundingMode::None>;

            DivInv<ComputeT> op;

            divInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInvInplaceSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

// ForAllChannelsWithAlpha(8u);
// ForAllChannelsWithAlpha(8s);

// ForAllChannelsWithAlpha(16u);
// ForAllChannelsWithAlpha(16s);

// ForAllChannelsWithAlpha(32u);
// ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                                 scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode,
                                 const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT>,
                                           RoundingMode::NearestTiesToEven>;
                DivInv<ComputeT> op;
                divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                     aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT>,
                                           RoundingMode::NearestTiesAwayFromZero>;
                DivInv<ComputeT> op;
                divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                     aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                                  opp::DivInv<ComputeT>, RoundingMode::TowardZero>;
                DivInv<ComputeT> op;
                divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                     aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT>,
                                           RoundingMode::TowardNegativeInfinity>;
                DivInv<ComputeT> op;
                divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                     aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT>,
                                           RoundingMode::TowardPositiveInfinity>;
                DivInv<ComputeT> op;
                divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                     aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInvInplaceSrcScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, opp::RoundingMode aRoundingMode,     \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::DivInv<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divInplaceCSIMD = InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>,
                                                           RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            DivInv<ComputeT> op;
            simdOP_t opSIMD;
            Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            divInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
        }
        else
        {
            using divInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>, RoundingMode::None>;

            DivInv<ComputeT> op;

            divInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                          functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInvInplaceC<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

// ForAllChannelsWithAlpha(8u);
// ForAllChannelsWithAlpha(8s);

// ForAllChannelsWithAlpha(16u);
// ForAllChannelsWithAlpha(16s);

// ForAllChannelsWithAlpha(32u);
// ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst,
                               scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode,
                               const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>,
                                                                     RoundingMode::NearestTiesToEven>;
                DivInv<ComputeT> op;
                divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                   aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>,
                                                                     RoundingMode::NearestTiesAwayFromZero>;
                DivInv<ComputeT> op;
                divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                   aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>,
                                                                     RoundingMode::TowardZero>;
                DivInv<ComputeT> op;
                divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                   aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>,
                                                                     RoundingMode::TowardNegativeInfinity>;
                DivInv<ComputeT> op;
                divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                   aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>,
                                                                     RoundingMode::TowardPositiveInfinity>;
                DivInv<ComputeT> op;
                divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                   aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInvInplaceCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst,                               \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, opp::RoundingMode aRoundingMode,     \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                             const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using divInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>, RoundingMode::None>;

        DivInv<ComputeT> op;

        divInplaceDevC functor(aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInvInplaceDevC<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(         \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

// ForAllChannelsWithAlpha(8u);
// ForAllChannelsWithAlpha(8s);

// ForAllChannelsWithAlpha(16u);
// ForAllChannelsWithAlpha(16s);

// ForAllChannelsWithAlpha(32u);
// ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

// ForAllChannelsNoAlpha(16sc);
// ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceDevCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst,
                                  scalefactor_t<ComputeT> aScaleFactor, opp::RoundingMode aRoundingMode,
                                  const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>,
                                                   RoundingMode::NearestTiesToEven>;
                DivInv<ComputeT> op;
                divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                      aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>,
                                                   RoundingMode::NearestTiesAwayFromZero>;
                DivInv<ComputeT> op;
                divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                      aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>,
                                                   RoundingMode::TowardZero>;
                DivInv<ComputeT> op;
                divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                      aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>,
                                                   RoundingMode::TowardNegativeInfinity>;
                DivInv<ComputeT> op;
                divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                      aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT>,
                                                   RoundingMode::TowardPositiveInfinity>;
                DivInv<ComputeT> op;
                divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, divInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                      aStreamCtx, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInvInplaceDevCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst,                               \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, opp::RoundingMode aRoundingMode,     \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
