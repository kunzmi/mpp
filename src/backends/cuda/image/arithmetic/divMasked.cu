#if OPP_ENABLE_CUDA_BACKEND

#include "divMasked.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelMaskedKernel.h>
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
void InvokeDivSrcSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                         const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Div<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divSrcSrcSIMD = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                RoundingMode::None, ComputeT, simdOP_t>;

            const opp::Div<ComputeT, DstT> op;
            const simdOP_t opSIMD;

            const divSrcSrcSIMD functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrcSIMD>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                  aSize, aStreamCtx, functor);
        }
        else
        {
            using divSrcSrc =
                SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>, RoundingMode::None>;

            const opp::Div<ComputeT, DstT> op;

            const divSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrc>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                              aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivSrcSrcMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(             \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                   \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
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
void InvokeDivSrcSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                              const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst,
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
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                          RoundingMode::NearestTiesToEven>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrcScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrcScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                          RoundingMode::TowardZero>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrcScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                          RoundingMode::TowardNegativeInfinity>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrcScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                          RoundingMode::TowardPositiveInfinity>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrcScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
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
    InvokeDivSrcSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                   \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
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
void InvokeDivSrcCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                       const SrcT &aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                       const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Div<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divSrcCSIMD = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                   RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            const opp::Div<ComputeT, DstT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            const divSrcCSIMD functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcCSIMD>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                aSize, aStreamCtx, functor);
        }
        else
        {
            using divSrcC =
                SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>, RoundingMode::None>;

            const opp::Div<ComputeT, DstT> op;

            const divSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                            aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeDivSrcCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
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
void InvokeDivSrcCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                            const SrcT &aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                            opp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                             RoundingMode::NearestTiesToEven>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                     aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                             RoundingMode::NearestTiesAwayFromZero>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                     aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                             RoundingMode::TowardZero>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                     aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                             RoundingMode::TowardNegativeInfinity>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                     aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                             RoundingMode::TowardPositiveInfinity>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                     aSize, aStreamCtx, functor);
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
    InvokeDivSrcCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst,                                      \
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
void InvokeDivSrcDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                          const SrcT *aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using divSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>, RoundingMode::None>;

        const opp::Div<ComputeT, DstT> op;

        const divSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcDevC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                           aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivSrcDevCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
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
void InvokeDivSrcDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                               const SrcT *aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                               opp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divSrcDevCScale =
                    SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                               RoundingMode::NearestTiesToEven>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcDevCScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcDevCScale =
                    SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                               RoundingMode::NearestTiesAwayFromZero>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcDevCScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                                   opp::Div<ComputeT, DstT>, RoundingMode::TowardZero>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcDevCScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcDevCScale =
                    SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                               RoundingMode::TowardNegativeInfinity>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcDevCScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcDevCScale =
                    SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                               RoundingMode::TowardPositiveInfinity>;
                const opp::Div<ComputeT, DstT> op;
                const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcDevCScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
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
    InvokeDivSrcDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst,                                      \
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
void InvokeDivInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                             const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Div<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                        RoundingMode::None, ComputeT, simdOP_t>;

            const opp::Div<ComputeT, DstT> op;
            const simdOP_t opSIMD;

            const divInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using divInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>, RoundingMode::None>;

            const opp::Div<ComputeT, DstT> op;

            const divInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrc>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInplaceSrcMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(         \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx);

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
void InvokeDivInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                  const SrcT *aSrc2, size_t aPitchSrc2, scalefactor_t<ComputeT> aScaleFactor,
                                  opp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
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
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                           RoundingMode::NearestTiesToEven>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                           RoundingMode::NearestTiesAwayFromZero>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                                  opp::Div<ComputeT, DstT>, RoundingMode::TowardZero>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                           RoundingMode::TowardNegativeInfinity>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                           RoundingMode::TowardPositiveInfinity>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
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
    InvokeDivInplaceSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,                                                              \
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
void InvokeDivInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                           const SrcT &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Div<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divInplaceCSIMD = InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                           RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            const opp::Div<ComputeT, DstT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            const divInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using divInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>, RoundingMode::None>;

            const opp::Div<ComputeT, DstT> op;

            const divInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceC>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInplaceCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

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
void InvokeDivInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT &aConst, scalefactor_t<ComputeT> aScaleFactor,
                                opp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                RoundingMode::NearestTiesToEven>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                RoundingMode::NearestTiesAwayFromZero>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                RoundingMode::TowardZero>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                RoundingMode::TowardNegativeInfinity>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                RoundingMode::TowardPositiveInfinity>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
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
    InvokeDivInplaceCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(      \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
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
void InvokeDivInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                              const SrcT *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using divInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>, RoundingMode::None>;

        const opp::Div<ComputeT, DstT> op;

        const divInplaceDevC functor(aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevC>(aMask, aPitchMask, aSrcDst, aPitchSrcDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInplaceDevCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

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
void InvokeDivInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                   const SrcT *aConst, scalefactor_t<ComputeT> aScaleFactor,
                                   opp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
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
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                   RoundingMode::NearestTiesToEven>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                   RoundingMode::NearestTiesAwayFromZero>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                   RoundingMode::TowardZero>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                   RoundingMode::TowardNegativeInfinity>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Div<ComputeT, DstT>,
                                                   RoundingMode::TowardPositiveInfinity>;
                const opp::Div<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
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
    InvokeDivInplaceDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
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
void InvokeDivInvInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::DivInv<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                                        RoundingMode::None, ComputeT, simdOP_t>;

            const opp::DivInv<ComputeT, DstT> op;
            const simdOP_t opSIMD;

            const divInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using divInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT, DstT>, RoundingMode::None>;

            const opp::DivInv<ComputeT, DstT> op;

            const divInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrc>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInvInplaceSrcMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(      \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx);

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
void InvokeDivInvInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                     const SrcT *aSrc2, size_t aPitchSrc2, scalefactor_t<ComputeT> aScaleFactor,
                                     opp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
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
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                           RoundingMode::NearestTiesToEven>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                           RoundingMode::NearestTiesAwayFromZero>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                           RoundingMode::TowardZero>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                           RoundingMode::TowardNegativeInfinity>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                           RoundingMode::TowardPositiveInfinity>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
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
    InvokeDivInvInplaceSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,                                                              \
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
void InvokeDivInvInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                              const SrcT &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::DivInv<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using divInplaceCSIMD = InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                                           RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            const opp::DivInv<ComputeT, DstT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            const divInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using divInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>, RoundingMode::None>;

            const opp::DivInv<ComputeT, DstT> op;

            const divInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceC>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInvInplaceCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

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
void InvokeDivInvInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                   const SrcT &aConst, scalefactor_t<ComputeT> aScaleFactor,
                                   opp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                                RoundingMode::NearestTiesToEven>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                                RoundingMode::NearestTiesAwayFromZero>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                                RoundingMode::TowardZero>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                                RoundingMode::TowardNegativeInfinity>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                                RoundingMode::TowardPositiveInfinity>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
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
    InvokeDivInvInplaceCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
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
void InvokeDivInvInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                 const SrcT *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using divInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>, RoundingMode::None>;

        const opp::DivInv<ComputeT, DstT> op;

        const divInplaceDevC functor(aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevC>(aMask, aPitchMask, aSrcDst, aPitchSrcDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeDivInvInplaceDevCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

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
void InvokeDivInvInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                      const SrcT *aConst, scalefactor_t<ComputeT> aScaleFactor,
                                      opp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
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
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                                   RoundingMode::NearestTiesToEven>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                                   RoundingMode::NearestTiesAwayFromZero>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                                   RoundingMode::TowardZero>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                                   RoundingMode::TowardNegativeInfinity>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::DivInv<ComputeT, DstT>,
                                                   RoundingMode::TowardPositiveInfinity>;
                const opp::DivInv<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
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
    template void InvokeDivInvInplaceDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>,     \
                                                   typeSrcIsTypeDst>(                                                  \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
