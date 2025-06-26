#if MPP_ENABLE_CUDA_BACKEND

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
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                         const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
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

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrcSIMD>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                  aSize, aStreamCtx, functor);
        }
        else
        {
            using divSrcSrc =
                SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, RoundingMode::None>;

            const mpp::Div<ComputeT, DstT> op;

            const divSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrc>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                              aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivSrcSrcMask_For(typeSrcIsTypeDst)                                                           \
    template void                                                                                                      \
    InvokeDivSrcSrcMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(             \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                   \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivSrcSrcMask(type)                                                                 \
    InstantiateInvokeDivSrcSrcMask_For(Pixel##type##C1);                                                               \
    InstantiateInvokeDivSrcSrcMask_For(Pixel##type##C2);                                                               \
    InstantiateInvokeDivSrcSrcMask_For(Pixel##type##C3);                                                               \
    InstantiateInvokeDivSrcSrcMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivSrcSrcMask(type)                                                               \
    InstantiateInvokeDivSrcSrcMask_For(Pixel##type##C1);                                                               \
    InstantiateInvokeDivSrcSrcMask_For(Pixel##type##C2);                                                               \
    InstantiateInvokeDivSrcSrcMask_For(Pixel##type##C3);                                                               \
    InstantiateInvokeDivSrcSrcMask_For(Pixel##type##C4);                                                               \
    InstantiateInvokeDivSrcSrcMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                              const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst,
                              scalefactor_t<ComputeT> aScaleFactor, mpp::RoundingMode aRoundingMode,
                              const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                          RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT, DstT> op;
                const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrcScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT, DstT> op;
                const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrcScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                          RoundingMode::TowardZero>;
                const mpp::Div<ComputeT, DstT> op;
                const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrcScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                          RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT, DstT> op;
                const divSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcSrcScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                          RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT, DstT> op;
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
#define InstantiateInvokeDivSrcSrcScaleMask_For(typeSrcIsTypeDst)                                                      \
    template void                                                                                                      \
    InvokeDivSrcSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                   \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, mpp::RoundingMode aRoundingMode,     \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivSrcSrcScaleMask(type)                                                            \
    InstantiateInvokeDivSrcSrcScaleMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeDivSrcSrcScaleMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeDivSrcSrcScaleMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeDivSrcSrcScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivSrcSrcScaleMask(type)                                                          \
    InstantiateInvokeDivSrcSrcScaleMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeDivSrcSrcScaleMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeDivSrcSrcScaleMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeDivSrcSrcScaleMask_For(Pixel##type##C4);                                                          \
    InstantiateInvokeDivSrcSrcScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                       const SrcT &aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                       const StreamCtx &aStreamCtx)
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

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcCSIMD>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                aSize, aStreamCtx, functor);
        }
        else
        {
            using divSrcC =
                SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, RoundingMode::None>;

            const mpp::Div<ComputeT, DstT> op;

            const divSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                            aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivSrcCMask_For(typeSrcIsTypeDst)                                                             \
    template void InvokeDivSrcCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivSrcCMask(type)                                                                   \
    InstantiateInvokeDivSrcCMask_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeDivSrcCMask_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeDivSrcCMask_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeDivSrcCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivSrcCMask(type)                                                                 \
    InstantiateInvokeDivSrcCMask_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeDivSrcCMask_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeDivSrcCMask_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeDivSrcCMask_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeDivSrcCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                            const SrcT &aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                            mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                             RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT, DstT> op;
                const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                     aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                             RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT, DstT> op;
                const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                     aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                             RoundingMode::TowardZero>;
                const mpp::Div<ComputeT, DstT> op;
                const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                     aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                             RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT, DstT> op;
                const divSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                     aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                             RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT, DstT> op;
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
#define InstantiateInvokeDivSrcCScaleMask_For(typeSrcIsTypeDst)                                                        \
    template void                                                                                                      \
    InvokeDivSrcCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, mpp::RoundingMode aRoundingMode,     \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivSrcCScaleMask(type)                                                              \
    InstantiateInvokeDivSrcCScaleMask_For(Pixel##type##C1);                                                            \
    InstantiateInvokeDivSrcCScaleMask_For(Pixel##type##C2);                                                            \
    InstantiateInvokeDivSrcCScaleMask_For(Pixel##type##C3);                                                            \
    InstantiateInvokeDivSrcCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivSrcCScaleMask(type)                                                            \
    InstantiateInvokeDivSrcCScaleMask_For(Pixel##type##C1);                                                            \
    InstantiateInvokeDivSrcCScaleMask_For(Pixel##type##C2);                                                            \
    InstantiateInvokeDivSrcCScaleMask_For(Pixel##type##C3);                                                            \
    InstantiateInvokeDivSrcCScaleMask_For(Pixel##type##C4);                                                            \
    InstantiateInvokeDivSrcCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                          const SrcT *aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using divSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, RoundingMode::None>;

        const mpp::Div<ComputeT, DstT> op;

        const divSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcDevC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                           aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivSrcDevCMask_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeDivSrcDevCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivSrcDevCMask(type)                                                                \
    InstantiateInvokeDivSrcDevCMask_For(Pixel##type##C1);                                                              \
    InstantiateInvokeDivSrcDevCMask_For(Pixel##type##C2);                                                              \
    InstantiateInvokeDivSrcDevCMask_For(Pixel##type##C3);                                                              \
    InstantiateInvokeDivSrcDevCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivSrcDevCMask(type)                                                              \
    InstantiateInvokeDivSrcDevCMask_For(Pixel##type##C1);                                                              \
    InstantiateInvokeDivSrcDevCMask_For(Pixel##type##C2);                                                              \
    InstantiateInvokeDivSrcDevCMask_For(Pixel##type##C3);                                                              \
    InstantiateInvokeDivSrcDevCMask_For(Pixel##type##C4);                                                              \
    InstantiateInvokeDivSrcDevCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivSrcDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                               const SrcT *aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                               mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divSrcDevCScale =
                    SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                               RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT, DstT> op;
                const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcDevCScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcDevCScale =
                    SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                               RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT, DstT> op;
                const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcDevCScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                                   mpp::Div<ComputeT, DstT>, RoundingMode::TowardZero>;
                const mpp::Div<ComputeT, DstT> op;
                const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcDevCScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcDevCScale =
                    SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                               RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT, DstT> op;
                const divSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divSrcDevCScale>(
                    aMask, aPitchMask, aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcDevCScale =
                    SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                               RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT, DstT> op;
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
#define InstantiateInvokeDivSrcDevCScaleMask_For(typeSrcIsTypeDst)                                                     \
    template void                                                                                                      \
    InvokeDivSrcDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, mpp::RoundingMode aRoundingMode,     \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivSrcDevCScaleMask(type)                                                           \
    InstantiateInvokeDivSrcDevCScaleMask_For(Pixel##type##C1);                                                         \
    InstantiateInvokeDivSrcDevCScaleMask_For(Pixel##type##C2);                                                         \
    InstantiateInvokeDivSrcDevCScaleMask_For(Pixel##type##C3);                                                         \
    InstantiateInvokeDivSrcDevCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivSrcDevCScaleMask(type)                                                         \
    InstantiateInvokeDivSrcDevCScaleMask_For(Pixel##type##C1);                                                         \
    InstantiateInvokeDivSrcDevCScaleMask_For(Pixel##type##C2);                                                         \
    InstantiateInvokeDivSrcDevCScaleMask_For(Pixel##type##C3);                                                         \
    InstantiateInvokeDivSrcDevCScaleMask_For(Pixel##type##C4);                                                         \
    InstantiateInvokeDivSrcDevCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                             const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx)
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

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using divInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>, RoundingMode::None>;

            const mpp::Div<ComputeT, DstT> op;

            const divInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrc>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivInplaceSrcMask_For(typeSrcIsTypeDst)                                                       \
    template void                                                                                                      \
    InvokeDivInplaceSrcMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(         \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInplaceSrcMask(type)                                                             \
    InstantiateInvokeDivInplaceSrcMask_For(Pixel##type##C1);                                                           \
    InstantiateInvokeDivInplaceSrcMask_For(Pixel##type##C2);                                                           \
    InstantiateInvokeDivInplaceSrcMask_For(Pixel##type##C3);                                                           \
    InstantiateInvokeDivInplaceSrcMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInplaceSrcMask(type)                                                           \
    InstantiateInvokeDivInplaceSrcMask_For(Pixel##type##C1);                                                           \
    InstantiateInvokeDivInplaceSrcMask_For(Pixel##type##C2);                                                           \
    InstantiateInvokeDivInplaceSrcMask_For(Pixel##type##C3);                                                           \
    InstantiateInvokeDivInplaceSrcMask_For(Pixel##type##C4);                                                           \
    InstantiateInvokeDivInplaceSrcMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                  const SrcT *aSrc2, size_t aPitchSrc2, scalefactor_t<ComputeT> aScaleFactor,
                                  mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                           RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                           RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                                  mpp::Div<ComputeT, DstT>, RoundingMode::TowardZero>;
                const mpp::Div<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                           RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                           RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT, DstT> op;
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
#define InstantiateInvokeDivInplaceSrcScaleMask_For(typeSrcIsTypeDst)                                                  \
    template void                                                                                                      \
    InvokeDivInplaceSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,                                                              \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, mpp::RoundingMode aRoundingMode,     \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInplaceSrcScaleMask(type)                                                        \
    InstantiateInvokeDivInplaceSrcScaleMask_For(Pixel##type##C1);                                                      \
    InstantiateInvokeDivInplaceSrcScaleMask_For(Pixel##type##C2);                                                      \
    InstantiateInvokeDivInplaceSrcScaleMask_For(Pixel##type##C3);                                                      \
    InstantiateInvokeDivInplaceSrcScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInplaceSrcScaleMask(type)                                                      \
    InstantiateInvokeDivInplaceSrcScaleMask_For(Pixel##type##C1);                                                      \
    InstantiateInvokeDivInplaceSrcScaleMask_For(Pixel##type##C2);                                                      \
    InstantiateInvokeDivInplaceSrcScaleMask_For(Pixel##type##C3);                                                      \
    InstantiateInvokeDivInplaceSrcScaleMask_For(Pixel##type##C4);                                                      \
    InstantiateInvokeDivInplaceSrcScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                           const SrcT &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
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

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using divInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, RoundingMode::None>;

            const mpp::Div<ComputeT, DstT> op;

            const divInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceC>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivInplaceCMask_For(typeSrcIsTypeDst)                                                         \
    template void                                                                                                      \
    InvokeDivInplaceCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInplaceCMask(type)                                                               \
    InstantiateInvokeDivInplaceCMask_For(Pixel##type##C1);                                                             \
    InstantiateInvokeDivInplaceCMask_For(Pixel##type##C2);                                                             \
    InstantiateInvokeDivInplaceCMask_For(Pixel##type##C3);                                                             \
    InstantiateInvokeDivInplaceCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInplaceCMask(type)                                                             \
    InstantiateInvokeDivInplaceCMask_For(Pixel##type##C1);                                                             \
    InstantiateInvokeDivInplaceCMask_For(Pixel##type##C2);                                                             \
    InstantiateInvokeDivInplaceCMask_For(Pixel##type##C3);                                                             \
    InstantiateInvokeDivInplaceCMask_For(Pixel##type##C4);                                                             \
    InstantiateInvokeDivInplaceCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT &aConst, scalefactor_t<ComputeT> aScaleFactor,
                                mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                RoundingMode::TowardZero>;
                const mpp::Div<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT, DstT> op;
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
#define InstantiateInvokeDivInplaceCScaleMask_For(typeSrcIsTypeDst)                                                    \
    template void                                                                                                      \
    InvokeDivInplaceCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(      \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
        mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInplaceCScaleMask(type)                                                          \
    InstantiateInvokeDivInplaceCScaleMask_For(Pixel##type##C1);                                                        \
    InstantiateInvokeDivInplaceCScaleMask_For(Pixel##type##C2);                                                        \
    InstantiateInvokeDivInplaceCScaleMask_For(Pixel##type##C3);                                                        \
    InstantiateInvokeDivInplaceCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInplaceCScaleMask(type)                                                        \
    InstantiateInvokeDivInplaceCScaleMask_For(Pixel##type##C1);                                                        \
    InstantiateInvokeDivInplaceCScaleMask_For(Pixel##type##C2);                                                        \
    InstantiateInvokeDivInplaceCScaleMask_For(Pixel##type##C3);                                                        \
    InstantiateInvokeDivInplaceCScaleMask_For(Pixel##type##C4);                                                        \
    InstantiateInvokeDivInplaceCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                              const SrcT *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using divInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>, RoundingMode::None>;

        const mpp::Div<ComputeT, DstT> op;

        const divInplaceDevC functor(aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevC>(aMask, aPitchMask, aSrcDst, aPitchSrcDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivInplaceDevCMask_For(typeSrcIsTypeDst)                                                      \
    template void                                                                                                      \
    InvokeDivInplaceDevCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInplaceDevCMask(type)                                                            \
    InstantiateInvokeDivInplaceDevCMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeDivInplaceDevCMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeDivInplaceDevCMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeDivInplaceDevCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInplaceDevCMask(type)                                                          \
    InstantiateInvokeDivInplaceDevCMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeDivInplaceDevCMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeDivInplaceDevCMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeDivInplaceDevCMask_For(Pixel##type##C4);                                                          \
    InstantiateInvokeDivInplaceDevCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                   const SrcT *aConst, scalefactor_t<ComputeT> aScaleFactor,
                                   mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                   RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                   RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                   RoundingMode::TowardZero>;
                const mpp::Div<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                   RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Div<ComputeT, DstT>,
                                                   RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT, DstT> op;
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
#define InstantiateInvokeDivInplaceDevCScaleMask_For(typeSrcIsTypeDst)                                                 \
    template void                                                                                                      \
    InvokeDivInplaceDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
        mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInplaceDevCScaleMask(type)                                                       \
    InstantiateInvokeDivInplaceDevCScaleMask_For(Pixel##type##C1);                                                     \
    InstantiateInvokeDivInplaceDevCScaleMask_For(Pixel##type##C2);                                                     \
    InstantiateInvokeDivInplaceDevCScaleMask_For(Pixel##type##C3);                                                     \
    InstantiateInvokeDivInplaceDevCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInplaceDevCScaleMask(type)                                                     \
    InstantiateInvokeDivInplaceDevCScaleMask_For(Pixel##type##C1);                                                     \
    InstantiateInvokeDivInplaceDevCScaleMask_For(Pixel##type##C2);                                                     \
    InstantiateInvokeDivInplaceDevCScaleMask_For(Pixel##type##C3);                                                     \
    InstantiateInvokeDivInplaceDevCScaleMask_For(Pixel##type##C4);                                                     \
    InstantiateInvokeDivInplaceDevCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx)
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

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using divInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, RoundingMode::None>;

            const mpp::DivInv<ComputeT, DstT> op;

            const divInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrc>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivInvInplaceSrcMask_For(typeSrcIsTypeDst)                                                    \
    template void                                                                                                      \
    InvokeDivInvInplaceSrcMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(      \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInvInplaceSrcMask(type)                                                          \
    InstantiateInvokeDivInvInplaceSrcMask_For(Pixel##type##C1);                                                        \
    InstantiateInvokeDivInvInplaceSrcMask_For(Pixel##type##C2);                                                        \
    InstantiateInvokeDivInvInplaceSrcMask_For(Pixel##type##C3);                                                        \
    InstantiateInvokeDivInvInplaceSrcMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInvInplaceSrcMask(type)                                                        \
    InstantiateInvokeDivInvInplaceSrcMask_For(Pixel##type##C1);                                                        \
    InstantiateInvokeDivInvInplaceSrcMask_For(Pixel##type##C2);                                                        \
    InstantiateInvokeDivInvInplaceSrcMask_For(Pixel##type##C3);                                                        \
    InstantiateInvokeDivInvInplaceSrcMask_For(Pixel##type##C4);                                                        \
    InstantiateInvokeDivInvInplaceSrcMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                     const SrcT *aSrc2, size_t aPitchSrc2, scalefactor_t<ComputeT> aScaleFactor,
                                     mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                           RoundingMode::NearestTiesToEven>;
                const mpp::DivInv<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                           RoundingMode::NearestTiesAwayFromZero>;
                const mpp::DivInv<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                           RoundingMode::TowardZero>;
                const mpp::DivInv<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                           RoundingMode::TowardNegativeInfinity>;
                const mpp::DivInv<ComputeT, DstT> op;
                const divInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceSrcScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                           RoundingMode::TowardPositiveInfinity>;
                const mpp::DivInv<ComputeT, DstT> op;
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
#define InstantiateInvokeDivInvInplaceSrcScaleMask_For(typeSrcIsTypeDst)                                               \
    template void                                                                                                      \
    InvokeDivInvInplaceSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,                                                              \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, mpp::RoundingMode aRoundingMode,     \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInvInplaceSrcScaleMask(type)                                                     \
    InstantiateInvokeDivInvInplaceSrcScaleMask_For(Pixel##type##C1);                                                   \
    InstantiateInvokeDivInvInplaceSrcScaleMask_For(Pixel##type##C2);                                                   \
    InstantiateInvokeDivInvInplaceSrcScaleMask_For(Pixel##type##C3);                                                   \
    InstantiateInvokeDivInvInplaceSrcScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInvInplaceSrcScaleMask(type)                                                   \
    InstantiateInvokeDivInvInplaceSrcScaleMask_For(Pixel##type##C1);                                                   \
    InstantiateInvokeDivInvInplaceSrcScaleMask_For(Pixel##type##C2);                                                   \
    InstantiateInvokeDivInvInplaceSrcScaleMask_For(Pixel##type##C3);                                                   \
    InstantiateInvokeDivInvInplaceSrcScaleMask_For(Pixel##type##C4);                                                   \
    InstantiateInvokeDivInvInplaceSrcScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                              const SrcT &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
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

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using divInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, RoundingMode::None>;

            const mpp::DivInv<ComputeT, DstT> op;

            const divInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceC>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivInvInplaceCMask_For(typeSrcIsTypeDst)                                                      \
    template void                                                                                                      \
    InvokeDivInvInplaceCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInvInplaceCMask(type)                                                            \
    InstantiateInvokeDivInvInplaceCMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeDivInvInplaceCMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeDivInvInplaceCMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeDivInvInplaceCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInvInplaceCMask(type)                                                          \
    InstantiateInvokeDivInvInplaceCMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeDivInvInplaceCMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeDivInvInplaceCMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeDivInvInplaceCMask_For(Pixel##type##C4);                                                          \
    InstantiateInvokeDivInvInplaceCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                   const SrcT &aConst, scalefactor_t<ComputeT> aScaleFactor,
                                   mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                                RoundingMode::NearestTiesToEven>;
                const mpp::DivInv<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                                RoundingMode::NearestTiesAwayFromZero>;
                const mpp::DivInv<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                                RoundingMode::TowardZero>;
                const mpp::DivInv<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                                RoundingMode::TowardNegativeInfinity>;
                const mpp::DivInv<ComputeT, DstT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                                RoundingMode::TowardPositiveInfinity>;
                const mpp::DivInv<ComputeT, DstT> op;
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
#define InstantiateInvokeDivInvInplaceCScaleMask_For(typeSrcIsTypeDst)                                                 \
    template void                                                                                                      \
    InvokeDivInvInplaceCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
        mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInvInplaceCScaleMask(type)                                                       \
    InstantiateInvokeDivInvInplaceCScaleMask_For(Pixel##type##C1);                                                     \
    InstantiateInvokeDivInvInplaceCScaleMask_For(Pixel##type##C2);                                                     \
    InstantiateInvokeDivInvInplaceCScaleMask_For(Pixel##type##C3);                                                     \
    InstantiateInvokeDivInvInplaceCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInvInplaceCScaleMask(type)                                                     \
    InstantiateInvokeDivInvInplaceCScaleMask_For(Pixel##type##C1);                                                     \
    InstantiateInvokeDivInvInplaceCScaleMask_For(Pixel##type##C2);                                                     \
    InstantiateInvokeDivInvInplaceCScaleMask_For(Pixel##type##C3);                                                     \
    InstantiateInvokeDivInvInplaceCScaleMask_For(Pixel##type##C4);                                                     \
    InstantiateInvokeDivInvInplaceCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                 const SrcT *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using divInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>, RoundingMode::None>;

        const mpp::DivInv<ComputeT, DstT> op;

        const divInplaceDevC functor(aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevC>(aMask, aPitchMask, aSrcDst, aPitchSrcDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeDivInvInplaceDevCMask_For(typeSrcIsTypeDst)                                                   \
    template void                                                                                                      \
    InvokeDivInvInplaceDevCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInvInplaceDevCMask(type)                                                         \
    InstantiateInvokeDivInvInplaceDevCMask_For(Pixel##type##C1);                                                       \
    InstantiateInvokeDivInvInplaceDevCMask_For(Pixel##type##C2);                                                       \
    InstantiateInvokeDivInvInplaceDevCMask_For(Pixel##type##C3);                                                       \
    InstantiateInvokeDivInvInplaceDevCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInvInplaceDevCMask(type)                                                       \
    InstantiateInvokeDivInvInplaceDevCMask_For(Pixel##type##C1);                                                       \
    InstantiateInvokeDivInvInplaceDevCMask_For(Pixel##type##C2);                                                       \
    InstantiateInvokeDivInvInplaceDevCMask_For(Pixel##type##C3);                                                       \
    InstantiateInvokeDivInvInplaceDevCMask_For(Pixel##type##C4);                                                       \
    InstantiateInvokeDivInvInplaceDevCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeDivInvInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                      const SrcT *aConst, scalefactor_t<ComputeT> aScaleFactor,
                                      mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                                   RoundingMode::NearestTiesToEven>;
                const mpp::DivInv<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                                   RoundingMode::NearestTiesAwayFromZero>;
                const mpp::DivInv<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                                   RoundingMode::TowardZero>;
                const mpp::DivInv<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                                   RoundingMode::TowardNegativeInfinity>;
                const mpp::DivInv<ComputeT, DstT> op;
                const divInplaceDevCScale functor(aConst, op, aScaleFactor);
                InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, divInplaceDevCScale>(
                    aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceDevCScale =
                    InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::DivInv<ComputeT, DstT>,
                                                   RoundingMode::TowardPositiveInfinity>;
                const mpp::DivInv<ComputeT, DstT> op;
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
#define InstantiateInvokeDivInvInplaceDevCScaleMask_For(typeSrcIsTypeDst)                                              \
    template void InvokeDivInvInplaceDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>,     \
                                                   typeSrcIsTypeDst>(                                                  \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
        mpp::RoundingMode aRoundingMode, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeDivInvInplaceDevCScaleMask(type)                                                    \
    InstantiateInvokeDivInvInplaceDevCScaleMask_For(Pixel##type##C1);                                                  \
    InstantiateInvokeDivInvInplaceDevCScaleMask_For(Pixel##type##C2);                                                  \
    InstantiateInvokeDivInvInplaceDevCScaleMask_For(Pixel##type##C3);                                                  \
    InstantiateInvokeDivInvInplaceDevCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeDivInvInplaceDevCScaleMask(type)                                                  \
    InstantiateInvokeDivInvInplaceDevCScaleMask_For(Pixel##type##C1);                                                  \
    InstantiateInvokeDivInvInplaceDevCScaleMask_For(Pixel##type##C2);                                                  \
    InstantiateInvokeDivInvInplaceDevCScaleMask_For(Pixel##type##C3);                                                  \
    InstantiateInvokeDivInvInplaceDevCScaleMask_For(Pixel##type##C4);                                                  \
    InstantiateInvokeDivInvInplaceDevCScaleMask_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
