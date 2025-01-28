#if OPP_ENABLE_CUDA_BACKEND

#include "mulMasked.h"
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
void InvokeMulSrcSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                         const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Mul<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulSrcSrcSIMD = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None,
                                                ComputeT, simdOP_t>;

            Mul<ComputeT> op;
            simdOP_t opSIMD;

            mulSrcSrcSIMD functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcSrcSIMD>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                  aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

            Mul<ComputeT> op;

            mulSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcSrc>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                              aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulSrcSrcMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(             \
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

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulSrcSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                              const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst,
                              scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeMulSrcSrcMask(aMask, aPitchMask, aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aDst, aPitchDst, aSize,
                                aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using mulSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>,
                                                  RoundingMode::NearestTiesAwayFromZero>;

        Mul<ComputeT> op;

        mulSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcSrcScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulSrcSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                   \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
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
void InvokeMulSrcCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                       const SrcT &aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                       const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Mul<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulSrcCSIMD = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>,
                                                   RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            Mul<ComputeT> op;
            simdOP_t opSIMD;
            Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            mulSrcCSIMD functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcCSIMD>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

            Mul<ComputeT> op;

            mulSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                            aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeMulSrcCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
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

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulSrcCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                            const SrcT &aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                            const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeMulSrcCMask(aMask, aPitchMask, aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using mulSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>,
                                                     RoundingMode::NearestTiesAwayFromZero>;

        Mul<ComputeT> op;

        mulSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcCScale>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                             aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulSrcCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
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
void InvokeMulSrcDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                          const SrcT *aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
        using mulSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

        Mul<ComputeT> op;

        mulSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcDevC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                           aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulSrcDevCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
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

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulSrcDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                               const SrcT *aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                               const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeMulSrcDevCMask(aMask, aPitchMask, aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using mulSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>,
                                                           RoundingMode::NearestTiesAwayFromZero>;

        Mul<ComputeT> op;

        mulSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcDevCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulSrcDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
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
void InvokeMulInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                             const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Mul<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>,
                                                        RoundingMode::None, ComputeT, simdOP_t>;

            Mul<ComputeT> op;
            simdOP_t opSIMD;

            mulInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceSrcSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

            Mul<ComputeT> op;

            mulInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceSrc>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulInplaceSrcMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(         \
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

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                  const SrcT *aSrc2, size_t aPitchSrc2, scalefactor_t<ComputeT> aScaleFactor,
                                  const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeMulInplaceSrcMask(aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSrc2, aPitchSrc2, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using mulInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>,
                                                          RoundingMode::NearestTiesAwayFromZero>;

        Mul<ComputeT> op;

        mulInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceSrcScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulInplaceSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,                                                              \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
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
void InvokeMulInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                           const SrcT &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Mul<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulInplaceCSIMD = InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Mul<ComputeT>,
                                                           RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            Mul<ComputeT> op;
            simdOP_t opSIMD;
            Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            mulInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceCSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

            Mul<ComputeT> op;

            mulInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceC>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulInplaceCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
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

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT &aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeMulInplaceCMask(aMask, aPitchMask, aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using mulInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Mul<ComputeT>,
                                                             RoundingMode::NearestTiesAwayFromZero>;

        Mul<ComputeT> op;

        mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceCScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulInplaceCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(      \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
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
void InvokeMulInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                              const SrcT *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
        using mulInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

        Mul<ComputeT> op;

        mulInplaceDevC functor(aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceDevC>(aMask, aPitchMask, aSrcDst, aPitchSrcDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulInplaceDevCMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
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

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                   const SrcT *aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                   const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeMulInplaceDevCMask(aMask, aPitchMask, aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using mulInplaceDevCScale = InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Mul<ComputeT>,
                                                                   RoundingMode::NearestTiesAwayFromZero>;

        Mul<ComputeT> op;

        mulInplaceDevCScale functor(aConst, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceDevCScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulInplaceDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
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
