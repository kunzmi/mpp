#if OPP_ENABLE_CUDA_BACKEND

#include "mul.h"
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
void InvokeMulSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                     size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
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

            const opp::Mul<ComputeT> op;
            const simdOP_t opSIMD;

            const mulSrcSrcSIMD functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcSrcSIMD>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                            functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

            const opp::Mul<ComputeT> op;

            const mulSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulSrcSrc<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(             \
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
void InvokeMulSrcSrcScale(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeMulSrcSrc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using mulSrcSrcScale =
            SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::NearestTiesToEven>;

        const opp::Mul<ComputeT> op;

        const mulSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulSrcSrcScale<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst,                                                                      \
        scalefactor_t<default_ext_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,             \
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
void InvokeMulSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                   const Size2D &aSize, const StreamCtx &aStreamCtx)
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

            const opp::Mul<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            const mulSrcCSIMD functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcCSIMD>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

            const opp::Mul<ComputeT> op;

            const mulSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeMulSrcC<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
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
void InvokeMulSrcCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                        scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeMulSrcC(aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using mulSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>,
                                                     RoundingMode::NearestTiesToEven>;

        const opp::Mul<ComputeT> op;

        const mulSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulSrcCScale<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, scalefactor_t<default_ext_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,                \
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
void InvokeMulSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
        using mulSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

        const opp::Mul<ComputeT> op;

        const mulSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcDevC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulSrcDevC<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
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
void InvokeMulSrcDevCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                           scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeMulSrcDevC(aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using mulSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>,
                                                           RoundingMode::NearestTiesToEven>;

        const opp::Mul<ComputeT> op;

        const mulSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcDevCScale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulSrcDevCScale<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, scalefactor_t<default_ext_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,                \
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
void InvokeMulInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
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
            using mulInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>,
                                                        RoundingMode::None, ComputeT, simdOP_t>;

            const opp::Mul<ComputeT> op;
            const simdOP_t opSIMD;

            const mulInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceSrcSIMD>(aSrcDst, aPitchSrcDst, aSize,
                                                                                aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

            const opp::Mul<ComputeT> op;

            const mulInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulInplaceSrc<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(         \
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
void InvokeMulInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeMulInplaceSrc(aSrcDst, aPitchSrcDst, aSrc2, aPitchSrc2, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using mulInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>,
                                                          RoundingMode::NearestTiesToEven>;

        const opp::Mul<ComputeT> op;

        const mulInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                             functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulInplaceSrcScale<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        scalefactor_t<default_ext_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,             \
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
void InvokeMulInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
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
            using mulInplaceCSIMD = InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Mul<ComputeT>,
                                                           RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            const opp::Mul<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            const mulInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceCSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

            const opp::Mul<ComputeT> op;

            const mulInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                          functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulInplaceC<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
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
void InvokeMulInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst,
                            scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeMulInplaceC(aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using mulInplaceCScale =
            InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::NearestTiesToEven>;

        const opp::Mul<ComputeT> op;

        const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceCScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                           functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulInplaceCScale<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(      \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst,                               \
        scalefactor_t<default_ext_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,             \
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
void InvokeMulInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
        using mulInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

        const opp::Mul<ComputeT> op;

        const mulInplaceDevC functor(aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulInplaceDevC<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
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
void InvokeMulInplaceDevCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst,
                               scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeMulInplaceDevC(aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using mulInplaceDevCScale = InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Mul<ComputeT>,
                                                                   RoundingMode::NearestTiesToEven>;

        const opp::Mul<ComputeT> op;

        const mulInplaceDevCScale functor(aConst, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT without SIMD
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeMulInplaceDevCScale<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst,                               \
        scalefactor_t<default_ext_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,             \
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
