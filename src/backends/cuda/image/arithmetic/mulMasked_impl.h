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

            const opp::Mul<ComputeT> op;
            const simdOP_t opSIMD;

            const mulSrcSrcSIMD functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcSrcSIMD>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                  aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

            const opp::Mul<ComputeT> op;

            const mulSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcSrc>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                              aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeMulSrcSrcMask_For(typeSrcIsTypeDst)                                                           \
    template void                                                                                                      \
    InvokeMulSrcSrcMask<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(         \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                   \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulSrcSrcMask(type)                                                                 \
    InstantiateInvokeMulSrcSrcMask_For(Pixel##type##C1);                                                               \
    InstantiateInvokeMulSrcSrcMask_For(Pixel##type##C2);                                                               \
    InstantiateInvokeMulSrcSrcMask_For(Pixel##type##C3);                                                               \
    InstantiateInvokeMulSrcSrcMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulSrcSrcMask(type)                                                               \
    InstantiateInvokeMulSrcSrcMask_For(Pixel##type##C1);                                                               \
    InstantiateInvokeMulSrcSrcMask_For(Pixel##type##C2);                                                               \
    InstantiateInvokeMulSrcSrcMask_For(Pixel##type##C3);                                                               \
    InstantiateInvokeMulSrcSrcMask_For(Pixel##type##C4);                                                               \
    InstantiateInvokeMulSrcSrcMask_For(Pixel##type##C4A);

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

        using mulSrcSrcScale =
            SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::NearestTiesToEven>;

        const opp::Mul<ComputeT> op;

        const mulSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcSrcScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeMulSrcSrcScaleMask_For(typeSrcIsTypeDst)                                                      \
    template void                                                                                                      \
    InvokeMulSrcSrcScaleMask<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                   \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        scalefactor_t<default_ext_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,             \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulSrcSrcScaleMask(type)                                                            \
    InstantiateInvokeMulSrcSrcScaleMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeMulSrcSrcScaleMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeMulSrcSrcScaleMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeMulSrcSrcScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulSrcSrcScaleMask(type)                                                          \
    InstantiateInvokeMulSrcSrcScaleMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeMulSrcSrcScaleMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeMulSrcSrcScaleMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeMulSrcSrcScaleMask_For(Pixel##type##C4);                                                          \
    InstantiateInvokeMulSrcSrcScaleMask_For(Pixel##type##C4A);

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

            const opp::Mul<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            const mulSrcCSIMD functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcCSIMD>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

            const opp::Mul<ComputeT> op;

            const mulSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                            aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeMulSrcCMask_For(typeSrcIsTypeDst)                                                             \
    template void                                                                                                      \
    InvokeMulSrcCMask<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulSrcCMask(type)                                                                   \
    InstantiateInvokeMulSrcCMask_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeMulSrcCMask_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeMulSrcCMask_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeMulSrcCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulSrcCMask(type)                                                                 \
    InstantiateInvokeMulSrcCMask_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeMulSrcCMask_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeMulSrcCMask_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeMulSrcCMask_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeMulSrcCMask_For(Pixel##type##C4A);

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
                                                     RoundingMode::NearestTiesToEven>;

        const opp::Mul<ComputeT> op;

        const mulSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcCScale>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                             aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeMulSrcCScaleMask_For(typeSrcIsTypeDst)                                                        \
    template void                                                                                                      \
    InvokeMulSrcCScaleMask<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(      \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<default_ext_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,             \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulSrcCScaleMask(type)                                                              \
    InstantiateInvokeMulSrcCScaleMask_For(Pixel##type##C1);                                                            \
    InstantiateInvokeMulSrcCScaleMask_For(Pixel##type##C2);                                                            \
    InstantiateInvokeMulSrcCScaleMask_For(Pixel##type##C3);                                                            \
    InstantiateInvokeMulSrcCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulSrcCScaleMask(type)                                                            \
    InstantiateInvokeMulSrcCScaleMask_For(Pixel##type##C1);                                                            \
    InstantiateInvokeMulSrcCScaleMask_For(Pixel##type##C2);                                                            \
    InstantiateInvokeMulSrcCScaleMask_For(Pixel##type##C3);                                                            \
    InstantiateInvokeMulSrcCScaleMask_For(Pixel##type##C4);                                                            \
    InstantiateInvokeMulSrcCScaleMask_For(Pixel##type##C4A);

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

        const Mul<ComputeT> op;

        const mulSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcDevC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                           aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeMulSrcDevCMask_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeMulSrcDevCMask<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulSrcDevCMask(type)                                                                \
    InstantiateInvokeMulSrcDevCMask_For(Pixel##type##C1);                                                              \
    InstantiateInvokeMulSrcDevCMask_For(Pixel##type##C2);                                                              \
    InstantiateInvokeMulSrcDevCMask_For(Pixel##type##C3);                                                              \
    InstantiateInvokeMulSrcDevCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulSrcDevCMask(type)                                                              \
    InstantiateInvokeMulSrcDevCMask_For(Pixel##type##C1);                                                              \
    InstantiateInvokeMulSrcDevCMask_For(Pixel##type##C2);                                                              \
    InstantiateInvokeMulSrcDevCMask_For(Pixel##type##C3);                                                              \
    InstantiateInvokeMulSrcDevCMask_For(Pixel##type##C4);                                                              \
    InstantiateInvokeMulSrcDevCMask_For(Pixel##type##C4A);

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
                                                           RoundingMode::NearestTiesToEven>;

        const opp::Mul<ComputeT> op;

        const mulSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulSrcDevCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeMulSrcDevCScaleMask_For(typeSrcIsTypeDst)                                                     \
    template void                                                                                                      \
    InvokeMulSrcDevCScaleMask<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<default_ext_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,             \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulSrcDevCScaleMask(type)                                                           \
    InstantiateInvokeMulSrcDevCScaleMask_For(Pixel##type##C1);                                                         \
    InstantiateInvokeMulSrcDevCScaleMask_For(Pixel##type##C2);                                                         \
    InstantiateInvokeMulSrcDevCScaleMask_For(Pixel##type##C3);                                                         \
    InstantiateInvokeMulSrcDevCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulSrcDevCScaleMask(type)                                                         \
    InstantiateInvokeMulSrcDevCScaleMask_For(Pixel##type##C1);                                                         \
    InstantiateInvokeMulSrcDevCScaleMask_For(Pixel##type##C2);                                                         \
    InstantiateInvokeMulSrcDevCScaleMask_For(Pixel##type##C3);                                                         \
    InstantiateInvokeMulSrcDevCScaleMask_For(Pixel##type##C4);                                                         \
    InstantiateInvokeMulSrcDevCScaleMask_For(Pixel##type##C4A);

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

            const opp::Mul<ComputeT> op;
            const simdOP_t opSIMD;

            const mulInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceSrcSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

            const opp::Mul<ComputeT> op;

            const mulInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceSrc>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeMulInplaceSrcMask_For(typeSrcIsTypeDst)                                                       \
    template void                                                                                                      \
    InvokeMulInplaceSrcMask<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulInplaceSrcMask(type)                                                             \
    InstantiateInvokeMulInplaceSrcMask_For(Pixel##type##C1);                                                           \
    InstantiateInvokeMulInplaceSrcMask_For(Pixel##type##C2);                                                           \
    InstantiateInvokeMulInplaceSrcMask_For(Pixel##type##C3);                                                           \
    InstantiateInvokeMulInplaceSrcMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulInplaceSrcMask(type)                                                           \
    InstantiateInvokeMulInplaceSrcMask_For(Pixel##type##C1);                                                           \
    InstantiateInvokeMulInplaceSrcMask_For(Pixel##type##C2);                                                           \
    InstantiateInvokeMulInplaceSrcMask_For(Pixel##type##C3);                                                           \
    InstantiateInvokeMulInplaceSrcMask_For(Pixel##type##C4);                                                           \
    InstantiateInvokeMulInplaceSrcMask_For(Pixel##type##C4A);

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
                                                          RoundingMode::NearestTiesToEven>;

        const opp::Mul<ComputeT> op;

        const mulInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceSrcScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeMulInplaceSrcScaleMask_For(typeSrcIsTypeDst)                                                  \
    template void InvokeMulInplaceSrcScaleMask<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>,     \
                                               typeSrcIsTypeDst>(                                                      \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,                                                              \
        scalefactor_t<default_ext_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,             \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulInplaceSrcScaleMask(type)                                                        \
    InstantiateInvokeMulInplaceSrcScaleMask_For(Pixel##type##C1);                                                      \
    InstantiateInvokeMulInplaceSrcScaleMask_For(Pixel##type##C2);                                                      \
    InstantiateInvokeMulInplaceSrcScaleMask_For(Pixel##type##C3);                                                      \
    InstantiateInvokeMulInplaceSrcScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulInplaceSrcScaleMask(type)                                                      \
    InstantiateInvokeMulInplaceSrcScaleMask_For(Pixel##type##C1);                                                      \
    InstantiateInvokeMulInplaceSrcScaleMask_For(Pixel##type##C2);                                                      \
    InstantiateInvokeMulInplaceSrcScaleMask_For(Pixel##type##C3);                                                      \
    InstantiateInvokeMulInplaceSrcScaleMask_For(Pixel##type##C4);                                                      \
    InstantiateInvokeMulInplaceSrcScaleMask_For(Pixel##type##C4A);

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

            const opp::Mul<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            const mulInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceCSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::None>;

            const opp::Mul<ComputeT> op;

            const mulInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceC>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeMulInplaceCMask_For(typeSrcIsTypeDst)                                                         \
    template void                                                                                                      \
    InvokeMulInplaceCMask<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulInplaceCMask(type)                                                               \
    InstantiateInvokeMulInplaceCMask_For(Pixel##type##C1);                                                             \
    InstantiateInvokeMulInplaceCMask_For(Pixel##type##C2);                                                             \
    InstantiateInvokeMulInplaceCMask_For(Pixel##type##C3);                                                             \
    InstantiateInvokeMulInplaceCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulInplaceCMask(type)                                                             \
    InstantiateInvokeMulInplaceCMask_For(Pixel##type##C1);                                                             \
    InstantiateInvokeMulInplaceCMask_For(Pixel##type##C2);                                                             \
    InstantiateInvokeMulInplaceCMask_For(Pixel##type##C3);                                                             \
    InstantiateInvokeMulInplaceCMask_For(Pixel##type##C4);                                                             \
    InstantiateInvokeMulInplaceCMask_For(Pixel##type##C4A);

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

        using mulInplaceCScale =
            InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Mul<ComputeT>, RoundingMode::NearestTiesToEven>;

        const opp::Mul<ComputeT> op;

        const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceCScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeMulInplaceCScaleMask_For(typeSrcIsTypeDst)                                                    \
    template void                                                                                                      \
    InvokeMulInplaceCScaleMask<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(  \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, scalefactor_t<default_ext_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,  \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulInplaceCScaleMask(type)                                                          \
    InstantiateInvokeMulInplaceCScaleMask_For(Pixel##type##C1);                                                        \
    InstantiateInvokeMulInplaceCScaleMask_For(Pixel##type##C2);                                                        \
    InstantiateInvokeMulInplaceCScaleMask_For(Pixel##type##C3);                                                        \
    InstantiateInvokeMulInplaceCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulInplaceCScaleMask(type)                                                        \
    InstantiateInvokeMulInplaceCScaleMask_For(Pixel##type##C1);                                                        \
    InstantiateInvokeMulInplaceCScaleMask_For(Pixel##type##C2);                                                        \
    InstantiateInvokeMulInplaceCScaleMask_For(Pixel##type##C3);                                                        \
    InstantiateInvokeMulInplaceCScaleMask_For(Pixel##type##C4);                                                        \
    InstantiateInvokeMulInplaceCScaleMask_For(Pixel##type##C4A);

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

        const opp::Mul<ComputeT> op;

        const mulInplaceDevC functor(aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceDevC>(aMask, aPitchMask, aSrcDst, aPitchSrcDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeMulInplaceDevCMask_For(typeSrcIsTypeDst)                                                      \
    template void                                                                                                      \
    InvokeMulInplaceDevCMask<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulInplaceDevCMask(type)                                                            \
    InstantiateInvokeMulInplaceDevCMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeMulInplaceDevCMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeMulInplaceDevCMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeMulInplaceDevCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulInplaceDevCMask(type)                                                          \
    InstantiateInvokeMulInplaceDevCMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeMulInplaceDevCMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeMulInplaceDevCMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeMulInplaceDevCMask_For(Pixel##type##C4);                                                          \
    InstantiateInvokeMulInplaceDevCMask_For(Pixel##type##C4A);

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
                                                                   RoundingMode::NearestTiesToEven>;

        const opp::Mul<ComputeT> op;

        const mulInplaceDevCScale functor(aConst, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, mulInplaceDevCScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeMulInplaceDevCScaleMask_For(typeSrcIsTypeDst)                                                 \
    template void InvokeMulInplaceDevCScaleMask<typeSrcIsTypeDst, default_ext_compute_type_for_t<typeSrcIsTypeDst>,    \
                                                typeSrcIsTypeDst>(                                                     \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, scalefactor_t<default_ext_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,  \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulInplaceDevCScaleMask(type)                                                       \
    InstantiateInvokeMulInplaceDevCScaleMask_For(Pixel##type##C1);                                                     \
    InstantiateInvokeMulInplaceDevCScaleMask_For(Pixel##type##C2);                                                     \
    InstantiateInvokeMulInplaceDevCScaleMask_For(Pixel##type##C3);                                                     \
    InstantiateInvokeMulInplaceDevCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulInplaceDevCScaleMask(type)                                                     \
    InstantiateInvokeMulInplaceDevCScaleMask_For(Pixel##type##C1);                                                     \
    InstantiateInvokeMulInplaceDevCScaleMask_For(Pixel##type##C2);                                                     \
    InstantiateInvokeMulInplaceDevCScaleMask_For(Pixel##type##C3);                                                     \
    InstantiateInvokeMulInplaceDevCScaleMask_For(Pixel##type##C4);                                                     \
    InstantiateInvokeMulInplaceDevCScaleMask_For(Pixel##type##C4A);

#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
