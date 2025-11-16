#if MPP_ENABLE_CUDA_BACKEND

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
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                     size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Mul<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulSrcSrcSIMD = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>, RoundingMode::None,
                                                ComputeT, simdOP_t>;

            const mpp::Mul<ComputeT> op;
            const simdOP_t opSIMD;

            const mulSrcSrcSIMD functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcSrcSIMD>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                            functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>, RoundingMode::None>;

            const mpp::Mul<ComputeT> op;

            const mulSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeMulSrcSrc_For(typeSrcIsTypeDst)                                                               \
    template void                                                                                                      \
    InvokeMulSrcSrc<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(         \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulSrcSrc(type)                                                                     \
    InstantiateInvokeMulSrcSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeMulSrcSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeMulSrcSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeMulSrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulSrcSrc(type)                                                                   \
    InstantiateInvokeMulSrcSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeMulSrcSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeMulSrcSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeMulSrcSrc_For(Pixel##type##C4);                                                                   \
    InstantiateInvokeMulSrcSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulSrcSrcScale(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, double aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0)
        {
            InvokeMulSrcSrc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if (aScaleFactor > 1.0 || RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            using mulSrcSrcScale = SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>, ScalerT,
                                                      RoundingMode::NearestTiesToEven>;

            const mpp::Mul<ComputeT> op;

            const mulSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, scaler);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                             functor);
        }
        else
        {
            // Scaler performs NearestTiesToEven rounding:
            using ScalerT = Scale<ComputeT, true>;
            const ScalerT scaler(aScaleFactor);
            using mulSrcSrcScale =
                SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

            const mpp::Mul<ComputeT> op;

            const mulSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, scaler);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                             functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeMulSrcSrcScale_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeMulSrcSrcScale<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, double aScaleFactor, const Size2D &aSize,                            \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulSrcSrcScale(type)                                                                \
    InstantiateInvokeMulSrcSrcScale_For(Pixel##type##C1);                                                              \
    InstantiateInvokeMulSrcSrcScale_For(Pixel##type##C2);                                                              \
    InstantiateInvokeMulSrcSrcScale_For(Pixel##type##C3);                                                              \
    InstantiateInvokeMulSrcSrcScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulSrcSrcScale(type)                                                              \
    InstantiateInvokeMulSrcSrcScale_For(Pixel##type##C1);                                                              \
    InstantiateInvokeMulSrcSrcScale_For(Pixel##type##C2);                                                              \
    InstantiateInvokeMulSrcSrcScale_For(Pixel##type##C3);                                                              \
    InstantiateInvokeMulSrcSrcScale_For(Pixel##type##C4);                                                              \
    InstantiateInvokeMulSrcSrcScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                   const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Mul<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulSrcCSIMD = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>,
                                                   RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            const mpp::Mul<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT, TupelSize> tupelConstant =
                Tupel<ComputeT, TupelSize>::GetConstant(static_cast<ComputeT>(aConst));

            const mulSrcCSIMD functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcCSIMD>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>, RoundingMode::None>;

            const mpp::Mul<ComputeT> op;

            const mulSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeMulSrcC_For(typeSrcIsTypeDst)                                                                 \
    template void                                                                                                      \
    InvokeMulSrcC<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulSrcC(type)                                                                       \
    InstantiateInvokeMulSrcC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeMulSrcC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeMulSrcC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeMulSrcC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulSrcC(type)                                                                     \
    InstantiateInvokeMulSrcC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeMulSrcC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeMulSrcC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeMulSrcC_For(Pixel##type##C4);                                                                     \
    InstantiateInvokeMulSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulSrcCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                        double aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0)
        {
            InvokeMulSrcC(aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if (aScaleFactor > 1.0 || RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            using mulSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>, ScalerT,
                                                         RoundingMode::NearestTiesToEven>;

            const mpp::Mul<ComputeT> op;

            const mulSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, scaler);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // Scaler performs NearestTiesToEven rounding:
            using ScalerT = Scale<ComputeT, true>;
            const ScalerT scaler(aScaleFactor);
            using mulSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>, ScalerT,
                                                         RoundingMode::None>;

            const mpp::Mul<ComputeT> op;

            const mulSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, scaler);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeMulSrcCScale_For(typeSrcIsTypeDst)                                                            \
    template void                                                                                                      \
    InvokeMulSrcCScale<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(      \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, double aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulSrcCScale(type)                                                                  \
    InstantiateInvokeMulSrcCScale_For(Pixel##type##C1);                                                                \
    InstantiateInvokeMulSrcCScale_For(Pixel##type##C2);                                                                \
    InstantiateInvokeMulSrcCScale_For(Pixel##type##C3);                                                                \
    InstantiateInvokeMulSrcCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulSrcCScale(type)                                                                \
    InstantiateInvokeMulSrcCScale_For(Pixel##type##C1);                                                                \
    InstantiateInvokeMulSrcCScale_For(Pixel##type##C2);                                                                \
    InstantiateInvokeMulSrcCScale_For(Pixel##type##C3);                                                                \
    InstantiateInvokeMulSrcCScale_For(Pixel##type##C4);                                                                \
    InstantiateInvokeMulSrcCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
        using mulSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;

        const mulSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcDevC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeMulSrcDevC_For(typeSrcIsTypeDst)                                                              \
    template void                                                                                                      \
    InvokeMulSrcDevC<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulSrcDevC(type)                                                                    \
    InstantiateInvokeMulSrcDevC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeMulSrcDevC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeMulSrcDevC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeMulSrcDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulSrcDevC(type)                                                                  \
    InstantiateInvokeMulSrcDevC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeMulSrcDevC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeMulSrcDevC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeMulSrcDevC_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeMulSrcDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulSrcDevCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                           double aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0)
        {
            InvokeMulSrcDevC(aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if (aScaleFactor > 1.0 || RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            using mulSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>,
                                                               ScalerT, RoundingMode::NearestTiesToEven>;

            const mpp::Mul<ComputeT> op;

            const mulSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, scaler);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcDevCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
        }
        else
        {
            // Scaler performs NearestTiesToEven rounding:
            using ScalerT = Scale<ComputeT, true>;
            const ScalerT scaler(aScaleFactor);
            using mulSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>,
                                                               ScalerT, RoundingMode::None>;

            const mpp::Mul<ComputeT> op;

            const mulSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, scaler);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulSrcDevCScale>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeMulSrcDevCScale_For(typeSrcIsTypeDst)                                                         \
    template void                                                                                                      \
    InvokeMulSrcDevCScale<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, double aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulSrcDevCScale(type)                                                               \
    InstantiateInvokeMulSrcDevCScale_For(Pixel##type##C1);                                                             \
    InstantiateInvokeMulSrcDevCScale_For(Pixel##type##C2);                                                             \
    InstantiateInvokeMulSrcDevCScale_For(Pixel##type##C3);                                                             \
    InstantiateInvokeMulSrcDevCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulSrcDevCScale(type)                                                             \
    InstantiateInvokeMulSrcDevCScale_For(Pixel##type##C1);                                                             \
    InstantiateInvokeMulSrcDevCScale_For(Pixel##type##C2);                                                             \
    InstantiateInvokeMulSrcDevCScale_For(Pixel##type##C3);                                                             \
    InstantiateInvokeMulSrcDevCScale_For(Pixel##type##C4);                                                             \
    InstantiateInvokeMulSrcDevCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Mul<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>,
                                                        RoundingMode::None, ComputeT, simdOP_t>;

            const mpp::Mul<ComputeT> op;
            const simdOP_t opSIMD;

            const mulInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceSrcSIMD>(aSrcDst, aPitchSrcDst, aSize,
                                                                                aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>, RoundingMode::None>;

            const mpp::Mul<ComputeT> op;

            const mulInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeMulInplaceSrc_For(typeSrcIsTypeDst)                                                           \
    template void                                                                                                      \
    InvokeMulInplaceSrc<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulInplaceSrc(type)                                                                 \
    InstantiateInvokeMulInplaceSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeMulInplaceSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeMulInplaceSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeMulInplaceSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulInplaceSrc(type)                                                               \
    InstantiateInvokeMulInplaceSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeMulInplaceSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeMulInplaceSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeMulInplaceSrc_For(Pixel##type##C4);                                                               \
    InstantiateInvokeMulInplaceSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              double aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0)
        {
            InvokeMulInplaceSrc(aSrcDst, aPitchSrcDst, aSrc2, aPitchSrc2, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if (aScaleFactor > 1.0 || RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            using mulInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>,
                                                              ScalerT, RoundingMode::NearestTiesToEven>;

            const mpp::Mul<ComputeT> op;

            const mulInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                 aStreamCtx, functor);
        }
        else
        {
            // Scaler performs NearestTiesToEven rounding:
            using ScalerT = Scale<ComputeT, true>;
            const ScalerT scaler(aScaleFactor);
            using mulInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Mul<ComputeT>,
                                                              ScalerT, RoundingMode::None>;

            const mpp::Mul<ComputeT> op;

            const mulInplaceSrcScale functor(aSrc2, aPitchSrc2, op, scaler);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                 aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeMulInplaceSrcScale_For(typeSrcIsTypeDst)                                                      \
    template void InvokeMulInplaceSrcScale<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>,     \
                                           typeSrcIsTypeDst>(                                                          \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        double aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulInplaceSrcScale(type)                                                            \
    InstantiateInvokeMulInplaceSrcScale_For(Pixel##type##C1);                                                          \
    InstantiateInvokeMulInplaceSrcScale_For(Pixel##type##C2);                                                          \
    InstantiateInvokeMulInplaceSrcScale_For(Pixel##type##C3);                                                          \
    InstantiateInvokeMulInplaceSrcScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulInplaceSrcScale(type)                                                          \
    InstantiateInvokeMulInplaceSrcScale_For(Pixel##type##C1);                                                          \
    InstantiateInvokeMulInplaceSrcScale_For(Pixel##type##C2);                                                          \
    InstantiateInvokeMulInplaceSrcScale_For(Pixel##type##C3);                                                          \
    InstantiateInvokeMulInplaceSrcScale_For(Pixel##type##C4);                                                          \
    InstantiateInvokeMulInplaceSrcScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                       const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Mul<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
            using mulInplaceCSIMD = InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Mul<ComputeT>,
                                                           RoundingMode::None, Tupel<ComputeT, TupelSize>, simdOP_t>;

            const mpp::Mul<ComputeT> op;
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
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Mul<ComputeT>, RoundingMode::None>;

            const mpp::Mul<ComputeT> op;

            const mulInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                          functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeMulInplaceC_For(typeSrcIsTypeDst)                                                             \
    template void                                                                                                      \
    InvokeMulInplaceC<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulInplaceC(type)                                                                   \
    InstantiateInvokeMulInplaceC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeMulInplaceC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeMulInplaceC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeMulInplaceC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulInplaceC(type)                                                                 \
    InstantiateInvokeMulInplaceC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeMulInplaceC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeMulInplaceC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeMulInplaceC_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeMulInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, double aScaleFactor,
                            const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0)
        {
            InvokeMulInplaceC(aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if (aScaleFactor > 1.0 || RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            using mulInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Mul<ComputeT>, ScalerT,
                                                                 RoundingMode::NearestTiesToEven>;

            const mpp::Mul<ComputeT> op;

            const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceCScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                               functor);
        }
        else
        {
            // Scaler performs NearestTiesToEven rounding:
            using ScalerT = Scale<ComputeT, true>;
            const ScalerT scaler(aScaleFactor);
            using mulInplaceCScale =
                InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

            const mpp::Mul<ComputeT> op;

            const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceCScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                               functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeMulInplaceCScale_For(typeSrcIsTypeDst)                                                        \
    template void                                                                                                      \
    InvokeMulInplaceCScale<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(  \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, double aScaleFactor,          \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulInplaceCScale(type)                                                              \
    InstantiateInvokeMulInplaceCScale_For(Pixel##type##C1);                                                            \
    InstantiateInvokeMulInplaceCScale_For(Pixel##type##C2);                                                            \
    InstantiateInvokeMulInplaceCScale_For(Pixel##type##C3);                                                            \
    InstantiateInvokeMulInplaceCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulInplaceCScale(type)                                                            \
    InstantiateInvokeMulInplaceCScale_For(Pixel##type##C1);                                                            \
    InstantiateInvokeMulInplaceCScale_For(Pixel##type##C2);                                                            \
    InstantiateInvokeMulInplaceCScale_For(Pixel##type##C3);                                                            \
    InstantiateInvokeMulInplaceCScale_For(Pixel##type##C4);                                                            \
    InstantiateInvokeMulInplaceCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
        using mulInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::Mul<ComputeT>, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;

        const mulInplaceDevC functor(aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeMulInplaceDevC_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeMulInplaceDevC<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulInplaceDevC(type)                                                                \
    InstantiateInvokeMulInplaceDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeMulInplaceDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeMulInplaceDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeMulInplaceDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulInplaceDevC(type)                                                              \
    InstantiateInvokeMulInplaceDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeMulInplaceDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeMulInplaceDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeMulInplaceDevC_For(Pixel##type##C4);                                                              \
    InstantiateInvokeMulInplaceDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMulInplaceDevCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, double aScaleFactor,
                               const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0)
        {
            InvokeMulInplaceDevC(aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        if (aScaleFactor > 1.0 || RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            using mulInplaceDevCScale = InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Mul<ComputeT>,
                                                                       ScalerT, RoundingMode::NearestTiesToEven>;

            const mpp::Mul<ComputeT> op;

            const mulInplaceDevCScale functor(aConst, op, scaler);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                  aStreamCtx, functor);
        }
        else
        {
            // Scaler performs NearestTiesToEven rounding:
            using ScalerT = Scale<ComputeT, true>;
            const ScalerT scaler(aScaleFactor);
            using mulInplaceDevCScale = InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Mul<ComputeT>,
                                                                       ScalerT, RoundingMode::None>;

            const mpp::Mul<ComputeT> op;

            const mulInplaceDevCScale functor(aConst, op, scaler);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, mulInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize,
                                                                                  aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeMulInplaceDevCScale_For(typeSrcIsTypeDst)                                                     \
    template void InvokeMulInplaceDevCScale<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>,    \
                                            typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst,         \
                                                              const typeSrcIsTypeDst *aConst, double aScaleFactor,     \
                                                              const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMulInplaceDevCScale(type)                                                           \
    InstantiateInvokeMulInplaceDevCScale_For(Pixel##type##C1);                                                         \
    InstantiateInvokeMulInplaceDevCScale_For(Pixel##type##C2);                                                         \
    InstantiateInvokeMulInplaceDevCScale_For(Pixel##type##C3);                                                         \
    InstantiateInvokeMulInplaceDevCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMulInplaceDevCScale(type)                                                         \
    InstantiateInvokeMulInplaceDevCScale_For(Pixel##type##C1);                                                         \
    InstantiateInvokeMulInplaceDevCScale_For(Pixel##type##C2);                                                         \
    InstantiateInvokeMulInplaceDevCScale_For(Pixel##type##C3);                                                         \
    InstantiateInvokeMulInplaceDevCScale_For(Pixel##type##C4);                                                         \
    InstantiateInvokeMulInplaceDevCScale_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
