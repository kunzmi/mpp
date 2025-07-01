#if MPP_ENABLE_CUDA_BACKEND

#include "subMasked.h"
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
void InvokeSubSrcSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                         const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Sub<Tupel<DstT, TupelSize>>;

        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = sub_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subSrcSrcSIMD = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None,
                                                ComputeT_SIMD, simdOP_t>;

            const mpp::Sub<ComputeT> op;
            const simdOP_t opSIMD;

            const subSrcSrcSIMD functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subSrcSrcSIMD>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                  aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None>;

            const mpp::Sub<ComputeT> op;

            const subSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subSrcSrc>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                              aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubSrcSrcMask_For(typeSrcIsTypeDst)                                                           \
    template void                                                                                                      \
    InvokeSubSrcSrcMask<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                   \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubSrcSrcMask(type)                                                                 \
    InstantiateInvokeSubSrcSrcMask_For(Pixel##type##C1);                                                               \
    InstantiateInvokeSubSrcSrcMask_For(Pixel##type##C2);                                                               \
    InstantiateInvokeSubSrcSrcMask_For(Pixel##type##C3);                                                               \
    InstantiateInvokeSubSrcSrcMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubSrcSrcMask(type)                                                               \
    InstantiateInvokeSubSrcSrcMask_For(Pixel##type##C1);                                                               \
    InstantiateInvokeSubSrcSrcMask_For(Pixel##type##C2);                                                               \
    InstantiateInvokeSubSrcSrcMask_For(Pixel##type##C3);                                                               \
    InstantiateInvokeSubSrcSrcMask_For(Pixel##type##C4);                                                               \
    InstantiateInvokeSubSrcSrcMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubSrcSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                              const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst,
                              scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubSrcSrcMask(aMask, aPitchMask, aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aDst, aPitchDst, aSize,
                                aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subSrcSrcScale =
            SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subSrcSrcScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubSrcSrcScaleMask_For(typeSrcIsTypeDst)                                                      \
    template void                                                                                                      \
    InvokeSubSrcSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                   \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubSrcSrcScaleMask(type)                                                            \
    InstantiateInvokeSubSrcSrcScaleMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeSubSrcSrcScaleMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeSubSrcSrcScaleMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeSubSrcSrcScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubSrcSrcScaleMask(type)                                                          \
    InstantiateInvokeSubSrcSrcScaleMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeSubSrcSrcScaleMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeSubSrcSrcScaleMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeSubSrcSrcScaleMask_For(Pixel##type##C4);                                                          \
    InstantiateInvokeSubSrcSrcScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubSrcCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                       const SrcT &aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                       const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Sub<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = sub_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subSrcCSIMD = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>,
                                                   RoundingMode::None, Tupel<ComputeT_SIMD, TupelSize>, simdOP_t>;

            const mpp::Sub<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT_SIMD, TupelSize> tupelConstant =
                Tupel<ComputeT_SIMD, TupelSize>::GetConstant(static_cast<ComputeT_SIMD>(aConst));

            const subSrcCSIMD functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subSrcCSIMD>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None>;

            const mpp::Sub<ComputeT> op;

            const subSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subSrcC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                            aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubSrcCMask_For(typeSrcIsTypeDst)                                                             \
    template void                                                                                                      \
    InvokeSubSrcCMask<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubSrcCMask(type)                                                                   \
    InstantiateInvokeSubSrcCMask_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeSubSrcCMask_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeSubSrcCMask_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeSubSrcCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubSrcCMask(type)                                                                 \
    InstantiateInvokeSubSrcCMask_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeSubSrcCMask_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeSubSrcCMask_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeSubSrcCMask_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeSubSrcCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubSrcCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                            const SrcT &aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                            const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubSrcCMask(aMask, aPitchMask, aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>,
                                                     RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subSrcCScale>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                             aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubSrcCScaleMask_For(typeSrcIsTypeDst)                                                        \
    template void                                                                                                      \
    InvokeSubSrcCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubSrcCScaleMask(type)                                                              \
    InstantiateInvokeSubSrcCScaleMask_For(Pixel##type##C1);                                                            \
    InstantiateInvokeSubSrcCScaleMask_For(Pixel##type##C2);                                                            \
    InstantiateInvokeSubSrcCScaleMask_For(Pixel##type##C3);                                                            \
    InstantiateInvokeSubSrcCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubSrcCScaleMask(type)                                                            \
    InstantiateInvokeSubSrcCScaleMask_For(Pixel##type##C1);                                                            \
    InstantiateInvokeSubSrcCScaleMask_For(Pixel##type##C2);                                                            \
    InstantiateInvokeSubSrcCScaleMask_For(Pixel##type##C3);                                                            \
    InstantiateInvokeSubSrcCScaleMask_For(Pixel##type##C4);                                                            \
    InstantiateInvokeSubSrcCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubSrcDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                          const SrcT *aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
        using subSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None>;

        const mpp::Sub<ComputeT> op;

        const subSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subSrcDevC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                           aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubSrcDevCMask_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeSubSrcDevCMask<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubSrcDevCMask(type)                                                                \
    InstantiateInvokeSubSrcDevCMask_For(Pixel##type##C1);                                                              \
    InstantiateInvokeSubSrcDevCMask_For(Pixel##type##C2);                                                              \
    InstantiateInvokeSubSrcDevCMask_For(Pixel##type##C3);                                                              \
    InstantiateInvokeSubSrcDevCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubSrcDevCMask(type)                                                              \
    InstantiateInvokeSubSrcDevCMask_For(Pixel##type##C1);                                                              \
    InstantiateInvokeSubSrcDevCMask_For(Pixel##type##C2);                                                              \
    InstantiateInvokeSubSrcDevCMask_For(Pixel##type##C3);                                                              \
    InstantiateInvokeSubSrcDevCMask_For(Pixel##type##C4);                                                              \
    InstantiateInvokeSubSrcDevCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubSrcDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                               const SrcT *aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                               const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubSrcDevCMask(aMask, aPitchMask, aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>,
                                                           RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subSrcDevCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubSrcDevCScaleMask_For(typeSrcIsTypeDst)                                                     \
    template void                                                                                                      \
    InvokeSubSrcDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubSrcDevCScaleMask(type)                                                           \
    InstantiateInvokeSubSrcDevCScaleMask_For(Pixel##type##C1);                                                         \
    InstantiateInvokeSubSrcDevCScaleMask_For(Pixel##type##C2);                                                         \
    InstantiateInvokeSubSrcDevCScaleMask_For(Pixel##type##C3);                                                         \
    InstantiateInvokeSubSrcDevCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubSrcDevCScaleMask(type)                                                         \
    InstantiateInvokeSubSrcDevCScaleMask_For(Pixel##type##C1);                                                         \
    InstantiateInvokeSubSrcDevCScaleMask_For(Pixel##type##C2);                                                         \
    InstantiateInvokeSubSrcDevCScaleMask_For(Pixel##type##C3);                                                         \
    InstantiateInvokeSubSrcDevCScaleMask_For(Pixel##type##C4);                                                         \
    InstantiateInvokeSubSrcDevCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                             const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Sub<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = sub_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>,
                                                        RoundingMode::None, ComputeT_SIMD, simdOP_t>;

            const mpp::Sub<ComputeT> op;
            const simdOP_t opSIMD;

            const subInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceSrcSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None>;

            const mpp::Sub<ComputeT> op;

            const subInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceSrc>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubInplaceSrcMask_For(typeSrcIsTypeDst)                                                       \
    template void                                                                                                      \
    InvokeSubInplaceSrcMask<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInplaceSrcMask(type)                                                             \
    InstantiateInvokeSubInplaceSrcMask_For(Pixel##type##C1);                                                           \
    InstantiateInvokeSubInplaceSrcMask_For(Pixel##type##C2);                                                           \
    InstantiateInvokeSubInplaceSrcMask_For(Pixel##type##C3);                                                           \
    InstantiateInvokeSubInplaceSrcMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInplaceSrcMask(type)                                                           \
    InstantiateInvokeSubInplaceSrcMask_For(Pixel##type##C1);                                                           \
    InstantiateInvokeSubInplaceSrcMask_For(Pixel##type##C2);                                                           \
    InstantiateInvokeSubInplaceSrcMask_For(Pixel##type##C3);                                                           \
    InstantiateInvokeSubInplaceSrcMask_For(Pixel##type##C4);                                                           \
    InstantiateInvokeSubInplaceSrcMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                  const SrcT *aSrc2, size_t aPitchSrc2, scalefactor_t<ComputeT> aScaleFactor,
                                  const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubInplaceSrcMask(aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSrc2, aPitchSrc2, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>,
                                                          RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceSrcScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubInplaceSrcScaleMask_For(typeSrcIsTypeDst)                                                  \
    template void                                                                                                      \
    InvokeSubInplaceSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,                                                              \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInplaceSrcScaleMask(type)                                                        \
    InstantiateInvokeSubInplaceSrcScaleMask_For(Pixel##type##C1);                                                      \
    InstantiateInvokeSubInplaceSrcScaleMask_For(Pixel##type##C2);                                                      \
    InstantiateInvokeSubInplaceSrcScaleMask_For(Pixel##type##C3);                                                      \
    InstantiateInvokeSubInplaceSrcScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInplaceSrcScaleMask(type)                                                      \
    InstantiateInvokeSubInplaceSrcScaleMask_For(Pixel##type##C1);                                                      \
    InstantiateInvokeSubInplaceSrcScaleMask_For(Pixel##type##C2);                                                      \
    InstantiateInvokeSubInplaceSrcScaleMask_For(Pixel##type##C3);                                                      \
    InstantiateInvokeSubInplaceSrcScaleMask_For(Pixel##type##C4);                                                      \
    InstantiateInvokeSubInplaceSrcScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                           const SrcT &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Sub<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = sub_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subInplaceCSIMD =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None,
                                       Tupel<ComputeT_SIMD, TupelSize>, simdOP_t>;

            const mpp::Sub<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT_SIMD, TupelSize> tupelConstant =
                Tupel<ComputeT_SIMD, TupelSize>::GetConstant(static_cast<ComputeT_SIMD>(aConst));

            const subInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceCSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None>;

            const mpp::Sub<ComputeT> op;

            const subInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceC>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubInplaceCMask_For(typeSrcIsTypeDst)                                                         \
    template void                                                                                                      \
    InvokeSubInplaceCMask<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInplaceCMask(type)                                                               \
    InstantiateInvokeSubInplaceCMask_For(Pixel##type##C1);                                                             \
    InstantiateInvokeSubInplaceCMask_For(Pixel##type##C2);                                                             \
    InstantiateInvokeSubInplaceCMask_For(Pixel##type##C3);                                                             \
    InstantiateInvokeSubInplaceCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInplaceCMask(type)                                                             \
    InstantiateInvokeSubInplaceCMask_For(Pixel##type##C1);                                                             \
    InstantiateInvokeSubInplaceCMask_For(Pixel##type##C2);                                                             \
    InstantiateInvokeSubInplaceCMask_For(Pixel##type##C3);                                                             \
    InstantiateInvokeSubInplaceCMask_For(Pixel##type##C4);                                                             \
    InstantiateInvokeSubInplaceCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT &aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubInplaceCMask(aMask, aPitchMask, aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subInplaceCScale =
            InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceCScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubInplaceCScaleMask_For(typeSrcIsTypeDst)                                                    \
    template void                                                                                                      \
    InvokeSubInplaceCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(      \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInplaceCScaleMask(type)                                                          \
    InstantiateInvokeSubInplaceCScaleMask_For(Pixel##type##C1);                                                        \
    InstantiateInvokeSubInplaceCScaleMask_For(Pixel##type##C2);                                                        \
    InstantiateInvokeSubInplaceCScaleMask_For(Pixel##type##C3);                                                        \
    InstantiateInvokeSubInplaceCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInplaceCScaleMask(type)                                                        \
    InstantiateInvokeSubInplaceCScaleMask_For(Pixel##type##C1);                                                        \
    InstantiateInvokeSubInplaceCScaleMask_For(Pixel##type##C2);                                                        \
    InstantiateInvokeSubInplaceCScaleMask_For(Pixel##type##C3);                                                        \
    InstantiateInvokeSubInplaceCScaleMask_For(Pixel##type##C4);                                                        \
    InstantiateInvokeSubInplaceCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                              const SrcT *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
        using subInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None>;

        const mpp::Sub<ComputeT> op;

        const subInplaceDevC functor(aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceDevC>(aMask, aPitchMask, aSrcDst, aPitchSrcDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubInplaceDevCMask_For(typeSrcIsTypeDst)                                                      \
    template void InvokeSubInplaceDevCMask<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>,     \
                                           typeSrcIsTypeDst>(                                                          \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInplaceDevCMask(type)                                                            \
    InstantiateInvokeSubInplaceDevCMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeSubInplaceDevCMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeSubInplaceDevCMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeSubInplaceDevCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInplaceDevCMask(type)                                                          \
    InstantiateInvokeSubInplaceDevCMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeSubInplaceDevCMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeSubInplaceDevCMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeSubInplaceDevCMask_For(Pixel##type##C4);                                                          \
    InstantiateInvokeSubInplaceDevCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                   const SrcT *aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                   const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubInplaceDevCMask(aMask, aPitchMask, aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subInplaceDevCScale = InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Sub<ComputeT>,
                                                                   RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subInplaceDevCScale functor(aConst, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceDevCScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubInplaceDevCScaleMask_For(typeSrcIsTypeDst)                                                 \
    template void                                                                                                      \
    InvokeSubInplaceDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInplaceDevCScaleMask(type)                                                       \
    InstantiateInvokeSubInplaceDevCScaleMask_For(Pixel##type##C1);                                                     \
    InstantiateInvokeSubInplaceDevCScaleMask_For(Pixel##type##C2);                                                     \
    InstantiateInvokeSubInplaceDevCScaleMask_For(Pixel##type##C3);                                                     \
    InstantiateInvokeSubInplaceDevCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInplaceDevCScaleMask(type)                                                     \
    InstantiateInvokeSubInplaceDevCScaleMask_For(Pixel##type##C1);                                                     \
    InstantiateInvokeSubInplaceDevCScaleMask_For(Pixel##type##C2);                                                     \
    InstantiateInvokeSubInplaceDevCScaleMask_For(Pixel##type##C3);                                                     \
    InstantiateInvokeSubInplaceDevCScaleMask_For(Pixel##type##C4);                                                     \
    InstantiateInvokeSubInplaceDevCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInvInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::SubInv<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = sub_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::SubInv<ComputeT>,
                                                        RoundingMode::None, ComputeT_SIMD, simdOP_t>;

            const mpp::SubInv<ComputeT> op;
            const simdOP_t opSIMD;

            const subInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceSrcSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::SubInv<ComputeT>, RoundingMode::None>;

            const mpp::SubInv<ComputeT> op;

            const subInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceSrc>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubInvInplaceSrcMask_For(typeSrcIsTypeDst)                                                    \
    template void InvokeSubInvInplaceSrcMask<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>,   \
                                             typeSrcIsTypeDst>(                                                        \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInvInplaceSrcMask(type)                                                          \
    InstantiateInvokeSubInvInplaceSrcMask_For(Pixel##type##C1);                                                        \
    InstantiateInvokeSubInvInplaceSrcMask_For(Pixel##type##C2);                                                        \
    InstantiateInvokeSubInvInplaceSrcMask_For(Pixel##type##C3);                                                        \
    InstantiateInvokeSubInvInplaceSrcMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInvInplaceSrcMask(type)                                                        \
    InstantiateInvokeSubInvInplaceSrcMask_For(Pixel##type##C1);                                                        \
    InstantiateInvokeSubInvInplaceSrcMask_For(Pixel##type##C2);                                                        \
    InstantiateInvokeSubInvInplaceSrcMask_For(Pixel##type##C3);                                                        \
    InstantiateInvokeSubInvInplaceSrcMask_For(Pixel##type##C4);                                                        \
    InstantiateInvokeSubInvInplaceSrcMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInvInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                     const SrcT *aSrc2, size_t aPitchSrc2, scalefactor_t<ComputeT> aScaleFactor,
                                     const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubInvInplaceSrcMask(aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSrc2, aPitchSrc2, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::SubInv<ComputeT>,
                                                          RoundingMode::NearestTiesToEven>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceSrcScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubInvInplaceSrcScaleMask_For(typeSrcIsTypeDst)                                               \
    template void                                                                                                      \
    InvokeSubInvInplaceSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,                                                              \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInvInplaceSrcScaleMask(type)                                                     \
    InstantiateInvokeSubInvInplaceSrcScaleMask_For(Pixel##type##C1);                                                   \
    InstantiateInvokeSubInvInplaceSrcScaleMask_For(Pixel##type##C2);                                                   \
    InstantiateInvokeSubInvInplaceSrcScaleMask_For(Pixel##type##C3);                                                   \
    InstantiateInvokeSubInvInplaceSrcScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScaleMask(type)                                                   \
    InstantiateInvokeSubInvInplaceSrcScaleMask_For(Pixel##type##C1);                                                   \
    InstantiateInvokeSubInvInplaceSrcScaleMask_For(Pixel##type##C2);                                                   \
    InstantiateInvokeSubInvInplaceSrcScaleMask_For(Pixel##type##C3);                                                   \
    InstantiateInvokeSubInvInplaceSrcScaleMask_For(Pixel##type##C4);                                                   \
    InstantiateInvokeSubInvInplaceSrcScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInvInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                              const SrcT &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::SubInv<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = sub_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subInplaceCSIMD =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::SubInv<ComputeT>, RoundingMode::None,
                                       Tupel<ComputeT_SIMD, TupelSize>, simdOP_t>;

            const mpp::SubInv<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT_SIMD, TupelSize> tupelConstant =
                Tupel<ComputeT_SIMD, TupelSize>::GetConstant(static_cast<ComputeT_SIMD>(aConst));

            const subInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceCSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::SubInv<ComputeT>, RoundingMode::None>;

            const mpp::SubInv<ComputeT> op;

            const subInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceC>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubInvInplaceCMask_For(typeSrcIsTypeDst)                                                      \
    template void InvokeSubInvInplaceCMask<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>,     \
                                           typeSrcIsTypeDst>(                                                          \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInvInplaceCMask(type)                                                            \
    InstantiateInvokeSubInvInplaceCMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeSubInvInplaceCMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeSubInvInplaceCMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeSubInvInplaceCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInvInplaceCMask(type)                                                          \
    InstantiateInvokeSubInvInplaceCMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeSubInvInplaceCMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeSubInvInplaceCMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeSubInvInplaceCMask_For(Pixel##type##C4);                                                          \
    InstantiateInvokeSubInvInplaceCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInvInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                   const SrcT &aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                   const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubInvInplaceCMask(aMask, aPitchMask, aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::SubInv<ComputeT>,
                                                             RoundingMode::NearestTiesToEven>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceCScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubInvInplaceCScaleMask_For(typeSrcIsTypeDst)                                                 \
    template void                                                                                                      \
    InvokeSubInvInplaceCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInvInplaceCScaleMask(type)                                                       \
    InstantiateInvokeSubInvInplaceCScaleMask_For(Pixel##type##C1);                                                     \
    InstantiateInvokeSubInvInplaceCScaleMask_For(Pixel##type##C2);                                                     \
    InstantiateInvokeSubInvInplaceCScaleMask_For(Pixel##type##C3);                                                     \
    InstantiateInvokeSubInvInplaceCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInvInplaceCScaleMask(type)                                                     \
    InstantiateInvokeSubInvInplaceCScaleMask_For(Pixel##type##C1);                                                     \
    InstantiateInvokeSubInvInplaceCScaleMask_For(Pixel##type##C2);                                                     \
    InstantiateInvokeSubInvInplaceCScaleMask_For(Pixel##type##C3);                                                     \
    InstantiateInvokeSubInvInplaceCScaleMask_For(Pixel##type##C4);                                                     \
    InstantiateInvokeSubInvInplaceCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInvInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                 const SrcT *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
        using subInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::SubInv<ComputeT>, RoundingMode::None>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceDevC functor(aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceDevC>(aMask, aPitchMask, aSrcDst, aPitchSrcDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubInvInplaceDevCMask_For(typeSrcIsTypeDst)                                                   \
    template void InvokeSubInvInplaceDevCMask<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>,  \
                                              typeSrcIsTypeDst>(                                                       \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInvInplaceDevCMask(type)                                                         \
    InstantiateInvokeSubInvInplaceDevCMask_For(Pixel##type##C1);                                                       \
    InstantiateInvokeSubInvInplaceDevCMask_For(Pixel##type##C2);                                                       \
    InstantiateInvokeSubInvInplaceDevCMask_For(Pixel##type##C3);                                                       \
    InstantiateInvokeSubInvInplaceDevCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInvInplaceDevCMask(type)                                                       \
    InstantiateInvokeSubInvInplaceDevCMask_For(Pixel##type##C1);                                                       \
    InstantiateInvokeSubInvInplaceDevCMask_For(Pixel##type##C2);                                                       \
    InstantiateInvokeSubInvInplaceDevCMask_For(Pixel##type##C3);                                                       \
    InstantiateInvokeSubInvInplaceDevCMask_For(Pixel##type##C4);                                                       \
    InstantiateInvokeSubInvInplaceDevCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInvInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                      const SrcT *aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                      const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubInvInplaceDevCMask(aMask, aPitchMask, aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subInplaceDevCScale = InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::SubInv<ComputeT>,
                                                                   RoundingMode::NearestTiesToEven>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceDevCScale functor(aConst, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, subInplaceDevCScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubInvInplaceDevCScaleMask_For(typeSrcIsTypeDst)                                              \
    template void InvokeSubInvInplaceDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>,     \
                                                   typeSrcIsTypeDst>(                                                  \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInvInplaceDevCScaleMask(type)                                                    \
    InstantiateInvokeSubInvInplaceDevCScaleMask_For(Pixel##type##C1);                                                  \
    InstantiateInvokeSubInvInplaceDevCScaleMask_For(Pixel##type##C2);                                                  \
    InstantiateInvokeSubInvInplaceDevCScaleMask_For(Pixel##type##C3);                                                  \
    InstantiateInvokeSubInvInplaceDevCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScaleMask(type)                                                  \
    InstantiateInvokeSubInvInplaceDevCScaleMask_For(Pixel##type##C1);                                                  \
    InstantiateInvokeSubInvInplaceDevCScaleMask_For(Pixel##type##C2);                                                  \
    InstantiateInvokeSubInvInplaceDevCScaleMask_For(Pixel##type##C3);                                                  \
    InstantiateInvokeSubInvInplaceDevCScaleMask_For(Pixel##type##C4);                                                  \
    InstantiateInvokeSubInvInplaceDevCScaleMask_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
