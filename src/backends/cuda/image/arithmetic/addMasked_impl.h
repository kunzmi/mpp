#if MPP_ENABLE_CUDA_BACKEND

#include "addMasked.h"
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
void InvokeAddSrcSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                         const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Add<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = add_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addSrcSrcSIMD = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Add<ComputeT>, RoundingMode::None,
                                                ComputeT_SIMD, simdOP_t>;

            const mpp::Add<ComputeT> op;
            const simdOP_t opSIMD;

            const addSrcSrcSIMD functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addSrcSrcSIMD>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                  aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Add<ComputeT>, RoundingMode::None>;

            const mpp::Add<ComputeT> op;

            const addSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addSrcSrc>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                              aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using add_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeAddSrcSrcMask_For(typeSrcIsTypeDst)                                                           \
    template void                                                                                                      \
    InvokeAddSrcSrcMask<typeSrcIsTypeDst, add_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                   \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddSrcSrcMask(type)                                                                 \
    InstantiateInvokeAddSrcSrcMask_For(Pixel##type##C1);                                                               \
    InstantiateInvokeAddSrcSrcMask_For(Pixel##type##C2);                                                               \
    InstantiateInvokeAddSrcSrcMask_For(Pixel##type##C3);                                                               \
    InstantiateInvokeAddSrcSrcMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddSrcSrcMask(type)                                                               \
    InstantiateInvokeAddSrcSrcMask_For(Pixel##type##C1);                                                               \
    InstantiateInvokeAddSrcSrcMask_For(Pixel##type##C2);                                                               \
    InstantiateInvokeAddSrcSrcMask_For(Pixel##type##C3);                                                               \
    InstantiateInvokeAddSrcSrcMask_For(Pixel##type##C4);                                                               \
    InstantiateInvokeAddSrcSrcMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddSrcSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc1, size_t aPitchSrc1,
                              const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst, size_t aPitchDst,
                              scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeAddSrcSrcMask(aMask, aPitchMask, aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aDst, aPitchDst, aSize,
                                aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addSrcSrcScale =
            SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Add<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addSrcSrcScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeAddSrcSrcScaleMask_For(typeSrcIsTypeDst)                                                      \
    template void                                                                                                      \
    InvokeAddSrcSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1,                   \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddSrcSrcScaleMask(type)                                                            \
    InstantiateInvokeAddSrcSrcScaleMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeAddSrcSrcScaleMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeAddSrcSrcScaleMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeAddSrcSrcScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddSrcSrcScaleMask(type)                                                          \
    InstantiateInvokeAddSrcSrcScaleMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeAddSrcSrcScaleMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeAddSrcSrcScaleMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeAddSrcSrcScaleMask_For(Pixel##type##C4);                                                          \
    InstantiateInvokeAddSrcSrcScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddSrcCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                       const SrcT &aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                       const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Add<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = add_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addSrcCSIMD = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Add<ComputeT>,
                                                   RoundingMode::None, Tupel<ComputeT_SIMD, TupelSize>, simdOP_t>;

            const mpp::Add<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT_SIMD, TupelSize> tupelConstant =
                Tupel<ComputeT_SIMD, TupelSize>::GetConstant(static_cast<ComputeT_SIMD>(aConst));

            const addSrcCSIMD functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addSrcCSIMD>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Add<ComputeT>, RoundingMode::None>;

            const mpp::Add<ComputeT> op;

            const addSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addSrcC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                            aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using add_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeAddSrcCMask_For(typeSrcIsTypeDst)                                                             \
    template void                                                                                                      \
    InvokeAddSrcCMask<typeSrcIsTypeDst, add_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddSrcCMask(type)                                                                   \
    InstantiateInvokeAddSrcCMask_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeAddSrcCMask_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeAddSrcCMask_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeAddSrcCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddSrcCMask(type)                                                                 \
    InstantiateInvokeAddSrcCMask_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeAddSrcCMask_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeAddSrcCMask_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeAddSrcCMask_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeAddSrcCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddSrcCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                            const SrcT &aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                            const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeAddSrcCMask(aMask, aPitchMask, aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Add<ComputeT>,
                                                     RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addSrcCScale>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                             aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeAddSrcCScaleMask_For(typeSrcIsTypeDst)                                                        \
    template void                                                                                                      \
    InvokeAddSrcCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddSrcCScaleMask(type)                                                              \
    InstantiateInvokeAddSrcCScaleMask_For(Pixel##type##C1);                                                            \
    InstantiateInvokeAddSrcCScaleMask_For(Pixel##type##C2);                                                            \
    InstantiateInvokeAddSrcCScaleMask_For(Pixel##type##C3);                                                            \
    InstantiateInvokeAddSrcCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddSrcCScaleMask(type)                                                            \
    InstantiateInvokeAddSrcCScaleMask_For(Pixel##type##C1);                                                            \
    InstantiateInvokeAddSrcCScaleMask_For(Pixel##type##C2);                                                            \
    InstantiateInvokeAddSrcCScaleMask_For(Pixel##type##C3);                                                            \
    InstantiateInvokeAddSrcCScaleMask_For(Pixel##type##C4);                                                            \
    InstantiateInvokeAddSrcCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddSrcDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                          const SrcT *aConst, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
        using addSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Add<ComputeT>, RoundingMode::None>;

        const mpp::Add<ComputeT> op;

        const addSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addSrcDevC>(aMask, aPitchMask, aDst, aPitchDst, aSize,
                                                                           aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using add_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeAddSrcDevCMask_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeAddSrcDevCMask<typeSrcIsTypeDst, add_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddSrcDevCMask(type)                                                                \
    InstantiateInvokeAddSrcDevCMask_For(Pixel##type##C1);                                                              \
    InstantiateInvokeAddSrcDevCMask_For(Pixel##type##C2);                                                              \
    InstantiateInvokeAddSrcDevCMask_For(Pixel##type##C3);                                                              \
    InstantiateInvokeAddSrcDevCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddSrcDevCMask(type)                                                              \
    InstantiateInvokeAddSrcDevCMask_For(Pixel##type##C1);                                                              \
    InstantiateInvokeAddSrcDevCMask_For(Pixel##type##C2);                                                              \
    InstantiateInvokeAddSrcDevCMask_For(Pixel##type##C3);                                                              \
    InstantiateInvokeAddSrcDevCMask_For(Pixel##type##C4);                                                              \
    InstantiateInvokeAddSrcDevCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddSrcDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                               const SrcT *aConst, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor,
                               const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeAddSrcDevCMask(aMask, aPitchMask, aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Add<ComputeT>,
                                                           RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addSrcDevCScale>(aMask, aPitchMask, aDst, aPitchDst,
                                                                                aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeAddSrcDevCScaleMask_For(typeSrcIsTypeDst)                                                     \
    template void                                                                                                      \
    InvokeAddSrcDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        const Pixel8uC1 *aMask, size_t aPitchMask, const typeSrcIsTypeDst *aSrc, size_t aPitchSrc,                     \
        const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddSrcDevCScaleMask(type)                                                           \
    InstantiateInvokeAddSrcDevCScaleMask_For(Pixel##type##C1);                                                         \
    InstantiateInvokeAddSrcDevCScaleMask_For(Pixel##type##C2);                                                         \
    InstantiateInvokeAddSrcDevCScaleMask_For(Pixel##type##C3);                                                         \
    InstantiateInvokeAddSrcDevCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddSrcDevCScaleMask(type)                                                         \
    InstantiateInvokeAddSrcDevCScaleMask_For(Pixel##type##C1);                                                         \
    InstantiateInvokeAddSrcDevCScaleMask_For(Pixel##type##C2);                                                         \
    InstantiateInvokeAddSrcDevCScaleMask_For(Pixel##type##C3);                                                         \
    InstantiateInvokeAddSrcDevCScaleMask_For(Pixel##type##C4);                                                         \
    InstantiateInvokeAddSrcDevCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                             const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Add<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = add_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Add<ComputeT>,
                                                        RoundingMode::None, ComputeT_SIMD, simdOP_t>;

            const mpp::Add<ComputeT> op;
            const simdOP_t opSIMD;

            const addInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addInplaceSrcSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Add<ComputeT>, RoundingMode::None>;

            const mpp::Add<ComputeT> op;

            const addInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addInplaceSrc>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using add_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeAddInplaceSrcMask_For(typeSrcIsTypeDst)                                                       \
    template void                                                                                                      \
    InvokeAddInplaceSrcMask<typeSrcIsTypeDst, add_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddInplaceSrcMask(type)                                                             \
    InstantiateInvokeAddInplaceSrcMask_For(Pixel##type##C1);                                                           \
    InstantiateInvokeAddInplaceSrcMask_For(Pixel##type##C2);                                                           \
    InstantiateInvokeAddInplaceSrcMask_For(Pixel##type##C3);                                                           \
    InstantiateInvokeAddInplaceSrcMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(type)                                                           \
    InstantiateInvokeAddInplaceSrcMask_For(Pixel##type##C1);                                                           \
    InstantiateInvokeAddInplaceSrcMask_For(Pixel##type##C2);                                                           \
    InstantiateInvokeAddInplaceSrcMask_For(Pixel##type##C3);                                                           \
    InstantiateInvokeAddInplaceSrcMask_For(Pixel##type##C4);                                                           \
    InstantiateInvokeAddInplaceSrcMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddInplaceSrcScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                  const SrcT *aSrc2, size_t aPitchSrc2, scalefactor_t<ComputeT> aScaleFactor,
                                  const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeAddInplaceSrcMask(aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSrc2, aPitchSrc2, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Add<ComputeT>,
                                                          RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addInplaceSrcScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeAddInplaceSrcScaleMask_For(typeSrcIsTypeDst)                                                  \
    template void                                                                                                      \
    InvokeAddInplaceSrcScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,                                                              \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddInplaceSrcScaleMask(type)                                                        \
    InstantiateInvokeAddInplaceSrcScaleMask_For(Pixel##type##C1);                                                      \
    InstantiateInvokeAddInplaceSrcScaleMask_For(Pixel##type##C2);                                                      \
    InstantiateInvokeAddInplaceSrcScaleMask_For(Pixel##type##C3);                                                      \
    InstantiateInvokeAddInplaceSrcScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddInplaceSrcScaleMask(type)                                                      \
    InstantiateInvokeAddInplaceSrcScaleMask_For(Pixel##type##C1);                                                      \
    InstantiateInvokeAddInplaceSrcScaleMask_For(Pixel##type##C2);                                                      \
    InstantiateInvokeAddInplaceSrcScaleMask_For(Pixel##type##C3);                                                      \
    InstantiateInvokeAddInplaceSrcScaleMask_For(Pixel##type##C4);                                                      \
    InstantiateInvokeAddInplaceSrcScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddInplaceCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                           const SrcT &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Add<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = add_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addInplaceCSIMD =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Add<ComputeT>, RoundingMode::None,
                                       Tupel<ComputeT_SIMD, TupelSize>, simdOP_t>;

            const mpp::Add<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT_SIMD, TupelSize> tupelConstant =
                Tupel<ComputeT_SIMD, TupelSize>::GetConstant(static_cast<ComputeT_SIMD>(aConst));

            const addInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addInplaceCSIMD>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Add<ComputeT>, RoundingMode::None>;

            const mpp::Add<ComputeT> op;

            const addInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addInplaceC>(
                aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using add_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeAddInplaceCMask_For(typeSrcIsTypeDst)                                                         \
    template void                                                                                                      \
    InvokeAddInplaceCMask<typeSrcIsTypeDst, add_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddInplaceCMask(type)                                                               \
    InstantiateInvokeAddInplaceCMask_For(Pixel##type##C1);                                                             \
    InstantiateInvokeAddInplaceCMask_For(Pixel##type##C2);                                                             \
    InstantiateInvokeAddInplaceCMask_For(Pixel##type##C3);                                                             \
    InstantiateInvokeAddInplaceCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddInplaceCMask(type)                                                             \
    InstantiateInvokeAddInplaceCMask_For(Pixel##type##C1);                                                             \
    InstantiateInvokeAddInplaceCMask_For(Pixel##type##C2);                                                             \
    InstantiateInvokeAddInplaceCMask_For(Pixel##type##C3);                                                             \
    InstantiateInvokeAddInplaceCMask_For(Pixel##type##C4);                                                             \
    InstantiateInvokeAddInplaceCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddInplaceCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                const SrcT &aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeAddInplaceCMask(aMask, aPitchMask, aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addInplaceCScale =
            InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Add<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addInplaceCScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeAddInplaceCScaleMask_For(typeSrcIsTypeDst)                                                    \
    template void                                                                                                      \
    InvokeAddInplaceCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(      \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst &aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddInplaceCScaleMask(type)                                                          \
    InstantiateInvokeAddInplaceCScaleMask_For(Pixel##type##C1);                                                        \
    InstantiateInvokeAddInplaceCScaleMask_For(Pixel##type##C2);                                                        \
    InstantiateInvokeAddInplaceCScaleMask_For(Pixel##type##C3);                                                        \
    InstantiateInvokeAddInplaceCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddInplaceCScaleMask(type)                                                        \
    InstantiateInvokeAddInplaceCScaleMask_For(Pixel##type##C1);                                                        \
    InstantiateInvokeAddInplaceCScaleMask_For(Pixel##type##C2);                                                        \
    InstantiateInvokeAddInplaceCScaleMask_For(Pixel##type##C3);                                                        \
    InstantiateInvokeAddInplaceCScaleMask_For(Pixel##type##C4);                                                        \
    InstantiateInvokeAddInplaceCScaleMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddInplaceDevCMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                              const SrcT *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
        using addInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, mpp::Add<ComputeT>, RoundingMode::None>;

        const mpp::Add<ComputeT> op;

        const addInplaceDevC functor(aConst, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addInplaceDevC>(aMask, aPitchMask, aSrcDst, aPitchSrcDst,
                                                                               aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using add_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeAddInplaceDevCMask_For(typeSrcIsTypeDst)                                                      \
    template void InvokeAddInplaceDevCMask<typeSrcIsTypeDst, add_simd_vector_compute_type_for_t<typeSrcIsTypeDst>,     \
                                           typeSrcIsTypeDst>(                                                          \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddInplaceDevCMask(type)                                                            \
    InstantiateInvokeAddInplaceDevCMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeAddInplaceDevCMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeAddInplaceDevCMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeAddInplaceDevCMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(type)                                                          \
    InstantiateInvokeAddInplaceDevCMask_For(Pixel##type##C1);                                                          \
    InstantiateInvokeAddInplaceDevCMask_For(Pixel##type##C2);                                                          \
    InstantiateInvokeAddInplaceDevCMask_For(Pixel##type##C3);                                                          \
    InstantiateInvokeAddInplaceDevCMask_For(Pixel##type##C4);                                                          \
    InstantiateInvokeAddInplaceDevCMask_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddInplaceDevCScaleMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                   const SrcT *aConst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                   const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeAddInplaceDevCMask(aMask, aPitchMask, aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addInplaceDevCScale = InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Add<ComputeT>,
                                                                   RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addInplaceDevCScale functor(aConst, op, aScaleFactor);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addInplaceDevCScale>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeAddInplaceDevCScaleMask_For(typeSrcIsTypeDst)                                                 \
    template void                                                                                                      \
    InvokeAddInplaceDevCScaleMask<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const Pixel8uC1 *aMask, size_t aPitchMask, typeSrcIsTypeDst *aSrcDst, size_t aPitchSrcDst,                     \
        const typeSrcIsTypeDst *aConst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,      \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddInplaceDevCScaleMask(type)                                                       \
    InstantiateInvokeAddInplaceDevCScaleMask_For(Pixel##type##C1);                                                     \
    InstantiateInvokeAddInplaceDevCScaleMask_For(Pixel##type##C2);                                                     \
    InstantiateInvokeAddInplaceDevCScaleMask_For(Pixel##type##C3);                                                     \
    InstantiateInvokeAddInplaceDevCScaleMask_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddInplaceDevCScaleMask(type)                                                     \
    InstantiateInvokeAddInplaceDevCScaleMask_For(Pixel##type##C1);                                                     \
    InstantiateInvokeAddInplaceDevCScaleMask_For(Pixel##type##C2);                                                     \
    InstantiateInvokeAddInplaceDevCScaleMask_For(Pixel##type##C3);                                                     \
    InstantiateInvokeAddInplaceDevCScaleMask_For(Pixel##type##C4);                                                     \
    InstantiateInvokeAddInplaceDevCScaleMask_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
