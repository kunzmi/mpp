#if MPP_ENABLE_CUDA_BACKEND

#include "sub.h"
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
void InvokeSubSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                     size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
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

            InvokeForEachPixelKernelDefault<DstT, TupelSize, subSrcSrcSIMD>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                            functor);
        }
        else
        {
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None>;

            const mpp::Sub<ComputeT> op;

            const subSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, subSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubSrcSrc_For(typeSrcIsTypeDst)                                                               \
    template void                                                                                                      \
    InvokeSubSrcSrc<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(         \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubSrcSrc(type)                                                                     \
    InstantiateInvokeSubSrcSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeSubSrcSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeSubSrcSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeSubSrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubSrcSrc(type)                                                                   \
    InstantiateInvokeSubSrcSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeSubSrcSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeSubSrcSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeSubSrcSrc_For(Pixel##type##C4);                                                                   \
    InstantiateInvokeSubSrcSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubSrcSrcScale(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubSrcSrc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subSrcSrcScale =
            SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, subSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubSrcSrcScale_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeSubSrcSrcScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst,                                                                      \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubSrcSrcScale(type)                                                                \
    InstantiateInvokeSubSrcSrcScale_For(Pixel##type##C1);                                                              \
    InstantiateInvokeSubSrcSrcScale_For(Pixel##type##C2);                                                              \
    InstantiateInvokeSubSrcSrcScale_For(Pixel##type##C3);                                                              \
    InstantiateInvokeSubSrcSrcScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubSrcSrcScale(type)                                                              \
    InstantiateInvokeSubSrcSrcScale_For(Pixel##type##C1);                                                              \
    InstantiateInvokeSubSrcSrcScale_For(Pixel##type##C2);                                                              \
    InstantiateInvokeSubSrcSrcScale_For(Pixel##type##C3);                                                              \
    InstantiateInvokeSubSrcSrcScale_For(Pixel##type##C4);                                                              \
    InstantiateInvokeSubSrcSrcScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                   const Size2D &aSize, const StreamCtx &aStreamCtx)
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

            InvokeForEachPixelKernelDefault<DstT, TupelSize, subSrcCSIMD>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None>;

            const mpp::Sub<ComputeT> op;

            const subSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, subSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubSrcC_For(typeSrcIsTypeDst)                                                                 \
    template void                                                                                                      \
    InvokeSubSrcC<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubSrcC(type)                                                                       \
    InstantiateInvokeSubSrcC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeSubSrcC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeSubSrcC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeSubSrcC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubSrcC(type)                                                                     \
    InstantiateInvokeSubSrcC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeSubSrcC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeSubSrcC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeSubSrcC_For(Pixel##type##C4);                                                                     \
    InstantiateInvokeSubSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubSrcCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                        scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubSrcC(aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>,
                                                     RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, subSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubSrcCScale_For(typeSrcIsTypeDst)                                                            \
    template void                                                                                                      \
    InvokeSubSrcCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(              \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubSrcCScale(type)                                                                  \
    InstantiateInvokeSubSrcCScale_For(Pixel##type##C1);                                                                \
    InstantiateInvokeSubSrcCScale_For(Pixel##type##C2);                                                                \
    InstantiateInvokeSubSrcCScale_For(Pixel##type##C3);                                                                \
    InstantiateInvokeSubSrcCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubSrcCScale(type)                                                                \
    InstantiateInvokeSubSrcCScale_For(Pixel##type##C1);                                                                \
    InstantiateInvokeSubSrcCScale_For(Pixel##type##C2);                                                                \
    InstantiateInvokeSubSrcCScale_For(Pixel##type##C3);                                                                \
    InstantiateInvokeSubSrcCScale_For(Pixel##type##C4);                                                                \
    InstantiateInvokeSubSrcCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
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

        InvokeForEachPixelKernelDefault<DstT, TupelSize, subSrcDevC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubSrcDevC_For(typeSrcIsTypeDst)                                                              \
    template void                                                                                                      \
    InvokeSubSrcDevC<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubSrcDevC(type)                                                                    \
    InstantiateInvokeSubSrcDevC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeSubSrcDevC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeSubSrcDevC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeSubSrcDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubSrcDevC(type)                                                                  \
    InstantiateInvokeSubSrcDevC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeSubSrcDevC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeSubSrcDevC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeSubSrcDevC_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeSubSrcDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubSrcDevCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                           scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubSrcDevC(aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>,
                                                           RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, subSrcDevCScale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubSrcDevCScale_For(typeSrcIsTypeDst)                                                         \
    template void                                                                                                      \
    InvokeSubSrcDevCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubSrcDevCScale(type)                                                               \
    InstantiateInvokeSubSrcDevCScale_For(Pixel##type##C1);                                                             \
    InstantiateInvokeSubSrcDevCScale_For(Pixel##type##C2);                                                             \
    InstantiateInvokeSubSrcDevCScale_For(Pixel##type##C3);                                                             \
    InstantiateInvokeSubSrcDevCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubSrcDevCScale(type)                                                             \
    InstantiateInvokeSubSrcDevCScale_For(Pixel##type##C1);                                                             \
    InstantiateInvokeSubSrcDevCScale_For(Pixel##type##C2);                                                             \
    InstantiateInvokeSubSrcDevCScale_For(Pixel##type##C3);                                                             \
    InstantiateInvokeSubSrcDevCScale_For(Pixel##type##C4);                                                             \
    InstantiateInvokeSubSrcDevCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
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
            using subInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>,
                                                        RoundingMode::None, ComputeT_SIMD, simdOP_t>;

            const mpp::Sub<ComputeT> op;
            const simdOP_t opSIMD;

            const subInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceSrcSIMD>(aSrcDst, aPitchSrcDst, aSize,
                                                                                aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None>;

            const mpp::Sub<ComputeT> op;

            const subInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
        }
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubInplaceSrc_For(typeSrcIsTypeDst)                                                           \
    template void                                                                                                      \
    InvokeSubInplaceSrc<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInplaceSrc(type)                                                                 \
    InstantiateInvokeSubInplaceSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeSubInplaceSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeSubInplaceSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeSubInplaceSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInplaceSrc(type)                                                               \
    InstantiateInvokeSubInplaceSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeSubInplaceSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeSubInplaceSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeSubInplaceSrc_For(Pixel##type##C4);                                                               \
    InstantiateInvokeSubInplaceSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubInplaceSrc(aSrcDst, aPitchSrcDst, aSrc2, aPitchSrc2, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sub<ComputeT>,
                                                          RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                             functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubInplaceSrcScale_For(typeSrcIsTypeDst)                                                      \
    template void                                                                                                      \
    InvokeSubInplaceSrcScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInplaceSrcScale(type)                                                            \
    InstantiateInvokeSubInplaceSrcScale_For(Pixel##type##C1);                                                          \
    InstantiateInvokeSubInplaceSrcScale_For(Pixel##type##C2);                                                          \
    InstantiateInvokeSubInplaceSrcScale_For(Pixel##type##C3);                                                          \
    InstantiateInvokeSubInplaceSrcScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInplaceSrcScale(type)                                                          \
    InstantiateInvokeSubInplaceSrcScale_For(Pixel##type##C1);                                                          \
    InstantiateInvokeSubInplaceSrcScale_For(Pixel##type##C2);                                                          \
    InstantiateInvokeSubInplaceSrcScale_For(Pixel##type##C3);                                                          \
    InstantiateInvokeSubInplaceSrcScale_For(Pixel##type##C4);                                                          \
    InstantiateInvokeSubInplaceSrcScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
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
            using subInplaceCSIMD =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None,
                                       Tupel<ComputeT_SIMD, TupelSize>, simdOP_t>;

            const mpp::Sub<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT_SIMD, TupelSize> tupelConstant =
                Tupel<ComputeT_SIMD, TupelSize>::GetConstant(static_cast<ComputeT_SIMD>(aConst));

            const subInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceCSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
        }
        else
        {
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::None>;

            const mpp::Sub<ComputeT> op;

            const subInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                          functor);
        }
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubInplaceC_For(typeSrcIsTypeDst)                                                             \
    template void                                                                                                      \
    InvokeSubInplaceC<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInplaceC(type)                                                                   \
    InstantiateInvokeSubInplaceC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeSubInplaceC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeSubInplaceC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeSubInplaceC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInplaceC(type)                                                                 \
    InstantiateInvokeSubInplaceC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeSubInplaceC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeSubInplaceC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeSubInplaceC_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeSubInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst,
                            scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubInplaceC(aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subInplaceCScale =
            InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Sub<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceCScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                           functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubInplaceCScale_For(typeSrcIsTypeDst)                                                        \
    template void                                                                                                      \
    InvokeSubInplaceCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst,                               \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInplaceCScale(type)                                                              \
    InstantiateInvokeSubInplaceCScale_For(Pixel##type##C1);                                                            \
    InstantiateInvokeSubInplaceCScale_For(Pixel##type##C2);                                                            \
    InstantiateInvokeSubInplaceCScale_For(Pixel##type##C3);                                                            \
    InstantiateInvokeSubInplaceCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInplaceCScale(type)                                                            \
    InstantiateInvokeSubInplaceCScale_For(Pixel##type##C1);                                                            \
    InstantiateInvokeSubInplaceCScale_For(Pixel##type##C2);                                                            \
    InstantiateInvokeSubInplaceCScale_For(Pixel##type##C3);                                                            \
    InstantiateInvokeSubInplaceCScale_For(Pixel##type##C4);                                                            \
    InstantiateInvokeSubInplaceCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
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

        InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubInplaceDevC_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeSubInplaceDevC<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInplaceDevC(type)                                                                \
    InstantiateInvokeSubInplaceDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeSubInplaceDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeSubInplaceDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeSubInplaceDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInplaceDevC(type)                                                              \
    InstantiateInvokeSubInplaceDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeSubInplaceDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeSubInplaceDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeSubInplaceDevC_For(Pixel##type##C4);                                                              \
    InstantiateInvokeSubInplaceDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInplaceDevCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst,
                               scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubInplaceDevC(aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subInplaceDevCScale = InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::Sub<ComputeT>,
                                                                   RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subInplaceDevCScale functor(aConst, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubInplaceDevCScale_For(typeSrcIsTypeDst)                                                     \
    template void                                                                                                      \
    InvokeSubInplaceDevCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst,                               \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInplaceDevCScale(type)                                                           \
    InstantiateInvokeSubInplaceDevCScale_For(Pixel##type##C1);                                                         \
    InstantiateInvokeSubInplaceDevCScale_For(Pixel##type##C2);                                                         \
    InstantiateInvokeSubInplaceDevCScale_For(Pixel##type##C3);                                                         \
    InstantiateInvokeSubInplaceDevCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInplaceDevCScale(type)                                                         \
    InstantiateInvokeSubInplaceDevCScale_For(Pixel##type##C1);                                                         \
    InstantiateInvokeSubInplaceDevCScale_For(Pixel##type##C2);                                                         \
    InstantiateInvokeSubInplaceDevCScale_For(Pixel##type##C3);                                                         \
    InstantiateInvokeSubInplaceDevCScale_For(Pixel##type##C4);                                                         \
    InstantiateInvokeSubInplaceDevCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInvInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                            const Size2D &aSize, const StreamCtx &aStreamCtx)
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

            InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceSrcSIMD>(aSrcDst, aPitchSrcDst, aSize,
                                                                                aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::SubInv<ComputeT>, RoundingMode::None>;

            const mpp::SubInv<ComputeT> op;

            const subInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
        }
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubInvInplaceSrc_For(typeSrcIsTypeDst)                                                        \
    template void                                                                                                      \
    InvokeSubInvInplaceSrc<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(  \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInvInplaceSrc(type)                                                              \
    InstantiateInvokeSubInvInplaceSrc_For(Pixel##type##C1);                                                            \
    InstantiateInvokeSubInvInplaceSrc_For(Pixel##type##C2);                                                            \
    InstantiateInvokeSubInvInplaceSrc_For(Pixel##type##C3);                                                            \
    InstantiateInvokeSubInvInplaceSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInvInplaceSrc(type)                                                            \
    InstantiateInvokeSubInvInplaceSrc_For(Pixel##type##C1);                                                            \
    InstantiateInvokeSubInvInplaceSrc_For(Pixel##type##C2);                                                            \
    InstantiateInvokeSubInvInplaceSrc_For(Pixel##type##C3);                                                            \
    InstantiateInvokeSubInvInplaceSrc_For(Pixel##type##C4);                                                            \
    InstantiateInvokeSubInvInplaceSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInvInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                                 scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubInvInplaceSrc(aSrcDst, aPitchSrcDst, aSrc2, aPitchSrc2, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::SubInv<ComputeT>,
                                                          RoundingMode::NearestTiesToEven>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                             functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubInvInplaceSrcScale_For(typeSrcIsTypeDst)                                                   \
    template void                                                                                                      \
    InvokeSubInvInplaceSrcScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInvInplaceSrcScale(type)                                                         \
    InstantiateInvokeSubInvInplaceSrcScale_For(Pixel##type##C1);                                                       \
    InstantiateInvokeSubInvInplaceSrcScale_For(Pixel##type##C2);                                                       \
    InstantiateInvokeSubInvInplaceSrcScale_For(Pixel##type##C3);                                                       \
    InstantiateInvokeSubInvInplaceSrcScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInvInplaceSrcScale(type)                                                       \
    InstantiateInvokeSubInvInplaceSrcScale_For(Pixel##type##C1);                                                       \
    InstantiateInvokeSubInvInplaceSrcScale_For(Pixel##type##C2);                                                       \
    InstantiateInvokeSubInvInplaceSrcScale_For(Pixel##type##C3);                                                       \
    InstantiateInvokeSubInvInplaceSrcScale_For(Pixel##type##C4);                                                       \
    InstantiateInvokeSubInvInplaceSrcScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInvInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
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

            InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceCSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
        }
        else
        {
            // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
            using subInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, mpp::SubInv<ComputeT>, RoundingMode::None>;

            const mpp::SubInv<ComputeT> op;

            const subInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                          functor);
        }
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubInvInplaceC_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeSubInvInplaceC<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInvInplaceC(type)                                                                \
    InstantiateInvokeSubInvInplaceC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeSubInvInplaceC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeSubInvInplaceC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeSubInvInplaceC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInvInplaceC(type)                                                              \
    InstantiateInvokeSubInvInplaceC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeSubInvInplaceC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeSubInvInplaceC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeSubInvInplaceC_For(Pixel##type##C4);                                                              \
    InstantiateInvokeSubInvInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInvInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst,
                               scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubInvInplaceC(aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subInplaceCScale = InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::SubInv<ComputeT>,
                                                             RoundingMode::NearestTiesToEven>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceCScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                           functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubInvInplaceCScale_For(typeSrcIsTypeDst)                                                     \
    template void                                                                                                      \
    InvokeSubInvInplaceCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst,                               \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInvInplaceCScale(type)                                                           \
    InstantiateInvokeSubInvInplaceCScale_For(Pixel##type##C1);                                                         \
    InstantiateInvokeSubInvInplaceCScale_For(Pixel##type##C2);                                                         \
    InstantiateInvokeSubInvInplaceCScale_For(Pixel##type##C3);                                                         \
    InstantiateInvokeSubInvInplaceCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInvInplaceCScale(type)                                                         \
    InstantiateInvokeSubInvInplaceCScale_For(Pixel##type##C1);                                                         \
    InstantiateInvokeSubInvInplaceCScale_For(Pixel##type##C2);                                                         \
    InstantiateInvokeSubInvInplaceCScale_For(Pixel##type##C3);                                                         \
    InstantiateInvokeSubInvInplaceCScale_For(Pixel##type##C4);                                                         \
    InstantiateInvokeSubInvInplaceCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInvInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                             const StreamCtx &aStreamCtx)
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

        InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
    }
}

#pragma region Instantiate
// using sub_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSubInvInplaceDevC_For(typeSrcIsTypeDst)                                                       \
    template void                                                                                                      \
    InvokeSubInvInplaceDevC<typeSrcIsTypeDst, sub_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInvInplaceDevC(type)                                                             \
    InstantiateInvokeSubInvInplaceDevC_For(Pixel##type##C1);                                                           \
    InstantiateInvokeSubInvInplaceDevC_For(Pixel##type##C2);                                                           \
    InstantiateInvokeSubInvInplaceDevC_For(Pixel##type##C3);                                                           \
    InstantiateInvokeSubInvInplaceDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInvInplaceDevC(type)                                                           \
    InstantiateInvokeSubInvInplaceDevC_For(Pixel##type##C1);                                                           \
    InstantiateInvokeSubInvInplaceDevC_For(Pixel##type##C2);                                                           \
    InstantiateInvokeSubInvInplaceDevC_For(Pixel##type##C3);                                                           \
    InstantiateInvokeSubInvInplaceDevC_For(Pixel##type##C4);                                                           \
    InstantiateInvokeSubInvInplaceDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSubInvInplaceDevCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst,
                                  scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                                  const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeSubInvInplaceDevC(aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using subInplaceDevCScale = InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, mpp::SubInv<ComputeT>,
                                                                   RoundingMode::NearestTiesToEven>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceDevCScale functor(aConst, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, subInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeSubInvInplaceDevCScale_For(typeSrcIsTypeDst)                                                  \
    template void                                                                                                      \
    InvokeSubInvInplaceDevCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst,                               \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSubInvInplaceDevCScale(type)                                                        \
    InstantiateInvokeSubInvInplaceDevCScale_For(Pixel##type##C1);                                                      \
    InstantiateInvokeSubInvInplaceDevCScale_For(Pixel##type##C2);                                                      \
    InstantiateInvokeSubInvInplaceDevCScale_For(Pixel##type##C3);                                                      \
    InstantiateInvokeSubInvInplaceDevCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSubInvInplaceDevCScale(type)                                                      \
    InstantiateInvokeSubInvInplaceDevCScale_For(Pixel##type##C1);                                                      \
    InstantiateInvokeSubInvInplaceDevCScale_For(Pixel##type##C2);                                                      \
    InstantiateInvokeSubInvInplaceDevCScale_For(Pixel##type##C3);                                                      \
    InstantiateInvokeSubInvInplaceDevCScale_For(Pixel##type##C4);                                                      \
    InstantiateInvokeSubInvInplaceDevCScale_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
