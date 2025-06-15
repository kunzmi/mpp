#if OPP_ENABLE_CUDA_BACKEND

#include "add.h"
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
void InvokeAddSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                     size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Add<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = add_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addSrcSrcSIMD = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>, RoundingMode::None,
                                                ComputeT_SIMD, simdOP_t>;

            const opp::Add<ComputeT> op;
            const simdOP_t opSIMD;

            const addSrcSrcSIMD functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, addSrcSrcSIMD>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                            functor);
        }
        else
        {
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>, RoundingMode::None>;

            const opp::Add<ComputeT> op;

            const addSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, addSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using add_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeAddSrcSrc_For(typeSrcIsTypeDst)                                                               \
    template void                                                                                                      \
    InvokeAddSrcSrc<typeSrcIsTypeDst, add_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(         \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddSrcSrc(type)                                                                     \
    InstantiateInvokeAddSrcSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeAddSrcSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeAddSrcSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeAddSrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddSrcSrc(type)                                                                   \
    InstantiateInvokeAddSrcSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeAddSrcSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeAddSrcSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeAddSrcSrc_For(Pixel##type##C4);                                                                   \
    InstantiateInvokeAddSrcSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddSrcSrcScale(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                          size_t aPitchDst, scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeAddSrcSrc(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addSrcSrcScale =
            SrcSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>, RoundingMode::NearestTiesToEven>;

        const opp::Add<ComputeT> op;

        const addSrcSrcScale functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, addSrcSrcScale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeAddSrcSrcScale_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeAddSrcSrcScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst,                                                                      \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddSrcSrcScale(type)                                                                \
    InstantiateInvokeAddSrcSrcScale_For(Pixel##type##C1);                                                              \
    InstantiateInvokeAddSrcSrcScale_For(Pixel##type##C2);                                                              \
    InstantiateInvokeAddSrcSrcScale_For(Pixel##type##C3);                                                              \
    InstantiateInvokeAddSrcSrcScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddSrcSrcScale(type)                                                              \
    InstantiateInvokeAddSrcSrcScale_For(Pixel##type##C1);                                                              \
    InstantiateInvokeAddSrcSrcScale_For(Pixel##type##C2);                                                              \
    InstantiateInvokeAddSrcSrcScale_For(Pixel##type##C3);                                                              \
    InstantiateInvokeAddSrcSrcScale_For(Pixel##type##C4);                                                              \
    InstantiateInvokeAddSrcSrcScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                   const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Add<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = add_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addSrcCSIMD = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>,
                                                   RoundingMode::None, Tupel<ComputeT_SIMD, TupelSize>, simdOP_t>;

            const opp::Add<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT_SIMD, TupelSize> tupelConstant =
                Tupel<ComputeT_SIMD, TupelSize>::GetConstant(static_cast<ComputeT_SIMD>(aConst));

            const addSrcCSIMD functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, addSrcCSIMD>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>, RoundingMode::None>;

            const opp::Add<ComputeT> op;

            const addSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, addSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using add_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeAddSrcC_For(typeSrcIsTypeDst)                                                                 \
    template void                                                                                                      \
    InvokeAddSrcC<typeSrcIsTypeDst, add_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddSrcC(type)                                                                       \
    InstantiateInvokeAddSrcC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeAddSrcC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeAddSrcC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeAddSrcC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddSrcC(type)                                                                     \
    InstantiateInvokeAddSrcC_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeAddSrcC_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeAddSrcC_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeAddSrcC_For(Pixel##type##C4);                                                                     \
    InstantiateInvokeAddSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddSrcCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                        scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeAddSrcC(aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addSrcCScale = SrcConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>,
                                                     RoundingMode::NearestTiesToEven>;

        const opp::Add<ComputeT> op;

        const addSrcCScale functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, addSrcCScale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeAddSrcCScale_For(typeSrcIsTypeDst)                                                            \
    template void                                                                                                      \
    InvokeAddSrcCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(              \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst &aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddSrcCScale(type)                                                                  \
    InstantiateInvokeAddSrcCScale_For(Pixel##type##C1);                                                                \
    InstantiateInvokeAddSrcCScale_For(Pixel##type##C2);                                                                \
    InstantiateInvokeAddSrcCScale_For(Pixel##type##C3);                                                                \
    InstantiateInvokeAddSrcCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddSrcCScale(type)                                                                \
    InstantiateInvokeAddSrcCScale_For(Pixel##type##C1);                                                                \
    InstantiateInvokeAddSrcCScale_For(Pixel##type##C2);                                                                \
    InstantiateInvokeAddSrcCScale_For(Pixel##type##C3);                                                                \
    InstantiateInvokeAddSrcCScale_For(Pixel##type##C4);                                                                \
    InstantiateInvokeAddSrcCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
        using addSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>, RoundingMode::None>;

        const opp::Add<ComputeT> op;

        const addSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, addSrcDevC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using add_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeAddSrcDevC_For(typeSrcIsTypeDst)                                                              \
    template void                                                                                                      \
    InvokeAddSrcDevC<typeSrcIsTypeDst, add_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddSrcDevC(type)                                                                    \
    InstantiateInvokeAddSrcDevC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeAddSrcDevC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeAddSrcDevC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeAddSrcDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddSrcDevC(type)                                                                  \
    InstantiateInvokeAddSrcDevC_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeAddSrcDevC_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeAddSrcDevC_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeAddSrcDevC_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeAddSrcDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddSrcDevCScale(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                           scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeAddSrcDevC(aSrc, aPitchSrc, aConst, aDst, aPitchDst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addSrcDevCScale = SrcDevConstantScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>,
                                                           RoundingMode::NearestTiesToEven>;

        const opp::Add<ComputeT> op;

        const addSrcDevCScale functor(aSrc, aPitchSrc, aConst, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, addSrcDevCScale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeAddSrcDevCScale_For(typeSrcIsTypeDst)                                                         \
    template void                                                                                                      \
    InvokeAddSrcDevCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, const typeSrcIsTypeDst *aConst, typeSrcIsTypeDst *aDst,        \
        size_t aPitchDst, scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddSrcDevCScale(type)                                                               \
    InstantiateInvokeAddSrcDevCScale_For(Pixel##type##C1);                                                             \
    InstantiateInvokeAddSrcDevCScale_For(Pixel##type##C2);                                                             \
    InstantiateInvokeAddSrcDevCScale_For(Pixel##type##C3);                                                             \
    InstantiateInvokeAddSrcDevCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddSrcDevCScale(type)                                                             \
    InstantiateInvokeAddSrcDevCScale_For(Pixel##type##C1);                                                             \
    InstantiateInvokeAddSrcDevCScale_For(Pixel##type##C2);                                                             \
    InstantiateInvokeAddSrcDevCScale_For(Pixel##type##C3);                                                             \
    InstantiateInvokeAddSrcDevCScale_For(Pixel##type##C4);                                                             \
    InstantiateInvokeAddSrcDevCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Add<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = add_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>,
                                                        RoundingMode::None, ComputeT_SIMD, simdOP_t>;

            const opp::Add<ComputeT> op;
            const simdOP_t opSIMD;

            const addInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, addInplaceSrcSIMD>(aSrcDst, aPitchSrcDst, aSize,
                                                                                aStreamCtx, functor);
        }
        else
        {
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>, RoundingMode::None>;

            const opp::Add<ComputeT> op;

            const addInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, addInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
        }
    }
}

#pragma region Instantiate
// using add_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeAddInplaceSrc_For(typeSrcIsTypeDst)                                                           \
    template void                                                                                                      \
    InvokeAddInplaceSrc<typeSrcIsTypeDst, add_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddInplaceSrc(type)                                                                 \
    InstantiateInvokeAddInplaceSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeAddInplaceSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeAddInplaceSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeAddInplaceSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddInplaceSrc(type)                                                               \
    InstantiateInvokeAddInplaceSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeAddInplaceSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeAddInplaceSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeAddInplaceSrc_For(Pixel##type##C4);                                                               \
    InstantiateInvokeAddInplaceSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddInplaceSrcScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                              scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeAddInplaceSrc(aSrcDst, aPitchSrcDst, aSrc2, aPitchSrc2, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addInplaceSrcScale = InplaceSrcScaleFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>,
                                                          RoundingMode::NearestTiesToEven>;

        const opp::Add<ComputeT> op;

        const addInplaceSrcScale functor(aSrc2, aPitchSrc2, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, addInplaceSrcScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                             functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeAddInplaceSrcScale_For(typeSrcIsTypeDst)                                                      \
    template void                                                                                                      \
    InvokeAddInplaceSrcScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(        \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddInplaceSrcScale(type)                                                            \
    InstantiateInvokeAddInplaceSrcScale_For(Pixel##type##C1);                                                          \
    InstantiateInvokeAddInplaceSrcScale_For(Pixel##type##C2);                                                          \
    InstantiateInvokeAddInplaceSrcScale_For(Pixel##type##C3);                                                          \
    InstantiateInvokeAddInplaceSrcScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddInplaceSrcScale(type)                                                          \
    InstantiateInvokeAddInplaceSrcScale_For(Pixel##type##C1);                                                          \
    InstantiateInvokeAddInplaceSrcScale_For(Pixel##type##C2);                                                          \
    InstantiateInvokeAddInplaceSrcScale_For(Pixel##type##C3);                                                          \
    InstantiateInvokeAddInplaceSrcScale_For(Pixel##type##C4);                                                          \
    InstantiateInvokeAddInplaceSrcScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                       const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Add<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = add_simd_tupel_compute_type_for_t<SrcT>;
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addInplaceCSIMD =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Add<ComputeT>, RoundingMode::None,
                                       Tupel<ComputeT_SIMD, TupelSize>, simdOP_t>;

            const opp::Add<ComputeT> op;
            const simdOP_t opSIMD;
            const Tupel<ComputeT_SIMD, TupelSize> tupelConstant =
                Tupel<ComputeT_SIMD, TupelSize>::GetConstant(static_cast<ComputeT_SIMD>(aConst));

            const addInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, addInplaceCSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
        }
        else
        {
            // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
            using addInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::Add<ComputeT>, RoundingMode::None>;

            const opp::Add<ComputeT> op;

            const addInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, addInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                          functor);
        }
    }
}

#pragma region Instantiate
// using add_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeAddInplaceC_For(typeSrcIsTypeDst)                                                             \
    template void                                                                                                      \
    InvokeAddInplaceC<typeSrcIsTypeDst, add_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddInplaceC(type)                                                                   \
    InstantiateInvokeAddInplaceC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeAddInplaceC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeAddInplaceC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeAddInplaceC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddInplaceC(type)                                                                 \
    InstantiateInvokeAddInplaceC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeAddInplaceC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeAddInplaceC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeAddInplaceC_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeAddInplaceC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddInplaceCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst,
                            scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeAddInplaceC(aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addInplaceCScale =
            InplaceConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Add<ComputeT>, RoundingMode::NearestTiesToEven>;

        const opp::Add<ComputeT> op;

        const addInplaceCScale functor(static_cast<ComputeT>(aConst), op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, addInplaceCScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                           functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeAddInplaceCScale_For(typeSrcIsTypeDst)                                                        \
    template void                                                                                                      \
    InvokeAddInplaceCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst &aConst,                               \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddInplaceCScale(type)                                                              \
    InstantiateInvokeAddInplaceCScale_For(Pixel##type##C1);                                                            \
    InstantiateInvokeAddInplaceCScale_For(Pixel##type##C2);                                                            \
    InstantiateInvokeAddInplaceCScale_For(Pixel##type##C3);                                                            \
    InstantiateInvokeAddInplaceCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddInplaceCScale(type)                                                            \
    InstantiateInvokeAddInplaceCScale_For(Pixel##type##C1);                                                            \
    InstantiateInvokeAddInplaceCScale_For(Pixel##type##C2);                                                            \
    InstantiateInvokeAddInplaceCScale_For(Pixel##type##C3);                                                            \
    InstantiateInvokeAddInplaceCScale_For(Pixel##type##C4);                                                            \
    InstantiateInvokeAddInplaceCScale_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
        using addInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, opp::Add<ComputeT>, RoundingMode::None>;

        const opp::Add<ComputeT> op;

        const addInplaceDevC functor(aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, addInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
    }
}

#pragma region Instantiate
// using add_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeAddInplaceDevC_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeAddInplaceDevC<typeSrcIsTypeDst, add_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(    \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst, const Size2D &aSize,          \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddInplaceDevC(type)                                                                \
    InstantiateInvokeAddInplaceDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeAddInplaceDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeAddInplaceDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeAddInplaceDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddInplaceDevC(type)                                                              \
    InstantiateInvokeAddInplaceDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeAddInplaceDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeAddInplaceDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeAddInplaceDevC_For(Pixel##type##C4);                                                              \
    InstantiateInvokeAddInplaceDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddInplaceDevCScale(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst,
                               scalefactor_t<ComputeT> aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        // if no scale, use SIMD versions if possible:
        if (aScaleFactor == 1.0f)
        {
            InvokeAddInplaceDevC(aSrcDst, aPitchSrcDst, aConst, aSize, aStreamCtx);
            return;
        }

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addInplaceDevCScale = InplaceDevConstantScaleFunctor<TupelSize, ComputeT, DstT, opp::Add<ComputeT>,
                                                                   RoundingMode::NearestTiesToEven>;

        const opp::Add<ComputeT> op;

        const addInplaceDevCScale functor(aConst, op, aScaleFactor);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, addInplaceDevCScale>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT without SIMD
#define InstantiateInvokeAddInplaceDevCScale_For(typeSrcIsTypeDst)                                                     \
    template void                                                                                                      \
    InvokeAddInplaceDevCScale<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(       \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aConst,                               \
        scalefactor_t<default_compute_type_for_t<typeSrcIsTypeDst>> aScaleFactor, const Size2D &aSize,                 \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeAddInplaceDevCScale(type)                                                           \
    InstantiateInvokeAddInplaceDevCScale_For(Pixel##type##C1);                                                         \
    InstantiateInvokeAddInplaceDevCScale_For(Pixel##type##C2);                                                         \
    InstantiateInvokeAddInplaceDevCScale_For(Pixel##type##C3);                                                         \
    InstantiateInvokeAddInplaceDevCScale_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeAddInplaceDevCScale(type)                                                         \
    InstantiateInvokeAddInplaceDevCScale_For(Pixel##type##C1);                                                         \
    InstantiateInvokeAddInplaceDevCScale_For(Pixel##type##C2);                                                         \
    InstantiateInvokeAddInplaceDevCScale_For(Pixel##type##C3);                                                         \
    InstantiateInvokeAddInplaceDevCScale_For(Pixel##type##C4);                                                         \
    InstantiateInvokeAddInplaceDevCScale_For(Pixel##type##C4A);

#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
