#if OPP_ENABLE_CUDA_BACKEND

#include "absDiff.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/binary_operators.h>
#include <backends/cuda/simd_operators/simd_types.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceConstantFunctor.h>
#include <common/image/functors/inplaceDevConstantFunctor.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
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
void InvokeAbsDiffSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                         size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::AbsDiff<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD     = absDiff_simd_tupel_compute_type_for_t<SrcT>;
            using absDiffSrcSrcSIMD = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AbsDiff<ComputeT>,
                                                    RoundingMode::None, ComputeT_SIMD, simdOP_t>;

            AbsDiff<ComputeT> op;
            simdOP_t opSIMD;

            absDiffSrcSrcSIMD functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, absDiffSrcSrcSIMD>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                functor);
        }
        else
        {
            using absDiffSrcSrc =
                SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AbsDiff<ComputeT>, RoundingMode::None>;

            AbsDiff<ComputeT> op;

            absDiffSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, absDiffSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                            functor);
        }
    }
}

#pragma region Instantiate
// using absDiff_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeAbsDiffSrcSrc<typeSrcIsTypeDst, absDiff_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
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

ForAllChannelsWithAlpha(16u);

ForAllChannelsWithAlpha(32u);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAbsDiffSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                       const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::AbsDiff<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD   = absDiff_simd_tupel_compute_type_for_t<SrcT>;
            using absDiffSrcCSIMD = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AbsDiff<ComputeT>,
                                                       RoundingMode::None, Tupel<ComputeT_SIMD, TupelSize>, simdOP_t>;

            AbsDiff<ComputeT> op;
            simdOP_t opSIMD;
            Tupel<ComputeT_SIMD, TupelSize> tupelConstant =
                Tupel<ComputeT_SIMD, TupelSize>::GetConstant(static_cast<ComputeT_SIMD>(aConst));

            absDiffSrcCSIMD functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, absDiffSrcCSIMD>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
        }
        else
        {
            using absDiffSrcC =
                SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AbsDiff<ComputeT>, RoundingMode::None>;

            AbsDiff<ComputeT> op;

            absDiffSrcC functor(aSrc, aPitchSrc, static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, absDiffSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using absDiff_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeAbsDiffSrcC<typeSrcIsTypeDst, absDiff_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
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

ForAllChannelsWithAlpha(16u);

ForAllChannelsWithAlpha(32u);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAbsDiffSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                          const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // set to roundingmode NONE, because AbsDiff cannot produce non-integers in computations with ints:
        using absDiffSrcDevC =
            SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AbsDiff<ComputeT>, RoundingMode::None>;

        AbsDiff<ComputeT> op;

        absDiffSrcDevC functor(aSrc, aPitchSrc, aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, absDiffSrcDevC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using absDiff_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeAbsDiffSrcDevC<typeSrcIsTypeDst, absDiff_simd_vector_compute_type_for_t<typeSrcIsTypeDst>,     \
                                       typeSrcIsTypeDst>(                                                              \
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

ForAllChannelsWithAlpha(16u);

ForAllChannelsWithAlpha(32u);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAbsDiffInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                             const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::AbsDiff<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD         = absDiff_simd_tupel_compute_type_for_t<SrcT>;
            using absDiffInplaceSrcSIMD = InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AbsDiff<ComputeT>,
                                                            RoundingMode::None, ComputeT_SIMD, simdOP_t>;

            AbsDiff<ComputeT> op;
            simdOP_t opSIMD;

            absDiffInplaceSrcSIMD functor(aSrc2, aPitchSrc2, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, absDiffInplaceSrcSIMD>(aSrcDst, aPitchSrcDst, aSize,
                                                                                    aStreamCtx, functor);
        }
        else
        {
            using absDiffInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AbsDiff<ComputeT>, RoundingMode::None>;

            AbsDiff<ComputeT> op;

            absDiffInplaceSrc functor(aSrc2, aPitchSrc2, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, absDiffInplaceSrc>(aSrcDst, aPitchSrcDst, aSize,
                                                                                aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using absDiff_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeAbsDiffInplaceSrc<typeSrcIsTypeDst, absDiff_simd_vector_compute_type_for_t<typeSrcIsTypeDst>,  \
                                          typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst,           \
                                                            const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,          \
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

ForAllChannelsWithAlpha(16u);

ForAllChannelsWithAlpha(32u);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAbsDiffInplaceC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT &aConst, const Size2D &aSize,
                           const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::AbsDiff<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = absDiff_simd_tupel_compute_type_for_t<SrcT>;
            using absDiffInplaceCSIMD =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::AbsDiff<ComputeT>, RoundingMode::None,
                                       Tupel<ComputeT_SIMD, TupelSize>, simdOP_t>;

            AbsDiff<ComputeT> op;
            simdOP_t opSIMD;
            Tupel<ComputeT_SIMD, TupelSize> tupelConstant =
                Tupel<ComputeT_SIMD, TupelSize>::GetConstant(static_cast<ComputeT_SIMD>(aConst));

            absDiffInplaceCSIMD functor(static_cast<ComputeT>(aConst), op, tupelConstant, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, absDiffInplaceCSIMD>(aSrcDst, aPitchSrcDst, aSize,
                                                                                  aStreamCtx, functor);
        }
        else
        {
            using absDiffInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT, opp::AbsDiff<ComputeT>, RoundingMode::None>;

            AbsDiff<ComputeT> op;

            absDiffInplaceC functor(static_cast<ComputeT>(aConst), op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, absDiffInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
        }
    }
}

#pragma region Instantiate
// using absDiff_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeAbsDiffInplaceC<typeSrcIsTypeDst, absDiff_simd_vector_compute_type_for_t<typeSrcIsTypeDst>,    \
                                        typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst,             \
                                                          const typeSrcIsTypeDst &aConst, const Size2D &aSize,         \
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

ForAllChannelsWithAlpha(16u);

ForAllChannelsWithAlpha(32u);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAbsDiffInplaceDevC(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aConst, const Size2D &aSize,
                              const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using absDiffInplaceDevC =
            InplaceDevConstantFunctor<TupelSize, ComputeT, DstT, opp::AbsDiff<ComputeT>, RoundingMode::None>;

        AbsDiff<ComputeT> op;

        absDiffInplaceDevC functor(aConst, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, absDiffInplaceDevC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                             functor);
    }
}

#pragma region Instantiate
// using absDiff_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeAbsDiffInplaceDevC<typeSrcIsTypeDst, absDiff_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, \
                                           typeSrcIsTypeDst>(typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst,          \
                                                             const typeSrcIsTypeDst *aConst, const Size2D &aSize,      \
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

ForAllChannelsWithAlpha(16u);

ForAllChannelsWithAlpha(32u);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
