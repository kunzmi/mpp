#if OPP_ENABLE_CUDA_BACKEND

#include "abs.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/simd_types.h>
#include <backends/cuda/simd_operators/unary_operators.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/srcFunctor.h>
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
void InvokeAbsSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                  const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Abs<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD = abs_simd_tupel_compute_type_for_t<SrcT>;
            using absSrcSIMD    = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Abs<ComputeT>, RoundingMode::None,
                                             ComputeT_SIMD, simdOP_t>;

            const opp::Abs<ComputeT> op;
            const simdOP_t opSIMD;

            const absSrcSIMD functor(aSrc1, aPitchSrc1, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, absSrcSIMD>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using absSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Abs<ComputeT>, RoundingMode::None>;

            const opp::Abs<ComputeT> op;

            const absSrc functor(aSrc1, aPitchSrc1, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, absSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using abs_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeAbsSrc<typeSrcIsTypeDst, abs_simd_vector_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

template <typename DstT, typename ComputeT>
void InvokeAbsInplace(DstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_COMPUTE_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Abs<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using ComputeT_SIMD  = abs_simd_tupel_compute_type_for_t<DstT>;
            using absInplaceSIMD = InplaceFunctor<TupelSize, ComputeT, DstT, opp::Abs<ComputeT>, RoundingMode::None,
                                                  ComputeT_SIMD, simdOP_t>;

            const opp::Abs<ComputeT> op;
            const simdOP_t opSIMD;

            const absInplaceSIMD functor(op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, absInplaceSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                             functor);
        }
        else
        {
            using absInplace = InplaceFunctor<TupelSize, ComputeT, DstT, opp::Abs<ComputeT>, RoundingMode::None>;

            const opp::Abs<ComputeT> op;

            const absInplace functor(op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, absInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
        }
    }
}

#pragma region Instantiate
// using abs_simd_vector_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeAbsInplace<typeSrcIsTypeDst, abs_simd_vector_compute_type_for_t<typeSrcIsTypeDst>>(            \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
