#if OPP_ENABLE_CUDA_BACKEND

#include "sqrt.h"
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
void InvokeSqrtSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                   const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Sqrt<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using sqrtSrcSIMD = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Sqrt<ComputeT>, RoundingMode::None,
                                           ComputeT, simdOP_t>;

            Sqrt<ComputeT> op;
            simdOP_t opSIMD;

            sqrtSrcSIMD functor(aSrc1, aPitchSrc1, op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, sqrtSrcSIMD>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            using sqrtSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Sqrt<ComputeT>, RoundingMode::None>;

            Sqrt<ComputeT> op;

            sqrtSrc functor(aSrc1, aPitchSrc1, op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, sqrtSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeSqrtSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
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

template <typename DstT, typename ComputeT>
void InvokeSqrtInplace(DstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_COMPUTE_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using simdOP_t = simd::Sqrt<Tupel<DstT, TupelSize>>;
        if constexpr (simdOP_t::has_simd)
        {
            using sqrtInplaceSIMD =
                InplaceFunctor<TupelSize, ComputeT, DstT, opp::Sqrt<ComputeT>, RoundingMode::None, ComputeT, simdOP_t>;

            Sqrt<ComputeT> op;
            simdOP_t opSIMD;

            sqrtInplaceSIMD functor(op, opSIMD);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, sqrtInplaceSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
        }
        else
        {
            using sqrtInplace = InplaceFunctor<TupelSize, ComputeT, DstT, opp::Sqrt<ComputeT>, RoundingMode::None>;

            Sqrt<ComputeT> op;

            sqrtInplace functor(op);

            InvokeForEachPixelKernelDefault<DstT, TupelSize, sqrtInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                          functor);
        }
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT including SIMD activation if possible
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeSqrtInplace<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>>(                   \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
