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
void InvokeSqrtSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                   const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using simdOP_t = simd::Sqrt<Tupel<DstT, TupelSize>>;
    if constexpr (simdOP_t::has_simd)
    {
        using sqrtSrcSIMD = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sqrt<ComputeT>,
                                       RoundingMode::NearestTiesToEven, ComputeT, simdOP_t>;

        const mpp::Sqrt<ComputeT> op;
        const simdOP_t opSIMD;

        const sqrtSrcSIMD functor(aSrc1, aPitchSrc1, op, opSIMD);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, sqrtSrcSIMD>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else
    {
        using sqrtSrc =
            SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Sqrt<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Sqrt<ComputeT> op;

        const sqrtSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, sqrtSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_floating_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSqrtSrc_For(typeSrcIsTypeDst)                                                                 \
    template void                                                                                                      \
    InvokeSqrtSrc<typeSrcIsTypeDst, default_floating_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSqrtSrc(type)                                                                       \
    InstantiateInvokeSqrtSrc_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeSqrtSrc_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeSqrtSrc_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeSqrtSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSqrtSrc(type)                                                                     \
    InstantiateInvokeSqrtSrc_For(Pixel##type##C1);                                                                     \
    InstantiateInvokeSqrtSrc_For(Pixel##type##C2);                                                                     \
    InstantiateInvokeSqrtSrc_For(Pixel##type##C3);                                                                     \
    InstantiateInvokeSqrtSrc_For(Pixel##type##C4);                                                                     \
    InstantiateInvokeSqrtSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename DstT, typename ComputeT>
void InvokeSqrtInplace(DstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_COMPUTE_DST;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using simdOP_t = simd::Sqrt<Tupel<DstT, TupelSize>>;
    if constexpr (simdOP_t::has_simd)
    {
        using sqrtInplaceSIMD = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::Sqrt<ComputeT>,
                                               RoundingMode::NearestTiesToEven, ComputeT, simdOP_t>;

        const mpp::Sqrt<ComputeT> op;
        const simdOP_t opSIMD;

        const sqrtInplaceSIMD functor(op, opSIMD);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, sqrtInplaceSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                          functor);
    }
    else
    {
        using sqrtInplace =
            InplaceFunctor<TupelSize, ComputeT, DstT, mpp::Sqrt<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Sqrt<ComputeT> op;

        const sqrtInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, sqrtInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                      functor);
    }
}

#pragma region Instantiate
// using default_floating_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeSqrtInplace_For(typeSrcIsTypeDst)                                                             \
    template void InvokeSqrtInplace<typeSrcIsTypeDst, default_floating_compute_type_for_t<typeSrcIsTypeDst>>(          \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeSqrtInplace(type)                                                                   \
    InstantiateInvokeSqrtInplace_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeSqrtInplace_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeSqrtInplace_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeSqrtInplace_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeSqrtInplace(type)                                                                 \
    InstantiateInvokeSqrtInplace_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeSqrtInplace_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeSqrtInplace_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeSqrtInplace_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeSqrtInplace_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
