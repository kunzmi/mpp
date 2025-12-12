#include "ln.h"
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
void InvokeLnSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                 const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using simdOP_t = simd::Ln<Tupel<DstT, TupelSize>>;
    if constexpr (simdOP_t::has_simd)
    {
        using lnSrcSIMD = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Ln<ComputeT>,
                                     RoundingMode::NearestTiesToEven, ComputeT, simdOP_t>;

        const mpp::Ln<ComputeT> op;
        const simdOP_t opSIMD;

        const lnSrcSIMD functor(aSrc1, aPitchSrc1, op, opSIMD);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, lnSrcSIMD>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else
    {
        using lnSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Ln<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Ln<ComputeT> op;

        const lnSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, lnSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_floating_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeLnSrc_For(typeSrcIsTypeDst)                                                                   \
    template void                                                                                                      \
    InvokeLnSrc<typeSrcIsTypeDst, default_floating_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeLnSrc(type)                                                                         \
    InstantiateInvokeLnSrc_For(Pixel##type##C1);                                                                       \
    InstantiateInvokeLnSrc_For(Pixel##type##C2);                                                                       \
    InstantiateInvokeLnSrc_For(Pixel##type##C3);                                                                       \
    InstantiateInvokeLnSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeLnSrc(type)                                                                       \
    InstantiateInvokeLnSrc_For(Pixel##type##C1);                                                                       \
    InstantiateInvokeLnSrc_For(Pixel##type##C2);                                                                       \
    InstantiateInvokeLnSrc_For(Pixel##type##C3);                                                                       \
    InstantiateInvokeLnSrc_For(Pixel##type##C4);                                                                       \
    InstantiateInvokeLnSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename DstT, typename ComputeT>
void InvokeLnInplace(DstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_COMPUTE_DST;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using simdOP_t = simd::Ln<Tupel<DstT, TupelSize>>;
    if constexpr (simdOP_t::has_simd)
    {
        using lnInplaceSIMD = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::Ln<ComputeT>,
                                             RoundingMode::NearestTiesToEven, ComputeT, simdOP_t>;

        const mpp::Ln<ComputeT> op;
        const simdOP_t opSIMD;

        const lnInplaceSIMD functor(op, opSIMD);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, lnInplaceSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                        functor);
    }
    else
    {
        using lnInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::Ln<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Ln<ComputeT> op;

        const lnInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, lnInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_floating_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeLnInplace_For(typeSrcIsTypeDst)                                                               \
    template void InvokeLnInplace<typeSrcIsTypeDst, default_floating_compute_type_for_t<typeSrcIsTypeDst>>(            \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeLnInplace(type)                                                                     \
    InstantiateInvokeLnInplace_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeLnInplace_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeLnInplace_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeLnInplace_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeLnInplace(type)                                                                   \
    InstantiateInvokeLnInplace_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeLnInplace_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeLnInplace_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeLnInplace_For(Pixel##type##C4);                                                                   \
    InstantiateInvokeLnInplace_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
