#include "exp.h"
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
void InvokeExpSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                  const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using simdOP_t = simd::Exp<Tupel<DstT, TupelSize>>;
    if constexpr (simdOP_t::has_simd)
    {
        using expSrcSIMD = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Exp<ComputeT>,
                                      RoundingMode::NearestTiesToEven, ComputeT, simdOP_t>;

        const mpp::Exp<ComputeT> op;
        const simdOP_t opSIMD;

        const expSrcSIMD functor(aSrc1, aPitchSrc1, op, opSIMD);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, expSrcSIMD>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else
    {
        using expSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Exp<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Exp<ComputeT> op;

        const expSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, expSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_floating_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeExpSrc_For(typeSrcIsTypeDst)                                                                  \
    template void                                                                                                      \
    InvokeExpSrc<typeSrcIsTypeDst, default_floating_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(           \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, typeSrcIsTypeDst *aDst, size_t aPitchDst,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeExpSrc(type)                                                                        \
    InstantiateInvokeExpSrc_For(Pixel##type##C1);                                                                      \
    InstantiateInvokeExpSrc_For(Pixel##type##C2);                                                                      \
    InstantiateInvokeExpSrc_For(Pixel##type##C3);                                                                      \
    InstantiateInvokeExpSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeExpSrc(type)                                                                      \
    InstantiateInvokeExpSrc_For(Pixel##type##C1);                                                                      \
    InstantiateInvokeExpSrc_For(Pixel##type##C2);                                                                      \
    InstantiateInvokeExpSrc_For(Pixel##type##C3);                                                                      \
    InstantiateInvokeExpSrc_For(Pixel##type##C4);                                                                      \
    InstantiateInvokeExpSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename DstT, typename ComputeT>
void InvokeExpInplace(DstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_COMPUTE_DST;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using simdOP_t = simd::Exp<Tupel<DstT, TupelSize>>;
    if constexpr (simdOP_t::has_simd)
    {
        using expInplaceSIMD = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::Exp<ComputeT>,
                                              RoundingMode::NearestTiesToEven, ComputeT, simdOP_t>;

        const mpp::Exp<ComputeT> op;
        const simdOP_t opSIMD;

        const expInplaceSIMD functor(op, opSIMD);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, expInplaceSIMD>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
    }
    else
    {
        using expInplace =
            InplaceFunctor<TupelSize, ComputeT, DstT, mpp::Exp<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Exp<ComputeT> op;

        const expInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, expInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_floating_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeExpInplace_For(typeSrcIsTypeDst)                                                              \
    template void InvokeExpInplace<typeSrcIsTypeDst, default_floating_compute_type_for_t<typeSrcIsTypeDst>>(           \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeExpInplace(type)                                                                    \
    InstantiateInvokeExpInplace_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeExpInplace_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeExpInplace_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeExpInplace_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeExpInplace(type)                                                                  \
    InstantiateInvokeExpInplace_For(Pixel##type##C1);                                                                  \
    InstantiateInvokeExpInplace_For(Pixel##type##C2);                                                                  \
    InstantiateInvokeExpInplace_For(Pixel##type##C3);                                                                  \
    InstantiateInvokeExpInplace_For(Pixel##type##C4);                                                                  \
    InstantiateInvokeExpInplace_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
