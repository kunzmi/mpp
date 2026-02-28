#include "alphaPremul.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/unary_operators.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
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
void InvokeAlphaPremulSrc(const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using alphaPremulSrc =
        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::AlphaPremul<ComputeT, SrcT>, RoundingMode::NearestTiesToEven>;

    const mpp::AlphaPremul<ComputeT, SrcT> op;

    const alphaPremulSrc functor(aSrc, aPitchSrc, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaPremulSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
// using default_floating_compute_type_for_t for computeT
#define InstantiateInvokeAlphaPremulSrc_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeAlphaPremulSrc<typeSrcIsTypeDst, default_floating_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, \
        const mpp::cuda::StreamCtx &aStreamCtx);

#pragma endregion

template <typename SrcDstT, typename ComputeT>
void InvokeAlphaPremulInplace(SrcDstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize,
                              const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT = SrcDstT;
    using DstT = SrcDstT;

    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using alphaPremulInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::AlphaPremul<ComputeT, SrcT>, RoundingMode::NearestTiesToEven>;

    const mpp::AlphaPremul<ComputeT, SrcT> op;

    const alphaPremulInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaPremulInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                         functor);
}

#pragma region Instantiate
// using default_floating_compute_type_for_t for computeT
#define InstantiateInvokeAlphaPremulInplace_For(typeSrcIsTypeDst)                                                      \
    template void InvokeAlphaPremulInplace<typeSrcIsTypeDst, default_floating_compute_type_for_t<typeSrcIsTypeDst>>(   \
        typeSrcIsTypeDst * aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAlphaPremulACSrc(const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst,
                            const remove_vector_t<SrcT> &aAlpha, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using alphaPremulACSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::AlphaPremulAC<ComputeT, SrcT>,
                                        RoundingMode::NearestTiesToEven>;

    const mpp::AlphaPremulAC<ComputeT, SrcT> op(aAlpha);

    const alphaPremulACSrc functor(aSrc, aPitchSrc, op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaPremulACSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
// using default_floating_compute_type_for_t for computeT
#define InstantiateInvokeAlphaPremulACSrc_For(typeSrcIsTypeDst)                                                        \
    template void                                                                                                      \
    InvokeAlphaPremulACSrc<typeSrcIsTypeDst, default_floating_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, typeSrcIsTypeDst *aDst, size_t aPitchDst,                      \
        const remove_vector_t<typeSrcIsTypeDst> &aAlpha, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#pragma endregion

template <typename SrcDstT, typename ComputeT>
void InvokeAlphaPremulACInplace(SrcDstT *aSrcDst, size_t aPitchSrcDst, const remove_vector_t<SrcDstT> &aAlpha,
                                const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT = SrcDstT;
    using DstT = SrcDstT;

    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using alphaPremulACInplace =
        InplaceFunctor<TupelSize, ComputeT, DstT, mpp::AlphaPremulAC<ComputeT, SrcT>, RoundingMode::NearestTiesToEven>;

    const mpp::AlphaPremulAC<ComputeT, SrcT> op(aAlpha);

    const alphaPremulACInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaPremulACInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                           functor);
}

#pragma region Instantiate
// using default_floating_compute_type_for_t for computeT
#define InstantiateInvokeAlphaPremulACInplace_For(typeSrcIsTypeDst)                                                    \
    template void InvokeAlphaPremulACInplace<typeSrcIsTypeDst, default_floating_compute_type_for_t<typeSrcIsTypeDst>>( \
        typeSrcIsTypeDst * aDst, size_t aPitchDst, const remove_vector_t<typeSrcIsTypeDst> &aAlpha,                    \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#pragma endregion

} // namespace mpp::image::cuda
