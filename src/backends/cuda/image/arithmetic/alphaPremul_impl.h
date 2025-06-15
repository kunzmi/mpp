#if OPP_ENABLE_CUDA_BACKEND

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
void InvokeAlphaPremulSrc(const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using alphaPremulSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AlphaPremul<ComputeT, SrcT>,
                                          RoundingMode::NearestTiesToEven>;

        const opp::AlphaPremul<ComputeT, SrcT> op;

        const alphaPremulSrc functor(aSrc, aPitchSrc, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaPremulSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT
#define InstantiateInvokeAlphaPremulSrc_For(typeSrcIsTypeDst)                                                          \
    template void                                                                                                      \
    InvokeAlphaPremulSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, \
        const opp::cuda::StreamCtx &aStreamCtx);

#pragma endregion

template <typename SrcDstT, typename ComputeT>
void InvokeAlphaPremulInplace(SrcDstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize,
                              const opp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT = SrcDstT;
    using DstT = SrcDstT;
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using alphaPremulInplace = InplaceFunctor<TupelSize, ComputeT, DstT, opp::AlphaPremul<ComputeT, SrcT>,
                                                  RoundingMode::NearestTiesToEven>;

        const opp::AlphaPremul<ComputeT, SrcT> op;

        const alphaPremulInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaPremulInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                             functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT
#define InstantiateInvokeAlphaPremulInplace_For(typeSrcIsTypeDst)                                                      \
    template void InvokeAlphaPremulInplace<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>>(            \
        typeSrcIsTypeDst * aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAlphaPremulACSrc(const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst,
                            remove_vector_t<SrcT> aAlpha, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using alphaPremulACSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AlphaPremulAC<ComputeT, SrcT>,
                                            RoundingMode::NearestTiesToEven>;

        const opp::AlphaPremulAC<ComputeT, SrcT> op(aAlpha);

        const alphaPremulACSrc functor(aSrc, aPitchSrc, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaPremulACSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT
#define InstantiateInvokeAlphaPremulACSrc_For(typeSrcIsTypeDst)                                                        \
    template void                                                                                                      \
    InvokeAlphaPremulACSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(          \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, typeSrcIsTypeDst *aDst, size_t aPitchDst,                      \
        remove_vector_t<typeSrcIsTypeDst> aAlpha, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

#pragma endregion

template <typename SrcDstT, typename ComputeT>
void InvokeAlphaPremulACInplace(SrcDstT *aSrcDst, size_t aPitchSrcDst, remove_vector_t<SrcDstT> aAlpha,
                                const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    using SrcT = SrcDstT;
    using DstT = SrcDstT;
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using alphaPremulACInplace = InplaceFunctor<TupelSize, ComputeT, DstT, opp::AlphaPremulAC<ComputeT, SrcT>,
                                                    RoundingMode::NearestTiesToEven>;

        const opp::AlphaPremulAC<ComputeT, SrcT> op(aAlpha);

        const alphaPremulACInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaPremulACInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                               functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT
#define InstantiateInvokeAlphaPremulACInplace_For(typeSrcIsTypeDst)                                                    \
    template void InvokeAlphaPremulACInplace<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>>(          \
        typeSrcIsTypeDst * aDst, size_t aPitchDst, remove_vector_t<typeSrcIsTypeDst> aAlpha, const Size2D &aSize,      \
        const StreamCtx &aStreamCtx);

#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
