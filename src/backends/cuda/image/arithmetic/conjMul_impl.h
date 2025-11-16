#if MPP_ENABLE_CUDA_BACKEND

#include "conjMul.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/binary_operators.h>
#include <backends/cuda/simd_operators/simd_types.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
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
void InvokeConjMulSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                         size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using conjMulSrcSrc =
            SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::ConjMul<ComputeT>, RoundingMode::None>;

        const mpp::ConjMul<ComputeT> op;

        const conjMulSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, conjMulSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT
#define InstantiateInvokeConjMulSrcSrc_For(typeSrcIsTypeDst)                                                           \
    template void                                                                                                      \
    InvokeConjMulSrcSrc<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(     \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeConjMulSrcSrc(type)                                                                 \
    InstantiateInvokeConjMulSrcSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeConjMulSrcSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeConjMulSrcSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeConjMulSrcSrc_For(Pixel##type##C4);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeConjMulInplaceSrc(DstT *aSrcDst, size_t aPitchSrcDst, const SrcT *aSrc2, size_t aPitchSrc2,
                             const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        // integer multiplication results in integers, so no rounding needed:
        using conjMulInplaceSrc =
            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::ConjMul<ComputeT>, RoundingMode::None>;

        const mpp::ConjMul<ComputeT> op;

        const conjMulInplaceSrc functor(aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, conjMulInplaceSrc>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                            functor);
    }
}

#pragma region Instantiate
// using default_ext_int_compute_type_for_t for computeT including SIMD activation if possible
#define InstantiateInvokeConjMulInplaceSrc_For(typeSrcIsTypeDst)                                                       \
    template void                                                                                                      \
    InvokeConjMulInplaceSrc<typeSrcIsTypeDst, default_ext_int_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>( \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,             \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeConjMulInplaceSrc(type)                                                             \
    InstantiateInvokeConjMulInplaceSrc_For(Pixel##type##C1);                                                           \
    InstantiateInvokeConjMulInplaceSrc_For(Pixel##type##C2);                                                           \
    InstantiateInvokeConjMulInplaceSrc_For(Pixel##type##C3);                                                           \
    InstantiateInvokeConjMulInplaceSrc_For(Pixel##type##C4);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
