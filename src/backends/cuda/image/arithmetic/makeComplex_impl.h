#if MPP_ENABLE_CUDA_BACKEND

#include "magnitude.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/unary_operators.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/functors/srcFunctor.h>
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
void InvokeMakeComplexSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using makeComplexSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::MakeComplex<DstT>, RoundingMode::None>;

        const mpp::MakeComplex<DstT> op;

        const makeComplexSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, makeComplexSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using SrcT for computeT
#define InstantiateInvokeMakeComplexSrc_For(typeSrc, typeDst)                                                          \
    template void InvokeMakeComplexSrc<typeSrc, typeSrc, typeDst>(const typeSrc *aSrc1, size_t aPitchSrc1,             \
                                                                  typeDst *aDst, size_t aPitchDst,                     \
                                                                  const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMakeComplexSrc(typeSrc, typeDst)                                                    \
    InstantiateInvokeMakeComplexSrc_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                       \
    InstantiateInvokeMakeComplexSrc_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                       \
    InstantiateInvokeMakeComplexSrc_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                       \
    InstantiateInvokeMakeComplexSrc_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMakeComplexSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                             size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<DstT> && mppEnableCudaBackend<DstT>)
    {
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using makeComplexSrcSrc =
            SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::MakeComplex<DstT>, RoundingMode::None>;

        const mpp::MakeComplex<DstT> op;

        const makeComplexSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, makeComplexSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                            functor);
    }
}

#pragma region Instantiate
// using SrcT for computeT
#define InstantiateInvokeMakeComplexSrcSrc_For(typeSrc, typeDst)                                                       \
    template void InvokeMakeComplexSrcSrc<typeSrc, typeSrc, typeDst>(                                                  \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2, typeDst *aDst,               \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMakeComplexSrcSrc(typeSrc, typeDst)                                                 \
    InstantiateInvokeMakeComplexSrcSrc_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                    \
    InstantiateInvokeMakeComplexSrcSrc_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                    \
    InstantiateInvokeMakeComplexSrcSrc_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                    \
    InstantiateInvokeMakeComplexSrcSrc_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#pragma endregion
} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
