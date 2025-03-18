#if OPP_ENABLE_CUDA_BACKEND

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
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMakeComplexSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                          const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using makeComplexSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::MakeComplex<DstT>, RoundingMode::None>;

        const opp::MakeComplex<DstT> op;

        const makeComplexSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, makeComplexSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using SrcT for computeT
#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeMakeComplexSrc<typeSrc, typeSrc, typeDst>(const typeSrc *aSrc1, size_t aPitchSrc1,             \
                                                                  typeDst *aDst, size_t aPitchDst,                     \
                                                                  const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(typeSrc, typeDst)                                                                        \
    Instantiate_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                           \
    Instantiate_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                           \
    Instantiate_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                           \
    Instantiate_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

ForAllChannelsNoAlpha(16s, 16sc);
ForAllChannelsNoAlpha(32s, 32sc);
ForAllChannelsNoAlpha(32f, 32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMakeComplexSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                             size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using makeComplexSrcSrc =
            SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::MakeComplex<DstT>, RoundingMode::None>;

        const opp::MakeComplex<DstT> op;

        const makeComplexSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, makeComplexSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                            functor);
    }
}

#pragma region Instantiate
// using SrcT for computeT
#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeMakeComplexSrcSrc<typeSrc, typeSrc, typeDst>(                                                  \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2, typeDst *aDst,               \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(typeSrc, typeDst)                                                                        \
    Instantiate_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                           \
    Instantiate_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                           \
    Instantiate_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                           \
    Instantiate_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

ForAllChannelsNoAlpha(16s, 16sc);
ForAllChannelsNoAlpha(32s, 32sc);
ForAllChannelsNoAlpha(32f, 32fc);

#undef Instantiate_For
#undef ForAllChannelsNoAlpha
#pragma endregion
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
