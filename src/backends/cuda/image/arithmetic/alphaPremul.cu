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

        using alphaPremulSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AlphaPremul<ComputeT>,
                                          RoundingMode::NearestTiesAwayFromZero>;

        remove_vector_t<ComputeT> alphaScaleVal = static_cast<remove_vector_t<ComputeT>>(1);
        if constexpr (RealIntVector<SrcT>)
        {
            alphaScaleVal = static_cast<remove_vector_t<ComputeT>>(numeric_limits<remove_vector_t<SrcT>>::max());
        }

        AlphaPremul<ComputeT> op(alphaScaleVal);

        alphaPremulSrc functor(aSrc, aPitchSrc, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaPremulSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void                                                                                                      \
    InvokeAlphaPremulSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(            \
        const typeSrcIsTypeDst *aSrc, size_t aPitchSrc, typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, \
        const opp::cuda::StreamCtx &aStreamCtx);

// we treat 4th channel as alpha channel, but don't need any pre-load of alpha
Instantiate_For(Pixel8sC4);
Instantiate_For(Pixel8uC4);

Instantiate_For(Pixel16sC4);
Instantiate_For(Pixel16uC4);

Instantiate_For(Pixel32sC4);
Instantiate_For(Pixel32uC4);

Instantiate_For(Pixel16fC4);
Instantiate_For(Pixel16bfC4);
Instantiate_For(Pixel32fC4);
Instantiate_For(Pixel64fC4);

#undef Instantiate_For
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

        using alphaPremulInplace = InplaceFunctor<TupelSize, ComputeT, DstT, opp::AlphaPremul<ComputeT>,
                                                  RoundingMode::NearestTiesAwayFromZero>;

        remove_vector_t<ComputeT> alphaScaleVal = static_cast<remove_vector_t<ComputeT>>(1);
        if constexpr (RealIntVector<SrcT>)
        {
            alphaScaleVal = static_cast<remove_vector_t<ComputeT>>(numeric_limits<remove_vector_t<SrcT>>::max());
        }

        AlphaPremul<ComputeT> op(alphaScaleVal);

        alphaPremulInplace functor(op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, alphaPremulInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                             functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT
#define Instantiate_For(typeSrcIsTypeDst)                                                                              \
    template void InvokeAlphaPremulInplace<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>>(            \
        typeSrcIsTypeDst * aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

// we treat 4th channel as alpha channel, but don't need any pre-load of alpha
Instantiate_For(Pixel8sC4);
Instantiate_For(Pixel8uC4);

Instantiate_For(Pixel16sC4);
Instantiate_For(Pixel16uC4);

Instantiate_For(Pixel32sC4);
Instantiate_For(Pixel32uC4);

Instantiate_For(Pixel16fC4);
Instantiate_For(Pixel16bfC4);
Instantiate_For(Pixel32fC4);
Instantiate_For(Pixel64fC4);

#undef Instantiate_For
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
