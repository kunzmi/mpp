#include "test.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixel422C2Kernel.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/colorConversion/color_operators.h>
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
void InvokeTestSrc(const SrcT *aSrc1, size_t aPitchSrc1, Vector2<remove_vector_t<DstT>> *aDst, size_t aPitchDst,
                   const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = 2; //    ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using absSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::image::ColorTwist3x4<ComputeT>, RoundingMode::None>;

    // const mpp::image::RGBtoXYZ<ComputeT> op;
    constexpr Matrix3x4<float> color(mpp::image::color::RGB_FRtoYCbCrBT601_LR, {16, 128, 128});
    const mpp::image::ColorTwist3x4<ComputeT> op(color);

    const absSrc functor(aSrc1, aPitchSrc1, op);

    InvokeForEachPixel422C2KernelDefault<DstT, absSrc>(aDst, aPitchDst, aSize, ChromaSubsamplePos::Center,
                                                       Dst422C2Layout::YCbCr, aStreamCtx, functor);
    // InvokeForEachPixelKernelDefault<DstT, TupelSize, absSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate
#define InstantiateInvokeTestSrc_For(typeSrcIsTypeDst)                                                                 \
    template void                                                                                                      \
    InvokeTestSrc<typeSrcIsTypeDst, same_vector_size_different_type_t<typeSrcIsTypeDst, float>, typeSrcIsTypeDst>(     \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector2<remove_vector_t<typeSrcIsTypeDst>> *aDst,            \
        size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeTestSrc(type) InstantiateInvokeTestSrc_For(Pixel##type##C3);

#pragma endregion

template <typename DstT, typename ComputeT>
void InvokeTestInplace(DstT *aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    /*MPP_CUDA_REGISTER_TEMPALTE_COMPUTE_DST;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using absInplace = InplaceFunctor<TupelSize, ComputeT, DstT, mpp::image::RGBtoXYZ<ComputeT>, RoundingMode::None>;

    const mpp::image::RGBtoXYZ<ComputeT> op;

    const absInplace functor(op);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, absInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);*/
}

#pragma region Instantiate
#define InstantiateInvokeTestInplace_For(typeSrcIsTypeDst)                                                             \
    template void InvokeTestInplace<typeSrcIsTypeDst, same_vector_size_different_type_t<typeSrcIsTypeDst, float>>(     \
        typeSrcIsTypeDst * aSrcDst, size_t aPitchSrcDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeTestInplace(type) InstantiateInvokeTestInplace_For(Pixel##type##C3);

#pragma endregion

} // namespace mpp::image::cuda
