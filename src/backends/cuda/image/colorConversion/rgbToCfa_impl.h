#include "cfaToRgb.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelBlockKernel.h>
#include <backends/cuda/streamCtx.h>
#include <common/colorConversion/color_operators.h>
#include <common/defines.h>
#include <common/image/functors/rgbToCfaFunctor.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{

template <typename SrcT>
void InvokeRgbToCfaSrc(const SrcT *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<SrcT>> *aDst, size_t aPitchDst,
                       BayerGridPosition aBayerGrid, const Vector2<int> &aAllowedReadRoiOffset,
                       const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using DstT   = Vector1<remove_vector_t<SrcT>>;
    using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

    if (aBayerGrid == BayerGridPosition::BGGR)
    {
        using cfa = RGBToCFAFunctor<DstT, BCType, BayerGridPosition::BGGR>;
        const cfa functor(bc);

        InvokeForEachPixelBlockKernelDefault(aDst, aPitchDst, aSizeSrc, aStreamCtx, functor);
    }
    else if (aBayerGrid == BayerGridPosition::GBRG)
    {
        using cfa = RGBToCFAFunctor<DstT, BCType, BayerGridPosition::GBRG>;
        const cfa functor(bc);

        InvokeForEachPixelBlockKernelDefault(aDst, aPitchDst, aSizeSrc, aStreamCtx, functor);
    }
    else if (aBayerGrid == BayerGridPosition::GRBG)
    {
        using cfa = RGBToCFAFunctor<DstT, BCType, BayerGridPosition::GRBG>;
        const cfa functor(bc);

        InvokeForEachPixelBlockKernelDefault(aDst, aPitchDst, aSizeSrc, aStreamCtx, functor);
    }
    else if (aBayerGrid == BayerGridPosition::RGGB)
    {
        using cfa = RGBToCFAFunctor<DstT, BCType, BayerGridPosition::RGGB>;
        const cfa functor(bc);

        InvokeForEachPixelBlockKernelDefault(aDst, aPitchDst, aSizeSrc, aStreamCtx, functor);
    }
    else
    {
        INVALIDARGUMENT(aBayerGrid, "Unknown BayerGridPosition: " << aBayerGrid);
    }
}

#pragma region Instantiate
#define InstantiateInvokeRgbToCfaSrc_For(typeSrcIsTypeDst)                                                             \
    template void InvokeRgbToCfaSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector1<remove_vector_t<typeSrcIsTypeDst>> *aDst,            \
        size_t aPitchDst, BayerGridPosition aBayerGrid, const Vector2<int> &aAllowedReadRoiOffset,                     \
        const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeRgbToCfaSrc(type)                                                                 \
    InstantiateInvokeRgbToCfaSrc_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeRgbToCfaSrc_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
