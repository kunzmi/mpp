#include "cfaToRgb.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelBlockKernel.h>
#include <backends/cuda/streamCtx.h>
#include <common/colorConversion/color_operators.h>
#include <common/defines.h>
#include <common/image/functors/cfaToRgbFunctor.h>
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
void InvokeCfaToRgbSrc(const SrcT *aSrc1, size_t aPitchSrc1, Vector3<remove_vector_t<SrcT>> *aDst, size_t aPitchDst,
                       BayerGridPosition aBayerGrid, const Vector2<int> &aAllowedReadRoiOffset,
                       const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,
                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    using DstT   = Vector3<remove_vector_t<SrcT>>;
    using OpT    = NOP<DstT>;
    using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
    const OpT op;
    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

    if (aBayerGrid == BayerGridPosition::BGGR)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::BGGR>;
        const cfa functor(bc, op);

        InvokeForEachPixelBlockKernelDefault(aDst, aPitchDst, aSizeSrc, aStreamCtx, functor);
    }
    else if (aBayerGrid == BayerGridPosition::GBRG)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::GBRG>;
        const cfa functor(bc, op);

        InvokeForEachPixelBlockKernelDefault(aDst, aPitchDst, aSizeSrc, aStreamCtx, functor);
    }
    else if (aBayerGrid == BayerGridPosition::GRBG)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::GRBG>;
        const cfa functor(bc, op);

        InvokeForEachPixelBlockKernelDefault(aDst, aPitchDst, aSizeSrc, aStreamCtx, functor);
    }
    else if (aBayerGrid == BayerGridPosition::RGGB)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::RGGB>;
        const cfa functor(bc, op);

        InvokeForEachPixelBlockKernelDefault(aDst, aPitchDst, aSizeSrc, aStreamCtx, functor);
    }
    else
    {
        INVALIDARGUMENT(aBayerGrid, "Unknown BayerGridPosition: " << aBayerGrid);
    }
}

#pragma region Instantiate
#define InstantiateInvokeCfaToRgb3CSrc_For(typeSrcIsTypeDst)                                                           \
    template void InvokeCfaToRgbSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector3<remove_vector_t<typeSrcIsTypeDst>> *aDst,            \
        size_t aPitchDst, BayerGridPosition aBayerGrid, const Vector2<int> &aAllowedReadRoiOffset,                     \
        const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeCfaToRgb3CSrc(type) InstantiateInvokeCfaToRgb3CSrc_For(Pixel##type##C1);

#pragma endregion

template <typename SrcT>
void InvokeCfaToRgbSrc(const SrcT *aSrc1, size_t aPitchSrc1, Vector4<remove_vector_t<SrcT>> *aDst, size_t aPitchDst,
                       remove_vector_t<SrcT> aAlpha, BayerGridPosition aBayerGrid,
                       const Vector2<int> &aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize,
                       const Size2D &aSizeSrc, const mpp::cuda::StreamCtx &aStreamCtx)
{
    using DstT   = Vector4<remove_vector_t<SrcT>>;
    using OpT    = SetAlphaConst<DstT, NOP<DstT>>;
    using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
    const OpT op(aAlpha, {});
    const BCType bc(aSrc1, aPitchSrc1, aAllowedReadRoiSize, aAllowedReadRoiOffset);

    if (aBayerGrid == BayerGridPosition::BGGR)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::BGGR>;
        const cfa functor(bc, op);

        InvokeForEachPixelBlockKernelDefault(aDst, aPitchDst, aSizeSrc, aStreamCtx, functor);
    }
    else if (aBayerGrid == BayerGridPosition::GBRG)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::GBRG>;
        const cfa functor(bc, op);

        InvokeForEachPixelBlockKernelDefault(aDst, aPitchDst, aSizeSrc, aStreamCtx, functor);
    }
    else if (aBayerGrid == BayerGridPosition::GRBG)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::GRBG>;
        const cfa functor(bc, op);

        InvokeForEachPixelBlockKernelDefault(aDst, aPitchDst, aSizeSrc, aStreamCtx, functor);
    }
    else if (aBayerGrid == BayerGridPosition::RGGB)
    {
        using cfa = CFAToRGBFunctor<DstT, BCType, OpT, BayerGridPosition::RGGB>;
        const cfa functor(bc, op);

        InvokeForEachPixelBlockKernelDefault(aDst, aPitchDst, aSizeSrc, aStreamCtx, functor);
    }
    else
    {
        INVALIDARGUMENT(aBayerGrid, "Unknown BayerGridPosition: " << aBayerGrid);
    }
}

#pragma region Instantiate
#define InstantiateInvokeCfaToRgb4CSrc_For(typeSrcIsTypeDst)                                                           \
    template void InvokeCfaToRgbSrc<typeSrcIsTypeDst>(                                                                 \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, Vector4<remove_vector_t<typeSrcIsTypeDst>> *aDst,            \
        size_t aPitchDst, remove_vector_t<typeSrcIsTypeDst> aAlpha, BayerGridPosition aBayerGrid,                      \
        const Vector2<int> &aAllowedReadRoiOffset, const Size2D &aAllowedReadRoiSize, const Size2D &aSizeSrc,          \
        const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlphaInvokeCfaToRgb4CSrc(type) InstantiateInvokeCfaToRgb4CSrc_For(Pixel##type##C1);

#pragma endregion
} // namespace mpp::image::cuda
