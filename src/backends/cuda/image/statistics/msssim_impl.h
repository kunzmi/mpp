#include "msssim.h"
#include "ssim.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongXKernel.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
#include <backends/cuda/image/SSIMFilterKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcReductionFunctor.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/statistics/operators.h>
#include <common/statistics/postOperators.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{
template <typename T> struct pixel_block_size_y
{
    constexpr static int value = 1;
};
template <typename T>
    requires(sizeof(remove_vector_t<T>) == 2)
struct pixel_block_size_y<T>
{
    constexpr static int value = 2;
};
template <typename T>
    requires(sizeof(remove_vector_t<T>) == 1)
struct pixel_block_size_y<T>
{
    constexpr static int value = 4;
};

template <typename SrcT, typename DstT>
void InvokeMSSSIMSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aTempBuffer,
                        size_t aPitchTempBuffer, DstT *aTempBufferAvg, DstT *aDst, int aIteration,
                        remove_vector_t<DstT> aDynamicRange, remove_vector_t<DstT> aK1, remove_vector_t<DstT> aK2,
                        const Size2D &aAllowedReadRoiSize1, const Vector2<int> &aOffsetToActualRoi1,
                        const Size2D &aAllowedReadRoiSize2, const Vector2<int> &aOffsetToActualRoi2,
                        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    if (aIteration < 0 || aIteration >= 5)
    {
        throw INVALIDARGUMENT(aIteration, "The iteration must be in range 0..4.");
    }

    static constexpr float weights[] = {0.0448f, 0.2856f, 0.3000f, 0.2363f, 0.1333f}; // sums up to 1
    constexpr size_t TupelSize       = ConfigTupelSize<"Default", sizeof(DstT)>::value;
    using ComputeT                   = DstT;
    using FilterT                    = FixedFilterKernelSSIM;

    constexpr int pixelBlockSizeY = pixel_block_size_y<DstT>::value;
    constexpr int filterSize      = 11;

    using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
    const BCType bc1(aSrc1, aPitchSrc1, aAllowedReadRoiSize1, aOffsetToActualRoi1);
    const BCType bc2(aSrc2, aPitchSrc2, aAllowedReadRoiSize2, aOffsetToActualRoi2);

    if (aIteration < 4)
    {
        const mpp::MSSSIM<DstT> postOp(aDynamicRange, aK2);

        InvokeSSIMFilterKernelDefault<ComputeT, DstT, TupelSize, filterSize, pixelBlockSizeY, BCType, FilterT,
                                      mpp::MSSSIM<DstT>>(bc1, bc2, aTempBuffer, aPitchTempBuffer, postOp, aSize,
                                                         aStreamCtx);
    }
    else
    {
        const mpp::SSIM<DstT> postOp(aDynamicRange, aK1, aK2);

        InvokeSSIMFilterKernelDefault<ComputeT, DstT, TupelSize, filterSize, pixelBlockSizeY, BCType, FilterT,
                                      mpp::SSIM<DstT>>(bc1, bc2, aTempBuffer, aPitchTempBuffer, postOp, aSize,
                                                       aStreamCtx);
    }

    using sumSrc = SrcReductionFunctor<TupelSize, ComputeT, ComputeT, mpp::Sum<ComputeT, ComputeT>>;

    const mpp::Sum<ComputeT, ComputeT> op;

    const sumSrc functor(aTempBuffer, aPitchTempBuffer, op);

    InvokeReductionAlongXKernelDefault<ComputeT, ComputeT, TupelSize, sumSrc, mpp::Sum<ComputeT, ComputeT>,
                                       ReductionInitValue::Zero>(aTempBuffer, aTempBufferAvg, aSize, aStreamCtx,
                                                                 functor);

    const mpp::DivScalar<DstT> postOpScalar(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(aSize.TotalSize()));
    if (aIteration == 0)
    {
        const mpp::DivPostOp<DstT> postOpAvg(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(aSize.TotalSize()) /
                                             weights[to_size_t(aIteration)]);

        InvokeReductionAlongYKernelDefault<ComputeT, DstT, mpp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                           mpp::DivPostOp<DstT>, mpp::DivScalar<DstT>>(
            aTempBufferAvg, aDst + aIteration, nullptr, aSize.y, postOpAvg, postOpScalar, aStreamCtx);
    }
    else
    {
        const mpp::DivAddToPostOp<DstT> postOpAvg(
            static_cast<complex_basetype_t<remove_vector_t<DstT>>>(aSize.TotalSize()) / weights[to_size_t(aIteration)],
            aDst);

        InvokeReductionAlongYKernelDefault<ComputeT, DstT, mpp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                           mpp::DivAddToPostOp<DstT>, mpp::DivScalar<DstT>>(
            aTempBufferAvg, aDst, nullptr, aSize.y, postOpAvg, postOpScalar, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeMSSSIMSrcSrc<typeSrc, ssim_types_for_rt<typeSrc>>(                                             \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2,                              \
        ssim_types_for_rt<typeSrc> *aTempBuffer, size_t aPitchTempBuffer, ssim_types_for_rt<typeSrc> *aTempBufferAvg,  \
        ssim_types_for_rt<typeSrc> *aDst, int aIteration, remove_vector_t<ssim_types_for_rt<typeSrc>> aDynamicRange,   \
        remove_vector_t<ssim_types_for_rt<typeSrc>> aK1, remove_vector_t<ssim_types_for_rt<typeSrc>> aK2,              \
        const Size2D &aAllowedReadRoiSize1, const Vector2<int> &aOffsetToActualRoi1,                                   \
        const Size2D &aAllowedReadRoiSize2, const Vector2<int> &aOffsetToActualRoi2, const Size2D &aSize,              \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(typeIn)                                                                                  \
    Instantiate_For(Pixel##typeIn##C1);                                                                                \
    Instantiate_For(Pixel##typeIn##C2);                                                                                \
    Instantiate_For(Pixel##typeIn##C3);                                                                                \
    Instantiate_For(Pixel##typeIn##C4);

#define ForAllChannelsWithAlpha(typeIn)                                                                                \
    Instantiate_For(Pixel##typeIn##C1);                                                                                \
    Instantiate_For(Pixel##typeIn##C2);                                                                                \
    Instantiate_For(Pixel##typeIn##C3);                                                                                \
    Instantiate_For(Pixel##typeIn##C4);                                                                                \
    Instantiate_For(Pixel##typeIn##C4A);
#pragma endregion

} // namespace mpp::image::cuda
