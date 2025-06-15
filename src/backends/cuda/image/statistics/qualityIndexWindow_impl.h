#if OPP_ENABLE_CUDA_BACKEND

#include "qualityIndexWindow.h"
#include <backends/cuda/image/SSIMFilterKernel.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionAlongXKernel.h>
#include <backends/cuda/image/reductionAlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcReductionFunctor.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/statistics/operators.h>
#include <common/statistics/postOperators.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
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
void InvokeQIWSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aTempBuffer,
                     DstT *aTempBufferAvg, DstT *aDst, const Size2D &aAllowedReadRoiSize1,
                     const Vector2<int> &aOffsetToActualRoi1, const Size2D &aAllowedReadRoiSize2,
                     const Vector2<int> &aOffsetToActualRoi2, const Size2D &aSize,
                     const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr int filterSize   = 11;
        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;
        using ComputeT             = DstT;
        using FilterT              = FixedFilterKernel<opp::FixedFilter::LowPass, filterSize, float>;

        constexpr int pixelBlockSizeY = pixel_block_size_y<DstT>::value;

        using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
        const BCType bc1(aSrc1, aPitchSrc1, aAllowedReadRoiSize1, aOffsetToActualRoi1);
        const BCType bc2(aSrc2, aPitchSrc2, aAllowedReadRoiSize2, aOffsetToActualRoi2);

        const opp::QualityIndexWindow<DstT> postOp;

        InvokeSSIMFilterKernelDefault<ComputeT, DstT, TupelSize, filterSize, pixelBlockSizeY, BCType, FilterT,
                                      opp::QualityIndexWindow<DstT>>(bc1, bc2, aTempBuffer, aSize.x * sizeof(DstT),
                                                                     postOp, aSize, aStreamCtx);

        using sumSrc = SrcReductionFunctor<TupelSize, ComputeT, ComputeT, opp::Sum<ComputeT, ComputeT>>;

        const opp::Sum<ComputeT, ComputeT> op;

        const sumSrc functor(aTempBuffer, aSize.x * sizeof(DstT), op);

        InvokeReductionAlongXKernelDefault<ComputeT, ComputeT, TupelSize, sumSrc, opp::Sum<ComputeT, ComputeT>,
                                           ReductionInitValue::Zero>(aTempBuffer, aTempBufferAvg, aSize, aStreamCtx,
                                                                     functor);

        const opp::DivPostOp<DstT> postOpAvg(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(aSize.TotalSize()));
        const opp::DivScalar<DstT> postOpScalar(
            static_cast<complex_basetype_t<remove_vector_t<DstT>>>(aSize.TotalSize()));

        InvokeReductionAlongYKernelDefault<ComputeT, DstT, opp::Sum<DstT, DstT>, ReductionInitValue::Zero,
                                           opp::DivPostOp<DstT>, opp::DivScalar<DstT>>(
            aTempBufferAvg, aDst, nullptr, aSize.y, postOpAvg, postOpScalar, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeQIWSrcSrc<typeSrc, qiw_types_for_rt<typeSrc>>(                                                 \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2,                              \
        qiw_types_for_rt<typeSrc> *aTempBuffer, qiw_types_for_rt<typeSrc> *aTempBufferAvg,                             \
        qiw_types_for_rt<typeSrc> *aDst, const Size2D &aAllowedReadRoiSize1, const Vector2<int> &aOffsetToActualRoi1,  \
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
