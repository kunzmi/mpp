#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{
// compute and result types for ssim reduction:
template <typename SrcT> struct ssim_types_scalar_for
{
    using resultType = float;
};
template <> struct ssim_types_scalar_for<double>
{
    using resultType = double;
};

// compute and result types for ssim reduction:
template <typename SrcT> struct ssim_types_for
{
    using resultType =
        same_vector_size_different_type_t<SrcT, typename ssim_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using ssim_types_for_rt = typename ssim_types_for<T>::resultType;

template <typename SrcT, typename DstT>
void InvokeSSIMSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aTempBuffer,
                      DstT *aTempBufferAvg, DstT *aDst, remove_vector_t<DstT> aDynamicRange, remove_vector_t<DstT> aK1,
                      remove_vector_t<DstT> aK2, const Size2D &aAllowedReadRoiSize1,
                      const Vector2<int> &aOffsetToActualRoi1, const Size2D &aAllowedReadRoiSize2,
                      const Vector2<int> &aOffsetToActualRoi2, const Size2D &aSize,
                      const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
