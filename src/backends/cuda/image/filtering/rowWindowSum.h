#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/image/filtering/windowSumResultType.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/channel.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/opp_defs.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{
template <typename SrcT>
void InvokeRowWindowSum(const SrcT *aSrc1, size_t aPitchSrc1, window_sum_result_type_t<SrcT> *aDst, size_t aPitchDst,
                        complex_basetype_t<remove_vector_t<window_sum_result_type_t<SrcT>>> aScalingValue,
                        int aFilterSize, int aFilterCenter, BorderType aBorderType, const SrcT &aConstant,
                        const Size2D &aAllowedReadRoiSize, const Vector2<int> &aOffsetToActualRoi, const Size2D &aSize,
                        const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
