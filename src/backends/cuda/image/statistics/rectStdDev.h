#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/filterArea.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{
template <typename Src1T, typename Src2T, typename ComputeT, typename DstT>
void InvokeRectStdDev(const Src1T *aSrc1, size_t aPitchSrc1, const Src2T *aSrc2, size_t aPitchSrc2, DstT *aDst,
                      size_t aPitchDst, const FilterArea &aFilterArea, const Size2D &aSize,
                      const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
