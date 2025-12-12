#include "bilateralGaussFilter.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/precomputeBilateralGaussFilterKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/filterArea.h>
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

void InvokePrecomputeBilateralGaussFilter(Pixel32fC1 *aPreCompGeomDistCoeff, const FilterArea &aFilterArea,
                                          float aPosSquareSigma, const mpp::cuda::StreamCtx &aStreamCtx)
{
    InvokePrecomputeGeometryDistanceCoeffKernelDefault(aPreCompGeomDistCoeff, aFilterArea, aPosSquareSigma, aStreamCtx);
}

} // namespace mpp::image::cuda
