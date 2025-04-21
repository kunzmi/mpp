#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/statistics/indexMinMax.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{

template <typename SrcT>
void InvokeMinMaxIdxMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                              SrcT *aTempBufferMin, SrcT *aTempBufferMax,
                              same_vector_size_different_type_t<SrcT, int> *aTempMinIdxX,
                              same_vector_size_different_type_t<SrcT, int> *aTempMaxIdxX, SrcT *aDstMin, SrcT *aDstMax,
                              IndexMinMax *aDstIdx, remove_vector_t<SrcT> *aDstScalarMin,
                              remove_vector_t<SrcT> *aDstScalarMax, IndexMinMaxChannel *aDstScalarIdx,
                              const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
