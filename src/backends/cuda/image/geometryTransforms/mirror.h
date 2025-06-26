#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
template <typename SrcT>
void InvokeMirrorSrc(const SrcT *aSrc1, size_t aPitchSrc1, SrcT *aDst, size_t aPitchDst, MirrorAxis aAxis,
                     const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeMirrorInplace(SrcT *aSrcDst, size_t aPitchSrcDst, MirrorAxis aAxis, const Size2D &aSize,
                         const mpp::cuda::StreamCtx &aStreamCtx);
} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
