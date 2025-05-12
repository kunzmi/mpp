#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{
template <typename SrcT>
void InvokeMirrorSrc(const SrcT *aSrc1, size_t aPitchSrc1, SrcT *aDst, size_t aPitchDst, MirrorAxis aAxis,
                     const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

template <typename SrcT>
void InvokeMirrorInplace(SrcT *aSrcDst, size_t aPitchSrcDst, MirrorAxis aAxis, const Size2D &aSize,
                         const opp::cuda::StreamCtx &aStreamCtx);
} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
