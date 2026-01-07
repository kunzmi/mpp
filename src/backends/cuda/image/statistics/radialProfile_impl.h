#include "radialProfile.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/radialProfileKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace mpp::cuda;

namespace mpp::image::cuda
{
template <typename SrcT>
void InvokeRadialProfileSrc(const SrcT *aSrc, size_t aPitchSrc, int *aProfileCount,
                            same_vector_size_different_type_t<SrcT, float> *aProfileSum,
                            same_vector_size_different_type_t<SrcT, float> *aProfileSumSqr, int aProfileSize,
                            const AffineTransformation<float> &aTransformation, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx)
{
    InvokeRadialProfileKernelDefault<SrcT>(aSrc, aPitchSrc, aProfileCount, aProfileSum, aProfileSumSqr, aProfileSize,
                                           aTransformation, aSize, aStreamCtx);
}

#pragma region Instantiate

#define Instantiate_For(typeSrc)                                                                                       \
    template void InvokeRadialProfileSrc<typeSrc>(                                                                     \
        const typeSrc *aSrc, size_t aPitchSrc1, int *aProfileCount,                                                    \
        same_vector_size_different_type_t<typeSrc, float> *aProfileSum,                                                \
        same_vector_size_different_type_t<typeSrc, float> *aProfileSumSqr, int aProfileSize,                           \
        const AffineTransformation<float> &aTransformation, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(typeIn)                                                                                \
    Instantiate_For(Pixel##typeIn##C1);                                                                                \
    Instantiate_For(Pixel##typeIn##C2);                                                                                \
    Instantiate_For(Pixel##typeIn##C3);                                                                                \
    Instantiate_For(Pixel##typeIn##C4);                                                                                \
    Instantiate_For(Pixel##typeIn##C4A);
#pragma endregion

} // namespace mpp::image::cuda
