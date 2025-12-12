#include "minIdx.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionMinIndexAlongXKernel.h>
#include <backends/cuda/image/reductionMinIndexAlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcReduction2Functor.h>
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
template <typename SrcT>
void InvokeMinIdxSrc(const SrcT *aSrc, size_t aPitchSrc, SrcT *aTempBufferMin,
                     same_vector_size_different_type_t<SrcT, int> *aTempMinIdxX, SrcT *aDstMin,
                     same_vector_size_different_type_t<SrcT, int> *aDstMinIdxX,
                     same_vector_size_different_type_t<SrcT, int> *aDstMinIdxY, remove_vector_t<SrcT> *aDstScalarMin,
                     Vector3<int> *aDstScalarIdxMin, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    InvokeReductionMinIdxAlongXKernelDefault<SrcT, TupelSize>(aSrc, aPitchSrc, aTempBufferMin, aTempMinIdxX, aSize,
                                                              aStreamCtx);

    InvokeReductionMinIdxAlongYKernelDefault<SrcT>(aTempBufferMin, aTempMinIdxX, aDstMin, aDstMinIdxX, aDstMinIdxY,
                                                   aDstScalarMin, aDstScalarIdxMin, aSize.y, aStreamCtx); /**/
}

#pragma region Instantiate

#define Instantiate_For(type)                                                                                          \
    template void InvokeMinIdxSrc<type>(const type *aSrc, size_t aPitchSrc, type *aTempBufferMin,                      \
                                        same_vector_size_different_type_t<type, int> *aTempMinIdxX, type *aDstMin,     \
                                        same_vector_size_different_type_t<type, int> *aDstMinIdxX,                     \
                                        same_vector_size_different_type_t<type, int> *aDstMinIdxY,                     \
                                        remove_vector_t<type> *aDstScalarMin, Vector3<int> *aDstScalarIdxMin,          \
                                        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);
#pragma endregion

} // namespace mpp::image::cuda
