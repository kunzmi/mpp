#include "maxIdx.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionMaxIndexAlongXKernel.h>
#include <backends/cuda/image/reductionMaxIndexAlongYKernel.h>
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
void InvokeMaxIdxSrc(const SrcT *aSrc, size_t aPitchSrc, SrcT *aTempBufferMax,
                     same_vector_size_different_type_t<SrcT, int> *aTempMaxIdxX, SrcT *aDstMax,
                     same_vector_size_different_type_t<SrcT, int> *aDstMaxIdxX,
                     same_vector_size_different_type_t<SrcT, int> *aDstMaxIdxY, remove_vector_t<SrcT> *aDstScalarMax,
                     Vector3<int> *aDstScalarIdxMax, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    InvokeReductionMaxIdxAlongXKernelDefault<SrcT, TupelSize>(aSrc, aPitchSrc, aTempBufferMax, aTempMaxIdxX, aSize,
                                                              aStreamCtx);

    InvokeReductionMaxIdxAlongYKernelDefault<SrcT>(aTempBufferMax, aTempMaxIdxX, aDstMax, aDstMaxIdxX, aDstMaxIdxY,
                                                   aDstScalarMax, aDstScalarIdxMax, aSize.y, aStreamCtx);
}

#pragma region Instantiate

#define Instantiate_For(type)                                                                                          \
    template void InvokeMaxIdxSrc<type>(const type *aSrc, size_t aPitchSrc, type *aTempBufferMax,                      \
                                        same_vector_size_different_type_t<type, int> *aTempMaxIdxX, type *aDstMax,     \
                                        same_vector_size_different_type_t<type, int> *aDstMaxIdxX,                     \
                                        same_vector_size_different_type_t<type, int> *aDstMaxIdxY,                     \
                                        remove_vector_t<type> *aDstScalarMax, Vector3<int> *aDstScalarIdxMax,          \
                                        const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);
#pragma endregion

} // namespace mpp::image::cuda
