#if OPP_ENABLE_CUDA_BACKEND

#include "minMaxIdxMasked.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionMaskedMinMaxIndexAlongXKernel.h>
#include <backends/cuda/image/reductionMinMaxIndexAlongYKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcReduction2Functor.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/statistics/operators.h>
#include <common/statistics/postOperators.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{
template <typename SrcT>
void InvokeMinMaxIdxMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                              SrcT *aTempBufferMin, SrcT *aTempBufferMax,
                              same_vector_size_different_type_t<SrcT, int> *aTempMinIdxX,
                              same_vector_size_different_type_t<SrcT, int> *aTempMaxIdxX, SrcT *aDstMin, SrcT *aDstMax,
                              IndexMinMax *aDstIdx, remove_vector_t<SrcT> *aDstScalarMin,
                              remove_vector_t<SrcT> *aDstScalarMax, IndexMinMaxChannel *aDstScalarIdx,
                              const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        InvokeReductionMaskedMinMaxIdxAlongXKernelDefault<SrcT, TupelSize>(aMask, aPitchMask, aSrc, aPitchSrc,
                                                                           aTempBufferMin, aTempBufferMax, aTempMinIdxX,
                                                                           aTempMaxIdxX, aSize, aStreamCtx);

        InvokeReductionMinMaxIdxAlongYKernelDefault<SrcT>(aTempBufferMin, aTempBufferMax, aTempMinIdxX, aTempMaxIdxX,
                                                          aDstMin, aDstMax, aDstIdx, aDstScalarMin, aDstScalarMax,
                                                          aDstScalarIdx, aSize.y, aStreamCtx);
    }
}

#pragma region Instantiate

#define Instantiate_For(type)                                                                                          \
    template void InvokeMinMaxIdxMaskedSrc<type>(                                                                      \
        const Pixel8uC1 *aMask, size_t aPitchMask, const type *aSrc, size_t aPitchSrc, type *aTempBufferMin,           \
        type *aTempBufferMax, same_vector_size_different_type_t<type, int> *aTempMinIdxX,                              \
        same_vector_size_different_type_t<type, int> *aTempMaxIdxX, type *aDstMin, type *aDstMax,                      \
        IndexMinMax *aDstIdx, remove_vector_t<type> *aDstScalarMin, remove_vector_t<type> *aDstScalarMax,              \
        IndexMinMaxChannel *aDstScalarIdx, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
