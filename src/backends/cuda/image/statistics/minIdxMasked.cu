#if OPP_ENABLE_CUDA_BACKEND

#include "minIdxMasked.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/reductionMaskedMinIndexAlongXKernel.h>
#include <backends/cuda/image/reductionMinIndexAlongYKernel.h>
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
void InvokeMinIdxMaskedSrc(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc,
                           SrcT *aTempBufferMin, same_vector_size_different_type_t<SrcT, int> *aTempMinIdxX,
                           SrcT *aDstMin, same_vector_size_different_type_t<SrcT, int> *aDstMinIdxX,
                           same_vector_size_different_type_t<SrcT, int> *aDstMinIdxY,
                           remove_vector_t<SrcT> *aDstScalarMin, Vector3<int> *aDstScalarIdxMin, const Size2D &aSize,
                           const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<SrcT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        InvokeReductionMaskedMinIdxAlongXKernelDefault<SrcT, TupelSize>(
            aMask, aPitchMask, aSrc, aPitchSrc, aTempBufferMin, aTempMinIdxX, aSize, aStreamCtx);

        InvokeReductionMinIdxAlongYKernelDefault<SrcT>(aTempBufferMin, aTempMinIdxX, aDstMin, aDstMinIdxX, aDstMinIdxY,
                                                       aDstScalarMin, aDstScalarIdxMin, aSize.y, aStreamCtx); /**/
    }
}

#pragma region Instantiate

#define Instantiate_For(type)                                                                                          \
    template void InvokeMinIdxMaskedSrc<type>(                                                                         \
        const Pixel8uC1 *aMask, size_t aPitchMask, const type *aSrc, size_t aPitchSrc, type *aTempBufferMin,           \
        same_vector_size_different_type_t<type, int> *aTempMinIdxX, type *aDstMin,                                     \
        same_vector_size_different_type_t<type, int> *aDstMinIdxX,                                                     \
        same_vector_size_different_type_t<type, int> *aDstMinIdxY, remove_vector_t<type> *aDstScalarMin,               \
        Vector3<int> *aDstScalarIdxMin, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

ForAllChannelsWithAlpha(8u);
ForAllChannelsWithAlpha(8s);

ForAllChannelsWithAlpha(16u);
ForAllChannelsWithAlpha(16s);

ForAllChannelsWithAlpha(32u);
ForAllChannelsWithAlpha(32s);

ForAllChannelsWithAlpha(16f);
ForAllChannelsWithAlpha(16bf);
ForAllChannelsWithAlpha(32f);
ForAllChannelsWithAlpha(64f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
