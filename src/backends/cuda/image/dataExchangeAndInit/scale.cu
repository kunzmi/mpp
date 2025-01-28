#if OPP_ENABLE_CUDA_BACKEND

#include "convert.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/scaleConversionFunctor.h>
#include <common/image/pixelTypeEnabler.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeScale(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                 scalefactor_t<ComputeT> aScaleFactor, scalefactor_t<ComputeT> aSrcMin, scalefactor_t<ComputeT> aDstMin,
                 const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using scale = ScaleConversionFunctor<TupelSize, SrcT, ComputeT, DstT, RoundingMode::NearestTiesAwayFromZero>;

        const scale functor(aSrc1, aPitchSrc1, aScaleFactor, aSrcMin, aDstMin);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, scale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeScale<typeSrc, default_compute_type_for_t<typeSrc>, typeDst>(                                  \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<default_compute_type_for_t<typeSrc>> aScaleFactor,                                               \
        scalefactor_t<default_compute_type_for_t<typeSrc>> aSrcMin,                                                    \
        scalefactor_t<default_compute_type_for_t<typeSrc>> aDstMin, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(typeSrc, typeDst)                                                                        \
    Instantiate_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                           \
    Instantiate_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                           \
    Instantiate_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                           \
    Instantiate_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsWithAlpha(typeSrc, typeDst)                                                                      \
    Instantiate_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                           \
    Instantiate_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                           \
    Instantiate_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                           \
    Instantiate_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                                           \
    Instantiate_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

ForAllChannelsWithAlpha(8u, 8s);
ForAllChannelsWithAlpha(8u, 16u);
ForAllChannelsWithAlpha(8u, 16s);
ForAllChannelsWithAlpha(8u, 32u);
ForAllChannelsWithAlpha(8u, 32s);
ForAllChannelsWithAlpha(8u, 32f);
ForAllChannelsWithAlpha(8u, 64f);
ForAllChannelsWithAlpha(8u, 16f);
ForAllChannelsWithAlpha(8u, 16bf);

ForAllChannelsWithAlpha(8s, 8u);
ForAllChannelsWithAlpha(8s, 16u);
ForAllChannelsWithAlpha(8s, 16s);
ForAllChannelsWithAlpha(8s, 32u);
ForAllChannelsWithAlpha(8s, 32s);
ForAllChannelsWithAlpha(8s, 32f);
ForAllChannelsWithAlpha(8s, 64f);
ForAllChannelsWithAlpha(8s, 16f);
ForAllChannelsWithAlpha(8s, 16bf);

ForAllChannelsWithAlpha(16u, 8s);
ForAllChannelsWithAlpha(16u, 8u);
ForAllChannelsWithAlpha(16u, 16s);
ForAllChannelsWithAlpha(16u, 32u);
ForAllChannelsWithAlpha(16u, 32s);
ForAllChannelsWithAlpha(16u, 32f);
ForAllChannelsWithAlpha(16u, 64f);
ForAllChannelsWithAlpha(16u, 16f);
ForAllChannelsWithAlpha(16u, 16bf);

ForAllChannelsWithAlpha(16s, 8s);
ForAllChannelsWithAlpha(16s, 8u);
ForAllChannelsWithAlpha(16s, 16u);
ForAllChannelsWithAlpha(16s, 32u);
ForAllChannelsWithAlpha(16s, 32s);
ForAllChannelsWithAlpha(16s, 32f);
ForAllChannelsWithAlpha(16s, 64f);
ForAllChannelsWithAlpha(16s, 16f);
ForAllChannelsWithAlpha(16s, 16bf);

ForAllChannelsWithAlpha(32s, 8s);
ForAllChannelsWithAlpha(32s, 8u);
ForAllChannelsWithAlpha(32s, 16s);
ForAllChannelsWithAlpha(32s, 16u);
ForAllChannelsWithAlpha(32s, 32u);
ForAllChannelsWithAlpha(32s, 32f);
ForAllChannelsWithAlpha(32s, 64f);
ForAllChannelsWithAlpha(32s, 16f);
ForAllChannelsWithAlpha(32s, 16bf);

ForAllChannelsWithAlpha(32u, 8s);
ForAllChannelsWithAlpha(32u, 8u);
ForAllChannelsWithAlpha(32u, 16s);
ForAllChannelsWithAlpha(32u, 16u);
ForAllChannelsWithAlpha(32u, 32s);
ForAllChannelsWithAlpha(32u, 32f);
ForAllChannelsWithAlpha(32u, 64f);
ForAllChannelsWithAlpha(32u, 16f);
ForAllChannelsWithAlpha(32u, 16bf);

ForAllChannelsWithAlpha(32f, 8s);
ForAllChannelsWithAlpha(32f, 8u);
ForAllChannelsWithAlpha(32f, 16s);
ForAllChannelsWithAlpha(32f, 16u);
ForAllChannelsWithAlpha(32f, 32u);
ForAllChannelsWithAlpha(32f, 32s);
ForAllChannelsWithAlpha(32f, 64f);
ForAllChannelsWithAlpha(32f, 16f);
ForAllChannelsWithAlpha(32f, 16bf);

ForAllChannelsWithAlpha(64f, 8s);
ForAllChannelsWithAlpha(64f, 8u);
ForAllChannelsWithAlpha(64f, 16s);
ForAllChannelsWithAlpha(64f, 16u);
ForAllChannelsWithAlpha(64f, 32u);
ForAllChannelsWithAlpha(64f, 32s);
ForAllChannelsWithAlpha(64f, 32f);
ForAllChannelsWithAlpha(64f, 16f);
ForAllChannelsWithAlpha(64f, 16bf);

ForAllChannelsWithAlpha(16f, 8s);
ForAllChannelsWithAlpha(16f, 8u);
ForAllChannelsWithAlpha(16f, 16s);
ForAllChannelsWithAlpha(16f, 16u);
ForAllChannelsWithAlpha(16f, 32u);
ForAllChannelsWithAlpha(16f, 32s);
ForAllChannelsWithAlpha(16f, 32f);

ForAllChannelsWithAlpha(16bf, 8s);
ForAllChannelsWithAlpha(16bf, 8u);
ForAllChannelsWithAlpha(16bf, 16s);
ForAllChannelsWithAlpha(16bf, 16u);
ForAllChannelsWithAlpha(16bf, 32u);
ForAllChannelsWithAlpha(16bf, 32s);
ForAllChannelsWithAlpha(16bf, 32f);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
