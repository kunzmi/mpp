#if MPP_ENABLE_CUDA_BACKEND

#include "rectStdDev.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/functors/rectStdDevFunctor.h>
#include <common/image/pixelTypeEnabler.h>
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
template <typename Src1T, typename Src2T, typename ComputeT, typename DstT>
void InvokeRectStdDev(const Src1T *aSrc1, size_t aPitchSrc1, const Src2T *aSrc2, size_t aPitchSrc2, DstT *aDst,
                      size_t aPitchDst, const FilterArea &aFilterArea, const Size2D &aSize,
                      const mpp::cuda::StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<Src1T> && mppEnableCudaBackend<DstT>)
    {
        using SrcT = Src1T;
        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using rectStdDev = RectStdDevFunctor<TupelSize, Src1T, Src2T, ComputeT, DstT>;

        const rectStdDev functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, aSize, aFilterArea);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, rectStdDev>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define Instantiate_For(typeSrc1, typeSrc2, typeC, typeDst)                                                            \
    template void InvokeRectStdDev<typeSrc1, typeSrc2, typeC, typeDst>(                                                \
        const typeSrc1 *aSrc1, size_t aPitchSrc1, const typeSrc2 *aSrc2, size_t aPitchSrc2, typeDst *aDst,             \
        size_t aPitchDst, const FilterArea &aFilterArea, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(typeSrc1, typeSrc2, typeC, typeDst)                                                      \
    Instantiate_For(Pixel##typeSrc1##C1, Pixel##typeSrc2##C1, Pixel##typeC##C1, Pixel##typeDst##C1);                   \
    Instantiate_For(Pixel##typeSrc1##C2, Pixel##typeSrc2##C2, Pixel##typeC##C2, Pixel##typeDst##C2);                   \
    Instantiate_For(Pixel##typeSrc1##C3, Pixel##typeSrc2##C3, Pixel##typeC##C3, Pixel##typeDst##C3);                   \
    Instantiate_For(Pixel##typeSrc1##C4, Pixel##typeSrc2##C4, Pixel##typeC##C4, Pixel##typeDst##C4);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
