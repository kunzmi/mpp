#if OPP_ENABLE_CUDA_BACKEND

#include "addSquareMasked.h"
#include "addSquareProductWeightedOutputType.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelMaskedKernel.h>
#include <backends/cuda/simd_operators/binary_operators.h>
#include <backends/cuda/simd_operators/simd_types.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/image/functors/inplaceSrcFunctor.h>
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
void InvokeAddSquareInplaceSrcMask(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aSrcDst, size_t aPitchSrcDst,
                                   const SrcT *aSrc2, size_t aPitchSrc2, const Size2D &aSize,
                                   const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using addSqrInplaceSrc =
            InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::AddSqr<ComputeT>, RoundingMode::None>;

        const opp::AddSqr<ComputeT> op;

        const addSqrInplaceSrc functor(aSrc2, aPitchSrc2, op);

        InvokeForEachPixelMaskedKernelDefault<DstT, TupelSize, addSqrInplaceSrc>(
            aMask, aPitchMask, aSrcDst, aPitchSrcDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using add_spw_output_for_t for computeT and DstT
#define Instantiate_For(typeSrc)                                                                                       \
    template void                                                                                                      \
    InvokeAddSquareInplaceSrcMask<typeSrc, add_spw_output_for_t<typeSrc>, add_spw_output_for_t<typeSrc>>(              \
        const Pixel8uC1 *aMask, size_t aPitchMask, add_spw_output_for_t<typeSrc> *aSrcDst, size_t aPitchSrcDst,        \
        const typeSrc *aSrc2, size_t aPitchSrc2, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(type)                                                                                    \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);

#define ForAllChannelsWithAlpha(type)                                                                                  \
    Instantiate_For(Pixel##type##C1);                                                                                  \
    Instantiate_For(Pixel##type##C2);                                                                                  \
    Instantiate_For(Pixel##type##C3);                                                                                  \
    Instantiate_For(Pixel##type##C4);                                                                                  \
    Instantiate_For(Pixel##type##C4A);

#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
