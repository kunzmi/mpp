#include "magnitudeSqr.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/unary_operators.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/interpolator.h>
#include <common/image/functors/srcFunctor.h>
#include <common/image/functors/transformer.h>
#include <common/image/functors/transformerFunctor.h>
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
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMagnitudeSqrSrc(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSizeSrc,
                           const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    Size2D sizeDst = aSize;
    sizeDst.x      = sizeDst.x / 2 + 1;

    if (aSizeSrc == sizeDst)
    {
        constexpr RoundingMode roundingMode = RoundingMode::None;
        using CoordT                        = int;

        const TransformerPStoFFTW<CoordT> ps2fftw(aSize);

        using BCType = BorderControl<SrcT, BorderType::None>;
        const BCType bc(aSrc1, aPitchSrc1, aSizeSrc, {0});

        const mpp::MagnitudeSqr<ComputeT> op;
        using InterpolatorT = Interpolator<SrcT, BCType, CoordT, InterpolationMode::Undefined>;
        const InterpolatorT interpol(bc);

        using transformerT = TransformerFunctor<1, DstT, CoordT, false, InterpolatorT, TransformerPStoFFTW<CoordT>,
                                                roundingMode, mpp::MagnitudeSqr<ComputeT>>;
        const transformerT functor(interpol, ps2fftw, aSize, op);

        InvokeForEachPixelKernelDefault<DstT, 1, transformerT>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else
    {
        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        using magnitudeSqrSrc =
            SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::MagnitudeSqr<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::MagnitudeSqr<ComputeT> op;

        const magnitudeSqrSrc functor(aSrc1, aPitchSrc1, op);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, magnitudeSqrSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate
// using default_compute_type_for_t for computeT
#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeMagnitudeSqrSrc<typeSrc, default_compute_type_for_t<typeSrc>, typeDst>(                        \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst, const Size2D &aSizeSrc,              \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlpha(typeSrc, typeDst)                                                                        \
    Instantiate_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                           \
    Instantiate_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                           \
    Instantiate_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                           \
    Instantiate_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#pragma endregion

} // namespace mpp::image::cuda
