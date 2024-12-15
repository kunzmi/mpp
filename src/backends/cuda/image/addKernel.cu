#include "addKernel.h"
#include "forEachPixelKernel.cuh"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/simd_operators/binary_operators.h>
#include <backends/cuda/simd_operators/simd_types.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <cuda_runtime.h>

using namespace opp::cuda;

namespace opp::image::cuda
{
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeAddSrcSrc(const SrcT *aSrc1, size_t pitchSrc1, const SrcT *aSrc2, size_t pitchSrc2, DstT *aDst,
                     size_t pitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    OPP_CUDA_REGISTER_TEMPALTE;

    const dim3 BlockSize               = KernelConfiguration<sizeof(DstT)>::BlockSize;
    constexpr int WarpAlignmentInBytes = KernelConfiguration<sizeof(DstT)>::WarpAlignmentInBytes;
    constexpr size_t TupelSize         = KernelConfiguration<sizeof(DstT)>::TupelSize;
    constexpr uint SharedMemory        = 0;

    if constexpr (is_simd_type_v<DstT>)
    {
        // Note: we have two SIMD versions for CUDA: SIMD over Vector4 and Vector2 types for (s)byte/(u)short and SIMD
        // over Tupels for Vector1 and Vector2 types for (s)byte/(u)short. The Vector4/2 SIMD is activated using the
        // same ComputeT as DstT in the instantiate macro.
        // The Tupel-SIMD is activated here:

        // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
        using addSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>, RoudingMode::None, DstT,
                                        simd::Add<Tupel<DstT, TupelSize>>>;

        Add<ComputeT> op;
        simd::Add<Tupel<DstT, TupelSize>> opSIMD;

        addSrcSrc functor(aSrc1, pitchSrc1, aSrc2, pitchSrc2, op, opSIMD);

        InvokeForEachPixelKernel<SrcT, DstT, TupelSize, WarpAlignmentInBytes, addSrcSrc>(
            BlockSize, SharedMemory, aStreamCtx.Stream, aDst, pitchDst, aSize, functor);
    }
    else
    {
        // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
        using addSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>, RoudingMode::None>;

        Add<ComputeT> op;

        addSrcSrc functor(aSrc1, pitchSrc1, aSrc2, pitchSrc2, op);

        InvokeForEachPixelKernel<SrcT, DstT, TupelSize, WarpAlignmentInBytes, addSrcSrc>(
            BlockSize, SharedMemory, aStreamCtx.Stream, aDst, pitchDst, aSize, functor);
    }
}

#define DefaultInstantiate_For(typeSrcIsTypeDst)                                                                       \
    template void InvokeAddSrcSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define SIMDInstantiate_For(typeSrcIsTypeDst)                                                                          \
    template void InvokeAddSrcSrc<typeSrcIsTypeDst, typeSrcIsTypeDst, typeSrcIsTypeDst>(                               \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define DefaultForAllChannels(type)                                                                                    \
    DefaultInstantiate_For(Pixel##type##C1);                                                                           \
    DefaultInstantiate_For(Pixel##type##C2);                                                                           \
    DefaultInstantiate_For(Pixel##type##C3);                                                                           \
    DefaultInstantiate_For(Pixel##type##C4);

DefaultInstantiate_For(Pixel8uC1);
DefaultInstantiate_For(Pixel8uC2);
DefaultInstantiate_For(Pixel8uC3);
SIMDInstantiate_For(Pixel8uC4);

DefaultInstantiate_For(Pixel8sC1);
DefaultInstantiate_For(Pixel8sC2);
DefaultInstantiate_For(Pixel8sC3);
SIMDInstantiate_For(Pixel8sC4);

DefaultInstantiate_For(Pixel16uC1);
SIMDInstantiate_For(Pixel16uC2);
DefaultInstantiate_For(Pixel16uC3);
DefaultInstantiate_For(Pixel16uC4);

DefaultInstantiate_For(Pixel16sC1);
SIMDInstantiate_For(Pixel16sC2);
DefaultInstantiate_For(Pixel16sC3);
DefaultInstantiate_For(Pixel16sC4);

DefaultForAllChannels(32u);
DefaultForAllChannels(32s);

// forAllChannels(16f);
DefaultForAllChannels(32f);
DefaultForAllChannels(64f);

DefaultForAllChannels(16sc);
DefaultForAllChannels(32sc);
DefaultForAllChannels(32fc);

// alpha channels:
SIMDInstantiate_For(Pixel8uC4A);
SIMDInstantiate_For(Pixel8sC4A);
DefaultInstantiate_For(Pixel16uC4A);
DefaultInstantiate_For(Pixel16sC4A);

DefaultInstantiate_For(Pixel32uC4A);
DefaultInstantiate_For(Pixel32sC4A);

DefaultInstantiate_For(Pixel32fC4A);
DefaultInstantiate_For(Pixel64fC4A);

#undef DefaultForAllChannels
#undef DefaultInstantiate_For
#undef SIMDInstantiate_For

} // namespace opp::image::cuda
