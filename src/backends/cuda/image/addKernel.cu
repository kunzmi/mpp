#include "addKernel.h"
#include "forEachPixelKernel.cuh"

#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <cuda_runtime.h>
#include <iostream>

#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/ternary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/pixelTypes.h>
#include <common/vectorTypes.h>

#include <backends/cuda/image/configurations.h>

using namespace opp::image;

namespace opp::cuda::image
{
template <typename SrcT, typename ComputeT, typename DstT, int hardwareMajor, int hardwareMinor>
void InvokeAddSrcSrc(const SrcT *aSrc1, size_t pitchSrc1, const SrcT *aSrc2, size_t pitchSrc2, DstT *aDst,
                     size_t pitchDst, const Size2D &aSize)
{
    const dim3 BlockSize               = KernelConfiguration<sizeof(DstT)>::BlockSize;
    constexpr int WarpAlignmentInBytes = KernelConfiguration<sizeof(DstT)>::WarpAlignmentInBytes;
    constexpr size_t TupelSize         = KernelConfiguration<sizeof(DstT)>::TupelSize;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addSrcSrc =
        SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Add<ComputeT>, void, NullOp<void>, RoudingMode::None>;

    Add<ComputeT> op;

    addSrcSrc functor(aSrc1, pitchSrc1, aSrc2, pitchSrc2, op);

    InvokeForEachPixelKernel<SrcT, DstT, TupelSize, WarpAlignmentInBytes, addSrcSrc>(BlockSize, aDst, pitchDst, aSize,
                                                                                     functor);
}

#define InstantiateAddSrcSrc_For(typeSrcIsTypeDst)                                                                     \
    template void InvokeAddSrcSrc<typeSrcIsTypeDst, default_compute_type_for_t<typeSrcIsTypeDst>, typeSrcIsTypeDst>(   \
        const typeSrcIsTypeDst *aSrc1, size_t aPitchSrc1, const typeSrcIsTypeDst *aSrc2, size_t aPitchSrc2,            \
        typeSrcIsTypeDst *aDst, size_t aPitchDst, const Size2D &aSize);

#define forAllChannels(type)                                                                                           \
    InstantiateAddSrcSrc_For(Pixel##type##C1);                                                                         \
    InstantiateAddSrcSrc_For(Pixel##type##C2);                                                                         \
    InstantiateAddSrcSrc_For(Pixel##type##C3);                                                                         \
    InstantiateAddSrcSrc_For(Pixel##type##C4);

InstantiateAddSrcSrc_For(Pixel8uC1);
InstantiateAddSrcSrc_For(Pixel8uC2);
InstantiateAddSrcSrc_For(Pixel8uC3);
template void InvokeAddSrcSrc<Pixel8uC4, Pixel8uC4, Pixel8uC4>(const Pixel8uC4 *aSrc1, size_t aPitchSrc1,
                                                               const Pixel8uC4 *aSrc2, size_t aPitchSrc2,
                                                               Pixel8uC4 *aDst, size_t aPitchDst, const Size2D &aSize);

// forAllChannels(8u);
forAllChannels(8s);
forAllChannels(16u);
forAllChannels(16s);
forAllChannels(32u);
forAllChannels(32s);

// forAllChannels(16f);
forAllChannels(32f);
forAllChannels(64f);

forAllChannels(16sc);
forAllChannels(32sc);
forAllChannels(32fc);

// alpha channels:
InstantiateAddSrcSrc_For(Pixel8uC4A);
InstantiateAddSrcSrc_For(Pixel8sC4A);
InstantiateAddSrcSrc_For(Pixel16uC4A);
InstantiateAddSrcSrc_For(Pixel16sC4A);
InstantiateAddSrcSrc_For(Pixel32uC4A);
InstantiateAddSrcSrc_For(Pixel32sC4A);

InstantiateAddSrcSrc_For(Pixel32fC4A);
InstantiateAddSrcSrc_For(Pixel64fC4A);

#undef forAllChannels
#undef InstantiateAddSrcSrc_For

} // namespace opp::cuda::image
