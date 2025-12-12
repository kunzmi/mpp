#include "mirror.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelInHalfRoiSwapKernel.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/functors/inplaceTransformerFunctor.h>
#include <common/image/functors/interpolator.h>
#include <common/image/functors/transformer.h>
#include <common/image/functors/transformerFunctor.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
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
template <typename SrcT>
void InvokeMirrorSrc(const SrcT *aSrc1, size_t aPitchSrc1, SrcT *aDst, size_t aPitchDst, MirrorAxis aAxis,
                     const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

    using BCType = BorderControl<SrcT, BorderType::None>;
    const BCType bc(aSrc1, aPitchSrc1, aSize, {0});
    using InterpolatorT = Interpolator<SrcT, BCType, int, InterpolationMode::Undefined>;
    const InterpolatorT interpol(bc);

    switch (aAxis)
    {
        case mpp::MirrorAxis::Horizontal:
        {
            const TransformerMirror<int, MirrorAxis::Horizontal> mirror(aSize);
            using FunctorT = TransformerFunctor<TupelSize, SrcT, int, false, InterpolatorT,
                                                TransformerMirror<int, MirrorAxis::Horizontal>, RoundingMode::None>;
            const FunctorT functor(interpol, mirror, aSize);

            InvokeForEachPixelKernelDefault<SrcT, TupelSize, FunctorT>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::MirrorAxis::Vertical:
        {
            const TransformerMirror<int, MirrorAxis::Vertical> mirror(aSize);
            using FunctorT = TransformerFunctor<TupelSize, SrcT, int, false, InterpolatorT,
                                                TransformerMirror<int, MirrorAxis::Vertical>, RoundingMode::None>;
            const FunctorT functor(interpol, mirror, aSize);

            InvokeForEachPixelKernelDefault<SrcT, TupelSize, FunctorT>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::MirrorAxis::Both:
        {
            const TransformerMirror<int, MirrorAxis::Both> mirror(aSize);
            using FunctorT = TransformerFunctor<TupelSize, SrcT, int, false, InterpolatorT,
                                                TransformerMirror<int, MirrorAxis::Both>, RoundingMode::None>;
            const FunctorT functor(interpol, mirror, aSize);

            InvokeForEachPixelKernelDefault<SrcT, TupelSize, FunctorT>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aAxis, aAxis << " is not a supported mirror axis for Mirror.");
            break;
    }
}

#pragma region Instantiate

#define InstantiateInvokeMirrorSrc_For(typeSrc)                                                                        \
    template void InvokeMirrorSrc<typeSrc>(const typeSrc *aSrc1, size_t aPitchSrc1, typeSrc *aDst, size_t aPitchDst,   \
                                           MirrorAxis aAxis, const Size2D &aSize,                                      \
                                           const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMirrorSrc(type)                                                                     \
    InstantiateInvokeMirrorSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeMirrorSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeMirrorSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeMirrorSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMirrorSrc(type)                                                                   \
    InstantiateInvokeMirrorSrc_For(Pixel##type##C1);                                                                   \
    InstantiateInvokeMirrorSrc_For(Pixel##type##C2);                                                                   \
    InstantiateInvokeMirrorSrc_For(Pixel##type##C3);                                                                   \
    InstantiateInvokeMirrorSrc_For(Pixel##type##C4);                                                                   \
    InstantiateInvokeMirrorSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT>
void InvokeMirrorInplace(SrcT *aSrcDst, size_t aPitchSrcDst, MirrorAxis aAxis, const Size2D &aSize,
                         const mpp::cuda::StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE;

    using BCType = BorderControl<SrcT, BorderType::None>;
    const BCType bc(aSrcDst, aPitchSrcDst, aSize, {0});

    Size2D workROI = aSize;

    switch (aAxis)
    {
        case mpp::MirrorAxis::Horizontal:
        {
            const TransformerMirror<int, MirrorAxis::Horizontal> mirror(aSize);
            using FunctorT = InplaceTransformerFunctor<SrcT, BCType, TransformerMirror<int, MirrorAxis::Horizontal>>;
            const FunctorT functor(bc, mirror, aSize);

            workROI.y /= 2; // for uneven sizes, the center will remain unchanged

            InvokeForEachPixelInHalfRoiSwapKernelDefault<SrcT, FunctorT, false>(aSrcDst, aPitchSrcDst, workROI,
                                                                                aStreamCtx, functor);
        }
        break;
        case mpp::MirrorAxis::Vertical:
        {
            const TransformerMirror<int, MirrorAxis::Vertical> mirror(aSize);
            using FunctorT = InplaceTransformerFunctor<SrcT, BCType, TransformerMirror<int, MirrorAxis::Vertical>>;
            const FunctorT functor(bc, mirror, aSize);

            workROI.x /= 2; // for uneven sizes, the center will remain unchanged

            InvokeForEachPixelInHalfRoiSwapKernelDefault<SrcT, FunctorT, false>(aSrcDst, aPitchSrcDst, workROI,
                                                                                aStreamCtx, functor);
        }
        break;
        case mpp::MirrorAxis::Both:
        {
            const TransformerMirror<int, MirrorAxis::Both> mirror(aSize);
            using FunctorT = InplaceTransformerFunctor<SrcT, BCType, TransformerMirror<int, MirrorAxis::Both>>;
            const FunctorT functor(bc, mirror, aSize);

            if (workROI.x % 2 == 1)
            {
                workROI.x += 1;
                workROI.x /= 2; // for uneven sizes, the center will change!
                InvokeForEachPixelInHalfRoiSwapKernelDefault<SrcT, FunctorT, true>(aSrcDst, aPitchSrcDst, workROI,
                                                                                   aStreamCtx, functor);
            }
            else
            {
                workROI.x /= 2;
                InvokeForEachPixelInHalfRoiSwapKernelDefault<SrcT, FunctorT, false>(aSrcDst, aPitchSrcDst, workROI,
                                                                                    aStreamCtx, functor);
            }
        }
        break;
        default:
            throw INVALIDARGUMENT(aAxis, aAxis << " is not a supported mirror axis for Mirror.");
            break;
    }
}

#pragma region Instantiate

#define InstantiateInvokeMirrorInplace_For(typeSrc)                                                                    \
    template void InvokeMirrorInplace<typeSrc>(typeSrc * aSrcDst, size_t aPitchSrcDst, MirrorAxis aAxis,               \
                                               const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeMirrorInplace(type)                                                                 \
    InstantiateInvokeMirrorInplace_For(Pixel##type##C1);                                                               \
    InstantiateInvokeMirrorInplace_For(Pixel##type##C2);                                                               \
    InstantiateInvokeMirrorInplace_For(Pixel##type##C3);                                                               \
    InstantiateInvokeMirrorInplace_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeMirrorInplace(type)                                                               \
    InstantiateInvokeMirrorInplace_For(Pixel##type##C1);                                                               \
    InstantiateInvokeMirrorInplace_For(Pixel##type##C2);                                                               \
    InstantiateInvokeMirrorInplace_For(Pixel##type##C3);                                                               \
    InstantiateInvokeMirrorInplace_For(Pixel##type##C4);                                                               \
    InstantiateInvokeMirrorInplace_For(Pixel##type##C4A);

#pragma endregion
} // namespace mpp::image::cuda
