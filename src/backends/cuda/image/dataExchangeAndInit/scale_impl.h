#include "scale.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/defines.h>
#include <common/image/functors/scaleConversionFunctor.h>
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
void InvokeScale(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                 scalefactor_t<ComputeT> aScaleFactor, scalefactor_t<ComputeT> aSrcMin, scalefactor_t<ComputeT> aDstMin,
                 const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
    requires(!use_int_division_for_scale_v<SrcT, DstT>) && RealOrComplexFloatingVector<DstT>
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using scale = ScaleConversionFunctor<TupelSize, SrcT, ComputeT, DstT, RoundingMode::None>;

    const scale functor(aSrc1, aPitchSrc1, aScaleFactor, aSrcMin, aDstMin);

    InvokeForEachPixelKernelDefault<DstT, TupelSize, scale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
}

#pragma region Instantiate

#define InstantiateFloat_For(typeSrc, typeDst)                                                                         \
    template void InvokeScale<typeSrc, compute_type_scale_for_t<typeSrc, typeDst>, typeDst>(                           \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<compute_type_scale_for_t<typeSrc, typeDst>> aScaleFactor,                                        \
        scalefactor_t<compute_type_scale_for_t<typeSrc, typeDst>> aSrcMin,                                             \
        scalefactor_t<compute_type_scale_for_t<typeSrc, typeDst>> aDstMin, const Size2D &aSize,                        \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaFloat(typeSrc, typeDst)                                                                   \
    InstantiateFloat_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                      \
    InstantiateFloat_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                      \
    InstantiateFloat_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                      \
    InstantiateFloat_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsWithAlphaFloat(typeSrc, typeDst)                                                                 \
    InstantiateFloat_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                      \
    InstantiateFloat_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                      \
    InstantiateFloat_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                      \
    InstantiateFloat_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                                      \
    InstantiateFloat_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeScale(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                 scalefactor_t<ComputeT> aScaleFactor, scalefactor_t<ComputeT> aSrcMin, scalefactor_t<ComputeT> aDstMin,
                 RoundingMode aRoundingMode, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
    requires(!use_int_division_for_scale_v<SrcT, DstT>) && RealOrComplexIntVector<DstT>
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    switch (aRoundingMode)
    {
        case mpp::RoundingMode::NearestTiesToEven:
        {
            using scale = ScaleConversionFunctor<TupelSize, SrcT, ComputeT, DstT, RoundingMode::NearestTiesToEven>;
            const scale functor(aSrc1, aPitchSrc1, aScaleFactor, aSrcMin, aDstMin);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, scale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::RoundingMode::NearestTiesAwayFromZero:
        {
            using scale =
                ScaleConversionFunctor<TupelSize, SrcT, ComputeT, DstT, RoundingMode::NearestTiesAwayFromZero>;
            const scale functor(aSrc1, aPitchSrc1, aScaleFactor, aSrcMin, aDstMin);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, scale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::RoundingMode::TowardZero:
        {
            using scale = ScaleConversionFunctor<TupelSize, SrcT, ComputeT, DstT, RoundingMode::TowardZero>;
            const scale functor(aSrc1, aPitchSrc1, aScaleFactor, aSrcMin, aDstMin);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, scale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::RoundingMode::TowardNegativeInfinity:
        {
            using scale = ScaleConversionFunctor<TupelSize, SrcT, ComputeT, DstT, RoundingMode::TowardNegativeInfinity>;
            const scale functor(aSrc1, aPitchSrc1, aScaleFactor, aSrcMin, aDstMin);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, scale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::RoundingMode::TowardPositiveInfinity:
        {
            using scale = ScaleConversionFunctor<TupelSize, SrcT, ComputeT, DstT, RoundingMode::TowardPositiveInfinity>;
            const scale functor(aSrc1, aPitchSrc1, aScaleFactor, aSrcMin, aDstMin);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, scale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }
}

#pragma region Instantiate

#define InstantiateIntRound_For(typeSrc, typeDst)                                                                      \
    template void InvokeScale<typeSrc, compute_type_scale_for_t<typeSrc, typeDst>, typeDst>(                           \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<compute_type_scale_for_t<typeSrc, typeDst>> aScaleFactor,                                        \
        scalefactor_t<compute_type_scale_for_t<typeSrc, typeDst>> aSrcMin,                                             \
        scalefactor_t<compute_type_scale_for_t<typeSrc, typeDst>> aDstMin, RoundingMode aRoundingMode,                 \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaIntRound(typeSrc, typeDst)                                                                \
    InstantiateIntRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                   \
    InstantiateIntRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                   \
    InstantiateIntRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                   \
    InstantiateIntRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsWithAlphaIntRound(typeSrc, typeDst)                                                              \
    InstantiateIntRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                   \
    InstantiateIntRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                   \
    InstantiateIntRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                   \
    InstantiateIntRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                                   \
    InstantiateIntRound_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeScale(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, scalefactor_t<ComputeT> aSrcRange,
                 scalefactor_t<ComputeT> aDstRange, scalefactor_t<ComputeT> aSrcMin, scalefactor_t<ComputeT> aDstMin,
                 RoundingMode aRoundingMode, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
    requires(use_int_division_for_scale_v<SrcT, DstT>)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    switch (aRoundingMode)
    {
        case mpp::RoundingMode::NearestTiesToEven:
        {
            using scale = ScaleConversionFunctor<TupelSize, SrcT, ComputeT, DstT, RoundingMode::NearestTiesToEven>;
            const scale functor(aSrc1, aPitchSrc1, aSrcMin, aDstMin, aSrcRange, aDstRange);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, scale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::RoundingMode::NearestTiesAwayFromZero:
        {
            using scale =
                ScaleConversionFunctor<TupelSize, SrcT, ComputeT, DstT, RoundingMode::NearestTiesAwayFromZero>;
            const scale functor(aSrc1, aPitchSrc1, aSrcMin, aDstMin, aSrcRange, aDstRange);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, scale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::RoundingMode::TowardZero:
        {
            using scale = ScaleConversionFunctor<TupelSize, SrcT, ComputeT, DstT, RoundingMode::TowardZero>;
            const scale functor(aSrc1, aPitchSrc1, aSrcMin, aDstMin, aSrcRange, aDstRange);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, scale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::RoundingMode::TowardNegativeInfinity:
        {
            using scale = ScaleConversionFunctor<TupelSize, SrcT, ComputeT, DstT, RoundingMode::TowardNegativeInfinity>;
            const scale functor(aSrc1, aPitchSrc1, aSrcMin, aDstMin, aSrcRange, aDstRange);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, scale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::RoundingMode::TowardPositiveInfinity:
        {
            using scale = ScaleConversionFunctor<TupelSize, SrcT, ComputeT, DstT, RoundingMode::TowardPositiveInfinity>;
            const scale functor(aSrc1, aPitchSrc1, aSrcMin, aDstMin, aSrcRange, aDstRange);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, scale>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }
}

#pragma region Instantiate

#define InstantiateIntDiv_For(typeSrc, typeDst)                                                                        \
    template void InvokeScale<typeSrc, compute_type_scale_for_t<typeSrc, typeDst>, typeDst>(                           \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst,                                      \
        scalefactor_t<compute_type_scale_for_t<typeSrc, typeDst>> aSrcRange,                                           \
        scalefactor_t<compute_type_scale_for_t<typeSrc, typeDst>> aDstRange,                                           \
        scalefactor_t<compute_type_scale_for_t<typeSrc, typeDst>> aSrcMin,                                             \
        scalefactor_t<compute_type_scale_for_t<typeSrc, typeDst>> aDstMin, RoundingMode aRoundingMode,                 \
        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaIntDiv(typeSrc, typeDst)                                                                  \
    InstantiateIntDiv_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                     \
    InstantiateIntDiv_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                     \
    InstantiateIntDiv_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                     \
    InstantiateIntDiv_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsWithAlphaIntDiv(typeSrc, typeDst)                                                                \
    InstantiateIntDiv_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                                     \
    InstantiateIntDiv_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                                     \
    InstantiateIntDiv_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                                     \
    InstantiateIntDiv_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                                     \
    InstantiateIntDiv_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#pragma endregion

} // namespace mpp::image::cuda
