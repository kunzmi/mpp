#include "convert.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/simd_operators/simd_types.h>
#include <backends/cuda/simd_operators/unary_operators.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/image/functors/convertFunctor.h>
#include <common/image/functors/convertScaleFunctor.h>
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
template <typename SrcT, typename DstT>
void InvokeConvert(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                   const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using simdOP_t = simd::Convert<Tupel<DstT, TupelSize>>;
    if constexpr (simdOP_t::has_simd && std::same_as<SrcT, Vector1<float>>)
    {
        // rounding is only used in float to half/bfloat
        using convertSIMD = ConvertFunctor<TupelSize, SrcT, DstT, RoundingMode::NearestTiesToEven, simdOP_t>;

        const simdOP_t opSIMD;

        const convertSIMD functor(aSrc1, aPitchSrc1, opSIMD);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, convertSIMD>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
    else
    {
        // rounding is only used in float to half/bfloat
        using convert = ConvertFunctor<TupelSize, SrcT, DstT, RoundingMode::NearestTiesToEven>;

        const convert functor(aSrc1, aPitchSrc1);

        InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
    }
}

#pragma region Instantiate

#define InstantiateInvokeConvert_For(typeSrc, typeDst)                                                                 \
    template void InvokeConvert<typeSrc, typeDst>(const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst,              \
                                                  size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeConvert(typeSrc, typeDst)                                                           \
    InstantiateInvokeConvert_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateInvokeConvert_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateInvokeConvert_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateInvokeConvert_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsWithAlphaInvokeConvert(typeSrc, typeDst)                                                         \
    InstantiateInvokeConvert_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                              \
    InstantiateInvokeConvert_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                              \
    InstantiateInvokeConvert_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                              \
    InstantiateInvokeConvert_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                              \
    InstantiateInvokeConvert_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#pragma endregion

template <typename SrcT, typename DstT>
void InvokeConvertRound(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, RoundingMode aRoundingMode,
                        const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    static_assert(RealOrComplexFloatingVector<SrcT>,
                  "Rounding conversion is only valid for floating point source types.");

    MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    switch (aRoundingMode)
    {
        case mpp::RoundingMode::NearestTiesToEven:
        {
            using convert = ConvertFunctor<TupelSize, SrcT, DstT, RoundingMode::NearestTiesToEven>;
            const convert functor(aSrc1, aPitchSrc1);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::RoundingMode::NearestTiesAwayFromZero:
        {
            using convert = ConvertFunctor<TupelSize, SrcT, DstT, RoundingMode::NearestTiesAwayFromZero>;
            const convert functor(aSrc1, aPitchSrc1);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::RoundingMode::TowardZero:
        {
            using convert = ConvertFunctor<TupelSize, SrcT, DstT, RoundingMode::TowardZero>;
            const convert functor(aSrc1, aPitchSrc1);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::RoundingMode::TowardNegativeInfinity:
        {
            using convert = ConvertFunctor<TupelSize, SrcT, DstT, RoundingMode::TowardNegativeInfinity>;
            const convert functor(aSrc1, aPitchSrc1);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        case mpp::RoundingMode::TowardPositiveInfinity:
        {
            using convert = ConvertFunctor<TupelSize, SrcT, DstT, RoundingMode::TowardPositiveInfinity>;
            const convert functor(aSrc1, aPitchSrc1);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        break;
        default:
            break;
    }
}

#pragma region Instantiate
#define InstantiateInvokeConvertRound_For(typeSrc, typeDst)                                                            \
    template void InvokeConvertRound<typeSrc, typeDst>(const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst,         \
                                                       size_t aPitchDst, RoundingMode aRoundingMode,                   \
                                                       const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeConvertRound(typeSrc, typeDst)                                                      \
    InstantiateInvokeConvertRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                         \
    InstantiateInvokeConvertRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                         \
    InstantiateInvokeConvertRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                         \
    InstantiateInvokeConvertRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsWithAlphaInvokeConvertRound(typeSrc, typeDst)                                                    \
    InstantiateInvokeConvertRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                         \
    InstantiateInvokeConvertRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                         \
    InstantiateInvokeConvertRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                         \
    InstantiateInvokeConvertRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                         \
    InstantiateInvokeConvertRound_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#pragma endregion

template <typename SrcT, typename DstT>
void InvokeConvertScaleRound(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst,
                             RoundingMode aRoundingMode, double aScaleFactor, const Size2D &aSize,
                             const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

    constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

    using ComputeT = compute_type_convertScaleRound_for_t<SrcT, DstT>;

    if constexpr (RealOrComplexIntVector<ComputeT>)
    {
        if (aScaleFactor >= 1.0)
        {
            // integer to integer conversion with intgere scaling, so no rounding:
            using ScalerT = mpp::Scale<ComputeT, false>;
            const ScalerT scaler(aScaleFactor);
            using convert =
                ConvertScaleFunctor<TupelSize, SrcT, ComputeT, DstT, ScalerT, RoundingMode::NearestTiesToEven>;
            const convert functor(aSrc1, aPitchSrc1, scaler);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using ScalerT = mpp::Scale<ComputeT, true, mpp::RoundingMode::NearestTiesToEven>;
                    const ScalerT scaler(aScaleFactor);
                    using convert =
                        ConvertScaleFunctor<TupelSize, SrcT, ComputeT, DstT, ScalerT, mpp::RoundingMode::None>;
                    const convert functor(aSrc1, aPitchSrc1, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using ScalerT = mpp::Scale<ComputeT, true, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    const ScalerT scaler(aScaleFactor);
                    using convert =
                        ConvertScaleFunctor<TupelSize, SrcT, ComputeT, DstT, ScalerT, mpp::RoundingMode::None>;
                    const convert functor(aSrc1, aPitchSrc1, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using ScalerT = mpp::Scale<ComputeT, true, mpp::RoundingMode::TowardZero>;
                    const ScalerT scaler(aScaleFactor);
                    using convert =
                        ConvertScaleFunctor<TupelSize, SrcT, ComputeT, DstT, ScalerT, mpp::RoundingMode::None>;
                    const convert functor(aSrc1, aPitchSrc1, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using ScalerT = mpp::Scale<ComputeT, true, mpp::RoundingMode::TowardNegativeInfinity>;
                    const ScalerT scaler(aScaleFactor);
                    using convert =
                        ConvertScaleFunctor<TupelSize, SrcT, ComputeT, DstT, ScalerT, mpp::RoundingMode::None>;
                    const convert functor(aSrc1, aPitchSrc1, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using ScalerT = mpp::Scale<ComputeT, true, mpp::RoundingMode::TowardPositiveInfinity>;
                    const ScalerT scaler(aScaleFactor);
                    using convert =
                        ConvertScaleFunctor<TupelSize, SrcT, ComputeT, DstT, ScalerT, mpp::RoundingMode::None>;
                    const convert functor(aSrc1, aPitchSrc1, scaler);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }
    else
    {
        using ScalerT = Scale<ComputeT, false>;
        const ScalerT scaler(aScaleFactor);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using convert =
                    ConvertScaleFunctor<TupelSize, SrcT, ComputeT, DstT, ScalerT, RoundingMode::NearestTiesToEven>;
                const convert functor(aSrc1, aPitchSrc1, scaler);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using convert = ConvertScaleFunctor<TupelSize, SrcT, ComputeT, DstT, ScalerT,
                                                    RoundingMode::NearestTiesAwayFromZero>;
                const convert functor(aSrc1, aPitchSrc1, scaler);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using convert = ConvertScaleFunctor<TupelSize, SrcT, ComputeT, DstT, ScalerT, RoundingMode::TowardZero>;
                const convert functor(aSrc1, aPitchSrc1, scaler);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using convert =
                    ConvertScaleFunctor<TupelSize, SrcT, ComputeT, DstT, ScalerT, RoundingMode::TowardNegativeInfinity>;
                const convert functor(aSrc1, aPitchSrc1, scaler);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using convert =
                    ConvertScaleFunctor<TupelSize, SrcT, ComputeT, DstT, ScalerT, RoundingMode::TowardPositiveInfinity>;
                const convert functor(aSrc1, aPitchSrc1, scaler);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            default:
                break;
        }
    }
}

#pragma region Instantiate
#define InstantiateInvokeConvertScaleRound_For(typeSrc, typeDst)                                                       \
    template void InvokeConvertScaleRound<typeSrc, typeDst>(                                                           \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst, RoundingMode aRoundingMode,          \
        double aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeConvertScaleRound(typeSrc, typeDst)                                                 \
    InstantiateInvokeConvertScaleRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                    \
    InstantiateInvokeConvertScaleRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                    \
    InstantiateInvokeConvertScaleRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                    \
    InstantiateInvokeConvertScaleRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);

#define ForAllChannelsWithAlphaInvokeConvertScaleRound(typeSrc, typeDst)                                               \
    InstantiateInvokeConvertScaleRound_For(Pixel##typeSrc##C1, Pixel##typeDst##C1);                                    \
    InstantiateInvokeConvertScaleRound_For(Pixel##typeSrc##C2, Pixel##typeDst##C2);                                    \
    InstantiateInvokeConvertScaleRound_For(Pixel##typeSrc##C3, Pixel##typeDst##C3);                                    \
    InstantiateInvokeConvertScaleRound_For(Pixel##typeSrc##C4, Pixel##typeDst##C4);                                    \
    InstantiateInvokeConvertScaleRound_For(Pixel##typeSrc##C4A, Pixel##typeDst##C4A);

#pragma endregion

} // namespace mpp::image::cuda
