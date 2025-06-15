#if OPP_ENABLE_CUDA_BACKEND

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
template <typename SrcT, typename DstT>
void InvokeConvert(const SrcT *aSrc1, size_t aPitchSrc1, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                   const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

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
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using convert = ConvertFunctor<TupelSize, SrcT, DstT, RoundingMode::NearestTiesToEven>;
                const convert functor(aSrc1, aPitchSrc1);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using convert = ConvertFunctor<TupelSize, SrcT, DstT, RoundingMode::NearestTiesAwayFromZero>;
                const convert functor(aSrc1, aPitchSrc1);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using convert = ConvertFunctor<TupelSize, SrcT, DstT, RoundingMode::TowardZero>;
                const convert functor(aSrc1, aPitchSrc1);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using convert = ConvertFunctor<TupelSize, SrcT, DstT, RoundingMode::TowardNegativeInfinity>;
                const convert functor(aSrc1, aPitchSrc1);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
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
                             RoundingMode aRoundingMode, float aScaleFactor, const Size2D &aSize,
                             const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE_SRC_DST;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aRoundingMode)
        {
            case opp::RoundingMode::NearestTiesToEven:
            {
                using convert = ConvertScaleFunctor<TupelSize, SrcT, DstT, RoundingMode::NearestTiesToEven>;
                const convert functor(aSrc1, aPitchSrc1, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::NearestTiesAwayFromZero:
            {
                using convert = ConvertScaleFunctor<TupelSize, SrcT, DstT, RoundingMode::NearestTiesAwayFromZero>;
                const convert functor(aSrc1, aPitchSrc1, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardZero:
            {
                using convert = ConvertScaleFunctor<TupelSize, SrcT, DstT, RoundingMode::TowardZero>;
                const convert functor(aSrc1, aPitchSrc1, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardNegativeInfinity:
            {
                using convert = ConvertScaleFunctor<TupelSize, SrcT, DstT, RoundingMode::TowardNegativeInfinity>;
                const convert functor(aSrc1, aPitchSrc1, aScaleFactor);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, convert>(aDst, aPitchDst, aSize, aStreamCtx, functor);
            }
            break;
            case opp::RoundingMode::TowardPositiveInfinity:
            {
                using convert = ConvertScaleFunctor<TupelSize, SrcT, DstT, RoundingMode::TowardPositiveInfinity>;
                const convert functor(aSrc1, aPitchSrc1, aScaleFactor);
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
        float aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
