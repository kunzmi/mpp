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

#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeConvert<typeSrc, typeDst>(const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst,              \
                                                  size_t aPitchDst, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(8s, 8u);
ForAllChannelsWithAlpha(16u, 8u);
ForAllChannelsWithAlpha(16s, 8u);
ForAllChannelsWithAlpha(32s, 8u);

ForAllChannelsWithAlpha(32s, 8s);

ForAllChannelsWithAlpha(8u, 16u);
ForAllChannelsWithAlpha(8s, 16u);
ForAllChannelsWithAlpha(16s, 16u);

ForAllChannelsWithAlpha(8u, 16s);
ForAllChannelsWithAlpha(8s, 16s);

ForAllChannelsWithAlpha(8s, 32u);
ForAllChannelsWithAlpha(16u, 32u);
ForAllChannelsWithAlpha(16s, 32u);
ForAllChannelsWithAlpha(32s, 32u);

ForAllChannelsWithAlpha(8u, 32s);
ForAllChannelsWithAlpha(8s, 32s);
ForAllChannelsWithAlpha(16u, 32s);
ForAllChannelsWithAlpha(16s, 32s);

ForAllChannelsWithAlpha(8u, 16bf);
ForAllChannelsWithAlpha(8s, 16bf);
ForAllChannelsWithAlpha(16u, 16bf);
ForAllChannelsWithAlpha(16s, 16bf);
ForAllChannelsWithAlpha(32u, 16bf);
ForAllChannelsWithAlpha(32s, 16bf);
ForAllChannelsWithAlpha(32f, 16bf);

ForAllChannelsWithAlpha(8u, 16f);
ForAllChannelsWithAlpha(8s, 16f);
ForAllChannelsWithAlpha(16u, 16f);
ForAllChannelsWithAlpha(16s, 16f);
ForAllChannelsWithAlpha(32u, 16f);
ForAllChannelsWithAlpha(32s, 16f);
ForAllChannelsWithAlpha(32f, 16f);

ForAllChannelsWithAlpha(8u, 32f);
ForAllChannelsWithAlpha(8s, 32f);
ForAllChannelsWithAlpha(16u, 32f);
ForAllChannelsWithAlpha(16s, 32f);
ForAllChannelsWithAlpha(32u, 32f);
ForAllChannelsWithAlpha(32s, 32f);
ForAllChannelsWithAlpha(16f, 32f);
ForAllChannelsWithAlpha(16bf, 32f);

ForAllChannelsWithAlpha(8u, 64f);
ForAllChannelsWithAlpha(8s, 64f);
ForAllChannelsWithAlpha(16u, 64f);
ForAllChannelsWithAlpha(16s, 64f);
ForAllChannelsWithAlpha(32u, 64f);
ForAllChannelsWithAlpha(32s, 64f);
ForAllChannelsWithAlpha(16f, 64f);
ForAllChannelsWithAlpha(16bf, 64f);
ForAllChannelsWithAlpha(32f, 64f);

ForAllChannelsNoAlpha(16sc, 32sc);

ForAllChannelsNoAlpha(16sc, 32fc);
ForAllChannelsNoAlpha(16fc, 32fc);
ForAllChannelsNoAlpha(16bfc, 32fc);

ForAllChannelsNoAlpha(16sc, 64fc);
ForAllChannelsNoAlpha(16fc, 64fc);
ForAllChannelsNoAlpha(16bfc, 64fc);
ForAllChannelsNoAlpha(32fc, 64fc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
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
#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeConvertRound<typeSrc, typeDst>(const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst,         \
                                                       size_t aPitchDst, RoundingMode aRoundingMode,                   \
                                                       const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(16f, 8u);
ForAllChannelsWithAlpha(16bf, 8u);
ForAllChannelsWithAlpha(32f, 8u);
ForAllChannelsWithAlpha(64f, 8u);

ForAllChannelsWithAlpha(16f, 8s);
ForAllChannelsWithAlpha(16bf, 8s);
ForAllChannelsWithAlpha(32f, 8s);
ForAllChannelsWithAlpha(64f, 8s);

ForAllChannelsWithAlpha(16f, 16u);
ForAllChannelsWithAlpha(16bf, 16u);
ForAllChannelsWithAlpha(32f, 16u);
ForAllChannelsWithAlpha(64f, 16u);

ForAllChannelsWithAlpha(16f, 16s);
ForAllChannelsWithAlpha(16bf, 16s);
ForAllChannelsWithAlpha(32f, 16s);
ForAllChannelsWithAlpha(64f, 16s);

ForAllChannelsWithAlpha(32f, 16f);
ForAllChannelsWithAlpha(32f, 16bf);

ForAllChannelsNoAlpha(32fc, 16sc);
ForAllChannelsNoAlpha(64fc, 16sc);

ForAllChannelsNoAlpha(32fc, 16fc);
ForAllChannelsNoAlpha(32fc, 16bfc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
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
#define Instantiate_For(typeSrc, typeDst)                                                                              \
    template void InvokeConvertScaleRound<typeSrc, typeDst>(                                                           \
        const typeSrc *aSrc1, size_t aPitchSrc1, typeDst *aDst, size_t aPitchDst, RoundingMode aRoundingMode,          \
        float aScaleFactor, const Size2D &aSize, const StreamCtx &aStreamCtx);

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

ForAllChannelsWithAlpha(32u, 8u);
ForAllChannelsWithAlpha(32f, 8u);
ForAllChannelsWithAlpha(64f, 8u);

ForAllChannelsWithAlpha(8u, 8s);
ForAllChannelsWithAlpha(16u, 8s);
ForAllChannelsWithAlpha(16s, 8s);
ForAllChannelsWithAlpha(32u, 8s);
ForAllChannelsWithAlpha(32f, 8s);
ForAllChannelsWithAlpha(64f, 8s);

ForAllChannelsWithAlpha(32s, 16u);
ForAllChannelsWithAlpha(32u, 16u);
ForAllChannelsWithAlpha(32f, 16u);
ForAllChannelsWithAlpha(64f, 16u);

ForAllChannelsWithAlpha(16u, 16s);
ForAllChannelsWithAlpha(32s, 16s);
ForAllChannelsWithAlpha(32u, 16s);
ForAllChannelsWithAlpha(32f, 16s);
ForAllChannelsWithAlpha(64f, 16s);

ForAllChannelsWithAlpha(32f, 32u);
ForAllChannelsWithAlpha(64f, 32u);

ForAllChannelsWithAlpha(32u, 32s);
ForAllChannelsWithAlpha(32f, 32s);
ForAllChannelsWithAlpha(64f, 32s);

ForAllChannelsNoAlpha(32sc, 16sc);
ForAllChannelsNoAlpha(32fc, 16sc);
ForAllChannelsNoAlpha(64fc, 16sc);

ForAllChannelsNoAlpha(32fc, 32sc);
ForAllChannelsNoAlpha(64fc, 32sc);

#undef Instantiate_For
#undef ForAllChannelsWithAlpha
#undef ForAllChannelsNoAlpha
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
