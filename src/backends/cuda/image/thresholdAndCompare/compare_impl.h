#include "compare.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
#include <common/image/functors/srcFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
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
void InvokeCompareSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                         size_t aPitchDst, CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = vector_size_v<SrcT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (vector_active_size_v<SrcT> > 1 && vector_active_size_v<DstT> == 1 && CompareOp_IsPerChannel(aCompare))
    {
        throw INVALIDARGUMENT(
            aCompare,
            "CompareOp flag 'PerChannel' is not supported for multi channel images and single channel output.");
    }

    auto runOverAnyChannel = [&]<typename T>(T /*isAnyChannel*/) {
        constexpr bool anyChannel = T::value;
        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::Less:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Lt<ComputeT, anyChannel>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const mpp::Lt<ComputeT, anyChannel> op;
                    const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                    functor);
                }
            }
            break;
            case mpp::CompareOp::LessEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Le<ComputeT, anyChannel>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const mpp::Le<ComputeT, anyChannel> op;
                    const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                    functor);
                }
            }
            break;
            case mpp::CompareOp::Eq:
            {
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Eq<ComputeT, anyChannel>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const mpp::Eq<ComputeT, anyChannel> op;
                const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case mpp::CompareOp::Greater:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Gt<ComputeT, anyChannel>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const mpp::Gt<ComputeT, anyChannel> op;
                    const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                    functor);
                }
            }
            break;
            case mpp::CompareOp::GreaterEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Ge<ComputeT, anyChannel>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const mpp::Ge<ComputeT, anyChannel> op;
                    const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                    functor);
                }
            }
            break;
            case mpp::CompareOp::NEq:
            {
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::NEq<ComputeT, anyChannel>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const mpp::NEq<ComputeT, anyChannel> op;
                const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                    << aCompare << ". This function only supports binary comparisons.");
        }
    };

    if (CompareOp_IsPerChannel(aCompare) && vector_active_size_v<DstT> > 1)
    {
        // do not instantiate for single channel:
        if constexpr (vector_active_size_v<DstT> > 1)
        {
            switch (CompareOp_NoFlags(aCompare))
            {
                case mpp::CompareOp::Less:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare, "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
                    }
                    else
                    {
                        using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareLt<ComputeT>,
                                                            RoundingMode::None, voidType, voidType, true>;
                        const mpp::CompareLt<ComputeT> op;
                        const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize,
                                                                                        aStreamCtx, functor);
                    }
                }
                break;
                case mpp::CompareOp::LessEq:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare, "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
                    }
                    else
                    {
                        using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareLe<ComputeT>,
                                                            RoundingMode::None, voidType, voidType, true>;
                        const mpp::CompareLe<ComputeT> op;
                        const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize,
                                                                                        aStreamCtx, functor);
                    }
                }
                break;
                case mpp::CompareOp::Eq:
                {
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareEq<ComputeT>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const mpp::CompareEq<ComputeT> op;
                    const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                    functor);
                }
                break;
                case mpp::CompareOp::Greater:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare, "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
                    }
                    else
                    {
                        using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareGt<ComputeT>,
                                                            RoundingMode::None, voidType, voidType, true>;
                        const mpp::CompareGt<ComputeT> op;
                        const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize,
                                                                                        aStreamCtx, functor);
                    }
                }
                break;
                case mpp::CompareOp::GreaterEq:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare, "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
                    }
                    else
                    {
                        using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareGe<ComputeT>,
                                                            RoundingMode::None, voidType, voidType, true>;
                        const mpp::CompareGe<ComputeT> op;
                        const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize,
                                                                                        aStreamCtx, functor);
                    }
                }
                break;
                case mpp::CompareOp::NEq:
                {
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareNEq<ComputeT>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const mpp::CompareNEq<ComputeT> op;
                    const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                    functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                        << aCompare
                                                        << ". This function only supports binary comparisons.");
            }
        }
    }
    else
    {
        // do not instantiate for multi channel:
        if constexpr (vector_active_size_v<DstT> == 1)
        {
            if (CompareOp_IsAnyChannel(aCompare))
            {
                runOverAnyChannel(std::true_type{});
            }
            else
            {
                runOverAnyChannel(std::false_type{});
            }
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeCompareSrcSrc_For(typeSrc, typeDst)                                                           \
    template void InvokeCompareSrcSrc<typeSrc, typeSrc, typeDst>(                                                      \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2, typeDst *aDst,               \
        size_t aPitchDst, CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeCompareSrcSrc(type)                                                                 \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C1, Pixel8uC1);                                                    \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C2, Pixel8uC1);                                                    \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C3, Pixel8uC1);                                                    \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C4, Pixel8uC1);

#define ForAllChannelsWithAlphaInvokeCompareSrcSrc(type)                                                               \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C1, Pixel8uC1);                                                    \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C2, Pixel8uC1);                                                    \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C3, Pixel8uC1);                                                    \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C4, Pixel8uC1);                                                    \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C4A, Pixel8uC1);

#define ForAllChannelsNoAlphaInvokeCompareSrcSrcAnyChannel(type)                                                       \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C2, Pixel8uC2);                                                    \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C3, Pixel8uC3);                                                    \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C4, Pixel8uC4);

#define ForAllChannelsWithAlphaInvokeCompareSrcSrcAnyChannel(type)                                                     \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C2, Pixel8uC2);                                                    \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C3, Pixel8uC3);                                                    \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C4, Pixel8uC4);                                                    \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C4A, Pixel8uC4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeCompareSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                       CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = vector_size_v<SrcT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (vector_active_size_v<SrcT> > 1 && vector_active_size_v<DstT> == 1 && CompareOp_IsPerChannel(aCompare))
    {
        throw INVALIDARGUMENT(
            aCompare,
            "CompareOp flag 'PerChannel' is not supported for multi channel images and single channel output.");
    }

    auto runOverAnyChannel = [&]<typename T>(T /*isAnyChannel*/) {
        constexpr bool anyChannel = T::value;
        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::Less:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcC =
                        SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Lt<ComputeT, anyChannel>,
                                           RoundingMode::None, voidType, voidType, true>;
                    const mpp::Lt<ComputeT, anyChannel> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case mpp::CompareOp::LessEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcC =
                        SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Le<ComputeT, anyChannel>,
                                           RoundingMode::None, voidType, voidType, true>;
                    const mpp::Le<ComputeT, anyChannel> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case mpp::CompareOp::Eq:
            {
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Eq<ComputeT, anyChannel>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const mpp::Eq<ComputeT, anyChannel> op;
                const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
            }
            break;
            case mpp::CompareOp::Greater:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcC =
                        SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Gt<ComputeT, anyChannel>,
                                           RoundingMode::None, voidType, voidType, true>;
                    const mpp::Gt<ComputeT, anyChannel> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case mpp::CompareOp::GreaterEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcC =
                        SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Ge<ComputeT, anyChannel>,
                                           RoundingMode::None, voidType, voidType, true>;
                    const mpp::Ge<ComputeT, anyChannel> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case mpp::CompareOp::NEq:
            {
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::NEq<ComputeT, anyChannel>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const mpp::NEq<ComputeT, anyChannel> op;
                const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                    << aCompare << ". This function only supports binary comparisons.");
        }
    };

    if (CompareOp_IsPerChannel(aCompare) && vector_active_size_v<DstT> > 1)
    {
        // do not instantiate for single channel:
        if constexpr (vector_active_size_v<DstT> > 1)
        {
            switch (CompareOp_NoFlags(aCompare))
            {
                case mpp::CompareOp::Less:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare, "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
                    }
                    else
                    {
                        using compareSrcC =
                            SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareLt<ComputeT>,
                                               RoundingMode::None, voidType, voidType, true>;
                        const mpp::CompareLt<ComputeT> op;
                        const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                    }
                }
                break;
                case mpp::CompareOp::LessEq:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare, "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
                    }
                    else
                    {
                        using compareSrcC =
                            SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareLe<ComputeT>,
                                               RoundingMode::None, voidType, voidType, true>;
                        const mpp::CompareLe<ComputeT> op;
                        const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                    }
                }
                break;
                case mpp::CompareOp::Eq:
                {
                    using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareEq<ComputeT>,
                                                           RoundingMode::None, voidType, voidType, true>;
                    const mpp::CompareEq<ComputeT> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
                break;
                case mpp::CompareOp::Greater:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare, "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
                    }
                    else
                    {
                        using compareSrcC =
                            SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareGt<ComputeT>,
                                               RoundingMode::None, voidType, voidType, true>;
                        const mpp::CompareGt<ComputeT> op;
                        const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                    }
                }
                break;
                case mpp::CompareOp::GreaterEq:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare, "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
                    }
                    else
                    {
                        using compareSrcC =
                            SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareGe<ComputeT>,
                                               RoundingMode::None, voidType, voidType, true>;
                        const mpp::CompareGe<ComputeT> op;
                        const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                    }
                }
                break;
                case mpp::CompareOp::NEq:
                {
                    using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareNEq<ComputeT>,
                                                           RoundingMode::None, voidType, voidType, true>;
                    const mpp::CompareNEq<ComputeT> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                        << aCompare
                                                        << ". This function only supports binary comparisons.");
            }
        }
    }
    else
    {
        // do not instantiate for multi channel:
        if constexpr (vector_active_size_v<DstT> == 1)
        {
            if (CompareOp_IsAnyChannel(aCompare))
            {
                runOverAnyChannel(std::true_type{});
            }
            else
            {
                runOverAnyChannel(std::false_type{});
            }
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeCompareSrcC_For(typeSrc, typeDst)                                                             \
    template void InvokeCompareSrcC<typeSrc, typeSrc, typeDst>(                                                        \
        const typeSrc *aSrc, size_t aPitchSrc, const typeSrc &aConst, typeDst *aDst, size_t aPitchDst,                 \
        CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeCompareSrcC(type)                                                                   \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C1, Pixel8uC1);                                                      \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C2, Pixel8uC1);                                                      \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C3, Pixel8uC1);                                                      \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C4, Pixel8uC1);

#define ForAllChannelsWithAlphaInvokeCompareSrcC(type)                                                                 \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C1, Pixel8uC1);                                                      \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C2, Pixel8uC1);                                                      \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C3, Pixel8uC1);                                                      \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C4, Pixel8uC1);                                                      \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C4A, Pixel8uC1);

#define ForAllChannelsNoAlphaInvokeCompareSrcCAnyChannel(type)                                                         \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C2, Pixel8uC2);                                                      \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C3, Pixel8uC3);                                                      \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C4, Pixel8uC4);

#define ForAllChannelsWithAlphaInvokeCompareSrcCAnyChannel(type)                                                       \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C2, Pixel8uC2);                                                      \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C3, Pixel8uC3);                                                      \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C4, Pixel8uC4);                                                      \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C4A, Pixel8uC4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeCompareSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                          CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = vector_size_v<SrcT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (vector_active_size_v<SrcT> > 1 && vector_active_size_v<DstT> == 1 && CompareOp_IsPerChannel(aCompare))
    {
        throw INVALIDARGUMENT(
            aCompare,
            "CompareOp flag 'PerChannel' is not supported for multi channel images and single channel output.");
    }

    auto runOverAnyChannel = [&]<typename T>(T /*isAnyChannel*/) {
        constexpr bool anyChannel = T::value;
        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::Less:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                              mpp::Lt<ComputeT, anyChannel>, RoundingMode::None, true>;
                    const mpp::Lt<ComputeT, anyChannel> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case mpp::CompareOp::LessEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {

                    using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                              mpp::Le<ComputeT, anyChannel>, RoundingMode::None, true>;
                    const mpp::Le<ComputeT, anyChannel> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case mpp::CompareOp::Eq:
            {
                using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                          mpp::Eq<ComputeT, anyChannel>, RoundingMode::None, true>;
                const mpp::Eq<ComputeT, anyChannel> op;
                const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
            }
            break;
            case mpp::CompareOp::Greater:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }

                else
                {
                    using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                              mpp::Gt<ComputeT, anyChannel>, RoundingMode::None, true>;
                    const mpp::Gt<ComputeT, anyChannel> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case mpp::CompareOp::GreaterEq:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, only Eq and NEq are supported.");
                }
                else
                {
                    using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                              mpp::Ge<ComputeT, anyChannel>, RoundingMode::None, true>;
                    const mpp::Ge<ComputeT, anyChannel> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case mpp::CompareOp::NEq:
            {
                using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                          mpp::NEq<ComputeT, anyChannel>, RoundingMode::None, true>;
                const mpp::NEq<ComputeT, anyChannel> op;
                const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                    << aCompare << ". This function only supports binary comparisons.");
        }
    };

    if (CompareOp_IsPerChannel(aCompare) && vector_active_size_v<DstT> > 1)
    {
        // do not instantiate for single channel:
        if constexpr (vector_active_size_v<DstT> > 1)
        {
            switch (CompareOp_NoFlags(aCompare))
            {
                case mpp::CompareOp::Less:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare, "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
                    }
                    else
                    {
                        using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                                  mpp::CompareLt<ComputeT>, RoundingMode::None, true>;
                        const mpp::CompareLt<ComputeT> op;
                        const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                    }
                }
                break;
                case mpp::CompareOp::LessEq:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare, "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
                    }
                    else
                    {

                        using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                                  mpp::CompareLe<ComputeT>, RoundingMode::None, true>;
                        const mpp::CompareLe<ComputeT> op;
                        const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                    }
                }
                break;
                case mpp::CompareOp::Eq:
                {
                    using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::CompareEq<ComputeT>,
                                                              RoundingMode::None, true>;
                    const mpp::CompareEq<ComputeT> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
                break;
                case mpp::CompareOp::Greater:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare, "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
                    }

                    else
                    {
                        using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                                  mpp::CompareGt<ComputeT>, RoundingMode::None, true>;
                        const mpp::CompareGt<ComputeT> op;
                        const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                    }
                }
                break;
                case mpp::CompareOp::GreaterEq:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare, "CompareOp "
                                          << aCompare
                                          << " is not supported for complex datatypes, only Eq and NEq are supported.");
                    }
                    else
                    {
                        using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                                  mpp::CompareGe<ComputeT>, RoundingMode::None, true>;
                        const mpp::CompareGe<ComputeT> op;
                        const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize,
                                                                                      aStreamCtx, functor);
                    }
                }
                break;
                case mpp::CompareOp::NEq:
                {
                    using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                              mpp::CompareNEq<ComputeT>, RoundingMode::None, true>;
                    const mpp::CompareNEq<ComputeT> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                        << aCompare
                                                        << ". This function only supports binary comparisons.");
            }
        }
    }
    else
    {
        // do not instantiate for multi channel:
        if constexpr (vector_active_size_v<DstT> == 1)
        {
            if (CompareOp_IsAnyChannel(aCompare))
            {
                runOverAnyChannel(std::true_type{});
            }
            else
            {
                runOverAnyChannel(std::false_type{});
            }
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeCompareSrcDevC_For(typeSrc, typeDst)                                                          \
    template void InvokeCompareSrcDevC<typeSrc, typeSrc, typeDst>(                                                     \
        const typeSrc *aSrc, size_t aPitchSrc, const typeSrc *aConst, typeDst *aDst, size_t aPitchDst,                 \
        CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeCompareSrcDevC(type)                                                                \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C1, Pixel8uC1);                                                   \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C2, Pixel8uC1);                                                   \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C3, Pixel8uC1);                                                   \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C4, Pixel8uC1);

#define ForAllChannelsWithAlphaInvokeCompareSrcDevC(type)                                                              \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C1, Pixel8uC1);                                                   \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C2, Pixel8uC1);                                                   \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C3, Pixel8uC1);                                                   \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C4, Pixel8uC1);                                                   \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C4A, Pixel8uC1);

#define ForAllChannelsNoAlphaInvokeCompareSrcDevCAnyChannel(type)                                                      \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C2, Pixel8uC2);                                                   \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C3, Pixel8uC3);                                                   \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C4, Pixel8uC4);

#define ForAllChannelsWithAlphaInvokeCompareSrcDevCAnyChannel(type)                                                    \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C2, Pixel8uC2);                                                   \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C3, Pixel8uC3);                                                   \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C4, Pixel8uC4);                                                   \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C4A, Pixel8uC4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeCompareSrc(const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst, CompareOp aCompare,
                      const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    MPP_CUDA_REGISTER_TEMPALTE;

    constexpr size_t TupelSize = vector_size_v<SrcT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(DstT)>::value;

    if (vector_active_size_v<SrcT> > 1 && vector_active_size_v<DstT> == 1 && CompareOp_IsPerChannel(aCompare))
    {
        throw INVALIDARGUMENT(
            aCompare,
            "CompareOp flag 'PerChannel' is not supported for multi channel images and single channel output.");
    }

    auto runOverAnyChannel = [&]<typename T>(T /*isAnyChannel*/) {
        constexpr bool anyChannel = T::value;
        switch (CompareOp_NoFlags(aCompare))
        {
            case mpp::CompareOp::IsFinite:
            {
                using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsFinite<ComputeT, anyChannel>,
                                              RoundingMode::None, voidType, voidType, true>;
                const mpp::IsFinite<ComputeT, anyChannel> op;
                const compareSrc functor(aSrc, aPitchSrc, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                             functor);
            }
            break;
            case mpp::CompareOp::IsNaN:
            {
                using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsNaN<ComputeT, anyChannel>,
                                              RoundingMode::None, voidType, voidType, true>;
                const mpp::IsNaN<ComputeT, anyChannel> op;
                const compareSrc functor(aSrc, aPitchSrc, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                             functor);
            }
            break;
            case mpp::CompareOp::IsInf:
            {
                using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsInf<ComputeT, anyChannel>,
                                              RoundingMode::None, voidType, voidType, true>;
                const mpp::IsInf<ComputeT, anyChannel> op;
                const compareSrc functor(aSrc, aPitchSrc, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                             functor);
            }
            break;
            case mpp::CompareOp::IsInfOrNaN:
            {
                using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsInfOrNaN<ComputeT, anyChannel>,
                                              RoundingMode::None, voidType, voidType, true>;
                const mpp::IsInfOrNaN<ComputeT, anyChannel> op;
                const compareSrc functor(aSrc, aPitchSrc, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                             functor);
            }
            break;
            case mpp::CompareOp::IsPositiveInf:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, use IsInf without sign instead.");
                }
                else
                {
                    using compareSrc =
                        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsPositiveInf<ComputeT, anyChannel>,
                                   RoundingMode::None, voidType, voidType, true>;
                    const mpp::IsPositiveInf<ComputeT, anyChannel> op;
                    const compareSrc functor(aSrc, aPitchSrc, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                 functor);
                }
            }
            break;
            case mpp::CompareOp::IsNegativeInf:
            {
                if constexpr (ComplexVector<SrcT>)
                {
                    throw INVALIDARGUMENT(
                        aCompare, "CompareOp "
                                      << aCompare
                                      << " is not supported for complex datatypes, use IsInf without sign instead.");
                }
                else
                {
                    using compareSrc =
                        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsNegativeInf<ComputeT, anyChannel>,
                                   RoundingMode::None, voidType, voidType, true>;
                    const mpp::IsNegativeInf<ComputeT, anyChannel> op;
                    const compareSrc functor(aSrc, aPitchSrc, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                 functor);
                }
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare,
                                      "Unsupported CompareOp: "
                                          << aCompare
                                          << ". This function only supports unary comparisons (IsInf, IsNaN, etc.).");
        }
    };

    if (CompareOp_IsPerChannel(aCompare) && vector_active_size_v<DstT> > 1)
    {
        // do not instantiate for single channel:
        if constexpr (vector_active_size_v<DstT> > 1)
        {
            constexpr bool anyChannel = false;
            switch (CompareOp_NoFlags(aCompare))
            {
                case mpp::CompareOp::IsFinite:
                {
                    using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsFinite<ComputeT, anyChannel>,
                                                  RoundingMode::None, voidType, voidType, true>;
                    const mpp::IsFinite<ComputeT, anyChannel> op;
                    const compareSrc functor(aSrc, aPitchSrc, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                 functor);
                }
                break;
                case mpp::CompareOp::IsNaN:
                {
                    using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsNaN<ComputeT, anyChannel>,
                                                  RoundingMode::None, voidType, voidType, true>;
                    const mpp::IsNaN<ComputeT, anyChannel> op;
                    const compareSrc functor(aSrc, aPitchSrc, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                 functor);
                }
                break;
                case mpp::CompareOp::IsInf:
                {
                    using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsInf<ComputeT, anyChannel>,
                                                  RoundingMode::None, voidType, voidType, true>;
                    const mpp::IsInf<ComputeT, anyChannel> op;
                    const compareSrc functor(aSrc, aPitchSrc, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                 functor);
                }
                break;
                case mpp::CompareOp::IsInfOrNaN:
                {
                    using compareSrc =
                        SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsInfOrNaN<ComputeT, anyChannel>,
                                   RoundingMode::None, voidType, voidType, true>;
                    const mpp::IsInfOrNaN<ComputeT, anyChannel> op;
                    const compareSrc functor(aSrc, aPitchSrc, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                 functor);
                }
                break;
                case mpp::CompareOp::IsPositiveInf:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare,
                            "CompareOp " << aCompare
                                         << " is not supported for complex datatypes, use IsInf without sign instead.");
                    }
                    else
                    {
                        using compareSrc =
                            SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsPositiveInf<ComputeT, anyChannel>,
                                       RoundingMode::None, voidType, voidType, true>;
                        const mpp::IsPositiveInf<ComputeT, anyChannel> op;
                        const compareSrc functor(aSrc, aPitchSrc, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                     functor);
                    }
                }
                break;
                case mpp::CompareOp::IsNegativeInf:
                {
                    if constexpr (ComplexVector<SrcT>)
                    {
                        throw INVALIDARGUMENT(
                            aCompare,
                            "CompareOp " << aCompare
                                         << " is not supported for complex datatypes, use IsInf without sign instead.");
                    }
                    else
                    {
                        using compareSrc =
                            SrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::IsNegativeInf<ComputeT, anyChannel>,
                                       RoundingMode::None, voidType, voidType, true>;
                        const mpp::IsNegativeInf<ComputeT, anyChannel> op;
                        const compareSrc functor(aSrc, aPitchSrc, op);
                        InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                     functor);
                    }
                }
                break;
                default:
                    throw INVALIDARGUMENT(
                        aCompare, "Unsupported CompareOp: "
                                      << aCompare
                                      << ". This function only supports unary comparisons (IsInf, IsNaN, etc.).");
            }
        }
    }
    else
    {
        // do not instantiate for multi channel:
        if constexpr (vector_active_size_v<DstT> == 1)
        {
            if (CompareOp_IsAnyChannel(aCompare))
            {
                runOverAnyChannel(std::true_type{});
            }
            else
            {
                runOverAnyChannel(std::false_type{});
            }
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeCompareSrc_For(typeSrc, typeDst)                                                              \
    template void InvokeCompareSrc<typeSrc, typeSrc, typeDst>(const typeSrc *aSrc, size_t aPitchSrc, typeDst *aDst,    \
                                                              size_t aPitchDst, CompareOp aCompare,                    \
                                                              const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeCompareSrc(type)                                                                    \
    InstantiateInvokeCompareSrc_For(Pixel##type##C1, Pixel8uC1);                                                       \
    InstantiateInvokeCompareSrc_For(Pixel##type##C2, Pixel8uC1);                                                       \
    InstantiateInvokeCompareSrc_For(Pixel##type##C3, Pixel8uC1);                                                       \
    InstantiateInvokeCompareSrc_For(Pixel##type##C4, Pixel8uC1);

#define ForAllChannelsWithAlphaInvokeCompareSrc(type)                                                                  \
    InstantiateInvokeCompareSrc_For(Pixel##type##C1, Pixel8uC1);                                                       \
    InstantiateInvokeCompareSrc_For(Pixel##type##C2, Pixel8uC1);                                                       \
    InstantiateInvokeCompareSrc_For(Pixel##type##C3, Pixel8uC1);                                                       \
    InstantiateInvokeCompareSrc_For(Pixel##type##C4, Pixel8uC1);                                                       \
    InstantiateInvokeCompareSrc_For(Pixel##type##C4A, Pixel8uC1);

#define ForAllChannelsNoAlphaInvokeCompareSrcAnyChannel(type)                                                          \
    InstantiateInvokeCompareSrc_For(Pixel##type##C2, Pixel8uC2);                                                       \
    InstantiateInvokeCompareSrc_For(Pixel##type##C3, Pixel8uC3);                                                       \
    InstantiateInvokeCompareSrc_For(Pixel##type##C4, Pixel8uC4);

#define ForAllChannelsWithAlphaInvokeCompareSrcAnyChannel(type)                                                        \
    InstantiateInvokeCompareSrc_For(Pixel##type##C2, Pixel8uC2);                                                       \
    InstantiateInvokeCompareSrc_For(Pixel##type##C3, Pixel8uC3);                                                       \
    InstantiateInvokeCompareSrc_For(Pixel##type##C4, Pixel8uC4);                                                       \
    InstantiateInvokeCompareSrc_For(Pixel##type##C4A, Pixel8uC4A);

#pragma endregion

} // namespace mpp::image::cuda
