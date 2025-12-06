#if MPP_ENABLE_CUDA_BACKEND

#include "replaceIf.h"
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/forEachPixelKernel.h>
#include <backends/cuda/streamCtx.h>
#include <backends/cuda/templateRegistry.h>
#include <common/arithmetic/binary_operators.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/functors/inplaceConstantFunctor.h>
#include <common/image/functors/inplaceDevConstantFunctor.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
#include <common/image/functors/srcFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/pixelTypeEnabler.h>
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
template <typename SrcT, typename ComperatorT, typename CompareT> struct instantiationHelper
{
    using src_t        = SrcT;
    using comperator_t = ComperatorT;
    using compare_t    = CompareT;
};

template <typename CompareT, bool IsAnyChannel> struct instantiationHelper2
{
    using compare_t                      = CompareT;
    static constexpr bool is_any_channel = IsAnyChannel;
};

template <typename SrcDstT>
void InvokeReplaceIfSrcSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, const SrcDstT *aSrc2, size_t aPitchSrc2,
                           const SrcDstT &aValue, SrcDstT *aDst, size_t aPitchDst, CompareOp aCompare,
                           const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcDstT> && mppEnableCudaBackend<SrcDstT>)
    {
        using SrcT     = SrcDstT;
        using DstT     = SrcDstT;
        using ComputeT = SrcDstT;

        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                         << " is not supported: Flags CompareOp::AnyChannel and "
                                                            "CompareOp::PerChannel cannot be set at the same time.");
        }

        auto runComperator = [&]<typename T>(T /*instantiationHelper*/) {
            using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                ReplaceIf<SrcT, typename T::comperator_t, typename T::compare_t>,
                                                RoundingMode::None, voidType, voidType, true>;
            const ReplaceIf<SrcT, typename T::comperator_t, typename T::compare_t> op(aValue);

            const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                            functor);
        };

        auto runAnyChannel = [&]<typename T>(T /*instantiationHelper2*/) {
            constexpr bool anyChannel = T::is_any_channel;
            constexpr bool perChannel = vector_size_v<typename T::compare_t> > 1;

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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareLt<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Lt<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareLe<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Le<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
                    }
                }
                break;
                case mpp::CompareOp::Eq:
                {
                    if constexpr (perChannel)
                    {
                        runComperator(instantiationHelper<SrcT, mpp::CompareEq<ComputeT>, typename T::compare_t>{});
                    }
                    else
                    {
                        runComperator(
                            instantiationHelper<SrcT, mpp::Eq<ComputeT, anyChannel>, typename T::compare_t>{});
                    }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareGt<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Gt<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareGe<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Ge<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
                    }
                }
                break;
                case mpp::CompareOp::NEq:
                {
                    if constexpr (perChannel)
                    {
                        runComperator(instantiationHelper<SrcT, mpp::CompareNEq<ComputeT>, typename T::compare_t>{});
                    }
                    else
                    {
                        runComperator(
                            instantiationHelper<SrcT, mpp::NEq<ComputeT, anyChannel>, typename T::compare_t>{});
                    }
                }
                break;
                default:
                    throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                        << aCompare
                                                        << ". This function only supports binary comparisons.");
            }
        };

        if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            using CompareT = same_vector_size_different_type_t<SrcT, byte>;
            runAnyChannel(instantiationHelper2<CompareT, false>{});
        }
        else
        {
            using CompareT = Vector1<byte>;
            if (CompareOp_IsAnyChannel(aCompare))
            {
                runAnyChannel(instantiationHelper2<CompareT, true>{});
            }
            else
            {
                runAnyChannel(instantiationHelper2<CompareT, false>{});
            }
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeReplaceIfSrcSrc_For(typeSrcDst)                                                               \
    template void InvokeReplaceIfSrcSrc<typeSrcDst>(                                                                   \
        const typeSrcDst *aSrc1, size_t aPitchSrc1, const typeSrcDst *aSrc2, size_t aPitchSrc2,                        \
        const typeSrcDst &aValue, typeSrcDst *aDst, size_t aPitchDst, CompareOp aCompare, const Size2D &aSize,         \
        const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeReplaceIfSrcSrc(type)                                                               \
    InstantiateInvokeReplaceIfSrcSrc_For(Pixel##type##C1);                                                             \
    InstantiateInvokeReplaceIfSrcSrc_For(Pixel##type##C2);                                                             \
    InstantiateInvokeReplaceIfSrcSrc_For(Pixel##type##C3);                                                             \
    InstantiateInvokeReplaceIfSrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeReplaceIfSrcSrc(type)                                                             \
    InstantiateInvokeReplaceIfSrcSrc_For(Pixel##type##C1);                                                             \
    InstantiateInvokeReplaceIfSrcSrc_For(Pixel##type##C2);                                                             \
    InstantiateInvokeReplaceIfSrcSrc_For(Pixel##type##C3);                                                             \
    InstantiateInvokeReplaceIfSrcSrc_For(Pixel##type##C4);                                                             \
    InstantiateInvokeReplaceIfSrcSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeReplaceIfSrcC(const SrcDstT *aSrc1, size_t aPitchSrc1, const SrcDstT &aConst, const SrcDstT &aValue,
                         SrcDstT *aDst, size_t aPitchDst, CompareOp aCompare, const Size2D &aSize,
                         const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcDstT> && mppEnableCudaBackend<SrcDstT>)
    {
        using SrcT     = SrcDstT;
        using DstT     = SrcDstT;
        using ComputeT = SrcDstT;

        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                         << " is not supported: Flags CompareOp::AnyChannel and "
                                                            "CompareOp::PerChannel cannot be set at the same time.");
        }

        auto runComperator = [&]<typename T>(T /*instantiationHelper*/) {
            using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                   ReplaceIf<SrcT, typename T::comperator_t, typename T::compare_t>,
                                                   RoundingMode::None, voidType, voidType, true>;
            const ReplaceIf<SrcT, typename T::comperator_t, typename T::compare_t> op(aValue);

            const compareSrcC functor(aSrc1, aPitchSrc1, aConst, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        };

        auto runAnyChannel = [&]<typename T>(T /*instantiationHelper2*/) {
            constexpr bool anyChannel = T::is_any_channel;
            constexpr bool perChannel = vector_size_v<typename T::compare_t> > 1;

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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareLt<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Lt<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareLe<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Le<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
                    }
                }
                break;
                case mpp::CompareOp::Eq:
                {
                    if constexpr (perChannel)
                    {
                        runComperator(instantiationHelper<SrcT, mpp::CompareEq<ComputeT>, typename T::compare_t>{});
                    }
                    else
                    {
                        runComperator(
                            instantiationHelper<SrcT, mpp::Eq<ComputeT, anyChannel>, typename T::compare_t>{});
                    }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareGt<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Gt<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareGe<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Ge<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
                    }
                }
                break;
                case mpp::CompareOp::NEq:
                {
                    if constexpr (perChannel)
                    {
                        runComperator(instantiationHelper<SrcT, mpp::CompareNEq<ComputeT>, typename T::compare_t>{});
                    }
                    else
                    {
                        runComperator(
                            instantiationHelper<SrcT, mpp::NEq<ComputeT, anyChannel>, typename T::compare_t>{});
                    }
                }
                break;
                default:
                    throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                        << aCompare
                                                        << ". This function only supports binary comparisons.");
            }
        };

        if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            using CompareT = same_vector_size_different_type_t<SrcT, byte>;
            runAnyChannel(instantiationHelper2<CompareT, false>{});
        }
        else
        {
            using CompareT = Vector1<byte>;
            if (CompareOp_IsAnyChannel(aCompare))
            {
                runAnyChannel(instantiationHelper2<CompareT, true>{});
            }
            else
            {
                runAnyChannel(instantiationHelper2<CompareT, false>{});
            }
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeReplaceIfSrcC_For(typeSrcDst)                                                                 \
    template void InvokeReplaceIfSrcC<typeSrcDst>(                                                                     \
        const typeSrcDst *aSrc1, size_t aPitchSrc1, const typeSrcDst &aConst, const typeSrcDst &aValue,                \
        typeSrcDst *aDst, size_t aPitchDst, CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeReplaceIfSrcC(type)                                                                 \
    InstantiateInvokeReplaceIfSrcC_For(Pixel##type##C1);                                                               \
    InstantiateInvokeReplaceIfSrcC_For(Pixel##type##C2);                                                               \
    InstantiateInvokeReplaceIfSrcC_For(Pixel##type##C3);                                                               \
    InstantiateInvokeReplaceIfSrcC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeReplaceIfSrcC(type)                                                               \
    InstantiateInvokeReplaceIfSrcC_For(Pixel##type##C1);                                                               \
    InstantiateInvokeReplaceIfSrcC_For(Pixel##type##C2);                                                               \
    InstantiateInvokeReplaceIfSrcC_For(Pixel##type##C3);                                                               \
    InstantiateInvokeReplaceIfSrcC_For(Pixel##type##C4);                                                               \
    InstantiateInvokeReplaceIfSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeReplaceIfSrcDevC(const SrcDstT *aSrc1, size_t aPitchSrc1, const SrcDstT *aConst, const SrcDstT &aValue,
                            SrcDstT *aDst, size_t aPitchDst, CompareOp aCompare, const Size2D &aSize,
                            const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcDstT> && mppEnableCudaBackend<SrcDstT>)
    {
        using SrcT     = SrcDstT;
        using DstT     = SrcDstT;
        using ComputeT = SrcDstT;

        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                         << " is not supported: Flags CompareOp::AnyChannel and "
                                                            "CompareOp::PerChannel cannot be set at the same time.");
        }

        auto runComperator = [&]<typename T>(T /*instantiationHelper*/) {
            using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT,
                                                      ReplaceIf<SrcT, typename T::comperator_t, typename T::compare_t>,
                                                      RoundingMode::None, true>;
            const ReplaceIf<SrcT, typename T::comperator_t, typename T::compare_t> op(aValue);

            const compareSrcC functor(aSrc1, aPitchSrc1, aConst, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        };

        auto runAnyChannel = [&]<typename T>(T /*instantiationHelper2*/) {
            constexpr bool anyChannel = T::is_any_channel;
            constexpr bool perChannel = vector_size_v<typename T::compare_t> > 1;

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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareLt<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Lt<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareLe<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Le<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
                    }
                }
                break;
                case mpp::CompareOp::Eq:
                {
                    if constexpr (perChannel)
                    {
                        runComperator(instantiationHelper<SrcT, mpp::CompareEq<ComputeT>, typename T::compare_t>{});
                    }
                    else
                    {
                        runComperator(
                            instantiationHelper<SrcT, mpp::Eq<ComputeT, anyChannel>, typename T::compare_t>{});
                    }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareGt<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Gt<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareGe<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Ge<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
                    }
                }
                break;
                case mpp::CompareOp::NEq:
                {
                    if constexpr (perChannel)
                    {
                        runComperator(instantiationHelper<SrcT, mpp::CompareNEq<ComputeT>, typename T::compare_t>{});
                    }
                    else
                    {
                        runComperator(
                            instantiationHelper<SrcT, mpp::NEq<ComputeT, anyChannel>, typename T::compare_t>{});
                    }
                }
                break;
                default:
                    throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                        << aCompare
                                                        << ". This function only supports binary comparisons.");
            }
        };

        if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            using CompareT = same_vector_size_different_type_t<SrcT, byte>;
            runAnyChannel(instantiationHelper2<CompareT, false>{});
        }
        else
        {
            using CompareT = Vector1<byte>;
            if (CompareOp_IsAnyChannel(aCompare))
            {
                runAnyChannel(instantiationHelper2<CompareT, true>{});
            }
            else
            {
                runAnyChannel(instantiationHelper2<CompareT, false>{});
            }
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeReplaceIfSrcDevC_For(typeSrcDst)                                                              \
    template void InvokeReplaceIfSrcDevC<typeSrcDst>(                                                                  \
        const typeSrcDst *aSrc1, size_t aPitchSrc1, const typeSrcDst *aConst, const typeSrcDst &aValue,                \
        typeSrcDst *aDst, size_t aPitchDst, CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeReplaceIfSrcDevC(type)                                                              \
    InstantiateInvokeReplaceIfSrcDevC_For(Pixel##type##C1);                                                            \
    InstantiateInvokeReplaceIfSrcDevC_For(Pixel##type##C2);                                                            \
    InstantiateInvokeReplaceIfSrcDevC_For(Pixel##type##C3);                                                            \
    InstantiateInvokeReplaceIfSrcDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeReplaceIfSrcDevC(type)                                                            \
    InstantiateInvokeReplaceIfSrcDevC_For(Pixel##type##C1);                                                            \
    InstantiateInvokeReplaceIfSrcDevC_For(Pixel##type##C2);                                                            \
    InstantiateInvokeReplaceIfSrcDevC_For(Pixel##type##C3);                                                            \
    InstantiateInvokeReplaceIfSrcDevC_For(Pixel##type##C4);                                                            \
    InstantiateInvokeReplaceIfSrcDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeReplaceIfSrc(const SrcDstT *aSrc1, size_t aPitchSrc1, const SrcDstT &aValue, SrcDstT *aDst, size_t aPitchDst,
                        CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcDstT> && mppEnableCudaBackend<SrcDstT>)
    {
        using SrcT     = SrcDstT;
        using DstT     = SrcDstT;
        using ComputeT = SrcDstT;

        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                         << " is not supported: Flags CompareOp::AnyChannel and "
                                                            "CompareOp::PerChannel cannot be set at the same time.");
        }

        auto runComperator = [&]<typename T>(T /*instantiationHelper*/) {
            using compareSrc = SrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                                          ReplaceIfFP<SrcT, typename T::comperator_t, typename T::compare_t>,
                                          RoundingMode::None, voidType, voidType, true>;
            const ReplaceIfFP<SrcT, typename T::comperator_t, typename T::compare_t> op(aValue);

            const compareSrc functor(aSrc1, aPitchSrc1, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrc>(aDst, aPitchDst, aSize, aStreamCtx, functor);
        };

        auto runAnyChannel = [&]<typename T>(T /*instantiationHelper2*/) {
            constexpr bool anyChannel = T::is_any_channel;

            switch (CompareOp_NoFlags(aCompare))
            {
                case mpp::CompareOp::IsFinite:
                {
                    runComperator(
                        instantiationHelper<SrcT, mpp::IsFinite<ComputeT, anyChannel>, typename T::compare_t>{});
                }
                break;
                case mpp::CompareOp::IsNaN:
                {
                    runComperator(instantiationHelper<SrcT, mpp::IsNaN<ComputeT, anyChannel>, typename T::compare_t>{});
                }
                break;
                case mpp::CompareOp::IsInf:
                {
                    runComperator(instantiationHelper<SrcT, mpp::IsInf<ComputeT, anyChannel>, typename T::compare_t>{});
                }
                break;
                case mpp::CompareOp::IsInfOrNaN:
                {
                    runComperator(
                        instantiationHelper<SrcT, mpp::IsInfOrNaN<ComputeT, anyChannel>, typename T::compare_t>{});
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
                        runComperator(instantiationHelper<SrcT, mpp::IsPositiveInf<ComputeT, anyChannel>,
                                                          typename T::compare_t>{});
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
                        runComperator(instantiationHelper<SrcT, mpp::IsNegativeInf<ComputeT, anyChannel>,
                                                          typename T::compare_t>{});
                    }
                }
                break;
                default:
                    throw INVALIDARGUMENT(
                        aCompare, "Unsupported CompareOp: "
                                      << aCompare
                                      << ". This function only supports unary comparisons (IsInf, IsNaN, etc.).");
            }
        };

        if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            using CompareT = same_vector_size_different_type_t<SrcT, byte>;
            runAnyChannel(instantiationHelper2<CompareT, false>{});
        }
        else
        {
            using CompareT = Vector1<byte>;
            if (CompareOp_IsAnyChannel(aCompare))
            {
                runAnyChannel(instantiationHelper2<CompareT, true>{});
            }
            else
            {
                runAnyChannel(instantiationHelper2<CompareT, false>{});
            }
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeReplaceIfSrc_For(typeSrcDst)                                                                  \
    template void InvokeReplaceIfSrc<typeSrcDst>(const typeSrcDst *aSrc1, size_t aPitchSrc1, const typeSrcDst &aValue, \
                                                 typeSrcDst *aDst, size_t aPitchDst, CompareOp aCompare,               \
                                                 const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeReplaceIfSrc(type)                                                                  \
    InstantiateInvokeReplaceIfSrc_For(Pixel##type##C1);                                                                \
    InstantiateInvokeReplaceIfSrc_For(Pixel##type##C2);                                                                \
    InstantiateInvokeReplaceIfSrc_For(Pixel##type##C3);                                                                \
    InstantiateInvokeReplaceIfSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeReplaceIfSrc(type)                                                                \
    InstantiateInvokeReplaceIfSrc_For(Pixel##type##C1);                                                                \
    InstantiateInvokeReplaceIfSrc_For(Pixel##type##C2);                                                                \
    InstantiateInvokeReplaceIfSrc_For(Pixel##type##C3);                                                                \
    InstantiateInvokeReplaceIfSrc_For(Pixel##type##C4);                                                                \
    InstantiateInvokeReplaceIfSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeReplaceIfInplaceSrcSrc(SrcDstT *aSrcDst, size_t aPitchSrcDst, const SrcDstT *aSrc2, size_t aPitchSrc2,
                                  const SrcDstT &aValue, CompareOp aCompare, const Size2D &aSize,
                                  const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcDstT> && mppEnableCudaBackend<SrcDstT>)
    {
        using SrcT     = SrcDstT;
        using DstT     = SrcDstT;
        using ComputeT = SrcDstT;

        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                         << " is not supported: Flags CompareOp::AnyChannel and "
                                                            "CompareOp::PerChannel cannot be set at the same time.");
        }

        auto runComperator = [&]<typename T>(T /*instantiationHelper*/) {
            using compareInplaceSrc =
                InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT,
                                  ReplaceIf<SrcT, typename T::comperator_t, typename T::compare_t>, RoundingMode::None,
                                  voidType, voidType>;
            const ReplaceIf<SrcT, typename T::comperator_t, typename T::compare_t> op(aValue);

            const compareInplaceSrc functor(aSrc2, aPitchSrc2, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, compareInplaceSrc>(aSrcDst, aPitchSrcDst, aSize,
                                                                                aStreamCtx, functor);
        };

        auto runAnyChannel = [&]<typename T>(T /*instantiationHelper2*/) {
            constexpr bool anyChannel = T::is_any_channel;
            constexpr bool perChannel = vector_size_v<typename T::compare_t> > 1;

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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareLt<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Lt<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareLe<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Le<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
                    }
                }
                break;
                case mpp::CompareOp::Eq:
                {
                    if constexpr (perChannel)
                    {
                        runComperator(instantiationHelper<SrcT, mpp::CompareEq<ComputeT>, typename T::compare_t>{});
                    }
                    else
                    {
                        runComperator(
                            instantiationHelper<SrcT, mpp::Eq<ComputeT, anyChannel>, typename T::compare_t>{});
                    }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareGt<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Gt<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareGe<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Ge<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
                    }
                }
                break;
                case mpp::CompareOp::NEq:
                {
                    if constexpr (perChannel)
                    {
                        runComperator(instantiationHelper<SrcT, mpp::CompareNEq<ComputeT>, typename T::compare_t>{});
                    }
                    else
                    {
                        runComperator(
                            instantiationHelper<SrcT, mpp::NEq<ComputeT, anyChannel>, typename T::compare_t>{});
                    }
                }
                break;
                default:
                    throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                        << aCompare
                                                        << ". This function only supports binary comparisons.");
            }
        };

        if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            using CompareT = same_vector_size_different_type_t<SrcT, byte>;
            runAnyChannel(instantiationHelper2<CompareT, false>{});
        }
        else
        {
            using CompareT = Vector1<byte>;
            if (CompareOp_IsAnyChannel(aCompare))
            {
                runAnyChannel(instantiationHelper2<CompareT, true>{});
            }
            else
            {
                runAnyChannel(instantiationHelper2<CompareT, false>{});
            }
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeReplaceIfInplaceSrcSrc_For(typeSrcDst)                                                        \
    template void InvokeReplaceIfInplaceSrcSrc<typeSrcDst>(                                                            \
        typeSrcDst * aSrcDst, size_t aPitchSrcDst, const typeSrcDst *aSrc2, size_t aPitchSrc2,                         \
        const typeSrcDst &aValue, CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrcSrc(type)                                                        \
    InstantiateInvokeReplaceIfInplaceSrcSrc_For(Pixel##type##C1);                                                      \
    InstantiateInvokeReplaceIfInplaceSrcSrc_For(Pixel##type##C2);                                                      \
    InstantiateInvokeReplaceIfInplaceSrcSrc_For(Pixel##type##C3);                                                      \
    InstantiateInvokeReplaceIfInplaceSrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcSrc(type)                                                      \
    InstantiateInvokeReplaceIfInplaceSrcSrc_For(Pixel##type##C1);                                                      \
    InstantiateInvokeReplaceIfInplaceSrcSrc_For(Pixel##type##C2);                                                      \
    InstantiateInvokeReplaceIfInplaceSrcSrc_For(Pixel##type##C3);                                                      \
    InstantiateInvokeReplaceIfInplaceSrcSrc_For(Pixel##type##C4);                                                      \
    InstantiateInvokeReplaceIfInplaceSrcSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeReplaceIfInplaceSrcC(SrcDstT *aSrcDst, size_t aPitchSrcDst, const SrcDstT &aConst, const SrcDstT &aValue,
                                CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcDstT> && mppEnableCudaBackend<SrcDstT>)
    {
        using SrcT     = SrcDstT;
        using DstT     = SrcDstT;
        using ComputeT = SrcDstT;

        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                         << " is not supported: Flags CompareOp::AnyChannel and "
                                                            "CompareOp::PerChannel cannot be set at the same time.");
        }

        auto runComperator = [&]<typename T>(T /*instantiationHelper*/) {
            using compareInplaceC =
                InplaceConstantFunctor<TupelSize, ComputeT, DstT,
                                       ReplaceIf<SrcT, typename T::comperator_t, typename T::compare_t>,
                                       RoundingMode::None, voidType, voidType>;
            const ReplaceIf<SrcT, typename T::comperator_t, typename T::compare_t> op(aValue);

            const compareInplaceC functor(aConst, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, compareInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
        };

        auto runAnyChannel = [&]<typename T>(T /*instantiationHelper2*/) {
            constexpr bool anyChannel = T::is_any_channel;
            constexpr bool perChannel = vector_size_v<typename T::compare_t> > 1;

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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareLt<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Lt<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareLe<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Le<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
                    }
                }
                break;
                case mpp::CompareOp::Eq:
                {
                    if constexpr (perChannel)
                    {
                        runComperator(instantiationHelper<SrcT, mpp::CompareEq<ComputeT>, typename T::compare_t>{});
                    }
                    else
                    {
                        runComperator(
                            instantiationHelper<SrcT, mpp::Eq<ComputeT, anyChannel>, typename T::compare_t>{});
                    }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareGt<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Gt<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareGe<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Ge<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
                    }
                }
                break;
                case mpp::CompareOp::NEq:
                {
                    if constexpr (perChannel)
                    {
                        runComperator(instantiationHelper<SrcT, mpp::CompareNEq<ComputeT>, typename T::compare_t>{});
                    }
                    else
                    {
                        runComperator(
                            instantiationHelper<SrcT, mpp::NEq<ComputeT, anyChannel>, typename T::compare_t>{});
                    }
                }
                break;
                default:
                    throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                        << aCompare
                                                        << ". This function only supports binary comparisons.");
            }
        };

        if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            using CompareT = same_vector_size_different_type_t<SrcT, byte>;
            runAnyChannel(instantiationHelper2<CompareT, false>{});
        }
        else
        {
            using CompareT = Vector1<byte>;
            if (CompareOp_IsAnyChannel(aCompare))
            {
                runAnyChannel(instantiationHelper2<CompareT, true>{});
            }
            else
            {
                runAnyChannel(instantiationHelper2<CompareT, false>{});
            }
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeReplaceIfInplaceSrcC_For(typeSrcDst)                                                          \
    template void InvokeReplaceIfInplaceSrcC<typeSrcDst>(                                                              \
        typeSrcDst * aSrcDst, size_t aPitchSrcDst, const typeSrcDst &aConst, const typeSrcDst &aValue,                 \
        CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrcC(type)                                                          \
    InstantiateInvokeReplaceIfInplaceSrcC_For(Pixel##type##C1);                                                        \
    InstantiateInvokeReplaceIfInplaceSrcC_For(Pixel##type##C2);                                                        \
    InstantiateInvokeReplaceIfInplaceSrcC_For(Pixel##type##C3);                                                        \
    InstantiateInvokeReplaceIfInplaceSrcC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcC(type)                                                        \
    InstantiateInvokeReplaceIfInplaceSrcC_For(Pixel##type##C1);                                                        \
    InstantiateInvokeReplaceIfInplaceSrcC_For(Pixel##type##C2);                                                        \
    InstantiateInvokeReplaceIfInplaceSrcC_For(Pixel##type##C3);                                                        \
    InstantiateInvokeReplaceIfInplaceSrcC_For(Pixel##type##C4);                                                        \
    InstantiateInvokeReplaceIfInplaceSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeReplaceIfInplaceSrcDevC(SrcDstT *aSrcDst, size_t aPitchSrcDst, const SrcDstT *aConst, const SrcDstT &aValue,
                                   CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcDstT> && mppEnableCudaBackend<SrcDstT>)
    {
        using SrcT     = SrcDstT;
        using DstT     = SrcDstT;
        using ComputeT = SrcDstT;

        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                         << " is not supported: Flags CompareOp::AnyChannel and "
                                                            "CompareOp::PerChannel cannot be set at the same time.");
        }

        auto runComperator = [&]<typename T>(T /*instantiationHelper*/) {
            using compareInplaceC =
                InplaceDevConstantFunctor<TupelSize, ComputeT, DstT,
                                          ReplaceIf<SrcT, typename T::comperator_t, typename T::compare_t>,
                                          RoundingMode::None>;
            const ReplaceIf<SrcT, typename T::comperator_t, typename T::compare_t> op(aValue);

            const compareInplaceC functor(aConst, op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, compareInplaceC>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                              functor);
        };

        auto runAnyChannel = [&]<typename T>(T /*instantiationHelper2*/) {
            constexpr bool anyChannel = T::is_any_channel;
            constexpr bool perChannel = vector_size_v<typename T::compare_t> > 1;

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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareLt<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Lt<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareLe<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Le<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
                    }
                }
                break;
                case mpp::CompareOp::Eq:
                {
                    if constexpr (perChannel)
                    {
                        runComperator(instantiationHelper<SrcT, mpp::CompareEq<ComputeT>, typename T::compare_t>{});
                    }
                    else
                    {
                        runComperator(
                            instantiationHelper<SrcT, mpp::Eq<ComputeT, anyChannel>, typename T::compare_t>{});
                    }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareGt<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Gt<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
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
                        if constexpr (perChannel)
                        {
                            runComperator(instantiationHelper<SrcT, mpp::CompareGe<ComputeT>, typename T::compare_t>{});
                        }
                        else
                        {
                            runComperator(
                                instantiationHelper<SrcT, mpp::Ge<ComputeT, anyChannel>, typename T::compare_t>{});
                        }
                    }
                }
                break;
                case mpp::CompareOp::NEq:
                {
                    if constexpr (perChannel)
                    {
                        runComperator(instantiationHelper<SrcT, mpp::CompareNEq<ComputeT>, typename T::compare_t>{});
                    }
                    else
                    {
                        runComperator(
                            instantiationHelper<SrcT, mpp::NEq<ComputeT, anyChannel>, typename T::compare_t>{});
                    }
                }
                break;
                default:
                    throw INVALIDARGUMENT(aCompare, "Unsupported CompareOp: "
                                                        << aCompare
                                                        << ". This function only supports binary comparisons.");
            }
        };

        if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            using CompareT = same_vector_size_different_type_t<SrcT, byte>;
            runAnyChannel(instantiationHelper2<CompareT, false>{});
        }
        else
        {
            using CompareT = Vector1<byte>;
            if (CompareOp_IsAnyChannel(aCompare))
            {
                runAnyChannel(instantiationHelper2<CompareT, true>{});
            }
            else
            {
                runAnyChannel(instantiationHelper2<CompareT, false>{});
            }
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeReplaceIfInplaceSrcDevC_For(typeSrcDst)                                                       \
    template void InvokeReplaceIfInplaceSrcDevC<typeSrcDst>(                                                           \
        typeSrcDst * aSrcDst, size_t aPitchSrcDst, const typeSrcDst *aConst, const typeSrcDst &aValue,                 \
        CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrcDevC(type)                                                       \
    InstantiateInvokeReplaceIfInplaceSrcDevC_For(Pixel##type##C1);                                                     \
    InstantiateInvokeReplaceIfInplaceSrcDevC_For(Pixel##type##C2);                                                     \
    InstantiateInvokeReplaceIfInplaceSrcDevC_For(Pixel##type##C3);                                                     \
    InstantiateInvokeReplaceIfInplaceSrcDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrcDevC(type)                                                     \
    InstantiateInvokeReplaceIfInplaceSrcDevC_For(Pixel##type##C1);                                                     \
    InstantiateInvokeReplaceIfInplaceSrcDevC_For(Pixel##type##C2);                                                     \
    InstantiateInvokeReplaceIfInplaceSrcDevC_For(Pixel##type##C3);                                                     \
    InstantiateInvokeReplaceIfInplaceSrcDevC_For(Pixel##type##C4);                                                     \
    InstantiateInvokeReplaceIfInplaceSrcDevC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcDstT>
void InvokeReplaceIfInplaceSrc(SrcDstT *aSrcDst, size_t aPitchSrcDst, const SrcDstT &aValue, CompareOp aCompare,
                               const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (mppEnablePixelType<SrcDstT> && mppEnableCudaBackend<SrcDstT>)
    {
        using SrcT     = SrcDstT;
        using DstT     = SrcDstT;
        using ComputeT = SrcDstT;

        MPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = ConfigTupelSize<"Default", sizeof(SrcT)>::value;

        if (CompareOp_IsAnyChannel(aCompare) && CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            throw INVALIDARGUMENT(aCompare, "CompareOp " << aCompare
                                                         << " is not supported: Flags CompareOp::AnyChannel and "
                                                            "CompareOp::PerChannel cannot be set at the same time.");
        }

        auto runComperator = [&]<typename T>(T /*instantiationHelper*/) {
            using compareInplace = InplaceFunctor<TupelSize, ComputeT, DstT,
                                                  ReplaceIfFP<SrcT, typename T::comperator_t, typename T::compare_t>,
                                                  RoundingMode::None, voidType, voidType>;
            const ReplaceIfFP<SrcT, typename T::comperator_t, typename T::compare_t> op(aValue);

            const compareInplace functor(op);
            InvokeForEachPixelKernelDefault<DstT, TupelSize, compareInplace>(aSrcDst, aPitchSrcDst, aSize, aStreamCtx,
                                                                             functor);
        };

        auto runAnyChannel = [&]<typename T>(T /*instantiationHelper2*/) {
            constexpr bool anyChannel = T::is_any_channel;

            switch (CompareOp_NoFlags(aCompare))
            {
                case mpp::CompareOp::IsFinite:
                {
                    runComperator(
                        instantiationHelper<SrcT, mpp::IsFinite<ComputeT, anyChannel>, typename T::compare_t>{});
                }
                break;
                case mpp::CompareOp::IsNaN:
                {
                    runComperator(instantiationHelper<SrcT, mpp::IsNaN<ComputeT, anyChannel>, typename T::compare_t>{});
                }
                break;
                case mpp::CompareOp::IsInf:
                {
                    runComperator(instantiationHelper<SrcT, mpp::IsInf<ComputeT, anyChannel>, typename T::compare_t>{});
                }
                break;
                case mpp::CompareOp::IsInfOrNaN:
                {
                    runComperator(
                        instantiationHelper<SrcT, mpp::IsInfOrNaN<ComputeT, anyChannel>, typename T::compare_t>{});
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
                        runComperator(instantiationHelper<SrcT, mpp::IsPositiveInf<ComputeT, anyChannel>,
                                                          typename T::compare_t>{});
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
                        runComperator(instantiationHelper<SrcT, mpp::IsNegativeInf<ComputeT, anyChannel>,
                                                          typename T::compare_t>{});
                    }
                }
                break;
                default:
                    throw INVALIDARGUMENT(
                        aCompare, "Unsupported CompareOp: "
                                      << aCompare
                                      << ". This function only supports unary comparisons (IsInf, IsNaN, etc.).");
            }
        };

        if (CompareOp_IsPerChannel(aCompare) && vector_size_v<SrcT> > 1)
        {
            using CompareT = same_vector_size_different_type_t<SrcT, byte>;
            runAnyChannel(instantiationHelper2<CompareT, false>{});
        }
        else
        {
            using CompareT = Vector1<byte>;
            if (CompareOp_IsAnyChannel(aCompare))
            {
                runAnyChannel(instantiationHelper2<CompareT, true>{});
            }
            else
            {
                runAnyChannel(instantiationHelper2<CompareT, false>{});
            }
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeReplaceIfInplaceSrc_For(typeSrcDst)                                                           \
    template void InvokeReplaceIfInplaceSrc<typeSrcDst>(typeSrcDst * aSrcDst, size_t aPitchSrcDst,                     \
                                                        const typeSrcDst &aValue, CompareOp aCompare,                  \
                                                        const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeReplaceIfInplaceSrc(type)                                                           \
    InstantiateInvokeReplaceIfInplaceSrc_For(Pixel##type##C1);                                                         \
    InstantiateInvokeReplaceIfInplaceSrc_For(Pixel##type##C2);                                                         \
    InstantiateInvokeReplaceIfInplaceSrc_For(Pixel##type##C3);                                                         \
    InstantiateInvokeReplaceIfInplaceSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeReplaceIfInplaceSrc(type)                                                         \
    InstantiateInvokeReplaceIfInplaceSrc_For(Pixel##type##C1);                                                         \
    InstantiateInvokeReplaceIfInplaceSrc_For(Pixel##type##C2);                                                         \
    InstantiateInvokeReplaceIfInplaceSrc_For(Pixel##type##C3);                                                         \
    InstantiateInvokeReplaceIfInplaceSrc_For(Pixel##type##C4);                                                         \
    InstantiateInvokeReplaceIfInplaceSrc_For(Pixel##type##C4A);

#pragma endregion

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
