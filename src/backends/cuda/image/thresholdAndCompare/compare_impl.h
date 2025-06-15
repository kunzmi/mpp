#if OPP_ENABLE_CUDA_BACKEND

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
#include <common/image/functors/srcSrcFunctor.h>
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
template <typename SrcT, typename ComputeT, typename DstT>
void InvokeCompareSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2, DstT *aDst,
                         size_t aPitchDst, CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<SrcT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = vector_size_v<SrcT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aCompare)
        {
            case opp::CompareOp::Less:
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
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Lt<ComputeT>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const opp::Lt<ComputeT> op;
                    const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                    functor);
                }
            }
            break;
            case opp::CompareOp::LessEq:
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
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Le<ComputeT>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const opp::Le<ComputeT> op;
                    const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                    functor);
                }
            }
            break;
            case opp::CompareOp::Eq:
            {
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Eq<ComputeT>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const opp::Eq<ComputeT> op;
                const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            case opp::CompareOp::Greater:
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
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Gt<ComputeT>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const opp::Gt<ComputeT> op;
                    const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                    functor);
                }
            }
            break;
            case opp::CompareOp::GreaterEq:
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
                    using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Ge<ComputeT>,
                                                        RoundingMode::None, voidType, voidType, true>;
                    const opp::Ge<ComputeT> op;
                    const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                    functor);
                }
            }
            break;
            case opp::CompareOp::NEq:
            {
                using compareSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, opp::NEq<ComputeT>,
                                                    RoundingMode::None, voidType, voidType, true>;
                const opp::NEq<ComputeT> op;
                const compareSrcSrc functor(aSrc1, aPitchSrc1, aSrc2, aPitchSrc2, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcSrc>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare, "Unknown CompareOp: " << aCompare);
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeCompareSrcSrc_For(typeSrc)                                                                    \
    template void InvokeCompareSrcSrc<typeSrc, typeSrc, Vector1<byte>>(                                                \
        const typeSrc *aSrc1, size_t aPitchSrc1, const typeSrc *aSrc2, size_t aPitchSrc2, Vector1<byte> *aDst,         \
        size_t aPitchDst, CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeCompareSrcSrc(type)                                                                 \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeCompareSrcSrc(type)                                                               \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C1);                                                               \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C2);                                                               \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C3);                                                               \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C4);                                                               \
    InstantiateInvokeCompareSrcSrc_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeCompareSrcC(const SrcT *aSrc, size_t aPitchSrc, const SrcT &aConst, DstT *aDst, size_t aPitchDst,
                       CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = vector_size_v<SrcT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aCompare)
        {
            case opp::CompareOp::Less:
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
                    using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Lt<ComputeT>,
                                                           RoundingMode::None, voidType, voidType, true>;
                    const opp::Lt<ComputeT> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case opp::CompareOp::LessEq:
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
                    using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Le<ComputeT>,
                                                           RoundingMode::None, voidType, voidType, true>;
                    const opp::Le<ComputeT> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case opp::CompareOp::Eq:
            {
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Eq<ComputeT>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const opp::Eq<ComputeT> op;
                const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
            }
            break;
            case opp::CompareOp::Greater:
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
                    using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Gt<ComputeT>,
                                                           RoundingMode::None, voidType, voidType, true>;
                    const opp::Gt<ComputeT> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case opp::CompareOp::GreaterEq:
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
                    using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Ge<ComputeT>,
                                                           RoundingMode::None, voidType, voidType, true>;
                    const opp::Ge<ComputeT> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case opp::CompareOp::NEq:
            {
                using compareSrcC = SrcConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::NEq<ComputeT>,
                                                       RoundingMode::None, voidType, voidType, true>;
                const opp::NEq<ComputeT> op;
                const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare, "Unknown CompareOp: " << aCompare);
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeCompareSrcC_For(typeSrc)                                                                      \
    template void InvokeCompareSrcC<typeSrc, typeSrc, Vector1<byte>>(                                                  \
        const typeSrc *aSrc, size_t aPitchSrc, const typeSrc &aConst, Vector1<byte> *aDst, size_t aPitchDst,           \
        CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeCompareSrcC(type)                                                                   \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeCompareSrcC(type)                                                                 \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C1);                                                                 \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C2);                                                                 \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C3);                                                                 \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C4);                                                                 \
    InstantiateInvokeCompareSrcC_For(Pixel##type##C4A);

#pragma endregion

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeCompareSrcDevC(const SrcT *aSrc, size_t aPitchSrc, const SrcT *aConst, DstT *aDst, size_t aPitchDst,
                          CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx)
{
    if constexpr (oppEnablePixelType<DstT> && oppEnableCudaBackend<DstT>)
    {
        OPP_CUDA_REGISTER_TEMPALTE;

        constexpr size_t TupelSize = vector_size_v<SrcT> == 3 ? 1 : ConfigTupelSize<"Default", sizeof(DstT)>::value;

        switch (aCompare)
        {
            case opp::CompareOp::Less:
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
                    using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Lt<ComputeT>,
                                                              RoundingMode::None, true>;
                    const opp::Lt<ComputeT> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case opp::CompareOp::LessEq:
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

                    using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Le<ComputeT>,
                                                              RoundingMode::None, true>;
                    const opp::Le<ComputeT> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case opp::CompareOp::Eq:
            {
                using compareSrcC =
                    SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Eq<ComputeT>, RoundingMode::None, true>;
                const opp::Eq<ComputeT> op;
                const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
            }
            break;
            case opp::CompareOp::Greater:
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
                    using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Gt<ComputeT>,
                                                              RoundingMode::None, true>;
                    const opp::Gt<ComputeT> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case opp::CompareOp::GreaterEq:
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
                    using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::Ge<ComputeT>,
                                                              RoundingMode::None, true>;
                    const opp::Ge<ComputeT> op;
                    const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                    InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                                  functor);
                }
            }
            break;
            case opp::CompareOp::NEq:
            {
                using compareSrcC = SrcDevConstantFunctor<TupelSize, SrcT, ComputeT, DstT, opp::NEq<ComputeT>,
                                                          RoundingMode::None, true>;
                const opp::NEq<ComputeT> op;
                const compareSrcC functor(aSrc, aPitchSrc, aConst, op);
                InvokeForEachPixelKernelDefault<DstT, TupelSize, compareSrcC>(aDst, aPitchDst, aSize, aStreamCtx,
                                                                              functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aCompare, "Unknown CompareOp: " << aCompare);
        }
    }
}

#pragma region Instantiate

#define InstantiateInvokeCompareSrcDevC_For(typeSrc)                                                                   \
    template void InvokeCompareSrcDevC<typeSrc, typeSrc, Vector1<byte>>(                                               \
        const typeSrc *aSrc, size_t aPitchSrc, const typeSrc *aConst, Vector1<byte> *aDst, size_t aPitchDst,           \
        CompareOp aCompare, const Size2D &aSize, const StreamCtx &aStreamCtx);

#define ForAllChannelsNoAlphaInvokeCompareSrcDevC(type)                                                                \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C4);

#define ForAllChannelsWithAlphaInvokeCompareSrcDevC(type)                                                              \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C1);                                                              \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C2);                                                              \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C3);                                                              \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C4);                                                              \
    InstantiateInvokeCompareSrcDevC_For(Pixel##type##C4A);

#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
