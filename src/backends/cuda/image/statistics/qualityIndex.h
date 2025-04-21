#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace opp::image::cuda
{
// compute and result types for qualityIndex reduction:
template <typename SrcT> struct qualityIndex_types_scalar_for
{
    using computeType1 = SrcT;
    using computeType2 = SrcT;
    using resultType   = SrcT;
};
template <> struct qualityIndex_types_scalar_for<byte>
{
    using computeType1 = ulong64;
    using computeType2 = ulong64;
    using resultType   = double;
};
template <> struct qualityIndex_types_scalar_for<sbyte>
{
    using computeType1 = long64;
    using computeType2 = long64;
    using resultType   = double;
};
template <> struct qualityIndex_types_scalar_for<ushort>
{
    using computeType1 = ulong64;
    using computeType2 = ulong64;
    using resultType   = double;
};
template <> struct qualityIndex_types_scalar_for<short>
{
    using computeType1 = long64;
    using computeType2 = long64;
    using resultType   = double;
};
template <> struct qualityIndex_types_scalar_for<uint>
{
    using computeType1 = ulong64;
    using computeType2 = ulong64;
    using resultType   = double;
};
template <> struct qualityIndex_types_scalar_for<int>
{
    using computeType1 = long64;
    using computeType2 = long64;
    using resultType   = double;
};

template <> struct qualityIndex_types_scalar_for<HalfFp16>
{
    using computeType1 = float;
    using computeType2 = float;
    using resultType   = float;
};
template <> struct qualityIndex_types_scalar_for<BFloat16>
{
    using computeType1 = float;
    using computeType2 = float;
    using resultType   = float;
};

template <> struct qualityIndex_types_scalar_for<float>
{
    using computeType1 = float;
    using computeType2 = double;
    using resultType   = double;
};
template <> struct qualityIndex_types_scalar_for<double>
{
    using computeType1 = double;
    using computeType2 = double;
    using resultType   = double;
};

// compute and result types for qualityIndex reduction:
template <typename SrcT> struct qualityIndex_types_for
{
    using computeType1 =
        same_vector_size_different_type_t<SrcT,
                                          typename qualityIndex_types_scalar_for<remove_vector_t<SrcT>>::computeType1>;
    using computeType2 =
        same_vector_size_different_type_t<SrcT,
                                          typename qualityIndex_types_scalar_for<remove_vector_t<SrcT>>::computeType2>;
    using resultType =
        same_vector_size_different_type_t<SrcT,
                                          typename qualityIndex_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using qualityIndex_types_for_ct1 = typename qualityIndex_types_for<T>::computeType1;
template <typename T> using qualityIndex_types_for_ct2 = typename qualityIndex_types_for<T>::computeType2;
template <typename T> using qualityIndex_types_for_rt  = typename qualityIndex_types_for<T>::resultType;

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeQualityIndexSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2,
                              ComputeT *aTempBuffer1, ComputeT *aTempBuffer2, ComputeT *aTempBuffer3,
                              ComputeT *aTempBuffer4, ComputeT *aTempBuffer5, DstT *aDst, const Size2D &aSize,
                              const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
