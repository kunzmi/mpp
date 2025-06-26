#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/streamCtx.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <cuda_runtime.h>

namespace mpp::image::cuda
{
// compute and result types for normRelInf reduction:
template <typename SrcT> struct normRelInf_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct normRelInf_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelInf_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelInf_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelInf_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelInf_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct normRelInf_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct normRelInf_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct normRelInf_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct normRelInf_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelInf_types_scalar_for<double>
{
    using computeType = double;
    using resultType  = double;
};

// compute and result types for normRelInf reduction:
template <typename SrcT> struct normRelInf_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT,
                                          typename normRelInf_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT,
                                          typename normRelInf_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using normRelInf_types_for_ct = typename normRelInf_types_for<T>::computeType;
template <typename T> using normRelInf_types_for_rt = typename normRelInf_types_for<T>::resultType;

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeNormRelInfSrcSrc(const SrcT *aSrc1, size_t aPitchSrc1, const SrcT *aSrc2, size_t aPitchSrc2,
                            ComputeT *aTempBuffer1, ComputeT *aTempBuffer2, DstT *aDst,
                            remove_vector_t<DstT> *aDstScalar, const Size2D &aSize,
                            const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
