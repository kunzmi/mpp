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

// compute and result types for sum reduction:
template <typename SrcT> struct mean_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct mean_types_scalar_for<byte>
{
    using computeType = ulong64;
    using resultType  = double;
};
template <> struct mean_types_scalar_for<sbyte>
{
    using computeType = long64;
    using resultType  = double;
};
template <> struct mean_types_scalar_for<ushort>
{
    using computeType = ulong64;
    using resultType  = double;
};
template <> struct mean_types_scalar_for<short>
{
    using computeType = long64;
    using resultType  = double;
};
template <> struct mean_types_scalar_for<uint>
{
    using computeType = ulong64;
    using resultType  = double;
};
template <> struct mean_types_scalar_for<int>
{
    using computeType = long64;
    using resultType  = double;
};

template <> struct mean_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct mean_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct mean_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct mean_types_scalar_for<double>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct mean_types_scalar_for<c_short>
{
    using computeType = c_long;
    using resultType  = c_double;
};

template <> struct mean_types_scalar_for<c_int>
{
    using computeType = c_long;
    using resultType  = c_double;
};

template <> struct mean_types_scalar_for<c_float>
{
    using computeType = c_float;
    using resultType  = c_double;
};

// compute and result types for sum reduction:
template <typename SrcT> struct mean_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT, typename mean_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT, typename mean_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using mean_types_for_ct = typename mean_types_for<T>::computeType;
template <typename T> using mean_types_for_rt = typename mean_types_for<T>::resultType;

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeMeanSrc(const SrcT *aSrc1, size_t aPitchSrc1, ComputeT *aTempBuffer, DstT *aDst,
                   remove_vector_t<DstT> *aDstScalar, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
