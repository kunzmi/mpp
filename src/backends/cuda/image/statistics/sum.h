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

// compute and result types for sum reduction:
template <typename SrcT, int variant> struct sum_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct sum_types_scalar_for<byte, 1>
{
    using computeType = ulong64;
    using resultType  = ulong64;
};
template <> struct sum_types_scalar_for<byte, 2>
{
    using computeType = ulong64;
    using resultType  = double;
};
template <> struct sum_types_scalar_for<sbyte, 1>
{
    using computeType = long64;
    using resultType  = long64;
};
template <> struct sum_types_scalar_for<sbyte, 2>
{
    using computeType = long64;
    using resultType  = double;
};
template <> struct sum_types_scalar_for<ushort, 1>
{
    using computeType = ulong64;
    using resultType  = ulong64;
};
template <> struct sum_types_scalar_for<ushort, 2>
{
    using computeType = ulong64;
    using resultType  = double;
};
template <> struct sum_types_scalar_for<short, 1>
{
    using computeType = long64;
    using resultType  = long64;
};
template <> struct sum_types_scalar_for<short, 2>
{
    using computeType = long64;
    using resultType  = double;
};
template <> struct sum_types_scalar_for<uint, 1>
{
    using computeType = ulong64;
    using resultType  = ulong64;
};
template <> struct sum_types_scalar_for<uint, 2>
{
    using computeType = ulong64;
    using resultType  = double;
};
template <> struct sum_types_scalar_for<int, 1>
{
    using computeType = long64;
    using resultType  = long64;
};
template <> struct sum_types_scalar_for<int, 2>
{
    using computeType = long64;
    using resultType  = double;
};

template <> struct sum_types_scalar_for<HalfFp16, 1>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct sum_types_scalar_for<BFloat16, 1>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct sum_types_scalar_for<float, 1>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct sum_types_scalar_for<double, 1>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct sum_types_scalar_for<c_short, 1>
{
    using computeType = c_long;
    using resultType  = c_long;
};
template <> struct sum_types_scalar_for<c_short, 2>
{
    using computeType = c_long;
    using resultType  = c_double;
};

template <> struct sum_types_scalar_for<c_int, 1>
{
    using computeType = c_long;
    using resultType  = c_long;
};
template <> struct sum_types_scalar_for<c_int, 2>
{
    using computeType = c_long;
    using resultType  = c_double;
};

template <> struct sum_types_scalar_for<c_float, 1>
{
    using computeType = c_float;
    using resultType  = c_double;
};

// compute and result types for sum reduction:
template <typename SrcT, int variant> struct sum_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT,
                                          typename sum_types_scalar_for<remove_vector_t<SrcT>, variant>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT,
                                          typename sum_types_scalar_for<remove_vector_t<SrcT>, variant>::resultType>;
};

template <typename T, int variant> using sum_types_for_ct = typename sum_types_for<T, variant>::computeType;
template <typename T, int variant> using sum_types_for_rt = typename sum_types_for<T, variant>::resultType;

template <typename SrcT, typename ComputeT, typename DstT>
void InvokeSumSrc(const SrcT *aSrc1, size_t aPitchSrc1, ComputeT *aTempBuffer, DstT *aDst,
                  remove_vector_t<DstT> *aDstScalar, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
