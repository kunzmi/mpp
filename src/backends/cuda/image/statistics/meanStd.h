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
// compute and result types for meanStd reduction:
template <typename SrcT> struct meanStd_types_scalar_for
{
    using computeType = SrcT;
    using resultType1 = SrcT;
    using resultType2 = SrcT;
};
template <> struct meanStd_types_scalar_for<byte>
{
    using computeType = ulong64;
    using resultType1 = double;
    using resultType2 = double;
};
template <> struct meanStd_types_scalar_for<sbyte>
{
    using computeType = long64;
    using resultType1 = double;
    using resultType2 = double;
};
template <> struct meanStd_types_scalar_for<ushort>
{
    using computeType = ulong64;
    using resultType1 = double;
    using resultType2 = double;
};
template <> struct meanStd_types_scalar_for<short>
{
    using computeType = long64;
    using resultType1 = double;
    using resultType2 = double;
};
template <> struct meanStd_types_scalar_for<uint>
{
    using computeType = ulong64;
    using resultType1 = double;
    using resultType2 = double;
};
template <> struct meanStd_types_scalar_for<int>
{
    using computeType = long64;
    using resultType1 = double;
    using resultType2 = double;
};

template <> struct meanStd_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType1 = float;
    using resultType2 = float;
};
template <> struct meanStd_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType1 = float;
    using resultType2 = float;
};

template <> struct meanStd_types_scalar_for<float>
{
    // using 64f as intermediate format, slows down by factor 2 (roughly) and
    // with 32f precision is still better than NPP compared to matlab...
    using computeType = float;
    using resultType1 = double;
    using resultType2 = double;
};
template <> struct meanStd_types_scalar_for<double>
{
    using computeType = double;
    using resultType1 = double;
    using resultType2 = double;
};

template <> struct meanStd_types_scalar_for<c_short>
{
    using computeType = c_long;
    using resultType1 = c_double;
    using resultType2 = double;
};

template <> struct meanStd_types_scalar_for<c_int>
{
    using computeType = c_long;
    using resultType1 = c_double;
    using resultType2 = double;
};

template <> struct meanStd_types_scalar_for<c_float>
{
    using computeType = c_float;
    using resultType1 = c_double;
    using resultType2 = double;
};

// compute and result types for sum reduction:
template <typename SrcT> struct meanStd_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT, typename meanStd_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType1 =
        same_vector_size_different_type_t<SrcT, typename meanStd_types_scalar_for<remove_vector_t<SrcT>>::resultType1>;
    using resultType2 =
        same_vector_size_different_type_t<SrcT, typename meanStd_types_scalar_for<remove_vector_t<SrcT>>::resultType2>;
};

template <typename T> using meanStd_types_for_ct  = typename meanStd_types_for<T>::computeType;
template <typename T> using meanStd_types_for_rt1 = typename meanStd_types_for<T>::resultType1;
template <typename T> using meanStd_types_for_rt2 = typename meanStd_types_for<T>::resultType2;

template <typename SrcT, typename ComputeT, typename DstT1, typename DstT2>
void InvokeMeanStdSrc(const SrcT *aSrc1, size_t aPitchSrc1, ComputeT *aTempBuffer1, ComputeT *aTempBuffer2,
                      DstT1 *aDst1, DstT2 *aDst2, remove_vector_t<DstT1> *aDstScalar1,
                      remove_vector_t<DstT2> *aDstScalar2, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
