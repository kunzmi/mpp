#pragma once
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/vectorTypes.h>

namespace mpp::image::cuda
{
#pragma region AverageError
// compute and result types for averageError reduction:
template <typename SrcT> struct averageError_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct averageError_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct averageError_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct averageError_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct averageError_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct averageError_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct averageError_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct averageError_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct averageError_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct averageError_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct averageError_types_scalar_for<double>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct averageError_types_scalar_for<c_short>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct averageError_types_scalar_for<c_int>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct averageError_types_scalar_for<c_float>
{
    using computeType = float;
    using resultType  = double;
};

// compute and result types for averageError reduction:
template <typename SrcT> struct averageError_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT,
                                          typename averageError_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT,
                                          typename averageError_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using averageError_types_for_ct = typename averageError_types_for<T>::computeType;
template <typename T> using averageError_types_for_rt = typename averageError_types_for<T>::resultType;
#pragma endregion

#pragma region AverageRelativeError
// compute and result types for averageRelativeError reduction:
template <typename SrcT> struct averageRelativeError_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct averageRelativeError_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct averageRelativeError_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct averageRelativeError_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct averageRelativeError_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct averageRelativeError_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct averageRelativeError_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct averageRelativeError_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct averageRelativeError_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct averageRelativeError_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct averageRelativeError_types_scalar_for<double>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct averageRelativeError_types_scalar_for<c_short>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct averageRelativeError_types_scalar_for<c_int>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct averageRelativeError_types_scalar_for<c_float>
{
    using computeType = float;
    using resultType  = double;
};

// compute and result types for averageRelativeError reduction:
template <typename SrcT> struct averageRelativeError_types_for
{
    using computeType = same_vector_size_different_type_t<
        SrcT, typename averageRelativeError_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType = same_vector_size_different_type_t<
        SrcT, typename averageRelativeError_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using averageRelativeError_types_for_ct = typename averageRelativeError_types_for<T>::computeType;
template <typename T> using averageRelativeError_types_for_rt = typename averageRelativeError_types_for<T>::resultType;
#pragma endregion

#pragma region DotProduct
// compute and result types for dotProduct reduction:
template <typename SrcT> struct dotProduct_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct dotProduct_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct dotProduct_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct dotProduct_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct dotProduct_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct dotProduct_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct dotProduct_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct dotProduct_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct dotProduct_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct dotProduct_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct dotProduct_types_scalar_for<double>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct dotProduct_types_scalar_for<c_short>
{
    using computeType = c_float;
    using resultType  = c_float;
};
template <> struct dotProduct_types_scalar_for<c_int>
{
    using computeType = c_double;
    using resultType  = c_double;
};
template <> struct dotProduct_types_scalar_for<c_float>
{
    using computeType = c_double;
    using resultType  = c_double;
};

// compute and result types for dotProduct reduction:
template <typename SrcT> struct dotProduct_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT,
                                          typename dotProduct_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT,
                                          typename dotProduct_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using dotProduct_types_for_ct = typename dotProduct_types_for<T>::computeType;
template <typename T> using dotProduct_types_for_rt = typename dotProduct_types_for<T>::resultType;
#pragma endregion

#pragma region MSE
// compute and result types for mse reduction:
template <typename SrcT> struct mse_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct mse_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct mse_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct mse_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct mse_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct mse_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct mse_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct mse_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct mse_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct mse_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct mse_types_scalar_for<double>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct mse_types_scalar_for<c_short>
{
    using computeType = c_float;
    using resultType  = c_float;
};
template <> struct mse_types_scalar_for<c_int>
{
    using computeType = c_double;
    using resultType  = c_double;
};
template <> struct mse_types_scalar_for<c_float>
{
    using computeType = c_double;
    using resultType  = c_double;
};

// compute and result types for mse reduction:
template <typename SrcT> struct mse_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT, typename mse_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT, typename mse_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using mse_types_for_ct = typename mse_types_for<T>::computeType;
template <typename T> using mse_types_for_rt = typename mse_types_for<T>::resultType;
#pragma endregion

#pragma region NormDiffInf
// compute and result types for normDiffInf reduction:
template <typename SrcT> struct normDiffInf_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct normDiffInf_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffInf_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffInf_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffInf_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffInf_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct normDiffInf_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct normDiffInf_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct normDiffInf_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct normDiffInf_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffInf_types_scalar_for<double>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct normDiffInf_types_scalar_for<c_short>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct normDiffInf_types_scalar_for<c_int>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct normDiffInf_types_scalar_for<c_float>
{
    using computeType = float;
    using resultType  = double;
};

// compute and result types for normDiffInf reduction:
template <typename SrcT> struct normDiffInf_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT,
                                          typename normDiffInf_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT,
                                          typename normDiffInf_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using normDiffInf_types_for_ct = typename normDiffInf_types_for<T>::computeType;
template <typename T> using normDiffInf_types_for_rt = typename normDiffInf_types_for<T>::resultType;
#pragma endregion

#pragma region MaxRelativeError
// compute and result types for maxRelativeError reduction:
template <typename SrcT> struct maxRelativeError_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct maxRelativeError_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct maxRelativeError_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct maxRelativeError_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct maxRelativeError_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct maxRelativeError_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct maxRelativeError_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct maxRelativeError_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct maxRelativeError_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct maxRelativeError_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct maxRelativeError_types_scalar_for<double>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct maxRelativeError_types_scalar_for<c_short>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct maxRelativeError_types_scalar_for<c_int>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct maxRelativeError_types_scalar_for<c_float>
{
    using computeType = float;
    using resultType  = double;
};

// compute and result types for maxRelativeError reduction:
template <typename SrcT> struct maxRelativeError_types_for
{
    using computeType = same_vector_size_different_type_t<
        SrcT, typename maxRelativeError_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType = same_vector_size_different_type_t<
        SrcT, typename maxRelativeError_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using maxRelativeError_types_for_ct = typename maxRelativeError_types_for<T>::computeType;
template <typename T> using maxRelativeError_types_for_rt = typename maxRelativeError_types_for<T>::resultType;
#pragma endregion

#pragma region Mean
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
#pragma endregion

#pragma region MeanStd
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

#pragma endregion

#pragma region NormDiffL1
// compute and result types for normDiffL1 reduction:
template <typename SrcT> struct normDiffL1_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct normDiffL1_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffL1_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffL1_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffL1_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffL1_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct normDiffL1_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct normDiffL1_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct normDiffL1_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct normDiffL1_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffL1_types_scalar_for<double>
{
    using computeType = double;
    using resultType  = double;
};

// compute and result types for normDiffL1 reduction:
template <typename SrcT> struct normDiffL1_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT,
                                          typename normDiffL1_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT,
                                          typename normDiffL1_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using normDiffL1_types_for_ct = typename normDiffL1_types_for<T>::computeType;
template <typename T> using normDiffL1_types_for_rt = typename normDiffL1_types_for<T>::resultType;
#pragma endregion

#pragma region NormDiffL2
// compute and result types for normDiffL2 reduction:
template <typename SrcT> struct normDiffL2_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct normDiffL2_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffL2_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffL2_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffL2_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffL2_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct normDiffL2_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct normDiffL2_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct normDiffL2_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct normDiffL2_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normDiffL2_types_scalar_for<double>
{
    using computeType = double;
    using resultType  = double;
};

// compute and result types for normDiffL2 reduction:
template <typename SrcT> struct normDiffL2_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT,
                                          typename normDiffL2_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT,
                                          typename normDiffL2_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using normDiffL2_types_for_ct = typename normDiffL2_types_for<T>::computeType;
template <typename T> using normDiffL2_types_for_rt = typename normDiffL2_types_for<T>::resultType;
#pragma endregion

#pragma region NormInf
// compute and result types for normInf reduction:
template <typename SrcT> struct normInf_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct normInf_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normInf_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normInf_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normInf_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normInf_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct normInf_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct normInf_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct normInf_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct normInf_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normInf_types_scalar_for<double>
{
    using computeType = double;
    using resultType  = double;
};

// compute and result types for normInf reduction:
template <typename SrcT> struct normInf_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT, typename normInf_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT, typename normInf_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using normInf_types_for_ct = typename normInf_types_for<T>::computeType;
template <typename T> using normInf_types_for_rt = typename normInf_types_for<T>::resultType;
#pragma endregion

#pragma region NormL1
// compute and result types for normL1 reduction:
template <typename SrcT> struct normL1_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct normL1_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normL1_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normL1_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normL1_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normL1_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct normL1_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct normL1_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct normL1_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct normL1_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normL1_types_scalar_for<double>
{
    using computeType = double;
    using resultType  = double;
};

// compute and result types for normL1 reduction:
template <typename SrcT> struct normL1_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT, typename normL1_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT, typename normL1_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using normL1_types_for_ct = typename normL1_types_for<T>::computeType;
template <typename T> using normL1_types_for_rt = typename normL1_types_for<T>::resultType;

#pragma endregion

#pragma region NormL2
// compute and result types for normL2 reduction:
template <typename SrcT> struct normL2_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct normL2_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normL2_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normL2_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normL2_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normL2_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct normL2_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct normL2_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct normL2_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct normL2_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normL2_types_scalar_for<double>
{
    using computeType = double;
    using resultType  = double;
};

// compute and result types for normL2 reduction:
template <typename SrcT> struct normL2_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT, typename normL2_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT, typename normL2_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using normL2_types_for_ct = typename normL2_types_for<T>::computeType;
template <typename T> using normL2_types_for_rt = typename normL2_types_for<T>::resultType;
#pragma endregion

#pragma region NormRelInf
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
#pragma endregion

#pragma region NormRelL1
// compute and result types for normRelL1 reduction:
template <typename SrcT> struct normRelL1_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct normRelL1_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelL1_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelL1_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelL1_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelL1_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct normRelL1_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct normRelL1_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct normRelL1_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct normRelL1_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelL1_types_scalar_for<double>
{
    using computeType = double;
    using resultType  = double;
};

// compute and result types for normRelL1 reduction:
template <typename SrcT> struct normRelL1_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT,
                                          typename normRelL1_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT, typename normRelL1_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using normRelL1_types_for_ct = typename normRelL1_types_for<T>::computeType;
template <typename T> using normRelL1_types_for_rt = typename normRelL1_types_for<T>::resultType;
#pragma endregion

#pragma region NormRelL2
// compute and result types for normRelL2 reduction:
template <typename SrcT> struct normRelL2_types_scalar_for
{
    using computeType = SrcT;
    using resultType  = SrcT;
};
template <> struct normRelL2_types_scalar_for<byte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelL2_types_scalar_for<sbyte>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelL2_types_scalar_for<ushort>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelL2_types_scalar_for<short>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelL2_types_scalar_for<uint>
{
    using computeType = double;
    using resultType  = double;
};
template <> struct normRelL2_types_scalar_for<int>
{
    using computeType = double;
    using resultType  = double;
};

template <> struct normRelL2_types_scalar_for<HalfFp16>
{
    using computeType = float;
    using resultType  = float;
};
template <> struct normRelL2_types_scalar_for<BFloat16>
{
    using computeType = float;
    using resultType  = float;
};

template <> struct normRelL2_types_scalar_for<float>
{
    using computeType = float;
    using resultType  = double;
};
template <> struct normRelL2_types_scalar_for<double>
{
    using computeType = double;
    using resultType  = double;
};

// compute and result types for normRelL2 reduction:
template <typename SrcT> struct normRelL2_types_for
{
    using computeType =
        same_vector_size_different_type_t<SrcT,
                                          typename normRelL2_types_scalar_for<remove_vector_t<SrcT>>::computeType>;
    using resultType =
        same_vector_size_different_type_t<SrcT, typename normRelL2_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using normRelL2_types_for_ct = typename normRelL2_types_for<T>::computeType;
template <typename T> using normRelL2_types_for_rt = typename normRelL2_types_for<T>::resultType;
#pragma endregion

#pragma region QualityIndex
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
#pragma endregion

#pragma region QualityIndexWindow
// compute and result types for qiw reduction:
template <typename SrcT> struct qiw_types_scalar_for
{
    using resultType = float;
};
template <> struct qiw_types_scalar_for<double>
{
    using resultType = double;
};

// compute and result types for qiw reduction:
template <typename SrcT> struct qiw_types_for
{
    using resultType =
        same_vector_size_different_type_t<SrcT, typename qiw_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using qiw_types_for_rt = typename qiw_types_for<T>::resultType;
#pragma endregion

#pragma region Sum
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

#pragma endregion

#pragma region SSIM
// compute and result types for ssim reduction:
template <typename SrcT> struct ssim_types_scalar_for
{
    using resultType = float;
};
template <> struct ssim_types_scalar_for<double>
{
    using resultType = double;
};

// compute and result types for ssim reduction:
template <typename SrcT> struct ssim_types_for
{
    using resultType =
        same_vector_size_different_type_t<SrcT, typename ssim_types_scalar_for<remove_vector_t<SrcT>>::resultType>;
};

template <typename T> using ssim_types_for_rt = typename ssim_types_for<T>::resultType;
#pragma endregion

#pragma region HistEven
// level types for histogram even:
template <typename SrcT> struct hist_even_level_types_scalar_for
{
    using levelType = int;
};
template <> struct hist_even_level_types_scalar_for<uint>
{
    using levelType = ulong64;
};
template <> struct hist_even_level_types_scalar_for<int>
{
    using levelType = long64;
};

template <> struct hist_even_level_types_scalar_for<HalfFp16>
{
    using levelType = float;
};
template <> struct hist_even_level_types_scalar_for<BFloat16>
{
    using levelType = float;
};

template <> struct hist_even_level_types_scalar_for<float>
{
    using levelType = float;
};
template <> struct hist_even_level_types_scalar_for<double>
{
    using levelType = double;
};

// compute and result types for sum reduction:
template <typename SrcT> struct hist_even_level_types_for
{
    using levelType =
        same_vector_size_different_type_t<SrcT,
                                          typename hist_even_level_types_scalar_for<remove_vector_t<SrcT>>::levelType>;
};

template <typename T> using hist_even_level_types_for_t = typename hist_even_level_types_for<T>::levelType;
#pragma endregion

#pragma region HistRange
// level types for histogram range:
template <typename SrcT> struct hist_range_types_scalar_for
{
    using levelType = int;
};
template <> struct hist_range_types_scalar_for<uint>
{
    using levelType = ulong64;
};
template <> struct hist_range_types_scalar_for<int>
{
    using levelType = long64;
};

template <> struct hist_range_types_scalar_for<HalfFp16>
{
    using levelType = float;
};
template <> struct hist_range_types_scalar_for<BFloat16>
{
    using levelType = float;
};

template <> struct hist_range_types_scalar_for<float>
{
    using levelType = float;
};
template <> struct hist_range_types_scalar_for<double>
{
    using levelType = double;
};

// compute and result types for histogram range:
template <typename SrcT> struct hist_range_types_for
{
    using levelType = typename hist_range_types_scalar_for<remove_vector_t<SrcT>>::levelType;
};

template <typename T> using hist_range_types_for_t = typename hist_range_types_for<T>::levelType;
#pragma endregion

#pragma region SSIM
#pragma endregion

#pragma region SSIM
#pragma endregion

} // namespace mpp::image::cuda