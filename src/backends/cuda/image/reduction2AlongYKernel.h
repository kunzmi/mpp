#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace opp::image::cuda
{
/// <summary>
/// runs aFunctor reduction on one image column, single output value.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <typename SrcT1, typename SrcT2, typename DstT1, typename DstT2, typename reductionOp1, typename reductionOp2,
          ReductionInitValue NeutralValue1, ReductionInitValue NeutralValue2, typename postOp1, typename postOp2,
          typename postOpScalar1, typename postOpScalar2>
__global__ void reduction2AlongYKernel(const SrcT1 *__restrict__ aSrc1, const SrcT2 *__restrict__ aSrc2,
                                       DstT1 *__restrict__ aDst1, DstT2 *__restrict__ aDst2,
                                       remove_vector_t<DstT1> *__restrict__ aDstScalar1,
                                       remove_vector_t<DstT2> *__restrict__ aDstScalar2, int aSize, postOp1 aPostOp1,
                                       postOp2 aPostOp2, postOpScalar1 aPostOpScalar1, postOpScalar2 aPostOpScalar2)
{
    int warpLaneId = threadIdx.x;
    int batchId    = threadIdx.y;

    reductionOp1 redOp1;
    reductionOp2 redOp2;
    DstT1 result1 = reduction_init_value_v<NeutralValue1, DstT1>;
    // currently we only use MeanStd with two different output types but until postOp,
    // DstT2 is still the same as DstT1, only for complex types the postOp diverges
    DstT1 result2 = reduction_init_value_v<NeutralValue2, DstT1>;

    extern __shared__ int sharedBuffer[];
    // Block dimension in X are the same for large and small configuration!
    DstT1(*buffer)[ConfigBlockSize<"DefaultReductionY">::value.x] =
        (DstT1(*)[ConfigBlockSize<"DefaultReductionY">::value.x])(sharedBuffer);

    // process 1D input array in threadBlock-junks
    for (int pixelYWarp0 = 0; pixelYWarp0 < aSize; pixelYWarp0 += warpSize * blockDim.y)
    {
        const int pixelY = pixelYWarp0 + warpLaneId + batchId * warpSize;
        if (pixelY < aSize)
        {
            redOp1(aSrc1[pixelY], result1);
            redOp2(aSrc2[pixelY], result2);
        }
    }

    // write intermediate threadBlock sums to shared memory for result 1:
    buffer[batchId][warpLaneId] = result1;

    __syncthreads();

    if (batchId == 0)
    {
        // reduce over Y the entire thread block:
#pragma unroll
        for (int i = 1; i < ConfigBlockSize<"DefaultReductionY">::value.y; i++) // i = 0 is already stored in result
        {
            redOp1(buffer[i][warpLaneId], result1);
        }

        // reduce over warp:
        if constexpr (ComplexVector<DstT1>)
        {
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.real, 16), result1.x.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.real, 8), result1.x.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.real, 4), result1.x.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.real, 2), result1.x.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.real, 1), result1.x.real);

            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.imag, 16), result1.x.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.imag, 8), result1.x.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.imag, 4), result1.x.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.imag, 2), result1.x.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.imag, 1), result1.x.imag);
        }
        else
        {
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x, 16), result1.x);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x, 8), result1.x);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x, 4), result1.x);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x, 2), result1.x);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x, 1), result1.x);
        }

        if constexpr (vector_active_size_v<DstT1> > 1)
        {
            if constexpr (ComplexVector<DstT1>)
            {
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.real, 16), result1.y.real);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.real, 8), result1.y.real);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.real, 4), result1.y.real);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.real, 2), result1.y.real);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.real, 1), result1.y.real);

                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.imag, 16), result1.y.imag);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.imag, 8), result1.y.imag);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.imag, 4), result1.y.imag);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.imag, 2), result1.y.imag);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.imag, 1), result1.y.imag);
            }
            else
            {
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y, 16), result1.y);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y, 8), result1.y);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y, 4), result1.y);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y, 2), result1.y);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y, 1), result1.y);
            }
        }
        if constexpr (vector_active_size_v<DstT1> > 2)
        {
            if constexpr (ComplexVector<DstT1>)
            {
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.real, 16), result1.z.real);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.real, 8), result1.z.real);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.real, 4), result1.z.real);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.real, 2), result1.z.real);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.real, 1), result1.z.real);

                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.imag, 16), result1.z.imag);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.imag, 8), result1.z.imag);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.imag, 4), result1.z.imag);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.imag, 2), result1.z.imag);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.imag, 1), result1.z.imag);
            }
            else
            {
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z, 16), result1.z);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z, 8), result1.z);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z, 4), result1.z);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z, 2), result1.z);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z, 1), result1.z);
            }
        }
        if constexpr (vector_active_size_v<DstT1> > 3)
        {
            if constexpr (ComplexVector<DstT1>)
            {
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.real, 16), result1.w.real);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.real, 8), result1.w.real);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.real, 4), result1.w.real);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.real, 2), result1.w.real);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.real, 1), result1.w.real);

                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.imag, 16), result1.w.imag);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.imag, 8), result1.w.imag);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.imag, 4), result1.w.imag);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.imag, 2), result1.w.imag);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.imag, 1), result1.w.imag);
            }
            else
            {
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w, 16), result1.w);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w, 8), result1.w);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w, 4), result1.w);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w, 2), result1.w);
                redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w, 1), result1.w);
            }
        }
    }

    __syncthreads();
    // write intermediate threadBlock sums to shared memory for result 2:
    buffer[batchId][warpLaneId] = result2;

    __syncthreads();

    if (batchId == 0)
    {
        // reduce over Y the entire thread block:
#pragma unroll
        for (int i = 1; i < ConfigBlockSize<"DefaultReductionY">::value.y; i++) // i = 0 is already stored in result
        {
            redOp2(buffer[i][warpLaneId], result2);
        }

        // reduce over warp:
        if constexpr (ComplexVector<DstT1>)
        {
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.real, 16), result2.x.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.real, 8), result2.x.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.real, 4), result2.x.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.real, 2), result2.x.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.real, 1), result2.x.real);

            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.imag, 16), result2.x.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.imag, 8), result2.x.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.imag, 4), result2.x.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.imag, 2), result2.x.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.imag, 1), result2.x.imag);
        }
        else
        {
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x, 16), result2.x);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x, 8), result2.x);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x, 4), result2.x);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x, 2), result2.x);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x, 1), result2.x);
        }

        if constexpr (vector_active_size_v<DstT2> > 1)
        {
            if constexpr (ComplexVector<DstT1>)
            {
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.real, 16), result2.y.real);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.real, 8), result2.y.real);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.real, 4), result2.y.real);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.real, 2), result2.y.real);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.real, 1), result2.y.real);

                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.imag, 16), result2.y.imag);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.imag, 8), result2.y.imag);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.imag, 4), result2.y.imag);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.imag, 2), result2.y.imag);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.imag, 1), result2.y.imag);
            }
            else
            {
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y, 16), result2.y);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y, 8), result2.y);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y, 4), result2.y);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y, 2), result2.y);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y, 1), result2.y);
            }
        }
        if constexpr (vector_active_size_v<DstT2> > 2)
        {
            if constexpr (ComplexVector<DstT1>)
            {
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.real, 16), result2.z.real);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.real, 8), result2.z.real);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.real, 4), result2.z.real);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.real, 2), result2.z.real);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.real, 1), result2.z.real);

                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.imag, 16), result2.z.imag);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.imag, 8), result2.z.imag);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.imag, 4), result2.z.imag);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.imag, 2), result2.z.imag);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.imag, 1), result2.z.imag);
            }
            else
            {
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z, 16), result2.z);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z, 8), result2.z);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z, 4), result2.z);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z, 2), result2.z);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z, 1), result2.z);
            }
        }
        if constexpr (vector_active_size_v<DstT2> > 3)
        {
            if constexpr (ComplexVector<DstT1>)
            {
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.real, 16), result2.w.real);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.real, 8), result2.w.real);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.real, 4), result2.w.real);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.real, 2), result2.w.real);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.real, 1), result2.w.real);

                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.imag, 16), result2.w.imag);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.imag, 8), result2.w.imag);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.imag, 4), result2.w.imag);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.imag, 2), result2.w.imag);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.imag, 1), result2.w.imag);
            }
            else
            {
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w, 16), result2.w);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w, 8), result2.w);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w, 4), result2.w);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w, 2), result2.w);
                redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w, 1), result2.w);
            }
        }
    }
    if (warpLaneId == 0 && batchId == 0)
    {
        if (aDstScalar2 != nullptr && aDstScalar1 != nullptr)
        {
            remove_vector_t<DstT2> res2Scalar;
            aPostOpScalar2(result1, result2, res2Scalar);
            *aDstScalar2 = res2Scalar;

            *aDstScalar1 = aPostOpScalar1(result1);
        }

        if (aDst2 != nullptr && aDst1 != nullptr)
        {
            DstT2 res2;
            aPostOp2(result1, result2, res2);
            aPostOp1(result1);

            // don't overwrite alpha channel if it exists:
            if constexpr (has_alpha_channel_v<DstT1>)
            {
                Vector3<remove_vector_t<DstT1>> *dstVec3_1 = reinterpret_cast<Vector3<remove_vector_t<DstT1>> *>(aDst1);
                *dstVec3_1                                 = result1.XYZ();

                Vector3<remove_vector_t<DstT2>> *dstVec3_2 = reinterpret_cast<Vector3<remove_vector_t<DstT2>> *>(aDst2);
                *dstVec3_2                                 = res2.XYZ();
            }
            else
            {
                *aDst1 = result1;
                *aDst2 = res2;
            }
        }
    }
}

template <typename SrcT1, typename SrcT2, typename DstT1, typename DstT2, typename reductionOp1, typename reductionOp2,
          ReductionInitValue NeutralValue1, ReductionInitValue NeutralValue2, typename postOp1, typename postOp2,
          typename postOpScalar1, typename postOpScalar2>
void InvokeReduction2AlongYKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                  const SrcT1 *aSrc1, const SrcT2 *aSrc2, DstT1 *aDst1, DstT2 *aDst2,
                                  remove_vector_t<DstT1> *aDstScalar1, remove_vector_t<DstT2> *aDstScalar2, int aSize,
                                  postOp1 aPostOp1, postOp2 aPostOp2, postOpScalar1 aPostOpScalar1,
                                  postOpScalar2 aPostOpScalar2)
{
    dim3 blocksPerGrid(1, 1, 1);

    const int size = DIV_UP(aSize, ConfigBlockSize<"DefaultReductionX">::value.y);

    reduction2AlongYKernel<SrcT1, SrcT2, DstT1, DstT2, reductionOp1, reductionOp2, NeutralValue1, NeutralValue2,
                           postOp1, postOp2><<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(
        aSrc1, aSrc2, aDst1, aDst2, aDstScalar1, aDstScalar2, size, aPostOp1, aPostOp2, aPostOpScalar1, aPostOpScalar2);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename SrcT1, typename SrcT2, typename DstT1, typename DstT2, typename reductionOp1, typename reductionOp2,
          ReductionInitValue NeutralValue1, ReductionInitValue NeutralValue2, typename postOp1, typename postOp2,
          typename postOpScalar1, typename postOpScalar2>
void InvokeReduction2AlongYKernelDefault(const SrcT1 *aSrc1, const SrcT2 *aSrc2, DstT1 *aDst1, DstT2 *aDst2,
                                         remove_vector_t<DstT1> *aDstScalar1, remove_vector_t<DstT2> *aDstScalar2,
                                         int aSize, postOp1 aPostOp1, postOp2 aPostOp2, postOpScalar1 aPostOpScalar1,
                                         postOpScalar2 aPostOpScalar2, const opp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        dim3 BlockSize    = ConfigBlockSize<"DefaultReductionY">::value;
        uint SharedMemory = sizeof(DstT1) * BlockSize.x * BlockSize.y * BlockSize.z;

        if (SharedMemory > aStreamCtx.SharedMemPerBlock)
        {
            // use a block config of half the size:
            BlockSize    = ConfigBlockSize<"DefaultReductionYLarge">::value;
            SharedMemory = sizeof(DstT1) * BlockSize.x * BlockSize.y * BlockSize.z;
        }

        InvokeReduction2AlongYKernel<SrcT1, SrcT2, DstT1, DstT2, reductionOp1, reductionOp2, NeutralValue1,
                                     NeutralValue2, postOp1, postOp2>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrc1, aSrc2, aDst1, aDst2, aDstScalar1,
            aDstScalar2, aSize, aPostOp1, aPostOp2, aPostOpScalar1, aPostOpScalar2);
    }
    else
    {
        throw CUDAUNSUPPORTED(reduction2AlongYKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND