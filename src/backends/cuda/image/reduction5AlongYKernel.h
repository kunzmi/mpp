#pragma once
#include "reduction5AlongXKernel.h" // for DuShuffle function
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

namespace mpp::image::cuda
{
/// <summary>
/// runs aFunctor reduction on one image column, single output value.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <typename SrcT1, typename SrcT2, typename SrcT3, typename SrcT4, typename SrcT5, typename ComputeT1,
          typename ComputeT2, typename ComputeT3, typename ComputeT4, typename ComputeT5, typename DstT,
          typename reductionOp1, typename reductionOp2, typename reductionOp3, typename reductionOp4,
          typename reductionOp5, ReductionInitValue NeutralValue1, ReductionInitValue NeutralValue2,
          ReductionInitValue NeutralValue3, ReductionInitValue NeutralValue4, ReductionInitValue NeutralValue5,
          typename postOp>
__global__ void reduction5AlongYKernel(const SrcT1 *__restrict__ aSrc1, const SrcT2 *__restrict__ aSrc2,
                                       const SrcT3 *__restrict__ aSrc3, const SrcT4 *__restrict__ aSrc4,
                                       const SrcT5 *__restrict__ aSrc5, DstT *__restrict__ aDst, int aSize,
                                       postOp aPostOp)
{
    int warpLaneId = threadIdx.x;
    int batchId    = threadIdx.y;

    reductionOp1 redOp1;
    reductionOp2 redOp2;
    reductionOp3 redOp3;
    reductionOp4 redOp4;
    reductionOp5 redOp5;
    ComputeT1 result1 = reduction_init_value_v<NeutralValue1, ComputeT1>;
    ComputeT2 result2 = reduction_init_value_v<NeutralValue2, ComputeT2>;
    ComputeT3 result3 = reduction_init_value_v<NeutralValue3, ComputeT3>;
    ComputeT4 result4 = reduction_init_value_v<NeutralValue4, ComputeT4>;
    ComputeT5 result5 = reduction_init_value_v<NeutralValue5, ComputeT5>;

    extern __shared__ int sharedBuffer[];
    // Block dimension in X are the same for large and small configuration!
    ComputeT1(*buffer1)[ConfigBlockSize<"DefaultReductionY">::value.x] =
        (ComputeT1(*)[ConfigBlockSize<"DefaultReductionY">::value.x])(sharedBuffer);
    ComputeT2(*buffer2)[ConfigBlockSize<"DefaultReductionY">::value.x] =
        (ComputeT2(*)[ConfigBlockSize<"DefaultReductionY">::value.x])(sharedBuffer);
    ComputeT3(*buffer3)[ConfigBlockSize<"DefaultReductionY">::value.x] =
        (ComputeT3(*)[ConfigBlockSize<"DefaultReductionY">::value.x])(sharedBuffer);
    ComputeT4(*buffer4)[ConfigBlockSize<"DefaultReductionY">::value.x] =
        (ComputeT4(*)[ConfigBlockSize<"DefaultReductionY">::value.x])(sharedBuffer);
    ComputeT5(*buffer5)[ConfigBlockSize<"DefaultReductionY">::value.x] =
        (ComputeT5(*)[ConfigBlockSize<"DefaultReductionY">::value.x])(sharedBuffer);

    // process 1D input array in threadBlock-junks
    for (int pixelYWarp0 = 0; pixelYWarp0 < aSize; pixelYWarp0 += warpSize * blockDim.y)
    {
        const int pixelY = pixelYWarp0 + warpLaneId + batchId * warpSize;
        if (pixelY < aSize)
        {
            redOp1(aSrc1[pixelY], result1);
            redOp2(aSrc2[pixelY], result2);
            redOp3(aSrc3[pixelY], result3);
            redOp4(aSrc4[pixelY], result4);
            redOp5(aSrc5[pixelY], result5);
        }
    }

    // write intermediate threadBlock sums to shared memory for result 1:
    buffer1[batchId][warpLaneId] = result1;

    __syncthreads();

    if (batchId == 0)
    {
        // reduce over Y the entire thread block:
#pragma unroll
        for (int i = 1; i < blockDim.y; i++) // i = 0 is already stored in result
        {
            redOp1(buffer1[i][warpLaneId], result1);
        }

        // reduce over warp:
        DoShuffle(result1, redOp1);
    }

    __syncthreads();

    // write intermediate threadBlock sums to shared memory for result 2:
    buffer2[batchId][warpLaneId] = result2;

    __syncthreads();

    if (batchId == 0)
    {
        // reduce over Y the entire thread block:
#pragma unroll
        for (int i = 1; i < blockDim.y; i++) // i = 0 is already stored in result
        {
            redOp2(buffer2[i][warpLaneId], result2);
        }

        // reduce over warp:
        DoShuffle(result2, redOp2);
    }

    __syncthreads();

    // write intermediate threadBlock sums to shared memory for result 3:
    buffer3[batchId][warpLaneId] = result3;

    __syncthreads();

    if (batchId == 0)
    {
        // reduce over Y the entire thread block:
#pragma unroll
        for (int i = 1; i < blockDim.y; i++) // i = 0 is already stored in result
        {
            redOp3(buffer3[i][warpLaneId], result3);
        }

        // reduce over warp:
        DoShuffle(result3, redOp3);
    }

    __syncthreads();

    // write intermediate threadBlock sums to shared memory for result 4:
    buffer4[batchId][warpLaneId] = result4;

    __syncthreads();

    if (batchId == 0)
    {
        // reduce over Y the entire thread block:
#pragma unroll
        for (int i = 1; i < blockDim.y; i++) // i = 0 is already stored in result
        {
            redOp4(buffer4[i][warpLaneId], result4);
        }

        // reduce over warp:
        DoShuffle(result4, redOp4);
    }

    __syncthreads();

    // write intermediate threadBlock sums to shared memory for result 5:
    buffer5[batchId][warpLaneId] = result5;

    __syncthreads();

    if (batchId == 0)
    {
        // reduce over Y the entire thread block:
#pragma unroll
        for (int i = 1; i < blockDim.y; i++) // i = 0 is already stored in result
        {
            redOp5(buffer5[i][warpLaneId], result5);
        }

        // reduce over warp:
        DoShuffle(result5, redOp5);
    }

    if (warpLaneId == 0 && batchId == 0)
    {
        DstT res;
        aPostOp(DstT(result1), DstT(result2), DstT(result3), DstT(result4), DstT(result5), res);

        // don't overwrite alpha channel if it exists:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            Vector3<remove_vector_t<DstT>> *dstVec3 = reinterpret_cast<Vector3<remove_vector_t<DstT>> *>(aDst);
            *dstVec3                                = res.XYZ();
        }
        else
        {
            *aDst = res;
        }
    }
}

template <typename SrcT1, typename SrcT2, typename SrcT3, typename SrcT4, typename SrcT5, typename ComputeT1,
          typename ComputeT2, typename ComputeT3, typename ComputeT4, typename ComputeT5, typename DstT,
          typename reductionOp1, typename reductionOp2, typename reductionOp3, typename reductionOp4,
          typename reductionOp5, ReductionInitValue NeutralValue1, ReductionInitValue NeutralValue2,
          ReductionInitValue NeutralValue3, ReductionInitValue NeutralValue4, ReductionInitValue NeutralValue5,
          typename postOp>
void InvokeReduction5AlongYKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                  const SrcT1 *aSrc1, const SrcT2 *aSrc2, const SrcT3 *aSrc3, const SrcT4 *aSrc4,
                                  const SrcT5 *aSrc5, DstT *aDst, int aSize, postOp aPostOp)
{
    dim3 blocksPerGrid(1, 1, 1);

    const int size = DIV_UP(aSize, ConfigBlockSize<"DefaultReductionX">::value.y);

    reduction5AlongYKernel<SrcT1, SrcT2, SrcT3, SrcT4, SrcT5, ComputeT1, ComputeT2, ComputeT3, ComputeT4, ComputeT5,
                           DstT, reductionOp1, reductionOp2, reductionOp3, reductionOp4, reductionOp5, NeutralValue1,
                           NeutralValue2, NeutralValue3, NeutralValue4, NeutralValue5, postOp>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrc1, aSrc2, aSrc3, aSrc4, aSrc5, aDst, size, aPostOp);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename SrcT1, typename SrcT2, typename SrcT3, typename SrcT4, typename SrcT5, typename ComputeT1,
          typename ComputeT2, typename ComputeT3, typename ComputeT4, typename ComputeT5, typename DstT,
          typename reductionOp1, typename reductionOp2, typename reductionOp3, typename reductionOp4,
          typename reductionOp5, ReductionInitValue NeutralValue1, ReductionInitValue NeutralValue2,
          ReductionInitValue NeutralValue3, ReductionInitValue NeutralValue4, ReductionInitValue NeutralValue5,
          typename postOp>
void InvokeReduction5AlongYKernelDefault(const SrcT1 *aSrc1, const SrcT2 *aSrc2, const SrcT3 *aSrc3, const SrcT4 *aSrc4,
                                         const SrcT5 *aSrc5, DstT *aDst, int aSize, postOp aPostOp,
                                         const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        // we define shared memory size using DstT, but we compute on ComputeTX, so make sure the sizes fit:
        static_assert(sizeof(ComputeT1) <= sizeof(DstT),
                      "expected type size of ComputeT1 smaller or equal to size of DstT");
        static_assert(sizeof(ComputeT2) <= sizeof(DstT),
                      "expected type size of ComputeT2 smaller or equal to size of DstT");
        static_assert(sizeof(ComputeT3) <= sizeof(DstT),
                      "expected type size of ComputeT3 smaller or equal to size of DstT");
        static_assert(sizeof(ComputeT4) <= sizeof(DstT),
                      "expected type size of ComputeT4 smaller or equal to size of DstT");
        static_assert(sizeof(ComputeT5) <= sizeof(DstT),
                      "expected type size of ComputeT5 smaller or equal to size of DstT");

        dim3 BlockSize    = ConfigBlockSize<"DefaultReductionYLarge">::value;
        uint SharedMemory = sizeof(DstT) * BlockSize.x * BlockSize.y * BlockSize.z;

        if (SharedMemory > aStreamCtx.SharedMemPerBlock)
        {
            // use a block config of half the size:
            BlockSize.y /= 2;
            SharedMemory = sizeof(DstT) * BlockSize.x * BlockSize.y * BlockSize.z;
        }

        InvokeReduction5AlongYKernel<SrcT1, SrcT2, SrcT3, SrcT4, SrcT5, ComputeT1, ComputeT2, ComputeT3, ComputeT4,
                                     ComputeT5, DstT, reductionOp1, reductionOp2, reductionOp3, reductionOp4,
                                     reductionOp5, NeutralValue1, NeutralValue2, NeutralValue3, NeutralValue4,
                                     NeutralValue5, postOp>(BlockSize, SharedMemory, aStreamCtx.WarpSize,
                                                            aStreamCtx.Stream, aSrc1, aSrc2, aSrc3, aSrc4, aSrc5, aDst,
                                                            aSize, aPostOp);
    }
    else
    {
        throw CUDAUNSUPPORTED(reduction5AlongYKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
