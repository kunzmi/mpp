#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
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
/// runs aFunctor reduction on every image line, single output value.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <typename SrcT, typename DstT>
__global__ void reductionAlongYKernel(const SrcT *__restrict__ aSrc, DstT *__restrict__ aDst, int aSize)
{
    int warpLaneID = threadIdx.x;
    int batchId    = threadIdx.y;

    DstT result(0);

    __shared__ DstT buffer[32][32];

    buffer[batchId][warpLaneID] = DstT(0);

    for (int pixelYWarp0 = 0; pixelYWarp0 < aSize; pixelYWarp0 += warpSize * blockDim.y)
    {
        const int pixelY = pixelYWarp0 + warpLaneID + batchId * warpSize;
        if (pixelY < aSize)
        {
            buffer[batchId][warpLaneID] += aSrc[pixelY];
        }
    }

    __syncthreads();
    if (batchId == 0)
    {
#pragma unroll
        for (int i = 1; i < 32; i++)
        {
            buffer[0][warpLaneID] += buffer[i][warpLaneID];
        }

        SrcT threadValue = buffer[0][warpLaneID];
        // reduce over warp:
        threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 16);
        threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 8);
        threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 4);
        threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 2);
        threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 1);

        if constexpr (vector_active_size_v<DstT> > 1)
        {
            threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 16);
            threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 8);
            threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 4);
            threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 2);
            threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 1);
        }
        if constexpr (vector_active_size_v<DstT> > 2)
        {
            threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 16);
            threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 8);
            threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 4);
            threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 2);
            threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 1);
        }
        if constexpr (vector_active_size_v<DstT> > 3)
        {
            threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 16);
            threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 8);
            threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 4);
            threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 2);
            threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 1);
        }

        result = threadValue;
    }

    // for (int pixelYWarp0 = 0; pixelYWarp0 < aSize; pixelYWarp0 += warpSize * blockDim.y)
    //{
    //     SrcT threadValue(0);
    //     const int pixelY = pixelYWarp0 + warpLaneID + batchId * warpSize;
    //     if (pixelY < aSize)
    //     {
    //         threadValue += aSrc[pixelY];
    //     }
    //
    //    // reduce over warp:
    //    threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 16);
    //    threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 8);
    //    threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 4);
    //    threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 2);
    //    threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 1);
    //
    //    if constexpr (vector_active_size_v<DstT> > 1)
    //    {
    //        threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 16);
    //        threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 8);
    //        threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 4);
    //        threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 2);
    //        threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 1);
    //    }
    //    if constexpr (vector_active_size_v<DstT> > 2)
    //    {
    //        threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 16);
    //        threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 8);
    //        threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 4);
    //        threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 2);
    //        threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 1);
    //    }
    //    if constexpr (vector_active_size_v<DstT> > 3)
    //    {
    //        threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 16);
    //        threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 8);
    //        threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 4);
    //        threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 2);
    //        threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 1);
    //    }
    //
    //    result += threadValue;
    //}
    // buffer[batchId] = result;

    //__syncthreads();

    if (warpLaneID == 0 && batchId == 0)
    {
        /*result += buffer[1];
        result += buffer[2];
        result += buffer[3];
        result += buffer[4];
        result += buffer[5];
        result += buffer[6];
        result += buffer[7];*/
        *aDst = result;
    }
}

template <typename SrcT, typename DstT>
void InvokeReductionAlongYKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                 const SrcT *aSrc, DstT *aDst, int aSize)
{
    dim3 blocksPerGrid(1, 1, 1);

    reductionAlongYKernel<SrcT, DstT><<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrc, aDst, aSize);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename SrcT, typename DstT>
void InvokeReductionAlongYKernelDefault(const SrcT *aSrc, DstT *aDst, int aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        // const dim3 BlockSize               = ConfigBlockSize<"Default">::value;
        const dim3 BlockSize{32, 32, 1};
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeReductionAlongYKernel<SrcT, DstT>(BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrc,
                                                aDst, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(reductionAlongYKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND