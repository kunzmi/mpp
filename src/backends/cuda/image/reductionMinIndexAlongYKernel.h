#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/channel.h>
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
/// runs Min with index reduction on one image column, single output value.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <typename SrcT>
__global__ void reductionMinIdxAlongYKernel(
    const SrcT *__restrict__ aSrcMin, const same_vector_size_different_type_t<SrcT, int> *__restrict__ aSrcMinIdxX,
    SrcT *__restrict__ aDstMin, same_vector_size_different_type_t<SrcT, int> *__restrict__ aDstMinIdxX,
    same_vector_size_different_type_t<SrcT, int> *__restrict__ aDstMinIdxY,
    remove_vector_t<SrcT> *__restrict__ aDstScalarMin, Vector3<int> *__restrict__ aDstScalarIdxMin, int aSize)
{
    using idxT = same_vector_size_different_type_t<SrcT, int>;

    int warpLaneId = threadIdx.x;
    int batchId    = threadIdx.y;

    opp::MinIdx<SrcT> redOpMin;

    SrcT resultMin(reduction_init_value_v<ReductionInitValue::Max, SrcT>);
    idxT resultMinIdx(-1);

    __shared__ SrcT bufferMinVal[ConfigBlockSize<"DefaultReductionY">::value.y];
    __shared__ idxT bufferMinIdx[ConfigBlockSize<"DefaultReductionY">::value.y];

    // process 1D input array in threadBlock-junks
    for (int pixelYWarp0 = 0; pixelYWarp0 < aSize; pixelYWarp0 += warpSize * blockDim.y)
    {
        const int pixelY = pixelYWarp0 + warpLaneId + batchId * warpSize;
        if (pixelY < aSize)
        {
            SrcT pxMin = aSrcMin[pixelY];

            redOpMin(pxMin.x, pixelY, resultMin.x, resultMinIdx.x);
            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                redOpMin(pxMin.y, pixelY, resultMin.y, resultMinIdx.y);
            }
            if constexpr (vector_active_size_v<SrcT> > 2)
            {
                redOpMin(pxMin.z, pixelY, resultMin.z, resultMinIdx.z);
            }
            if constexpr (vector_active_size_v<SrcT> > 3)
            {
                redOpMin(pxMin.w, pixelY, resultMin.w, resultMinIdx.w);
            }
        }
    }

    // reduce over warps:
    redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.x, 16), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.x, 16),
             resultMin.x, resultMinIdx.x);
    redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.x, 8), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.x, 8), resultMin.x,
             resultMinIdx.x);
    redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.x, 4), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.x, 4), resultMin.x,
             resultMinIdx.x);
    redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.x, 2), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.x, 2), resultMin.x,
             resultMinIdx.x);
    redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.x, 1), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.x, 1), resultMin.x,
             resultMinIdx.x);

    if constexpr (vector_active_size_v<SrcT> > 1)
    {
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.y, 16), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.y, 16),
                 resultMin.y, resultMinIdx.y);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.y, 8), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.y, 8),
                 resultMin.y, resultMinIdx.y);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.y, 4), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.y, 4),
                 resultMin.y, resultMinIdx.y);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.y, 2), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.y, 2),
                 resultMin.y, resultMinIdx.y);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.y, 1), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.y, 1),
                 resultMin.y, resultMinIdx.y);
    }
    if constexpr (vector_active_size_v<SrcT> > 2)
    {
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.z, 16), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.z, 16),
                 resultMin.z, resultMinIdx.z);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.z, 8), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.z, 8),
                 resultMin.z, resultMinIdx.z);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.z, 4), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.z, 4),
                 resultMin.z, resultMinIdx.z);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.z, 2), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.z, 2),
                 resultMin.z, resultMinIdx.z);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.z, 1), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.z, 1),
                 resultMin.z, resultMinIdx.z);
    }
    if constexpr (vector_active_size_v<SrcT> > 3)
    {
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.w, 16), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.w, 16),
                 resultMin.w, resultMinIdx.w);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.w, 8), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.w, 8),
                 resultMin.w, resultMinIdx.w);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.w, 4), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.w, 4),
                 resultMin.w, resultMinIdx.w);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.w, 2), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.w, 2),
                 resultMin.w, resultMinIdx.w);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.w, 1), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.w, 1),
                 resultMin.w, resultMinIdx.w);
    }

    // write intermediate threadBlock sums to shared memory for result 1:
    bufferMinVal[batchId] = resultMin;
    bufferMinIdx[batchId] = resultMinIdx;

    __syncthreads();

    if (warpLaneId == 0 && batchId == 0)
    {
        // reduce over Y the entire thread block:
        for (int i = 1; i < blockDim.y; i++) // i = 0 is already stored in result
        {
            redOpMin(bufferMinVal[i].x, bufferMinIdx[i].x, resultMin.x, resultMinIdx.x);
            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                redOpMin(bufferMinVal[i].y, bufferMinIdx[i].y, resultMin.y, resultMinIdx.y);
            }
            if constexpr (vector_active_size_v<SrcT> > 2)
            {
                redOpMin(bufferMinVal[i].z, bufferMinIdx[i].z, resultMin.z, resultMinIdx.z);
            }
            if constexpr (vector_active_size_v<SrcT> > 3)
            {
                redOpMin(bufferMinVal[i].w, bufferMinIdx[i].w, resultMin.w, resultMinIdx.w);
            }
        }

        // fetch X coordinates:
        idxT minIdxX;
        minIdxX.x = aSrcMinIdxX[resultMinIdx.x].x;
        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            minIdxX.y = aSrcMinIdxX[resultMinIdx.y].y;
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            minIdxX.z = aSrcMinIdxX[resultMinIdx.z].z;
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            minIdxX.w = aSrcMinIdxX[resultMinIdx.w].w;
        }

        if (aDstScalarMin != nullptr && aDstScalarIdxMin != nullptr)
        {
            int minIdxVec                   = 0;
            remove_vector_t<SrcT> minScalar = resultMin.x;

            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                redOpMin(resultMin.y, 1, minScalar, minIdxVec);
            }
            if constexpr (vector_active_size_v<SrcT> > 2)
            {
                redOpMin(resultMin.z, 2, minScalar, minIdxVec);
            }
            if constexpr (vector_active_size_v<SrcT> > 3)
            {
                redOpMin(resultMin.w, 3, minScalar, minIdxVec);
            }

            Vector3<int> minIdx(minIdxX[Channel(minIdxVec)], resultMinIdx[Channel(minIdxVec)], minIdxVec);

            *aDstScalarMin = minScalar;

            *aDstScalarIdxMin = minIdx;
        }

        if (aDstMin != nullptr && aDstMinIdxX != nullptr && aDstMinIdxY != nullptr)
        {
            // don't overwrite alpha channel if it exists:
            if constexpr (has_alpha_channel_v<SrcT>)
            {
                Vector3<remove_vector_t<SrcT>> *dstVec3 = reinterpret_cast<Vector3<remove_vector_t<SrcT>> *>(aDstMin);
                *dstVec3                                = resultMin.XYZ();
                Vector3<int> *dstMinIdxXVec3            = reinterpret_cast<Vector3<int> *>(aDstMinIdxX);
                *dstMinIdxXVec3                         = minIdxX.XYZ();
                Vector3<int> *dstMinIdxYVec3            = reinterpret_cast<Vector3<int> *>(aDstMinIdxY);
                *dstMinIdxYVec3                         = resultMinIdx.XYZ();
            }
            else
            {
                *aDstMin     = resultMin;
                *aDstMinIdxX = minIdxX;
                *aDstMinIdxY = resultMinIdx;
            }
        }
    }
}

template <typename SrcT>
void InvokeReductionMinIdxAlongYKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                       const SrcT *aSrcMin,
                                       const same_vector_size_different_type_t<SrcT, int> *aSrcMinIdxX, SrcT *aDstMin,
                                       same_vector_size_different_type_t<SrcT, int> *aDstMinIdxX,
                                       same_vector_size_different_type_t<SrcT, int> *aDstMinIdxY,
                                       remove_vector_t<SrcT> *aDstScalarMin, Vector3<int> *aDstScalarIdxMin, int aSize)
{
    dim3 blocksPerGrid(1, 1, 1);

    const int size = DIV_UP(aSize, ConfigBlockSize<"DefaultReductionX">::value.y);

    reductionMinIdxAlongYKernel<SrcT><<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(
        aSrcMin, aSrcMinIdxX, aDstMin, aDstMinIdxX, aDstMinIdxY, aDstScalarMin, aDstScalarIdxMin, aSize);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename SrcT>
void InvokeReductionMinIdxAlongYKernelDefault(const SrcT *aSrcMin,
                                              const same_vector_size_different_type_t<SrcT, int> *aSrcMinIdxX,
                                              SrcT *aDstMin, same_vector_size_different_type_t<SrcT, int> *aDstMinIdxX,
                                              same_vector_size_different_type_t<SrcT, int> *aDstMinIdxY,
                                              remove_vector_t<SrcT> *aDstScalarMin, Vector3<int> *aDstScalarIdxMin,
                                              int aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize = ConfigBlockSize<"DefaultReductionY">::value;

        const uint SharedMemory = 0;

        InvokeReductionMinIdxAlongYKernel<SrcT>(BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream,
                                                aSrcMin, aSrcMinIdxX, aDstMin, aDstMinIdxX, aDstMinIdxY, aDstScalarMin,
                                                aDstScalarIdxMin, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(reductionMinIdxAlongYKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND