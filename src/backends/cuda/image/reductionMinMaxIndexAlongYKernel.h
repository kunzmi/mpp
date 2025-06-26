#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/channel.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/statistics/indexMinMax.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{
/// <summary>
/// runs MinMax with index reduction on one image column, single output value.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <typename SrcT>
__global__ void reductionMinMaxIdxAlongYKernel(
    const SrcT *__restrict__ aSrcMin, const SrcT *__restrict__ aSrcMax,
    const same_vector_size_different_type_t<SrcT, int> *__restrict__ aSrcMinIdxX,
    const same_vector_size_different_type_t<SrcT, int> *__restrict__ aSrcMaxIdxX, SrcT *__restrict__ aDstMin,
    SrcT *__restrict__ aDstMax, IndexMinMax *__restrict__ aDstIdx, remove_vector_t<SrcT> *__restrict__ aDstScalarMin,
    remove_vector_t<SrcT> *__restrict__ aDstScalarMax, IndexMinMaxChannel *__restrict__ aDstScalarIdx, int aSize)
{
    using idxT = same_vector_size_different_type_t<SrcT, int>;

    int warpLaneId = threadIdx.x;
    int batchId    = threadIdx.y;

    mpp::MinIdx<SrcT> redOpMin;
    mpp::MaxIdx<SrcT> redOpMax;

    SrcT resultMin(reduction_init_value_v<ReductionInitValue::Max, SrcT>);
    SrcT resultMax(reduction_init_value_v<ReductionInitValue::Min, SrcT>);
    idxT resultMinIdx(INT_MAX);
    idxT resultMaxIdx(INT_MAX);

    __shared__ SrcT bufferMinVal[ConfigBlockSize<"DefaultReductionY">::value.y];
    __shared__ idxT bufferMinIdx[ConfigBlockSize<"DefaultReductionY">::value.y];
    __shared__ SrcT bufferMaxVal[ConfigBlockSize<"DefaultReductionY">::value.y];
    __shared__ idxT bufferMaxIdx[ConfigBlockSize<"DefaultReductionY">::value.y];

    // process 1D input array in threadBlock-junks
    for (int pixelYWarp0 = 0; pixelYWarp0 < aSize; pixelYWarp0 += warpSize * blockDim.y)
    {
        const int pixelY = pixelYWarp0 + warpLaneId + batchId * warpSize;
        if (pixelY < aSize)
        {
            SrcT pxMin = aSrcMin[pixelY];
            SrcT pxMax = aSrcMax[pixelY];

            redOpMin(pxMin.x, pixelY, resultMin.x, resultMinIdx.x);
            redOpMax(pxMax.x, pixelY, resultMax.x, resultMaxIdx.x);
            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                redOpMin(pxMin.y, pixelY, resultMin.y, resultMinIdx.y);
                redOpMax(pxMax.y, pixelY, resultMax.y, resultMaxIdx.y);
            }
            if constexpr (vector_active_size_v<SrcT> > 2)
            {
                redOpMin(pxMin.z, pixelY, resultMin.z, resultMinIdx.z);
                redOpMax(pxMax.z, pixelY, resultMax.z, resultMaxIdx.z);
            }
            if constexpr (vector_active_size_v<SrcT> > 3)
            {
                redOpMin(pxMin.w, pixelY, resultMin.w, resultMinIdx.w);
                redOpMax(pxMax.w, pixelY, resultMax.w, resultMaxIdx.w);
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

    redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.x, 16), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.x, 16),
             resultMax.x, resultMaxIdx.x);
    redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.x, 8), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.x, 8), resultMax.x,
             resultMaxIdx.x);
    redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.x, 4), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.x, 4), resultMax.x,
             resultMaxIdx.x);
    redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.x, 2), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.x, 2), resultMax.x,
             resultMaxIdx.x);
    redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.x, 1), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.x, 1), resultMax.x,
             resultMaxIdx.x);

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

        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.y, 16), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.y, 16),
                 resultMax.y, resultMaxIdx.y);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.y, 8), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.y, 8),
                 resultMax.y, resultMaxIdx.y);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.y, 4), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.y, 4),
                 resultMax.y, resultMaxIdx.y);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.y, 2), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.y, 2),
                 resultMax.y, resultMaxIdx.y);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.y, 1), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.y, 1),
                 resultMax.y, resultMaxIdx.y);
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

        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.z, 16), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.z, 16),
                 resultMax.z, resultMaxIdx.z);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.z, 8), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.z, 8),
                 resultMax.z, resultMaxIdx.z);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.z, 4), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.z, 4),
                 resultMax.z, resultMaxIdx.z);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.z, 2), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.z, 2),
                 resultMax.z, resultMaxIdx.z);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.z, 1), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.z, 1),
                 resultMax.z, resultMaxIdx.z);
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

        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.w, 16), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.w, 16),
                 resultMax.w, resultMaxIdx.w);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.w, 8), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.w, 8),
                 resultMax.w, resultMaxIdx.w);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.w, 4), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.w, 4),
                 resultMax.w, resultMaxIdx.w);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.w, 2), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.w, 2),
                 resultMax.w, resultMaxIdx.w);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.w, 1), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.w, 1),
                 resultMax.w, resultMaxIdx.w);
    }

    // write intermediate threadBlock sums to shared memory for result 1:
    bufferMinVal[batchId] = resultMin;
    bufferMinIdx[batchId] = resultMinIdx;
    bufferMaxVal[batchId] = resultMax;
    bufferMaxIdx[batchId] = resultMaxIdx;

    __syncthreads();

    if (warpLaneId == 0 && batchId == 0)
    {
        // reduce over Y the entire thread block:
        for (int i = 1; i < blockDim.y; i++) // i = 0 is already stored in result
        {
            redOpMin(bufferMinVal[i].x, bufferMinIdx[i].x, resultMin.x, resultMinIdx.x);
            redOpMax(bufferMaxVal[i].x, bufferMaxIdx[i].x, resultMax.x, resultMaxIdx.x);
            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                redOpMin(bufferMinVal[i].y, bufferMinIdx[i].y, resultMin.y, resultMinIdx.y);
                redOpMax(bufferMaxVal[i].y, bufferMaxIdx[i].y, resultMax.y, resultMaxIdx.y);
            }
            if constexpr (vector_active_size_v<SrcT> > 2)
            {
                redOpMin(bufferMinVal[i].z, bufferMinIdx[i].z, resultMin.z, resultMinIdx.z);
                redOpMax(bufferMaxVal[i].z, bufferMaxIdx[i].z, resultMax.z, resultMaxIdx.z);
            }
            if constexpr (vector_active_size_v<SrcT> > 3)
            {
                redOpMin(bufferMinVal[i].w, bufferMinIdx[i].w, resultMin.w, resultMinIdx.w);
                redOpMax(bufferMaxVal[i].w, bufferMaxIdx[i].w, resultMax.w, resultMaxIdx.w);
            }
        }

        // fetch X coordinates:
        idxT minIdxX;
        idxT maxIdxX;
        minIdxX.x = aSrcMinIdxX[resultMinIdx.x].x;
        maxIdxX.x = aSrcMaxIdxX[resultMaxIdx.x].x;
        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            minIdxX.y = aSrcMinIdxX[resultMinIdx.y].y;
            maxIdxX.y = aSrcMaxIdxX[resultMaxIdx.y].y;
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            minIdxX.z = aSrcMinIdxX[resultMinIdx.z].z;
            maxIdxX.z = aSrcMaxIdxX[resultMaxIdx.z].z;
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            minIdxX.w = aSrcMinIdxX[resultMinIdx.w].w;
            maxIdxX.w = aSrcMaxIdxX[resultMaxIdx.w].w;
        }

        if (aDstScalarMin != nullptr && aDstScalarMax != nullptr && aDstScalarIdx != nullptr)
        {
            int minIdxVec                   = 0;
            int maxIdxVec                   = 0;
            remove_vector_t<SrcT> minScalar = resultMin.x;
            remove_vector_t<SrcT> maxScalar = resultMax.x;

            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                redOpMin(resultMin.y, 1, minScalar, minIdxVec);
                redOpMax(resultMax.y, 1, maxScalar, maxIdxVec);
            }
            if constexpr (vector_active_size_v<SrcT> > 2)
            {
                redOpMin(resultMin.z, 2, minScalar, minIdxVec);
                redOpMax(resultMax.z, 2, maxScalar, maxIdxVec);
            }
            if constexpr (vector_active_size_v<SrcT> > 3)
            {
                redOpMin(resultMin.w, 3, minScalar, minIdxVec);
                redOpMax(resultMax.w, 3, maxScalar, maxIdxVec);
            }
            IndexMinMaxChannel idx;
            idx.IndexMin.x = minIdxX[Channel(minIdxVec)];
            idx.IndexMin.y = resultMinIdx[Channel(minIdxVec)];
            idx.ChannelMin = minIdxVec;

            idx.IndexMax.x = maxIdxX[Channel(maxIdxVec)];
            idx.IndexMax.y = resultMaxIdx[Channel(maxIdxVec)];
            idx.ChannelMax = maxIdxVec;

            *aDstScalarMin = minScalar;
            *aDstScalarMax = maxScalar;

            *aDstScalarIdx = idx;
        }

        if (aDstMin != nullptr && aDstMax != nullptr && aDstIdx != nullptr)
        {
            *aDstMin = resultMin;
            *aDstMax = resultMax;
            IndexMinMax idx;
            idx.IndexMin.x = minIdxX.x;
            idx.IndexMin.y = resultMinIdx.x;
            idx.IndexMax.x = maxIdxX.x;
            idx.IndexMax.y = resultMaxIdx.x;
            *aDstIdx       = idx;

            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                idx.IndexMin.x = minIdxX.y;
                idx.IndexMin.y = resultMinIdx.y;
                idx.IndexMax.x = maxIdxX.y;
                idx.IndexMax.y = resultMaxIdx.y;
                aDstIdx[1]     = idx;
            }
            if constexpr (vector_active_size_v<SrcT> > 2)
            {
                idx.IndexMin.x = minIdxX.z;
                idx.IndexMin.y = resultMinIdx.z;
                idx.IndexMax.x = maxIdxX.z;
                idx.IndexMax.y = resultMaxIdx.z;
                aDstIdx[2]     = idx;
            }
            if constexpr (vector_active_size_v<SrcT> > 3)
            {
                idx.IndexMin.x = minIdxX.w;
                idx.IndexMin.y = resultMinIdx.w;
                idx.IndexMax.x = maxIdxX.w;
                idx.IndexMax.y = resultMaxIdx.w;
                aDstIdx[3]     = idx;
            }
        }
    }
}

template <typename SrcT>
void InvokeReductionMinMaxIdxAlongYKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize,
                                          cudaStream_t aStream, const SrcT *aSrcMin, const SrcT *aSrcMax,
                                          const same_vector_size_different_type_t<SrcT, int> *aSrcMinIdxX,
                                          const same_vector_size_different_type_t<SrcT, int> *aSrcMaxIdxX,
                                          SrcT *aDstMin, SrcT *aDstMax, IndexMinMax *aDstIdx,
                                          remove_vector_t<SrcT> *aDstScalarMin, remove_vector_t<SrcT> *aDstScalarMax,
                                          IndexMinMaxChannel *aDstScalarIdx, int aSize)
{
    dim3 blocksPerGrid(1, 1, 1);

    const int size = DIV_UP(aSize, ConfigBlockSize<"DefaultReductionX">::value.y);

    reductionMinMaxIdxAlongYKernel<SrcT><<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(
        aSrcMin, aSrcMax, aSrcMinIdxX, aSrcMaxIdxX, aDstMin, aDstMax, aDstIdx, aDstScalarMin, aDstScalarMax,
        aDstScalarIdx, size);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename SrcT>
void InvokeReductionMinMaxIdxAlongYKernelDefault(
    const SrcT *aSrcMin, const SrcT *aSrcMax, const same_vector_size_different_type_t<SrcT, int> *aSrcMinIdxX,
    const same_vector_size_different_type_t<SrcT, int> *aSrcMaxIdxX, SrcT *aDstMin, SrcT *aDstMax, IndexMinMax *aDstIdx,
    remove_vector_t<SrcT> *aDstScalarMin, remove_vector_t<SrcT> *aDstScalarMax, IndexMinMaxChannel *aDstScalarIdx,
    int aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize = ConfigBlockSize<"DefaultReductionY">::value;

        const uint SharedMemory = 0;

        InvokeReductionMinMaxIdxAlongYKernel<SrcT>(BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream,
                                                   aSrcMin, aSrcMax, aSrcMinIdxX, aSrcMaxIdxX, aDstMin, aDstMax,
                                                   aDstIdx, aDstScalarMin, aDstScalarMax, aDstScalarIdx, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(reductionMinMaxIdxAlongYKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND