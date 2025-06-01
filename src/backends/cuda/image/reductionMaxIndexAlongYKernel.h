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
/// runs Max with index reduction on one image column, single output value.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <typename SrcT>
__global__ void reductionMaxIdxAlongYKernel(
    const SrcT *__restrict__ aSrcMax, const same_vector_size_different_type_t<SrcT, int> *__restrict__ aSrcMaxIdxX,
    SrcT *__restrict__ aDstMax, same_vector_size_different_type_t<SrcT, int> *__restrict__ aDstMaxIdxX,
    same_vector_size_different_type_t<SrcT, int> *__restrict__ aDstMaxIdxY,
    remove_vector_t<SrcT> *__restrict__ aDstScalarMax, Vector3<int> *__restrict__ aDstScalarIdxMax, int aSize)
{
    using idxT = same_vector_size_different_type_t<SrcT, int>;

    int warpLaneId = threadIdx.x;
    int batchId    = threadIdx.y;

    opp::MaxIdx<SrcT> redOpMax;

    SrcT resultMax(reduction_init_value_v<ReductionInitValue::Min, SrcT>);
    idxT resultMaxIdx(INT_MAX);

    __shared__ SrcT bufferMaxVal[ConfigBlockSize<"DefaultReductionY">::value.y];
    __shared__ idxT bufferMaxIdx[ConfigBlockSize<"DefaultReductionY">::value.y];

    // process 1D input array in threadBlock-junks
    for (int pixelYWarp0 = 0; pixelYWarp0 < aSize; pixelYWarp0 += warpSize * blockDim.y)
    {
        const int pixelY = pixelYWarp0 + warpLaneId + batchId * warpSize;
        if (pixelY < aSize)
        {
            SrcT pxMax = aSrcMax[pixelY];

            redOpMax(pxMax.x, pixelY, resultMax.x, resultMaxIdx.x);
            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                redOpMax(pxMax.y, pixelY, resultMax.y, resultMaxIdx.y);
            }
            if constexpr (vector_active_size_v<SrcT> > 2)
            {
                redOpMax(pxMax.z, pixelY, resultMax.z, resultMaxIdx.z);
            }
            if constexpr (vector_active_size_v<SrcT> > 3)
            {
                redOpMax(pxMax.w, pixelY, resultMax.w, resultMaxIdx.w);
            }
        }
    }

    // reduce over warps:
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
    bufferMaxVal[batchId] = resultMax;
    bufferMaxIdx[batchId] = resultMaxIdx;

    __syncthreads();

    if (warpLaneId == 0 && batchId == 0)
    {
        // reduce over Y the entire thread block:
#pragma unroll
        for (int i = 1; i < ConfigBlockSize<"DefaultReductionYLargeType">::value.y;
             i++) // i = 0 is already stored in result
        {
            redOpMax(bufferMaxVal[i].x, bufferMaxIdx[i].x, resultMax.x, resultMaxIdx.x);
            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                redOpMax(bufferMaxVal[i].y, bufferMaxIdx[i].y, resultMax.y, resultMaxIdx.y);
            }
            if constexpr (vector_active_size_v<SrcT> > 2)
            {
                redOpMax(bufferMaxVal[i].z, bufferMaxIdx[i].z, resultMax.z, resultMaxIdx.z);
            }
            if constexpr (vector_active_size_v<SrcT> > 3)
            {
                redOpMax(bufferMaxVal[i].w, bufferMaxIdx[i].w, resultMax.w, resultMaxIdx.w);
            }
        }

        // fetch X coordinates:
        idxT maxIdxX;
        maxIdxX.x = aSrcMaxIdxX[resultMaxIdx.x].x;
        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            maxIdxX.y = aSrcMaxIdxX[resultMaxIdx.y].y;
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            maxIdxX.z = aSrcMaxIdxX[resultMaxIdx.z].z;
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            maxIdxX.w = aSrcMaxIdxX[resultMaxIdx.w].w;
        }

        if (aDstScalarMax != nullptr && aDstScalarIdxMax != nullptr)
        {
            int maxIdxVec                   = 0;
            remove_vector_t<SrcT> maxScalar = resultMax.x;

            if constexpr (vector_active_size_v<SrcT> > 1)
            {
                redOpMax(resultMax.y, 1, maxScalar, maxIdxVec);
            }
            if constexpr (vector_active_size_v<SrcT> > 2)
            {
                redOpMax(resultMax.z, 2, maxScalar, maxIdxVec);
            }
            if constexpr (vector_active_size_v<SrcT> > 3)
            {
                redOpMax(resultMax.w, 3, maxScalar, maxIdxVec);
            }

            Vector3<int> maxIdx(maxIdxX[Channel(maxIdxVec)], resultMaxIdx[Channel(maxIdxVec)], maxIdxVec);

            *aDstScalarMax = maxScalar;

            *aDstScalarIdxMax = maxIdx;
        }

        if (aDstMax != nullptr && aDstMaxIdxX != nullptr && aDstMaxIdxY != nullptr)
        {
            // don't overwrite alpha channel if it exists:
            if constexpr (has_alpha_channel_v<SrcT>)
            {
                Vector3<remove_vector_t<SrcT>> *dstVec3 = reinterpret_cast<Vector3<remove_vector_t<SrcT>> *>(aDstMax);
                *dstVec3                                = resultMax.XYZ();
                Vector3<int> *dstMaxIdxXVec3            = reinterpret_cast<Vector3<int> *>(aDstMaxIdxX);
                *dstMaxIdxXVec3                         = maxIdxX.XYZ();
                Vector3<int> *dstMaxIdxYVec3            = reinterpret_cast<Vector3<int> *>(aDstMaxIdxY);
                *dstMaxIdxYVec3                         = resultMaxIdx.XYZ();
            }
            else
            {
                *aDstMax     = resultMax;
                *aDstMaxIdxX = maxIdxX;
                *aDstMaxIdxY = resultMaxIdx;
            }
        }
    }
}

template <typename SrcT>
void InvokeReductionMaxIdxAlongYKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                       const SrcT *aSrcMax,
                                       const same_vector_size_different_type_t<SrcT, int> *aSrcMaxIdxX, SrcT *aDstMax,
                                       same_vector_size_different_type_t<SrcT, int> *aDstMaxIdxX,
                                       same_vector_size_different_type_t<SrcT, int> *aDstMaxIdxY,
                                       remove_vector_t<SrcT> *aDstScalarMax, Vector3<int> *aDstScalarIdxMax, int aSize)
{
    dim3 blocksPerGrid(1, 1, 1);

    const int size = DIV_UP(aSize, ConfigBlockSize<"DefaultReductionX">::value.y);

    reductionMaxIdxAlongYKernel<SrcT><<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(
        aSrcMax, aSrcMaxIdxX, aDstMax, aDstMaxIdxX, aDstMaxIdxY, aDstScalarMax, aDstScalarIdxMax, size);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename SrcT>
void InvokeReductionMaxIdxAlongYKernelDefault(const SrcT *aSrcMax,
                                              const same_vector_size_different_type_t<SrcT, int> *aSrcMaxIdxX,
                                              SrcT *aDstMax, same_vector_size_different_type_t<SrcT, int> *aDstMaxIdxX,
                                              same_vector_size_different_type_t<SrcT, int> *aDstMaxIdxY,
                                              remove_vector_t<SrcT> *aDstScalarMax, Vector3<int> *aDstScalarIdxMax,
                                              int aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize = ConfigBlockSize<"DefaultReductionYLargeType">::value;

        const uint SharedMemory = 0;

        InvokeReductionMaxIdxAlongYKernel<SrcT>(BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream,
                                                aSrcMax, aSrcMaxIdxX, aDstMax, aDstMaxIdxX, aDstMaxIdxY, aDstScalarMax,
                                                aDstScalarIdxMax, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(reductionMaxIdxAlongYKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND