#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcReductionMaxIdxFunctor.h>
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
/// runs Max-Index reduction on every image line.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class SrcT>
__global__ void reductionMaxIdxAlongXKernel(const SrcT *aSrc, size_t aPitchSrc, SrcT *__restrict__ aDstMax,
                                            same_vector_size_different_type_t<SrcT, int> *__restrict__ aDstMaxIdx,
                                            Size2D aSize, ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    using idxT = same_vector_size_different_type_t<SrcT, int>;

    int warpLaneID = threadIdx.x;
    int pixelY     = blockIdx.y * blockDim.y + threadIdx.y;

    mpp::MaxIdx<SrcT> redOpMax;
    SrcReductionMaxIdxFunctor<TupelSize, SrcT> functor(aSrc, aPitchSrc);

    SrcT resultMax(reduction_init_value_v<ReductionInitValue::Min, SrcT>);
    idxT resultMaxIdx(INT_MAX);

    if (pixelY >= aSize.y)
    {
        return;
    }

    // simple case, no tupels
    if constexpr (TupelSize == 1)
    {
        // loop over x-dimension in warp steps:
        for (int pixelXWarp0 = 0; pixelXWarp0 < aSize.x; pixelXWarp0 += warpSize)
        {
            const int pixelX = pixelXWarp0 + warpLaneID;
            if (pixelX < aSize.x)
            {
                functor(pixelX, pixelY, resultMax, resultMaxIdx);
            }
        }
    }
    else
    {
        // compute left unaligned part:
        for (int pixelXWarp0 = 0; pixelXWarp0 < aSplit.MutedAndLeft(); pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelLeft(pixelXWarp0 + warpLaneID);
            if (pixelX >= 0) // i.e. thread is active
            {
                functor(pixelX, pixelY, resultMax, resultMaxIdx);
            }
        }

        // computer center part as tupels:
        for (int pixelXWarp0 = aSplit.MutedAndLeft(); pixelXWarp0 < aSplit.MutedAndLeftAndCenter();
             pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelCenter(pixelXWarp0 + warpLaneID);
            // center part is always aligned to warpSize, no need to check
            functor(pixelX, pixelY, resultMax, resultMaxIdx, true);
        }

        // compute right unaligned part:
        for (int pixelXWarp0 = aSplit.MutedAndLeftAndCenter(); pixelXWarp0 < aSplit.Total(); pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelRight(pixelXWarp0 + warpLaneID);
            if (pixelX < aSize.x) // i.e. thread is active
            {
                functor(pixelX, pixelY, resultMax, resultMaxIdx);
            }
        }
    }

    // reduce over warp:
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

    // In other X-reduction kernels we further do a reduction over the block also in Y, but we won't do this for the
    // Index-kernels, in order to maintain the Y position.
    if (warpLaneID == 0)
    {
        {
            aDstMax[pixelY]    = resultMax;
            aDstMaxIdx[pixelY] = resultMaxIdx;
        }
    }
}

template <typename SrcT, size_t TupelSize, int WarpAlignmentInBytes>
void InvokeReductionMaxIdxAlongXKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                       const SrcT *aSrc, size_t aPitchSrc, SrcT *aDstMax,
                                       same_vector_size_different_type_t<SrcT, int> *aDstMaxIdx, Size2D aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aSrc, aSize.x, aWarpSize);

    dim3 blocksPerGrid(1, DIV_UP(aSize.y, aBlockSize.y), 1);

    reductionMaxIdxAlongXKernel<WarpAlignmentInBytes, TupelSize>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrc, aPitchSrc, aDstMax, aDstMaxIdx, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename SrcT, size_t TupelSize>
void InvokeReductionMaxIdxAlongXKernelDefault(const SrcT *aSrc, size_t aPitchSrc, SrcT *aDstMax,
                                              same_vector_size_different_type_t<SrcT, int> *aDstMaxIdx,
                                              const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize               = ConfigBlockSize<"DefaultReductionX">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeReductionMaxIdxAlongXKernel<SrcT, TupelSize, WarpAlignmentInBytes>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrc, aPitchSrc, aDstMax, aDstMaxIdx,
            aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(reductionMaxIdxAlongXKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
