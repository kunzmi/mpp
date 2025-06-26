#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include "integralXKernel.h" // for doShuffleUp functions
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

namespace mpp::image::cuda
{
/// <summary>
/// performs integral image computation in X direction but on a transposed image, thus in Y direction (inplace).
/// </summary>
template <class SrcDstT, int TupelSize, int WarpAlignmentInBytes>
__global__ void integralYKernel(SrcDstT *__restrict__ aSrcDst, size_t aPitchSrcDst, SrcDstT aStartValue, Size2D aSize,
                                ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    int warpLaneID = threadIdx.x;
    int pixelY     = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelY >= aSize.y)
    {
        return;
    }
    SrcDstT previousWarp(aStartValue); // result of righmost value in previous warp

    // simple case, no tupels
    if constexpr (TupelSize == 1)
    {
        // loop over x-dimension in warp steps:
        for (int pixelXWarp0 = 0; pixelXWarp0 < aSize.x; pixelXWarp0 += warpSize)
        {
            const int pixelX = pixelXWarp0 + warpLaneID;
            SrcDstT resVal(0);

            SrcDstT *pixelSrcDst = gotoPtr(aSrcDst, aPitchSrcDst, pixelX, pixelY);
            if (pixelX < aSize.x)
            {
                resVal += *pixelSrcDst;
            }

            doShuffleUp(warpLaneID, resVal.x);

            resVal += previousWarp;

            previousWarp.x = __shfl_sync(0xFFFFFFFF, resVal.x, warpSize - 1);

            if (pixelX < aSize.x)
            {
                *pixelSrcDst = resVal;
            }
        }
    }
    else
    {
        // compute left unaligned part:
        for (int pixelXWarp0 = 0; pixelXWarp0 < aSplit.MutedAndLeft(); pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelLeft(pixelXWarp0 + warpLaneID);
            SrcDstT resVal(0);

            SrcDstT *pixelSrcDst = gotoPtr(aSrcDst, aPitchSrcDst, pixelX, pixelY);
            if (pixelX >= 0) // i.e. thread is active
            {
                resVal += *pixelSrcDst;
            }

            doShuffleUp(warpLaneID, resVal.x);

            resVal += previousWarp;

            previousWarp.x = __shfl_sync(0xFFFFFFFF, resVal.x, warpSize - 1);

            if (pixelX < aSize.x)
            {
                *pixelSrcDst = resVal;
            }
        }

        // computer center part as tupels:
        for (int pixelXWarp0 = aSplit.MutedAndLeft(); pixelXWarp0 < aSplit.MutedAndLeftAndCenter();
             pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelCenter(pixelXWarp0 + warpLaneID);

            Tupel<SrcDstT, TupelSize> tupelSrc;
            Tupel<SrcDstT, TupelSize> tupelDst;

            SrcDstT *pixelSrcDst = gotoPtr(aSrcDst, aPitchSrcDst, pixelX, pixelY);
            tupelSrc             = Tupel<SrcDstT, TupelSize>::Load(pixelSrcDst);

            tupelDst.value[0] = tupelSrc.value[0];
#pragma unroll
            for (size_t i = 1; i < TupelSize; i++)
            {
                tupelDst.value[i] = tupelDst.value[i - 1] + tupelSrc.value[i];
            }

            doShuffleUpTupel<TupelSize, SrcDstT>(warpLaneID, tupelDst);

#pragma unroll
            for (size_t i = 0; i < TupelSize; i++)
            {
                tupelDst.value[i] += previousWarp;
            }

            previousWarp.x = __shfl_sync(0xFFFFFFFF, tupelDst.value[TupelSize - 1].x, warpSize - 1);

            Tupel<SrcDstT, TupelSize>::StoreAligned(tupelDst, pixelSrcDst);
        }

        // compute right unaligned part:
        for (int pixelXWarp0 = aSplit.MutedAndLeftAndCenter(); pixelXWarp0 < aSplit.Total(); pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelRight(pixelXWarp0 + warpLaneID);
            SrcDstT resVal(0);

            SrcDstT *pixelSrcDst = gotoPtr(aSrcDst, aPitchSrcDst, pixelX, pixelY);
            if (pixelX < aSize.x) // i.e. thread is active
            {
                resVal += *pixelSrcDst;
            }

            doShuffleUp(warpLaneID, resVal.x);

            resVal += previousWarp;

            previousWarp.x = __shfl_sync(0xFFFFFFFF, resVal.x, warpSize - 1);

            if (pixelX < aSize.x)
            {
                *pixelSrcDst = resVal;
            }
        }
    }
}

template <class SrcDstT, int TupelSize, int WarpAlignmentInBytes>
void InvokeIntegralYKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                           SrcDstT *aSrcDst, size_t aPitchSrcDst, SrcDstT aStartValue, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aSrcDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(1, DIV_UP(aSize.y, aBlockSize.y), 1);

    integralYKernel<SrcDstT, TupelSize, WarpAlignmentInBytes>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcDst, aPitchSrcDst, aStartValue, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename SrcDstT, size_t TupelSize>
void InvokeIntegralYKernelDefault(SrcDstT *aSrcDst, size_t aPitchSrcDst, SrcDstT aStartValue, const Size2D &aSize,
                                  const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize = {32, 16, 1};
        // ConfigBlockSize<"DefaultReductionX">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeIntegralYKernel<SrcDstT, TupelSize, WarpAlignmentInBytes>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcDst, aPitchSrcDst, aStartValue, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(integralYKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND