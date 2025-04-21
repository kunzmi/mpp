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

template <typename T> __device__ void doShuffleUp(int warpLaneID, T &value)
{
    T temp = 0;
    temp   = __shfl_up_sync(0xFFFFFFFF, value, 1);
    if (warpLaneID >= 1)
    {
        value += temp;
    }
    temp = __shfl_up_sync(0xFFFFFFFF, value, 2);
    if (warpLaneID >= 2)
    {
        value += temp;
    }
    temp = __shfl_up_sync(0xFFFFFFFF, value, 4);
    if (warpLaneID >= 4)
    {
        value += temp;
    }
    temp = __shfl_up_sync(0xFFFFFFFF, value, 8);
    if (warpLaneID >= 8)
    {
        value += temp;
    }
    temp = __shfl_up_sync(0xFFFFFFFF, value, 16);
    if (warpLaneID >= 16)
    {
        value += temp;
    }
}

template <int TupelSize, typename T> __device__ void doShuffleUpTupel(int warpLaneID, Tupel<T, TupelSize> &value)
{

    remove_vector_t<T> temp = 0;

    temp = __shfl_up_sync(0xFFFFFFFF, value.value[TupelSize - 1].x, 1);
    if (warpLaneID >= 1)
    {
#pragma unroll
        for (size_t i = 0; i < TupelSize; i++)
        {
            value.value[i].x += temp;
        }
    }

    temp = __shfl_up_sync(0xFFFFFFFF, value.value[TupelSize - 1].x, 2);
    if (warpLaneID >= 2)
    {
#pragma unroll
        for (size_t i = 0; i < TupelSize; i++)
        {
            value.value[i].x += temp;
        }
    }

    temp = __shfl_up_sync(0xFFFFFFFF, value.value[TupelSize - 1].x, 4);
    if (warpLaneID >= 4)
    {
#pragma unroll
        for (size_t i = 0; i < TupelSize; i++)
        {
            value.value[i].x += temp;
        }
    }

    temp = __shfl_up_sync(0xFFFFFFFF, value.value[TupelSize - 1].x, 8);
    if (warpLaneID >= 8)
    {
#pragma unroll
        for (size_t i = 0; i < TupelSize; i++)
        {
            value.value[i].x += temp;
        }
    }

    temp = __shfl_up_sync(0xFFFFFFFF, value.value[TupelSize - 1].x, 16);
    if (warpLaneID >= 16)
    {
#pragma unroll
        for (size_t i = 0; i < TupelSize; i++)
        {
            value.value[i].x += temp;
        }
    }
}

/// <summary>
/// performs integral image computation in X direction.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <int TupelSize, int WarpAlignmentInBytes, class SrcT, class DstT>
__global__ void integralXKernel(const SrcT *__restrict__ aSrc, size_t aPitchSrc, DstT *__restrict__ aDst,
                                size_t aPitchDst, Size2D aSize, ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    int warpLaneID = threadIdx.x;
    int pixelY     = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelY >= aSize.y)
    {
        return;
    }
    DstT previousWarp(0); // result of righmost value in previous warp

    // simple case, no tupels
    if constexpr (TupelSize == 1)
    {
        // loop over x-dimension in warp steps:
        for (int pixelXWarp0 = 0; pixelXWarp0 < aSize.x; pixelXWarp0 += warpSize)
        {
            const int pixelX = pixelXWarp0 + warpLaneID;
            DstT resVal(0);
            if (pixelY > 0)
            {
                if (pixelX < aSize.x)
                {
                    if (pixelX > 0)
                    {
                        // pixelX - 1 do add 0 border at the left:
                        const DstT pixelSrc = DstT(*gotoPtr(aSrc, aPitchSrc, pixelX - 1, pixelY - 1));
                        resVal += pixelSrc;
                    }
                }

                doShuffleUp(warpLaneID, resVal.x);

                resVal.x += previousWarp.x;

                previousWarp.x = __shfl_sync(0xFFFFFFFF, resVal.x, warpSize - 1);
            }
            if (pixelX < aSize.x)
            {
                DstT *pixelDst = gotoPtr(aDst, aPitchDst, pixelX, pixelY);
                *pixelDst      = resVal;
            }
        }
    }
    else
    {
        // compute left unaligned part:
        for (int pixelXWarp0 = 0; pixelXWarp0 < aSplit.MutedAndLeft(); pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelLeft(pixelXWarp0 + warpLaneID);
            DstT resVal(0);
            if (pixelY > 0)
            {
                if (pixelX >= 0) // i.e. thread is active
                {
                    if (pixelX > 0)
                    {
                        // pixelX - 1 do add 0 border at the left:
                        const DstT pixelSrc = DstT(*gotoPtr(aSrc, aPitchSrc, pixelX - 1, pixelY - 1));
                        resVal += pixelSrc;
                    }
                }

                doShuffleUp(warpLaneID, resVal.x);

                resVal.x += previousWarp.x;

                previousWarp.x = __shfl_sync(0xFFFFFFFF, resVal.x, warpSize - 1);
            }
            if (pixelX < aSize.x)
            {
                DstT *pixelDst = gotoPtr(aDst, aPitchDst, pixelX, pixelY);
                *pixelDst      = resVal;
            }
        }

        // computer center part as tupels:
        for (int pixelXWarp0 = aSplit.MutedAndLeft(); pixelXWarp0 < aSplit.MutedAndLeftAndCenter();
             pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelCenter(pixelXWarp0 + warpLaneID);

            Tupel<SrcT, TupelSize> tupelSrc;
            Tupel<DstT, TupelSize> tupelDst;

            if (pixelY > 0)
            {
                if (pixelX == 0)
                {
                    tupelSrc.value[0] = DstT(0);
                    // don't load the zero padded pixel, as it doesn't exist in source image:
#pragma unroll
                    for (size_t i = 1; i < TupelSize; i++)
                    {
                        tupelSrc.value[i] = *gotoPtr(aSrc, aPitchSrc, pixelX - 1 + (int)i, pixelY - 1);
                    }
                }
                else
                {
                    const SrcT *pixelSrc = gotoPtr(aSrc, aPitchSrc, pixelX - 1, pixelY - 1);
                    tupelSrc             = Tupel<SrcT, TupelSize>::Load(pixelSrc);
                }

                tupelDst.value[0] = DstT(tupelSrc.value[0]);
#pragma unroll
                for (size_t i = 1; i < TupelSize; i++)
                {
                    tupelDst.value[i] = tupelDst.value[i - 1] + DstT(tupelSrc.value[i]);
                }

                doShuffleUpTupel<TupelSize, DstT>(warpLaneID, tupelDst);

#pragma unroll
                for (size_t i = 0; i < TupelSize; i++)
                {
                    tupelDst.value[i] += previousWarp;
                }

                previousWarp.x = __shfl_sync(0xFFFFFFFF, tupelDst.value[TupelSize - 1].x, warpSize - 1);
            }
            else
            {
#pragma unroll
                for (size_t i = 0; i < TupelSize; i++)
                {
                    tupelDst.value[i] = DstT(0);
                }
            }
            if (pixelX < aSize.x)
            {
                DstT *pixelDst = gotoPtr(aDst, aPitchDst, pixelX, pixelY);

                Tupel<DstT, TupelSize>::StoreAligned(tupelDst, pixelDst);
            }
        }

        // compute right unaligned part:
        for (int pixelXWarp0 = aSplit.MutedAndLeftAndCenter(); pixelXWarp0 < aSplit.Total(); pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelRight(pixelXWarp0 + warpLaneID);
            DstT resVal(0);
            if (pixelY > 0)
            {
                if (pixelX < aSize.x) // i.e. thread is active
                {
                    if (pixelX > 0)
                    {
                        // pixelX - 1 do add 0 border at the left:
                        const DstT pixelSrc = DstT(*gotoPtr(aSrc, aPitchSrc, pixelX - 1, pixelY - 1));
                        resVal += pixelSrc;
                    }
                }

                doShuffleUp(warpLaneID, resVal.x);

                resVal.x += previousWarp.x;

                previousWarp.x = __shfl_sync(0xFFFFFFFF, resVal.x, warpSize - 1);
            }
            if (pixelX < aSize.x)
            {
                DstT *pixelDst = gotoPtr(aDst, aPitchDst, pixelX, pixelY);
                *pixelDst      = resVal;
            }
        }
    }
}

template <typename SrcT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes>
void InvokeIntegralXKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                           const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(1, DIV_UP(aSize.y, aBlockSize.y), 1);

    integralXKernel<TupelSize, WarpAlignmentInBytes, SrcT, DstT>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrc, aPitchSrc, aDst, aPitchDst, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename SrcT, typename DstT, size_t TupelSize>
void InvokeIntegralXKernelDefault(const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                                  const opp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize = {32, 16, 1};
        // ConfigBlockSize<"DefaultReductionX">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeIntegralXKernel<SrcT, DstT, TupelSize, WarpAlignmentInBytes>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrc, aPitchSrc, aDst, aPitchDst, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(integralXKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND