#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/maskTupel.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{
/// <summary>
/// runs aFunctor reduction on every image line, then reduces the thread block along Y - two value reduction.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class DstT1, class DstT2, typename funcType, typename reductionOp1,
          typename reductionOp2, ReductionInitValue NeutralValue1, ReductionInitValue NeutralValue2>
__global__ void reduction2MaskedCountingAlongXKernel(const byte *__restrict__ aMask, size_t aPitchMask,
                                                     DstT1 *__restrict__ aDst1, DstT2 *__restrict__ aDst2,
                                                     ulong64 *__restrict__ aMaskCounters, Size2D aSize,
                                                     ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit,
                                                     funcType aFunctor)
{
    int warpLaneID = threadIdx.x;
    int pixelY     = blockIdx.y * blockDim.y + threadIdx.y;

    reductionOp1 redOp1;
    reductionOp2 redOp2;
    DstT1 result1(reduction_init_value_v<NeutralValue1, DstT1>);
    DstT2 result2(reduction_init_value_v<NeutralValue2, DstT2>);
    __shared__ DstT1 buffer1[ConfigBlockSize<"DefaultReductionX">::value.y];
    __shared__ DstT2 buffer2[ConfigBlockSize<"DefaultReductionX">::value.y];
    __shared__ int bufferMask[ConfigBlockSize<"DefaultReductionX">::value.y];
    int maskCounter = 0; // counts the pixels in mask

    if (pixelY >= aSize.y)
    {
        if (warpLaneID == 0)
        {
            // set shared memory buffer to zero for later reduction
            buffer1[threadIdx.y]    = reduction_init_value_v<NeutralValue1, DstT1>;
            buffer2[threadIdx.y]    = reduction_init_value_v<NeutralValue2, DstT2>;
            bufferMask[threadIdx.y] = 0;
        }
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
                const byte mask = *gotoPtr(aMask, aPitchMask, pixelX, pixelY);
                if (mask)
                {
                    aFunctor(pixelX, pixelY, result1, result2);
                    maskCounter++;
                }
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
                const byte mask = *gotoPtr(aMask, aPitchMask, pixelX, pixelY);
                if (mask)
                {
                    aFunctor(pixelX, pixelY, result1, result2);
                    maskCounter++;
                }
            }
        }

        // compute center part as tupels:
        for (int pixelXWarp0 = aSplit.MutedAndLeft(); pixelXWarp0 < aSplit.MutedAndLeftAndCenter();
             pixelXWarp0 += warpSize)
        {
            const int pixelX               = aSplit.GetPixelCenter(pixelXWarp0 + warpLaneID);
            const byte *pixelsMask         = gotoPtr(aMask, aPitchMask, pixelX, pixelY);
            MaskTupel<TupelSize> maskTupel = MaskTupel<TupelSize>::Load(pixelsMask);
            if (maskTupel.AreAllFalse())
            {
                // nothing to do for these pixels
                continue;
            }

            // center part is always aligned to warpSize, no need to check
            aFunctor(pixelX, pixelY, result1, result2, maskTupel, maskCounter);
        }

        // compute right unaligned part:
        for (int pixelXWarp0 = aSplit.MutedAndLeftAndCenter(); pixelXWarp0 < aSplit.Total(); pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelRight(pixelXWarp0 + warpLaneID);
            if (pixelX < aSize.x) // i.e. thread is active
            {
                const byte mask = *gotoPtr(aMask, aPitchMask, pixelX, pixelY);
                if (mask)
                {
                    aFunctor(pixelX, pixelY, result1, result2);
                    maskCounter++;
                }
            }
        }
    }

    // reduce over warp:
    if constexpr (ComplexVector<DstT1>) // DstT2 is also complex in case DstT1 is complex
    {
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.real, 16), result1.x.real);
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.real, 8), result1.x.real);
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.real, 4), result1.x.real);
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.real, 2), result1.x.real);
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.real, 1), result1.x.real);

        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.real, 16), result2.x.real);
        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.real, 8), result2.x.real);
        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.real, 4), result2.x.real);
        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.real, 2), result2.x.real);
        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.real, 1), result2.x.real);

        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.imag, 16), result1.x.imag);
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.imag, 8), result1.x.imag);
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.imag, 4), result1.x.imag);
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.imag, 2), result1.x.imag);
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x.imag, 1), result1.x.imag);

        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.imag, 16), result2.x.imag);
        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.imag, 8), result2.x.imag);
        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.imag, 4), result2.x.imag);
        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.imag, 2), result2.x.imag);
        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x.imag, 1), result2.x.imag);
    }
    else
    {
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x, 16), result1.x);
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x, 8), result1.x);
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x, 4), result1.x);
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x, 2), result1.x);
        redOp1(__shfl_down_sync(0xFFFFFFFF, result1.x, 1), result1.x);

        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x, 16), result2.x);
        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x, 8), result2.x);
        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x, 4), result2.x);
        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x, 2), result2.x);
        redOp2(__shfl_down_sync(0xFFFFFFFF, result2.x, 1), result2.x);
    }

    if constexpr (vector_active_size_v<DstT1> > 1)
    {
        if constexpr (ComplexVector<DstT1>) // DstT2 is also complex in case DstT1 is complex
        {
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.real, 16), result1.y.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.real, 8), result1.y.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.real, 4), result1.y.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.real, 2), result1.y.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.real, 1), result1.y.real);

            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.real, 16), result2.y.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.real, 8), result2.y.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.real, 4), result2.y.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.real, 2), result2.y.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.real, 1), result2.y.real);

            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.imag, 16), result1.y.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.imag, 8), result1.y.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.imag, 4), result1.y.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.imag, 2), result1.y.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y.imag, 1), result1.y.imag);

            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.imag, 16), result2.y.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.imag, 8), result2.y.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.imag, 4), result2.y.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.imag, 2), result2.y.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y.imag, 1), result2.y.imag);
        }
        else
        {
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y, 16), result1.y);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y, 8), result1.y);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y, 4), result1.y);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y, 2), result1.y);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.y, 1), result1.y);

            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y, 16), result2.y);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y, 8), result2.y);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y, 4), result2.y);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y, 2), result2.y);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.y, 1), result2.y);
        }
    }
    if constexpr (vector_active_size_v<DstT1> > 2)
    {
        if constexpr (ComplexVector<DstT1>) // DstT2 is also complex in case DstT1 is complex
        {
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.real, 16), result1.z.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.real, 8), result1.z.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.real, 4), result1.z.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.real, 2), result1.z.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.real, 1), result1.z.real);

            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.real, 16), result2.z.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.real, 8), result2.z.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.real, 4), result2.z.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.real, 2), result2.z.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.real, 1), result2.z.real);

            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.imag, 16), result1.z.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.imag, 8), result1.z.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.imag, 4), result1.z.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.imag, 2), result1.z.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z.imag, 1), result1.z.imag);

            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.imag, 16), result2.z.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.imag, 8), result2.z.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.imag, 4), result2.z.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.imag, 2), result2.z.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z.imag, 1), result2.z.imag);
        }
        else
        {
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z, 16), result1.z);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z, 8), result1.z);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z, 4), result1.z);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z, 2), result1.z);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.z, 1), result1.z);

            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z, 16), result2.z);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z, 8), result2.z);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z, 4), result2.z);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z, 2), result2.z);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.z, 1), result2.z);
        }
    }
    if constexpr (vector_active_size_v<DstT1> > 3)
    {
        if constexpr (ComplexVector<DstT1>) // DstT2 is also complex in case DstT1 is complex
        {
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.real, 16), result1.w.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.real, 8), result1.w.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.real, 4), result1.w.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.real, 2), result1.w.real);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.real, 1), result1.w.real);

            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.real, 16), result2.w.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.real, 8), result2.w.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.real, 4), result2.w.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.real, 2), result2.w.real);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.real, 1), result2.w.real);

            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.imag, 16), result1.w.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.imag, 8), result1.w.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.imag, 4), result1.w.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.imag, 2), result1.w.imag);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w.imag, 1), result1.w.imag);

            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.imag, 16), result2.w.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.imag, 8), result2.w.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.imag, 4), result2.w.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.imag, 2), result2.w.imag);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w.imag, 1), result2.w.imag);
        }
        else
        {
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w, 16), result1.w);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w, 8), result1.w);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w, 4), result1.w);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w, 2), result1.w);
            redOp1(__shfl_down_sync(0xFFFFFFFF, result1.w, 1), result1.w);

            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w, 16), result2.w);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w, 8), result2.w);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w, 4), result2.w);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w, 2), result2.w);
            redOp2(__shfl_down_sync(0xFFFFFFFF, result2.w, 1), result2.w);
        }
    }

    // reduce maskCounter over warp:
    maskCounter += __shfl_down_sync(0xFFFFFFFF, maskCounter, 16);
    maskCounter += __shfl_down_sync(0xFFFFFFFF, maskCounter, 8);
    maskCounter += __shfl_down_sync(0xFFFFFFFF, maskCounter, 4);
    maskCounter += __shfl_down_sync(0xFFFFFFFF, maskCounter, 2);
    maskCounter += __shfl_down_sync(0xFFFFFFFF, maskCounter, 1);

    // at this stage, we have reduced the entire image along X to one single column. But we have already all values in
    // registers, so lets further reduce along Y all values that are in the same block:
    if (warpLaneID == 0)
    {
        buffer1[threadIdx.y]    = result1;
        buffer2[threadIdx.y]    = result2;
        bufferMask[threadIdx.y] = maskCounter;

        __syncthreads();
        if (threadIdx.y == 0)
        {
            ulong64 maskCounterUL = maskCounter; // thread0
#pragma unroll
            for (int i = 1; i < blockDim.y; i++)
            {
                redOp1(buffer1[i], result1);
                redOp2(buffer2[i], result2);
                maskCounterUL += bufferMask[i];
            }

            // Now we have reduced the image to 1/blockDim.y of its original height: store result in global memory and
            // do final reduction with another kernel:
            aDst1[pixelY / blockDim.y]         = result1;
            aDst2[pixelY / blockDim.y]         = result2;
            aMaskCounters[pixelY / blockDim.y] = maskCounterUL;
        }
    }
}

template <typename SrcT, typename DstT1, typename DstT2, size_t TupelSize, int WarpAlignmentInBytes, typename funcType,
          typename reductionOp1, typename reductionOp2, ReductionInitValue NeutralValue1,
          ReductionInitValue NeutralValue2>
void InvokeReduction2MaskedCountingAlongXKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize,
                                                cudaStream_t aStream, const byte *aMask, size_t aPitchMask,
                                                const SrcT *aSrc, DstT1 *aDst1, DstT2 *aDst2, ulong64 *aMaskCounters,
                                                const Size2D &aSize, const funcType &aFunctor)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aSrc, aSize.x, aWarpSize);

    dim3 blocksPerGrid(1, DIV_UP(aSize.y, aBlockSize.y), 1);

    reduction2MaskedCountingAlongXKernel<WarpAlignmentInBytes, TupelSize, DstT1, DstT2, funcType, reductionOp1,
                                         reductionOp2, NeutralValue1, NeutralValue2>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aMask, aPitchMask, aDst1, aDst2, aMaskCounters, aSize,
                                                                ts, aFunctor);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename SrcT, typename DstT1, typename DstT2, size_t TupelSize, typename funcType, typename reductionOp1,
          typename reductionOp2, ReductionInitValue NeutralValue1, ReductionInitValue NeutralValue2>
void InvokeReduction2MaskedCountingAlongXKernelDefault(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc,
                                                       DstT1 *aDst1, DstT2 *aDst2, ulong64 *aMaskCounters,
                                                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx,
                                                       const funcType &aFunc)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize               = ConfigBlockSize<"DefaultReductionX">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeReduction2MaskedCountingAlongXKernel<SrcT, DstT1, DstT2, TupelSize, WarpAlignmentInBytes, funcType,
                                                   reductionOp1, reductionOp2, NeutralValue1, NeutralValue2>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, reinterpret_cast<const byte *>(aMask),
            aPitchMask, aSrc, aDst1, aDst2, aMaskCounters, aSize, aFunc);
    }
    else
    {
        throw CUDAUNSUPPORTED(reduction2MaskedCountingAlongXKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
