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
template <typename T, typename reductionOp> __device__ void DoShuffle(T &aVal, reductionOp aOp)
{
    if constexpr (ComplexVector<T>)
    {
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x.real, 16), aVal.x.real);
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x.real, 8), aVal.x.real);
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x.real, 4), aVal.x.real);
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x.real, 2), aVal.x.real);
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x.real, 1), aVal.x.real);

        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x.imag, 16), aVal.x.imag);
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x.imag, 8), aVal.x.imag);
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x.imag, 4), aVal.x.imag);
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x.imag, 2), aVal.x.imag);
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x.imag, 1), aVal.x.imag);
    }
    else
    {
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x, 16), aVal.x);
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x, 8), aVal.x);
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x, 4), aVal.x);
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x, 2), aVal.x);
        aOp(__shfl_down_sync(0xFFFFFFFF, aVal.x, 1), aVal.x);
    }
    if constexpr (vector_active_size_v<T> > 1)
    {
        if constexpr (ComplexVector<T>)
        {
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y.real, 16), aVal.y.real);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y.real, 8), aVal.y.real);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y.real, 4), aVal.y.real);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y.real, 2), aVal.y.real);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y.real, 1), aVal.y.real);

            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y.imag, 16), aVal.y.imag);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y.imag, 8), aVal.y.imag);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y.imag, 4), aVal.y.imag);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y.imag, 2), aVal.y.imag);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y.imag, 1), aVal.y.imag);
        }
        else
        {
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y, 16), aVal.y);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y, 8), aVal.y);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y, 4), aVal.y);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y, 2), aVal.y);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.y, 1), aVal.y);
        }
    }
    if constexpr (vector_active_size_v<T> > 2)
    {
        if constexpr (ComplexVector<T>)
        {
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z.real, 16), aVal.z.real);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z.real, 8), aVal.z.real);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z.real, 4), aVal.z.real);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z.real, 2), aVal.z.real);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z.real, 1), aVal.z.real);

            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z.imag, 16), aVal.z.imag);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z.imag, 8), aVal.z.imag);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z.imag, 4), aVal.z.imag);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z.imag, 2), aVal.z.imag);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z.imag, 1), aVal.z.imag);
        }
        else
        {
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z, 16), aVal.z);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z, 8), aVal.z);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z, 4), aVal.z);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z, 2), aVal.z);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.z, 1), aVal.z);
        }
    }
    if constexpr (vector_active_size_v<T> > 3)
    {
        if constexpr (ComplexVector<T>)
        {
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w.real, 16), aVal.w.real);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w.real, 8), aVal.w.real);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w.real, 4), aVal.w.real);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w.real, 2), aVal.w.real);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w.real, 1), aVal.w.real);

            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w.imag, 16), aVal.w.imag);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w.imag, 8), aVal.w.imag);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w.imag, 4), aVal.w.imag);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w.imag, 2), aVal.w.imag);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w.imag, 1), aVal.w.imag);
        }
        else
        {
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w, 16), aVal.w);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w, 8), aVal.w);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w, 4), aVal.w);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w, 2), aVal.w);
            aOp(__shfl_down_sync(0xFFFFFFFF, aVal.w, 1), aVal.w);
        }
    }
}

/// <summary>
/// runs aFunctor reduction on every image line, then reduces the thread block along Y - 5 value reduction.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, typename DstT1, typename DstT2, typename DstT3, typename DstT4,
          typename DstT5, typename funcType, typename reductionOp1, typename reductionOp2, typename reductionOp3,
          typename reductionOp4, typename reductionOp5, ReductionInitValue NeutralValue1,
          ReductionInitValue NeutralValue2, ReductionInitValue NeutralValue3, ReductionInitValue NeutralValue4,
          ReductionInitValue NeutralValue5>
__global__ void reduction5AlongXKernel(DstT1 *__restrict__ aDst1, DstT2 *__restrict__ aDst2, DstT3 *__restrict__ aDst3,
                                       DstT4 *__restrict__ aDst4, DstT5 *__restrict__ aDst5, Size2D aSize,
                                       ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit, funcType aFunctor)
{
    int warpLaneID = threadIdx.x;
    int pixelY     = blockIdx.y * blockDim.y + threadIdx.y;

    reductionOp1 redOp1;
    reductionOp2 redOp2;
    reductionOp3 redOp3;
    reductionOp4 redOp4;
    reductionOp5 redOp5;
    DstT1 result1(reduction_init_value_v<NeutralValue1, DstT1>);
    DstT2 result2(reduction_init_value_v<NeutralValue2, DstT2>);
    DstT3 result3(reduction_init_value_v<NeutralValue3, DstT3>);
    DstT4 result4(reduction_init_value_v<NeutralValue4, DstT4>);
    DstT5 result5(reduction_init_value_v<NeutralValue5, DstT5>);
    __shared__ DstT1 buffer1[ConfigBlockSize<"DefaultReductionX">::value.y];
    __shared__ DstT2 buffer2[ConfigBlockSize<"DefaultReductionX">::value.y];
    __shared__ DstT3 buffer3[ConfigBlockSize<"DefaultReductionX">::value.y];
    __shared__ DstT4 buffer4[ConfigBlockSize<"DefaultReductionX">::value.y];
    __shared__ DstT5 buffer5[ConfigBlockSize<"DefaultReductionX">::value.y];

    if (pixelY >= aSize.y)
    {
        if (warpLaneID == 0)
        {
            buffer1[threadIdx.y] =
                reduction_init_value_v<NeutralValue1, DstT1>; // set shared memory buffer to zero for later reduction
            buffer2[threadIdx.y] =
                reduction_init_value_v<NeutralValue2, DstT2>; // set shared memory buffer to zero for later reduction
            buffer3[threadIdx.y] =
                reduction_init_value_v<NeutralValue3, DstT3>; // set shared memory buffer to zero for later reduction
            buffer4[threadIdx.y] =
                reduction_init_value_v<NeutralValue4, DstT4>; // set shared memory buffer to zero for later reduction
            buffer5[threadIdx.y] =
                reduction_init_value_v<NeutralValue5, DstT5>; // set shared memory buffer to zero for later reduction
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
                aFunctor(pixelX, pixelY, result1, result2, result3, result4, result5);
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
                aFunctor(pixelX, pixelY, result1, result2, result3, result4, result5);
            }
        }

        // computer center part as tupels:
        for (int pixelXWarp0 = aSplit.MutedAndLeft(); pixelXWarp0 < aSplit.MutedAndLeftAndCenter();
             pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelCenter(pixelXWarp0 + warpLaneID);
            // center part is always aligned to warpSize, no need to check
            aFunctor(pixelX, pixelY, result1, result2, result3, result4, result5, true);
        }

        // compute right unaligned part:
        for (int pixelXWarp0 = aSplit.MutedAndLeftAndCenter(); pixelXWarp0 < aSplit.Total(); pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelRight(pixelXWarp0 + warpLaneID);
            if (pixelX < aSize.x) // i.e. thread is active
            {
                aFunctor(pixelX, pixelY, result1, result2, result3, result4, result5);
            }
        }
    }

    // reduce over warp:
    DoShuffle(result1, redOp1);
    DoShuffle(result2, redOp2);
    DoShuffle(result3, redOp3);
    DoShuffle(result4, redOp4);
    DoShuffle(result5, redOp5);

    // at this stage, we have reduced the entire image along X to one single column. But we have already all values in
    // registers, so lets further reduce along Y all values that are in the same block:
    if (warpLaneID == 0)
    {
        buffer1[threadIdx.y] = result1;
        buffer2[threadIdx.y] = result2;
        buffer3[threadIdx.y] = result3;
        buffer4[threadIdx.y] = result4;
        buffer5[threadIdx.y] = result5;

        __syncthreads();
        if (threadIdx.y == 0)
        {
#pragma unroll
            for (int i = 1; i < blockDim.y; i++)
            {
                redOp1(buffer1[i], result1);
                redOp2(buffer2[i], result2);
                redOp3(buffer3[i], result3);
                redOp4(buffer4[i], result4);
                redOp5(buffer5[i], result5);
            }

            // Now we have reduced the image to 1/blockDim.y of its original height: store result in global memory and
            // do final reduction with another kernel:
            aDst1[pixelY / blockDim.y] = result1;
            aDst2[pixelY / blockDim.y] = result2;
            aDst3[pixelY / blockDim.y] = result3;
            aDst4[pixelY / blockDim.y] = result4;
            aDst5[pixelY / blockDim.y] = result5;
        }
    }
}

template <typename SrcT, typename DstT1, typename DstT2, typename DstT3, typename DstT4, typename DstT5,
          size_t TupelSize, int WarpAlignmentInBytes, typename funcType, typename reductionOp1, typename reductionOp2,
          typename reductionOp3, typename reductionOp4, typename reductionOp5, ReductionInitValue NeutralValue1,
          ReductionInitValue NeutralValue2, ReductionInitValue NeutralValue3, ReductionInitValue NeutralValue4,
          ReductionInitValue NeutralValue5>
void InvokeReduction5AlongXKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                  const SrcT *aSrc, DstT1 *aDst1, DstT2 *aDst2, DstT3 *aDst3, DstT4 *aDst4,
                                  DstT5 *aDst5, const Size2D &aSize, const funcType &aFunctor)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aSrc, aSize.x, aWarpSize);

    dim3 blocksPerGrid(1, DIV_UP(aSize.y, aBlockSize.y), 1);

    reduction5AlongXKernel<WarpAlignmentInBytes, TupelSize, DstT1, DstT2, DstT3, DstT4, DstT5, funcType, reductionOp1,
                           reductionOp2, reductionOp3, reductionOp4, reductionOp5, NeutralValue1, NeutralValue2,
                           NeutralValue3, NeutralValue4, NeutralValue5>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aDst1, aDst2, aDst3, aDst4, aDst5, aSize, ts, aFunctor);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename SrcT, typename DstT1, typename DstT2, typename DstT3, typename DstT4, typename DstT5,
          size_t TupelSize, typename funcType, typename reductionOp1, typename reductionOp2, typename reductionOp3,
          typename reductionOp4, typename reductionOp5, ReductionInitValue NeutralValue1,
          ReductionInitValue NeutralValue2, ReductionInitValue NeutralValue3, ReductionInitValue NeutralValue4,
          ReductionInitValue NeutralValue5>
void InvokeReduction5AlongXKernelDefault(const SrcT *aSrc, DstT1 *aDst1, DstT2 *aDst2, DstT3 *aDst3, DstT4 *aDst4,
                                         DstT5 *aDst5, const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx,
                                         const funcType &aFunc)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize               = ConfigBlockSize<"DefaultReductionX">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeReduction5AlongXKernel<SrcT, DstT1, DstT2, DstT3, DstT4, DstT5, TupelSize, WarpAlignmentInBytes, funcType,
                                     reductionOp1, reductionOp2, reductionOp3, reductionOp4, reductionOp5,
                                     NeutralValue1, NeutralValue2, NeutralValue3, NeutralValue4, NeutralValue5>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrc, aDst1, aDst2, aDst3, aDst4, aDst5,
            aSize, aFunc);
    }
    else
    {
        throw CUDAUNSUPPORTED(reduction5AlongXKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND