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
#include <common/maskTupel.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace opp::image::cuda
{
/// <summary>
/// runs aFunctor reduction on every image line, then reduces the thread block along Y - single value reduction.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class DstT, typename funcType, typename reductionOp,
          ReductionInitValue NeutralValue>
__global__ void reductionMaskedAlongXKernel(const byte *__restrict__ aMask, size_t aPitchMask, DstT *__restrict__ aDst,
                                            Size2D aSize, ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit,
                                            funcType aFunctor)
{
    int warpLaneID = threadIdx.x;
    int pixelY     = blockIdx.y * blockDim.y + threadIdx.y;

    reductionOp redOp;
    DstT result(reduction_init_value_v<NeutralValue, DstT>);
    __shared__ DstT buffer[ConfigBlockSize<"DefaultReductionX">::value.y];

    if (pixelY >= aSize.y)
    {
        if (warpLaneID == 0)
        {
            // set shared memory buffer to zero for later reduction
            buffer[threadIdx.y] = DstT(reduction_init_value_v<NeutralValue, DstT>);
        }
        return;
    }

    // simple case, no tupels
    if constexpr (TupelSize == 1)
    {
        // loop over x-dimension in warp steps:
        for (int pixelXWarp0 = 0; pixelXWarp0 < aSize.x; pixelXWarp0 += warpSize)
        {
            // DstT threadValue(0);
            const int pixelX = pixelXWarp0 + warpLaneID;
            if (pixelX < aSize.x)
            {
                const byte mask = *gotoPtr(aMask, aPitchMask, pixelX, pixelY);
                if (mask)
                {
                    aFunctor(pixelX, pixelY, result);
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
                    aFunctor(pixelX, pixelY, result);
                }
            }
        }

        // computer center part as tupels:
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
            aFunctor(pixelX, pixelY, result, maskTupel);
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
                    aFunctor(pixelX, pixelY, result);
                }
            }
        }
    }

    // reduce over warp:
    if constexpr (ComplexVector<DstT>)
    {
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x.real, 16), result.x.real);
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x.real, 8), result.x.real);
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x.real, 4), result.x.real);
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x.real, 2), result.x.real);
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x.real, 1), result.x.real);

        redOp(__shfl_down_sync(0xFFFFFFFF, result.x.imag, 16), result.x.imag);
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x.imag, 8), result.x.imag);
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x.imag, 4), result.x.imag);
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x.imag, 2), result.x.imag);
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x.imag, 1), result.x.imag);
    }
    else
    {
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x, 16), result.x);
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x, 8), result.x);
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x, 4), result.x);
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x, 2), result.x);
        redOp(__shfl_down_sync(0xFFFFFFFF, result.x, 1), result.x);
    }

    if constexpr (vector_active_size_v<DstT> > 1)
    {
        if constexpr (ComplexVector<DstT>)
        {
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y.real, 16), result.y.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y.real, 8), result.y.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y.real, 4), result.y.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y.real, 2), result.y.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y.real, 1), result.y.real);

            redOp(__shfl_down_sync(0xFFFFFFFF, result.y.imag, 16), result.y.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y.imag, 8), result.y.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y.imag, 4), result.y.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y.imag, 2), result.y.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y.imag, 1), result.y.imag);
        }
        else
        {
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y, 16), result.y);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y, 8), result.y);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y, 4), result.y);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y, 2), result.y);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.y, 1), result.y);
        }
    }
    if constexpr (vector_active_size_v<DstT> > 2)
    {
        if constexpr (ComplexVector<DstT>)
        {
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z.real, 16), result.z.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z.real, 8), result.z.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z.real, 4), result.z.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z.real, 2), result.z.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z.real, 1), result.z.real);

            redOp(__shfl_down_sync(0xFFFFFFFF, result.z.imag, 16), result.z.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z.imag, 8), result.z.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z.imag, 4), result.z.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z.imag, 2), result.z.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z.imag, 1), result.z.imag);
        }
        else
        {
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z, 16), result.z);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z, 8), result.z);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z, 4), result.z);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z, 2), result.z);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.z, 1), result.z);
        }
    }
    if constexpr (vector_active_size_v<DstT> > 3)
    {
        if constexpr (ComplexVector<DstT>)
        {
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w.real, 16), result.w.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w.real, 8), result.w.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w.real, 4), result.w.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w.real, 2), result.w.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w.real, 1), result.w.real);

            redOp(__shfl_down_sync(0xFFFFFFFF, result.w.imag, 16), result.w.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w.imag, 8), result.w.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w.imag, 4), result.w.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w.imag, 2), result.w.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w.imag, 1), result.w.imag);
        }
        else
        {
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w, 16), result.w);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w, 8), result.w);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w, 4), result.w);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w, 2), result.w);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.w, 1), result.w);
        }
    }

    // at this stage, we have reduced the entire image along X to one single column. But we have already all values in
    // registers, so lets further reduce along Y all values that are in the same block:
    if (warpLaneID == 0)
    {
        buffer[threadIdx.y] = result;

        __syncthreads();
        if (threadIdx.y == 0)
        {
#pragma unroll
            for (int i = 1; i < blockDim.y; i++)
            {
                redOp(buffer[i], result);
            }

            // Now we have reduced the image to 1/blockDim.y of its original height: store result in global memory and
            // do final reduction with another kernel:
            aDst[pixelY / blockDim.y] = result;
        }
    }
}

template <typename SrcT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, typename funcType,
          typename reductionOp, ReductionInitValue NeutralValue>
void InvokeReductionMaskedAlongXKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                       const byte *aMask, size_t aPitchMask, const SrcT *aSrc, DstT *aDst,
                                       const Size2D &aSize, const funcType &aFunctor)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aSrc, aSize.x, aWarpSize);

    dim3 blocksPerGrid(1, DIV_UP(aSize.y, aBlockSize.y), 1);

    reductionMaskedAlongXKernel<WarpAlignmentInBytes, TupelSize, DstT, funcType, reductionOp, NeutralValue>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aMask, aPitchMask, aDst, aSize, ts, aFunctor);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename SrcT, typename DstT, size_t TupelSize, typename funcType, typename reductionOp,
          ReductionInitValue NeutralValue>
void InvokeReductionMaskedAlongXKernelDefault(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc, DstT *aDst,
                                              const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx,
                                              const funcType &aFunc)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize               = ConfigBlockSize<"DefaultReductionX">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeReductionMaskedAlongXKernel<SrcT, DstT, TupelSize, WarpAlignmentInBytes, funcType, reductionOp,
                                          NeutralValue>(BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream,
                                                        reinterpret_cast<const byte *>(aMask), aPitchMask, aSrc, aDst,
                                                        aSize, aFunc);
    }
    else
    {
        throw CUDAUNSUPPORTED(reductionMaskedAlongXKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND