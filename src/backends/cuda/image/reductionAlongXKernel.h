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
template <int WarpAlignmentInBytes, int TupelSize, class DstT, typename funcType>
__global__ void reductionAlongXKernel(DstT *__restrict__ aDst, Size2D aSize,
                                      ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit, funcType aFunctor)
{
    int warpLaneID = threadIdx.x;
    int pixelY     = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelY >= aSize.y)
    {
        return;
    }

    // simple case, no tupels
    if constexpr (TupelSize == 1)
    {
        DstT result(0);

        for (int pixelXWarp0 = 0; pixelXWarp0 < aSize.x; pixelXWarp0 += warpSize)
        {
            // DstT threadValue(0);
            const int pixelX = pixelXWarp0 + warpLaneID;
            if (pixelX < aSize.x)
            {
                aFunctor(pixelX, pixelY, result);
                // aFunctor(pixelX, pixelY, threadValue);
            }

            //// reduce over warp:
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 16);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 8);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 4);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 2);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 1);
            //
            // if constexpr (vector_active_size_v<DstT> > 1)
            //{
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 16);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 8);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 4);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 2);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 1);
            // }
            // if constexpr (vector_active_size_v<DstT> > 2)
            //{
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 16);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 8);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 4);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 2);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 1);
            // }
            // if constexpr (vector_active_size_v<DstT> > 3)
            //{
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 16);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 8);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 4);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 2);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 1);
            // }
            //
            // result += threadValue;
        }

        // reduce over warp:
        result.x += __shfl_down_sync(0xFFFFFFFF, result.x, 16);
        result.x += __shfl_down_sync(0xFFFFFFFF, result.x, 8);
        result.x += __shfl_down_sync(0xFFFFFFFF, result.x, 4);
        result.x += __shfl_down_sync(0xFFFFFFFF, result.x, 2);
        result.x += __shfl_down_sync(0xFFFFFFFF, result.x, 1);

        if constexpr (vector_active_size_v<DstT> > 1)
        {
            result.y += __shfl_down_sync(0xFFFFFFFF, result.y, 16);
            result.y += __shfl_down_sync(0xFFFFFFFF, result.y, 8);
            result.y += __shfl_down_sync(0xFFFFFFFF, result.y, 4);
            result.y += __shfl_down_sync(0xFFFFFFFF, result.y, 2);
            result.y += __shfl_down_sync(0xFFFFFFFF, result.y, 1);
        }
        if constexpr (vector_active_size_v<DstT> > 2)
        {
            result.z += __shfl_down_sync(0xFFFFFFFF, result.z, 16);
            result.z += __shfl_down_sync(0xFFFFFFFF, result.z, 8);
            result.z += __shfl_down_sync(0xFFFFFFFF, result.z, 4);
            result.z += __shfl_down_sync(0xFFFFFFFF, result.z, 2);
            result.z += __shfl_down_sync(0xFFFFFFFF, result.z, 1);
        }
        if constexpr (vector_active_size_v<DstT> > 3)
        {
            result.w += __shfl_down_sync(0xFFFFFFFF, result.w, 16);
            result.w += __shfl_down_sync(0xFFFFFFFF, result.w, 8);
            result.w += __shfl_down_sync(0xFFFFFFFF, result.w, 4);
            result.w += __shfl_down_sync(0xFFFFFFFF, result.w, 2);
            result.w += __shfl_down_sync(0xFFFFFFFF, result.w, 1);
        }

        if (warpLaneID == 0)
        {
            aDst[pixelY] = result;
        }
    }
    else
    {
        DstT result(0);

        // compute left unaligned part:
        for (int pixelXWarp0 = 0; pixelXWarp0 < aSplit.MutedAndLeft(); pixelXWarp0 += warpSize)
        {
            // DstT threadValue(0);
            const int pixelX = aSplit.GetPixel(pixelXWarp0 + warpLaneID);
            if (pixelX >= 0 && pixelX < aSplit.MutedAndLeft()) // i.e. thread is active
            {
                aFunctor(pixelX, pixelY, result);
                // aFunctor(pixelX, pixelY, threadValue);
            }

            //// reduce over warp:
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 16);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 8);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 4);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 2);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 1);

            // if constexpr (vector_active_size_v<DstT> > 1)
            //{
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 16);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 8);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 4);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 2);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 1);
            // }
            // if constexpr (vector_active_size_v<DstT> > 2)
            //{
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 16);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 8);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 4);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 2);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 1);
            // }
            // if constexpr (vector_active_size_v<DstT> > 3)
            //{
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 16);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 8);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 4);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 2);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 1);
            // }
            //
            // result += threadValue;
        }

        // computer center part as tupels:
        for (int pixelXWarp0 = aSplit.MutedAndLeft(); pixelXWarp0 < aSplit.MutedAndLeftAndCenter();
             pixelXWarp0 += warpSize)
        {
            // DstT threadValue(0);

            const int pixelX = aSplit.GetPixel(pixelXWarp0 + warpLaneID);
            // if (pixelX < aSize.x) center part is always aligned to warpSize, no need to check
            {
                // aFunctor(pixelX, pixelY, threadValue, true);
                aFunctor(pixelX, pixelY, result, true);
            }

            //// reduce over warp:
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 16);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 8);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 4);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 2);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 1);
            //
            // if constexpr (vector_active_size_v<DstT> > 1)
            //{
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 16);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 8);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 4);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 2);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 1);
            // }
            // if constexpr (vector_active_size_v<DstT> > 2)
            //{
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 16);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 8);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 4);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 2);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 1);
            // }
            // if constexpr (vector_active_size_v<DstT> > 3)
            //{
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 16);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 8);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 4);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 2);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 1);
            // }
            //
            // result += threadValue;
        }

        // compute right unaligned part:
        for (int pixelXWarp0 = aSplit.MutedAndLeftAndCenter(); pixelXWarp0 < aSplit.Total(); pixelXWarp0 += warpSize)
        {
            // DstT threadValue(0);
            const int pixelX = aSplit.GetPixel(pixelXWarp0 + warpLaneID);
            if (pixelX < aSize.x) // i.e. thread is active
            {
                // aFunctor(pixelX, pixelY, threadValue);
                aFunctor(pixelX, pixelY, result);
            }

            //// reduce over warp:
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 16);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 8);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 4);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 2);
            // threadValue.x += __shfl_down_sync(0xFFFFFFFF, threadValue.x, 1);
            //
            // if constexpr (vector_active_size_v<DstT> > 1)
            //{
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 16);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 8);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 4);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 2);
            //     threadValue.y += __shfl_down_sync(0xFFFFFFFF, threadValue.y, 1);
            // }
            // if constexpr (vector_active_size_v<DstT> > 2)
            //{
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 16);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 8);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 4);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 2);
            //     threadValue.z += __shfl_down_sync(0xFFFFFFFF, threadValue.z, 1);
            // }
            // if constexpr (vector_active_size_v<DstT> > 3)
            //{
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 16);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 8);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 4);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 2);
            //     threadValue.w += __shfl_down_sync(0xFFFFFFFF, threadValue.w, 1);
            // }
            //
            // result += threadValue;
        }

        // reduce over warp:
        result.x += __shfl_down_sync(0xFFFFFFFF, result.x, 16);
        result.x += __shfl_down_sync(0xFFFFFFFF, result.x, 8);
        result.x += __shfl_down_sync(0xFFFFFFFF, result.x, 4);
        result.x += __shfl_down_sync(0xFFFFFFFF, result.x, 2);
        result.x += __shfl_down_sync(0xFFFFFFFF, result.x, 1);

        if constexpr (vector_active_size_v<DstT> > 1)
        {
            result.y += __shfl_down_sync(0xFFFFFFFF, result.y, 16);
            result.y += __shfl_down_sync(0xFFFFFFFF, result.y, 8);
            result.y += __shfl_down_sync(0xFFFFFFFF, result.y, 4);
            result.y += __shfl_down_sync(0xFFFFFFFF, result.y, 2);
            result.y += __shfl_down_sync(0xFFFFFFFF, result.y, 1);
        }
        if constexpr (vector_active_size_v<DstT> > 2)
        {
            result.z += __shfl_down_sync(0xFFFFFFFF, result.z, 16);
            result.z += __shfl_down_sync(0xFFFFFFFF, result.z, 8);
            result.z += __shfl_down_sync(0xFFFFFFFF, result.z, 4);
            result.z += __shfl_down_sync(0xFFFFFFFF, result.z, 2);
            result.z += __shfl_down_sync(0xFFFFFFFF, result.z, 1);
        }
        if constexpr (vector_active_size_v<DstT> > 3)
        {
            result.w += __shfl_down_sync(0xFFFFFFFF, result.w, 16);
            result.w += __shfl_down_sync(0xFFFFFFFF, result.w, 8);
            result.w += __shfl_down_sync(0xFFFFFFFF, result.w, 4);
            result.w += __shfl_down_sync(0xFFFFFFFF, result.w, 2);
            result.w += __shfl_down_sync(0xFFFFFFFF, result.w, 1);
        }

        if (warpLaneID == 0)
        {
            aDst[pixelY] = result;
        }
    }
}

template <typename SrcT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, typename funcType>
void InvokeReductionAlongXKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                 const SrcT *aSrc, DstT *aDst, const Size2D &aSize, const funcType &aFunctor)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aSrc, aSize.x, aWarpSize);

    dim3 blocksPerGrid(1, DIV_UP(aSize.y, aBlockSize.y), 1);

    reductionAlongXKernel<WarpAlignmentInBytes, TupelSize, DstT, funcType>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aDst, aSize, ts, aFunctor);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename SrcT, typename DstT, size_t TupelSize, typename funcType>
void InvokeReductionAlongXKernelDefault(const SrcT *aSrc, DstT *aDst, const Size2D &aSize,
                                        const opp::cuda::StreamCtx &aStreamCtx, const funcType &aFunc)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize               = ConfigBlockSize<"Default">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeReductionAlongXKernel<SrcT, DstT, TupelSize, WarpAlignmentInBytes, funcType>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrc, aDst, aSize, aFunc);
    }
    else
    {
        throw CUDAUNSUPPORTED(forEachPixelKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND