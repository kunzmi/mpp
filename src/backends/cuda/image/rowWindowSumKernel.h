#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/borderControl.h>
#include <common/image/gotoPtr.h>
#include <common/image/pitchException.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{
/// <summary>
/// Applies a row window sum filter to each pixel in an image with border control.<para/>
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, RoundingMode roundingMode,
          typename BorderControlT, typename FilterT>
__global__ void rowWindowSumKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                   FilterT aScalingValue, int aFilterSize, int aFilterCenter, Size2D aSize,
                                   ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    constexpr int myWarpSize = 32; // warpSize itself is not const nor constexpr...
    RoundFunctor<roundingMode, ComputeT> round;

    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    // we need the full warp in x direction to do the load...
    if (threadY >= aSize.y)
    {
        return;
    }

    extern __shared__ int sharedBuffer[];
    ComputeT *buffer = reinterpret_cast<ComputeT *>(sharedBuffer);

    const int pixelX  = aSplit.GetPixel(threadX);
    const int pixelX0 = aSplit.GetPixel(blockIdx.x * blockDim.x);
    const int pixelY  = threadY;

    // don't check for warp alignment if TupelSize <= 1
    if constexpr (TupelSize > 1) // evaluated at compile time
    {
        const int extendedBlockW = myWarpSize * TupelSize + (aFilterSize - 1);
        ComputeT *bufferLine     = buffer + extendedBlockW * threadIdx.y;

        // as threads in warp-aligned area are always the full warp, no need to check for X pixel limits here
        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            // load the full pixel-line for one warp to shared mem in warp-consecutive blocks:
            for (int i = 0; i < extendedBlockW; i += myWarpSize)
            {
                const int offsetX = i + threadIdx.x;

                if (offsetX < extendedBlockW)
                {
                    const int srcPixelY = pixelY;
                    const int srcPixelX = pixelX0 - aFilterCenter + offsetX;

                    bufferLine[offsetX] = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));
                }
            }

            //__syncthreads();

            // now we have values for the entire warp in shared memory
            // --> filter in lines

            ComputeT bufferReg[TupelSize] = {0};
#pragma unroll
            for (int kx = 0; kx < TupelSize; kx++)
            {
                const int idxPixelOut = threadIdx.x + myWarpSize * kx;

                for (int i = 0; i < aFilterSize; i++)
                {
                    const int idx = idxPixelOut + i;

                    bufferReg[kx] += bufferLine[idx];
                }
            }

            //__syncthreads();
            // now the data is filtered in registers, but the tupels are not consecutive, they are warp-interleaved and
            // need to be reshuffled via shared mem where we write them still warp-contionous to avoid bank conflicts
            // during write:
#pragma unroll
            for (int kx = 0; kx < TupelSize; kx++)
            {
                const int idxPixelOut   = threadIdx.x + myWarpSize * kx;
                bufferLine[idxPixelOut] = bufferReg[kx];
            }

            const int pixelYDst = pixelY;

            DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX, pixelYDst);
            Tupel<DstT, TupelSize> res;

#pragma unroll
            for (int t = 0; t < TupelSize; t++)
            {
                // now read again from shared mem, despite some bank conflicts...
                ComputeT temp = bufferLine[threadIdx.x * TupelSize + t];

                temp = temp * aScalingValue;

                if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                {
                    round(temp);
                }

                res.value[t] = DstT(temp);

                // restore alpha channel values:
                if constexpr (has_alpha_channel_v<DstT>)
                {
                    res.value[t].w = (pixelsOut + t)->w;
                }
            }
            Tupel<DstT, TupelSize>::StoreAligned(res, pixelsOut);

            return;
        }
    }

    {

        const int extendedBlockW = myWarpSize + (aFilterSize - 1);
        ComputeT *bufferLine     = buffer + extendedBlockW * threadIdx.y;

#pragma unroll
        for (int i = 0; i < extendedBlockW; i += myWarpSize)
        {
            const int offsetX = i + threadIdx.x;

            if (offsetX < extendedBlockW)
            {
                const int srcPixelY = pixelY;
                const int srcPixelX = pixelX0 - aFilterCenter + offsetX;

                bufferLine[offsetX] = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));
            }
        }

        //__syncthreads();

        // now we have values for the entire warp in shared memory
        // --> filter in lines

        // now that the entire warp has done the loading, we need to check for correct X-pixel:
        if (pixelX >= 0 && pixelX < aSize.x)
        {
            const int pixelYDst = pixelY;

            DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX, pixelYDst);

            ComputeT temp(0);

            for (int i = 0; i < aFilterSize; i++)
            {
                const int elementIndex = i + threadIdx.x;
                const int idxShared    = (threadIdx.y) * extendedBlockW + elementIndex;
                temp += buffer[idxShared];
            }

            temp = temp * aScalingValue;

            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
            {
                round(temp);
            }

            DstT res = DstT(temp);

            // restore alpha channel values:
            if constexpr (has_alpha_channel_v<DstT>)
            {
                res.w = pixelsOut->w;
            }
            *pixelsOut = res;
        }

        return;
    }
}

template <typename ComputeT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, RoundingMode roundingMode,
          typename BorderControlT, typename FilterT>
void InvokeRowWindowSumKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                              const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst, FilterT aScalingValue,
                              int aFilterSize, int aFilterCenter, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y, aBlockSize.y), 1);

    rowWindowSumKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, roundingMode, BorderControlT, FilterT>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aDst, aPitchDst, aScalingValue, aFilterSize,
                                                                aFilterCenter, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, RoundingMode roundingMode, typename BorderControlT,
          typename FilterT>
void InvokeRowWindowSumKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                     FilterT aScalingValue, int aFilterSize, int aFilterCenter, const Size2D &aSize,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        checkPitchIsMultiple(aPitchDst, ConfigWarpAlignment<"Default">::value, TupelSize);

        dim3 BlockSize = ConfigBlockSize<"Default">::value;

        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        const uint extendedBlockW          = BlockSize.x * TupelSize + (aFilterSize - 1);
        const uint SharedMemory            = sizeof(ComputeT) * (extendedBlockW)*BlockSize.y;

        InvokeRowWindowSumKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, roundingMode, BorderControlT,
                                 FilterT>(BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC,
                                          aDst, aPitchDst, aScalingValue, aFilterSize, aFilterCenter, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(rowWindowSumKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
