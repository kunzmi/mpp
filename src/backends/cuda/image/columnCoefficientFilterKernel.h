#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/borderControl.h>
#include <common/image/gotoPtr.h>
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
/// Applies a column filter to each pixel in an image with border control.<para/>
/// Each warp of the kernel operates on a small patch (or pixel block) of size (warpSize * TupelSize) x blockHeight.
/// First, the area is consecutively read by the warp, filtered in columns
/// and the result is stored in shared memory. Then in a second pass, the patch is rearanged and the
/// result is then stored in the destination image.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, typename FilterT, bool needsScaling>
__global__ void columnCoefficientFilterKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                              const FilterT *__restrict__ aFilter, FilterT aScalingValueInv,
                                              int aFilterSize, int aFilterCenter, Size2D aSize,
                                              ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    constexpr int myWarpSize = 32; // warpSize itself is not const nor constexpr...
    RoundFunctor<roundingMode, ComputeT> round;

    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    // we need the full warp in x direction to do the load...
    if (threadY >= DIV_UP(aSize.y, blockHeight))
    {
        return;
    }

    extern __shared__ int sharedBuffer[];
    ComputeT *buffer = reinterpret_cast<ComputeT *>(sharedBuffer);

    const int pixelX  = aSplit.GetPixel(threadX);
    const int pixelX0 = aSplit.GetPixel(blockIdx.x * blockDim.x);
    const int pixelY  = threadY * blockHeight;

    // don't check for warp alignment if TupelSize <= 1
    if constexpr (TupelSize > 1) // evaluated at compile time
    {
        const int extendedBlockW = myWarpSize * TupelSize;
        const int extendedBlockH = blockHeight + (aFilterSize - 1);

        // as threads in warp-aligned area are always the full warp, no need to check for X pixel limits here
        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            for (int i = 0; i < extendedBlockW; i += myWarpSize)
            {
                const int offsetX = i + threadIdx.x;

                if (offsetX < extendedBlockW)
                {
                    ComputeT localBuffer[blockHeight] = {0};

                    for (int ry = 0; ry < extendedBlockH; ry++)
                    {
                        const int srcPixelY = ry - aFilterCenter + pixelY;

                        const int srcPixelX = pixelX0 + offsetX;
                        ComputeT pixel      = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                        for (int ky = 0; ky < blockHeight; ky++)
                        {
                            const int filterIndex = ry - ky;

                            if (filterIndex >= 0 && filterIndex < aFilterSize)
                            {
                                localBuffer[ky] += pixel * aFilter[filterIndex];
                            }
                        }
                    }

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        buffer[(ky + threadIdx.y * blockHeight) * extendedBlockW + offsetX] = localBuffer[ky];
                    }
                }
            }

            __syncthreads();

            // now we have column filtered values for the entire warp in shared memory

#pragma unroll
            for (int bl = 0; bl < blockHeight; bl++)
            {
                const int pixelYDst = pixelY + bl;

                if (pixelYDst < aSize.y)
                {
                    DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX, pixelYDst);
                    Tupel<DstT, TupelSize> res;

#pragma unroll
                    for (int t = 0; t < TupelSize; t++)
                    {

                        const int elementIndex = threadIdx.x * TupelSize + t;
                        ComputeT temp = buffer[(bl + threadIdx.y * blockHeight) * extendedBlockW + elementIndex];

                        if constexpr (needsScaling)
                        {
                            temp = temp * aScalingValueInv;
                        }

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
                }
            }
            return;
        }
    }

    {

        const int extendedBlockW = myWarpSize;
        const int extendedBlockH = blockHeight + (aFilterSize - 1);

#pragma unroll
        for (int i = 0; i < extendedBlockW; i += myWarpSize)
        {
            const int offsetX = i + threadIdx.x;

            if (offsetX < extendedBlockW)
            {
                ComputeT localBuffer[blockHeight] = {0};
#pragma unroll
                for (int ry = 0; ry < extendedBlockH; ry++)
                {
                    const int srcPixelY = ry - aFilterCenter + pixelY;

                    const int srcPixelX = pixelX0 + offsetX;

                    ComputeT pixel = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        const int filterIndex = ry - ky;

                        if (filterIndex >= 0 && filterIndex < aFilterSize)
                        {
                            localBuffer[ky] += pixel * aFilter[filterIndex];
                        }
                    }
                }

#pragma unroll
                for (int ky = 0; ky < blockHeight; ky++)
                {
                    buffer[(ky + threadIdx.y * blockHeight) * extendedBlockW + offsetX] = localBuffer[ky];
                }
            }
        }

        __syncthreads();

        // now we have column filtered values for the entire warp in shared memory

        // now that the entire warp has done the loading, we need to check for correct X-pixel:
        if (pixelX >= 0 && pixelX < aSize.x)
        {
#pragma unroll
            for (int bl = 0; bl < blockHeight; bl++)
            {
                const int pixelYDst = pixelY + bl;

                if (pixelYDst < aSize.y)
                {
                    DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX, pixelYDst);

                    const int elementIndex = threadIdx.x;
                    ComputeT temp          = buffer[(bl + threadIdx.y * blockHeight) * extendedBlockW + elementIndex];

                    if (needsScaling)
                    {
                        temp = temp * aScalingValueInv;
                    }

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
            }
        }

        return;
    }
}

template <typename ComputeT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, typename FilterT, bool needsScaling>
void InvokeColumnCoefficientFilterKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize,
                                         cudaStream_t aStream, const BorderControlT &aSrcWithBC, DstT *aDst,
                                         size_t aPitchDst, const FilterT *aFilter, FilterT aScalingValueInv,
                                         int aFilterSize, int aFilterCenter, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    columnCoefficientFilterKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, blockHeight, roundingMode,
                                  BorderControlT, FilterT, needsScaling>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aDst, aPitchDst, aFilter, aScalingValueInv,
                                                                aFilterSize, aFilterCenter, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, int blockHeight, RoundingMode roundingMode,
          typename BorderControlT, typename FilterT, bool needsScaling>
void InvokeColumnCoefficientFilterKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                                const FilterT *aFilter, FilterT aScalingValueInv, int aFilterSize,
                                                int aFilterCenter, const Size2D &aSize,
                                                const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        dim3 BlockSize = ConfigBlockSize<"Default">::value;

        if (blockHeight > 1)
        {
            BlockSize.y = 2;
        } /**/

        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        const uint extendedBlockW          = BlockSize.x * TupelSize;
        const uint SharedMemory            = sizeof(ComputeT) * (extendedBlockW)*blockHeight * BlockSize.y;

        InvokeColumnCoefficientFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight, roundingMode,
                                            BorderControlT, FilterT, needsScaling>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst, aFilter,
            aScalingValueInv, aFilterSize, aFilterCenter, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(columnCoefficientFilterKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
