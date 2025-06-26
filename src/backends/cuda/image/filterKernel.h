#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/filterArea.h>
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
/// Applies a filter to each pixel in an image with border control.<para/>
/// Each thread of the kernel operates on a small patch (or pixel block) of size blockWidth x blockHeight: Line by line
/// (block width + filter padding), all source pixels are read from global memory into registers. Then, for each line
/// read, the pixel values are multiplied with the corresponding filter weight and added to the image patch (again in
/// registers) at the corresponding coordinates. This approach reduces the number of read accesses to global memory with
/// a tradeoff for register consumption. For tupel-ed data, the tupel size equals the pixelblock width in aligned image
/// areas, the blockWidth-parameter must be set to 1 for the non-aligned areas. For non-tupled data, the
/// blockWidth-parameter can be freely set.<para/>
/// The filter is copied to shared memory before any processing.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, typename FilterT, int blockWidth,
          int blockHeight, RoundingMode roundingMode, typename BorderControlT>
__global__ void filterKernelSharedMem(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                      const FilterT *__restrict__ aFilter, FilterArea aFilterArea, Size2D aSize,
                                      ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    RoundFunctor<roundingMode, ComputeT> round;

    extern __shared__ int sharedBuffer[];
    FilterT *filterBuffer  = reinterpret_cast<FilterT *>(sharedBuffer);
    const int filterLength = aFilterArea.Size.x * aFilterArea.Size.y;

    // load filter to shared memory (entire block):
    for (int i = 0; i < filterLength; i += blockDim.x * blockDim.y)
    {
        int idx = i + threadIdx.x + threadIdx.y * blockDim.x;
        if (idx < filterLength)
        {
            filterBuffer[idx] = aFilter[idx];
        }
    }
    __syncthreads();

    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    // in case of tuples, blockWidth == 1, and ThreadSplit handles the thread limits:
    if constexpr (TupelSize > 1)
    {
        if (aSplit.ThreadIsOutsideOfRange(threadX) || threadY >= DIV_UP(aSize.y, blockHeight))
        {
            return;
        }
    }
    else
    {
        if (threadX >= DIV_UP(aSize.x, blockWidth) || threadY >= DIV_UP(aSize.y, blockHeight))
        {
            return;
        }
    }

    const int pixelX = aSplit.GetPixel(threadX) * blockWidth;
    const int pixelY = threadY * blockHeight;

    // don't check for warp alignment if TupelSize <= 1
    if constexpr (TupelSize > 1) // evaluated at compile time
    {
        const int extendedBlockW = TupelSize + (aFilterArea.Size.x - 1);
        const int extendedBlockH = blockHeight + (aFilterArea.Size.y - 1);

        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            ComputeT result[blockHeight][TupelSize] = {0};

            for (int ry = 0; ry < extendedBlockH; ry++)
            {
                const int srcPixelY = ry - aFilterArea.Center.y + pixelY;

                for (int rx = 0; rx < extendedBlockW; rx++)
                {
                    const int srcPixelX = pixelX - aFilterArea.Center.x + rx;
                    ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int by = 0; by < blockHeight; by++)
                    {
                        const int filterY = ry - by;
#pragma unroll
                        for (int bx = 0; bx < TupelSize; bx++)
                        {
                            const int filterX = rx - bx;
                            if (filterY >= 0 && filterY < aFilterArea.Size.y && filterX >= 0 &&
                                filterX < aFilterArea.Size.x)
                            {
                                result[by][bx] += srcPixel * filterBuffer[filterY * aFilterArea.Size.x + filterX];
                            }
                        }
                    }
                }
            }

#pragma unroll
            for (int bl = 0; bl < blockHeight; bl++)
            {
                if (pixelY + bl < aSize.y)
                {
                    DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX, pixelY + bl);
                    Tupel<DstT, TupelSize> res;

#pragma unroll
                    for (size_t t = 0; t < TupelSize; t++)
                    {
                        if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                        {
                            round(result[bl][t]);
                        }

                        res.value[t] = DstT(result[bl][t]);
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
        const int extendedBlockW                 = blockWidth + (aFilterArea.Size.x - 1);
        const int extendedBlockH                 = blockHeight + (aFilterArea.Size.y - 1);
        ComputeT result[blockHeight][blockWidth] = {0};

        for (int ry = 0; ry < extendedBlockH; ry++)
        {
            const int srcPixelY = ry - aFilterArea.Center.y + pixelY;

            for (int rx = 0; rx < extendedBlockW; rx++)
            {
                const int srcPixelX = pixelX - aFilterArea.Center.x + rx;
                ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                for (int by = 0; by < blockHeight; by++)
                {
                    const int filterY = ry - by;
#pragma unroll
                    for (int bx = 0; bx < blockWidth; bx++)
                    {
                        const int filterX = rx - bx;
                        if (filterY >= 0 && filterY < aFilterArea.Size.y && filterX >= 0 &&
                            filterX < aFilterArea.Size.x)
                        {
                            result[by][bx] += srcPixel * filterBuffer[filterY * aFilterArea.Size.x + filterX];
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int bl = 0; bl < blockHeight; bl++)
        {
            if (pixelY + bl < aSize.y)
            {
#pragma unroll
                for (int bc = 0; bc < blockWidth; bc++)
                {
                    if (pixelX + bc < aSize.x)
                    {
                        DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX + bc, pixelY + bl);
                        DstT res;

                        if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                        {
                            round(result[bl][bc]);
                        }

                        res = DstT(result[bl][bc]);

                        // restore alpha channel values:
                        if constexpr (has_alpha_channel_v<DstT>)
                        {
                            res.w = pixelsOut->w;
                        }
                        *pixelsOut = res;
                    }
                }
            }
        }
    }
    return;
}

/// <summary>
/// Applies a filter to each pixel in an image with border control.<para/>
/// Each thread of the kernel operates on a small patch (or pixel block) of size blockWidth x blockHeight: Line by line
/// (block width + filter padding), all source pixels are read from global memory into registers. Then, for each line
/// read, the pixel values are multiplied with the corresponding filter weight and added to the image patch (again in
/// registers) at the corresponding coordinates. This approach reduces the number of read accesses to global memory with
/// a tradeoff for register consumption. For tupel-ed data, the tupel size equals the pixelblock width in aligned image
/// areas, the blockWidth-parameter must be set to 1 for the non-aligned areas. For non-tupled data, the
/// blockWidth-parameter can be freely set.<para/>
/// The filter is directly read from global memory. The code is duplicated, otherwise the compiler loses track of
/// __restrict__ and doesn't use LDG.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, typename FilterT, int blockWidth,
          int blockHeight, RoundingMode roundingMode, typename BorderControlT>
__global__ void filterKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                             const FilterT *__restrict__ aFilter, FilterArea aFilterArea, Size2D aSize,
                             ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    RoundFunctor<roundingMode, ComputeT> round;

    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    // in case of tuples, blockWidth == 1, and ThreadSplit handles the thread limits:
    if constexpr (TupelSize > 1)
    {
        if (aSplit.ThreadIsOutsideOfRange(threadX) || threadY >= DIV_UP(aSize.y, blockHeight))
        {
            return;
        }
    }
    else
    {
        if (threadX >= DIV_UP(aSize.x, blockWidth) || threadY >= DIV_UP(aSize.y, blockHeight))
        {
            return;
        }
    }

    const int pixelX = aSplit.GetPixel(threadX) * blockWidth;
    const int pixelY = threadY * blockHeight;

    // don't check for warp alignment if TupelSize <= 1
    if constexpr (TupelSize > 1) // evaluated at compile time
    {
        const int extendedBlockW = TupelSize + (aFilterArea.Size.x - 1);
        const int extendedBlockH = blockHeight + (aFilterArea.Size.y - 1);

        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            ComputeT result[blockHeight][TupelSize] = {0};

            for (int ry = 0; ry < extendedBlockH; ry++)
            {
                const int srcPixelY = ry - aFilterArea.Center.y + pixelY;

                for (int rx = 0; rx < extendedBlockW; rx++)
                {
                    const int srcPixelX = pixelX - aFilterArea.Center.x + rx;
                    ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int by = 0; by < blockHeight; by++)
                    {
                        const int filterY = ry - by;
#pragma unroll
                        for (int bx = 0; bx < TupelSize; bx++)
                        {
                            const int filterX = rx - bx;
                            if (filterY >= 0 && filterY < aFilterArea.Size.y && filterX >= 0 &&
                                filterX < aFilterArea.Size.x)
                            {
                                result[by][bx] += srcPixel * aFilter[filterY * aFilterArea.Size.x + filterX];
                            }
                        }
                    }
                }
            }

#pragma unroll
            for (int bl = 0; bl < blockHeight; bl++)
            {
                if (pixelY + bl < aSize.y)
                {
                    DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX, pixelY + bl);
                    Tupel<DstT, TupelSize> res;

#pragma unroll
                    for (size_t t = 0; t < TupelSize; t++)
                    {
                        if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                        {
                            round(result[bl][t]);
                        }

                        res.value[t] = DstT(result[bl][t]);
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
        const int extendedBlockW                 = blockWidth + (aFilterArea.Size.x - 1);
        const int extendedBlockH                 = blockHeight + (aFilterArea.Size.y - 1);
        ComputeT result[blockHeight][blockWidth] = {0};

        for (int ry = 0; ry < extendedBlockH; ry++)
        {
            const int srcPixelY = ry - aFilterArea.Center.y + pixelY;

            for (int rx = 0; rx < extendedBlockW; rx++)
            {
                const int srcPixelX = pixelX - aFilterArea.Center.x + rx;
                ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                for (int by = 0; by < blockHeight; by++)
                {
                    const int filterY = ry - by;
#pragma unroll
                    for (int bx = 0; bx < blockWidth; bx++)
                    {
                        const int filterX = rx - bx;
                        if (filterY >= 0 && filterY < aFilterArea.Size.y && filterX >= 0 &&
                            filterX < aFilterArea.Size.x)
                        {
                            result[by][bx] += srcPixel * aFilter[filterY * aFilterArea.Size.x + filterX];
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int bl = 0; bl < blockHeight; bl++)
        {
            if (pixelY + bl < aSize.y)
            {
#pragma unroll
                for (int bc = 0; bc < blockWidth; bc++)
                {
                    if (pixelX + bc < aSize.x)
                    {
                        DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX + bc, pixelY + bl);
                        DstT res;

                        if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                        {
                            round(result[bl][bc]);
                        }

                        res = DstT(result[bl][bc]);

                        // restore alpha channel values:
                        if constexpr (has_alpha_channel_v<DstT>)
                        {
                            res.w = pixelsOut->w;
                        }
                        *pixelsOut = res;
                    }
                }
            }
        }
    }
    return;
}

template <typename ComputeT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, typename FilterT,
          int blockWidth, int blockHeight, RoundingMode roundingMode, typename BorderControlT>
void InvokeFilterKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                        const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst, const FilterT *aFilter,
                        const FilterArea &aFilterArea, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total() / blockWidth, aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    if (aSharedMemory == 0)
    {
        filterKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, FilterT, blockWidth, blockHeight, roundingMode,
                     BorderControlT><<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(
            aSrcWithBC, aDst, aPitchDst, aFilter, aFilterArea, aSize, ts);
    }
    else
    {
        filterKernelSharedMem<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, FilterT, blockWidth, blockHeight,
                              roundingMode, BorderControlT><<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(
            aSrcWithBC, aDst, aPitchDst, aFilter, aFilterArea, aSize, ts);
    }

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, typename FilterT, int blockWidth, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT>
void InvokeFilterKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst, const FilterT *aFilter,
                               const FilterArea &aFilterArea, const Size2D &aSize,
                               const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        dim3 BlockSize = ConfigBlockSize<"Default">::value;

        if (blockHeight > 1)
        {
            BlockSize.y = 2;
        }

        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        uint SharedMemory                  = sizeof(FilterT) * aFilterArea.Size.TotalSize();
        if (SharedMemory > aStreamCtx.SharedMemPerBlock)
        {
            SharedMemory = 0; // will switch to kernel without shared memory
        }

        InvokeFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, FilterT, blockWidth, blockHeight,
                           roundingMode, BorderControlT>(BlockSize, SharedMemory, aStreamCtx.WarpSize,
                                                         aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst, aFilter,
                                                         aFilterArea, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(filterKernel, "Trying to execute on a platform with an unsupported compute capability: "
                                                << aStreamCtx.ComputeCapabilityMajor << "."
                                                << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND