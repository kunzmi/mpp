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
/// Applies a fixed size filter to each pixel in an image with border control.<para/>
/// Each thread of the kernel operates on a small patch (or pixel block) of size blockWidth x blockHeight: Line by line
/// (block width + filter padding), all source pixels are read from global memory into registers. Then, for each line
/// read, the pixel values are multiplied with the corresponding filter weight and added to the image patch (again in
/// registers) at the corresponding coordinates. This approach reduces the number of read accesses to global memory with
/// a tradeoff for register consumption. For tupel-ed data, the tupel size equals the pixelblock width in aligned image
/// areas, the blockWidth-parameter must be set to 1 for the non-aligned areas. For non-tupled data, the
/// blockWidth-parameter can be freely set.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, int kernelWidth, int kernelHeight,
          int kernelCenterX, int kernelCenterY, int blockWidth, int blockHeight, RoundingMode roundingMode,
          typename BorderControlT, typename FixedFilterKernelT>
__global__ void fixedFilterKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst, Size2D aSize,
                                  ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    RoundFunctor<roundingMode, ComputeT> round;
    constexpr FixedFilterKernelT filterKernel;

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
        constexpr int extendedBlockW = TupelSize + (kernelWidth - 1);
        constexpr int extendedBlockH = blockHeight + (kernelHeight - 1);

        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            ComputeT result[blockHeight][TupelSize] = {0};

#pragma unroll
            for (int ry = 0; ry < extendedBlockH; ry++)
            {
                const int srcPixelY = ry - kernelCenterY + pixelY;

#pragma unroll
                for (int rx = 0; rx < extendedBlockW; rx++)
                {
                    const int srcPixelX = pixelX - kernelCenterX + rx;
                    ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int ky = 0; ky < kernelHeight; ky++)
                    {
                        const int pixelDstY = ry - ky;

#pragma unroll
                        for (int kx = 0; kx < kernelWidth; kx++)
                        {
                            const int pixelDstX = rx - kx;

                            if (pixelDstY >= 0 && pixelDstY < blockHeight)
                            {
                                if (pixelDstX >= 0 && pixelDstX < TupelSize)
                                {
                                    result[pixelDstY][pixelDstX] += srcPixel * filterKernel.Values[ky][kx];
                                }
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
                        if (filterKernel.NeedsScaling)
                        {
                            result[bl][t] = result[bl][t] / filterKernel.Scaling;
                        }

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
        constexpr int extendedBlockW             = blockWidth + (kernelWidth - 1);
        constexpr int extendedBlockH             = blockHeight + (kernelHeight - 1);
        ComputeT result[blockHeight][blockWidth] = {0};

#pragma unroll
        for (int ry = 0; ry < extendedBlockH; ry++)
        {
            const int srcPixelY = ry - kernelCenterY + pixelY;

#pragma unroll
            for (int rx = 0; rx < extendedBlockW; rx++)
            {
                const int srcPixelX = pixelX - kernelCenterX + rx;
                ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));
#pragma unroll
                for (int ky = 0; ky < kernelHeight; ky++)
                {
                    const int pixelDstY = ry - ky;

#pragma unroll
                    for (int kx = 0; kx < kernelWidth; kx++)
                    {
                        const int pixelDstX = rx - kx;

                        if (pixelDstY >= 0 && pixelDstY < blockHeight)
                        {
                            if (pixelDstX >= 0 && pixelDstX < blockWidth)
                            {
                                result[pixelDstY][pixelDstX] += srcPixel * filterKernel.Values[ky][kx];
                            }
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

                        if (filterKernel.NeedsScaling)
                        {
                            result[bl][bc] = result[bl][bc] / filterKernel.Scaling;
                        }

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

template <typename ComputeT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, int kernelWidth,
          int kernelHeight, int kernelCenterX, int kernelCenterY, int blockWidth, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, typename FixedFilterKernelT>
void InvokeFixedFilterKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                             const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total() / blockWidth, aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    fixedFilterKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, kernelWidth, kernelHeight, kernelCenterX,
                      kernelCenterY, blockWidth, blockHeight, roundingMode, BorderControlT, FixedFilterKernelT>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aDst, aPitchDst, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, int kernelWidth, int kernelHeight, int kernelCenterX,
          int kernelCenterY, int blockWidth, int blockHeight, RoundingMode roundingMode, typename BorderControlT,
          typename FixedFilterKernelT>
void InvokeFixedFilterKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
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
        constexpr uint SharedMemory        = 0;

        InvokeFixedFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, kernelWidth, kernelHeight,
                                kernelCenterX, kernelCenterY, blockWidth, blockHeight, roundingMode, BorderControlT,
                                FixedFilterKernelT>(BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream,
                                                    aSrcWithBC, aDst, aPitchDst, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(fixedFilterKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
