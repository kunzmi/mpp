#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

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

namespace opp::image::cuda
{
/// <summary>
/// Similar to filterKernel, but adapted slightly for CrossCorrelation
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, typename SrcT, int blockWidth,
          int blockHeight, RoundingMode roundingMode, typename BorderControlT>
__global__ void crossCorrelationKernelSharedMem(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                                const SrcT *__restrict__ aTemplate, size_t aPitchTemplate,
                                                Size2D aTemplateSize, Size2D aSize,
                                                ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    RoundFunctor<roundingMode, ComputeT> round;

    extern __shared__ int sharedBuffer[];
    SrcT *templateBuffer = reinterpret_cast<SrcT *>(sharedBuffer);

    // load template to shared memory (entire block):
    for (int y = 0; y < aTemplateSize.y; y += blockDim.y)
    {
        if (y + threadIdx.y < aTemplateSize.y)
        {
            for (int x = 0; x < aTemplateSize.x; x += blockDim.x)
            {
                if (x + threadIdx.x < aTemplateSize.x)
                {
                    const int idxShared = (y + threadIdx.y) * aTemplateSize.x + x + threadIdx.x;
                    templateBuffer[idxShared] =
                        *gotoPtr(aTemplate, aPitchTemplate, x + (int)threadIdx.x, y + (int)threadIdx.y);
                }
            }
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
        const int extendedBlockW = TupelSize + (aTemplateSize.x - 1);
        const int extendedBlockH = blockHeight + (aTemplateSize.y - 1);

        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            ComputeT result[blockHeight][TupelSize] = {0};

            for (int ry = 0; ry < extendedBlockH; ry++)
            {
                const int srcPixelY = ry - aTemplateSize.y / 2 + pixelY;

                for (int rx = 0; rx < extendedBlockW; rx++)
                {
                    const int srcPixelX = pixelX - aTemplateSize.x / 2 + rx;
                    ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int by = 0; by < blockHeight; by++)
                    {
                        const int tplY = ry - by;
#pragma unroll
                        for (int bx = 0; bx < TupelSize; bx++)
                        {
                            const int tplX = rx - bx;
                            if (tplY >= 0 && tplY < aTemplateSize.y && tplX >= 0 && tplX < aTemplateSize.x)
                            {
                                result[by][bx] += srcPixel * templateBuffer[tplY * aTemplateSize.x + tplX];
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
        const int extendedBlockW                 = blockWidth + (aTemplateSize.x - 1);
        const int extendedBlockH                 = blockHeight + (aTemplateSize.y - 1);
        ComputeT result[blockHeight][blockWidth] = {0};

        for (int ry = 0; ry < extendedBlockH; ry++)
        {
            const int srcPixelY = ry - aTemplateSize.y / 2 + pixelY;

            for (int rx = 0; rx < extendedBlockW; rx++)
            {
                const int srcPixelX = pixelX - aTemplateSize.x / 2 + rx;
                ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                for (int by = 0; by < blockHeight; by++)
                {
                    const int tplY = ry - by;
#pragma unroll
                    for (int bx = 0; bx < blockWidth; bx++)
                    {
                        const int tplX = rx - bx;
                        if (tplY >= 0 && tplY < aTemplateSize.y && tplX >= 0 && tplX < aTemplateSize.x)
                        {
                            result[by][bx] += srcPixel * templateBuffer[tplY * aTemplateSize.x + tplX];
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
/// The template is directly read from global memory. The code is duplicated, otherwise the compiler loses track of
/// __restrict__ and doesn't use LDG.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, typename SrcT, int blockWidth,
          int blockHeight, RoundingMode roundingMode, typename BorderControlT>
__global__ void crossCorrelationKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                       const SrcT *__restrict__ aTemplate, size_t aPitchTemplate, Size2D aTemplateSize,
                                       Size2D aSize, ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
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
        const int extendedBlockW = TupelSize + (aTemplateSize.x - 1);
        const int extendedBlockH = blockHeight + (aTemplateSize.y - 1);

        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            ComputeT result[blockHeight][TupelSize] = {0};

            for (int ry = 0; ry < extendedBlockH; ry++)
            {
                const int srcPixelY = ry - aTemplateSize.y / 2 + pixelY;

                for (int rx = 0; rx < extendedBlockW; rx++)
                {
                    const int srcPixelX = pixelX - aTemplateSize.x / 2 + rx;
                    ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int by = 0; by < blockHeight; by++)
                    {
                        const int tplY = ry - by;
#pragma unroll
                        for (int bx = 0; bx < TupelSize; bx++)
                        {
                            const int tplX = rx - bx;
                            if (tplY >= 0 && tplY < aTemplateSize.y && tplX >= 0 && tplX < aTemplateSize.x)
                            {
                                auto tpl = *gotoPtr(aTemplate, aPitchTemplate, tplX, tplY);
                                result[by][bx] += srcPixel * tpl;
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
        const int extendedBlockW                 = blockWidth + (aTemplateSize.x - 1);
        const int extendedBlockH                 = blockHeight + (aTemplateSize.y - 1);
        ComputeT result[blockHeight][blockWidth] = {0};

        for (int ry = 0; ry < extendedBlockH; ry++)
        {
            const int srcPixelY = ry - aTemplateSize.y / 2 + pixelY;

            for (int rx = 0; rx < extendedBlockW; rx++)
            {
                const int srcPixelX = pixelX - aTemplateSize.x / 2 + rx;
                ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                for (int by = 0; by < blockHeight; by++)
                {
                    const int tplY = ry - by;
#pragma unroll
                    for (int bx = 0; bx < blockWidth; bx++)
                    {
                        const int tplX = rx - bx;
                        if (tplY >= 0 && tplY < aTemplateSize.y && tplX >= 0 && tplX < aTemplateSize.x)
                        {
                            auto tpl = *gotoPtr(aTemplate, aPitchTemplate, tplX, tplY);
                            result[by][bx] += srcPixel * tpl;
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

template <typename ComputeT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, typename SrcT, int blockWidth,
          int blockHeight, RoundingMode roundingMode, typename BorderControlT>
void InvokeCrossCorrelationKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                  const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst, const SrcT *aTemplate,
                                  size_t aPitchTemplate, const Size2D &aTemplateSize, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total() / blockWidth, aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    if (aSharedMemory == 0)
    {
        crossCorrelationKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, SrcT, blockWidth, blockHeight,
                               roundingMode, BorderControlT><<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(
            aSrcWithBC, aDst, aPitchDst, aTemplate, aPitchTemplate, aTemplateSize, aSize, ts);
    }
    else
    {
        crossCorrelationKernelSharedMem<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, SrcT, blockWidth, blockHeight,
                                        roundingMode, BorderControlT>
            <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aDst, aPitchDst, aTemplate,
                                                                    aPitchTemplate, aTemplateSize, aSize, ts);
    }

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, typename SrcT, int blockWidth, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT>
void InvokeCrossCorrelationKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                         const SrcT *aTemplate, size_t aPitchTemplate, const Size2D &aTemplateSize,
                                         const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        dim3 BlockSize = ConfigBlockSize<"Default">::value;

        if (blockHeight > 1)
        {
            BlockSize.y = 2;
        }

        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        uint SharedMemory                  = sizeof(SrcT) * aTemplateSize.TotalSize();
        if (SharedMemory > aStreamCtx.SharedMemPerBlock)
        {
            SharedMemory = 0; // will switch to kernel without shared memory
        }

        InvokeCrossCorrelationKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, SrcT, blockWidth, blockHeight,
                                     roundingMode, BorderControlT>(BlockSize, SharedMemory, aStreamCtx.WarpSize,
                                                                   aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                                                                   aTemplate, aPitchTemplate, aTemplateSize, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(crossCorrelationKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND