#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/morphology/morphologyComputeT.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/borderControl.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/numeric_limits.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace opp::image::cuda
{
/// <summary>
/// Applies a morphology dilation and erosion operation to each pixel in an image with border control in a neighborhood
/// where the mask is non-zero. The result is the difference of both value.<para/>
/// Each thread of the kernel operates on a small patch (or pixel block) of size blockWidth x blockHeight: Line by line
/// (block width + filter padding), all source pixels are read from global memory into registers. Then, for each line
/// read, the pixel values are multiplied with the corresponding filter weight and added to the image patch (again in
/// registers) at the corresponding coordinates. This approach reduces the number of read accesses to global memory with
/// a tradeoff for register consumption. For tupel-ed data, the tupel size equals the pixelblock width in aligned image
/// areas, the blockWidth-parameter must be set to 1 for the non-aligned areas. For non-tupled data, the
/// blockWidth-parameter can be freely set.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class DstT, typename FilterT, int kernelWidth, int kernelHeight,
          int blockWidth, int blockHeight, typename BorderControlT>
__global__ void fixedSizeMorphologyGradientKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                                  const FilterT *__restrict__ aFilter, Vector2<int> aFilterCenter,
                                                  Size2D aSize, ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    using ComputeT = same_vector_size_different_type_t<DstT, remove_vector_t<morph_compute_type_t<DstT>>>;
    extern __shared__ int sharedBuffer[];
    FilterT *filterBuffer      = reinterpret_cast<FilterT *>(sharedBuffer);
    constexpr int filterLength = kernelWidth * kernelHeight;

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
        constexpr int extendedBlockW = TupelSize + (kernelWidth - 1);
        constexpr int extendedBlockH = blockHeight + (kernelHeight - 1);

        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            DstT resultDilate[blockHeight][TupelSize];
            DstT resultErode[blockHeight][TupelSize];

#pragma unroll
            for (int ry = 0; ry < blockHeight; ry++)
            {
#pragma unroll
                for (int rx = 0; rx < TupelSize; rx++)
                {
                    resultDilate[ry][rx] = DstT(numeric_limits<remove_vector_t<DstT>>::min());
                    resultErode[ry][rx]  = DstT(numeric_limits<remove_vector_t<DstT>>::max());
                }
            }

#pragma unroll
            for (int ry = 0; ry < extendedBlockH; ry++)
            {
                const int srcPixelY = ry - aFilterCenter.y + pixelY;

#pragma unroll
                for (int rx = 0; rx < extendedBlockW; rx++)
                {
                    const int srcPixelX = pixelX - aFilterCenter.x + rx;
                    DstT srcPixel       = aSrcWithBC(srcPixelX, srcPixelY);

#pragma unroll
                    for (int ky = 0; ky < kernelHeight; ky++)
                    {
                        const int pixelDstY = ry + ky - kernelHeight + 1;

#pragma unroll
                        for (int kx = 0; kx < kernelWidth; kx++)
                        {
                            const int pixelDstX = rx + kx - kernelWidth + 1;

                            if (pixelDstY >= 0 && pixelDstY < blockHeight)
                            {
                                if (pixelDstX >= 0 && pixelDstX < TupelSize)
                                {
                                    if (filterBuffer[ky * kernelWidth + kx] > 0)
                                    {
                                        resultDilate[pixelDstY][pixelDstX].Max(srcPixel);
                                        resultErode[pixelDstY][pixelDstX].Min(srcPixel);
                                    }
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
                        ComputeT erosion  = ComputeT(resultErode[bl][t]);
                        ComputeT dilation = ComputeT(resultDilate[bl][t]);

                        res.value[t] = DstT(dilation - erosion);
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
        constexpr int extendedBlockW = blockWidth + (kernelWidth - 1);
        constexpr int extendedBlockH = blockHeight + (kernelHeight - 1);
        DstT resultDilate[blockHeight][blockWidth];
        DstT resultErode[blockHeight][blockWidth];

#pragma unroll
        for (int ry = 0; ry < blockHeight; ry++)
        {
#pragma unroll
            for (int rx = 0; rx < blockWidth; rx++)
            {
                resultDilate[ry][rx] = DstT(numeric_limits<remove_vector_t<DstT>>::min());
                resultErode[ry][rx]  = DstT(numeric_limits<remove_vector_t<DstT>>::max());
            }
        }

#pragma unroll
        for (int ry = 0; ry < extendedBlockH; ry++)
        {
            const int srcPixelY = ry - aFilterCenter.y + pixelY;

#pragma unroll
            for (int rx = 0; rx < extendedBlockW; rx++)
            {
                const int srcPixelX = pixelX - aFilterCenter.x + rx;
                DstT srcPixel       = aSrcWithBC(srcPixelX, srcPixelY);
#pragma unroll
                for (int ky = 0; ky < kernelHeight; ky++)
                {
                    const int pixelDstY = ry + ky - kernelHeight + 1;

#pragma unroll
                    for (int kx = 0; kx < kernelWidth; kx++)
                    {
                        const int pixelDstX = rx + kx - kernelWidth + 1;

                        if (pixelDstY >= 0 && pixelDstY < blockHeight)
                        {
                            if (pixelDstX >= 0 && pixelDstX < blockWidth)
                            {
                                if (filterBuffer[ky * kernelWidth + kx] > 0)
                                {
                                    resultDilate[pixelDstY][pixelDstX].Max(srcPixel);
                                    resultErode[pixelDstY][pixelDstX].Min(srcPixel);
                                }
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
                        ComputeT erosion  = ComputeT(resultErode[bl][bc]);
                        ComputeT dilation = ComputeT(resultDilate[bl][bc]);

                        res = DstT(dilation - erosion);
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

template <typename DstT, size_t TupelSize, int WarpAlignmentInBytes, typename FilterT, int kernelWidth,
          int kernelHeight, int blockWidth, int blockHeight, typename BorderControlT>
void InvokeFixedSizeMorphologyGradientKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize,
                                             cudaStream_t aStream, const BorderControlT &aSrcWithBC, DstT *aDst,
                                             size_t aPitchDst, const FilterT *aFilter,
                                             const Vector2<int> &aFilterCenter, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total() / blockWidth, aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    fixedSizeMorphologyGradientKernel<WarpAlignmentInBytes, TupelSize, DstT, FilterT, kernelWidth, kernelHeight,
                                      blockWidth, blockHeight, BorderControlT>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aDst, aPitchDst, aFilter, aFilterCenter,
                                                                aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename DstT, size_t TupelSize, typename FilterT, int kernelWidth, int kernelHeight, int blockWidth,
          int blockHeight, typename BorderControlT>
void InvokeFixedSizeMorphologyGradientKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                                    const FilterT *aFilter, const Vector2<int> &aFilterCenter,
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
        constexpr uint SharedMemory        = sizeof(FilterT) * kernelWidth * kernelHeight;

        InvokeFixedSizeMorphologyGradientKernel<DstT, TupelSize, WarpAlignmentInBytes, FilterT, kernelWidth,
                                                kernelHeight, blockWidth, blockHeight, BorderControlT>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst, aFilter,
            aFilterCenter, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(fixedSizeMorphologyGradientKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND