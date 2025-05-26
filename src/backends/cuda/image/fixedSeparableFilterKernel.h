#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

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

namespace opp::image::cuda
{
/// <summary>
/// Applies a fixed size separable filter to each pixel in an image with border control.<para/>
/// Each warp of the kernel operates on a small patch (or pixel block) of size (warpSize * TupelSize) x blockHeight.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, int kernelSize, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, typename FixedFilterKernelT>
__global__ void fixedSeparableFilterKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                           Size2D aSize, ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    constexpr int myWarpSize = 32; // warpSize itself is not const nor constexpr...
    RoundFunctor<roundingMode, ComputeT> round;
    constexpr FixedFilterKernelT filterKernel;

    constexpr int centerFilter = kernelSize / 2;

    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    // we need the full warp in x direction to do the load...
    if (threadY >= DIV_UP(aSize.y, blockHeight))
    {
        return;
    }

    extern __shared__ int sharedBuffer[];

    const int pixelX  = aSplit.GetPixel(threadX);
    const int pixelX0 = aSplit.GetPixel(blockIdx.x * blockDim.x);
    const int pixelY  = threadY * blockHeight;

    // don't check for warp alignment if TupelSize <= 1
    if constexpr (TupelSize > 1) // evaluated at compile time
    {
        constexpr int extendedBlockW = myWarpSize * TupelSize + (kernelSize - 1);
        constexpr int extendedBlockH = blockHeight + (kernelSize - 1);

        ComputeT(*buffer)[extendedBlockW] = (ComputeT(*)[extendedBlockW])(sharedBuffer);

        // as threads in warp-aligned area are always the full warp, no need to check for X pixel limits here
        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
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
                        const int srcPixelY = ry - centerFilter + pixelY;

                        const int srcPixelX = pixelX0 - centerFilter + offsetX;
                        ComputeT pixel      = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                        for (int ky = 0; ky < blockHeight; ky++)
                        {
                            const int filterIndex = ry - ky;

                            if (filterIndex >= 0 && filterIndex < kernelSize)
                            {
                                localBuffer[ky] += pixel * filterKernel.ValuesSeparable[filterIndex];
                            }
                        }
                    }

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        buffer[ky + threadIdx.y * blockHeight][offsetX] = localBuffer[ky];
                    }
                }
            }

            __syncthreads();

            // now we have column filtered values for the entire warp in shared memory
            // --> filter in lines

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
                        ComputeT temp(0);

#pragma unroll
                        for (int i = 0; i < kernelSize; i++)
                        {
                            const int elementIndex = i + threadIdx.x * TupelSize + t;
                            temp +=
                                buffer[bl + threadIdx.y * blockHeight][elementIndex] * filterKernel.ValuesSeparable[i];
                        }

                        if (filterKernel.NeedsScaling)
                        {
                            temp = temp / filterKernel.Scaling;
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

        constexpr int extendedBlockW = myWarpSize + (kernelSize - 1);
        constexpr int extendedBlockH = blockHeight + (kernelSize - 1);

        ComputeT(*buffer)[extendedBlockW] = (ComputeT(*)[extendedBlockW])(sharedBuffer);

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
                    const int srcPixelY = ry - centerFilter + pixelY;

                    const int srcPixelX = pixelX0 - centerFilter + offsetX;

                    ComputeT pixel = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        const int filterIndex = ry - ky;

                        if (filterIndex >= 0 && filterIndex < kernelSize)
                        {
                            localBuffer[ky] += pixel * filterKernel.ValuesSeparable[filterIndex];
                        }
                    }
                }

#pragma unroll
                for (int ky = 0; ky < blockHeight; ky++)
                {
                    buffer[ky + threadIdx.y * blockHeight][offsetX] = localBuffer[ky];
                }
            }
        }

        __syncthreads();

        // now we have column filtered values for the entire warp in shared memory
        // --> filter in lines

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

                    ComputeT temp(0);

#pragma unroll
                    for (int i = 0; i < kernelSize; i++)
                    {
                        const int elementIndex = i + threadIdx.x;
                        temp += buffer[bl + threadIdx.y * blockHeight][elementIndex] * filterKernel.ValuesSeparable[i];
                    }

                    if (filterKernel.NeedsScaling)
                    {
                        temp = temp / filterKernel.Scaling;
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

template <typename ComputeT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, int kernelSize, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, typename FixedFilterKernelT>
void InvokeFixedSeparableFilterKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                      const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                      const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    fixedSeparableFilterKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, kernelSize, blockHeight, roundingMode,
                               BorderControlT, FixedFilterKernelT>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aDst, aPitchDst, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, int kernelSize, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, typename FixedFilterKernelT>
void InvokeFixedSeparableFilterKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                             const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        dim3 BlockSize = ConfigBlockSize<"Default">::value;

        if (blockHeight > 1)
        {
            BlockSize.y = 2;
        } /**/

        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        const uint extendedBlockW          = BlockSize.x * TupelSize + (kernelSize - 1);
        const uint SharedMemory            = sizeof(ComputeT) * (extendedBlockW)*blockHeight * BlockSize.y;

        InvokeFixedSeparableFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, kernelSize, blockHeight,
                                         roundingMode, BorderControlT, FixedFilterKernelT>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(fixedSeparableFilterKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND