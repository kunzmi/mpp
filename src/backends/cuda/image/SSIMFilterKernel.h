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
/// Applies a fixed size separable filter to each pixel in an image with border control.<para/>
/// Each warp of the kernel operates on a small patch (or pixel block) of size (warpSize * TupelSize) x blockHeight.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, int kernelSize, int blockHeight,
          typename BorderControlT, typename FixedFilterKernelT, typename postOp>
__global__ void SSIMFilterKernel(BorderControlT aSrc1WithBC, BorderControlT aSrc2WithBC, DstT *__restrict__ aDst,
                                 size_t aPitchDst, postOp aPostOp, Size2D aSize,
                                 ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    constexpr int myWarpSize = 32; // warpSize itself is not const nor constexpr...
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
        // 5 different results computed per pixel
        ComputeT(*buffer)[5][extendedBlockW] = (ComputeT(*)[5][extendedBlockW])(sharedBuffer);

        // as threads in warp-aligned area are always the full warp, no need to check for X pixel limits here
        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
#pragma unroll
            for (int i = 0; i < extendedBlockW; i += myWarpSize)
            {
                const int offsetX = i + threadIdx.x;

                if (offsetX < extendedBlockW)
                {
                    ComputeT localMean1[blockHeight]       = {0};
                    ComputeT localMean2[blockHeight]       = {0};
                    ComputeT localVar1Sqr[blockHeight]     = {0};
                    ComputeT localVar2Sqr[blockHeight]     = {0};
                    ComputeT localCrossVarSqr[blockHeight] = {0};
#pragma unroll
                    for (int ry = 0; ry < extendedBlockH; ry++)
                    {
                        const int srcPixelY = ry - centerFilter + pixelY;

                        const int srcPixelX = pixelX0 - centerFilter + offsetX;
                        ComputeT pixelSrc1  = ComputeT(aSrc1WithBC(srcPixelX, srcPixelY));
                        ComputeT pixelSrc2  = ComputeT(aSrc2WithBC(srcPixelX, srcPixelY));

#pragma unroll
                        for (int ky = 0; ky < blockHeight; ky++)
                        {
                            const int filterIndex = ry - ky;

                            if (filterIndex >= 0 && filterIndex < kernelSize)
                            {
                                localMean1[ky] += pixelSrc1 * filterKernel.ValuesSeparable[filterIndex];
                                localMean2[ky] += pixelSrc2 * filterKernel.ValuesSeparable[filterIndex];
                                localVar1Sqr[ky] += pixelSrc1 * pixelSrc1 * filterKernel.ValuesSeparable[filterIndex];
                                localVar2Sqr[ky] += pixelSrc2 * pixelSrc2 * filterKernel.ValuesSeparable[filterIndex];
                                localCrossVarSqr[ky] +=
                                    pixelSrc1 * pixelSrc2 * filterKernel.ValuesSeparable[filterIndex];
                            }
                        }
                    }

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        const int idxY = ky + threadIdx.y * blockHeight;

                        buffer[idxY][0][offsetX] = localMean1[ky];
                        buffer[idxY][1][offsetX] = localMean2[ky];
                        buffer[idxY][2][offsetX] = localVar1Sqr[ky];
                        buffer[idxY][3][offsetX] = localVar2Sqr[ky];
                        buffer[idxY][4][offsetX] = localCrossVarSqr[ky];
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
                        ComputeT mean1(0);
                        ComputeT mean2(0);
                        ComputeT var1Sqr(0);
                        ComputeT var2Sqr(0);
                        ComputeT crossVar(0);

#pragma unroll
                        for (int i = 0; i < kernelSize; i++)
                        {
                            const int elementIndex = i + threadIdx.x * TupelSize + t;
                            const int idxY         = bl + threadIdx.y * blockHeight;
                            mean1 += buffer[idxY][0][elementIndex] * filterKernel.ValuesSeparable[i];
                            mean2 += buffer[idxY][1][elementIndex] * filterKernel.ValuesSeparable[i];
                            var1Sqr += buffer[idxY][2][elementIndex] * filterKernel.ValuesSeparable[i];
                            var2Sqr += buffer[idxY][3][elementIndex] * filterKernel.ValuesSeparable[i];
                            crossVar += buffer[idxY][4][elementIndex] * filterKernel.ValuesSeparable[i];
                        }

                        aPostOp(mean1, var1Sqr, mean2, var2Sqr, crossVar, res.value[t]);
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
        // 5 different results computed per pixel
        ComputeT(*buffer)[5][extendedBlockW] = (ComputeT(*)[5][extendedBlockW])(sharedBuffer);

#pragma unroll
        for (int i = 0; i < extendedBlockW; i += myWarpSize)
        {
            const int offsetX = i + threadIdx.x;

            if (offsetX < extendedBlockW)
            {
                ComputeT localMean1[blockHeight]       = {0};
                ComputeT localMean2[blockHeight]       = {0};
                ComputeT localVar1Sqr[blockHeight]     = {0};
                ComputeT localVar2Sqr[blockHeight]     = {0};
                ComputeT localCrossVarSqr[blockHeight] = {0};
#pragma unroll
                for (int ry = 0; ry < extendedBlockH; ry++)
                {
                    const int srcPixelY = ry - centerFilter + pixelY;

                    const int srcPixelX = pixelX0 - centerFilter + offsetX;

                    ComputeT pixelSrc1 = ComputeT(aSrc1WithBC(srcPixelX, srcPixelY));
                    ComputeT pixelSrc2 = ComputeT(aSrc2WithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        const int filterIndex = ry - ky;

                        if (filterIndex >= 0 && filterIndex < kernelSize)
                        {
                            localMean1[ky] += pixelSrc1 * filterKernel.ValuesSeparable[filterIndex];
                            localMean2[ky] += pixelSrc2 * filterKernel.ValuesSeparable[filterIndex];
                            localVar1Sqr[ky] += pixelSrc1 * pixelSrc1 * filterKernel.ValuesSeparable[filterIndex];
                            localVar2Sqr[ky] += pixelSrc2 * pixelSrc2 * filterKernel.ValuesSeparable[filterIndex];
                            localCrossVarSqr[ky] += pixelSrc1 * pixelSrc2 * filterKernel.ValuesSeparable[filterIndex];
                        }
                    }
                }

#pragma unroll
                for (int ky = 0; ky < blockHeight; ky++)
                {
                    const int idxY = ky + threadIdx.y * blockHeight;

                    buffer[idxY][0][offsetX] = localMean1[ky];
                    buffer[idxY][1][offsetX] = localMean2[ky];
                    buffer[idxY][2][offsetX] = localVar1Sqr[ky];
                    buffer[idxY][3][offsetX] = localVar2Sqr[ky];
                    buffer[idxY][4][offsetX] = localCrossVarSqr[ky];
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

                    ComputeT mean1(0);
                    ComputeT mean2(0);
                    ComputeT var1Sqr(0);
                    ComputeT var2Sqr(0);
                    ComputeT crossVar(0);

#pragma unroll
                    for (int i = 0; i < kernelSize; i++)
                    {
                        const int elementIndex = i + threadIdx.x;
                        const int idxY         = bl + threadIdx.y * blockHeight;
                        mean1 += buffer[idxY][0][elementIndex] * filterKernel.ValuesSeparable[i];
                        mean2 += buffer[idxY][1][elementIndex] * filterKernel.ValuesSeparable[i];
                        var1Sqr += buffer[idxY][2][elementIndex] * filterKernel.ValuesSeparable[i];
                        var2Sqr += buffer[idxY][3][elementIndex] * filterKernel.ValuesSeparable[i];
                        crossVar += buffer[idxY][4][elementIndex] * filterKernel.ValuesSeparable[i];
                    }
                    DstT res;

                    aPostOp(mean1, var1Sqr, mean2, var2Sqr, crossVar, res);

                    *pixelsOut = res;
                }
            }
        }

        return;
    }
}

template <typename ComputeT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, int kernelSize, int blockHeight,
          typename BorderControlT, typename FixedFilterKernelT, typename postOp>
void InvokeSSIMFilterKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                            const BorderControlT &aSrc1WithBC, const BorderControlT &aSrc2WithBC, DstT *aDst,
                            size_t aPitchDst, postOp aPostOp, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    SSIMFilterKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, kernelSize, blockHeight, BorderControlT,
                     FixedFilterKernelT, postOp><<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(
        aSrc1WithBC, aSrc2WithBC, aDst, aPitchDst, aPostOp, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, int kernelSize, int blockHeight, typename BorderControlT,
          typename FixedFilterKernelT, typename postOp>
void InvokeSSIMFilterKernelDefault(const BorderControlT &aSrc1WithBC, const BorderControlT &aSrc2WithBC, DstT *aDst,
                                   size_t aPitchDst, postOp aPostOp, const Size2D &aSize,
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
        const uint extendedBlockW          = BlockSize.x * TupelSize + (kernelSize - 1);
        uint SharedMemory                  = sizeof(ComputeT) * 5 * extendedBlockW * blockHeight * BlockSize.y;

        if (SharedMemory > aStreamCtx.SharedMemPerBlock)
        {
            BlockSize.y /= 2; // this now also fits for Pixel64fC4
            SharedMemory = sizeof(ComputeT) * 5 * extendedBlockW * blockHeight * BlockSize.y;
        }

        InvokeSSIMFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, kernelSize, blockHeight, BorderControlT,
                               FixedFilterKernelT, postOp>(BlockSize, SharedMemory, aStreamCtx.WarpSize,
                                                           aStreamCtx.Stream, aSrc1WithBC, aSrc2WithBC, aDst, aPitchDst,
                                                           aPostOp, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(SSIMFilterKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
