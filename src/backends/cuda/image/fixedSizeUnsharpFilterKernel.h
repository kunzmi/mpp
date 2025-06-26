#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

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
/// Applies a fixed size separable filter to each pixel in an image with border control and then merges the original
/// image pixel with the filtered pixel according to Result = Original + aWeight * (Original - filter(Orignal)) *
/// (filter(Original) &lt;= aThreshold).<para/>
/// Each warp of the kernel operates on a small patch (or pixel block) of size (warpSize * TupelSize) x blockHeight.
/// First, the width-extended area (patch size + filter size) is consecutively read by the warp, filtered in columns
/// and the result is stored in shared memory. Then in a second pass, the extended patch is filtered in rows and the
/// result is then stored in the destination image.
/// The layout in shared memory leads to bank-conflicts when reading the pixels, but I haven't found any more performant
/// layout yet...
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, typename FilterT, int filterSize>
__global__ void fixedSizeUnsharpFilterKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                             const FilterT *__restrict__ aFilter, FilterT aWeight, FilterT aThreshold,
                                             int aFilterCenter, Size2D aSize,
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
        constexpr int extendedBlockW = myWarpSize * TupelSize + (filterSize - 1);
        constexpr int extendedBlockH = blockHeight + (filterSize - 1);

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
                        const int srcPixelY = ry - aFilterCenter + pixelY;

                        const int srcPixelX = pixelX0 - aFilterCenter + offsetX;
                        ComputeT pixel      = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                        for (int ky = 0; ky < blockHeight; ky++)
                        {
                            const int filterIndex = ry - ky;

                            if (filterIndex >= 0 && filterIndex < filterSize)
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
                        for (int i = 0; i < filterSize; i++)
                        {
                            const int elementIndex = i + threadIdx.x * TupelSize + t;
                            temp +=
                                buffer[(bl + threadIdx.y * blockHeight) * extendedBlockW + elementIndex] * aFilter[i];
                        }

                        ComputeT origPixel = ComputeT(aSrcWithBC(pixelX + t, pixelYDst));
                        ComputeT highPass  = origPixel - temp;
                        ComputeT activator;
                        activator.x = abs(highPass.x) >= aThreshold ? 1.0f : 0.0f;
                        if constexpr (vector_active_size_v<ComputeT> > 1)
                        {
                            activator.y = abs(highPass.y) >= aThreshold ? 1.0f : 0.0f;
                        }
                        if constexpr (vector_active_size_v<ComputeT> > 2)
                        {
                            activator.z = abs(highPass.z) >= aThreshold ? 1.0f : 0.0f;
                        }
                        if constexpr (vector_active_size_v<ComputeT> > 3)
                        {
                            activator.w = abs(highPass.w) >= aThreshold ? 1.0f : 0.0f;
                        }
                        temp = origPixel + aWeight * highPass * activator;

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

        constexpr int extendedBlockW = myWarpSize + (filterSize - 1);
        constexpr int extendedBlockH = blockHeight + (filterSize - 1);

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

                    const int srcPixelX = pixelX0 - aFilterCenter + offsetX;

                    ComputeT pixel = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        const int filterIndex = ry - ky;

                        if (filterIndex >= 0 && filterIndex < filterSize)
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
                    for (int i = 0; i < filterSize; i++)
                    {
                        const int elementIndex = i + threadIdx.x;
                        temp += buffer[(bl + threadIdx.y * blockHeight) * extendedBlockW + elementIndex] * aFilter[i];
                    }

                    ComputeT origPixel = ComputeT(aSrcWithBC(pixelX, pixelYDst));
                    ComputeT highPass  = origPixel - temp;
                    ComputeT activator;
                    activator.x = abs(highPass.x) >= aThreshold ? 1.0f : 0.0f;
                    if constexpr (vector_active_size_v<ComputeT> > 1)
                    {
                        activator.y = abs(highPass.y) >= aThreshold ? 1.0f : 0.0f;
                    }
                    if constexpr (vector_active_size_v<ComputeT> > 2)
                    {
                        activator.z = abs(highPass.z) >= aThreshold ? 1.0f : 0.0f;
                    }
                    if constexpr (vector_active_size_v<ComputeT> > 3)
                    {
                        activator.w = abs(highPass.w) >= aThreshold ? 1.0f : 0.0f;
                    }
                    temp = origPixel + aWeight * highPass * activator;

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
          RoundingMode roundingMode, typename BorderControlT, typename FilterT, int filterSize>
void InvokeFixedSizeUnsharpFilterKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                        const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                        const FilterT *aFilter, FilterT aWeight, FilterT aThreshold, int aFilterCenter,
                                        const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    fixedSizeUnsharpFilterKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, blockHeight, roundingMode,
                                 BorderControlT, FilterT, filterSize>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aDst, aPitchDst, aFilter, aWeight,
                                                                aThreshold, aFilterCenter, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, int blockHeight, RoundingMode roundingMode,
          typename BorderControlT, typename FilterT>
void InvokeFixedSizeUnsharpFilterKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                               const FilterT *aFilter, FilterT aWeight, FilterT aThreshold,
                                               int aFilterSize, int aFilterCenter, const Size2D &aSize,
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
        const uint extendedBlockW          = BlockSize.x * TupelSize + (aFilterSize - 1);
        const uint SharedMemory            = sizeof(ComputeT) * (extendedBlockW)*blockHeight * BlockSize.y;

        switch (aFilterSize)
        {
            case 3:
                InvokeFixedSizeUnsharpFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight,
                                                   roundingMode, BorderControlT, FilterT, 3>(
                    BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                    aFilter, aWeight, aThreshold, aFilterCenter, aSize);
                break;
            case 5:
                InvokeFixedSizeUnsharpFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight,
                                                   roundingMode, BorderControlT, FilterT, 5>(
                    BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                    aFilter, aWeight, aThreshold, aFilterCenter, aSize);
                break;
            case 7:
                InvokeFixedSizeUnsharpFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight,
                                                   roundingMode, BorderControlT, FilterT, 7>(
                    BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                    aFilter, aWeight, aThreshold, aFilterCenter, aSize);
                break;
            case 9:
                InvokeFixedSizeUnsharpFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight,
                                                   roundingMode, BorderControlT, FilterT, 9>(
                    BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                    aFilter, aWeight, aThreshold, aFilterCenter, aSize);
                break;
            default:
                throw INVALIDARGUMENT(
                    aFilterSize, "Only sizes 3, 5, 7, and 9 are implemented as fixed size kernels. Provided size is: "
                                     << aFilterSize << ".");
                break;
        }
    }
    else
    {
        throw CUDAUNSUPPORTED(fixedSizeUnsharpFilterKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND