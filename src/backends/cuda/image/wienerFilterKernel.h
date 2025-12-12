#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/filterArea.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vector2.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{
/// <summary>
/// Same as separableWindowOpKernel with Add operator, but accumulates pixel and pixel^2 for variance computation.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT>
__global__ void wienerFilterKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                   FilterArea aFilterArea, ComputeT aNoise, Size2D aSize,
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
        const int extendedBlockW = myWarpSize * TupelSize + (aFilterArea.Size.x - 1);
        const int extendedBlockH = blockHeight + (aFilterArea.Size.y - 1);

        ComputeT *bufferSqr = buffer + (extendedBlockW * blockHeight * blockDim.y);
        // as threads in warp-aligned area are always the full warp, no need to check for X pixel limits here
        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            for (int i = 0; i < extendedBlockW; i += myWarpSize)
            {
                const int offsetX = i + threadIdx.x;

                if (offsetX < extendedBlockW)
                {
                    ComputeT localBuffer[blockHeight]    = {0};
                    ComputeT localBufferSqr[blockHeight] = {0};

                    for (int ry = 0; ry < extendedBlockH; ry++)
                    {
                        const int srcPixelY = ry - aFilterArea.Center.y + pixelY;

                        const int srcPixelX = pixelX0 - aFilterArea.Center.x + offsetX;
                        ComputeT pixel      = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));
                        ComputeT pixelSqr   = ComputeT::Sqr(pixel);

#pragma unroll
                        for (int ky = 0; ky < blockHeight; ky++)
                        {
                            const int filterIndex = ry - ky;

                            if (filterIndex >= 0 && filterIndex < aFilterArea.Size.y)
                            {
                                localBuffer[ky] += pixel;
                                localBufferSqr[ky] += pixelSqr;
                            }
                        }
                    }

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        buffer[(ky + threadIdx.y * blockHeight) * extendedBlockW + offsetX]    = localBuffer[ky];
                        bufferSqr[(ky + threadIdx.y * blockHeight) * extendedBlockW + offsetX] = localBufferSqr[ky];
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
                        ComputeT tempSqr(0);

                        for (int i = 0; i < aFilterArea.Size.x; i++)
                        {
                            const int elementIndex = i + threadIdx.x * TupelSize + t;
                            const int idxShared    = (bl + threadIdx.y * blockHeight) * extendedBlockW + elementIndex;
                            temp += buffer[idxShared];
                            tempSqr += bufferSqr[idxShared];
                        }

                        const int srcPixelX = pixelX + t;
                        const int srcPixelY = bl + pixelY;

                        ComputeT pixelSrc = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));
                        const remove_vector_t<ComputeT> maskSizeInv =
                            static_cast<remove_vector_t<ComputeT>>(1) /
                            static_cast<remove_vector_t<ComputeT>>(aFilterArea.Size.x * aFilterArea.Size.y);

                        temp *= maskSizeInv;                           // sum --> mean
                        tempSqr = tempSqr * maskSizeInv - temp * temp; // sumSqr --> variance (with bias as no N-1)

                        // wiener filter:
                        temp = temp + ComputeT::Max(tempSqr - aNoise, ComputeT(0)) / ComputeT::Max(tempSqr, aNoise) *
                                          (pixelSrc - temp);

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

        const int extendedBlockW = myWarpSize + (aFilterArea.Size.x - 1);
        const int extendedBlockH = blockHeight + (aFilterArea.Size.y - 1);
        ComputeT *bufferSqr      = buffer + (extendedBlockW * blockHeight * blockDim.y);

#pragma unroll
        for (int i = 0; i < extendedBlockW; i += myWarpSize)
        {
            const int offsetX = i + threadIdx.x;

            if (offsetX < extendedBlockW)
            {
                ComputeT localBuffer[blockHeight]    = {0};
                ComputeT localBufferSqr[blockHeight] = {0};

#pragma unroll
                for (int ry = 0; ry < extendedBlockH; ry++)
                {
                    const int srcPixelY = ry - aFilterArea.Center.y + pixelY;

                    const int srcPixelX = pixelX0 - aFilterArea.Center.x + offsetX;
                    ComputeT pixel      = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));
                    ComputeT pixelSqr   = ComputeT::Sqr(pixel);

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        const int filterIndex = ry - ky;

                        if (filterIndex >= 0 && filterIndex < aFilterArea.Size.y)
                        {
                            localBuffer[ky] += pixel;
                            localBufferSqr[ky] += pixelSqr;
                        }
                    }
                }

#pragma unroll
                for (int ky = 0; ky < blockHeight; ky++)
                {
                    buffer[(ky + threadIdx.y * blockHeight) * extendedBlockW + offsetX]    = localBuffer[ky];
                    bufferSqr[(ky + threadIdx.y * blockHeight) * extendedBlockW + offsetX] = localBufferSqr[ky];
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
                    ComputeT tempSqr(0);

                    for (int i = 0; i < aFilterArea.Size.x; i++)
                    {
                        const int elementIndex = i + threadIdx.x;
                        const int idxShared    = (bl + threadIdx.y * blockHeight) * extendedBlockW + elementIndex;
                        temp += buffer[idxShared];
                        tempSqr += bufferSqr[idxShared];
                    }

                    const int srcPixelX = pixelX;
                    const int srcPixelY = bl + pixelY;

                    ComputeT pixelSrc = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));
                    const remove_vector_t<ComputeT> maskSizeInv =
                        static_cast<remove_vector_t<ComputeT>>(1) /
                        static_cast<remove_vector_t<ComputeT>>(aFilterArea.Size.x * aFilterArea.Size.y);

                    temp *= maskSizeInv;                           // sum --> mean
                    tempSqr = tempSqr * maskSizeInv - temp * temp; // sumSqr --> variance (with bias as no N-1)

                    // wiener filter:
                    temp = temp + ComputeT::Max(tempSqr - aNoise, ComputeT(0)) / ComputeT::Max(tempSqr, aNoise) *
                                      (pixelSrc - temp);

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
          RoundingMode roundingMode, typename BorderControlT>
void InvokeWienerFilterKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                              const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                              const FilterArea &aFilterArea, ComputeT aNoise, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    wienerFilterKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, blockHeight, roundingMode, BorderControlT>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aDst, aPitchDst, aFilterArea, aNoise, aSize,
                                                                ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, int blockHeight, RoundingMode roundingMode,
          typename BorderControlT>
void InvokeWienerFilterKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                     const FilterArea &aFilterArea, ComputeT aNoise, const Size2D &aSize,
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
        const uint extendedBlockW          = BlockSize.x * TupelSize + (aFilterArea.Size.x - 1);
        const uint SharedMemory            = 2 * sizeof(ComputeT) * extendedBlockW * blockHeight * BlockSize.y;

        if (SharedMemory <= aStreamCtx.SharedMemPerBlock)
        {
            InvokeWienerFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight, roundingMode,
                                     BorderControlT>(BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream,
                                                     aSrcWithBC, aDst, aPitchDst, aFilterArea, aNoise, aSize);
        }
        else if (blockHeight > 1)
        {
            // try with reduced blockHeight:
            const uint SharedMemory2 = 2 * sizeof(ComputeT) * extendedBlockW * 1 * BlockSize.y;

            if (SharedMemory2 > aStreamCtx.SharedMemPerBlock)
            {
                throw INVALIDARGUMENT(aFilterArea.Size,
                                      "Even with reduced launch configuration, the filter kernel width ("
                                          << aFilterArea.Size.x << " pixels) requires too much shared memory ("
                                          << SharedMemory2 << " Bytes). The limit is " << aStreamCtx.SharedMemPerBlock
                                          << " Bytes - cannot launch kernel.");
            }
            InvokeWienerFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, 1, roundingMode, BorderControlT>(
                BlockSize, SharedMemory2, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                aFilterArea, aNoise, aSize);
        }
        else
        {
            throw INVALIDARGUMENT(aFilterArea.Size, "The filter kernel width ("
                                                        << aFilterArea.Size.x
                                                        << " pixels) requires too much shared memory (" << SharedMemory
                                                        << " Bytes). The limit is " << aStreamCtx.SharedMemPerBlock
                                                        << " Bytes - cannot launch kernel.");
        }
    }
    else
    {
        throw CUDAUNSUPPORTED(wienerFilterKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
