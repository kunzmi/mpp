#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/gotoPtr.h>
#include <common/image/pitchException.h>
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
template <typename T> struct pixel_block_size_for_small_fixed_kernel_x
{
    constexpr static int value = 1;
};
template <> struct pixel_block_size_for_small_fixed_kernel_x<Pixel8uC3>
{
    constexpr static int value = 4;
};
template <> struct pixel_block_size_for_small_fixed_kernel_x<Pixel8sC3>
{
    constexpr static int value = 4;
};

template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, int blockWidth, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, typename windowOp, typename postOp,
          ReductionInitValue NeutralValue, int filterSize>
__global__ void fixedSmallSizeSeparableWindowOpKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst,
                                                      size_t aPitchDst, Vector2<int> aFilterCenter, windowOp aWindowOp,
                                                      postOp aPostOp, Size2D aSize,
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
        constexpr int extendedBlockW = TupelSize + (filterSize - 1);
        constexpr int extendedBlockH = blockHeight + (filterSize - 1);

        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            ComputeT result[blockHeight][TupelSize];
#pragma unroll
            for (int bh = 0; bh < blockHeight; bh++)
            {
#pragma unroll
                for (size_t t = 0; t < TupelSize; t++)
                {
                    result[bh][t] = reduction_init_value_v<NeutralValue, ComputeT>;
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
                    ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int ky = 0; ky < filterSize; ky++)
                    {
                        const int pixelDstY = ry - ky;

#pragma unroll
                        for (int kx = 0; kx < filterSize; kx++)
                        {
                            const int pixelDstX = rx - kx;

                            if (pixelDstY >= 0 && pixelDstY < blockHeight)
                            {
                                if (pixelDstX >= 0 && pixelDstX < TupelSize)
                                {
                                    aWindowOp(srcPixel, result[pixelDstY][pixelDstX]);
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
                        if constexpr (vector_size_v<ComputeT> == vector_size_v<DstT>)
                        {
                            aPostOp(result[bl][t]);

                            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                            {
                                round(result[bl][t]);
                            }

                            res.value[t] = DstT(result[bl][t]);
                        }
                        else
                        {
                            aPostOp(result[bl][t], res.value[t]);
                        }

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
        constexpr int extendedBlockW = blockWidth + (filterSize - 1);
        constexpr int extendedBlockH = blockHeight + (filterSize - 1);
        ComputeT result[blockHeight][blockWidth];
#pragma unroll
        for (int bh = 0; bh < blockHeight; bh++)
        {
#pragma unroll
            for (size_t bw = 0; bw < blockWidth; bw++)
            {
                result[bh][bw] = reduction_init_value_v<NeutralValue, ComputeT>;
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
                ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));
#pragma unroll
                for (int ky = 0; ky < filterSize; ky++)
                {
                    const int pixelDstY = ry - ky;

#pragma unroll
                    for (int kx = 0; kx < filterSize; kx++)
                    {
                        const int pixelDstX = rx - kx;

                        if (pixelDstY >= 0 && pixelDstY < blockHeight)
                        {
                            if (pixelDstX >= 0 && pixelDstX < blockWidth)
                            {
                                aWindowOp(srcPixel, result[pixelDstY][pixelDstX]);
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

                        if constexpr (vector_size_v<ComputeT> == vector_size_v<DstT>)
                        {
                            aPostOp(result[bl][bc]);

                            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                            {
                                round(result[bl][bc]);
                            }

                            res = DstT(result[bl][bc]);
                        }
                        else
                        {
                            aPostOp(result[bl][bc], res);
                        }

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
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, typename windowOp, typename postOp,
          ReductionInitValue NeutralValue, int filterSize>
__global__ void fixedSizeSeparableWindowOpKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                                 Vector2<int> aFilterCenter, windowOp aWindowOp, postOp aPostOp,
                                                 Size2D aSize, ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
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
                    ComputeT localBuffer[blockHeight];
#pragma unroll
                    for (int bh = 0; bh < blockHeight; bh++)
                    {
                        localBuffer[bh] = reduction_init_value_v<NeutralValue, ComputeT>;
                    }

#pragma unroll
                    for (int ry = 0; ry < extendedBlockH; ry++)
                    {
                        const int srcPixelY = ry - aFilterCenter.y + pixelY;

                        const int srcPixelX = pixelX0 - aFilterCenter.x + offsetX;
                        ComputeT pixel      = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                        for (int ky = 0; ky < blockHeight; ky++)
                        {
                            const int filterIndex = ry - ky;

                            if (filterIndex >= 0 && filterIndex < filterSize)
                            {
                                aWindowOp(pixel, localBuffer[ky]);
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
                        ComputeT temp = reduction_init_value_v<NeutralValue, ComputeT>;

#pragma unroll
                        for (int i = 0; i < filterSize; i++)
                        {
                            const int elementIndex = i + threadIdx.x * TupelSize + t;
                            aWindowOp(buffer[(bl + threadIdx.y * blockHeight) * extendedBlockW + elementIndex], temp);
                        }

                        if constexpr (vector_size_v<ComputeT> == vector_size_v<DstT>)
                        {
                            aPostOp(temp);

                            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                            {
                                round(temp);
                            }

                            res.value[t] = DstT(temp);
                        }
                        else
                        {
                            aPostOp(temp, res.value[t]);
                        }

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
                ComputeT localBuffer[blockHeight];
#pragma unroll
                for (int bh = 0; bh < blockHeight; bh++)
                {
                    localBuffer[bh] = reduction_init_value_v<NeutralValue, ComputeT>;
                }
#pragma unroll
                for (int ry = 0; ry < extendedBlockH; ry++)
                {
                    const int srcPixelY = ry - aFilterCenter.y + pixelY;

                    const int srcPixelX = pixelX0 - aFilterCenter.x + offsetX;

                    ComputeT pixel = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        const int filterIndex = ry - ky;

                        if (filterIndex >= 0 && filterIndex < filterSize)
                        {
                            aWindowOp(pixel, localBuffer[ky]);
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

                    ComputeT temp = reduction_init_value_v<NeutralValue, ComputeT>;

#pragma unroll
                    for (int i = 0; i < filterSize; i++)
                    {
                        const int elementIndex = i + threadIdx.x;
                        aWindowOp(buffer[(bl + threadIdx.y * blockHeight) * extendedBlockW + elementIndex], temp);
                    }

                    DstT res;
                    if constexpr (vector_size_v<ComputeT> == vector_size_v<DstT>)
                    {
                        aPostOp(temp);

                        if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                        {
                            round(temp);
                        }

                        res = DstT(temp);
                    }
                    else
                    {
                        aPostOp(temp, res);
                    }

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
          RoundingMode roundingMode, typename BorderControlT, typename windowOp, typename postOp,
          ReductionInitValue NeutralValue, int filterSize>
void InvokeFixedSmallSizeSeparableWindowOpKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize,
                                                 cudaStream_t aStream, const BorderControlT &aSrcWithBC, DstT *aDst,
                                                 size_t aPitchDst, const Vector2<int> &aFilterCenter,
                                                 windowOp aWindowOp, postOp aPostOp, const Size2D &aSize)
{
    constexpr int blockWidth = pixel_block_size_for_small_fixed_kernel_x<DstT>::value;

    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total() / blockWidth, aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    fixedSmallSizeSeparableWindowOpKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, blockWidth, blockHeight,
                                          roundingMode, BorderControlT, windowOp, postOp, NeutralValue, filterSize>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aDst, aPitchDst, aFilterCenter, aWindowOp,
                                                                aPostOp, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, typename windowOp, typename postOp,
          ReductionInitValue NeutralValue, int filterSize>
void InvokeFixedSizeSeparableWindowOpKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize,
                                            cudaStream_t aStream, const BorderControlT &aSrcWithBC, DstT *aDst,
                                            size_t aPitchDst, const Vector2<int> &aFilterCenter, windowOp aWindowOp,
                                            postOp aPostOp, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    fixedSizeSeparableWindowOpKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, blockHeight, roundingMode,
                                     BorderControlT, windowOp, postOp, NeutralValue, filterSize>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aDst, aPitchDst, aFilterCenter, aWindowOp,
                                                                aPostOp, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, int blockHeight, RoundingMode roundingMode,
          typename BorderControlT, typename windowOp, typename postOp, ReductionInitValue NeutralValue>
void InvokeFixedSizeSeparableWindowOpKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                                   int aFilterSize, const Vector2<int> &aFilterCenter,
                                                   windowOp aWindowOp, postOp aPostOp, const Size2D &aSize,
                                                   const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        checkPitchIsMultiple(aPitchDst, ConfigWarpAlignment<"Default">::value, TupelSize);

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
                InvokeFixedSmallSizeSeparableWindowOpKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes,
                                                            blockHeight, roundingMode, BorderControlT, windowOp, postOp,
                                                            NeutralValue, 3>(
                    BlockSize, 0, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst, aFilterCenter,
                    aWindowOp, aPostOp, aSize);
                break;
            case 5:
                InvokeFixedSizeSeparableWindowOpKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight,
                                                       roundingMode, BorderControlT, windowOp, postOp, NeutralValue, 5>(
                    BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                    aFilterCenter, aWindowOp, aPostOp, aSize);
                break;
            case 7:
                InvokeFixedSizeSeparableWindowOpKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight,
                                                       roundingMode, BorderControlT, windowOp, postOp, NeutralValue, 7>(
                    BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                    aFilterCenter, aWindowOp, aPostOp, aSize);
                break;
            case 9:
                InvokeFixedSizeSeparableWindowOpKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight,
                                                       roundingMode, BorderControlT, windowOp, postOp, NeutralValue, 9>(
                    BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                    aFilterCenter, aWindowOp, aPostOp, aSize);
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
        throw CUDAUNSUPPORTED(fixedSizeSeparableWindowOpKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
