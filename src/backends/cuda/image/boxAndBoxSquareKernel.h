#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/filterArea.h>
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
/// <summary>
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, int blockHeight, typename BorderControlT>
__global__ void boxAndBoxSquareKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                      FilterArea aFilterArea, Size2D aSize,
                                      ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    constexpr int myWarpSize = 32; // warpSize itself is not const nor constexpr...

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
                        const int srcPixelY = ry - aFilterArea.Center.y + pixelY;

                        const int srcPixelX             = pixelX0 - aFilterArea.Center.x + offsetX;
                        remove_vector_t<ComputeT> pixel = remove_vector_t<ComputeT>(aSrcWithBC(srcPixelX, srcPixelY).x);

#pragma unroll
                        for (int ky = 0; ky < blockHeight; ky++)
                        {
                            const int filterIndex = ry - ky;

                            if (filterIndex >= 0 && filterIndex < aFilterArea.Size.y)
                            {
                                localBuffer[ky].x += pixel;
                                localBuffer[ky].y += pixel * pixel;
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

                        for (int i = 0; i < aFilterArea.Size.x; i++)
                        {
                            const int elementIndex = i + threadIdx.x * TupelSize + t;
                            temp += buffer[(bl + threadIdx.y * blockHeight) * extendedBlockW + elementIndex];
                        }

                        temp.x /= remove_vector_t<ComputeT>(aFilterArea.Size.x * aFilterArea.Size.y);
                        res.value[t] = DstT(temp);
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
                    const int srcPixelY = ry - aFilterArea.Center.y + pixelY;

                    const int srcPixelX = pixelX0 - aFilterArea.Center.x + offsetX;

                    remove_vector_t<ComputeT> pixel = remove_vector_t<ComputeT>(aSrcWithBC(srcPixelX, srcPixelY).x);

#pragma unroll
                    for (int ky = 0; ky < blockHeight; ky++)
                    {
                        const int filterIndex = ry - ky;

                        if (filterIndex >= 0 && filterIndex < aFilterArea.Size.y)
                        {
                            localBuffer[ky].x += pixel;
                            localBuffer[ky].y += pixel * pixel;
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
                    for (int i = 0; i < aFilterArea.Size.x; i++)
                    {
                        const int elementIndex = i + threadIdx.x;
                        temp += buffer[(bl + threadIdx.y * blockHeight) * extendedBlockW + elementIndex];
                    }

                    temp.x /= remove_vector_t<ComputeT>(aFilterArea.Size.x * aFilterArea.Size.y);
                    DstT res = DstT(temp);

                    *pixelsOut = res;
                }
            }
        }

        return;
    }
}

template <typename ComputeT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, int blockHeight,
          typename BorderControlT>
void InvokeBoxAndBoxSquareKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                 const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                 const FilterArea &aFilterArea, const Size2D &aSize)
{
    static_assert(vector_size_v<DstT> == 2, "For this specialised kernel, only two channel output is possible: .x for "
                                            "mean value and .y for summed square.");

    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    boxAndBoxSquareKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, blockHeight, BorderControlT>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aDst, aPitchDst, aFilterArea, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, int blockHeight, typename BorderControlT>
void InvokeBoxAndBoxSquareKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                        const FilterArea &aFilterArea, const Size2D &aSize,
                                        const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        checkPitchIsMultiple(aPitchDst, ConfigWarpAlignment<"Default">::value, TupelSize);

        dim3 BlockSize = ConfigBlockSize<"Default">::value;

        if (blockHeight > 1)
        {
            BlockSize.y = 2;
        } /**/

        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        const uint extendedBlockW          = BlockSize.x * TupelSize + (aFilterArea.Size.x - 1);
        const uint SharedMemory            = sizeof(ComputeT) * (extendedBlockW)*blockHeight * BlockSize.y;

        if (SharedMemory <= aStreamCtx.SharedMemPerBlock)
        {
            InvokeBoxAndBoxSquareKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockHeight, BorderControlT>(
                BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                aFilterArea, aSize);
        }
        else if (blockHeight > 1)
        {
            // try with reduced blockHeight:
            const uint SharedMemory2 = sizeof(ComputeT) * (extendedBlockW) * 1 * BlockSize.y;

            if (SharedMemory2 > aStreamCtx.SharedMemPerBlock)
            {
                throw INVALIDARGUMENT(aFilterArea.Size,
                                      "Even with reduced launch configuration, the filter kernel width ("
                                          << aFilterArea.Size.x << " pixels) requires too much shared memory ("
                                          << SharedMemory2 << " Bytes). The limit is " << aStreamCtx.SharedMemPerBlock
                                          << " Bytes - cannot launch kernel.");
            }
            InvokeBoxAndBoxSquareKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, 1, BorderControlT>(
                BlockSize, SharedMemory2, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst,
                aFilterArea, aSize);
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
        throw CUDAUNSUPPORTED(boxAndBoxSquareKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
