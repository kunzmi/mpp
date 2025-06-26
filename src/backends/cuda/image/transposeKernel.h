#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{

/// <summary>
/// performs matrix transpose operation.
/// </summary>
template <typename SrcDstT, int blockDimX, int blockDimY>
__global__ void transposeKernel(const SrcDstT *__restrict__ aSrc, size_t aPitchSrc, SrcDstT *__restrict__ aDst,
                                size_t aPitchDst, Size2D aSizeDst)
{
    constexpr int bankConflictOffset =
        sizeof(SrcDstT) == 1 ? 4 : (sizeof(SrcDstT) == 2 ? 2 : (sizeof(SrcDstT) == 4 ? 1 : 0));

    __shared__ SrcDstT buffer[blockDimY][blockDimX + bankConflictOffset];
    int xIn = blockIdx.x * blockDimX + threadIdx.x;
    int yIn = blockIdx.y * blockDimY + threadIdx.y;

    if (xIn < aSizeDst.y && yIn < aSizeDst.x) // aSizeDst transposed
    {
        const SrcDstT *pixelIn           = gotoPtr(aSrc, aPitchSrc, xIn, yIn);
        buffer[threadIdx.y][threadIdx.x] = *pixelIn;
    }
    __syncthreads();

    int xOut = blockIdx.y * blockDimX + threadIdx.x;
    int yOut = blockIdx.x * blockDimY + threadIdx.y;
    if (xOut < aSizeDst.x && yOut < aSizeDst.y)
    {
        SrcDstT *pixelOut = gotoPtr(aDst, aPitchDst, xOut, yOut);
        *pixelOut         = buffer[threadIdx.x][threadIdx.y];
    }
}

template <typename SrcDstT, int blockDimX, int blockDimY>
void InvokeTransposeKernel(const dim3 &aBlockSize, uint aSharedMemory, cudaStream_t aStream, const SrcDstT *aSrc,
                           size_t aPitchSrc, SrcDstT *aDst, size_t aPitchDst, const Size2D &aSizeDst)
{

    dim3 blocksPerGrid(DIV_UP(aSizeDst.y, aBlockSize.x), DIV_UP(aSizeDst.x, aBlockSize.y), 1);

    transposeKernel<SrcDstT, blockDimX, blockDimY>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrc, aPitchSrc, aDst, aPitchDst, aSizeDst);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename SrcDstT>
void InvokeTransposeKernelDefault(const SrcDstT *aSrc, size_t aPitchSrc, SrcDstT *aDst, size_t aPitchDst,
                                  const Size2D &aSizeDst, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        if constexpr (sizeof(SrcDstT) > 4 * 8) //==32 bytes or for ex. Pixel64fC4 is still OK
        {
            // large type, eg Pixel64fcC4 (complex double)
            constexpr ConstExprDim3 BlockSize{16, 16, 1};
            uint SharedMemory = 0;

            InvokeTransposeKernel<SrcDstT, BlockSize.x, BlockSize.y>(BlockSize, SharedMemory, aStreamCtx.Stream, aSrc,
                                                                     aPitchSrc, aDst, aPitchDst, aSizeDst);
        }
        else
        {
            constexpr ConstExprDim3 BlockSize{16, 16, 1};
            uint SharedMemory = 0;

            InvokeTransposeKernel<SrcDstT, BlockSize.x, BlockSize.y>(BlockSize, SharedMemory, aStreamCtx.Stream, aSrc,
                                                                     aPitchSrc, aDst, aPitchDst, aSizeDst);
        }
    }
    else
    {
        throw CUDAUNSUPPORTED(transposeKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND