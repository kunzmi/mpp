#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/mpp_defs.h>
#include <common/utilities.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{
/// <summary>
/// Create an accelerator index array for LUT value lookup.
/// </summary>
__global__ void lutAcceleratorKernel(const float *__restrict__ aX, int aLutSize, int *__restrict__ aAccelerator,
                                     int aAccerlatorSize)
{
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadX >= aAccerlatorSize)
    {
        return;
    }

    const float LUTMin   = aX[0];
    const float LUTMax   = aX[aLutSize - 1];
    const float stepSize = (LUTMax - LUTMin) / static_cast<float>(aAccerlatorSize - 1);

    int idx = LastIndexSmallerOrEqual(aX, aLutSize, 0, LUTMin + static_cast<float>(threadX) * stepSize);
    if (idx >= 0)
    {
        aAccelerator[threadX] = idx;
    }
    else
    {
        // we should never end up here, but who knows...
        aAccelerator[threadX] = 0;
    }
}

void InvokeLutAcceleratorKernel(const dim3 &aBlockSize, uint aSharedMemory, cudaStream_t aStream, const float *aX,
                                int aLutSize, int *aAccelerator, int aAccerlatorSize)
{
    dim3 blocksPerGrid(DIV_UP(aAccerlatorSize, aBlockSize.x), 1, 1);

    lutAcceleratorKernel<<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aX, aLutSize, aAccelerator,
                                                                                aAccerlatorSize);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

void InvokeLutAcceleratorKernelDefault(const float *aX, int aLutSize, int *aAccelerator, int aAccerlatorSize,
                                       const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize        = {64, 1, 1};
        constexpr uint SharedMemory = 0;

        InvokeLutAcceleratorKernel(BlockSize, SharedMemory, aStreamCtx.Stream, aX, aLutSize, aAccelerator,
                                   aAccerlatorSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(lutAcceleratorKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
