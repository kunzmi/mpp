#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/filterArea.h>
#include <common/mpp_defs.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{
__device__ float getWeight(int aIndexX, int aIndexY, int aCenterX, int aCenterY, float aPosSquareSigma)
{
    const float idxX = static_cast<float>(aIndexX - aCenterX);
    const float idxY = static_cast<float>(aIndexY - aCenterY);
    float distSqr    = idxX * idxX + idxY * idxY;

    return __expf(-distSqr / (2.0f * aPosSquareSigma));
}

__global__ void precomputeGeometryDistanceCoeffKernel(Pixel32fC1 *__restrict__ aPreCompGeomDistCoeff,
                                                      FilterArea aFilterArea, float aPosSquareSigma)
{
    const int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelX >= aFilterArea.Size.x || pixelY >= aFilterArea.Size.y)
    {
        return;
    }

    const int idx = pixelY * aFilterArea.Size.x + pixelX;

    aPreCompGeomDistCoeff[idx] = getWeight(pixelX, pixelY, aFilterArea.Center.x, aFilterArea.Center.y, aPosSquareSigma);
}

void InvokePrecomputeGeometryDistanceCoeffKernel(const dim3 &aBlockSize, uint aSharedMemory, cudaStream_t aStream,
                                                 Pixel32fC1 *aPreCompGeomDistCoeff, const FilterArea &aFilterArea,
                                                 float aPosSquareSigma)
{
    dim3 blocksPerGrid(DIV_UP(aFilterArea.Size.x, aBlockSize.x), DIV_UP(aFilterArea.Size.y, aBlockSize.y), 1);

    precomputeGeometryDistanceCoeffKernel<<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(
        aPreCompGeomDistCoeff, aFilterArea, aPosSquareSigma);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

void InvokePrecomputeGeometryDistanceCoeffKernelDefault(Pixel32fC1 *aPreCompGeomDistCoeff,
                                                        const FilterArea &aFilterArea, float aPosSquareSigma,
                                                        const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize        = ConfigBlockSize<"Default">::value;
        constexpr uint SharedMemory = 0;

        InvokePrecomputeGeometryDistanceCoeffKernel(BlockSize, SharedMemory, aStreamCtx.Stream, aPreCompGeomDistCoeff,
                                                    aFilterArea, aPosSquareSigma);
    }
    else
    {
        throw CUDAUNSUPPORTED(bilateralGaussFilterKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND