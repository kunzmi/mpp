#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/channel.h>
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
/// sets a specific channel to a provided value, for each pixel.
/// </summary>
template <class DstT>
__global__ void setChannelKernel(DstT *__restrict__ aDst, size_t aPitchDst, remove_vector_t<DstT> aValue,
                                 Channel aChannel, Size2D aSize)
{
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelX >= aSize.x || pixelY >= aSize.y)
    {
        return;
    }

    DstT *pixelOut = gotoPtr(aDst, aPitchDst, pixelX, pixelY);

    (*pixelOut)[aChannel] = aValue;
    return;
}

/// <summary>
/// sets a specific channel to a provided value in device memory, for each pixel.
/// </summary>
template <class DstT>
__global__ void setChannelKernel(DstT *__restrict__ aDst, size_t aPitchDst, const remove_vector_t<DstT> *aValue,
                                 Channel aChannel, Size2D aSize)
{
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelX >= aSize.x || pixelY >= aSize.y)
    {
        return;
    }

    DstT *pixelOut = gotoPtr(aDst, aPitchDst, pixelX, pixelY);

    (*pixelOut)[aChannel] = *aValue;
    return;
}

template <typename DstT>
void InvokeSetChannelKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream, DstT *aDst,
                            size_t aPitchDst, remove_vector_t<DstT> aValue, Channel aChannel, const Size2D &aSize)
{
    dim3 blocksPerGrid(DIV_UP(aSize.x, aBlockSize.x), DIV_UP(aSize.y, aBlockSize.y), 1);

    setChannelKernel<DstT>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aDst, aPitchDst, aValue, aChannel, aSize);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename DstT>
void InvokeSetChannelKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream, DstT *aDst,
                            size_t aPitchDst, const remove_vector_t<DstT> *aValue, Channel aChannel,
                            const Size2D &aSize)
{
    dim3 blocksPerGrid(DIV_UP(aSize.x, aBlockSize.x), DIV_UP(aSize.y, aBlockSize.y), 1);

    setChannelKernel<DstT>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aDst, aPitchDst, aValue, aChannel, aSize);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename DstT>
void InvokeSetChannelKernelDefault(DstT *aDst, size_t aPitchDst, remove_vector_t<DstT> aValue, Channel aChannel,
                                   const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize        = ConfigBlockSize<"Default">::value;
        constexpr uint SharedMemory = 0;

        InvokeSetChannelKernel<DstT>(BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aDst, aPitchDst,
                                     aValue, aChannel, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(setChannelKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

template <typename DstT>
void InvokeSetChannelKernelDefault(DstT *aDst, size_t aPitchDst, const remove_vector_t<DstT> *aValue, Channel aChannel,
                                   const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize        = ConfigBlockSize<"Default">::value;
        constexpr uint SharedMemory = 0;

        InvokeSetChannelKernel<DstT>(BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aDst, aPitchDst,
                                     aValue, aChannel, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(setChannelKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
