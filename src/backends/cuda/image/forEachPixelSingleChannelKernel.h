#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace opp::image::cuda
{
/// <summary>
/// runs aFunctor on every pixel (only one channel of a multi-channel pixel) of an image. no inplace operation, no mask.
/// </summary>
template <class DstT, typename funcType>
__global__ void forEachPixelSingleChannelKernel(DstT *__restrict__ aDst, size_t aPitchDst, Channel aDstChannel,
                                                Size2D aSize, funcType aFunctor)
{
    const int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelX >= aSize.x || pixelY >= aSize.y)
    {
        return;
    }

    Vector1<remove_vector_t<DstT>> res;
    DstT *pixelOut                     = gotoPtr(aDst, aPitchDst, pixelX, pixelY);
    remove_vector_t<DstT> *subPixelOut = reinterpret_cast<remove_vector_t<DstT> *>(pixelOut) + aDstChannel.Value();

    aFunctor(pixelX, pixelY, res);

    *subPixelOut = res.x;
    return;
}

template <typename DstT, typename funcType>
void InvokeForEachPixelSingleChannelKernel(const dim3 &aBlockSize, uint aSharedMemory, cudaStream_t aStream, DstT *aDst,
                                           size_t aPitchDst, Channel aDstChannel, const Size2D &aSize,
                                           const funcType &aFunctor)
{
    dim3 blocksPerGrid(DIV_UP(aSize.x, aBlockSize.x), DIV_UP(aSize.y, aBlockSize.y), 1);

    forEachPixelSingleChannelKernel<DstT, funcType>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aDst, aPitchDst, aDstChannel, aSize, aFunctor);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename DstT, typename funcType>
void InvokeForEachPixelSingleChannelKernelDefault(DstT *aDst, size_t aPitchDst, Channel aDstChannel,
                                                  const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx,
                                                  const funcType &aFunc)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize        = ConfigBlockSize<"Default">::value;
        constexpr uint SharedMemory = 0;

        InvokeForEachPixelSingleChannelKernel<DstT, funcType>(BlockSize, SharedMemory, aStreamCtx.Stream, aDst,
                                                              aPitchDst, aDstChannel, aSize, aFunc);
    }
    else
    {
        throw CUDAUNSUPPORTED(forEachPixelSingleChannelKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND