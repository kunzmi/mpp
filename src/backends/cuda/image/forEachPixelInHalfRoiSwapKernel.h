#pragma once
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
/// runs aFunctor on every pixel of a (half-)sub-image and swaps the value with another pixel. Inplace.
/// </summary>
template <class DstT, typename funcType, bool xUneven>
__global__ void forEachPixelInHalfRoiSwapKernel(DstT *__restrict__ aDst, size_t aPitchDst, const Size2D &aSizeSub,
                                                funcType aFunctor)
{
    const int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelX >= aSizeSub.x || pixelY >= aSizeSub.y)
    {
        return;
    }

    if constexpr (xUneven)
    {
        // process the center vertical line only once:
        if (pixelX == aSizeSub.x - 1 && pixelY > aSizeSub.y / 2)
        {
            return;
        }
    }

    int otherX = 0;
    int otherY = 0;

    // get corresponding other pixel:
    aFunctor(pixelX, pixelY, otherX, otherY);

    DstT *pixel      = gotoPtr(aDst, aPitchDst, pixelX, pixelY);
    DstT *otherPixel = gotoPtr(aDst, aPitchDst, otherX, otherY);

    DstT pixelInRegister      = *pixel;
    DstT otherPixelInRegister = *otherPixel;

    // keep alpha channel value (simpleCPU version):
    if constexpr (has_alpha_channel_v<DstT>)
    {
        pixel_basetype_t<DstT> alphaChannelPixel = pixelInRegister.w;
        pixel_basetype_t<DstT> alphaChannelOther = otherPixelInRegister.w;

        DstT temp              = otherPixelInRegister;
        otherPixelInRegister   = pixelInRegister;
        otherPixelInRegister.w = alphaChannelOther;
        pixelInRegister        = temp;
        pixelInRegister.w      = alphaChannelPixel;
    }
    else
    {
        DstT temp            = otherPixelInRegister;
        otherPixelInRegister = pixelInRegister;
        pixelInRegister      = temp;
    }

    *pixel      = pixelInRegister;
    *otherPixel = otherPixelInRegister;
    return;
}

template <typename DstT, typename funcType, bool xUneven>
void InvokeForEachPixelInHalfRoiSwapKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize,
                                           cudaStream_t aStream, DstT *aDst, size_t aPitchDst, const Size2D &aSizeSub,
                                           const funcType &aFunctor)
{
    dim3 blocksPerGrid(DIV_UP(aSizeSub.x, aBlockSize.x), DIV_UP(aSizeSub.y, aBlockSize.y), 1);

    forEachPixelInHalfRoiSwapKernel<DstT, funcType, xUneven>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aDst, aPitchDst, aSizeSub, aFunctor);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename DstT, typename funcType, bool xUneven>
void InvokeForEachPixelInHalfRoiSwapKernelDefault(DstT *aDst, size_t aPitchDst, const Size2D &aSizeSub,
                                                  const mpp::cuda::StreamCtx &aStreamCtx, const funcType &aFunc)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize        = ConfigBlockSize<"Default">::value;
        constexpr uint SharedMemory = 0;

        InvokeForEachPixelInHalfRoiSwapKernel<DstT, funcType, xUneven>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aDst, aPitchDst, aSizeSub, aFunc);
    }
    else
    {
        throw CUDAUNSUPPORTED(forEachPixelKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
