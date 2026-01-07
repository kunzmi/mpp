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
/// runs aFunctor on every 2x2 pixel block of an image. Outplace operation, no mask.
/// </summary>
template <class DstT, typename funcType>
__global__ void forEachPixelBlockKernel(DstT *__restrict__ aDst, size_t aPitchDst, Size2D aSize, funcType aFunctor)
{
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelX = threadX * 2;
    int pixelY = threadY * 2;

    if (pixelX >= aSize.x || pixelY >= aSize.y)
    {
        return;
    }

    // will be optimized away as unused in case of no alpha channel:
    pixel_basetype_t<DstT> alphaChannel[4];

    DstT res[4];
    DstT *pixelOut0 = gotoPtr(aDst, aPitchDst, pixelX, pixelY);
    DstT *pixelOut1 = pixelOut0 + 1;
    DstT *pixelOut2 = gotoPtr(aDst, aPitchDst, pixelX, pixelY + 1);
    DstT *pixelOut3 = pixelOut2 + 1;

    // load the destination pixel in case of inplace operation or we load the full pixel for alpha operations:
    if constexpr (funcType::DoLoadBeforeOp || //
                  (has_alpha_channel_v<DstT> && load_full_vector_for_alpha_v<DstT>))
    {
        res[0] = *pixelOut0;
        res[1] = *pixelOut1;
        res[2] = *pixelOut2;
        res[3] = *pixelOut3;

        // save alpha channel value seperatly:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            alphaChannel[0] = res[0].w;
            alphaChannel[1] = res[1].w;
            alphaChannel[2] = res[2].w;
            alphaChannel[3] = res[3].w;
        }
    }

    // for nearly all functors aOp will evaluate to constant true.
    // Only transformerFunctor is capable of returning false if a source pixel is outside of the src-roi
    if (aFunctor(pixelX, pixelY, res))
    {
        // if we don't load the pixel anyhow but we still need just the alpha channel, load it:
        if constexpr (!funcType::DoLoadBeforeOp && //
                      (has_alpha_channel_v<DstT> && !load_full_vector_for_alpha_v<DstT>))
        {
            alphaChannel[0] = pixelOut0->w;
            alphaChannel[1] = pixelOut1->w;
            alphaChannel[2] = pixelOut2->w;
            alphaChannel[3] = pixelOut3->w;
        }

        // restore alpha channel value:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            res[0].w = alphaChannel[0];
            res[1].w = alphaChannel[1];
            res[2].w = alphaChannel[2];
            res[3].w = alphaChannel[3];
        }

        *pixelOut0 = res[0];
        *pixelOut1 = res[1];
        *pixelOut2 = res[2];
        *pixelOut3 = res[3];
    }
    return;
}

template <typename DstT, typename funcType>
void InvokeForEachPixelBlockKernel(const dim3 &aBlockSize, uint aSharedMemory, cudaStream_t aStream, DstT *aDst,
                                   size_t aPitchDst, const Size2D &aSize, const funcType &aFunctor)
{
    dim3 blocksPerGrid(DIV_UP(aSize.x / 2, aBlockSize.x), DIV_UP(aSize.y / 2, aBlockSize.y), 1);

    forEachPixelBlockKernel<DstT, funcType>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aDst, aPitchDst, aSize, aFunctor);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename DstT, typename funcType>
void InvokeForEachPixelBlockKernelDefault(DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                                          const mpp::cuda::StreamCtx &aStreamCtx, const funcType &aFunc)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize        = ConfigBlockSize<"Default">::value;
        constexpr uint SharedMemory = 0;

        InvokeForEachPixelBlockKernel<DstT, funcType>(BlockSize, SharedMemory, aStreamCtx.Stream, aDst, aPitchDst,
                                                      aSize, aFunc);
    }
    else
    {
        throw CUDAUNSUPPORTED(forEachPixelKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
