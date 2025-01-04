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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

namespace opp
{
namespace image
{
namespace cuda
{
/// <summary>
/// runs aOp on every pixel of an image. Inplace and outplace operation, no mask.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class DstT, typename functor>
__global__ void forEachPixelKernel(DstT *__restrict__ aDst, size_t aPitchDst, Size2D aSize,
                                   ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit, functor aOp)
{
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    if (aSplit.ThreadIsOutsideOfRange(threadX) || threadY >= aSize.y)
    {
        return;
    }

    const int pixelX = aSplit.GetPixel(threadX);
    const int pixelY = threadY;

    // don't check for warp alignment if TupelSize <= 1
    if constexpr (TupelSize > 1) // evaluated at compile time
    {
        // will be optimized away as unused in case of no alpha channel:
        pixel_basetype_t<DstT> alphaChannels[TupelSize];

        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            Tupel<DstT, TupelSize> res;

            DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX, pixelY);

            // load the destination pixel in case of inplace operation or we load the full pixel for alpha operations:
            if constexpr (functor::DoLoadBeforeOp || //
                          (has_alpha_channel_v<DstT> && load_full_vector_for_alpha_v<DstT>))
            {
                res = Tupel<DstT, TupelSize>::LoadAligned(pixelsOut);

                // save alpha channel values seperatly:
                if constexpr (has_alpha_channel_v<DstT>)
                {
#pragma unroll
                    for (size_t i = 0; i < TupelSize; i++)
                    {
                        alphaChannels[i] = res.value[i].w;
                    }
                }
            }

            // if we don't load the pixel anyhow but we still need just the alpha channel, load it:
            if constexpr (!functor::DoLoadBeforeOp && //
                          (has_alpha_channel_v<DstT> && !load_full_vector_for_alpha_v<DstT>))
            {
#pragma unroll
                for (size_t i = 0; i < TupelSize; i++)
                {
                    alphaChannels[i] = (pixelsOut + i)->w;
                }
            }

            aOp(pixelX, pixelY, res);

            // restore alpha channel values:
            if constexpr (has_alpha_channel_v<DstT>)
            {
#pragma unroll
                for (size_t i = 0; i < TupelSize; i++)
                {
                    res.value[i].w = alphaChannels[i];
                }
            }

            Tupel<DstT, TupelSize>::StoreAligned(res, pixelsOut);
            return;
        }
    }

    // will be optimized away as unused in case of no alpha channel:
    pixel_basetype_t<DstT> alphaChannel;

    DstT res;
    DstT *pixelOut = gotoPtr(aDst, aPitchDst, pixelX, pixelY);

    // load the destination pixel in case of inplace operation or we load the full pixel for alpha operations:
    if constexpr (functor::DoLoadBeforeOp || //
                  (has_alpha_channel_v<DstT> && load_full_vector_for_alpha_v<DstT>))
    {
        res = *pixelOut;

        // save alpha channel value seperatly:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            alphaChannel = res.w;
        }
    }
    // if we don't load the pixel anyhow but we still need just the alpha channel, load it:
    if constexpr (!functor::DoLoadBeforeOp && //
                  (has_alpha_channel_v<DstT> && !load_full_vector_for_alpha_v<DstT>))
    {
        alphaChannel = pixelOut->w;
    }

    aOp(pixelX, pixelY, res);

    // restore alpha channel value:
    if constexpr (has_alpha_channel_v<DstT>)
    {
        res.w = alphaChannel;
    }

    *pixelOut = res;
    return;
}

template <typename DstT, size_t TupelSize, int WarpAlignmentInBytes, typename funcType>
void InvokeForEachPixelKernel(const dim3 &aBlockSize, uint aSharedMemory, cudaStream_t aStream, DstT *aDst,
                              size_t pitchDst, const Size2D &aSize, const funcType &aOp)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x);

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y, aBlockSize.y), 1);

    forEachPixelKernel<WarpAlignmentInBytes, TupelSize, DstT, funcType>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aDst, pitchDst, aSize, ts, aOp);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename DstT, size_t TupelSize, typename funcType>
void InvokeForEachPixelKernelDefault(DstT *aDst, size_t pitchDst, const Size2D &aSize,
                                     const opp::cuda::StreamCtx &aStreamCtx, const funcType &aFunc)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize               = ConfigBlockSize<"Default">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeForEachPixelKernel<DstT, TupelSize, WarpAlignmentInBytes, funcType>(
            BlockSize, SharedMemory, aStreamCtx.Stream, aDst, pitchDst, aSize, aFunc);
    }
    else
    {
        throw CUDAUNSUPPORTED(forEachPixelKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace cuda
} // namespace image
} // namespace opp