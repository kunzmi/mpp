#pragma once
#include <backends/cuda/cudaException.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

namespace opp::cuda::image
{
/// <summary>
/// runs aOp on every pixel of an image. Inplace and outplace operation, no mask.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class DstT, class functor>
__global__ void forEachPixelKernel(DstT *__restrict__ aDst, size_t aPitchDst, opp::image::Size2D aSize,
                                   opp::image::ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit, functor aOp)
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
        opp::image::pixel_basetype_t<DstT> alphaChannels[TupelSize];

        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            Tupel<DstT, TupelSize> res;

            DstT *pixelsOut = opp::image::gotoPtr(aDst, aPitchDst, pixelX, pixelY);

            // load the destination pixel in case of inplace operation or we load the full pixel for alpha operations:
            if constexpr (functor::DoLoadBeforeOp || //
                          (opp::image::has_alpha_channel_v<DstT> && opp::image::load_full_vector_for_alpha_v<DstT>))
            {
                res = Tupel<DstT, TupelSize>::LoadAligned(pixelsOut);

                // save alpha channel values seperatly:
                if constexpr (opp::image::has_alpha_channel_v<DstT>)
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
                          (opp::image::has_alpha_channel_v<DstT> && !opp::image::load_full_vector_for_alpha_v<DstT>))
            {
#pragma unroll
                for (size_t i = 0; i < TupelSize; i++)
                {
                    alphaChannels[i] = (pixelsOut + i)->w;
                }
            }

            aOp(pixelX, pixelY, res);

            // restore alpha channel values:
            if constexpr (opp::image::has_alpha_channel_v<DstT>)
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
    opp::image::pixel_basetype_t<DstT> alphaChannel;

    DstT res;
    DstT *pixelOut = opp::image::gotoPtr(aDst, aPitchDst, pixelX, pixelY);

    // load the destination pixel in case of inplace operation or we load the full pixel for alpha operations:
    if constexpr (functor::DoLoadBeforeOp || //
                  (opp::image::has_alpha_channel_v<DstT> && opp::image::load_full_vector_for_alpha_v<DstT>))
    {
        res = *pixelOut;

        // save alpha channel value seperatly:
        if constexpr (opp::image::has_alpha_channel_v<DstT>)
        {
            alphaChannel = res.w;
        }
    }
    // if we don't load the pixel anyhow but we still need just the alpha channel, load it:
    if constexpr (!functor::DoLoadBeforeOp && //
                  (opp::image::has_alpha_channel_v<DstT> && !opp::image::load_full_vector_for_alpha_v<DstT>))
    {
        alphaChannel = pixelOut->w;
    }

    aOp(pixelX, pixelY, res);

    // restore alpha channel value:
    if constexpr (opp::image::has_alpha_channel_v<DstT>)
    {
        res.w = alphaChannel;
    }

    *pixelOut = res;
    return;
}

template <typename SrcT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, class funcType>
void InvokeForEachPixelKernel(const dim3 &aBlockSize, DstT *aDst, size_t pitchDst, const opp::image::Size2D &aSize,
                              const funcType &aOp)
{
    opp::image::ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x);

    dim3 blocksPerGrid((ts.Total() + (aBlockSize.x - 1)) / aBlockSize.x, (aSize.y + (aBlockSize.y - 1)) / aBlockSize.y,
                       1);

    forEachPixelKernel<WarpAlignmentInBytes, TupelSize, DstT, funcType>
        <<<blocksPerGrid, aBlockSize>>>(aDst, pitchDst, aSize, ts, aOp);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " Tupel size: " << TupelSize);
}
} // namespace opp::cuda::image