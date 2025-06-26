#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/maskTupel.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

namespace mpp::image::cuda
{
/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, with mask.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class DstT, class funcType>
__global__ void forEachPixelMaskedKernel(const byte *__restrict__ aMask, size_t aPitchMask, DstT *__restrict__ aDst,
                                         size_t aPitchDst, Size2D aSize,
                                         ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit, funcType aFunctor)
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
            const byte *pixelsMask         = gotoPtr(aMask, aPitchMask, pixelX, pixelY);
            MaskTupel<TupelSize> maskTupel = MaskTupel<TupelSize>::Load(pixelsMask);
            if (maskTupel.AreAllFalse())
            {
                // nothing to do for these pixels
                return;
            }

            // to avoid branching inside the warp, we keep processing all pixels, even if the mask is false for some
            // pixels...
            Tupel<DstT, TupelSize> res;

            DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX, pixelY);

            // load the destination pixel in case of inplace operation or we load the full pixel for alpha operations:
            if constexpr (funcType::DoLoadBeforeOp || //
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
            if constexpr (!funcType::DoLoadBeforeOp && //
                          (has_alpha_channel_v<DstT> && !load_full_vector_for_alpha_v<DstT>))
            {
#pragma unroll
                for (size_t i = 0; i < TupelSize; i++)
                {
                    alphaChannels[i] = (pixelsOut + i)->w;
                }
            }

            aFunctor(pixelX, pixelY, res);

            // restore alpha channel values:
            if constexpr (has_alpha_channel_v<DstT>)
            {
#pragma unroll
                for (size_t i = 0; i < TupelSize; i++)
                {
                    res.value[i].w = alphaChannels[i];
                }
            }

            if (maskTupel.AreAllTrue())
            {
                // save the entire Tupel
                Tupel<DstT, TupelSize>::StoreAligned(res, pixelsOut);
                return;
            }

            // save only those pixels that are active by the mask:
#pragma unroll
            for (size_t i = 0; i < TupelSize; i++)
            {
                if (maskTupel.value[i])
                {
                    *(pixelsOut + i) = res.value[i];
                }
            }
            return;
        }
    }

    const byte *pixelMask = gotoPtr(aMask, aPitchMask, pixelX, pixelY);
    byte mask             = *pixelMask;
    if (!mask)
    {
        // nothing to do as mask deactivated the pixel
        return;
    }

    // will be optimized away as unused in case of no alpha channel:
    pixel_basetype_t<DstT> alphaChannel;

    DstT res;
    DstT *pixelOut = gotoPtr(aDst, aPitchDst, pixelX, pixelY);

    // load the destination pixel in case of inplace operation or we load the full pixel for alpha operations:
    if constexpr (funcType::DoLoadBeforeOp || //
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
    if constexpr (!funcType::DoLoadBeforeOp && //
                  (has_alpha_channel_v<DstT> && !load_full_vector_for_alpha_v<DstT>))
    {
        alphaChannel = pixelOut->w;
    }

    aFunctor(pixelX, pixelY, res);

    // restore alpha channel value:
    if constexpr (has_alpha_channel_v<DstT>)
    {
        res.w = alphaChannel;
    }

    *pixelOut = res;
    return;
}

template <typename DstT, size_t TupelSize, int WarpAlignmentInBytes, class funcType>
void InvokeForEachPixelMaskedKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                    const byte *aMask, size_t aPitchMask, DstT *aDst, size_t aPitchDst,
                                    const Size2D &aSize, const funcType &aFunctor)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y, aBlockSize.y), 1);

    forEachPixelMaskedKernel<WarpAlignmentInBytes, TupelSize, DstT, funcType>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aMask, aPitchMask, aDst, aPitchDst, aSize, ts,
                                                                aFunctor);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename DstT, size_t TupelSize, typename funcType>
void InvokeForEachPixelMaskedKernelDefault(const Pixel8uC1 *aMask, size_t aPitchMask, DstT *aDst, size_t aPitchDst,
                                           const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx,
                                           const funcType &aFunc)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize               = ConfigBlockSize<"Default">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeForEachPixelMaskedKernel<DstT, TupelSize, WarpAlignmentInBytes, funcType>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, reinterpret_cast<const byte *>(aMask),
            aPitchMask, aDst, aPitchDst, aSize, aFunc);
    }
    else
    {
        throw CUDAUNSUPPORTED(forEachPixelMaskedKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}
} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND