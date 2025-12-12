#pragma once
#include "integralXKernel.h" // for doShuffleUp functions
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/reductionInitValues.h>
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
/// performs integral and integralSqr image computation in X direction and pads a row and a column.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <int TupelSize, int WarpAlignmentInBytes, class SrcT, class DstT, class DstSqrT>
__global__ void integralSqrXKernel(const SrcT *__restrict__ aSrc, size_t aPitchSrc, DstT *__restrict__ aDst,
                                   size_t aPitchDst, DstSqrT *__restrict__ aDstSqr, size_t aPitchDstSqr, Size2D aSize,
                                   ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    int warpLaneID = threadIdx.x;
    int pixelY     = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelY >= aSize.y)
    {
        return;
    }
    DstT previousWarp(0);       // result of righmost value in previous warp
    DstSqrT previousSqrWarp(0); // result of righmost value in previous warp

    // simple case, no tupels
    if constexpr (TupelSize == 1)
    {
        // loop over x-dimension in warp steps:
        for (int pixelXWarp0 = 0; pixelXWarp0 < aSize.x; pixelXWarp0 += warpSize)
        {
            const int pixelX = pixelXWarp0 + warpLaneID;
            DstT resVal(0);
            DstT resSqrVal(0);
            if (pixelY > 0)
            {
                if (pixelX < aSize.x)
                {
                    if (pixelX > 0)
                    {
                        // pixelX - 1 do add 0 border at the left:
                        SrcT pixel                = *gotoPtr(aSrc, aPitchSrc, pixelX - 1, pixelY - 1);
                        const DstT pixelSrc       = DstT(pixel);
                        const DstSqrT pixelSqrSrc = DstSqrT(pixel).Sqr();
                        resVal += pixelSrc;
                        resSqrVal += pixelSqrSrc;
                    }
                }

                doShuffleUp(warpLaneID, resVal.x);
                doShuffleUp(warpLaneID, resSqrVal.x);
                if constexpr (vector_active_size_v<SrcT> > 1)
                {
                    doShuffleUp(warpLaneID, resVal.y);
                    doShuffleUp(warpLaneID, resSqrVal.y);
                }
                if constexpr (vector_active_size_v<SrcT> > 2)
                {
                    doShuffleUp(warpLaneID, resVal.z);
                    doShuffleUp(warpLaneID, resSqrVal.z);
                }
                if constexpr (vector_active_size_v<SrcT> > 3)
                {
                    doShuffleUp(warpLaneID, resVal.w);
                    doShuffleUp(warpLaneID, resSqrVal.w);
                }

                resVal += previousWarp;
                resSqrVal += previousSqrWarp;

                previousWarp.x    = __shfl_sync(0xFFFFFFFF, resVal.x, warpSize - 1);
                previousSqrWarp.x = __shfl_sync(0xFFFFFFFF, resSqrVal.x, warpSize - 1);
                if constexpr (vector_active_size_v<SrcT> > 1)
                {
                    previousWarp.y    = __shfl_sync(0xFFFFFFFF, resVal.y, warpSize - 1);
                    previousSqrWarp.y = __shfl_sync(0xFFFFFFFF, resSqrVal.y, warpSize - 1);
                }
                if constexpr (vector_active_size_v<SrcT> > 2)
                {
                    previousWarp.z    = __shfl_sync(0xFFFFFFFF, resVal.z, warpSize - 1);
                    previousSqrWarp.z = __shfl_sync(0xFFFFFFFF, resSqrVal.z, warpSize - 1);
                }
                if constexpr (vector_active_size_v<SrcT> > 3)
                {
                    previousWarp.w    = __shfl_sync(0xFFFFFFFF, resVal.w, warpSize - 1);
                    previousSqrWarp.w = __shfl_sync(0xFFFFFFFF, resSqrVal.w, warpSize - 1);
                }
            }
            else
            {
                resVal          = previousWarp;
                previousWarp    = DstT(0);
                resSqrVal       = previousSqrWarp;
                previousSqrWarp = DstT(0);
            }
            if (pixelX < aSize.x)
            {
                DstT *pixelDst       = gotoPtr(aDst, aPitchDst, pixelX, pixelY);
                *pixelDst            = resVal;
                DstSqrT *pixelDstSqr = gotoPtr(aDstSqr, aPitchDstSqr, pixelX, pixelY);
                *pixelDstSqr         = resSqrVal;
            }
        }
    }
    else
    {
        // compute left unaligned part:
        for (int pixelXWarp0 = 0; pixelXWarp0 < aSplit.MutedAndLeft(); pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelLeft(pixelXWarp0 + warpLaneID);
            DstT resVal(0);
            DstSqrT resSqrVal(0);
            if (pixelY > 0)
            {
                if (pixelX >= 0) // i.e. thread is active
                {
                    if (pixelX > 0)
                    {
                        // pixelX - 1 do add 0 border at the left:
                        SrcT pixel                = *gotoPtr(aSrc, aPitchSrc, pixelX - 1, pixelY - 1);
                        const DstT pixelSrc       = DstT(pixel);
                        const DstSqrT pixelSqrSrc = DstSqrT(pixel).Sqr();
                        resVal += pixelSrc;
                        resSqrVal += pixelSqrSrc;
                    }
                }

                doShuffleUp(warpLaneID, resVal.x);
                doShuffleUp(warpLaneID, resSqrVal.x);
                if constexpr (vector_active_size_v<SrcT> > 1)
                {
                    doShuffleUp(warpLaneID, resVal.y);
                    doShuffleUp(warpLaneID, resSqrVal.y);
                }
                if constexpr (vector_active_size_v<SrcT> > 2)
                {
                    doShuffleUp(warpLaneID, resVal.z);
                    doShuffleUp(warpLaneID, resSqrVal.z);
                }
                if constexpr (vector_active_size_v<SrcT> > 3)
                {
                    doShuffleUp(warpLaneID, resVal.w);
                    doShuffleUp(warpLaneID, resSqrVal.w);
                }

                resVal += previousWarp;
                resSqrVal += previousSqrWarp;

                previousWarp.x    = __shfl_sync(0xFFFFFFFF, resVal.x, warpSize - 1);
                previousSqrWarp.x = __shfl_sync(0xFFFFFFFF, resSqrVal.x, warpSize - 1);
                if constexpr (vector_active_size_v<SrcT> > 1)
                {
                    previousWarp.y    = __shfl_sync(0xFFFFFFFF, resVal.y, warpSize - 1);
                    previousSqrWarp.y = __shfl_sync(0xFFFFFFFF, resSqrVal.y, warpSize - 1);
                }
                if constexpr (vector_active_size_v<SrcT> > 2)
                {
                    previousWarp.z    = __shfl_sync(0xFFFFFFFF, resVal.z, warpSize - 1);
                    previousSqrWarp.z = __shfl_sync(0xFFFFFFFF, resSqrVal.z, warpSize - 1);
                }
                if constexpr (vector_active_size_v<SrcT> > 3)
                {
                    previousWarp.w    = __shfl_sync(0xFFFFFFFF, resVal.w, warpSize - 1);
                    previousSqrWarp.w = __shfl_sync(0xFFFFFFFF, resSqrVal.w, warpSize - 1);
                }
            }
            else
            {
                resVal          = previousWarp;
                previousWarp    = DstT(0);
                resSqrVal       = previousSqrWarp;
                previousSqrWarp = DstT(0);
            }
            if (pixelX < aSize.x)
            {
                DstT *pixelDst       = gotoPtr(aDst, aPitchDst, pixelX, pixelY);
                *pixelDst            = resVal;
                DstSqrT *pixelDstSqr = gotoPtr(aDstSqr, aPitchDstSqr, pixelX, pixelY);
                *pixelDstSqr         = resSqrVal;
            }
        }

        // compute center part as tupels:
        for (int pixelXWarp0 = aSplit.MutedAndLeft(); pixelXWarp0 < aSplit.MutedAndLeftAndCenter();
             pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelCenter(pixelXWarp0 + warpLaneID);

            Tupel<SrcT, TupelSize> tupelSrc;
            Tupel<DstT, TupelSize> tupelDst;
            Tupel<DstSqrT, TupelSize> tupelDstSqr;

            if (pixelY > 0)
            {
                if (pixelX == 0)
                {
                    tupelSrc.value[0] = SrcT(0);
                    // don't load the zero padded pixel, as it doesn't exist in source image:
#pragma unroll
                    for (size_t i = 1; i < TupelSize; i++)
                    {
                        tupelSrc.value[i] = *gotoPtr(aSrc, aPitchSrc, pixelX - 1 + (int)i, pixelY - 1);
                    }
                }
                else
                {
                    const SrcT *pixelSrc = gotoPtr(aSrc, aPitchSrc, pixelX - 1, pixelY - 1);
                    tupelSrc             = Tupel<SrcT, TupelSize>::Load(pixelSrc);
                }

                tupelDst.value[0]    = DstT(tupelSrc.value[0]);
                tupelDstSqr.value[0] = DstSqrT(tupelSrc.value[0]) * DstSqrT(tupelSrc.value[0]);
#pragma unroll
                for (size_t i = 1; i < TupelSize; i++)
                {
                    tupelDst.value[i] = tupelDst.value[i - 1] + DstT(tupelSrc.value[i]);
                    tupelDstSqr.value[i] =
                        tupelDstSqr.value[i - 1] + DstSqrT(tupelSrc.value[i]) * DstSqrT(tupelSrc.value[i]);
                }

                doShuffleUpTupel<TupelSize, DstT>(warpLaneID, tupelDst);
                doShuffleUpTupel<TupelSize, DstSqrT>(warpLaneID, tupelDstSqr);

#pragma unroll
                for (size_t i = 0; i < TupelSize; i++)
                {
                    tupelDst.value[i] += previousWarp;
                    tupelDstSqr.value[i] += previousSqrWarp;
                }

                previousWarp.x    = __shfl_sync(0xFFFFFFFF, tupelDst.value[TupelSize - 1].x, warpSize - 1);
                previousSqrWarp.x = __shfl_sync(0xFFFFFFFF, tupelDstSqr.value[TupelSize - 1].x, warpSize - 1);
                if constexpr (vector_active_size_v<SrcT> > 1)
                {
                    previousWarp.y    = __shfl_sync(0xFFFFFFFF, tupelDst.value[TupelSize - 1].y, warpSize - 1);
                    previousSqrWarp.y = __shfl_sync(0xFFFFFFFF, tupelDstSqr.value[TupelSize - 1].y, warpSize - 1);
                }
                if constexpr (vector_active_size_v<SrcT> > 2)
                {
                    previousWarp.z    = __shfl_sync(0xFFFFFFFF, tupelDst.value[TupelSize - 1].z, warpSize - 1);
                    previousSqrWarp.z = __shfl_sync(0xFFFFFFFF, tupelDstSqr.value[TupelSize - 1].z, warpSize - 1);
                }
                if constexpr (vector_active_size_v<SrcT> > 3)
                {
                    previousWarp.w    = __shfl_sync(0xFFFFFFFF, tupelDst.value[TupelSize - 1].w, warpSize - 1);
                    previousSqrWarp.w = __shfl_sync(0xFFFFFFFF, tupelDstSqr.value[TupelSize - 1].w, warpSize - 1);
                }
            }
            else
            {
#pragma unroll
                for (size_t i = 0; i < TupelSize; i++)
                {
                    tupelDst.value[i]    = previousWarp;
                    tupelDstSqr.value[i] = previousSqrWarp;
                }
                previousWarp    = DstT(0);
                previousSqrWarp = DstSqrT(0);
            }
            if (pixelX < aSize.x)
            {
                DstT *pixelDst = gotoPtr(aDst, aPitchDst, pixelX, pixelY);
                Tupel<DstT, TupelSize>::StoreAligned(tupelDst, pixelDst);

                DstSqrT *pixelDstSqr = gotoPtr(aDstSqr, aPitchDstSqr, pixelX, pixelY);
                Tupel<DstSqrT, TupelSize>::StoreAligned(tupelDstSqr, pixelDstSqr);
            }
        }

        // compute right unaligned part:
        for (int pixelXWarp0 = aSplit.MutedAndLeftAndCenter(); pixelXWarp0 < aSplit.Total(); pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelRight(pixelXWarp0 + warpLaneID);
            DstT resVal(0);
            DstSqrT resSqrVal(0);
            if (pixelY > 0)
            {
                if (pixelX < aSize.x) // i.e. thread is active
                {
                    if (pixelX > 0)
                    {
                        // pixelX - 1 do add 0 border at the left:
                        SrcT pixel                = *gotoPtr(aSrc, aPitchSrc, pixelX - 1, pixelY - 1);
                        const DstT pixelSrc       = DstT(pixel);
                        const DstSqrT pixelSqrSrc = DstSqrT(pixel).Sqr();
                        resVal += pixelSrc;
                        resSqrVal += pixelSqrSrc;
                    }
                }

                doShuffleUp(warpLaneID, resVal.x);
                doShuffleUp(warpLaneID, resSqrVal.x);
                if constexpr (vector_active_size_v<SrcT> > 1)
                {
                    doShuffleUp(warpLaneID, resVal.y);
                    doShuffleUp(warpLaneID, resSqrVal.y);
                }
                if constexpr (vector_active_size_v<SrcT> > 2)
                {
                    doShuffleUp(warpLaneID, resVal.z);
                    doShuffleUp(warpLaneID, resSqrVal.z);
                }
                if constexpr (vector_active_size_v<SrcT> > 3)
                {
                    doShuffleUp(warpLaneID, resVal.w);
                    doShuffleUp(warpLaneID, resSqrVal.w);
                }

                resVal += previousWarp;
                resSqrVal += previousSqrWarp;

                previousWarp.x    = __shfl_sync(0xFFFFFFFF, resVal.x, warpSize - 1);
                previousSqrWarp.x = __shfl_sync(0xFFFFFFFF, resSqrVal.x, warpSize - 1);
                if constexpr (vector_active_size_v<SrcT> > 1)
                {
                    previousWarp.y    = __shfl_sync(0xFFFFFFFF, resVal.y, warpSize - 1);
                    previousSqrWarp.y = __shfl_sync(0xFFFFFFFF, resSqrVal.y, warpSize - 1);
                }
                if constexpr (vector_active_size_v<SrcT> > 2)
                {
                    previousWarp.z    = __shfl_sync(0xFFFFFFFF, resVal.z, warpSize - 1);
                    previousSqrWarp.z = __shfl_sync(0xFFFFFFFF, resSqrVal.z, warpSize - 1);
                }
                if constexpr (vector_active_size_v<SrcT> > 3)
                {
                    previousWarp.w    = __shfl_sync(0xFFFFFFFF, resVal.w, warpSize - 1);
                    previousSqrWarp.w = __shfl_sync(0xFFFFFFFF, resSqrVal.w, warpSize - 1);
                }
            }
            else
            {
                resVal          = previousWarp;
                previousWarp    = DstT(0);
                resSqrVal       = previousSqrWarp;
                previousSqrWarp = DstT(0);
            }
            if (pixelX < aSize.x)
            {
                DstT *pixelDst       = gotoPtr(aDst, aPitchDst, pixelX, pixelY);
                *pixelDst            = resVal;
                DstSqrT *pixelDstSqr = gotoPtr(aDstSqr, aPitchDstSqr, pixelX, pixelY);
                *pixelDstSqr         = resSqrVal;
            }
        }
    }
}

template <typename SrcT, typename DstT, class DstSqrT, size_t TupelSize, int WarpAlignmentInBytes>
void InvokeIntegralSqrXKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                              const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst, DstSqrT *aDstSqr,
                              size_t aPitchDstSqr, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(1, DIV_UP(aSize.y, aBlockSize.y), 1);

    integralSqrXKernel<TupelSize, WarpAlignmentInBytes, SrcT, DstT, DstSqrT>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrc, aPitchSrc, aDst, aPitchDst, aDstSqr, aPitchDstSqr,
                                                                aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename SrcT, typename DstT, class DstSqrT, size_t TupelSize>
void InvokeIntegralSqrXKernelDefault(const SrcT *aSrc, size_t aPitchSrc, DstT *aDst, size_t aPitchDst, DstSqrT *aDstSqr,
                                     size_t aPitchDstSqr, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize = {32, 16, 1};
        // ConfigBlockSize<"DefaultReductionX">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeIntegralSqrXKernel<SrcT, DstT, DstSqrT, TupelSize, WarpAlignmentInBytes>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrc, aPitchSrc, aDst, aPitchDst, aDstSqr,
            aPitchDstSqr, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(integralSqrXKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
