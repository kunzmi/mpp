#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/srcReductionMinMaxIdxFunctor.h>
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

namespace mpp::image::cuda
{
/// <summary>
/// runs MinMax-Index reduction on every image line.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class SrcT>
__global__ void reductionMaskedMinMaxIdxAlongXKernel(
    const byte *__restrict__ aMask, size_t aPitchMask, const SrcT *aSrc, size_t aPitchSrc, SrcT *__restrict__ aDstMin,
    SrcT *__restrict__ aDstMax, same_vector_size_different_type_t<SrcT, int> *__restrict__ aDstMinIdx,
    same_vector_size_different_type_t<SrcT, int> *__restrict__ aDstMaxIdx, Size2D aSize,
    ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    using idxT = same_vector_size_different_type_t<SrcT, int>;

    int warpLaneID = threadIdx.x;
    int pixelY     = blockIdx.y * blockDim.y + threadIdx.y;

    mpp::MinIdx<SrcT> redOpMin;
    mpp::MaxIdx<SrcT> redOpMax;
    SrcReductionMinMaxIdxFunctor<TupelSize, SrcT> functor(aSrc, aPitchSrc);

    SrcT resultMin(reduction_init_value_v<ReductionInitValue::Max, SrcT>);
    SrcT resultMax(reduction_init_value_v<ReductionInitValue::Min, SrcT>);
    idxT resultMinIdx(INT_MAX);
    idxT resultMaxIdx(INT_MAX);

    if (pixelY >= aSize.y)
    {
        return;
    }

    // simple case, no tupels
    if constexpr (TupelSize == 1)
    {
        // loop over x-dimension in warp steps:
        for (int pixelXWarp0 = 0; pixelXWarp0 < aSize.x; pixelXWarp0 += warpSize)
        {
            const int pixelX = pixelXWarp0 + warpLaneID;
            if (pixelX < aSize.x)
            {
                const byte mask = *gotoPtr(aMask, aPitchMask, pixelX, pixelY);
                if (mask)
                {
                    functor(pixelX, pixelY, resultMin, resultMax, resultMinIdx, resultMaxIdx);
                }
            }
        }
    }
    else
    {
        // compute left unaligned part:
        for (int pixelXWarp0 = 0; pixelXWarp0 < aSplit.MutedAndLeft(); pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelLeft(pixelXWarp0 + warpLaneID);
            if (pixelX >= 0) // i.e. thread is active
            {
                const byte mask = *gotoPtr(aMask, aPitchMask, pixelX, pixelY);
                if (mask)
                {
                    functor(pixelX, pixelY, resultMin, resultMax, resultMinIdx, resultMaxIdx);
                }
            }
        }

        // computer center part as tupels:
        for (int pixelXWarp0 = aSplit.MutedAndLeft(); pixelXWarp0 < aSplit.MutedAndLeftAndCenter();
             pixelXWarp0 += warpSize)
        {
            const int pixelX               = aSplit.GetPixelCenter(pixelXWarp0 + warpLaneID);
            const byte *pixelsMask         = gotoPtr(aMask, aPitchMask, pixelX, pixelY);
            MaskTupel<TupelSize> maskTupel = MaskTupel<TupelSize>::Load(pixelsMask);
            if (maskTupel.AreAllFalse())
            {
                // nothing to do for these pixels
                continue;
            }
            // center part is always aligned to warpSize, no need to check
            functor(pixelX, pixelY, resultMin, resultMax, resultMinIdx, resultMaxIdx, maskTupel);
        }

        // compute right unaligned part:
        for (int pixelXWarp0 = aSplit.MutedAndLeftAndCenter(); pixelXWarp0 < aSplit.Total(); pixelXWarp0 += warpSize)
        {
            const int pixelX = aSplit.GetPixelRight(pixelXWarp0 + warpLaneID);
            if (pixelX < aSize.x) // i.e. thread is active
            {
                const byte mask = *gotoPtr(aMask, aPitchMask, pixelX, pixelY);
                if (mask)
                {
                    functor(pixelX, pixelY, resultMin, resultMax, resultMinIdx, resultMaxIdx);
                }
            }
        }
    }

    // reduce over warp:
    redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.x, 16), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.x, 16),
             resultMin.x, resultMinIdx.x);
    redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.x, 8), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.x, 8), resultMin.x,
             resultMinIdx.x);
    redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.x, 4), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.x, 4), resultMin.x,
             resultMinIdx.x);
    redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.x, 2), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.x, 2), resultMin.x,
             resultMinIdx.x);
    redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.x, 1), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.x, 1), resultMin.x,
             resultMinIdx.x);

    redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.x, 16), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.x, 16),
             resultMax.x, resultMaxIdx.x);
    redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.x, 8), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.x, 8), resultMax.x,
             resultMaxIdx.x);
    redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.x, 4), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.x, 4), resultMax.x,
             resultMaxIdx.x);
    redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.x, 2), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.x, 2), resultMax.x,
             resultMaxIdx.x);
    redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.x, 1), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.x, 1), resultMax.x,
             resultMaxIdx.x);

    if constexpr (vector_active_size_v<SrcT> > 1)
    {
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.y, 16), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.y, 16),
                 resultMin.y, resultMinIdx.y);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.y, 8), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.y, 8),
                 resultMin.y, resultMinIdx.y);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.y, 4), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.y, 4),
                 resultMin.y, resultMinIdx.y);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.y, 2), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.y, 2),
                 resultMin.y, resultMinIdx.y);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.y, 1), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.y, 1),
                 resultMin.y, resultMinIdx.y);

        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.y, 16), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.y, 16),
                 resultMax.y, resultMaxIdx.y);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.y, 8), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.y, 8),
                 resultMax.y, resultMaxIdx.y);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.y, 4), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.y, 4),
                 resultMax.y, resultMaxIdx.y);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.y, 2), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.y, 2),
                 resultMax.y, resultMaxIdx.y);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.y, 1), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.y, 1),
                 resultMax.y, resultMaxIdx.y);
    }
    if constexpr (vector_active_size_v<SrcT> > 2)
    {
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.z, 16), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.z, 16),
                 resultMin.z, resultMinIdx.z);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.z, 8), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.z, 8),
                 resultMin.z, resultMinIdx.z);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.z, 4), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.z, 4),
                 resultMin.z, resultMinIdx.z);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.z, 2), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.z, 2),
                 resultMin.z, resultMinIdx.z);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.z, 1), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.z, 1),
                 resultMin.z, resultMinIdx.z);

        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.z, 16), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.z, 16),
                 resultMax.z, resultMaxIdx.z);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.z, 8), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.z, 8),
                 resultMax.z, resultMaxIdx.z);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.z, 4), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.z, 4),
                 resultMax.z, resultMaxIdx.z);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.z, 2), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.z, 2),
                 resultMax.z, resultMaxIdx.z);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.z, 1), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.z, 1),
                 resultMax.z, resultMaxIdx.z);
    }
    if constexpr (vector_active_size_v<SrcT> > 3)
    {
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.w, 16), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.w, 16),
                 resultMin.w, resultMinIdx.w);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.w, 8), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.w, 8),
                 resultMin.w, resultMinIdx.w);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.w, 4), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.w, 4),
                 resultMin.w, resultMinIdx.w);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.w, 2), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.w, 2),
                 resultMin.w, resultMinIdx.w);
        redOpMin(__shfl_down_sync(0xFFFFFFFF, resultMin.w, 1), __shfl_down_sync(0xFFFFFFFF, resultMinIdx.w, 1),
                 resultMin.w, resultMinIdx.w);

        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.w, 16), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.w, 16),
                 resultMax.w, resultMaxIdx.w);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.w, 8), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.w, 8),
                 resultMax.w, resultMaxIdx.w);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.w, 4), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.w, 4),
                 resultMax.w, resultMaxIdx.w);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.w, 2), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.w, 2),
                 resultMax.w, resultMaxIdx.w);
        redOpMax(__shfl_down_sync(0xFFFFFFFF, resultMax.w, 1), __shfl_down_sync(0xFFFFFFFF, resultMaxIdx.w, 1),
                 resultMax.w, resultMaxIdx.w);
    }

    // In other X-reduction kernels we further do a reduction over the block also in Y, but we won't do this for the
    // Index-kernels, in order to maintain the Y position.
    if (warpLaneID == 0)
    {
        aDstMin[pixelY]    = resultMin;
        aDstMax[pixelY]    = resultMax;
        aDstMinIdx[pixelY] = resultMinIdx;
        aDstMaxIdx[pixelY] = resultMaxIdx;
    }
}

template <typename SrcT, size_t TupelSize, int WarpAlignmentInBytes>
void InvokeReductionMaskedMinMaxIdxAlongXKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize,
                                                cudaStream_t aStream, const byte *aMask, size_t aPitchMask,
                                                const SrcT *aSrc, size_t aPitchSrc, SrcT *aDstMin, SrcT *aDstMax,
                                                same_vector_size_different_type_t<SrcT, int> *aDstMinIdx,
                                                same_vector_size_different_type_t<SrcT, int> *aDstMaxIdx, Size2D aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aSrc, aSize.x, aWarpSize);

    dim3 blocksPerGrid(1, DIV_UP(aSize.y, aBlockSize.y), 1);

    reductionMaskedMinMaxIdxAlongXKernel<WarpAlignmentInBytes, TupelSize>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aMask, aPitchMask, aSrc, aPitchSrc, aDstMin, aDstMax,
                                                                aDstMinIdx, aDstMaxIdx, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename SrcT, size_t TupelSize>
void InvokeReductionMaskedMinMaxIdxAlongXKernelDefault(const Pixel8uC1 *aMask, size_t aPitchMask, const SrcT *aSrc,
                                                       size_t aPitchSrc, SrcT *aDstMin, SrcT *aDstMax,
                                                       same_vector_size_different_type_t<SrcT, int> *aDstMinIdx,
                                                       same_vector_size_different_type_t<SrcT, int> *aDstMaxIdx,
                                                       const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize               = ConfigBlockSize<"DefaultReductionX">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeReductionMaskedMinMaxIdxAlongXKernel<SrcT, TupelSize, WarpAlignmentInBytes>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, reinterpret_cast<const byte *>(aMask),
            aPitchMask, aSrc, aPitchSrc, aDstMin, aDstMax, aDstMinIdx, aDstMaxIdx, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(reductionMaskedMinMaxIdxAlongXKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND