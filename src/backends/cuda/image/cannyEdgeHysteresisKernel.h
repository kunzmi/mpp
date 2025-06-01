#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

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
#include <numbers>

namespace opp::image::cuda
{
template <class DstT, typename BorderControlT>
__device__ DstT processOnePixel(int pixelX, int pixelY, BorderControlT &aSrcWithBC,
                                const Pixel32fC1 *__restrict__ aSrcAngle, size_t aPitchSrcAngle)
{
    constexpr DstT Nothing = 0;
    constexpr DstT Weak    = 1;
    constexpr DstT Strong  = 255;

    DstT pixel = aSrcWithBC(pixelX, pixelY);
    if (pixel == Nothing)
    {
        return Nothing;
    }
    if (pixel == Strong)
    {
        return Strong;
    }

    Pixel32fC1 angle = *gotoPtr(aSrcAngle, aPitchSrcAngle, pixelX, pixelY);

    // get quantized direction from angle, angles from atan2-function are given in range -pi..pi
    // map this range to 0..3, where 0 = horizontal, 1 = 45deg diagonal, 2 = vertical, 3 = -45deg = 135deg diagonal
    angle.x = round((angle.x / std::numbers::pi_v<float> * 180.0f + 180.0f) / 45.0f);
    int dir = static_cast<int>(angle.x) % 4; // the modulo maps the negative / opposite direction to the positive one

    int pixelMinus = 0;
    int pixelPlus  = 0;
    switch (dir)
    {
        case 0:
            // gradient horizontal direction -> check in Y direction
            pixelMinus = aSrcWithBC(pixelX, pixelY - 1).x;
            pixelPlus  = aSrcWithBC(pixelX, pixelY + 1).x;
            break;
        case 1:
            // gradient +45deg -> check in -45deg direction
            pixelMinus = aSrcWithBC(pixelX - 1, pixelY - 1).x;
            pixelPlus  = aSrcWithBC(pixelX + 1, pixelY + 1).x;
            break;
        case 2:
            // gradient vertical direction -> check in X direction
            pixelMinus = aSrcWithBC(pixelX - 1, pixelY).x;
            pixelPlus  = aSrcWithBC(pixelX + 1, pixelY).x;
            break;
        case 3:
            // gradient -45deg=135deg -> check in +45deg direction
            pixelMinus = aSrcWithBC(pixelX - 1, pixelY + 1).x;
            pixelPlus  = aSrcWithBC(pixelX + 1, pixelY - 1).x;
            break;
        default:
            break;
    }

    if (pixelMinus + pixelPlus >= Strong.x)
    {
        // at least one of the neighbor pixels is at least "Strong"
        return Strong;
    }
    return Nothing;
}

/// <summary>
/// runs canny edge max supression on every pixel of an image.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class DstT, typename BorderControlT>
__global__ void cannyEdgeHysteresisKernel(BorderControlT aSrcWithBC, const Pixel32fC1 *__restrict__ aSrcAngle,
                                          size_t aPitchSrcAngle, DstT *__restrict__ aDst, size_t aPitchDst,
                                          Size2D aSize, ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
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
        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            Tupel<DstT, TupelSize> res;

#pragma unroll
            for (size_t t = 0; t < TupelSize; t++)
            {
                res.value[t] =
                    processOnePixel<DstT, BorderControlT>(pixelX + t, pixelY, aSrcWithBC, aSrcAngle, aPitchSrcAngle);
            }

            DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX, pixelY);
            Tupel<DstT, TupelSize>::StoreAligned(res, pixelsOut);
            return;
        }
    }

    DstT *pixelOut = gotoPtr(aDst, aPitchDst, pixelX, pixelY);
    *pixelOut      = processOnePixel<DstT, BorderControlT>(pixelX, pixelY, aSrcWithBC, aSrcAngle, aPitchSrcAngle);

    return;
}

template <typename DstT, size_t TupelSize, int WarpAlignmentInBytes, typename BorderControlT>
void InvokeCannyEdgeHysteresisKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                     const BorderControlT &aSrcWithBC, const Pixel32fC1 *aSrcAngle,
                                     size_t aPitchSrcAngle, DstT *aDst, size_t aPitchDst, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y, aBlockSize.y), 1);

    cannyEdgeHysteresisKernel<WarpAlignmentInBytes, TupelSize, DstT, BorderControlT>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrcWithBC, aSrcAngle, aPitchSrcAngle, aDst, aPitchDst,
                                                                aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename DstT, size_t TupelSize, typename BorderControlT>
void InvokeCannyEdgeHysteresisKernelDefault(const BorderControlT &aSrcWithBC, const Pixel32fC1 *aSrcAngle,
                                            size_t aPitchSrcAngle, DstT *aDst, size_t aPitchDst, const Size2D &aSize,
                                            const opp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize               = ConfigBlockSize<"Default">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeCannyEdgeHysteresisKernel<DstT, TupelSize, WarpAlignmentInBytes, BorderControlT>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aSrcAngle, aPitchSrcAngle,
            aDst, aPitchDst, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(cannyEdgeHysteresisKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND