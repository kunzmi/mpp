#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/exception.h>
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
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 3 channel destination, for
/// Tupled operation, all Dst-pointers must fulfill the same alignment constraints.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class DstT, typename funcType>
__global__ void forEachPixelPlanar3Kernel(Vector1<remove_vector_t<DstT>> *__restrict__ aDst1, size_t aPitchDst1,
                                          Vector1<remove_vector_t<DstT>> *__restrict__ aDst2, size_t aPitchDst2,
                                          Vector1<remove_vector_t<DstT>> *__restrict__ aDst3, size_t aPitchDst3,
                                          Size2D aSize, ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit,
                                          funcType aFunctor)
{
    using DstPlaneT = Vector1<remove_vector_t<DstT>>;
    int threadX     = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY     = blockIdx.y * blockDim.y + threadIdx.y;

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
            Tupel<DstPlaneT, TupelSize> resPlane;

            DstPlaneT *pixelsOut1 = gotoPtr(aDst1, aPitchDst1, pixelX, pixelY);
            DstPlaneT *pixelsOut2 = gotoPtr(aDst2, aPitchDst2, pixelX, pixelY);
            DstPlaneT *pixelsOut3 = gotoPtr(aDst3, aPitchDst3, pixelX, pixelY);

            // load the destination pixel in case of inplace operation:
            if constexpr (funcType::DoLoadBeforeOp)
            {
                resPlane = Tupel<DstPlaneT, TupelSize>::LoadAligned(pixelsOut1);
#pragma unroll
                for (size_t i = 0; i < TupelSize; i++)
                {
                    res.value[i].x = resPlane.value[i].x;
                }
                resPlane = Tupel<DstPlaneT, TupelSize>::LoadAligned(pixelsOut2);
#pragma unroll
                for (size_t i = 0; i < TupelSize; i++)
                {
                    res.value[i].y = resPlane.value[i].x;
                }
                resPlane = Tupel<DstPlaneT, TupelSize>::LoadAligned(pixelsOut3);
#pragma unroll
                for (size_t i = 0; i < TupelSize; i++)
                {
                    res.value[i].z = resPlane.value[i].x;
                }
            }

            aFunctor(pixelX, pixelY, res);

#pragma unroll
            for (size_t i = 0; i < TupelSize; i++)
            {
                resPlane.value[i].x = res.value[i].x;
            }
            Tupel<DstPlaneT, TupelSize>::StoreAligned(resPlane, pixelsOut1);
#pragma unroll
            for (size_t i = 0; i < TupelSize; i++)
            {
                resPlane.value[i].x = res.value[i].y;
            }
            Tupel<DstPlaneT, TupelSize>::StoreAligned(resPlane, pixelsOut2);
#pragma unroll
            for (size_t i = 0; i < TupelSize; i++)
            {
                resPlane.value[i].x = res.value[i].z;
            }
            Tupel<DstPlaneT, TupelSize>::StoreAligned(resPlane, pixelsOut3);
            return;
        }
    }

    DstT res;
    DstPlaneT *pixelOut1 = gotoPtr(aDst1, aPitchDst1, pixelX, pixelY);
    DstPlaneT *pixelOut2 = gotoPtr(aDst2, aPitchDst2, pixelX, pixelY);
    DstPlaneT *pixelOut3 = gotoPtr(aDst3, aPitchDst3, pixelX, pixelY);

    // load the destination pixel in case of inplace operation:
    if constexpr (funcType::DoLoadBeforeOp)
    {
        res.x = pixelOut1->x;
        res.y = pixelOut2->x;
        res.z = pixelOut3->x;
    }

    aFunctor(pixelX, pixelY, res);

    *pixelOut1 = res.x;
    *pixelOut2 = res.y;
    *pixelOut3 = res.z;
    return;
}

template <typename DstT, size_t TupelSize, int WarpAlignmentInBytes, typename funcType>
void InvokeForEachPixelPlanar3Kernel(const dim3 &aBlockSize, uint aSharedMemory, cudaStream_t aStream,
                                     Vector1<remove_vector_t<DstT>> *__restrict__ aDst1, size_t aPitchDst1,
                                     Vector1<remove_vector_t<DstT>> *__restrict__ aDst2, size_t aPitchDst2,
                                     Vector1<remove_vector_t<DstT>> *__restrict__ aDst3, size_t aPitchDst3,
                                     const Size2D &aSize, const funcType &aFunctor)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst1, aSize.x);

    if (TupelSize != 1)
    {
        ThreadSplit<WarpAlignmentInBytes, TupelSize> tsCheck(aDst2, aSize.x);
        if (ts != tsCheck)
        {
            throw INVALIDARGUMENT(
                aDst1 and aDst2,
                "All destination images must fulfill the same byte-alignments in order to use tupeled memory access.");
        }
        ThreadSplit<WarpAlignmentInBytes, TupelSize> tsCheck2(aDst3, aSize.x);
        if (ts != tsCheck2)
        {
            throw INVALIDARGUMENT(
                aDst1 and aDst3,
                "All destination images must fulfill the same byte-alignments in order to use tupeled memory access.");
        }
    }

    dim3 blocksPerGrid(DIV_UP(ts.Total(), aBlockSize.x), DIV_UP(aSize.y, aBlockSize.y), 1);

    forEachPixelPlanar3Kernel<WarpAlignmentInBytes, TupelSize, DstT, funcType>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3,
                                                                aSize, ts, aFunctor);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename DstT, size_t TupelSize, typename funcType>
void InvokeForEachPixelPlanar3KernelDefault(Vector1<remove_vector_t<DstT>> *__restrict__ aDst1, size_t aPitchDst1,
                                            Vector1<remove_vector_t<DstT>> *__restrict__ aDst2, size_t aPitchDst2,
                                            Vector1<remove_vector_t<DstT>> *__restrict__ aDst3, size_t aPitchDst3,
                                            const Size2D &aSize, const opp::cuda::StreamCtx &aStreamCtx,
                                            const funcType &aFunc)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize               = ConfigBlockSize<"Default">::value;
        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeForEachPixelPlanar3Kernel<DstT, TupelSize, WarpAlignmentInBytes, funcType>(
            BlockSize, SharedMemory, aStreamCtx.Stream, aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize,
            aFunc);
    }
    else
    {
        throw CUDAUNSUPPORTED(forEachPixelPlanar3Kernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND