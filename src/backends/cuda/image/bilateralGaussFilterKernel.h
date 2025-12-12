#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/filterArea.h>
#include <common/image/functors/borderControl.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/mpp_defs.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vector2.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{

template <typename ComputeT, Norm norm>
__device__ __forceinline__ float getWeight(const ComputeT &aPixel00, const ComputeT &aPixel, float aValSquareSigma)
{
    ComputeT diff = aPixel - aPixel00;
    float dist    = 0;
    if constexpr (vector_active_size_v<ComputeT> > 1)
    {
        if constexpr (norm == Norm::L2)
        {
            diff.Sqr();
        }
        else
        {
            diff.Abs();
        }
        dist = diff.x;
        dist += diff.y;
        if constexpr (vector_active_size_v<ComputeT> > 2)
        {
            dist += diff.z;
        }
        if constexpr (vector_active_size_v<ComputeT> > 3)
        {
            dist += diff.w;
        }
    }
    else
    {
        dist = diff.x;
    }

    if constexpr (norm == Norm::L1)
    {
        dist *= dist;
    }

    return __expf(-dist / (2.0f * aValSquareSigma));
}

/// <summary>
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, int blockWidth, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, Norm norm>
__global__ void bilateralGaussFilterKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDst, size_t aPitchDst,
                                           FilterArea aFilterArea, const Pixel32fC1 *__restrict__ aPreCompGeomDistCoeff,
                                           float aValSquareSigma, Size2D aSize,
                                           ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    RoundFunctor<roundingMode, ComputeT> round;

    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    // in case of tuples, blockWidth == 1, and ThreadSplit handles the thread limits:
    if constexpr (TupelSize > 1)
    {
        if (aSplit.ThreadIsOutsideOfRange(threadX) || threadY >= DIV_UP(aSize.y, blockHeight))
        {
            return;
        }
    }
    else
    {
        if (threadX >= DIV_UP(aSize.x, blockWidth) || threadY >= DIV_UP(aSize.y, blockHeight))
        {
            return;
        }
    }

    const int pixelX = aSplit.GetPixel(threadX) * blockWidth;
    const int pixelY = threadY * blockHeight;

    // don't check for warp alignment if TupelSize <= 1
    if constexpr (TupelSize > 1) // evaluated at compile time
    {
        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            ComputeT pixel00[blockHeight][TupelSize];
            ComputeT result[blockHeight][TupelSize]  = {0};
            float sumWeights[blockHeight][TupelSize] = {0};

#pragma unroll
            for (int by = 0; by < blockHeight; by++)
            {
                const int srcPixelY0 = by + pixelY;
#pragma unroll
                for (int bx = 0; bx < TupelSize; bx++)
                {
                    const int srcPixelX0 = bx + pixelX;
                    pixel00[by][bx]      = ComputeT(aSrcWithBC(srcPixelX0, srcPixelY0));
                }
            }

            const int extendedBlockW = TupelSize + (aFilterArea.Size.x - 1);
            const int extendedBlockH = blockHeight + (aFilterArea.Size.y - 1);

            for (int ry = 0; ry < extendedBlockH; ry++)
            {
                const int srcPixelY = ry - aFilterArea.Center.y + pixelY;

                for (int rx = 0; rx < extendedBlockW; rx++)
                {
                    const int srcPixelX = pixelX - aFilterArea.Center.x + rx;
                    ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int by = 0; by < blockHeight; by++)
                    {
                        const int filterY = ry - by;
#pragma unroll
                        for (int bx = 0; bx < TupelSize; bx++)
                        {
                            const int filterX = rx - bx;
                            if (filterY >= 0 && filterY < aFilterArea.Size.y && filterX >= 0 &&
                                filterX < aFilterArea.Size.x)
                            {
                                const float wColor =
                                    getWeight<ComputeT, norm>(pixel00[by][bx], srcPixel, aValSquareSigma);
                                const float wDist  = aPreCompGeomDistCoeff[filterY * aFilterArea.Size.x + filterX].x;
                                const float weight = wDist * wColor;
                                sumWeights[by][bx] += weight;
                                result[by][bx] += srcPixel * weight;
                            }
                        }
                    }
                }
            }

#pragma unroll
            for (int bl = 0; bl < blockHeight; bl++)
            {
                if (pixelY + bl < aSize.y)
                {
                    DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX, pixelY + bl);
                    Tupel<DstT, TupelSize> res;

#pragma unroll
                    for (size_t t = 0; t < TupelSize; t++)
                    {
                        result[bl][t] /= sumWeights[bl][t];

                        if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                        {
                            round(result[bl][t]);
                        }

                        res.value[t] = DstT(result[bl][t]);
                        // restore alpha channel values:
                        if constexpr (has_alpha_channel_v<DstT>)
                        {
                            res.value[t].w = (pixelsOut + t)->w;
                        }
                    }

                    Tupel<DstT, TupelSize>::StoreAligned(res, pixelsOut);
                }
            }
            return;
        }
    }

    {
        ComputeT pixel00[blockHeight][blockWidth];
        ComputeT result[blockHeight][blockWidth]  = {0};
        float sumWeights[blockHeight][blockWidth] = {0};

#pragma unroll
        for (int by = 0; by < blockHeight; by++)
        {
            const int srcPixelY0 = by + pixelY;
#pragma unroll
            for (int bx = 0; bx < blockWidth; bx++)
            {
                const int srcPixelX0 = bx + pixelX;
                pixel00[by][bx]      = ComputeT(aSrcWithBC(srcPixelX0, srcPixelY0));
            }
        }
        const int extendedBlockW = blockWidth + (aFilterArea.Size.x - 1);
        const int extendedBlockH = blockHeight + (aFilterArea.Size.y - 1);

        for (int ry = 0; ry < extendedBlockH; ry++)
        {
            const int srcPixelY = ry - aFilterArea.Center.y + pixelY;

            for (int rx = 0; rx < extendedBlockW; rx++)
            {
                const int srcPixelX = pixelX - aFilterArea.Center.x + rx;
                ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                for (int by = 0; by < blockHeight; by++)
                {
                    const int filterY = ry - by;
#pragma unroll
                    for (int bx = 0; bx < blockWidth; bx++)
                    {
                        const int filterX = rx - bx;
                        if (filterY >= 0 && filterY < aFilterArea.Size.y && filterX >= 0 &&
                            filterX < aFilterArea.Size.x)
                        {
                            const float wColor = getWeight<ComputeT, norm>(pixel00[by][bx], srcPixel, aValSquareSigma);
                            const float wDist  = aPreCompGeomDistCoeff[filterY * aFilterArea.Size.x + filterX].x;
                            const float weight = wDist * wColor;
                            sumWeights[by][bx] += weight;
                            result[by][bx] += srcPixel * weight;
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int bl = 0; bl < blockHeight; bl++)
        {
            if (pixelY + bl < aSize.y)
            {
#pragma unroll
                for (int bc = 0; bc < blockWidth; bc++)
                {
                    if (pixelX + bc < aSize.x)
                    {
                        DstT *pixelsOut = gotoPtr(aDst, aPitchDst, pixelX + bc, pixelY + bl);
                        DstT res;

                        result[bl][bc] /= sumWeights[bl][bc];

                        if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                        {
                            round(result[bl][bc]);
                        }

                        res = DstT(result[bl][bc]);

                        // restore alpha channel values:
                        if constexpr (has_alpha_channel_v<DstT>)
                        {
                            res.w = pixelsOut->w;
                        }
                        *pixelsOut = res;
                    }
                }
            }
        }
    }
    return;
}

template <typename ComputeT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, int blockWidth, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, Norm norm>
void InvokeBilateralGaussFilterKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                      const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                      const FilterArea &aFilterArea, const Pixel32fC1 *aPreCompGeomDistCoeff,
                                      float aValSquareSigma, const Size2D &aSize)
{
    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(aDst, aSize.x, aWarpSize);

    dim3 blocksPerGrid(DIV_UP(ts.Total() / blockWidth, aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    bilateralGaussFilterKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, blockWidth, blockHeight, roundingMode,
                               BorderControlT, norm><<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(
        aSrcWithBC, aDst, aPitchDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, int blockWidth, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, Norm norm>
void InvokeBilateralGaussFilterKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDst, size_t aPitchDst,
                                             const FilterArea &aFilterArea, const Pixel32fC1 *aPreCompGeomDistCoeff,
                                             float aValSquareSigma, const Size2D &aSize,
                                             const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        dim3 BlockSize = ConfigBlockSize<"Default">::value;

        if (blockHeight > 1)
        {
            BlockSize.y = 2;
        }

        constexpr int WarpAlignmentInBytes = ConfigWarpAlignment<"Default">::value;
        constexpr uint SharedMemory        = 0;

        InvokeBilateralGaussFilterKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, blockWidth, blockHeight,
                                         roundingMode, BorderControlT, norm>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDst, aPitchDst, aFilterArea,
            aPreCompGeomDistCoeff, aValSquareSigma, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(bilateralGaussFilterKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
