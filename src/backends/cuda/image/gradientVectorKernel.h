#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/channel.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/borderControl.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{
/// <summary>
/// Basically the same kernel as FixedFilterKernel, but computes X and Y gradients at the same time.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class ComputeT, class DstT, int kernelWidth, int kernelHeight,
          int kernelCenterX, int kernelCenterY, int blockWidth, int blockHeight, RoundingMode roundingMode,
          typename BorderControlT, typename FixedFilterKernelXT, typename FixedFilterKernelYT>
__global__ void gradientVectorKernel(BorderControlT aSrcWithBC, DstT *__restrict__ aDstX, size_t aPitchDstX,
                                     DstT *__restrict__ aDstY, size_t aPitchDstY, DstT *__restrict__ aDstMag,
                                     size_t aPitchDstMag, Pixel32fC1 *__restrict__ aDstAngle, size_t aPitchDstAngle,
                                     Pixel32fC4 *__restrict__ aDstCovariance, size_t aPitchDstCovariance, Norm aNorm,
                                     Size2D aSize, ThreadSplit<WarpAlignmentInBytes, TupelSize> aSplit)
{
    RoundFunctor<roundingMode, Vector1<remove_vector_t<ComputeT>>> round;
    constexpr FixedFilterKernelXT filterKernelX;
    constexpr FixedFilterKernelYT filterKernelY;

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
        constexpr int extendedBlockW = TupelSize + (kernelWidth - 1);
        constexpr int extendedBlockH = blockHeight + (kernelHeight - 1);

        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            ComputeT gradientX[blockHeight][TupelSize] = {0};
            ComputeT gradientY[blockHeight][TupelSize] = {0};

#pragma unroll
            for (int ry = 0; ry < extendedBlockH; ry++)
            {
                const int srcPixelY = ry - kernelCenterY + pixelY;

#pragma unroll
                for (int rx = 0; rx < extendedBlockW; rx++)
                {
                    const int srcPixelX = pixelX - kernelCenterX + rx;
                    ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));

#pragma unroll
                    for (int ky = 0; ky < kernelHeight; ky++)
                    {
                        const int pixelDstY = ry - ky;

#pragma unroll
                        for (int kx = 0; kx < kernelWidth; kx++)
                        {
                            const int pixelDstX = rx - kx;

                            if (pixelDstY >= 0 && pixelDstY < blockHeight)
                            {
                                if (pixelDstX >= 0 && pixelDstX < TupelSize)
                                {
                                    gradientX[pixelDstY][pixelDstX] += srcPixel * filterKernelX.Values[ky][kx];
                                    gradientY[pixelDstY][pixelDstX] += srcPixel * filterKernelY.Values[ky][kx];
                                }
                            }
                        }
                    }
                }
            }

            // for multi channel, find channel with largest L2 gradient and store it in first channel:
            if constexpr (vector_active_size_v<ComputeT> > 1)
            {
#pragma unroll
                for (int bl = 0; bl < blockHeight; bl++)
                {
#pragma unroll
                    for (size_t t = 0; t < TupelSize; t++)
                    {
                        remove_vector_t<ComputeT> maxMagSqr =
                            gradientX[bl][t].x * gradientX[bl][t].x + gradientY[bl][t].x * gradientY[bl][t].x;
#pragma unroll
                        for (int c = 1; c < vector_active_size_v<ComputeT>; c++)
                        {
                            remove_vector_t<ComputeT> magSqr =
                                gradientX[bl][t][Channel(c)] * gradientX[bl][t][Channel(c)] +
                                gradientY[bl][t][Channel(c)] * gradientY[bl][t][Channel(c)];

                            if (magSqr > maxMagSqr)
                            {
                                maxMagSqr          = magSqr;
                                gradientX[bl][t].x = gradientX[bl][t][Channel(c)];
                                gradientY[bl][t].x = gradientY[bl][t][Channel(c)];
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
                    if (aDstX != nullptr)
                    {
                        DstT *dstX = gotoPtr(aDstX, aPitchDstX, pixelX, pixelY + bl);
                        Tupel<DstT, TupelSize> res;

#pragma unroll
                        for (size_t t = 0; t < TupelSize; t++)
                        {
                            Vector1<remove_vector_t<ComputeT>> temp = gradientX[bl][t].x;
                            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                            {
                                round(temp);
                            }

                            res.value[t] = DstT(temp);
                        }

                        Tupel<DstT, TupelSize>::StoreAligned(res, dstX);
                    }
                    if (aDstY != nullptr)
                    {
                        DstT *dstY = gotoPtr(aDstY, aPitchDstY, pixelX, pixelY + bl);
                        Tupel<DstT, TupelSize> res;

#pragma unroll
                        for (size_t t = 0; t < TupelSize; t++)
                        {
                            Vector1<remove_vector_t<ComputeT>> temp = gradientY[bl][t].x;
                            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                            {
                                round(temp);
                            }

                            res.value[t] = DstT(temp);
                        }

                        Tupel<DstT, TupelSize>::StoreAligned(res, dstY);
                    }
                    if (aDstMag != nullptr)
                    {
                        DstT *dstMag = gotoPtr(aDstMag, aPitchDstMag, pixelX, pixelY + bl);
                        Tupel<DstT, TupelSize> res;

#pragma unroll
                        for (size_t t = 0; t < TupelSize; t++)
                        {
                            Vector1<remove_vector_t<ComputeT>> temp;

                            switch (aNorm)
                            {
                                case Norm::Inf:
                                    temp = max(gradientX[bl][t].x, gradientY[bl][t].x);
                                    break;
                                case Norm::L1:
                                    temp = abs(gradientX[bl][t].x) + abs(gradientY[bl][t].x);
                                    break;
                                case Norm::L2:
                                    temp = sqrt(gradientX[bl][t].x * gradientX[bl][t].x +
                                                gradientY[bl][t].x * gradientY[bl][t].x);
                                    break;
                                default:
                                    // well, that shouldn't happen...
                                    break;
                            }

                            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                            {
                                round(temp);
                            }

                            res.value[t] = DstT(temp);
                        }

                        Tupel<DstT, TupelSize>::StoreAligned(res, dstMag);
                    }
                    if (aDstAngle != nullptr)
                    {
                        if constexpr (std::same_as<DstT, Pixel32fC1>)
                        {
                            // i.e. we can use tupels:
                            DstT *dstAngle = gotoPtr(aDstAngle, aPitchDstAngle, pixelX, pixelY + bl);
                            Tupel<DstT, TupelSize> res;

#pragma unroll
                            for (size_t t = 0; t < TupelSize; t++)
                            {
                                res.value[t] = atan2(gradientY[bl][t].x, gradientX[bl][t].x);
                            }

                            Tupel<DstT, TupelSize>::StoreAligned(res, dstAngle);
                        }
                        else
                        {
                            // write out untupeled:
                            Pixel32fC1 *dstAngle = gotoPtr(aDstAngle, aPitchDstAngle, pixelX, pixelY + bl);

#pragma unroll
                            for (size_t t = 0; t < TupelSize; t++)
                            {
                                *dstAngle = atan2(gradientY[bl][t].x, gradientX[bl][t].x);
                                dstAngle++;
                            }
                        }
                    }
                    if (aDstCovariance != nullptr)
                    {
                        // write out untupeled:
                        Pixel32fC4 *dstCovariance = gotoPtr(aDstCovariance, aPitchDstCovariance, pixelX, pixelY + bl);
#pragma unroll
                        for (size_t t = 0; t < TupelSize; t++)
                        {
                            Pixel32fC4 res;
                            res.x = static_cast<float>(gradientX[bl][t].x) * static_cast<float>(gradientX[bl][t].x);
                            res.y = static_cast<float>(gradientY[bl][t].x) * static_cast<float>(gradientY[bl][t].x);
                            res.z = static_cast<float>(gradientX[bl][t].x) * static_cast<float>(gradientY[bl][t].x);
                            res.w = res.z;
                            *dstCovariance = res;
                            dstCovariance++;
                        }
                    }
                }
            }
            return;
        }
    }

    {
        constexpr int extendedBlockW                = blockWidth + (kernelWidth - 1);
        constexpr int extendedBlockH                = blockHeight + (kernelHeight - 1);
        ComputeT gradientX[blockHeight][blockWidth] = {0};
        ComputeT gradientY[blockHeight][blockWidth] = {0};

#pragma unroll
        for (int ry = 0; ry < extendedBlockH; ry++)
        {
            const int srcPixelY = ry - kernelCenterY + pixelY;

#pragma unroll
            for (int rx = 0; rx < extendedBlockW; rx++)
            {
                const int srcPixelX = pixelX - kernelCenterX + rx;
                ComputeT srcPixel   = ComputeT(aSrcWithBC(srcPixelX, srcPixelY));
#pragma unroll
                for (int ky = 0; ky < kernelHeight; ky++)
                {
                    const int pixelDstY = ry - ky;

#pragma unroll
                    for (int kx = 0; kx < kernelWidth; kx++)
                    {
                        const int pixelDstX = rx - kx;

                        if (pixelDstY >= 0 && pixelDstY < blockHeight)
                        {
                            if (pixelDstX >= 0 && pixelDstX < blockWidth)
                            {
                                gradientX[pixelDstY][pixelDstX] += srcPixel * filterKernelX.Values[ky][kx];
                                gradientY[pixelDstY][pixelDstX] += srcPixel * filterKernelY.Values[ky][kx];
                            }
                        }
                    }
                }
            }
        }

        // for multi channel, find channel with largest L2 gradient and store it in first channel:
        if constexpr (vector_active_size_v<ComputeT> > 1)
        {
#pragma unroll
            for (int bl = 0; bl < blockHeight; bl++)
            {
#pragma unroll
                for (size_t bc = 0; bc < blockWidth; bc++)
                {
                    remove_vector_t<ComputeT> maxMagSqr =
                        gradientX[bl][bc].x * gradientX[bl][bc].x + gradientY[bl][bc].x * gradientY[bl][bc].x;
#pragma unroll
                    for (int c = 1; c < vector_active_size_v<ComputeT>; c++)
                    {
                        remove_vector_t<ComputeT> magSqr =
                            gradientX[bl][bc][Channel(c)] * gradientX[bl][bc][Channel(c)] +
                            gradientY[bl][bc][Channel(c)] * gradientY[bl][bc][Channel(c)];

                        if (magSqr > maxMagSqr)
                        {
                            maxMagSqr           = magSqr;
                            gradientX[bl][bc].x = gradientX[bl][bc][Channel(c)];
                            gradientY[bl][bc].x = gradientY[bl][bc][Channel(c)];
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
                        if (aDstX != nullptr)
                        {
                            DstT *dstX = gotoPtr(aDstX, aPitchDstX, pixelX + bc, pixelY + bl);

                            Vector1<remove_vector_t<ComputeT>> temp = gradientX[bl][bc].x;
                            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                            {
                                round(temp);
                            }

                            *dstX = DstT(temp);
                        }
                        if (aDstY != nullptr)
                        {
                            DstT *dstY = gotoPtr(aDstY, aPitchDstY, pixelX + bc, pixelY + bl);

                            Vector1<remove_vector_t<ComputeT>> temp = gradientY[bl][bc].x;
                            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                            {
                                round(temp);
                            }

                            *dstY = DstT(temp);
                        }
                        if (aDstMag != nullptr)
                        {
                            DstT *dstMag = gotoPtr(aDstMag, aPitchDstMag, pixelX + bc, pixelY + bl);

                            Vector1<remove_vector_t<ComputeT>> temp;

                            switch (aNorm)
                            {
                                case Norm::Inf:
                                    temp = max(gradientX[bl][bc].x, gradientY[bl][bc].x);
                                    break;
                                case Norm::L1:
                                    temp = abs(gradientX[bl][bc].x) + abs(gradientY[bl][bc].x);
                                    break;
                                case Norm::L2:
                                    temp = sqrt(gradientX[bl][bc].x * gradientX[bl][bc].x +
                                                gradientY[bl][bc].x * gradientY[bl][bc].x);
                                    break;
                                default:
                                    // well, that shouldn't happen...
                                    break;
                            }

                            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
                            {
                                round(temp);
                            }

                            *dstMag = DstT(temp);
                        }
                        if (aDstAngle != nullptr)
                        {
                            Pixel32fC1 *dstAngle = gotoPtr(aDstAngle, aPitchDstAngle, pixelX + bc, pixelY + bl);

                            *dstAngle = atan2(gradientY[bl][bc].x, gradientX[bl][bc].x);
                        }
                        if (aDstCovariance != nullptr)
                        {
                            // write out untupeled:
                            Pixel32fC4 *dstCovariance =
                                gotoPtr(aDstCovariance, aPitchDstCovariance, pixelX + bc, pixelY + bl);

                            Pixel32fC4 res;
                            res.x = static_cast<float>(gradientX[bl][bc].x) * static_cast<float>(gradientX[bl][bc].x);
                            res.y = static_cast<float>(gradientY[bl][bc].x) * static_cast<float>(gradientY[bl][bc].x);
                            res.z = static_cast<float>(gradientX[bl][bc].x) * static_cast<float>(gradientY[bl][bc].x);
                            res.w = res.z;
                            *dstCovariance = res;
                        }
                    }
                }
            }
        }
    }
    return;
}

template <typename ComputeT, typename DstT, size_t TupelSize, int WarpAlignmentInBytes, int kernelWidth,
          int kernelHeight, int kernelCenterX, int kernelCenterY, int blockWidth, int blockHeight,
          RoundingMode roundingMode, typename BorderControlT, typename FixedFilterKernelXT,
          typename FixedFilterKernelYT>
void InvokeGradientVectorKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                const BorderControlT &aSrcWithBC, DstT *aDstX, size_t aPitchDstX, DstT *aDstY,
                                size_t aPitchDstY, DstT *aDstMag, size_t aPitchDstMag, Pixel32fC1 *aDstAngle,
                                size_t aPitchDstAngle, Pixel32fC4 *aDstCovariance, size_t aPitchDstCovariance,
                                Norm aNorm, const Size2D &aSize)
{
    // find first non-nullptr for thread alignments, if any:
    DstT *dst = aDstX;
    if (aDstX == nullptr)
    {
        dst = aDstY;
    }
    if (aDstX == nullptr && aDstY == nullptr)
    {
        dst = aDstMag;
    }
    if (aDstX == nullptr && aDstY == nullptr && aDstMag == nullptr)
    {
        if constexpr (std::same_as<DstT, Pixel32fC1>)
        {
            // in case of tuples and DstT is not Pixel32fC1, aDstAngle is written out untupled anyhow...
            dst = aDstAngle;
        }
    }
    // aDstCovariance as a 32fC3 is never tupled. If this is the only output, just tupel on nullptr alignment...

    ThreadSplit<WarpAlignmentInBytes, TupelSize> ts(dst, aSize.x, aWarpSize);

    if constexpr (TupelSize != 1)
    {
        ThreadSplit<WarpAlignmentInBytes, TupelSize> tsCheckX(aDstX, aSize.x, aWarpSize);
        if (aDstX != nullptr && ts != tsCheckX)
        {
            throw INVALIDARGUMENT(
                aDstX aDstY aDstMag,
                "All destination images must fulfill the same byte-alignments in order to use tupeled memory access.");
        }
        ThreadSplit<WarpAlignmentInBytes, TupelSize> tsCheckY(aDstY, aSize.x, aWarpSize);
        if (aDstY != nullptr && ts != tsCheckY)
        {
            throw INVALIDARGUMENT(
                aDstX aDstY aDstMag,
                "All destination images must fulfill the same byte-alignments in order to use tupeled memory access.");
        }
        ThreadSplit<WarpAlignmentInBytes, TupelSize> tsCheckMag(aDstMag, aSize.x, aWarpSize);
        if (aDstMag != nullptr && ts != tsCheckMag)
        {
            throw INVALIDARGUMENT(
                aDstX aDstY aDstMag,
                "All destination images must fulfill the same byte-alignments in order to use tupeled memory access.");
        }
        if constexpr (std::same_as<DstT, Pixel32fC1>)
        {
            ThreadSplit<WarpAlignmentInBytes, TupelSize> tsCheckAng(aDstAngle, aSize.x, aWarpSize);
            if (aDstAngle != nullptr && ts != tsCheckMag)
            {
                throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle,
                                      "All destination images must fulfill the same "
                                      "byte-alignments in order to use tupeled memory access.");
            }
        }
    }

    dim3 blocksPerGrid(DIV_UP(ts.Total() / blockWidth, aBlockSize.x), DIV_UP(aSize.y / blockHeight, aBlockSize.y), 1);

    gradientVectorKernel<WarpAlignmentInBytes, TupelSize, ComputeT, DstT, kernelWidth, kernelHeight, kernelCenterX,
                         kernelCenterY, blockWidth, blockHeight, roundingMode, BorderControlT, FixedFilterKernelXT,
                         FixedFilterKernelYT><<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(
        aSrcWithBC, aDstX, aPitchDstX, aDstY, aPitchDstY, aDstMag, aPitchDstMag, aDstAngle, aPitchDstAngle,
        aDstCovariance, aPitchDstCovariance, aNorm, aSize, ts);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream
                                             << " Tupel size: " << TupelSize);
}

template <typename ComputeT, typename DstT, size_t TupelSize, int kernelWidth, int kernelHeight, int kernelCenterX,
          int kernelCenterY, int blockWidth, int blockHeight, RoundingMode roundingMode, typename BorderControlT,
          typename FixedFilterKernelXT, typename FixedFilterKernelYT>
void InvokeGradientVectorKernelDefault(const BorderControlT &aSrcWithBC, DstT *aDstX, size_t aPitchDstX, DstT *aDstY,
                                       size_t aPitchDstY, DstT *aDstMag, size_t aPitchDstMag, Pixel32fC1 *aDstAngle,
                                       size_t aPitchDstAngle, Pixel32fC4 *aDstCovariance, size_t aPitchDstCovariance,
                                       Norm aNorm, const Size2D &aSize, const mpp::cuda::StreamCtx &aStreamCtx)
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

        InvokeGradientVectorKernel<ComputeT, DstT, TupelSize, WarpAlignmentInBytes, kernelWidth, kernelHeight,
                                   kernelCenterX, kernelCenterY, blockWidth, blockHeight, roundingMode, BorderControlT,
                                   FixedFilterKernelXT, FixedFilterKernelYT>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrcWithBC, aDstX, aPitchDstX, aDstY,
            aPitchDstY, aDstMag, aPitchDstMag, aDstAngle, aPitchDstAngle, aDstCovariance, aPitchDstCovariance, aNorm,
            aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(gradientVectorKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
