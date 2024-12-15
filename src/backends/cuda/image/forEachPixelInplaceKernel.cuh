#include <common/image/gotoPtr.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace opp::image::cuda
{
/// <summary>
/// runs aOp on every pixel of an image. No inplace operation, no mask.
/// </summary>
template <int WarpAlignmentInBytes, int TupelSize, class T, class functor>
__global__ void forEachPixelInplaceKernel(T *__restrict__ aSrcDst, size_t aPitchSrcDst, Size2D aSize,
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
        if (aSplit.ThreadIsAlignedToWarp(threadX))
        {
            T *pixelsOut = gotoPtr(aSrcDst, aPitchSrcDst, pixelX, pixelY);

            Tupel<T, TupelSize> res = Tupel<T, TupelSize>::LoadAligned(pixelsOut);

            aOp(pixelX, pixelY, res);

            Tupel<T, TupelSize>::StoreAligned(res, pixelsOut);
            return;
        }
    }

    T *pixelOut = gotoPtr(aSrcDst, aPitchSrcDst, pixelX, pixelY);

    aOp(pixelX, pixelY, *pixelOut);
    return;
}

} // namespace opp::image::cuda