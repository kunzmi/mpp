#include <common/image/gotoPtr.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/safeCast.h>
#include <common/tupel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/ternary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/image/arithmetic/functors.h>

using namespace opp;
using namespace opp::image;

template <int tupelSize, class T, class functor>
__global__ void forEachPixelKernel(T *__restrict__ out, size_t pitchOut, Size2D aSize,
                                   ThreadSplit<256, tupelSize> aSplit, functor aOp)
{
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadX < aSplit.Muted() || threadX >= aSplit.Total() || threadY >= aSize.y)
    {
        return;
    }

    const int pixelX = aSplit.GetPixel(threadX);
    const int pixelY = threadY;

    if (aSplit.ThreadIsAlignedToWarp(threadX))
    {
        T *aDst = gotoPtr(out, pitchOut, pixelX, pixelY);

        Tupel<T, tupelSize> res;
        aOp(pixelX, pixelY, res);
        /*res.value[0] = pixelX;*/

        Tupel<T, tupelSize>::StoreAligned(res, aDst);
        return;
    }
    else
    {
        T *pixelOut = gotoPtr(out, pitchOut, pixelX, pixelY);

        aOp(pixelX, pixelY, *pixelOut);
        return;
    }
}

void forEachPixelKernelWithCuda(const int *in1, size_t pitch1, const int *in2, size_t pitch2, int *out, size_t pitchOut,
                                Size2D aSize)
{
    Add<Vector1<float>> op;

    SrcSrcScaleFunctor<2, Vector1<float>, Vector1<int>, Vector1<int>, Add<Vector1<float>>> functor(
        (Vector1<int> *)in1, pitch1, (Vector1<int> *)in2, pitch2, op, 1.0f);

    ThreadSplit<256, 2> ts(out, aSize.x);

    dim3 threadsPerBlock(32, 8, 1);
    dim3 blocksPerGrid((ts.Total() + 31) / 32, (aSize.y + 7) / 8, 1);

    forEachPixelKernel<2><<<blocksPerGrid, threadsPerBlock>>>((Vector1<int> *)out, pitchOut, aSize, ts, functor);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "addKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching addKernel!"
                  << std::endl;
        return;
    }
}
