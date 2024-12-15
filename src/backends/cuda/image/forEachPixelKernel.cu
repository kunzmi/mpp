#include "forEachPixelKernel.cuh"

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
#include <common/image/arithmetic/srcSrcFunctor.h>
#include <common/image/pixelTypes.h>
#include <common/vectorTypes.h>

#include <backends/cuda/image/configurations.h>

namespace opp::image::cuda
{
constexpr int WarpAlignmentInBytes = 64;

using funcType = SrcSrcFunctor<2, Pixel32sC4, Pixel8uC4, Pixel8uC4, Add<Pixel32sC4>>;

void forEachPixelKernelWithCuda(Pixel8uC4 *in1, size_t pitch1, Pixel8uC4 *in2, size_t pitch2, Pixel8uC4 *out,
                                size_t pitchOut, Size2D aSize)
{
    Add<Pixel32sC4> op;

    funcType functor(in1, pitch1, in2, pitch2, op);

    ThreadSplit<WarpAlignmentInBytes, 2> ts(out, aSize.x);

    dim3 threadsPerBlock(32, 8, 1);
    dim3 blocksPerGrid((ts.Total() + 31) / 32, (aSize.y + 7) / 8, 1);

    forEachPixelKernel<WarpAlignmentInBytes, 2, Pixel8uC4, funcType>
        <<<blocksPerGrid, threadsPerBlock>>>(out, pitchOut, aSize, ts, functor);

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

} // namespace opp::image::cuda
