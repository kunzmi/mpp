#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/affineTransformation.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/utilities.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{

/// <summary>
/// Computes a radial profile. As we use atomics to accumulate the values, we compute for each thread a pixel block.
/// Doing so, we can exclude that two threads in a warp fall on the same entry in the radial profile.
/// </summary>
template <typename SrcT, int blockSize>
__global__ void radialProfileKernel(const SrcT *__restrict__ aSrc, size_t aPitchSrc, int *__restrict__ aProfileCount,
                                    same_vector_size_different_type_t<SrcT, float> *__restrict__ aProfileSum,
                                    same_vector_size_different_type_t<SrcT, float> *__restrict__ aProfileSumSqr,
                                    int aProfileSize, AffineTransformation<float> aTransformation, Size2D aSize)
{
    int xIn = blockIdx.x * blockDim.x + threadIdx.x;
    int yIn = blockIdx.y * blockDim.y + threadIdx.y;

    int constexpr maxRadiiCount =
        (blockSize * 3 + 1) / 2; // The actual factor is sqrt(2), approximate it with 1.5 and round up

    if (xIn * blockSize >= aSize.x && yIn * blockSize >= aSize.y)
    {
        return;
    }

    int radii[blockSize][blockSize];
    Pixel32sC1 radMin = numeric_limits<int>::max();

    __shared__ int radiusCount[maxRadiiCount][32 * 8];

#pragma unroll
    for (int by = 0; by < blockSize; by++)
    {
#pragma unroll
        for (int bx = 0; bx < blockSize; bx++)
        {
            // move to center and rotate/squeeze in case of an ellipse.
            int px = xIn * blockSize + bx;
            int py = yIn * blockSize + by;

            if (px < aSize.x && py < aSize.y)
            {
                Pixel32fC2 coord = aTransformation * Vec2f(xIn * blockSize + bx, yIn * blockSize + by);
                coord.Sqr();
                Pixel32fC1 radius = coord.x + coord.y;

                radius.Sqrt();
                radius += 0.5f; // we know that we have no negative values, so avoid the round() function

                int r = static_cast<int>(radius.x);
                radMin.Min(r);
                radii[by][bx] = r;
            }
            else
            {
                // indicate that outside of image
                radii[by][bx] = -1;
            }
        }
    }

    if (radMin == numeric_limits<int>::max())
    {
        // this should not happen, but who knows...
        // not a single pixel of the pixel block was inside the image...
        return;
    }

#pragma unroll
    for (int r = 0; r < maxRadiiCount; r++)
    {
        radiusCount[r][threadIdx.y * 32 + threadIdx.x] = 0;
    }

#pragma unroll
    for (int by = 0; by < blockSize; by++)
    {
#pragma unroll
        for (int bx = 0; bx < blockSize; bx++)
        {
            if (radii[by][bx] >= 0)
            {
                radiusCount[radii[by][bx] - radMin.x][threadIdx.y * 32 + threadIdx.x]++;
            }
        }
    }

#pragma unroll
    for (int ri = 0; ri < maxRadiiCount; ri++)
    {
        same_vector_size_different_type_t<SrcT, float> pixel    = 0;
        same_vector_size_different_type_t<SrcT, float> pixelSqr = 0;

        int radCount = radiusCount[ri][threadIdx.y * 32 + threadIdx.x];
        if (radCount > 0)
        {
#pragma unroll
            for (int by = 0; by < blockSize; by++)
            {
#pragma unroll
                for (int bx = 0; bx < blockSize; bx++)
                {
                    if (ri == radii[by][bx] - radMin.x)
                    {
                        const same_vector_size_different_type_t<SrcT, float> p =
                            same_vector_size_different_type_t<SrcT, float>(
                                *gotoPtr(aSrc, aPitchSrc, xIn * blockSize + bx, yIn * blockSize + by));
                        pixel += p;
                        pixelSqr += p * p;
                    }
                }
            }

            if (radMin.x + ri < aProfileSize)
            {
                atomicAdd(aProfileCount + radMin.x + ri, radCount);

                atomicAdd(reinterpret_cast<float *>(aProfileSum) + radMin.x + ri, pixel.x);
                if constexpr (vector_active_size_v<SrcT> > 1)
                {
                    atomicAdd(reinterpret_cast<float *>(aProfileSum) + (radMin.x + ri) * vector_active_size_v<SrcT> + 1,
                              pixel.y);
                }
                if constexpr (vector_active_size_v<SrcT> > 2)
                {
                    atomicAdd(reinterpret_cast<float *>(aProfileSum) + (radMin.x + ri) * vector_active_size_v<SrcT> + 2,
                              pixel.z);
                }
                if constexpr (vector_active_size_v<SrcT> > 3)
                {
                    atomicAdd(reinterpret_cast<float *>(aProfileSum) + (radMin.x + ri) * vector_active_size_v<SrcT> + 3,
                              pixel.w);
                }

                if (aProfileSumSqr != nullptr)
                {
                    atomicAdd(reinterpret_cast<float *>(aProfileSumSqr) + (radMin.x + ri) * vector_active_size_v<SrcT> +
                                  0,
                              pixelSqr.x);
                    if constexpr (vector_active_size_v<SrcT> > 1)
                    {
                        atomicAdd(reinterpret_cast<float *>(aProfileSumSqr) +
                                      (radMin.x + ri) * vector_active_size_v<SrcT> + 1,
                                  pixelSqr.y);
                    }
                    if constexpr (vector_active_size_v<SrcT> > 2)
                    {
                        atomicAdd(reinterpret_cast<float *>(aProfileSumSqr) +
                                      (radMin.x + ri) * vector_active_size_v<SrcT> + 2,
                                  pixelSqr.z);
                    }
                    if constexpr (vector_active_size_v<SrcT> > 3)
                    {
                        atomicAdd(reinterpret_cast<float *>(aProfileSumSqr) +
                                      (radMin.x + ri) * vector_active_size_v<SrcT> + 3,
                                  pixelSqr.w);
                    }
                }
            }
        }
    }
}

template <typename SrcT>
void InvokeRadialProfileKernel(const dim3 &aBlockSize, uint aSharedMemory, cudaStream_t aStream, const SrcT *aSrc,
                               size_t aPitchSrc, int *aProfileCount,
                               same_vector_size_different_type_t<SrcT, float> *aProfileSum,
                               same_vector_size_different_type_t<SrcT, float> *aProfileSumSqr, int aProfileSize,
                               const AffineTransformation<float> &aTransformation, const Size2D &aSize)
{

    dim3 blocksPerGrid(DIV_UP((aSize.x + 3) / 4, aBlockSize.x), DIV_UP((aSize.y + 3) / 4, aBlockSize.y), 1);

    radialProfileKernel<SrcT, 4><<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(
        aSrc, aPitchSrc, aProfileCount, aProfileSum, aProfileSumSqr, aProfileSize, aTransformation, aSize);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename SrcT>
void InvokeRadialProfileKernelDefault(const SrcT *aSrc, size_t aPitchSrc, int *aProfileCount,
                                      same_vector_size_different_type_t<SrcT, float> *aProfileSum,
                                      same_vector_size_different_type_t<SrcT, float> *aProfileSumSqr, int aProfileSize,
                                      const AffineTransformation<float> &aTransformation, const Size2D &aSize,
                                      const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        dim3 BlockSize    = ConfigBlockSize<"Default">::value;
        uint SharedMemory = 0;

        InvokeRadialProfileKernel<SrcT>(BlockSize, SharedMemory, aStreamCtx.Stream, aSrc, aPitchSrc, aProfileCount,
                                        aProfileSum, aProfileSumSqr, aProfileSize, aTransformation, aSize);
    }
    else
    {
        throw CUDAUNSUPPORTED(transposeKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
