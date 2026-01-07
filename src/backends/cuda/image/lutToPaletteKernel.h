#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/mpp_defs.h>
#include <common/utilities.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{

template <typename TT> struct lut_compute_type_for
{
    using type = double;
};
template <> struct lut_compute_type_for<byte>
{
    using type = float;
};

template <typename T> using lut_compute_type_for_t = typename lut_compute_type_for<T>::type;

/// <summary>
/// Create a color palette from a LUT with interpolation.
/// </summary>
template <typename LutT, InterpolationMode interpolationMode>
__global__ void lutToPaletteKernel(const int *__restrict__ aX, const int *__restrict__ aY, int aLutSize,
                                   LutT *__restrict__ aPalette)
{
    constexpr int PaletteSize = 1 << (sizeof(LutT) * 8);

    int threadX = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadX >= PaletteSize)
    {
        return;
    }

    int idx = LastIndexSmallerOrEqual(aX, aLutSize, 0, threadX);

    if constexpr (interpolationMode == InterpolationMode::NearestNeighbor)
    {
        if (idx >= 0)
        {
            aPalette[threadX] = static_cast<LutT>(aY[idx]);
        }
        else
        {
            aPalette[threadX] = static_cast<LutT>(threadX - numeric_limits<LutT>::lowest());
        }
    }
    if constexpr (interpolationMode == InterpolationMode::Linear)
    {
        lut_compute_type_for_t<LutT> valX[2];
        lut_compute_type_for_t<LutT> valY[2];

        if (idx >= 0 && idx < aLutSize - 1)
        {
            valX[0] = static_cast<lut_compute_type_for_t<LutT>>(aX[idx]);
            valX[1] = static_cast<lut_compute_type_for_t<LutT>>(aX[idx + 1]);

            valY[0] = static_cast<lut_compute_type_for_t<LutT>>(aY[idx]);
            valY[1] = static_cast<lut_compute_type_for_t<LutT>>(aY[idx + 1]);

            const lut_compute_type_for_t<LutT> val = LinearInterpolate(
                valX, valY, static_cast<lut_compute_type_for_t<LutT>>(threadX - numeric_limits<LutT>::lowest()));

            if constexpr (RealUnsignedIntegral<LutT>)
            {
                aPalette[threadX] = static_cast<LutT>(val + static_cast<lut_compute_type_for_t<LutT>>(0.5));
            }
            else
            {
                aPalette[threadX] = static_cast<LutT>(round(val));
            }
        }
        else if (idx == aLutSize - 1)
        {
            aPalette[threadX] = static_cast<LutT>(aY[idx]);
        }
        else
        {
            aPalette[threadX] = static_cast<LutT>(threadX - numeric_limits<LutT>::lowest());
        }
    }
    if constexpr (interpolationMode == InterpolationMode::CubicLagrange)
    {
        lut_compute_type_for_t<LutT> valX[4];
        lut_compute_type_for_t<LutT> valY[4];

        if (idx >= 0)
        {
            idx--;
            idx = min(max(0, idx), aLutSize - 4);

            valX[0] = static_cast<lut_compute_type_for_t<LutT>>(aX[idx]);
            valX[1] = static_cast<lut_compute_type_for_t<LutT>>(aX[idx + 1]);
            valX[2] = static_cast<lut_compute_type_for_t<LutT>>(aX[idx + 2]);
            valX[3] = static_cast<lut_compute_type_for_t<LutT>>(aX[idx + 3]);

            valY[0] = static_cast<lut_compute_type_for_t<LutT>>(aY[idx]);
            valY[1] = static_cast<lut_compute_type_for_t<LutT>>(aY[idx + 1]);
            valY[2] = static_cast<lut_compute_type_for_t<LutT>>(aY[idx + 2]);
            valY[3] = static_cast<lut_compute_type_for_t<LutT>>(aY[idx + 3]);

            const lut_compute_type_for_t<LutT> val = CubicInterpolate(
                valX, valY, static_cast<lut_compute_type_for_t<LutT>>(threadX - numeric_limits<LutT>::lowest()));

            if constexpr (RealUnsignedIntegral<LutT>)
            {
                aPalette[threadX] = static_cast<LutT>(val + static_cast<lut_compute_type_for_t<LutT>>(0.5));
            }
            else
            {
                aPalette[threadX] = static_cast<LutT>(round(val));
            }
        }
        else
        {
            aPalette[threadX] = static_cast<LutT>(threadX - numeric_limits<LutT>::lowest());
        }
    }
}

template <typename LutT>
void InvokeLutToPaletteKernel(const dim3 &aBlockSize, uint aSharedMemory, cudaStream_t aStream,
                              const int *__restrict__ aX, const int *__restrict__ aY, int aLutSize,
                              LutT *__restrict__ aPalette, InterpolationMode aInterpolationMode)
{
    constexpr int PaletteSize = 1 << (sizeof(LutT) * 8);
    dim3 blocksPerGrid(DIV_UP(PaletteSize, aBlockSize.x), 1, 1);

    if (aInterpolationMode == InterpolationMode::NearestNeighbor)
    {
        lutToPaletteKernel<LutT, InterpolationMode::NearestNeighbor>
            <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aX, aY, aLutSize, aPalette);
    }
    else if (aInterpolationMode == InterpolationMode::Linear)
    {
        lutToPaletteKernel<LutT, InterpolationMode::Linear>
            <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aX, aY, aLutSize, aPalette);
    }
    else if (aInterpolationMode == InterpolationMode::CubicLagrange)
    {
        lutToPaletteKernel<LutT, InterpolationMode::CubicLagrange>
            <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aX, aY, aLutSize, aPalette);
    }
    else
    {
        throw INVALIDARGUMENT(aInterpolationMode, "Unsupported interpolation mode. Only NearestNeighbor, Linear and "
                                                  "CubicLagrange are supported, but provided aInterpolationMode is "
                                                      << aInterpolationMode);
    }

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename LutT>
void InvokeLutToPaletteKernelDefault(const int *__restrict__ aX, const int *__restrict__ aY, int aLutSize,
                                     LutT *__restrict__ aPalette, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize        = {sizeof(LutT) == 1 ? 32 : 128, 1, 1};
        constexpr uint SharedMemory = 0;

        InvokeLutToPaletteKernel<LutT>(BlockSize, SharedMemory, aStreamCtx.Stream, aX, aY, aLutSize, aPalette,
                                       aInterpolationMode);
    }
    else
    {
        throw CUDAUNSUPPORTED(lutToPaletteKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
