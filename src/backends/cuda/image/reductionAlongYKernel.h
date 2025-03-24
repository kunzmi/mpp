#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace opp::image::cuda
{

/// <summary>
/// runs aFunctor reduction on one image column, single output value.
/// The kernel is supposed to launch warpSize threads on x-block dimension.
/// </summary>
template <typename SrcT, typename DstT, typename reductionOp, ReductionInitValue NeutralValue, typename postOp,
          typename postOpScalar>
__global__ void reductionAlongYKernel(const SrcT *__restrict__ aSrc, DstT *__restrict__ aDst,
                                      remove_vector_t<DstT> *__restrict__ aDstScalar, int aSize, postOp aPostOp,
                                      postOpScalar aPostOpScalar)
{
    int warpLaneId = threadIdx.x;
    int batchId    = threadIdx.y;

    reductionOp redOp;
    DstT result(reduction_init_value_v<NeutralValue, DstT>);

    /*__shared__ DstT
        buffer[ConfigBlockSize<"DefaultReductionY">::value.y][ConfigBlockSize<"DefaultReductionY">::value.x];*/
    extern __shared__ int sharedBuffer[];

    // Block dimension in X are the same for large and small configuration!
    DstT(*buffer)[ConfigBlockSize<"DefaultReductionY">::value.x] =
        (DstT(*)[ConfigBlockSize<"DefaultReductionY">::value.x])(sharedBuffer);

    // process 1D input array in threadBlock-junks
    for (int pixelYWarp0 = 0; pixelYWarp0 < aSize; pixelYWarp0 += warpSize * blockDim.y)
    {
        const int pixelY = pixelYWarp0 + warpLaneId + batchId * warpSize;
        if (pixelY < aSize)
        {
            redOp(aSrc[pixelY], result);
        }
    }

    // write intermediate threadBlock sums to shared memory:
    buffer[batchId][warpLaneId] = result;

    __syncthreads();

    if (batchId == 0)
    {
        // reduce over Y the entire thread block:
#pragma unroll
        for (int i = 1; i < ConfigBlockSize<"DefaultReductionY">::value.y; i++) // i = 0 is already stored in result
        {
            redOp(buffer[i][warpLaneId], result);
        }

        // reduce over warp:
        if constexpr (ComplexVector<DstT>)
        {
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x.real, 16), result.x.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x.real, 8), result.x.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x.real, 4), result.x.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x.real, 2), result.x.real);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x.real, 1), result.x.real);

            redOp(__shfl_down_sync(0xFFFFFFFF, result.x.imag, 16), result.x.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x.imag, 8), result.x.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x.imag, 4), result.x.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x.imag, 2), result.x.imag);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x.imag, 1), result.x.imag);
        }
        else
        {
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x, 16), result.x);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x, 8), result.x);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x, 4), result.x);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x, 2), result.x);
            redOp(__shfl_down_sync(0xFFFFFFFF, result.x, 1), result.x);
        }

        if constexpr (vector_active_size_v<DstT> > 1)
        {
            if constexpr (ComplexVector<DstT>)
            {
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y.real, 16), result.y.real);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y.real, 8), result.y.real);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y.real, 4), result.y.real);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y.real, 2), result.y.real);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y.real, 1), result.y.real);

                redOp(__shfl_down_sync(0xFFFFFFFF, result.y.imag, 16), result.y.imag);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y.imag, 8), result.y.imag);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y.imag, 4), result.y.imag);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y.imag, 2), result.y.imag);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y.imag, 1), result.y.imag);
            }
            else
            {
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y, 16), result.y);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y, 8), result.y);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y, 4), result.y);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y, 2), result.y);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.y, 1), result.y);
            }
        }
        if constexpr (vector_active_size_v<DstT> > 2)
        {
            if constexpr (ComplexVector<DstT>)
            {
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z.real, 16), result.z.real);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z.real, 8), result.z.real);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z.real, 4), result.z.real);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z.real, 2), result.z.real);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z.real, 1), result.z.real);

                redOp(__shfl_down_sync(0xFFFFFFFF, result.z.imag, 16), result.z.imag);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z.imag, 8), result.z.imag);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z.imag, 4), result.z.imag);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z.imag, 2), result.z.imag);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z.imag, 1), result.z.imag);
            }
            else
            {
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z, 16), result.z);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z, 8), result.z);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z, 4), result.z);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z, 2), result.z);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.z, 1), result.z);
            }
        }
        if constexpr (vector_active_size_v<DstT> > 3)
        {
            if constexpr (ComplexVector<DstT>)
            {
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w.real, 16), result.w.real);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w.real, 8), result.w.real);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w.real, 4), result.w.real);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w.real, 2), result.w.real);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w.real, 1), result.w.real);

                redOp(__shfl_down_sync(0xFFFFFFFF, result.w.imag, 16), result.w.imag);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w.imag, 8), result.w.imag);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w.imag, 4), result.w.imag);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w.imag, 2), result.w.imag);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w.imag, 1), result.w.imag);
            }
            else
            {
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w, 16), result.w);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w, 8), result.w);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w, 4), result.w);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w, 2), result.w);
                redOp(__shfl_down_sync(0xFFFFFFFF, result.w, 1), result.w);
            }
        }
    }

    if (warpLaneId == 0 && batchId == 0)
    {
        if (aDstScalar != nullptr)
        {
            *aDstScalar = aPostOpScalar(result);
        }
        if (aDst != nullptr)
        {
            aPostOp(result);
            *aDst = result;
        }
    }
}

template <typename SrcT, typename DstT, typename reductionOp, ReductionInitValue NeutralValue, typename postOp,
          typename postOpScalar>
void InvokeReductionAlongYKernel(const dim3 &aBlockSize, uint aSharedMemory, int aWarpSize, cudaStream_t aStream,
                                 const SrcT *aSrc, DstT *aDst, remove_vector_t<DstT> *aDstScalar, int aSize,
                                 postOp aPostOp, postOpScalar aPostOpScalar)
{
    dim3 blocksPerGrid(1, 1, 1);

    int size = aSize / ConfigBlockSize<"DefaultReductionX">::value.y;
    size     = std::max(size, 1);

    reductionAlongYKernel<SrcT, DstT, reductionOp, NeutralValue, postOp, postOpScalar>
        <<<blocksPerGrid, aBlockSize, aSharedMemory, aStream>>>(aSrc, aDst, aDstScalar, size, aPostOp, aPostOpScalar);

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid
                                             << " SharedMemory: " << aSharedMemory << " Stream: " << aStream);
}

template <typename SrcT, typename DstT, typename reductionOp, ReductionInitValue NeutralValue, typename postOp,
          typename postOpScalar>
void InvokeReductionAlongYKernelDefault(const SrcT *aSrc, DstT *aDst, remove_vector_t<DstT> *aDstScalar, int aSize,
                                        postOp aPostOp, postOpScalar aPostOpScalar,
                                        const opp::cuda::StreamCtx &aStreamCtx)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        dim3 BlockSize    = ConfigBlockSize<"DefaultReductionY">::value;
        uint SharedMemory = sizeof(DstT) * BlockSize.x * BlockSize.y * BlockSize.z;

        if (SharedMemory > aStreamCtx.SharedMemPerBlock)
        {
            // use a block config of half the size:
            BlockSize    = ConfigBlockSize<"DefaultReductionYLarge">::value;
            SharedMemory = sizeof(DstT) * BlockSize.x * BlockSize.y * BlockSize.z;
        }

        InvokeReductionAlongYKernel<SrcT, DstT, reductionOp, NeutralValue, postOp, postOpScalar>(
            BlockSize, SharedMemory, aStreamCtx.WarpSize, aStreamCtx.Stream, aSrc, aDst, aDstScalar, aSize, aPostOp,
            aPostOpScalar);
    }
    else
    {
        throw CUDAUNSUPPORTED(reductionAlongYKernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND