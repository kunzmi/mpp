#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/streamCtx.h>
#include <common/exception.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/mpp_defs.h>
#include <common/tupel.h>
#include <common/utilities.h>
#include <common/vectorTypes_impl.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mpp::image::cuda
{
enum class Dst422C2Layout
{
    YCrCb,
    YCbCr,
    CbYCr,
    CrYCb // this variant is technically possible but doesn't seem to exist in real world scenarios.
};

/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. C 2 channel destination,
/// with 422 chroma sub-sampling. Every second pixel has either Cb or Cr component.
/// </summary>
template <class DstT, typename funcType, ChromaSubsamplePos chromaSubsamplePos, Dst422C2Layout layout>
__global__ void forEachPixel422C2Kernel(Vector2<remove_vector_t<DstT>> *__restrict__ aDst1, size_t aPitchDst1,
                                        Size2D aSize, funcType aFunctor)
{
    using DstPlaneT             = Vector2<remove_vector_t<DstT>>;
    int threadX                 = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY                 = blockIdx.y * blockDim.y + threadIdx.y;
    constexpr int subSampleSize = 2;

    const int pixelX = threadX * subSampleSize;
    const int pixelY = threadY;

    // skip uneven pixels at the border
    if (threadX >= aSize.x / subSampleSize || threadY >= aSize.y)
    {
        return;
    }

    DstT res[subSampleSize];

    // load the destination pixel in case of inplace operation:
    static_assert(!funcType::DoLoadBeforeOp, "Pre-loading on 422 sub-sampled data is not implemented.");

    Tupel<DstPlaneT, subSampleSize> resLuma2Chroma1;
#pragma unroll
    for (int i = 0; i < subSampleSize; i++)
    {
        // ignore functor result, as only transformerFunctor can return false and they are not used in chroma
        // sub-sampled kernels
        aFunctor(pixelX + i, pixelY, res[i]);

        if constexpr (layout == Dst422C2Layout::YCbCr || layout == Dst422C2Layout::YCrCb)
        {
            resLuma2Chroma1.value[i].x = res[i].x;
        }
        else
        {
            resLuma2Chroma1.value[i].y = res[i].x;
        }
    }

    DstPlaneT resChroma;
    if constexpr (chromaSubsamplePos == ChromaSubsamplePos::Center)
    {
        // average chroma values:
        if constexpr (RealFloatingVector<DstT>)
        {
            resChroma = res[0].YZ();
            resChroma += res[1].YZ();
            resChroma /= static_cast<remove_vector_t<DstT>>(subSampleSize);
        }
        else
        {
            Vector2<int> temp = static_cast<Vector2<int>>(res[0].YZ());
            temp += static_cast<Vector2<int>>(res[1].YZ());
            temp.DivScaleRoundZero(subSampleSize); // simple integer division, same as in NPP
            resChroma = static_cast<DstPlaneT>(temp);
        }
    }
    else // CenterLeft is treated as Left
    {
        resChroma = res[0].YZ();
    }

    if constexpr (layout == Dst422C2Layout::YCbCr)
    {
        resLuma2Chroma1.value[0].y = resChroma.x;
        resLuma2Chroma1.value[1].y = resChroma.y;
    }
    else if constexpr (layout == Dst422C2Layout::YCrCb)
    {
        resLuma2Chroma1.value[0].y = resChroma.y;
        resLuma2Chroma1.value[1].y = resChroma.x;
    }
    else if constexpr (layout == Dst422C2Layout::CbYCr)
    {
        resLuma2Chroma1.value[0].x = resChroma.x;
        resLuma2Chroma1.value[1].x = resChroma.y;
    }
    else
    {
        resLuma2Chroma1.value[0].x = resChroma.y;
        resLuma2Chroma1.value[1].x = resChroma.x;
    }

    DstPlaneT *pixelOutLuma = gotoPtr(aDst1, aPitchDst1, pixelX, pixelY);
    // maybe aligned or un-aligned:
    Tupel<DstPlaneT, subSampleSize>::Store(resLuma2Chroma1, pixelOutLuma);

    return;
}

template <typename DstT, typename funcType>
void InvokeForEachPixel422C2Kernel(const dim3 &aBlockSize, cudaStream_t aStream,
                                   Vector2<remove_vector_t<DstT>> *__restrict__ aDst1, size_t aPitchDst1,
                                   const Size2D &aSize, const funcType &aFunctor,
                                   ChromaSubsamplePos aChromaSubsamplePos, Dst422C2Layout aLayout)
{

    dim3 blocksPerGrid(DIV_UP(aSize.x / 2, aBlockSize.x), DIV_UP(aSize.y, aBlockSize.y), 1);

    if (aChromaSubsamplePos == ChromaSubsamplePos::Center || aChromaSubsamplePos == ChromaSubsamplePos::Undefined)
    {
        if (aLayout == Dst422C2Layout::YCbCr)
        {
            forEachPixel422C2Kernel<DstT, funcType, ChromaSubsamplePos::Center, Dst422C2Layout::YCbCr>
                <<<blocksPerGrid, aBlockSize, 0, aStream>>>(aDst1, aPitchDst1, aSize, aFunctor);
        }
        else if (aLayout == Dst422C2Layout::YCrCb)
        {
            forEachPixel422C2Kernel<DstT, funcType, ChromaSubsamplePos::Center, Dst422C2Layout::YCrCb>
                <<<blocksPerGrid, aBlockSize, 0, aStream>>>(aDst1, aPitchDst1, aSize, aFunctor);
        }
        else if (aLayout == Dst422C2Layout::CbYCr)
        {
            forEachPixel422C2Kernel<DstT, funcType, ChromaSubsamplePos::Center, Dst422C2Layout::CbYCr>
                <<<blocksPerGrid, aBlockSize, 0, aStream>>>(aDst1, aPitchDst1, aSize, aFunctor);
        }
        else if (aLayout == Dst422C2Layout::CrYCb)
        {
            forEachPixel422C2Kernel<DstT, funcType, ChromaSubsamplePos::Center, Dst422C2Layout::CrYCb>
                <<<blocksPerGrid, aBlockSize, 0, aStream>>>(aDst1, aPitchDst1, aSize, aFunctor);
        }
    }
    else
    {
        if (aLayout == Dst422C2Layout::YCbCr)
        {
            forEachPixel422C2Kernel<DstT, funcType, ChromaSubsamplePos::Left, Dst422C2Layout::YCbCr>
                <<<blocksPerGrid, aBlockSize, 0, aStream>>>(aDst1, aPitchDst1, aSize, aFunctor);
        }
        else if (aLayout == Dst422C2Layout::YCrCb)
        {
            forEachPixel422C2Kernel<DstT, funcType, ChromaSubsamplePos::Left, Dst422C2Layout::YCrCb>
                <<<blocksPerGrid, aBlockSize, 0, aStream>>>(aDst1, aPitchDst1, aSize, aFunctor);
        }
        else if (aLayout == Dst422C2Layout::CbYCr)
        {
            forEachPixel422C2Kernel<DstT, funcType, ChromaSubsamplePos::Left, Dst422C2Layout::CbYCr>
                <<<blocksPerGrid, aBlockSize, 0, aStream>>>(aDst1, aPitchDst1, aSize, aFunctor);
        }
        else if (aLayout == Dst422C2Layout::CrYCb)
        {
            forEachPixel422C2Kernel<DstT, funcType, ChromaSubsamplePos::Left, Dst422C2Layout::CrYCb>
                <<<blocksPerGrid, aBlockSize, 0, aStream>>>(aDst1, aPitchDst1, aSize, aFunctor);
        }
    }

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid << " SharedMemory: " << 0
                                             << " Stream: " << aStream);
}

template <typename DstT, typename funcType>
void InvokeForEachPixel422C2KernelDefault(Vector2<remove_vector_t<DstT>> *__restrict__ aDst1, size_t aPitchDst1,
                                          const Size2D &aSize, ChromaSubsamplePos aChromaSubsamplePos,
                                          Dst422C2Layout aLayout, const mpp::cuda::StreamCtx &aStreamCtx,
                                          const funcType &aFunc)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize = ConfigBlockSize<"Default">::value;

        InvokeForEachPixel422C2Kernel<DstT, funcType>(BlockSize, aStreamCtx.Stream, aDst1, aPitchDst1, aSize, aFunc,
                                                      aChromaSubsamplePos, aLayout);
    }
    else
    {
        throw CUDAUNSUPPORTED(forEachPixel422C2Kernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
