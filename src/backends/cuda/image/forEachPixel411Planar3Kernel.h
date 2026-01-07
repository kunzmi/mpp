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
/// <summary>
/// runs aFunctor on every pixel of an image. Inplace and outplace operation, no mask. Planar 3 channel destination,
/// with 411 chroma sub-sampling.
/// </summary>
template <class DstT, typename funcType, ChromaSubsamplePos chromaSubsamplePos>
__global__ void forEachPixel411Planar3Kernel(Vector1<remove_vector_t<DstT>> *__restrict__ aDst1, size_t aPitchDst1,
                                             Vector1<remove_vector_t<DstT>> *__restrict__ aDst2, size_t aPitchDst2,
                                             Vector1<remove_vector_t<DstT>> *__restrict__ aDst3, size_t aPitchDst3,
                                             Size2D aSize, funcType aFunctor)
{
    using DstPlane              = Vector1<remove_vector_t<DstT>>;
    int threadX                 = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY                 = blockIdx.y * blockDim.y + threadIdx.y;
    constexpr int subSampleSize = 4;

    const int pixelX       = threadX * subSampleSize;
    const int pixelXChroma = threadX;
    const int pixelY       = threadY;

    // skip non-multiple of 4 pixels at the border
    if (threadX >= aSize.x / subSampleSize || threadY >= aSize.y)
    {
        return;
    }

    DstT res[subSampleSize];
    DstPlane *pixelOutLuma    = gotoPtr(aDst1, aPitchDst1, pixelX, pixelY);
    DstPlane *pixelOutChroma1 = gotoPtr(aDst2, aPitchDst2, pixelXChroma, pixelY);
    DstPlane *pixelOutChroma2 = gotoPtr(aDst3, aPitchDst3, pixelXChroma, pixelY);

    // load the destination pixel in case of inplace operation:
    static_assert(!funcType::DoLoadBeforeOp, "Pre-loading on 411 sub-sampled data is not implemented.");

    Tupel<DstPlane, subSampleSize> resLuma;
#pragma unroll
    for (int i = 0; i < subSampleSize; i++)
    {
        // ignore functor result, as only transformerFunctor can return false and they are not used in chroma
        // sub-sampled kernels
        aFunctor(pixelX + i, pixelY, res[i]);
        resLuma.value[i].x = res[i].x;
    }
    // maybe aligned or un-aligned:
    Tupel<DstPlane, subSampleSize>::Store(resLuma, pixelOutLuma);

    DstPlane resChroma1;
    DstPlane resChroma2;
    if constexpr (chromaSubsamplePos == ChromaSubsamplePos::Center)
    {
        // average chroma values:
        if constexpr (RealFloatingVector<DstT>)
        {
            resChroma1.x = res[0].y;
            resChroma1.x += res[1].y;
            resChroma1.x += res[2].y;
            resChroma1.x += res[3].y;
            resChroma1.x /= static_cast<remove_vector_t<DstT>>(subSampleSize);

            resChroma2.x = res[0].z;
            resChroma2.x += res[1].z;
            resChroma2.x += res[2].z;
            resChroma2.x += res[3].z;
            resChroma2.x /= static_cast<remove_vector_t<DstT>>(subSampleSize);
        }
        else
        {
            int temp = static_cast<int>(res[0].y);
            temp += static_cast<int>(res[1].y);
            temp += static_cast<int>(res[2].y);
            temp += static_cast<int>(res[3].y);
            temp /= subSampleSize; // simple integer division, same as in NPP
            resChroma1.x = static_cast<remove_vector_t<DstPlane>>(temp);

            temp = static_cast<int>(res[0].z);
            temp += static_cast<int>(res[1].z);
            temp += static_cast<int>(res[2].z);
            temp += static_cast<int>(res[3].z);
            temp /= subSampleSize; // simple integer division, same as in NPP
            resChroma2.x = static_cast<remove_vector_t<DstPlane>>(temp);
        }
    }
    else // CenterLeft is treated as Left
    {
        resChroma1.x = res[0].y;
        resChroma2.x = res[0].z;
    }
    *pixelOutChroma1 = resChroma1;
    *pixelOutChroma2 = resChroma2;

    return;
}

template <typename DstT, typename funcType>
void InvokeForEachPixel411Planar3Kernel(const dim3 &aBlockSize, cudaStream_t aStream,
                                        Vector1<remove_vector_t<DstT>> *__restrict__ aDst1, size_t aPitchDst1,
                                        Vector1<remove_vector_t<DstT>> *__restrict__ aDst2, size_t aPitchDst2,
                                        Vector1<remove_vector_t<DstT>> *__restrict__ aDst3, size_t aPitchDst3,
                                        const Size2D &aSize, const funcType &aFunctor,
                                        ChromaSubsamplePos chromaSubsamplePos)
{

    dim3 blocksPerGrid(DIV_UP(aSize.x / 4, aBlockSize.x), DIV_UP(aSize.y, aBlockSize.y), 1);

    if (chromaSubsamplePos == ChromaSubsamplePos::Center || chromaSubsamplePos == ChromaSubsamplePos::Undefined)
    {
        forEachPixel411Planar3Kernel<DstT, funcType, ChromaSubsamplePos::Center>
            <<<blocksPerGrid, aBlockSize, 0, aStream>>>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize,
                                                        aFunctor);
    }
    else
    {
        forEachPixel411Planar3Kernel<DstT, funcType, ChromaSubsamplePos::Left>
            <<<blocksPerGrid, aBlockSize, 0, aStream>>>(aDst1, aPitchDst1, aDst2, aPitchDst2, aDst3, aPitchDst3, aSize,
                                                        aFunctor);
    }

    peekAndCheckLastCudaError("Block size: " << aBlockSize << " Grid size: " << blocksPerGrid << " SharedMemory: " << 0
                                             << " Stream: " << aStream);
}

template <typename DstT, typename funcType>
void InvokeForEachPixel411Planar3KernelDefault(Vector1<remove_vector_t<DstT>> *__restrict__ aDst1, size_t aPitchDst1,
                                               Vector1<remove_vector_t<DstT>> *__restrict__ aDst2, size_t aPitchDst2,
                                               Vector1<remove_vector_t<DstT>> *__restrict__ aDst3, size_t aPitchDst3,
                                               const Size2D &aSize, ChromaSubsamplePos chromaSubsamplePos,
                                               const mpp::cuda::StreamCtx &aStreamCtx, const funcType &aFunc)
{
    if (aStreamCtx.ComputeCapabilityMajor < INT_MAX)
    {
        const dim3 BlockSize = ConfigBlockSize<"Default">::value;

        InvokeForEachPixel411Planar3Kernel<DstT, funcType>(BlockSize, aStreamCtx.Stream, aDst1, aPitchDst1, aDst2,
                                                           aPitchDst2, aDst3, aPitchDst3, aSize, aFunc,
                                                           chromaSubsamplePos);
    }
    else
    {
        throw CUDAUNSUPPORTED(forEachPixel411Planar3Kernel,
                              "Trying to execute on a platform with an unsupported compute capability: "
                                  << aStreamCtx.ComputeCapabilityMajor << "." << aStreamCtx.ComputeCapabilityMinor);
    }
}

} // namespace mpp::image::cuda
