#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if MPP_ENABLE_CUDA_BACKEND

#include "imageView.h"
#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/configurations.h>
#include <backends/cuda/image/statistics/statisticsKernel.h>
#include <backends/cuda/streamCtx.h>
#include <cmath>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/safeCast.h>
#include <common/scratchBuffer.h>
#include <common/scratchBufferException.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <concepts>
#include <limits>

namespace mpp::image::cuda
{
#pragma region AverageError
template <PixelType T> size_t ImageView<T>::AverageErrorBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = averageError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T> size_t ImageView<T>::AverageErrorMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = averageError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ulong64> buffers(nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
void ImageView<T>::AverageError(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<averageError_types_for_rt<T>> &aDst,
                                mpp::cuda::DevVarView<remove_vector_t<averageError_types_for_rt<T>>> &aDstScalar,
                                mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = averageError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeAverageErrorSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                             aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::AverageErrorMasked(const ImageView<T> &aSrc2,
                                      mpp::cuda::DevVarView<averageError_types_for_rt<T>> &aDst,
                                      mpp::cuda::DevVarView<remove_vector_t<averageError_types_for_rt<T>>> &aDstScalar,
                                      const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = averageError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ulong64> buffers(aBuffer.Pointer(),
                                             BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeAverageErrorMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                   aSrc2.Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDst.Pointer(),
                                   aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::AverageError(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<averageError_types_for_rt<T>> &aDst,
                                mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = averageError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeAverageErrorSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                             aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::AverageErrorMasked(const ImageView<T> &aSrc2,
                                      mpp::cuda::DevVarView<averageError_types_for_rt<T>> &aDst,
                                      const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = averageError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ulong64> buffers(aBuffer.Pointer(),
                                             BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeAverageErrorMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                   aSrc2.Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDst.Pointer(),
                                   nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region AverageRelativeError
template <PixelType T> size_t ImageView<T>::AverageRelativeErrorBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = averageRelativeError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::AverageRelativeErrorMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = averageRelativeError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ulong64> buffers(nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
void ImageView<T>::AverageRelativeError(
    const ImageView<T> &aSrc2, mpp::cuda::DevVarView<averageRelativeError_types_for_rt<T>> &aDst,
    mpp::cuda::DevVarView<remove_vector_t<averageRelativeError_types_for_rt<T>>> &aDstScalar,
    mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = averageRelativeError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeAverageRelativeErrorSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                                     buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(),
                                     aStreamCtx);
}

template <PixelType T>
void ImageView<T>::AverageRelativeErrorMasked(
    const ImageView<T> &aSrc2, mpp::cuda::DevVarView<averageRelativeError_types_for_rt<T>> &aDst,
    mpp::cuda::DevVarView<remove_vector_t<averageRelativeError_types_for_rt<T>>> &aDstScalar,
    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = averageRelativeError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ulong64> buffers(aBuffer.Pointer(),
                                             BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeAverageRelativeErrorMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                           aSrc2.Pitch(), buffers.template Get<0>(), buffers.template Get<1>(),
                                           aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::AverageRelativeError(const ImageView<T> &aSrc2,
                                        mpp::cuda::DevVarView<averageRelativeError_types_for_rt<T>> &aDst,
                                        mpp::cuda::DevVarView<byte> &aBuffer,
                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = averageRelativeError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeAverageRelativeErrorSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                                     buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::AverageRelativeErrorMasked(const ImageView<T> &aSrc2,
                                              mpp::cuda::DevVarView<averageRelativeError_types_for_rt<T>> &aDst,
                                              const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = averageRelativeError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ulong64> buffers(aBuffer.Pointer(),
                                             BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeAverageRelativeErrorMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                           aSrc2.Pitch(), buffers.template Get<0>(), buffers.template Get<1>(),
                                           aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region DotProduct
template <PixelType T> size_t ImageView<T>::DotProductBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = dotProduct_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T> size_t ImageView<T>::DotProductMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    // same as unmasked:
    return DotProductBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::DotProduct(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<dotProduct_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<remove_vector_t<dotProduct_types_for_rt<T>>> &aDstScalar,
                              mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = dotProduct_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeDotProductSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                           aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::DotProductMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<dotProduct_types_for_rt<T>> &aDst,
                                    mpp::cuda::DevVarView<remove_vector_t<dotProduct_types_for_rt<T>>> &aDstScalar,
                                    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = dotProduct_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeDotProductMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(),
                                 SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::DotProduct(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<dotProduct_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = dotProduct_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeDotProductSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                           aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::DotProductMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<dotProduct_types_for_rt<T>> &aDst,
                                    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = dotProduct_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeDotProductMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(),
                                 aStreamCtx);
}
#pragma endregion

#pragma region MSE
template <PixelType T> size_t ImageView<T>::MSEBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = mse_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T> size_t ImageView<T>::MSEMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = mse_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ulong64> buffers(nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
void ImageView<T>::MSE(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<mse_types_for_rt<T>> &aDst,
                       mpp::cuda::DevVarView<remove_vector_t<mse_types_for_rt<T>>> &aDstScalar,
                       mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = mse_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMSESrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(), aDst.Pointer(),
                    aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MSEMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<mse_types_for_rt<T>> &aDst,
                             mpp::cuda::DevVarView<remove_vector_t<mse_types_for_rt<T>>> &aDstScalar,
                             const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                             const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = mse_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ulong64> buffers(aBuffer.Pointer(),
                                             BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMSEMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                          buffers.template Get<0>(), buffers.template Get<1>(), aDst.Pointer(), aDstScalar.Pointer(),
                          SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MSE(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<mse_types_for_rt<T>> &aDst,
                       mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = mse_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMSESrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(), aDst.Pointer(),
                    nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MSEMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<mse_types_for_rt<T>> &aDst,
                             const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                             const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = mse_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ulong64> buffers(aBuffer.Pointer(),
                                             BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMSEMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
                          buffers.template Get<0>(), buffers.template Get<1>(), aDst.Pointer(), nullptr, SizeRoi(),
                          aStreamCtx);
}
#pragma endregion

#pragma region MaximumError
template <PixelType T> size_t ImageView<T>::MaximumErrorBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = normDiffInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T> size_t ImageView<T>::MaximumErrorMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    // same as unmasked:
    return MaximumErrorBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaximumError(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                                mpp::cuda::DevVarView<remove_vector_t<normDiffInf_types_for_rt<T>>> &aDstScalar,
                                mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = normDiffInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormDiffInfSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                            aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaximumErrorMasked(const ImageView<T> &aSrc2,
                                      mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                                      mpp::cuda::DevVarView<remove_vector_t<normDiffInf_types_for_rt<T>>> &aDstScalar,
                                      const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normDiffInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormDiffInfMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                  aSrc2.Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(),
                                  SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaximumError(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                                mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = normDiffInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormDiffInfSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                            aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaximumErrorMasked(const ImageView<T> &aSrc2,
                                      mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                                      const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normDiffInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormDiffInfMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                  aSrc2.Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(),
                                  aStreamCtx);
}
#pragma endregion

#pragma region MaximumRelativeError
template <PixelType T> size_t ImageView<T>::MaximumRelativeErrorBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = maxRelativeError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::MaximumRelativeErrorMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    // same as unmasked:
    return MaximumRelativeErrorBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaximumRelativeError(
    const ImageView<T> &aSrc2, mpp::cuda::DevVarView<maxRelativeError_types_for_rt<T>> &aDst,
    mpp::cuda::DevVarView<remove_vector_t<maxRelativeError_types_for_rt<T>>> &aDstScalar,
    mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = maxRelativeError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMaxRelativeErrorSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                                 aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaximumRelativeErrorMasked(
    const ImageView<T> &aSrc2, mpp::cuda::DevVarView<maxRelativeError_types_for_rt<T>> &aDst,
    mpp::cuda::DevVarView<remove_vector_t<maxRelativeError_types_for_rt<T>>> &aDstScalar,
    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = maxRelativeError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMaxRelativeErrorMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                       aSrc2.Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(),
                                       SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaximumRelativeError(const ImageView<T> &aSrc2,
                                        mpp::cuda::DevVarView<maxRelativeError_types_for_rt<T>> &aDst,
                                        mpp::cuda::DevVarView<byte> &aBuffer,
                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = maxRelativeError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMaxRelativeErrorSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                                 aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaximumRelativeErrorMasked(const ImageView<T> &aSrc2,
                                              mpp::cuda::DevVarView<maxRelativeError_types_for_rt<T>> &aDst,
                                              const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = maxRelativeError_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMaxRelativeErrorMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                       aSrc2.Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(),
                                       aStreamCtx);
}
#pragma endregion

#pragma region NormDiffInf
template <PixelType T>
size_t ImageView<T>::NormDiffInfBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return MaximumErrorBufferSize(aStreamCtx);
}

template <PixelType T>
size_t ImageView<T>::NormDiffInfMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return MaximumErrorMaskedBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormDiffInf(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                               mpp::cuda::DevVarView<remove_vector_t<normDiffInf_types_for_rt<T>>> &aDstScalar,
                               mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    MaximumError(aSrc2, aDst, aDstScalar, aBuffer, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormDiffInfMasked(const ImageView<T> &aSrc2,
                                     mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                                     mpp::cuda::DevVarView<remove_vector_t<normDiffInf_types_for_rt<T>>> &aDstScalar,
                                     const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    MaximumErrorMasked(aSrc2, aDst, aDstScalar, aMask, aBuffer, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormDiffInf(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                               mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    MaximumError(aSrc2, aDst, aBuffer, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormDiffInfMasked(const ImageView<T> &aSrc2,
                                     mpp::cuda::DevVarView<normDiffInf_types_for_rt<T>> &aDst,
                                     const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    MaximumErrorMasked(aSrc2, aDst, aMask, aBuffer, aStreamCtx);
}
#pragma endregion

#pragma region NormDiffL1
template <PixelType T>
size_t ImageView<T>::NormDiffL1BufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = normDiffL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::NormDiffL1MaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    // same as unmasked:
    return NormDiffL1BufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormDiffL1(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL1_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<remove_vector_t<normDiffL1_types_for_rt<T>>> &aDstScalar,
                              mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = normDiffL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormDiffL1SrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                           aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormDiffL1Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL1_types_for_rt<T>> &aDst,
                                    mpp::cuda::DevVarView<remove_vector_t<normDiffL1_types_for_rt<T>>> &aDstScalar,
                                    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normDiffL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormDiffL1MaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(),
                                 SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormDiffL1(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL1_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = normDiffL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormDiffL1SrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                           aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormDiffL1Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL1_types_for_rt<T>> &aDst,
                                    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normDiffL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormDiffL1MaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(),
                                 aStreamCtx);
}
#pragma endregion

#pragma region NormDiffL2
template <PixelType T>
size_t ImageView<T>::NormDiffL2BufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = normDiffL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::NormDiffL2MaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    // same as unmasked:
    return NormDiffL2BufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormDiffL2(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL2_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<remove_vector_t<normDiffL2_types_for_rt<T>>> &aDstScalar,
                              mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = normDiffL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormDiffL2SrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                           aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormDiffL2Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL2_types_for_rt<T>> &aDst,
                                    mpp::cuda::DevVarView<remove_vector_t<normDiffL2_types_for_rt<T>>> &aDstScalar,
                                    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normDiffL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormDiffL2MaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(),
                                 SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormDiffL2(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL2_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = normDiffL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormDiffL2SrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                           aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormDiffL2Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normDiffL2_types_for_rt<T>> &aDst,
                                    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normDiffL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormDiffL2MaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(),
                                 aStreamCtx);
}
#pragma endregion

#pragma region NormRelInf
template <PixelType T>
size_t ImageView<T>::NormRelInfBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = normDiffInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::NormRelInfMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    // same as unmasked:
    return NormRelInfBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormRelInf(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelInf_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<remove_vector_t<normRelInf_types_for_rt<T>>> &aDstScalar,
                              mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = normRelInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormRelInfSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                           buffers.template Get<1>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormRelInfMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelInf_types_for_rt<T>> &aDst,
                                    mpp::cuda::DevVarView<remove_vector_t<normRelInf_types_for_rt<T>>> &aDstScalar,
                                    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normRelInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormRelInfMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDst.Pointer(),
                                 aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormRelInf(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelInf_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = normRelInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormRelInfSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                           buffers.template Get<1>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormRelInfMasked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelInf_types_for_rt<T>> &aDst,
                                    const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normRelInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormRelInfMaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                 aSrc2.Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDst.Pointer(),
                                 nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region NormRelL1
template <PixelType T>
size_t ImageView<T>::NormRelL1BufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = normRelL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::NormRelL1MaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    // same as unmasked:
    return NormRelL1BufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormRelL1(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL1_types_for_rt<T>> &aDst,
                             mpp::cuda::DevVarView<remove_vector_t<normRelL1_types_for_rt<T>>> &aDstScalar,
                             mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = normRelL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormRelL1SrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                          buffers.template Get<1>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormRelL1Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL1_types_for_rt<T>> &aDst,
                                   mpp::cuda::DevVarView<remove_vector_t<normRelL1_types_for_rt<T>>> &aDstScalar,
                                   const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normRelL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormRelL1MaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                aSrc2.Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDst.Pointer(),
                                aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormRelL1(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL1_types_for_rt<T>> &aDst,
                             mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = normRelL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormRelL1SrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                          buffers.template Get<1>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormRelL1Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL1_types_for_rt<T>> &aDst,
                                   const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normRelL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormRelL1MaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                aSrc2.Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDst.Pointer(),
                                nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region NormRelL2
template <PixelType T>
size_t ImageView<T>::NormRelL2BufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = normRelL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::NormRelL2MaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    // same as unmasked:
    return NormRelL2BufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormRelL2(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL2_types_for_rt<T>> &aDst,
                             mpp::cuda::DevVarView<remove_vector_t<normRelL2_types_for_rt<T>>> &aDstScalar,
                             mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = normRelL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormRelL2SrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                          buffers.template Get<1>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormRelL2Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL2_types_for_rt<T>> &aDst,
                                   mpp::cuda::DevVarView<remove_vector_t<normRelL2_types_for_rt<T>>> &aDstScalar,
                                   const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normRelL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormRelL2MaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                aSrc2.Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDst.Pointer(),
                                aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormRelL2(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL2_types_for_rt<T>> &aDst,
                             mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = normRelL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormRelL2SrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                          buffers.template Get<1>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormRelL2Masked(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<normRelL2_types_for_rt<T>> &aDst,
                                   const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normRelL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormRelL2MaskedSrcSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), aSrc2.PointerRoi(),
                                aSrc2.Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDst.Pointer(),
                                nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region PSNR
template <PixelType T>
size_t ImageView<T>::PSNRBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = mse_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
void ImageView<T>::PSNR(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<mse_types_for_rt<T>> &aDst,
                        mpp::cuda::DevVarView<remove_vector_t<mse_types_for_rt<T>>> &aDstScalar,
                        remove_vector_t<mse_types_for_rt<T>> aValueRange, mpp::cuda::DevVarView<byte> &aBuffer,
                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = mse_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokePSNRSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aValueRange, buffers.template Get<0>(),
                     aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::PSNR(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<mse_types_for_rt<T>> &aDst,
                        remove_vector_t<mse_types_for_rt<T>> aValueRange, mpp::cuda::DevVarView<byte> &aBuffer,
                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = mse_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokePSNRSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aValueRange, buffers.template Get<0>(),
                     aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region QualityIndex
template <PixelType T>
size_t ImageView<T>::QualityIndexBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = qualityIndex_types_for_ct1<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT, ComputeT, ComputeT, ComputeT> buffers(
        nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor, SizeRoi().y / divisor, SizeRoi().y / divisor,
                             SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
void ImageView<T>::QualityIndex(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<qualityIndex_types_for_rt<T>> &aDst,
                                mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    using ComputeT = qualityIndex_types_for_ct1<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT, ComputeT, ComputeT, ComputeT> buffers(
        aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor, SizeRoi().y / divisor,
                                       SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeQualityIndexSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                             buffers.template Get<1>(), buffers.template Get<2>(), buffers.template Get<3>(),
                             buffers.template Get<4>(), aDst.Pointer(), SizeRoi(), aStreamCtx);
}
template <PixelType T>
size_t ImageView<T>::QualityIndexWindowBufferSize(const mpp::cuda::StreamCtx & /*aStreamCtx*/) const
    requires RealVector<T>
{
    using DstT = qiw_types_for_rt<T>;

    ScratchBuffer<DstT, DstT> buffers(nullptr, BufferSizes{SizeRoi().TotalSize(), SizeRoi().y});
    return buffers.GetTotalBufferSize();
}

template <PixelType T>
void ImageView<T>::QualityIndexWindow(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<qiw_types_for_rt<T>> &aDst,
                                      mpp::cuda::DevVarView<byte> &aBuffer,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using DstT = qiw_types_for_rt<T>;

    const Vector2<int> roiOffset(0, 0);

    ScratchBuffer<DstT, DstT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().TotalSize(), SizeRoi().y});
    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeQIWSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                    buffers.template Get<1>(), aDst.Pointer(), SizeRoi(), roiOffset, SizeRoi(), roiOffset, SizeRoi(),
                    aStreamCtx);
}
#pragma endregion

#pragma region SSIM
template <PixelType T>
size_t ImageView<T>::SSIMBufferSize(const mpp::cuda::StreamCtx & /*aStreamCtx*/) const
    requires RealVector<T>
{
    using DstT = ssim_types_for_rt<T>;

    const float scaleFactor = std::max(1.0f, std::round(to_float(SizeRoi().Min()) / 256.0f));

    if (scaleFactor > 1)
    {
        const Size2D scaledRoi = SizeRoi() / to_int(scaleFactor);

        ScratchBuffer<T, T, DstT, DstT> buffers(nullptr,
                                                BufferSizes{scaledRoi, scaledRoi, scaledRoi.TotalSize(), scaledRoi.y});
        return buffers.GetTotalBufferSize();
    }

    ScratchBuffer<DstT, DstT> buffers(nullptr, BufferSizes{SizeRoi().TotalSize(), SizeRoi().y});
    return buffers.GetTotalBufferSize();
}

template <PixelType T>
void ImageView<T>::SSIM(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<ssim_types_for_rt<T>> &aDst,
                        mpp::cuda::DevVarView<byte> &aBuffer, remove_vector_t<ssim_types_for_rt<T>> aDynamicRange,
                        remove_vector_t<ssim_types_for_rt<T>> aK1, remove_vector_t<ssim_types_for_rt<T>> aK2,
                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using DstT = ssim_types_for_rt<T>;

    const float scaleFactor = std::max(1.0f, std::round(to_float(SizeRoi().Min()) / 256.0f));

    const Vector2<int> roiOffset(0, 0);

    if (scaleFactor > 1)
    {
        const Size2D scaledRoi = SizeRoi() / to_int(scaleFactor);

        ScratchBuffer<T, T, DstT, DstT> buffers(aBuffer.Pointer(),
                                                BufferSizes{scaledRoi, scaledRoi, scaledRoi.TotalSize(), scaledRoi.y});

        CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());
        assert(buffers.GetSubBufferPitch(0) == buffers.GetSubBufferPitch(1));

        const size_t pitchTmp = buffers.GetSubBufferPitch(0);

        ImageView<T> resizeSrc1(buffers.template Get<0>(), scaledRoi, pitchTmp);
        ImageView<T> resizeSrc2(buffers.template Get<1>(), scaledRoi, pitchTmp);

        if constexpr (SingleChannel<T>)
        {
            using twoChannel = ImageView<Vector2<remove_vector_t<T>>>;
            twoChannel::Resize(*this, aSrc2, resizeSrc1, resizeSrc2, 1.0 / to_double(scaleFactor), 0,
                               InterpolationMode::Super, BorderType::Replicate, Roi(), aStreamCtx);
        }
        else
        {
            this->Resize(resizeSrc1, 1.0 / to_double(scaleFactor), 0, InterpolationMode::Super, BorderType::Replicate,
                         Roi(), aStreamCtx);
            aSrc2.Resize(resizeSrc2, 1.0 / to_double(scaleFactor), 0, InterpolationMode::Super, BorderType::Replicate,
                         Roi(), aStreamCtx);
        }

        InvokeSSIMSrcSrc(resizeSrc1.PointerRoi(), resizeSrc1.Pitch(), resizeSrc2.PointerRoi(), resizeSrc2.Pitch(),
                         buffers.template Get<2>(), buffers.template Get<3>(), aDst.Pointer(), aDynamicRange, aK1, aK2,
                         scaledRoi, roiOffset, scaledRoi, roiOffset, scaledRoi, aStreamCtx);
    }
    else
    {
        ScratchBuffer<DstT, DstT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().TotalSize(), SizeRoi().y});
        CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

        InvokeSSIMSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<0>(),
                         buffers.template Get<1>(), aDst.Pointer(), aDynamicRange, aK1, aK2, SizeRoi(), roiOffset,
                         SizeRoi(), roiOffset, SizeRoi(), aStreamCtx);
    }
}
#pragma endregion

#pragma region MSSSIM
template <PixelType T>
size_t ImageView<T>::MSSSIMBufferSize(const mpp::cuda::StreamCtx & /*aStreamCtx*/) const
    requires RealVector<T>
{
    using DstT = ssim_types_for_rt<T>;

    const Size2D scaledRoi1 = SizeRoi() / 2;
    const Size2D scaledRoi2 = scaledRoi1 / 2;

    ScratchBuffer<T, T, T, T, DstT, DstT> buffers(
        nullptr, BufferSizes{scaledRoi1, scaledRoi1, scaledRoi2, scaledRoi2, SizeRoi().TotalSize(), SizeRoi().y});
    return buffers.GetTotalBufferSize();
}

template <PixelType T>
void ImageView<T>::MSSSIM(const ImageView<T> &aSrc2, mpp::cuda::DevVarView<ssim_types_for_rt<T>> &aDst,
                          mpp::cuda::DevVarView<byte> &aBuffer, remove_vector_t<ssim_types_for_rt<T>> aDynamicRange,
                          remove_vector_t<ssim_types_for_rt<T>> aK1, remove_vector_t<ssim_types_for_rt<T>> aK2,
                          const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using DstT       = ssim_types_for_rt<T>;
    using twoChannel = ImageView<Vector2<remove_vector_t<T>>>;

    const Vector2<int> roiOffset(0, 0);

    const Size2D scaledRoi1 = SizeRoi() / 2;
    const Size2D scaledRoi2 = scaledRoi1 / 2;

    ScratchBuffer<T, T, T, T, DstT, DstT> buffers(
        aBuffer.Pointer(),
        BufferSizes{scaledRoi1, scaledRoi1, scaledRoi2, scaledRoi2, SizeRoi().TotalSize(), SizeRoi().y});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    ImageView<T> resizeSrc11(buffers.template Get<0>(), scaledRoi1, buffers.GetSubBufferPitch(0));
    ImageView<T> resizeSrc12(buffers.template Get<1>(), scaledRoi1, buffers.GetSubBufferPitch(1));
    ImageView<T> resizeSrc21(buffers.template Get<2>(), scaledRoi2, buffers.GetSubBufferPitch(2));
    ImageView<T> resizeSrc22(buffers.template Get<3>(), scaledRoi2, buffers.GetSubBufferPitch(3));

    Size2D resizedRoiSize = SizeRoi();

    int level = 0;
    InvokeMSSSIMSrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), buffers.template Get<4>(),
                       buffers.template Get<5>(), aDst.Pointer(), level, aDynamicRange, aK1, aK2, SizeRoi(), roiOffset,
                       SizeRoi(), roiOffset, SizeRoi(), aStreamCtx);

    resizedRoiSize /= 2;
    if constexpr (SingleChannel<T>)
    {
        twoChannel::Resize(*this, aSrc2, resizeSrc11, resizeSrc12, 0.5, 0, InterpolationMode::Super,
                           BorderType::Replicate, Roi(), aStreamCtx);
    }
    else
    {
        this->Resize(resizeSrc11, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi(), aStreamCtx);
        aSrc2.Resize(resizeSrc12, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi(), aStreamCtx);
    }

    level = 1;
    InvokeMSSSIMSrcSrc(resizeSrc11.PointerRoi(), resizeSrc11.Pitch(), resizeSrc12.PointerRoi(), resizeSrc12.Pitch(),
                       buffers.template Get<4>(), buffers.template Get<5>(), aDst.Pointer(), level, aDynamicRange, aK1,
                       aK2, resizedRoiSize, roiOffset, resizedRoiSize, roiOffset, resizedRoiSize, aStreamCtx);

    resizedRoiSize /= 2;
    if constexpr (SingleChannel<T>)
    {
        twoChannel::Resize(resizeSrc11, resizeSrc12, resizeSrc21, resizeSrc22, 0.5, 0, InterpolationMode::Super,
                           BorderType::Replicate, Roi(), aStreamCtx);
    }
    else
    {
        resizeSrc11.Resize(resizeSrc21, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi(), aStreamCtx);
        resizeSrc12.Resize(resizeSrc22, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi(), aStreamCtx);
    }

    level = 2;
    InvokeMSSSIMSrcSrc(resizeSrc21.PointerRoi(), resizeSrc21.Pitch(), resizeSrc22.PointerRoi(), resizeSrc22.Pitch(),
                       buffers.template Get<4>(), buffers.template Get<5>(), aDst.Pointer(), level, aDynamicRange, aK1,
                       aK2, resizedRoiSize, roiOffset, resizedRoiSize, roiOffset, resizedRoiSize, aStreamCtx);

    resizedRoiSize /= 2;
    resizeSrc11.SetRoi(Roi({0}, resizedRoiSize));
    resizeSrc12.SetRoi(Roi({0}, resizedRoiSize));
    if constexpr (SingleChannel<T>)
    {
        twoChannel::Resize(resizeSrc21, resizeSrc22, resizeSrc11, resizeSrc12, 0.5, 0, InterpolationMode::Super,
                           BorderType::Replicate, Roi(), aStreamCtx);
    }
    else
    {
        resizeSrc21.Resize(resizeSrc11, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi(), aStreamCtx);
        resizeSrc22.Resize(resizeSrc12, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi(), aStreamCtx);
    }

    level = 3;
    InvokeMSSSIMSrcSrc(resizeSrc11.PointerRoi(), resizeSrc11.Pitch(), resizeSrc12.PointerRoi(), resizeSrc12.Pitch(),
                       buffers.template Get<4>(), buffers.template Get<5>(), aDst.Pointer(), level, aDynamicRange, aK1,
                       aK2, resizedRoiSize, roiOffset, resizedRoiSize, roiOffset, resizedRoiSize, aStreamCtx);

    resizedRoiSize /= 2;
    resizeSrc21.SetRoi(Roi({0}, resizedRoiSize));
    resizeSrc22.SetRoi(Roi({0}, resizedRoiSize));
    if constexpr (SingleChannel<T>)
    {
        twoChannel::Resize(resizeSrc11, resizeSrc12, resizeSrc21, resizeSrc22, 0.5, 0, InterpolationMode::Super,
                           BorderType::Replicate, Roi(), aStreamCtx);
    }
    else
    {
        resizeSrc11.Resize(resizeSrc21, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi(), aStreamCtx);
        resizeSrc12.Resize(resizeSrc22, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi(), aStreamCtx);
    }

    level = 4;
    InvokeMSSSIMSrcSrc(resizeSrc21.PointerRoi(), resizeSrc21.Pitch(), resizeSrc22.PointerRoi(), resizeSrc22.Pitch(),
                       buffers.template Get<4>(), buffers.template Get<5>(), aDst.Pointer(), level, aDynamicRange, aK1,
                       aK2, resizedRoiSize, roiOffset, resizedRoiSize, roiOffset, resizedRoiSize, aStreamCtx);
}
#pragma endregion

#pragma region NormInf
template <PixelType T>
size_t ImageView<T>::NormInfBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = normInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::NormInfMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    // same as unmasked
    return NormInfBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormInf(mpp::cuda::DevVarView<normInf_types_for_rt<T>> &aDst,
                           mpp::cuda::DevVarView<remove_vector_t<normInf_types_for_rt<T>>> &aDstScalar,
                           mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using ComputeT = normInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormInfSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(),
                     aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormInfMasked(mpp::cuda::DevVarView<normInf_types_for_rt<T>> &aDst,
                                 mpp::cuda::DevVarView<remove_vector_t<normInf_types_for_rt<T>>> &aDstScalar,
                                 const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                 const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormInfMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                           aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormInf(mpp::cuda::DevVarView<normInf_types_for_rt<T>> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
                           const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using ComputeT = normInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormInfSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormInfMasked(mpp::cuda::DevVarView<normInf_types_for_rt<T>> &aDst,
                                 const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                 const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normInf_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormInfMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                           aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region NormL1
template <PixelType T>
size_t ImageView<T>::NormL1BufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = normL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::NormL1MaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    // same as unmasked
    return NormL1BufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormL1(mpp::cuda::DevVarView<normL1_types_for_rt<T>> &aDst,
                          mpp::cuda::DevVarView<remove_vector_t<normL1_types_for_rt<T>>> &aDstScalar,
                          mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using ComputeT = normL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormL1Src(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(),
                    aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormL1Masked(mpp::cuda::DevVarView<normL1_types_for_rt<T>> &aDst,
                                mpp::cuda::DevVarView<remove_vector_t<normL1_types_for_rt<T>>> &aDstScalar,
                                const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormL1MaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                          aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormL1(mpp::cuda::DevVarView<normL1_types_for_rt<T>> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
                          const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using ComputeT = normL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormL1Src(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormL1Masked(mpp::cuda::DevVarView<normL1_types_for_rt<T>> &aDst, const ImageView<Pixel8uC1> &aMask,
                                mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normL1_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormL1MaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                          aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region NormL2
template <PixelType T>
size_t ImageView<T>::NormL2BufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = normL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::NormL2MaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    // same as unmasked
    return NormL2BufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormL2(mpp::cuda::DevVarView<normL2_types_for_rt<T>> &aDst,
                          mpp::cuda::DevVarView<remove_vector_t<normL2_types_for_rt<T>>> &aDstScalar,
                          mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using ComputeT = normL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormL2Src(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(),
                    aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormL2Masked(mpp::cuda::DevVarView<normL2_types_for_rt<T>> &aDst,
                                mpp::cuda::DevVarView<remove_vector_t<normL2_types_for_rt<T>>> &aDstScalar,
                                const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormL2MaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                          aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormL2(mpp::cuda::DevVarView<normL2_types_for_rt<T>> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
                          const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using ComputeT = normL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormL2Src(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::NormL2Masked(mpp::cuda::DevVarView<normL2_types_for_rt<T>> &aDst, const ImageView<Pixel8uC1> &aMask,
                                mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = normL2_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeNormL2MaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                          aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region Sum
template <PixelType T>
size_t ImageView<T>::SumBufferSize(const mpp::cuda::DevVarView<sum_types_for_rt<T, 1>> & /*aDst*/,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = sum_types_for_ct<T, 1>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::SumMaskedBufferSize(const mpp::cuda::DevVarView<sum_types_for_rt<T, 1>> &aDst,
                                         const mpp::cuda::StreamCtx &aStreamCtx) const
{
    return SumBufferSize(aDst, aStreamCtx);
}

template <PixelType T>
size_t ImageView<T>::SumBufferSize(const mpp::cuda::DevVarView<sum_types_for_rt<T, 2>> & /*aDst*/,
                                   const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    using ComputeT = sum_types_for_ct<T, 2>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::SumMaskedBufferSize(const mpp::cuda::DevVarView<sum_types_for_rt<T, 2>> &aDst,
                                         const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T>
{
    return SumBufferSize(aDst, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Sum(mpp::cuda::DevVarView<sum_types_for_rt<T, 1>> &aDst,
                       mpp::cuda::DevVarView<remove_vector_t<sum_types_for_rt<T, 1>>> &aDstScalar,
                       mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    using ComputeT = sum_types_for_ct<T, 1>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeSumSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(),
                 aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Sum(mpp::cuda::DevVarView<sum_types_for_rt<T, 2>> &aDst,
                       mpp::cuda::DevVarView<remove_vector_t<sum_types_for_rt<T, 2>>> &aDstScalar,
                       mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T> && (vector_active_size_v<T> > 1)
{
    using ComputeT = sum_types_for_ct<T, 2>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeSumSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(),
                 aStreamCtx);
}

template <PixelType T>
void ImageView<T>::SumMasked(mpp::cuda::DevVarView<sum_types_for_rt<T, 1>> &aDst,
                             mpp::cuda::DevVarView<remove_vector_t<sum_types_for_rt<T, 1>>> &aDstScalar,
                             const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                             const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = sum_types_for_ct<T, 1>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeSumMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                       aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::SumMasked(mpp::cuda::DevVarView<sum_types_for_rt<T, 2>> &aDst,
                             mpp::cuda::DevVarView<remove_vector_t<sum_types_for_rt<T, 2>>> &aDstScalar,
                             const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                             const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = sum_types_for_ct<T, 2>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeSumMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                       aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Sum(mpp::cuda::DevVarView<sum_types_for_rt<T, 1>> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
                       const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    using ComputeT = sum_types_for_ct<T, 1>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeSumSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Sum(mpp::cuda::DevVarView<sum_types_for_rt<T, 2>> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
                       const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T> && (vector_active_size_v<T> == 1)
{
    using ComputeT = sum_types_for_ct<T, 2>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeSumSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::SumMasked(mpp::cuda::DevVarView<sum_types_for_rt<T, 1>> &aDst, const ImageView<Pixel8uC1> &aMask,
                             mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = sum_types_for_ct<T, 1>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeSumMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                       aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::SumMasked(mpp::cuda::DevVarView<sum_types_for_rt<T, 2>> &aDst, const ImageView<Pixel8uC1> &aMask,
                             mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealOrComplexIntVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = sum_types_for_ct<T, 2>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeSumMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                       aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region Mean
template <PixelType T> size_t ImageView<T>::MeanBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = mean_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T> size_t ImageView<T>::MeanMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = mean_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ulong64> buffers(nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
void ImageView<T>::Mean(mpp::cuda::DevVarView<mean_types_for_rt<T>> &aDst,
                        mpp::cuda::DevVarView<remove_vector_t<mean_types_for_rt<T>>> &aDstScalar,
                        mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    using ComputeT = mean_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMeanSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(),
                  aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MeanMasked(mpp::cuda::DevVarView<mean_types_for_rt<T>> &aDst,
                              mpp::cuda::DevVarView<remove_vector_t<mean_types_for_rt<T>>> &aDstScalar,
                              const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = mean_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ulong64> buffers(aBuffer.Pointer(),
                                             BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMeanMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                        buffers.template Get<1>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Mean(mpp::cuda::DevVarView<mean_types_for_rt<T>> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    using ComputeT = mean_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMeanSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MeanMasked(mpp::cuda::DevVarView<mean_types_for_rt<T>> &aDst, const ImageView<Pixel8uC1> &aMask,
                              mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = mean_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ulong64> buffers(aBuffer.Pointer(),
                                             BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMeanMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                        buffers.template Get<1>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region MeanStd
template <PixelType T> size_t ImageView<T>::MeanStdBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = meanStd_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T> size_t ImageView<T>::MeanStdMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
{
    using ComputeT = meanStd_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT, ulong64> buffers(
        nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor, SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
void ImageView<T>::MeanStd(mpp::cuda::DevVarView<meanStd_types_for_rt1<T>> &aMean,
                           mpp::cuda::DevVarView<meanStd_types_for_rt2<T>> &aStd,
                           mpp::cuda::DevVarView<remove_vector_t<meanStd_types_for_rt1<T>>> &aMeanScalar,
                           mpp::cuda::DevVarView<remove_vector_t<meanStd_types_for_rt2<T>>> &aStdScalar,
                           mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    using ComputeT = meanStd_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMeanStdSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aMean.Pointer(),
                     aStd.Pointer(), aMeanScalar.Pointer(), aStdScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MeanStdMasked(mpp::cuda::DevVarView<meanStd_types_for_rt1<T>> &aMean,
                                 mpp::cuda::DevVarView<meanStd_types_for_rt2<T>> &aStd,
                                 mpp::cuda::DevVarView<remove_vector_t<meanStd_types_for_rt1<T>>> &aMeanScalar,
                                 mpp::cuda::DevVarView<remove_vector_t<meanStd_types_for_rt2<T>>> &aStdScalar,
                                 const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                 const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = meanStd_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT, ulong64> buffers(
        aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMeanStdMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                           buffers.template Get<1>(), buffers.template Get<2>(), aMean.Pointer(), aStd.Pointer(),
                           aMeanScalar.Pointer(), aStdScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MeanStd(mpp::cuda::DevVarView<meanStd_types_for_rt1<T>> &aMean,
                           mpp::cuda::DevVarView<meanStd_types_for_rt2<T>> &aStd, mpp::cuda::DevVarView<byte> &aBuffer,
                           const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    using ComputeT = meanStd_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMeanStdSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aMean.Pointer(),
                     aStd.Pointer(), nullptr, nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MeanStdMasked(mpp::cuda::DevVarView<meanStd_types_for_rt1<T>> &aMean,
                                 mpp::cuda::DevVarView<meanStd_types_for_rt2<T>> &aStd,
                                 const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                 const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = meanStd_types_for_ct<T>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT, ulong64> buffers(
        aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMeanStdMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                           buffers.template Get<1>(), buffers.template Get<2>(), aMean.Pointer(), aStd.Pointer(),
                           nullptr, nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region CountInRange
template <PixelType T>
size_t ImageView<T>::CountInRangeBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = same_vector_size_different_type_t<T, size_t>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::CountInRangeMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return CountInRangeBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::CountInRange(const T &aLowerLimit, const T &aUpperLimit,
                                mpp::cuda::DevVarView<same_vector_size_different_type_t<T, size_t>> &aDst,
                                mpp::cuda::DevVarView<size_t> &aDstScalar, mpp::cuda::DevVarView<byte> &aBuffer,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using ComputeT = same_vector_size_different_type_t<T, size_t>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeCountInRangeSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(),
                          aLowerLimit, aUpperLimit, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::CountInRangeMasked(const T &aLowerLimit, const T &aUpperLimit,
                                      mpp::cuda::DevVarView<same_vector_size_different_type_t<T, size_t>> &aDst,
                                      mpp::cuda::DevVarView<size_t> &aDstScalar, const ImageView<Pixel8uC1> &aMask,
                                      mpp::cuda::DevVarView<byte> &aBuffer,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = same_vector_size_different_type_t<T, size_t>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeCountInRangeMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                                aDst.Pointer(), aDstScalar.Pointer(), aLowerLimit, aUpperLimit, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::CountInRange(const T &aLowerLimit, const T &aUpperLimit,
                                mpp::cuda::DevVarView<same_vector_size_different_type_t<T, size_t>> &aDst,
                                mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using ComputeT = same_vector_size_different_type_t<T, size_t>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeCountInRangeSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, aLowerLimit,
                          aUpperLimit, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::CountInRangeMasked(const T &aLowerLimit, const T &aUpperLimit,
                                      mpp::cuda::DevVarView<same_vector_size_different_type_t<T, size_t>> &aDst,
                                      const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = same_vector_size_different_type_t<T, size_t>;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeCountInRangeMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                                aDst.Pointer(), nullptr, aLowerLimit, aUpperLimit, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region Min
template <PixelType T>
size_t ImageView<T>::MinBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::MinMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return MinBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Min(mpp::cuda::DevVarView<T> &aDst, mpp::cuda::DevVarView<remove_vector_t<T>> &aDstScalar,
                       mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(),
                 aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinMasked(mpp::cuda::DevVarView<T> &aDst, mpp::cuda::DevVarView<remove_vector_t<T>> &aDstScalar,
                             const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                             const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                       aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Min(mpp::cuda::DevVarView<T> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
                       const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinMasked(mpp::cuda::DevVarView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                             mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                       aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion
#pragma region Max
template <PixelType T>
size_t ImageView<T>::MaxBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::MaxMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return MaxBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Max(mpp::cuda::DevVarView<T> &aDst, mpp::cuda::DevVarView<remove_vector_t<T>> &aDstScalar,
                       mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMaxSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(),
                 aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaxMasked(mpp::cuda::DevVarView<T> &aDst, mpp::cuda::DevVarView<remove_vector_t<T>> &aDstScalar,
                             const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                             const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMaxMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                       aDst.Pointer(), aDstScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::Max(mpp::cuda::DevVarView<T> &aDst, mpp::cuda::DevVarView<byte> &aBuffer,
                       const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMaxSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaxMasked(mpp::cuda::DevVarView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                             mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMaxMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                       aDst.Pointer(), nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion
#pragma region MinMax
template <PixelType T>
size_t ImageView<T>::MinMaxBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::MinMaxMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return MinMaxBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinMax(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                          mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMinScalar,
                          mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMaxScalar,
                          mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinMaxSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDstMin.Pointer(),
                    aDstMax.Pointer(), aDstMinScalar.Pointer(), aDstMaxScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinMaxMasked(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                                mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMinScalar,
                                mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMaxScalar,
                                const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinMaxMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                          buffers.template Get<1>(), aDstMin.Pointer(), aDstMax.Pointer(), aDstMinScalar.Pointer(),
                          aDstMaxScalar.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinMax(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                          mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinMaxSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDstMin.Pointer(),
                    aDstMax.Pointer(), nullptr, nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinMaxMasked(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                                const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT> buffers(aBuffer.Pointer(),
                                              BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinMaxMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                          buffers.template Get<1>(), aDstMin.Pointer(), aDstMax.Pointer(), nullptr, nullptr, SizeRoi(),
                          aStreamCtx);
}
#pragma endregion
#pragma region MinIndex
template <PixelType T>
size_t ImageView<T>::MinIndexBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, same_vector_size_different_type_t<T, int>> buffers(
        nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::MinIndexMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return MinIndexBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinIndex(mpp::cuda::DevVarView<T> &aDstMin,
                            mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                            mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                            mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMinScalar,
                            mpp::cuda::DevVarView<Vector3<int>> &aDstScalarIdx, mpp::cuda::DevVarView<byte> &aBuffer,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, same_vector_size_different_type_t<T, int>> buffers(
        aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinIdxSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDstMin.Pointer(),
                    aDstIndexX.Pointer(), aDstIndexY.Pointer(), aDstMinScalar.Pointer(), aDstScalarIdx.Pointer(),
                    SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinIndexMasked(mpp::cuda::DevVarView<T> &aDstMin,
                                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                                  mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMinScalar,
                                  mpp::cuda::DevVarView<Vector3<int>> &aDstScalarIdx, const ImageView<Pixel8uC1> &aMask,
                                  mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, same_vector_size_different_type_t<T, int>> buffers(
        aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinIdxMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                          buffers.template Get<1>(), aDstMin.Pointer(), aDstIndexX.Pointer(), aDstIndexY.Pointer(),
                          aDstMinScalar.Pointer(), aDstScalarIdx.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinIndex(mpp::cuda::DevVarView<T> &aDstMin,
                            mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                            mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                            mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, same_vector_size_different_type_t<T, int>> buffers(
        aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinIdxSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDstMin.Pointer(),
                    aDstIndexX.Pointer(), aDstIndexY.Pointer(), nullptr, nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinIndexMasked(mpp::cuda::DevVarView<T> &aDstMin,
                                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                                  const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                  const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, same_vector_size_different_type_t<T, int>> buffers(
        aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinIdxMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                          buffers.template Get<1>(), aDstMin.Pointer(), aDstIndexX.Pointer(), aDstIndexY.Pointer(),
                          nullptr, nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion
#pragma region MaxIndex
template <PixelType T>
size_t ImageView<T>::MaxIndexBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, same_vector_size_different_type_t<T, int>> buffers(
        nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::MaxIndexMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return MaxIndexBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaxIndex(mpp::cuda::DevVarView<T> &aDstMax,
                            mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                            mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                            mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMaxScalar,
                            mpp::cuda::DevVarView<Vector3<int>> &aDstScalarIdx, mpp::cuda::DevVarView<byte> &aBuffer,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, same_vector_size_different_type_t<T, int>> buffers(
        aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMaxIdxSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDstMax.Pointer(),
                    aDstIndexX.Pointer(), aDstIndexY.Pointer(), aDstMaxScalar.Pointer(), aDstScalarIdx.Pointer(),
                    SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaxIndexMasked(mpp::cuda::DevVarView<T> &aDstMax,
                                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                                  mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMaxScalar,
                                  mpp::cuda::DevVarView<Vector3<int>> &aDstScalarIdx, const ImageView<Pixel8uC1> &aMask,
                                  mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, same_vector_size_different_type_t<T, int>> buffers(
        aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMaxIdxMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                          buffers.template Get<1>(), aDstMax.Pointer(), aDstIndexX.Pointer(), aDstIndexY.Pointer(),
                          aDstMaxScalar.Pointer(), aDstScalarIdx.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaxIndex(mpp::cuda::DevVarView<T> &aDstMax,
                            mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                            mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                            mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, same_vector_size_different_type_t<T, int>> buffers(
        aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMaxIdxSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.template Get<1>(), aDstMax.Pointer(),
                    aDstIndexX.Pointer(), aDstIndexY.Pointer(), nullptr, nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MaxIndexMasked(mpp::cuda::DevVarView<T> &aDstMax,
                                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexX,
                                  mpp::cuda::DevVarView<same_vector_size_different_type_t<T, int>> &aDstIndexY,
                                  const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                  const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, same_vector_size_different_type_t<T, int>> buffers(
        aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMaxIdxMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                          buffers.template Get<1>(), aDstMax.Pointer(), aDstIndexX.Pointer(), aDstIndexY.Pointer(),
                          nullptr, nullptr, SizeRoi(), aStreamCtx);
}
#pragma endregion
#pragma region MinMaxIndex
template <PixelType T>
size_t ImageView<T>::MinMaxIndexBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT, same_vector_size_different_type_t<T, int>,
                  same_vector_size_different_type_t<T, int>>
        buffers(nullptr, BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor, SizeRoi().y / divisor,
                                     SizeRoi().y / divisor});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::MinMaxIndexMaskedBufferSize(const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return MinMaxIndexBufferSize(aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinMaxIndex(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                               mpp::cuda::DevVarView<IndexMinMax> &aDstIdx,
                               mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMinScalar,
                               mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMaxScalar,
                               mpp::cuda::DevVarView<IndexMinMaxChannel> &aDstScalarIdx,
                               mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT, same_vector_size_different_type_t<T, int>,
                  same_vector_size_different_type_t<T, int>>
        buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor, SizeRoi().y / divisor,
                                               SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinMaxIdxSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.template Get<1>(),
                       buffers.template Get<2>(), buffers.template Get<3>(), aDstMin.Pointer(), aDstMax.Pointer(),
                       aDstIdx.Pointer(), aDstMinScalar.Pointer(), aDstMaxScalar.Pointer(), aDstScalarIdx.Pointer(),
                       SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinMaxIndexMasked(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                                     mpp::cuda::DevVarView<IndexMinMax> &aDstIdx,
                                     mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMinScalar,
                                     mpp::cuda::DevVarView<remove_vector_t<T>> &aDstMaxScalar,
                                     mpp::cuda::DevVarView<IndexMinMaxChannel> &aDstScalarIdx,
                                     const ImageView<Pixel8uC1> &aMask, mpp::cuda::DevVarView<byte> &aBuffer,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT, same_vector_size_different_type_t<T, int>,
                  same_vector_size_different_type_t<T, int>>
        buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor, SizeRoi().y / divisor,
                                               SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinMaxIdxMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                             buffers.template Get<1>(), buffers.template Get<2>(), buffers.template Get<3>(),
                             aDstMin.Pointer(), aDstMax.Pointer(), aDstIdx.Pointer(), aDstMinScalar.Pointer(),
                             aDstMaxScalar.Pointer(), aDstScalarIdx.Pointer(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinMaxIndex(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                               mpp::cuda::DevVarView<IndexMinMax> &aDstIdx, mpp::cuda::DevVarView<byte> &aBuffer,
                               const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT, same_vector_size_different_type_t<T, int>,
                  same_vector_size_different_type_t<T, int>>
        buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor, SizeRoi().y / divisor,
                                               SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinMaxIdxSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.template Get<1>(),
                       buffers.template Get<2>(), buffers.template Get<3>(), aDstMin.Pointer(), aDstMax.Pointer(),
                       aDstIdx.Pointer(), nullptr, nullptr, nullptr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::MinMaxIndexMasked(mpp::cuda::DevVarView<T> &aDstMin, mpp::cuda::DevVarView<T> &aDstMax,
                                     mpp::cuda::DevVarView<IndexMinMax> &aDstIdx, const ImageView<Pixel8uC1> &aMask,
                                     mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());
    using ComputeT = T;

    int divisor = 1;
    if (aStreamCtx.ComputeCapabilityMajor < std::numeric_limits<int>::max())
    {
        divisor = ConfigBlockSize<"DefaultReductionX">::value.y;
    }

    ScratchBuffer<ComputeT, ComputeT, same_vector_size_different_type_t<T, int>,
                  same_vector_size_different_type_t<T, int>>
        buffers(aBuffer.Pointer(), BufferSizes{SizeRoi().y / divisor, SizeRoi().y / divisor, SizeRoi().y / divisor,
                                               SizeRoi().y / divisor});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeMinMaxIdxMaskedSrc(aMask.PointerRoi(), aMask.Pitch(), PointerRoi(), Pitch(), buffers.template Get<0>(),
                             buffers.template Get<1>(), buffers.template Get<2>(), buffers.template Get<3>(),
                             aDstMin.Pointer(), aDstMax.Pointer(), aDstIdx.Pointer(), nullptr, nullptr, nullptr,
                             SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region Integral
template <PixelType T>
size_t ImageView<T>::IntegralBufferSize(ImageView<same_vector_size_different_type_t<T, int>> & /*aDst*/,
                                        const mpp::cuda::StreamCtx & /*aStreamCtx*/) const
    requires RealIntVector<T> && NoAlpha<T>
{
    using DstT = same_vector_size_different_type_t<T, int>;

    ScratchBuffer<DstT> buffers(nullptr, BufferSizes{SizeRoi() + 1});

    return buffers.GetTotalBufferSize();
}
template <PixelType T>
size_t ImageView<T>::IntegralBufferSize(ImageView<same_vector_size_different_type_t<T, float>> & /*aDst*/,
                                        const mpp::cuda::StreamCtx & /*aStreamCtx*/) const
    requires RealVector<T> && NoAlpha<T> && (!std::same_as<double, remove_vector_t<T>>)
{
    using DstT = same_vector_size_different_type_t<T, float>;

    ScratchBuffer<DstT> buffers(nullptr, BufferSizes{SizeRoi() + 1});

    return buffers.GetTotalBufferSize();
}
template <PixelType T>
size_t ImageView<T>::IntegralBufferSize(ImageView<same_vector_size_different_type_t<T, long64>> & /*aDst*/,
                                        const mpp::cuda::StreamCtx & /*aStreamCtx*/) const
    requires RealIntVector<T> && NoAlpha<T>
{
    using DstT = same_vector_size_different_type_t<T, long64>;

    ScratchBuffer<DstT> buffers(nullptr, BufferSizes{SizeRoi() + 1});

    return buffers.GetTotalBufferSize();
}
template <PixelType T>
size_t ImageView<T>::IntegralBufferSize(ImageView<same_vector_size_different_type_t<T, double>> & /*aDst*/,
                                        const mpp::cuda::StreamCtx & /*aStreamCtx*/) const
    requires RealVector<T> && NoAlpha<T>
{
    using DstT = same_vector_size_different_type_t<T, double>;

    ScratchBuffer<DstT> buffers(nullptr, BufferSizes{SizeRoi() + 1});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::SqrIntegralBufferSize(ImageView<same_vector_size_different_type_t<T, int>> & /*aDst*/,
                                           ImageView<same_vector_size_different_type_t<T, int>> & /*aSqr*/,
                                           const mpp::cuda::StreamCtx & /*aStreamCtx*/) const
    requires RealIntVector<T> && NoAlpha<T>
{
    using DstT    = same_vector_size_different_type_t<T, int>;
    using DstSqrT = same_vector_size_different_type_t<T, int>;

    ScratchBuffer<DstT, DstSqrT> buffers(nullptr, BufferSizes{SizeRoi() + 1, SizeRoi() + 1});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::SqrIntegralBufferSize(ImageView<same_vector_size_different_type_t<T, int>> & /*aDst*/,
                                           ImageView<same_vector_size_different_type_t<T, long64>> & /*aSqr*/,
                                           const mpp::cuda::StreamCtx & /*aStreamCtx*/) const
    requires RealIntVector<T> && NoAlpha<T>
{
    using DstT    = same_vector_size_different_type_t<T, int>;
    using DstSqrT = same_vector_size_different_type_t<T, long64>;

    ScratchBuffer<DstT, DstSqrT> buffers(nullptr, BufferSizes{SizeRoi() + 1, SizeRoi() + 1});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::SqrIntegralBufferSize(ImageView<same_vector_size_different_type_t<T, float>> & /*aDst*/,
                                           ImageView<same_vector_size_different_type_t<T, double>> & /*aSqr*/,
                                           const mpp::cuda::StreamCtx & /*aStreamCtx*/) const
    requires RealVector<T> && NoAlpha<T> && (!std::same_as<double, remove_vector_t<T>>)
{
    using DstT    = same_vector_size_different_type_t<T, float>;
    using DstSqrT = same_vector_size_different_type_t<T, double>;

    ScratchBuffer<DstT, DstSqrT> buffers(nullptr, BufferSizes{SizeRoi() + 1, SizeRoi() + 1});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
size_t ImageView<T>::SqrIntegralBufferSize(ImageView<same_vector_size_different_type_t<T, double>> & /*aDst*/,
                                           ImageView<same_vector_size_different_type_t<T, double>> & /*aSqr*/,
                                           const mpp::cuda::StreamCtx & /*aStreamCtx*/) const
    requires RealVector<T> && NoAlpha<T>
{
    using DstT    = same_vector_size_different_type_t<T, double>;
    using DstSqrT = same_vector_size_different_type_t<T, double>;

    ScratchBuffer<DstT, DstSqrT> buffers(nullptr, BufferSizes{SizeRoi() + 1, SizeRoi() + 1});

    return buffers.GetTotalBufferSize();
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, int>> &ImageView<T>::Integral(
    ImageView<same_vector_size_different_type_t<T, int>> &aDst, const same_vector_size_different_type_t<T, int> &aVal,
    mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T> && NoAlpha<T>
{
    if (SizeRoi() != aDst.SizeRoi() - 1)
    {
        throw ROIEXCEPTION(
            "ROI of destination image must be one pixel larger in width and height than original ROI. Original size: "
            << SizeRoi() << " provided destination image size: " << aDst.SizeRoi());
    }

    using DstT = same_vector_size_different_type_t<T, int>;

    const size_t pitchTmp = PadImageWidthToPitch(sizeof(DstT) * to_size_t(aDst.ROI().width));
    const size_t sizeTmp  = pitchTmp * to_size_t(aDst.ROI().height);

    ScratchBuffer<DstT> buffers(aBuffer.Pointer(), BufferSizes{SizeRoi() + 1});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeIntegralSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.GetSubBufferPitch(0), aDst.Pointer(),
                      aDst.Pitch(), aVal, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::Integral(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst,
    const same_vector_size_different_type_t<T, float> &aVal, mpp::cuda::DevVarView<byte> &aBuffer,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && NoAlpha<T> && (!std::same_as<double, remove_vector_t<T>>)
{
    if (SizeRoi() != aDst.SizeRoi() - 1)
    {
        throw ROIEXCEPTION(
            "ROI of destination image must be one pixel larger in width and height than original ROI. Original size: "
            << SizeRoi() << " provided destination image size: " << aDst.SizeRoi());
    }

    using DstT = same_vector_size_different_type_t<T, float>;

    ScratchBuffer<DstT> buffers(aBuffer.Pointer(), BufferSizes{aDst.SizeRoi()});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeIntegralSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.GetSubBufferPitch(0), aDst.Pointer(),
                      aDst.Pitch(), aVal, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, long64>> &ImageView<T>::Integral(
    ImageView<same_vector_size_different_type_t<T, long64>> &aDst,
    const same_vector_size_different_type_t<T, long64> &aVal, mpp::cuda::DevVarView<byte> &aBuffer,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T> && NoAlpha<T>
{
    if (SizeRoi() != aDst.SizeRoi() - 1)
    {
        throw ROIEXCEPTION(
            "ROI of destination image must be one pixel larger in width and height than original ROI. Original size: "
            << SizeRoi() << " provided destination image size: " << aDst.SizeRoi());
    }

    using DstT = same_vector_size_different_type_t<T, long64>;

    const size_t pitchTmp = PadImageWidthToPitch(sizeof(DstT) * to_size_t(aDst.ROI().width));
    const size_t sizeTmp  = pitchTmp * to_size_t(aDst.ROI().height);

    ScratchBuffer<DstT> buffers(aBuffer.Pointer(), BufferSizes{aDst.SizeRoi()});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeIntegralSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.GetSubBufferPitch(0), aDst.Pointer(),
                      aDst.Pitch(), aVal, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, double>> &ImageView<T>::Integral(
    ImageView<same_vector_size_different_type_t<T, double>> &aDst,
    const same_vector_size_different_type_t<T, double> &aVal, mpp::cuda::DevVarView<byte> &aBuffer,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && NoAlpha<T>
{
    if (SizeRoi() != aDst.SizeRoi() - 1)
    {
        throw ROIEXCEPTION(
            "ROI of destination image must be one pixel larger in width and height than original ROI. Original size: "
            << SizeRoi() << " provided destination image size: " << aDst.SizeRoi());
    }

    using DstT = same_vector_size_different_type_t<T, double>;

    const size_t pitchTmp = PadImageWidthToPitch(sizeof(DstT) * to_size_t(aDst.ROI().width));
    const size_t sizeTmp  = pitchTmp * to_size_t(aDst.ROI().height);

    ScratchBuffer<DstT> buffers(aBuffer.Pointer(), BufferSizes{aDst.SizeRoi()});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeIntegralSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.GetSubBufferPitch(0), aDst.Pointer(),
                      aDst.Pitch(), aVal, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::SqrIntegral(ImageView<same_vector_size_different_type_t<T, int>> &aDst,
                               ImageView<same_vector_size_different_type_t<T, int>> &aSqr,
                               const same_vector_size_different_type_t<T, int> &aVal,
                               const same_vector_size_different_type_t<T, int> &aValSqr,
                               mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T> && NoAlpha<T>
{
    if (SizeRoi() != aDst.SizeRoi() - 1)
    {
        throw ROIEXCEPTION(
            "ROI of destination image must be one pixel larger in width and height than original ROI. Original size: "
            << SizeRoi() << " provided destination image size: " << aDst.SizeRoi());
    }
    if (SizeRoi() != aSqr.SizeRoi() - 1)
    {
        throw ROIEXCEPTION("ROI of destination image (squared) must be one pixel larger in width and height than "
                           "original ROI. Original size: "
                           << SizeRoi() << " provided destination image (squared) size: " << aSqr.SizeRoi());
    }

    using DstT    = same_vector_size_different_type_t<T, int>;
    using DstSqrT = same_vector_size_different_type_t<T, int>;

    ScratchBuffer<DstT, DstSqrT> buffers(aBuffer.Pointer(), BufferSizes{aDst.SizeRoi(), aSqr.SizeRoi()});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeIntegralSqrSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.GetSubBufferPitch(0),
                         buffers.template Get<1>(), buffers.GetSubBufferPitch(1), aDst.Pointer(), aDst.Pitch(),
                         aSqr.Pointer(), aSqr.Pitch(), aVal, aValSqr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::SqrIntegral(ImageView<same_vector_size_different_type_t<T, int>> &aDst,
                               ImageView<same_vector_size_different_type_t<T, long64>> &aSqr,
                               const same_vector_size_different_type_t<T, int> &aVal,
                               const same_vector_size_different_type_t<T, long64> &aValSqr,
                               mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealIntVector<T> && NoAlpha<T>
{
    if (SizeRoi() != aDst.SizeRoi() - 1)
    {
        throw ROIEXCEPTION(
            "ROI of destination image must be one pixel larger in width and height than original ROI. Original size: "
            << SizeRoi() << " provided destination image size: " << aDst.SizeRoi());
    }
    if (SizeRoi() != aSqr.SizeRoi() - 1)
    {
        throw ROIEXCEPTION("ROI of destination image (squared) must be one pixel larger in width and height than "
                           "original ROI. Original size: "
                           << SizeRoi() << " provided destination image (squared) size: " << aSqr.SizeRoi());
    }

    using DstT    = same_vector_size_different_type_t<T, int>;
    using DstSqrT = same_vector_size_different_type_t<T, long64>;

    ScratchBuffer<DstT, DstSqrT> buffers(aBuffer.Pointer(), BufferSizes{aDst.SizeRoi(), aSqr.SizeRoi()});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeIntegralSqrSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.GetSubBufferPitch(0),
                         buffers.template Get<1>(), buffers.GetSubBufferPitch(1), aDst.Pointer(), aDst.Pitch(),
                         aSqr.Pointer(), aSqr.Pitch(), aVal, aValSqr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::SqrIntegral(ImageView<same_vector_size_different_type_t<T, float>> &aDst,
                               ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                               const same_vector_size_different_type_t<T, float> &aVal,
                               const same_vector_size_different_type_t<T, double> &aValSqr,
                               mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && NoAlpha<T> && (!std::same_as<double, remove_vector_t<T>>)
{
    if (SizeRoi() != aDst.SizeRoi() - 1)
    {
        throw ROIEXCEPTION(
            "ROI of destination image must be one pixel larger in width and height than original ROI. Original size: "
            << SizeRoi() << " provided destination image size: " << aDst.SizeRoi());
    }
    if (SizeRoi() != aSqr.SizeRoi() - 1)
    {
        throw ROIEXCEPTION("ROI of destination image (squared) must be one pixel larger in width and height than "
                           "original ROI. Original size: "
                           << SizeRoi() << " provided destination image (squared) size: " << aSqr.SizeRoi());
    }

    using DstT    = same_vector_size_different_type_t<T, float>;
    using DstSqrT = same_vector_size_different_type_t<T, double>;

    ScratchBuffer<DstT, DstSqrT> buffers(aBuffer.Pointer(), BufferSizes{aDst.SizeRoi(), aSqr.SizeRoi()});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeIntegralSqrSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.GetSubBufferPitch(0),
                         buffers.template Get<1>(), buffers.GetSubBufferPitch(1), aDst.Pointer(), aDst.Pitch(),
                         aSqr.Pointer(), aSqr.Pitch(), aVal, aValSqr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::SqrIntegral(ImageView<same_vector_size_different_type_t<T, double>> &aDst,
                               ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                               const same_vector_size_different_type_t<T, double> &aVal,
                               const same_vector_size_different_type_t<T, double> &aValSqr,
                               mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T> && NoAlpha<T>
{
    if (SizeRoi() != aDst.SizeRoi() - 1)
    {
        throw ROIEXCEPTION(
            "ROI of destination image must be one pixel larger in width and height than original ROI. Original size: "
            << SizeRoi() << " provided destination image size: " << aDst.SizeRoi());
    }
    if (SizeRoi() != aSqr.SizeRoi() - 1)
    {
        throw ROIEXCEPTION("ROI of destination image (squared) must be one pixel larger in width and height than "
                           "original ROI. Original size: "
                           << SizeRoi() << " provided destination image (squared) size: " << aSqr.SizeRoi());
    }

    using DstT    = same_vector_size_different_type_t<T, double>;
    using DstSqrT = same_vector_size_different_type_t<T, double>;

    ScratchBuffer<DstT, DstSqrT> buffers(aBuffer.Pointer(), BufferSizes{aDst.SizeRoi(), aSqr.SizeRoi()});

    CHECK_BUFFER_SIZE(buffers, aBuffer.SizeInBytes());

    InvokeIntegralSqrSrc(PointerRoi(), Pitch(), buffers.template Get<0>(), buffers.GetSubBufferPitch(0),
                         buffers.template Get<1>(), buffers.GetSubBufferPitch(1), aDst.Pointer(), aDst.Pitch(),
                         aSqr.Pointer(), aSqr.Pitch(), aVal, aValSqr, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::RectStdDev(ImageView<same_vector_size_different_type_t<T, int>> &aSqr,
                              ImageView<same_vector_size_different_type_t<T, float>> &aDst,
                              const FilterArea &aFilterArea, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<remove_vector_t<T>, int>) && NoAlpha<T>
{
    using ComputeT = same_vector_size_different_type_t<T, double>;
    using Src2T    = same_vector_size_different_type_t<T, int>;
    using DstT     = same_vector_size_different_type_t<T, float>;

    checkSameSize(ROI(), aSqr.ROI());
    // aDst roi may differ

    InvokeRectStdDev<T, Src2T, ComputeT, DstT>(PointerRoi(), Pitch(), aSqr.PointerRoi(), aSqr.Pitch(),
                                               aDst.PointerRoi(), aDst.Pitch(), aFilterArea, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::RectStdDev(ImageView<same_vector_size_different_type_t<T, long64>> &aSqr,
                              ImageView<same_vector_size_different_type_t<T, float>> &aDst,
                              const FilterArea &aFilterArea, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<remove_vector_t<T>, int>) && NoAlpha<T>
{
    using ComputeT = same_vector_size_different_type_t<T, double>;
    using Src2T    = same_vector_size_different_type_t<T, long64>;
    using DstT     = same_vector_size_different_type_t<T, float>;

    checkSameSize(ROI(), aSqr.ROI());
    // aDst roi may differ

    InvokeRectStdDev<T, Src2T, ComputeT, DstT>(PointerRoi(), Pitch(), aSqr.PointerRoi(), aSqr.Pitch(),
                                               aDst.PointerRoi(), aDst.Pitch(), aFilterArea, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::RectStdDev(ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                              ImageView<same_vector_size_different_type_t<T, float>> &aDst,
                              const FilterArea &aFilterArea, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<remove_vector_t<T>, float>) && NoAlpha<T>
{
    using ComputeT = same_vector_size_different_type_t<T, double>;
    using Src2T    = same_vector_size_different_type_t<T, double>;
    using DstT     = same_vector_size_different_type_t<T, float>;

    checkSameSize(ROI(), aSqr.ROI());
    // aDst roi may differ

    InvokeRectStdDev<T, Src2T, ComputeT, DstT>(PointerRoi(), Pitch(), aSqr.PointerRoi(), aSqr.Pitch(),
                                               aDst.PointerRoi(), aDst.Pitch(), aFilterArea, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::RectStdDev(ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                              ImageView<same_vector_size_different_type_t<T, double>> &aDst,
                              const FilterArea &aFilterArea, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<remove_vector_t<T>, double>) && NoAlpha<T>
{
    using ComputeT = same_vector_size_different_type_t<T, double>;
    using Src2T    = same_vector_size_different_type_t<T, double>;
    using DstT     = same_vector_size_different_type_t<T, double>;

    checkSameSize(ROI(), aSqr.ROI());
    // aDst roi may differ

    InvokeRectStdDev<T, Src2T, ComputeT, DstT>(PointerRoi(), Pitch(), aSqr.PointerRoi(), aSqr.Pitch(),
                                               aDst.PointerRoi(), aDst.Pitch(), aFilterArea, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region MinEvery
template <PixelType T>
ImageView<T> &ImageView<T>::MinEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeMinEverySrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MinEvery(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeMinEveryInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion

#pragma region MaxEvery
template <PixelType T>
ImageView<T> &ImageView<T>::MaxEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeMaxEverySrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MaxEvery(const ImageView<T> &aSrc2, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeMaxEveryInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);
    return *this;
}
#pragma endregion

#pragma region Histogram
template <PixelType T>
void ImageView<T>::EvenLevels(int *aHPtrLevels, int aLevels, int aLowerLevel, int aUpperLevel,
                              HistorgamEvenMode aHistorgamEvenMode)
{
    if (aLevels < 2)
    {
        throw INVALIDARGUMENT(aLevels, "aLevels must be at least 2 but provided value is: " << aLevels);
    }
    if (aHPtrLevels == nullptr)
    {
        throw INVALIDARGUMENT(aHPtrLevels, "nullptr");
    }
    if (aLowerLevel >= aUpperLevel)
    {
        throw INVALIDARGUMENT(aLowerLevel, "aLowerLevel must be smaller than aUpperLevel, but aLowerLevel = "
                                               << aLowerLevel << " and aUpperLevel = " << aUpperLevel);
    }

    const double step =
        (static_cast<double>(aUpperLevel) - static_cast<double>(aLowerLevel)) / (static_cast<double>(aLevels) - 1.0);

    if (aHistorgamEvenMode == HistorgamEvenMode::NPP)
    {
        const int smallStep = static_cast<int>(std::floor(step));
        const int bigStep   = static_cast<int>(std::ceil(step));

        const int bigStepCount =
            static_cast<int>(std::round((step - smallStep) * (static_cast<double>(aLevels) - 1.0)));

        aHPtrLevels[0] = aLowerLevel;
        for (int i = 1; i <= bigStepCount; i++)
        {
            aHPtrLevels[i] = aHPtrLevels[i - 1] + bigStep;
        }
        for (int i = bigStepCount + 1; i < aLevels; i++)
        {
            aHPtrLevels[i] = aHPtrLevels[i - 1] + smallStep;
        }
        return;
    }

    // default mode as in CUB:
    for (int i = 0; i < aLevels; i++)
    {
        const double bin = step * static_cast<double>(i);
        aHPtrLevels[i]   = static_cast<int>(std::ceil(bin)) + aLowerLevel;
    }
}

template <PixelType T>
size_t ImageView<T>::HistogramEvenBufferSize(const same_vector_size_different_type_t<T, int> &aLevels,
                                             const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return InvokeHistogramEvenGetBufferSize(PointerRoi(), Pitch(), aLevels.data(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::HistogramEven(mpp::cuda::DevVarView<int> aHist[vector_active_size_v<T>],
                                 const hist_even_level_types_for_t<T> &aLowerLevel,
                                 const hist_even_level_types_for_t<T> &aUpperLevel,
                                 mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    int *histPtr[vector_active_size_v<T>];
    int histLevels[vector_active_size_v<T>];
    histPtr[0]    = aHist[0].Pointer();
    histLevels[0] = to_int(aHist[0].Size()) + 1; //+1 for bins to levels

    histPtr[1]    = aHist[1].Pointer();
    histLevels[1] = to_int(aHist[1].Size()) + 1;

    if constexpr (vector_active_size_v<T> > 2)
    {
        histPtr[2]    = aHist[2].Pointer();
        histLevels[2] = to_int(aHist[2].Size()) + 1;
    }
    if constexpr (vector_active_size_v<T> > 3)
    {
        histPtr[3]    = aHist[3].Pointer();
        histLevels[3] = to_int(aHist[3].Size()) + 1;
    }

    return InvokeHistogramEven(PointerRoi(), Pitch(), aBuffer.Pointer(), aBuffer.SizeInBytes(), histPtr, histLevels,
                               aLowerLevel, aUpperLevel, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::HistogramEven(mpp::cuda::DevVarView<int> &aHist, const hist_even_level_types_for_t<T> &aLowerLevel,
                                 const hist_even_level_types_for_t<T> &aUpperLevel,
                                 mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    int *histPtr[vector_active_size_v<T>];
    int histLevels[vector_active_size_v<T>];
    histPtr[0]    = aHist.Pointer();
    histLevels[0] = to_int(aHist.Size()) + 1; //+1 for bins to levels

    return InvokeHistogramEven(PointerRoi(), Pitch(), aBuffer.Pointer(), aBuffer.SizeInBytes(), histPtr, histLevels,
                               aLowerLevel, aUpperLevel, SizeRoi(), aStreamCtx);
}

template <PixelType T>
size_t ImageView<T>::HistogramRangeBufferSize(const same_vector_size_different_type_t<T, int> &aNumLevels,
                                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealVector<T>
{
    return InvokeHistogramRangeGetBufferSize(PointerRoi(), Pitch(), aNumLevels.data(), SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::HistogramRange(mpp::cuda::DevVarView<int> aHist[vector_active_size_v<T>],
                                  mpp::cuda::DevVarView<hist_range_types_for_t<T>> aLevels[vector_active_size_v<T>],
                                  mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    for (size_t i = 0; i < vector_active_size_v<T>; i++)
    {
        if (aHist[i].Size() + 1 != aLevels[i].Size())
        {
            throw INVALIDARGUMENT(
                aLevels, "The number of elements in the levels array must be one more than the number of "
                         "elements in the resulting histogram array. Number of elements in aHist[channel] = "
                             << aHist[i].Size() << " but number of elements in aLevels[channel] = " << aLevels[i].Size()
                             << " for channel = " << i << ".");
        }
    }

    int *histPtr[vector_active_size_v<T>];
    int histLevelCount[vector_active_size_v<T>];
    const hist_range_types_for_t<T> *histLevels[vector_active_size_v<T>];
    histPtr[0]        = aHist[0].Pointer();
    histLevelCount[0] = to_int(aHist[0].Size()) + 1; //+1 for bins to levels
    histLevels[0]     = aLevels[0].Pointer();

    histPtr[1]        = aHist[1].Pointer();
    histLevelCount[1] = to_int(aHist[1].Size()) + 1;
    histLevels[1]     = aLevels[1].Pointer();

    if constexpr (vector_active_size_v<T> > 2)
    {
        histPtr[2]        = aHist[2].Pointer();
        histLevelCount[2] = to_int(aHist[2].Size()) + 1;
        histLevels[2]     = aLevels[2].Pointer();
    }
    if constexpr (vector_active_size_v<T> > 3)
    {
        histPtr[3]        = aHist[3].Pointer();
        histLevelCount[3] = to_int(aHist[3].Size()) + 1;
        histLevels[3]     = aLevels[3].Pointer();
    }

    return InvokeHistogramRange(PointerRoi(), Pitch(), aBuffer.Pointer(), aBuffer.SizeInBytes(), histPtr,
                                histLevelCount, histLevels, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::HistogramRange(mpp::cuda::DevVarView<int> &aHist,
                                  const mpp::cuda::DevVarView<hist_range_types_for_t<T>> &aLevels,
                                  mpp::cuda::DevVarView<byte> &aBuffer, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    if (aHist.Size() + 1 != aLevels.Size())
    {
        throw INVALIDARGUMENT(aLevels, "The number of elements in the levels array must be one more than the number of "
                                       "elements in the resulting histogram array. Number of elements in aHist = "
                                           << aHist.Size() << " but number of elements in aLevels = " << aLevels.Size()
                                           << ".");
    }

    int *histPtr[vector_active_size_v<T>];
    int histLevelCount[vector_active_size_v<T>];
    const hist_range_types_for_t<T> *histLevels[vector_active_size_v<T>];
    histPtr[0]        = aHist.Pointer();
    histLevelCount[0] = to_int(aHist.Size()) + 1; //+1 for bins to levels
    histLevels[0]     = aLevels.Pointer();

    return InvokeHistogramRange(PointerRoi(), Pitch(), aBuffer.Pointer(), aBuffer.SizeInBytes(), histPtr,
                                histLevelCount, histLevels, SizeRoi(), aStreamCtx);
}
#pragma endregion

#pragma region Cross Correlation
template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelation(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                      BorderType aBorder, const Roi &aAllowedReadRoi,
                                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->CrossCorrelation(aTemplate, aDst, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelation(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                      const T &aConstant, BorderType aBorder,
                                                      const Roi &aAllowedReadRoi,
                                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeCrossCorrelation(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTemplate.PointerRoi(),
                           aTemplate.Pitch(), aTemplate.SizeRoi(), aBorder, aConstant, aAllowedReadRoi.Size(),
                           roiOffset, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationNormalized(const ImageView<T> &aTemplate,
                                                                ImageView<Pixel32fC1> &aDst, BorderType aBorder,
                                                                const Roi &aAllowedReadRoi,
                                                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->CrossCorrelationNormalized(aTemplate, aDst, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationNormalized(const ImageView<T> &aTemplate,
                                                                ImageView<Pixel32fC1> &aDst, const T &aConstant,
                                                                BorderType aBorder, const Roi &aAllowedReadRoi,
                                                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeCrossCorrelationNormalized(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTemplate.PointerRoi(),
                                     aTemplate.Pitch(), aTemplate.SizeRoi(), aBorder, aConstant, aAllowedReadRoi.Size(),
                                     roiOffset, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::SquareDistanceNormalized(const ImageView<T> &aTemplate,
                                                              ImageView<Pixel32fC1> &aDst, BorderType aBorder,
                                                              const Roi &aAllowedReadRoi,
                                                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->SquareDistanceNormalized(aTemplate, aDst, {0}, aBorder, aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::SquareDistanceNormalized(const ImageView<T> &aTemplate,
                                                              ImageView<Pixel32fC1> &aDst, const T &aConstant,
                                                              BorderType aBorder, const Roi &aAllowedReadRoi,
                                                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeSquareDistanceNormalized(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTemplate.PointerRoi(),
                                   aTemplate.Pitch(), aTemplate.SizeRoi(), aBorder, aConstant, aAllowedReadRoi.Size(),
                                   roiOffset, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationCoefficient(const ImageView<Pixel32fC2> &aSrcBoxFiltered,
                                                                 const ImageView<T> &aTemplate,
                                                                 const mpp::cuda::DevVarView<Pixel64fC1> &aMeanTemplate,
                                                                 ImageView<Pixel32fC1> &aDst, BorderType aBorder,
                                                                 const Roi &aAllowedReadRoi,
                                                                 const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->CrossCorrelationCoefficient(aSrcBoxFiltered, aTemplate, aMeanTemplate, aDst, {0}, aBorder,
                                             aAllowedReadRoi, aStreamCtx);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationCoefficient(const ImageView<Pixel32fC2> &aSrcBoxFiltered,
                                                                 const ImageView<T> &aTemplate,
                                                                 const mpp::cuda::DevVarView<Pixel64fC1> &aMeanTemplate,
                                                                 ImageView<Pixel32fC1> &aDst, const T &aConstant,
                                                                 BorderType aBorder, const Roi &aAllowedReadRoi,
                                                                 const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeCrossCorrelationCoefficient(allowedPtr, Pitch(), aSrcBoxFiltered.PointerRoi(), aSrcBoxFiltered.Pitch(),
                                      aDst.PointerRoi(), aDst.Pitch(), aTemplate.PointerRoi(), aTemplate.Pitch(),
                                      aTemplate.SizeRoi(), aMeanTemplate.Pointer(), aBorder, aConstant,
                                      aAllowedReadRoi.Size(), roiOffset, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelation(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                      BorderType aBorder, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->CrossCorrelation(aTemplate, aDst, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelation(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                      const T &aConstant, BorderType aBorder,
                                                      const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->CrossCorrelation(aTemplate, aDst, aConstant, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationNormalized(const ImageView<T> &aTemplate,
                                                                ImageView<Pixel32fC1> &aDst, BorderType aBorder,

                                                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->CrossCorrelationNormalized(aTemplate, aDst, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationNormalized(const ImageView<T> &aTemplate,
                                                                ImageView<Pixel32fC1> &aDst, const T &aConstant,
                                                                BorderType aBorder,

                                                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->CrossCorrelationNormalized(aTemplate, aDst, aConstant, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::SquareDistanceNormalized(const ImageView<T> &aTemplate,
                                                              ImageView<Pixel32fC1> &aDst, BorderType aBorder,

                                                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->SquareDistanceNormalized(aTemplate, aDst, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::SquareDistanceNormalized(const ImageView<T> &aTemplate,
                                                              ImageView<Pixel32fC1> &aDst, const T &aConstant,
                                                              BorderType aBorder,
                                                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->SquareDistanceNormalized(aTemplate, aDst, aConstant, aBorder, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationCoefficient(const ImageView<Pixel32fC2> &aSrcBoxFiltered,
                                                                 const ImageView<T> &aTemplate,
                                                                 const mpp::cuda::DevVarView<Pixel64fC1> &aMeanTemplate,
                                                                 ImageView<Pixel32fC1> &aDst, BorderType aBorder,

                                                                 const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->CrossCorrelationCoefficient(aSrcBoxFiltered, aTemplate, aMeanTemplate, aDst, aBorder, ROI(),
                                             aStreamCtx);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationCoefficient(const ImageView<Pixel32fC2> &aSrcBoxFiltered,
                                                                 const ImageView<T> &aTemplate,
                                                                 const mpp::cuda::DevVarView<Pixel64fC1> &aMeanTemplate,
                                                                 ImageView<Pixel32fC1> &aDst, const T &aConstant,
                                                                 BorderType aBorder,
                                                                 const mpp::cuda::StreamCtx &aStreamCtx) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->CrossCorrelationCoefficient(aSrcBoxFiltered, aTemplate, aMeanTemplate, aDst, aConstant, aBorder, ROI(),
                                             aStreamCtx);
}
#pragma endregion
} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND