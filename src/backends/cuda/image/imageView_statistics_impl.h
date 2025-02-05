#pragma once
#include <common/moduleEnabler.h> //NOLINT(misc-include-cleaner)
#if OPP_ENABLE_CUDA_BACKEND

#include "imageView.h"
#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/statistics/statisticsKernel.h>
#include <backends/cuda/streamCtx.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/opp_defs.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace opp::image::cuda
{
#pragma region MinEvery
template <PixelType T>
ImageView<T> &ImageView<T>::MinEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                     const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeMinEverySrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MinEvery(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
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
                                     const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    InvokeMaxEverySrcSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MaxEvery(const ImageView<T> &aSrc2, const opp::cuda::StreamCtx &aStreamCtx)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    InvokeMaxEveryInplaceSrc(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), SizeRoi(), aStreamCtx);
    return *this;
}
#pragma endregion

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND