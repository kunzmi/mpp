#pragma once
#include "dataExchangeAndInit/conversionRelations.h"
#include "dataExchangeAndInit/scale.h"
#include "dataExchangeAndInit/scaleRelations.h"
#include "imageView.h"
#include <array>
#include <backends/cuda/cudaException.h>
#include <backends/cuda/devVarView.h>
#include <backends/cuda/image/colorConversion/colorConversionKernel.h>
#include <backends/cuda/image/colorConversion/test.h>
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
#include <common/image/size2D.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <concepts>

namespace mpp::image::cuda
{
#pragma region ColorConversion
template <PixelType T>
ImageView<Vector2<remove_vector_t<T>>> &ImageView<T>::RGB2YCbCr422C2(ImageView<Vector2<remove_vector_t<T>>> &aDst,
                                                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeTestSrc<T, Vector3<float>, T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), SizeRoi(), aStreamCtx);

    return aDst;
}

#pragma region HLS
#pragma region RGBtoHLS
template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHLS(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeRGBtoHLSSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::RGBtoHLS(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeRGBtoHLSSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::RGBtoHLS(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeRGBtoHLSSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::RGBtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeRGBtoHLSSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeRGBtoHLSSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeRGBtoHLSSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHLS(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeRGBtoHLSInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion
#pragma region BGRtoHLS
template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHLS(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeBGRtoHLSSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::BGRtoHLS(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeBGRtoHLSSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::BGRtoHLS(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeBGRtoHLSSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::BGRtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeBGRtoHLSSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeBGRtoHLSSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHLS(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeBGRtoHLSSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHLS(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeBGRtoHLSInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region HLStoRGB
template <PixelType T>
ImageView<T> &ImageView<T>::HLStoRGB(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeHLStoRGBSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::HLStoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeHLStoRGBSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::HLStoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeHLStoRGBSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::HLStoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeHLStoRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::HLStoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeHLStoRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HLStoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeHLStoRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HLStoRGB(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeHLStoRGBInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion
#pragma region HLStoBGR
template <PixelType T>
ImageView<T> &ImageView<T>::HLStoBGR(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeHLStoBGRSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::HLStoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeHLStoBGRSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::HLStoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeHLStoBGRSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::HLStoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeHLStoBGRSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::HLStoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeHLStoBGRSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HLStoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeHLStoBGRSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HLStoBGR(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeHLStoBGRInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion
#pragma endregion

#pragma region HSV
#pragma region RGBtoHSV
template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHSV(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeRGBtoHSVSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::RGBtoHSV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeRGBtoHSVSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::RGBtoHSV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeRGBtoHSVSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::RGBtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeRGBtoHSVSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeRGBtoHSVSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeRGBtoHSVSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoHSV(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeRGBtoHSVInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion
#pragma region BGRtoHSV
template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHSV(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeBGRtoHSVSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::BGRtoHSV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeBGRtoHSVSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::BGRtoHSV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeBGRtoHSVSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::BGRtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeBGRtoHSVSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeBGRtoHSVSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHSV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeBGRtoHSVSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoHSV(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeBGRtoHSVInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region HSVtoRGB
template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoRGB(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeHSVtoRGBSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::HSVtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeHSVtoRGBSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::HSVtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeHSVtoRGBSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::HSVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeHSVtoRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeHSVtoRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeHSVtoRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoRGB(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeHSVtoRGBInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion
#pragma region HSVtoBGR
template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoBGR(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeHSVtoBGRSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::HSVtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeHSVtoBGRSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::HSVtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeHSVtoBGRSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::HSVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeHSVtoBGRSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeHSVtoBGRSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeHSVtoBGRSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::HSVtoBGR(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeHSVtoBGRInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion
#pragma endregion

#pragma region Lab
#pragma region RGBtoLab
template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLab(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeRGBtoLabSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::RGBtoLab(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeRGBtoLabSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::RGBtoLab(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeRGBtoLabSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::RGBtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeRGBtoLabSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeRGBtoLabSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeRGBtoLabSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLab(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeRGBtoLabInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion
#pragma region BGRtoLab
template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLab(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeBGRtoLabSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::BGRtoLab(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeBGRtoLabSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::BGRtoLab(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeBGRtoLabSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::BGRtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeBGRtoLabSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeBGRtoLabSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLab(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeBGRtoLabSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLab(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeBGRtoLabInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region LabtoRGB
template <PixelType T>
ImageView<T> &ImageView<T>::LabtoRGB(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeLabtoRGBSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::LabtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeLabtoRGBSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::LabtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeLabtoRGBSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::LabtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeLabtoRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::LabtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeLabtoRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LabtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeLabtoRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LabtoRGB(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeLabtoRGBInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion
#pragma region LabtoBGR
template <PixelType T>
ImageView<T> &ImageView<T>::LabtoBGR(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeLabtoBGRSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::LabtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeLabtoBGRSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::LabtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeLabtoBGRSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::LabtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeLabtoBGRSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::LabtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeLabtoBGRSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LabtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeLabtoBGRSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LabtoBGR(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeLabtoBGRInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion
#pragma endregion

#pragma region LUV
#pragma region RGBtoLUV
template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLUV(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeRGBtoLUVSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::RGBtoLUV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeRGBtoLUVSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::RGBtoLUV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeRGBtoLUVSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::RGBtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeRGBtoLUVSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeRGBtoLUVSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeRGBtoLUVSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RGBtoLUV(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeRGBtoLUVInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion
#pragma region BGRtoLUV
template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLUV(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeBGRtoLUVSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::BGRtoLUV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeBGRtoLUVSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::BGRtoLUV(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeBGRtoLUVSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::BGRtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeBGRtoLUVSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeBGRtoLUVSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLUV(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeBGRtoLUVSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BGRtoLUV(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeBGRtoLUVInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region LUVtoRGB
template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoRGB(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeLUVtoRGBSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::LUVtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeLUVtoRGBSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::LUVtoRGB(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeLUVtoRGBSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::LUVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeLUVtoRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeLUVtoRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeLUVtoRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoRGB(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeLUVtoRGBInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}
#pragma endregion
#pragma region LUVtoBGR
template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoBGR(ImageView<T> &aDst, float aNormalizationFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeLUVtoBGRSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::LUVtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeLUVtoBGRSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::LUVtoBGR(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeLUVtoBGRSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormalizationFactor,
                         SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::LUVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                            const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                            ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormalizationFactor,
                            const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16fC3, T> ||
             std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeLUVtoBGRSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
                         aDst2.PointerRoi(), aDst2.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeLUVtoBGRSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormalizationFactor, aSrc1.SizeRoi(),
                         aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoBGR(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                     float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16fC4, T> ||
             std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeLUVtoBGRSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                         aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                         aNormalizationFactor, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUVtoBGR(float aNormalizationFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> ||
             std::same_as<Pixel32fC3, T> || std::same_as<Pixel32fC4A, T>
{
    InvokeLUVtoBGRInplace<T>(PointerRoi(), Pitch(), aNormalizationFactor, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion
#pragma endregion

#pragma region ColorTwist3x3
template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(ImageView<T> &aDst, const Matrix<float> &aTwist,
                                       const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeColorTwist3x3Src<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeColorTwist3x3Src<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(),
                              aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Matrix<float> &aTwist,
                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeColorTwist3x3Src<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(),
                              aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(),
                              aTwist, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeColorTwist3x3Src<Vector4A<remove_vector_t<T>>>(
        aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
        aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist,
        aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                       const Matrix<float> &aTwist, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeColorTwist3x3Src<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                              aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                       remove_vector_t<T> aAlpha, const Matrix<float> &aTwist,
                                       const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeColorTwist3x3Src<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                              aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aAlpha, aTwist, aSrc1.SizeRoi(),
                              aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                       const Matrix<float> &aTwist, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeColorTwist3x3Src<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                              aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist,
                              aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const Matrix<float> &aTwist, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    InvokeColorTwist3x3Inplace<T>(PointerRoi(), Pitch(), aTwist, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<Vector2<remove_vector_t<T>>> &ImageView<T>::ColorTwistTo422(
    ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma, const Matrix<float> &aTwist,
    ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLumaChroma.ROI());

    InvokeColorTwist3x3Src444to422<T>(PointerRoi(), Pitch(), aDstLumaChroma.PointerRoi(), aDstLumaChroma.Pitch(),
                                      aTwist, SizeRoi(), aChromaSubsamplePos, aSwapLumaChroma, aStreamCtx);

    return aDstLumaChroma;
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi()), aDstChroma.SizeRoi());
    InvokeColorTwist3x3Src444to422<T>(PointerRoi(), Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma.PointerRoi(), aDstChroma.Pitch(), aTwist, SizeRoi(),
                                      aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi()), aDstChroma2.SizeRoi());

    InvokeColorTwist3x3Src444to422<T>(PointerRoi(), Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma1.PointerRoi(), aDstChroma1.Pitch(), aDstChroma2.PointerRoi(),
                                      aDstChroma2.Pitch(), aTwist, SizeRoi(), aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi() / 2), aDstChroma.SizeRoi());
    InvokeColorTwist3x3Src444to420<T>(PointerRoi(), Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma.PointerRoi(), aDstChroma.Pitch(), aTwist, SizeRoi(),
                                      aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi() / 2), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi() / 2), aDstChroma2.SizeRoi());

    InvokeColorTwist3x3Src444to420<T>(PointerRoi(), Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma1.PointerRoi(), aDstChroma1.Pitch(), aDstChroma2.PointerRoi(),
                                      aDstChroma2.Pitch(), aTwist, SizeRoi(), aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 4, HeightRoi()), aDstChroma.SizeRoi());
    InvokeColorTwist3x3Src444to411<T>(PointerRoi(), Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma.PointerRoi(), aDstChroma.Pitch(), aTwist, SizeRoi(),
                                      aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 4, HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(WidthRoi() / 4, HeightRoi()), aDstChroma2.SizeRoi());

    InvokeColorTwist3x3Src444to411<T>(PointerRoi(), Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma1.PointerRoi(), aDstChroma1.Pitch(), aDstChroma2.PointerRoi(),
                                      aDstChroma2.Pitch(), aTwist, SizeRoi(), aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma,
                                   const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLumaChroma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLumaChroma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLumaChroma.ROI());

    InvokeColorTwist3x3Src444to422<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLumaChroma.PointerRoi(),
                                      aDstLumaChroma.Pitch(), aTwist, aSrc0.SizeRoi(), aChromaSubsamplePos,
                                      aSwapLumaChroma, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi()), aDstChroma.SizeRoi());
    InvokeColorTwist3x3Src444to422<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma.PointerRoi(), aDstChroma.Pitch(), aTwist, aSrc0.SizeRoi(),
                                      aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi()), aDstChroma2.SizeRoi());

    InvokeColorTwist3x3Src444to422<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma1.PointerRoi(), aDstChroma1.Pitch(), aDstChroma2.PointerRoi(),
                                      aDstChroma2.Pitch(), aTwist, aSrc0.SizeRoi(), aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi() / 2), aDstChroma.SizeRoi());
    InvokeColorTwist3x3Src444to420<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma.PointerRoi(), aDstChroma.Pitch(), aTwist, aSrc0.SizeRoi(),
                                      aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi() / 2), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi() / 2), aDstChroma2.SizeRoi());

    InvokeColorTwist3x3Src444to420<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma1.PointerRoi(), aDstChroma1.Pitch(), aDstChroma2.PointerRoi(),
                                      aDstChroma2.Pitch(), aTwist, aSrc0.SizeRoi(), aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 4, aSrc0.HeightRoi()), aDstChroma.SizeRoi());
    InvokeColorTwist3x3Src444to411<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma.PointerRoi(), aDstChroma.Pitch(), aTwist, aSrc0.SizeRoi(),
                                      aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 4, aSrc0.HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 4, aSrc0.HeightRoi()), aDstChroma2.SizeRoi());

    InvokeColorTwist3x3Src444to411<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma1.PointerRoi(), aDstChroma1.Pitch(), aDstChroma2.PointerRoi(),
                                      aDstChroma2.Pitch(), aTwist, aSrc0.SizeRoi(), aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
ImageView<Vector3<remove_vector_t<T>>> &ImageView<T>::ColorTwistFrom422(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                                        const Matrix<float> &aTwist,
                                                                        bool aSwapLumaChroma,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC2, T> || std::same_as<Pixel16uC2, T> || std::same_as<Pixel16sC2, T> ||
             std::same_as<Pixel16fC2, T> || std::same_as<Pixel32fC2, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeColorTwist3x3Src422to444<Vector3<remove_vector_t<T>>>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                                                                aTwist, SizeRoi(), aSwapLumaChroma, aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     bool aSwapLumaChroma, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC2, T> || std::same_as<Pixel16uC2, T> || std::same_as<Pixel16sC2, T> ||
             std::same_as<Pixel16fC2, T> || std::same_as<Pixel32fC2, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeColorTwist3x3Src422to444<Vector4A<remove_vector_t<T>>>(
        PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(),
        aDst2.Pitch(), aTwist, SizeRoi(), aSwapLumaChroma, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                              const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi()));

    InvokeColorTwist3x3Src422to444<T>(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                      aSrcChroma.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrcLuma.SizeRoi(),
                                      aChromaSubsamplePos, aInterpolationMode, aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                              const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi()));

    InvokeColorTwist3x3Src422to444<T>(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                      aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                      aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrcLuma.SizeRoi(), aChromaSubsamplePos,
                                      aInterpolationMode, aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi()));

    InvokeColorTwist3x3Src422to444<Vector4A<remove_vector_t<T>>>(
        aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(), aSrcChroma.Pitch(), aDst0.PointerRoi(),
        aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, aSrcLuma.SizeRoi(),
        aChromaSubsamplePos, aInterpolationMode, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi()));

    InvokeColorTwist3x3Src422to444<Vector4A<remove_vector_t<T>>>(
        aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(), aSrcChroma1.Pitch(),
        aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(),
        aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, aSrcLuma.SizeRoi(), aChromaSubsamplePos,
        aInterpolationMode, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                              const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2));

    InvokeColorTwist3x3Src420to444<T>(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                      aSrcChroma.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrcLuma.SizeRoi(),
                                      aChromaSubsamplePos, aInterpolationMode, aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                              const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2));

    InvokeColorTwist3x3Src420to444<T>(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                      aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                      aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrcLuma.SizeRoi(), aChromaSubsamplePos,
                                      aInterpolationMode, aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2));

    InvokeColorTwist3x3Src420to444<Vector4A<remove_vector_t<T>>>(
        aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(), aSrcChroma.Pitch(), aDst0.PointerRoi(),
        aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, aSrcLuma.SizeRoi(),
        aChromaSubsamplePos, aInterpolationMode, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2));

    InvokeColorTwist3x3Src420to444<Vector4A<remove_vector_t<T>>>(
        aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(), aSrcChroma1.Pitch(),
        aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(),
        aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, aSrcLuma.SizeRoi(), aChromaSubsamplePos,
        aInterpolationMode, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                              const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst.WidthRoi() / 4, aDst.HeightRoi()));

    InvokeColorTwist3x3Src411to444<T>(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                      aSrcChroma.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrcLuma.SizeRoi(),
                                      aChromaSubsamplePos, aInterpolationMode, aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                              const Matrix<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst.WidthRoi() / 4, aDst.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst.WidthRoi() / 4, aDst.HeightRoi()));

    InvokeColorTwist3x3Src411to444<T>(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                      aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                      aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrcLuma.SizeRoi(), aChromaSubsamplePos,
                                      aInterpolationMode, aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst0.WidthRoi() / 4, aDst0.HeightRoi()));

    InvokeColorTwist3x3Src411to444<Vector4A<remove_vector_t<T>>>(
        aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(), aSrcChroma.Pitch(), aDst0.PointerRoi(),
        aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, aSrcLuma.SizeRoi(),
        aChromaSubsamplePos, aInterpolationMode, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst0.WidthRoi() / 4, aDst0.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst0.WidthRoi() / 4, aDst0.HeightRoi()));

    InvokeColorTwist3x3Src411to444<Vector4A<remove_vector_t<T>>>(
        aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(), aSrcChroma1.Pitch(),
        aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(),
        aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, aSrcLuma.SizeRoi(), aChromaSubsamplePos,
        aInterpolationMode, aStreamCtx);
}
#pragma endregion

#pragma region ColorTwist3x4
template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(ImageView<T> &aDst, const Matrix3x4<float> &aTwist,
                                       const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeColorTwist3x4Src<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeColorTwist3x4Src<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(),
                              aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Matrix3x4<float> &aTwist,
                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeColorTwist3x4Src<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(),
                              aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(),
                              aTwist, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                              const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeColorTwist3x4Src<Vector4A<remove_vector_t<T>>>(
        aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
        aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist,
        aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                       const Matrix3x4<float> &aTwist, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeColorTwist3x4Src<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                              aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, ImageView<T> &aDst,
                                       remove_vector_t<T> aAlpha, const Matrix3x4<float> &aTwist,
                                       const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst.ROI());

    InvokeColorTwist3x4Src<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                              aSrc2.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aAlpha, aTwist, aSrc1.SizeRoi(),
                              aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                       const Matrix3x4<float> &aTwist, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeColorTwist3x4Src<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                              aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist,
                              aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const Matrix3x4<float> &aTwist, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    InvokeColorTwist3x4Inplace<T>(PointerRoi(), Pitch(), aTwist, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<Vector2<remove_vector_t<T>>> &ImageView<T>::ColorTwistTo422(
    ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma, const Matrix3x4<float> &aTwist,
    ChromaSubsamplePos aChromaSubsamplePos, bool aSwapLumaChroma, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLumaChroma.ROI());

    InvokeColorTwist3x4Src444to422<T>(PointerRoi(), Pitch(), aDstLumaChroma.PointerRoi(), aDstLumaChroma.Pitch(),
                                      aTwist, SizeRoi(), aChromaSubsamplePos, aSwapLumaChroma, aStreamCtx);

    return aDstLumaChroma;
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi()), aDstChroma.SizeRoi());
    InvokeColorTwist3x4Src444to422<T>(PointerRoi(), Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma.PointerRoi(), aDstChroma.Pitch(), aTwist, SizeRoi(),
                                      aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi()), aDstChroma2.SizeRoi());

    InvokeColorTwist3x4Src444to422<T>(PointerRoi(), Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma1.PointerRoi(), aDstChroma1.Pitch(), aDstChroma2.PointerRoi(),
                                      aDstChroma2.Pitch(), aTwist, SizeRoi(), aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi() / 2), aDstChroma.SizeRoi());
    InvokeColorTwist3x4Src444to420<T>(PointerRoi(), Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma.PointerRoi(), aDstChroma.Pitch(), aTwist, SizeRoi(),
                                      aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi() / 2), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(WidthRoi() / 2, HeightRoi() / 2), aDstChroma2.SizeRoi());

    InvokeColorTwist3x4Src444to420<T>(PointerRoi(), Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma1.PointerRoi(), aDstChroma1.Pitch(), aDstChroma2.PointerRoi(),
                                      aDstChroma2.Pitch(), aTwist, SizeRoi(), aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 4, HeightRoi()), aDstChroma.SizeRoi());
    InvokeColorTwist3x4Src444to411<T>(PointerRoi(), Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma.PointerRoi(), aDstChroma.Pitch(), aTwist, SizeRoi(),
                                      aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(WidthRoi() / 4, HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(WidthRoi() / 4, HeightRoi()), aDstChroma2.SizeRoi());

    InvokeColorTwist3x4Src444to411<T>(PointerRoi(), Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma1.PointerRoi(), aDstChroma1.Pitch(), aDstChroma2.PointerRoi(),
                                      aDstChroma2.Pitch(), aTwist, SizeRoi(), aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstLumaChroma,
                                   const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                   bool aSwapLumaChroma, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLumaChroma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLumaChroma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLumaChroma.ROI());

    InvokeColorTwist3x4Src444to422<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLumaChroma.PointerRoi(),
                                      aDstLumaChroma.Pitch(), aTwist, aSrc0.SizeRoi(), aChromaSubsamplePos,
                                      aSwapLumaChroma, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi()), aDstChroma.SizeRoi());
    InvokeColorTwist3x4Src444to422<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma.PointerRoi(), aDstChroma.Pitch(), aTwist, aSrc0.SizeRoi(),
                                      aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo422(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi()), aDstChroma2.SizeRoi());

    InvokeColorTwist3x4Src444to422<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma1.PointerRoi(), aDstChroma1.Pitch(), aDstChroma2.PointerRoi(),
                                      aDstChroma2.Pitch(), aTwist, aSrc0.SizeRoi(), aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi() / 2), aDstChroma.SizeRoi());
    InvokeColorTwist3x4Src444to420<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma.PointerRoi(), aDstChroma.Pitch(), aTwist, aSrc0.SizeRoi(),
                                      aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo420(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi() / 2), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 2, aSrc0.HeightRoi() / 2), aDstChroma2.SizeRoi());

    InvokeColorTwist3x4Src444to420<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma1.PointerRoi(), aDstChroma1.Pitch(), aDstChroma2.PointerRoi(),
                                      aDstChroma2.Pitch(), aTwist, aSrc0.SizeRoi(), aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector2<remove_vector_t<T>>> &aDstChroma, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 4, aSrc0.HeightRoi()), aDstChroma.SizeRoi());
    InvokeColorTwist3x4Src444to411<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma.PointerRoi(), aDstChroma.Pitch(), aTwist, aSrc0.SizeRoi(),
                                      aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistTo411(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                   const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstLuma,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma1,
                                   ImageView<Vector1<remove_vector_t<T>>> &aDstChroma2, const Matrix3x4<float> &aTwist,
                                   ChromaSubsamplePos aChromaSubsamplePos, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrc0.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc1.ROI(), aDstLuma.ROI());
    checkSameSize(aSrc2.ROI(), aDstLuma.ROI());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 4, aSrc0.HeightRoi()), aDstChroma1.SizeRoi());
    checkSameSize(Size2D(aSrc0.WidthRoi() / 4, aSrc0.HeightRoi()), aDstChroma2.SizeRoi());

    InvokeColorTwist3x4Src444to411<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                      aSrc2.PointerRoi(), aSrc2.Pitch(), aDstLuma.PointerRoi(), aDstLuma.Pitch(),
                                      aDstChroma1.PointerRoi(), aDstChroma1.Pitch(), aDstChroma2.PointerRoi(),
                                      aDstChroma2.Pitch(), aTwist, aSrc0.SizeRoi(), aChromaSubsamplePos, aStreamCtx);
}

template <PixelType T>
ImageView<Vector3<remove_vector_t<T>>> &ImageView<T>::ColorTwistFrom422(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                                        const Matrix3x4<float> &aTwist,
                                                                        bool aSwapLumaChroma,
                                                                        const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC2, T> || std::same_as<Pixel16uC2, T> || std::same_as<Pixel16sC2, T> ||
             std::same_as<Pixel16fC2, T> || std::same_as<Pixel32fC2, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeColorTwist3x4Src422to444<Vector3<remove_vector_t<T>>>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                                                                aTwist, SizeRoi(), aSwapLumaChroma, aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     bool aSwapLumaChroma, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC2, T> || std::same_as<Pixel16uC2, T> || std::same_as<Pixel16sC2, T> ||
             std::same_as<Pixel16fC2, T> || std::same_as<Pixel32fC2, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());

    InvokeColorTwist3x4Src422to444<Vector4A<remove_vector_t<T>>>(
        PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(),
        aDst2.Pitch(), aTwist, SizeRoi(), aSwapLumaChroma, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                              const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi()));

    InvokeColorTwist3x4Src422to444<T>(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                      aSrcChroma.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrcLuma.SizeRoi(),
                                      aChromaSubsamplePos, aInterpolationMode, aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                              const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi()));

    InvokeColorTwist3x4Src422to444<T>(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                      aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                      aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrcLuma.SizeRoi(), aChromaSubsamplePos,
                                      aInterpolationMode, aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi()));

    InvokeColorTwist3x4Src422to444<Vector4A<remove_vector_t<T>>>(
        aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(), aSrcChroma.Pitch(), aDst0.PointerRoi(),
        aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, aSrcLuma.SizeRoi(),
        aChromaSubsamplePos, aInterpolationMode, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom422(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi()));

    InvokeColorTwist3x4Src422to444<Vector4A<remove_vector_t<T>>>(
        aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(), aSrcChroma1.Pitch(),
        aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(),
        aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, aSrcLuma.SizeRoi(), aChromaSubsamplePos,
        aInterpolationMode, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                              const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2));

    InvokeColorTwist3x4Src420to444<T>(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                      aSrcChroma.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrcLuma.SizeRoi(),
                                      aChromaSubsamplePos, aInterpolationMode, aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                              const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst.WidthRoi() / 2, aDst.HeightRoi() / 2));

    InvokeColorTwist3x4Src420to444<T>(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                      aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                      aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrcLuma.SizeRoi(), aChromaSubsamplePos,
                                      aInterpolationMode, aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2));

    InvokeColorTwist3x4Src420to444<Vector4A<remove_vector_t<T>>>(
        aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(), aSrcChroma.Pitch(), aDst0.PointerRoi(),
        aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, aSrcLuma.SizeRoi(),
        aChromaSubsamplePos, aInterpolationMode, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom420(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst0.WidthRoi() / 2, aDst0.HeightRoi() / 2));

    InvokeColorTwist3x4Src420to444<Vector4A<remove_vector_t<T>>>(
        aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(), aSrcChroma1.Pitch(),
        aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(),
        aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, aSrcLuma.SizeRoi(), aChromaSubsamplePos,
        aInterpolationMode, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma, ImageView<T> &aDst,
                                              const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst.WidthRoi() / 4, aDst.HeightRoi()));

    InvokeColorTwist3x4Src411to444<T>(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(),
                                      aSrcChroma.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrcLuma.SizeRoi(),
                                      aChromaSubsamplePos, aInterpolationMode, aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                              ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2, ImageView<T> &aDst,
                                              const Matrix3x4<float> &aTwist, ChromaSubsamplePos aChromaSubsamplePos,
                                              InterpolationMode aInterpolationMode,
                                              const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel8uC4A, T> || std::same_as<Pixel16uC3, T> ||
             std::same_as<Pixel16uC4A, T> || std::same_as<Pixel16sC3, T> || std::same_as<Pixel16sC4A, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel16fC4A, T> || std::same_as<Pixel32fC3, T> ||
             std::same_as<Pixel32fC4A, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst.WidthRoi() / 4, aDst.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst.WidthRoi() / 4, aDst.HeightRoi()));

    InvokeColorTwist3x4Src411to444<T>(aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(),
                                      aSrcChroma1.Pitch(), aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(),
                                      aDst.PointerRoi(), aDst.Pitch(), aTwist, aSrcLuma.SizeRoi(), aChromaSubsamplePos,
                                      aInterpolationMode, aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector2<remove_vector_t<T>>> &aSrcChroma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma.SizeRoi(), Size2D(aDst0.WidthRoi() / 4, aDst0.HeightRoi()));

    InvokeColorTwist3x4Src411to444<Vector4A<remove_vector_t<T>>>(
        aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma.PointerRoi(), aSrcChroma.Pitch(), aDst0.PointerRoi(),
        aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, aSrcLuma.SizeRoi(),
        aChromaSubsamplePos, aInterpolationMode, aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwistFrom411(ImageView<Vector1<remove_vector_t<T>>> &aSrcLuma,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrcChroma2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, const Matrix3x4<float> &aTwist,
                                     ChromaSubsamplePos aChromaSubsamplePos, InterpolationMode aInterpolationMode,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC3, T> || std::same_as<Pixel16uC3, T> || std::same_as<Pixel16sC3, T> ||
             std::same_as<Pixel16fC3, T> || std::same_as<Pixel32fC3, T>
{
    checkSameSize(aSrcLuma.ROI(), aDst0.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst1.ROI());
    checkSameSize(aSrcLuma.ROI(), aDst2.ROI());
    checkSameSize(aSrcChroma1.SizeRoi(), Size2D(aDst0.WidthRoi() / 4, aDst0.HeightRoi()));
    checkSameSize(aSrcChroma2.SizeRoi(), Size2D(aDst0.WidthRoi() / 4, aDst0.HeightRoi()));

    InvokeColorTwist3x4Src411to444<Vector4A<remove_vector_t<T>>>(
        aSrcLuma.PointerRoi(), aSrcLuma.Pitch(), aSrcChroma1.PointerRoi(), aSrcChroma1.Pitch(),
        aSrcChroma2.PointerRoi(), aSrcChroma2.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(),
        aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aTwist, aSrcLuma.SizeRoi(), aChromaSubsamplePos,
        aInterpolationMode, aStreamCtx);
}
#pragma endregion

#pragma region ColorTwist4x4
template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(ImageView<T> &aDst, const Matrix4x4<float> &aTwist,
                                       const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeColorTwist4x4Src<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Matrix4x4<float> &aTwist,
                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeColorTwist4x4Src<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(),
                              aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(),
                              aTwist, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwist(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
    ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const Matrix4x4<float> &aTwist, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());
    checkSameSize(aSrc0.ROI(), aDst3.ROI());

    InvokeColorTwist4x4Src<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                              aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(),
                              aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(),
                              aDst3.Pitch(), aTwist, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                       const Matrix4x4<float> &aTwist, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeColorTwist4x4Src<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                              aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist,
                              aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const Matrix4x4<float> &aTwist, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    InvokeColorTwist4x4Inplace<T>(PointerRoi(), Pitch(), aTwist, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region ColorTwist4x4C
template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(ImageView<T> &aDst, const Matrix4x4<float> &aTwist, const Pixel32fC4 &aConstant,
                                       const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeColorTwist4x4CSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aTwist, aConstant, SizeRoi(),
                               aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorTwist(ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                              ImageView<Vector1<remove_vector_t<T>>> &aDst3, const Matrix4x4<float> &aTwist,
                              const Pixel32fC4 &aConstant, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(ROI(), aDst0.ROI());
    checkSameSize(ROI(), aDst1.ROI());
    checkSameSize(ROI(), aDst2.ROI());
    checkSameSize(ROI(), aDst3.ROI());

    InvokeColorTwist4x4CSrc<T>(PointerRoi(), Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(),
                               aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(),
                               aTwist, aConstant, SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorTwist(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
    ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
    const Matrix4x4<float> &aTwist, const Pixel32fC4 &aConstant, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());
    checkSameSize(aSrc0.ROI(), aDst3.ROI());

    InvokeColorTwist4x4CSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                               aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(),
                               aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(),
                               aDst3.Pitch(), aTwist, aConstant, aSrc1.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<T> &aDst,
                                       const Matrix4x4<float> &aTwist, const Pixel32fC4 &aConstant,
                                       const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    checkSameSize(aSrc0.ROI(), aDst.ROI());
    checkSameSize(aSrc1.ROI(), aDst.ROI());
    checkSameSize(aSrc2.ROI(), aDst.ROI());
    checkSameSize(aSrc3.ROI(), aDst.ROI());

    InvokeColorTwist4x4CSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                               aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst.PointerRoi(), aDst.Pitch(),
                               aTwist, aConstant, aSrc1.SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColorTwist(const Matrix4x4<float> &aTwist, const Pixel32fC4 &aConstant,
                                       const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<Pixel8uC4, T> || std::same_as<Pixel16uC4, T> || std::same_as<Pixel16sC4, T> ||
             std::same_as<Pixel16fC4, T> || std::same_as<Pixel32fC4, T>
{
    InvokeColorTwist4x4CInplace<T>(PointerRoi(), Pitch(), aTwist, aConstant, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region GammaCorrBT709
template <PixelType T>
ImageView<T> &ImageView<T>::GammaCorrBT709(ImageView<T> &aDst, float aNormFactor,
                                           const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeGammaCorrBT709Src<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormFactor, SizeRoi(),
                               aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::GammaCorrBT709(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1, float aNormFactor,
                                  const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 2)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());

    InvokeGammaCorrBT709Src<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aDst0.PointerRoi(),
                               aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aNormFactor, aSrc0.SizeRoi(),
                               aStreamCtx);
}

template <PixelType T>
void ImageView<T>::GammaCorrBT709(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                  const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                  ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormFactor,
                                  const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeGammaCorrBT709Src<Vector4A<remove_vector_t<T>>>(
        aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
        aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(),
        aNormFactor, aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::GammaCorrBT709(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
    ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormFactor,
    const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 4)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());
    checkSameSize(aSrc0.ROI(), aDst3.ROI());

    InvokeGammaCorrBT709Src<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                               aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(),
                               aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(),
                               aDst3.Pitch(), aNormFactor, aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::GammaCorrBT709(float aNormFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    InvokeGammaCorrBT709Inplace<T>(PointerRoi(), Pitch(), aNormFactor, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region GammaInvCorrBT709
template <PixelType T>
ImageView<T> &ImageView<T>::GammaInvCorrBT709(ImageView<T> &aDst, float aNormFactor,
                                              const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeGammaInvCorrBT709Src<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormFactor, SizeRoi(),
                                  aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::GammaInvCorrBT709(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1, float aNormFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 2)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());

    InvokeGammaInvCorrBT709Src<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                  aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aNormFactor,
                                  aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::GammaInvCorrBT709(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                     const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                     ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormFactor,
                                     const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeGammaInvCorrBT709Src<Vector4A<remove_vector_t<T>>>(
        aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
        aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(),
        aNormFactor, aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::GammaInvCorrBT709(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
    ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormFactor,
    const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 4)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());
    checkSameSize(aSrc0.ROI(), aDst3.ROI());

    InvokeGammaInvCorrBT709Src<T>(
        aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
        aSrc3.PointerRoi(), aSrc3.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
        aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormFactor, aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::GammaInvCorrBT709(float aNormFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    InvokeGammaInvCorrBT709Inplace<T>(PointerRoi(), Pitch(), aNormFactor, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region GammaCorrsRGB
template <PixelType T>
ImageView<T> &ImageView<T>::GammaCorrsRGB(ImageView<T> &aDst, float aNormFactor,
                                          const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeGammaCorrsRGBSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormFactor, SizeRoi(),
                              aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::GammaCorrsRGB(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                 ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst1, float aNormFactor,
                                 const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 2)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());

    InvokeGammaCorrsRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aDst0.PointerRoi(),
                              aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aNormFactor, aSrc0.SizeRoi(),
                              aStreamCtx);
}

template <PixelType T>
void ImageView<T>::GammaCorrsRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                 const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                 const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                 ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormFactor,
                                 const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeGammaCorrsRGBSrc<Vector4A<remove_vector_t<T>>>(
        aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
        aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(),
        aNormFactor, aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::GammaCorrsRGB(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
    ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormFactor,
    const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 4)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());
    checkSameSize(aSrc0.ROI(), aDst3.ROI());

    InvokeGammaCorrsRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(),
                              aSrc2.Pitch(), aSrc3.PointerRoi(), aSrc3.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(),
                              aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(),
                              aDst3.Pitch(), aNormFactor, aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::GammaCorrsRGB(float aNormFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    InvokeGammaCorrsRGBInplace<T>(PointerRoi(), Pitch(), aNormFactor, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region GammaInvCorrsRGB
template <PixelType T>
ImageView<T> &ImageView<T>::GammaInvCorrsRGB(ImageView<T> &aDst, float aNormFactor,
                                             const mpp::cuda::StreamCtx &aStreamCtx) const
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeGammaInvCorrsRGBSrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNormFactor, SizeRoi(),
                                 aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::GammaInvCorrsRGB(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst1, float aNormFactor,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 2)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());

    InvokeGammaInvCorrsRGBSrc<T>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(),
                                 aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aNormFactor,
                                 aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::GammaInvCorrsRGB(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                                    ImageView<Vector1<remove_vector_t<T>>> &aDst2, float aNormFactor,
                                    const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());

    InvokeGammaInvCorrsRGBSrc<Vector4A<remove_vector_t<T>>>(
        aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
        aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(), aDst2.PointerRoi(), aDst2.Pitch(),
        aNormFactor, aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::GammaInvCorrsRGB(
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc0, const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
    const ImageView<Vector1<remove_vector_t<T>>> &aSrc2, const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
    ImageView<Vector1<remove_vector_t<T>>> &aDst0, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
    ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3, float aNormFactor,
    const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 4)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());
    checkSameSize(aSrc0.ROI(), aDst1.ROI());
    checkSameSize(aSrc0.ROI(), aDst2.ROI());
    checkSameSize(aSrc0.ROI(), aDst3.ROI());

    InvokeGammaInvCorrsRGBSrc<T>(
        aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
        aSrc3.PointerRoi(), aSrc3.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aDst1.PointerRoi(), aDst1.Pitch(),
        aDst2.PointerRoi(), aDst2.Pitch(), aDst3.PointerRoi(), aDst3.Pitch(), aNormFactor, aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::GammaInvCorrsRGB(float aNormFactor, const mpp::cuda::StreamCtx &aStreamCtx)
    requires std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>
{
    InvokeGammaInvCorrsRGBInplace<T>(PointerRoi(), Pitch(), aNormFactor, SizeRoi(), aStreamCtx);

    return *this;
}

#pragma endregion

#pragma region GradientColorToGray
template <PixelType T>
ImageView<Vector1<remove_vector_t<T>>> &ImageView<T>::GradientColorToGray(ImageView<Vector1<remove_vector_t<T>>> &aDst,
                                                                          Norm aNorm,
                                                                          const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_size_v<T> > 1)
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeColorGradientToGraySrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aNorm, SizeRoi(),
                                    aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::GradientColorToGray(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst0, Norm aNorm,
                                       const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 2)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());

    InvokeColorGradientToGraySrc<Vector2<remove_vector_t<T>>>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(),
                                                              aSrc1.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aNorm,
                                                              aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::GradientColorToGray(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst0, Norm aNorm,
                                       const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());

    InvokeColorGradientToGraySrc<Vector4A<remove_vector_t<T>>>(
        aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
        aDst0.PointerRoi(), aDst0.Pitch(), aNorm, aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::GradientColorToGray(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                                       const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                                       ImageView<Vector1<remove_vector_t<T>>> &aDst0, Norm aNorm,
                                       const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 4)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());

    InvokeColorGradientToGraySrc<Vector4<remove_vector_t<T>>>(
        aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
        aSrc3.PointerRoi(), aSrc3.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aNorm, aSrc0.SizeRoi(), aStreamCtx);
}

#pragma endregion

#pragma region ColorToGray
template <PixelType T>
ImageView<Vector1<remove_vector_t<T>>> &ImageView<T>::ColorToGray(
    ImageView<Vector1<remove_vector_t<T>>> &aDst, const same_vector_size_different_type_t<T, float> &aWeights,
    const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_size_v<T> > 1)
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeColorToGraySrc<T>(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aWeights, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
void ImageView<T>::ColorToGray(ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                               ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                               const same_vector_size_different_type_t<T, float> &aWeights,
                               const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 2)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());

    InvokeColorToGraySrc<Vector2<remove_vector_t<T>>>(aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(),
                                                      aSrc1.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aWeights,
                                                      aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorToGray(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                               const same_vector_size_different_type_t<T, float> &aWeights,
                               const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());

    InvokeColorToGraySrc<Vector4A<remove_vector_t<T>>>(
        aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
        aDst0.PointerRoi(), aDst0.Pitch(), aWeights, aSrc0.SizeRoi(), aStreamCtx);
}

template <PixelType T>
void ImageView<T>::ColorToGray(const ImageView<Vector1<remove_vector_t<T>>> &aSrc0,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc1,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                               const ImageView<Vector1<remove_vector_t<T>>> &aSrc3,
                               ImageView<Vector1<remove_vector_t<T>>> &aDst0,
                               const same_vector_size_different_type_t<T, float> &aWeights,
                               const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<HalfFp16, remove_vector_t<T>> ||
             std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 4)
{
    checkSameSize(aSrc0.ROI(), aSrc1.ROI());
    checkSameSize(aSrc0.ROI(), aSrc2.ROI());
    checkSameSize(aSrc0.ROI(), aSrc3.ROI());
    checkSameSize(aSrc0.ROI(), aDst0.ROI());

    InvokeColorToGraySrc<Vector4<remove_vector_t<T>>>(
        aSrc0.PointerRoi(), aSrc0.Pitch(), aSrc1.PointerRoi(), aSrc1.Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(),
        aSrc3.PointerRoi(), aSrc3.Pitch(), aDst0.PointerRoi(), aDst0.Pitch(), aWeights, aSrc0.SizeRoi(), aStreamCtx);
}

#pragma endregion

#pragma region CFAToRGB
template <PixelType T>
ImageView<Vector3<remove_vector_t<T>>> &ImageView<T>::CFAToRGB(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                               BayerGridPosition aBayerGrid, const Roi &aAllowedReadRoi,
                                                               const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16uC1, T> || std::same_as<Pixel32uC1, T> ||
             std::same_as<Pixel16sC1, T> || std::same_as<Pixel32sC1, T> || std::same_as<Pixel16bfC1, T> ||
             std::same_as<Pixel16fC1, T> || std::same_as<Pixel32fC1, T>)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));
    checkSameSize(ROI(), aDst.ROI());
    if (WidthRoi() % 2 != 0 || HeightRoi() % 2 != 0)
    {
        throw INVALIDARGUMENT(ROI, "The image ROI must have even width and height, but is: " << SizeRoi());
    }

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeCfaToRgbSrc(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aBayerGrid, roiOffset,
                      aAllowedReadRoi.Size(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Vector4<remove_vector_t<T>>> &ImageView<T>::CFAToRGB(ImageView<Vector4<remove_vector_t<T>>> &aDst,
                                                               remove_vector_t<T> aAlpha, BayerGridPosition aBayerGrid,
                                                               const Roi &aAllowedReadRoi,
                                                               const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16uC1, T> || std::same_as<Pixel32uC1, T> ||
             std::same_as<Pixel16sC1, T> || std::same_as<Pixel32sC1, T> || std::same_as<Pixel16bfC1, T> ||
             std::same_as<Pixel16fC1, T> || std::same_as<Pixel32fC1, T>)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));
    checkSameSize(ROI(), aDst.ROI());
    if (WidthRoi() % 2 != 0 || HeightRoi() % 2 != 0)
    {
        throw INVALIDARGUMENT(ROI, "The image ROI must have even width and height, but is: " << SizeRoi());
    }

    const Vector2<int> roiOffset = ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const T *allowedPtr          = gotoPtr(Pointer(), Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    InvokeCfaToRgbSrc(allowedPtr, Pitch(), aDst.PointerRoi(), aDst.Pitch(), aAlpha, aBayerGrid, roiOffset,
                      aAllowedReadRoi.Size(), SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Vector3<remove_vector_t<T>>> &ImageView<T>::CFAToRGB(ImageView<Vector3<remove_vector_t<T>>> &aDst,
                                                               BayerGridPosition aBayerGrid,
                                                               const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16uC1, T> || std::same_as<Pixel32uC1, T> ||
             std::same_as<Pixel16sC1, T> || std::same_as<Pixel32sC1, T> || std::same_as<Pixel16bfC1, T> ||
             std::same_as<Pixel16fC1, T> || std::same_as<Pixel32fC1, T>)
{
    return this->CFAToRGB(aDst, aBayerGrid, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<Vector4<remove_vector_t<T>>> &ImageView<T>::CFAToRGB(ImageView<Vector4<remove_vector_t<T>>> &aDst,
                                                               remove_vector_t<T> aAlpha, BayerGridPosition aBayerGrid,
                                                               const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16uC1, T> || std::same_as<Pixel32uC1, T> ||
             std::same_as<Pixel16sC1, T> || std::same_as<Pixel32sC1, T> || std::same_as<Pixel16bfC1, T> ||
             std::same_as<Pixel16fC1, T> || std::same_as<Pixel32fC1, T>)
{
    return this->CFAToRGB(aDst, aAlpha, aBayerGrid, ROI(), aStreamCtx);
}

template <PixelType T>
ImageView<Vector1<remove_vector_t<T>>> &ImageView<T>::RGBToCFA(ImageView<Vector1<remove_vector_t<T>>> &aDst,
                                                               BayerGridPosition aBayerGrid,
                                                               const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<int, remove_vector_t<T>> ||
             std::same_as<uint, remove_vector_t<T>> || std::same_as<BFloat16, remove_vector_t<T>> ||
             std::same_as<HalfFp16, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> == 3)
{
    checkSameSize(ROI(), aDst.ROI());
    if (WidthRoi() % 2 != 0 || HeightRoi() % 2 != 0)
    {
        throw INVALIDARGUMENT(ROI, "The image ROI must have even width and height, but is: " << SizeRoi());
    }

    InvokeRgbToCfaSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aBayerGrid, {0, 0}, SizeRoi(), SizeRoi(),
                      aStreamCtx);

    return aDst;
}
#pragma endregion

#pragma region LUTPalette
template <PixelType T>
void ImageView<T>::LUTToPalette(const int *aLevels, const int *aValues, int aLUTSize,
                                Vector1<remove_vector_t<T>> *aPalette, InterpolationMode aInterpolationMode)
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
             std::same_as<remove_vector_t<T>, ushort>)
{
    switch (aInterpolationMode)
    {
        case mpp::InterpolationMode::NearestNeighbor:
            LUTtoPalette(aLevels, aValues, aLUTSize, reinterpret_cast<remove_vector_t<T> *>(aPalette));
            break;
        case mpp::InterpolationMode::Linear:
            LUTtoPaletteLinear(aLevels, aValues, aLUTSize, reinterpret_cast<remove_vector_t<T> *>(aPalette));
            break;
        case mpp::InterpolationMode::CubicLagrange:
            LUTtoPaletteCubic(aLevels, aValues, aLUTSize, reinterpret_cast<remove_vector_t<T> *>(aPalette));
            break;
        default:
            throw INVALIDARGUMENT(
                aInterpolationMode,
                "Unsupported interpolation mode: "
                    << aInterpolationMode
                    << ". Only NearestNeighbor, Linear and CubicLagrange interpolation modes are supported.");
            break;
    }
}

template <PixelType T>
void ImageView<T>::LUTToPalette(const mpp::cuda::DevVarView<int> &aLevels, const mpp::cuda::DevVarView<int> &aValues,
                                mpp::cuda::DevVarView<Vector1<remove_vector_t<T>>> &aPalette,
                                InterpolationMode aInterpolationMode, const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
             std::same_as<remove_vector_t<T>, ushort>)
{
    if (aLevels.Size() != aValues.Size())
    {
        throw INVALIDARGUMENT(aLevels aValues, "aLevels and aValues must have the same size, but sizes are: "
                                                   << aLevels.Size() << " and " << aValues.Size());
    }

    const int lutSize = static_cast<int>(aLevels.Size());

    InvokeLutToPaletteKernelDefault(aLevels.Pointer(), aValues.Pointer(), lutSize,
                                    reinterpret_cast<remove_vector_t<T> *>(aPalette.Pointer()), aInterpolationMode,
                                    aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTPalette(ImageView<T> &aDst, const mpp::cuda::DevVarView<T> &aPalette, int aBitSize,
                                       const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
             std::same_as<remove_vector_t<T>, ushort>) &&
            (vector_size_v<T> == 1)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    InvokeLutPaletteSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aPalette.Pointer(), aBitSize, SizeRoi(),
                        aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTPalette(const mpp::cuda::DevVarView<T> &aPalette, int aBitSize,
                                       const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>)
{
    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    InvokeLutPaletteInplace(PointerRoi(), Pitch(), aPalette.Pointer(), aBitSize, SizeRoi(), aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<Vector3<remove_vector_t<T>>> &ImageView<T>::LUTPalette(
    ImageView<Vector3<remove_vector_t<T>>> &aDst, const mpp::cuda::DevVarView<Vector3<remove_vector_t<T>>> &aPalette,
    int aBitSize, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    InvokeLutPaletteSrc33(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aPalette.Pointer(), aBitSize,
                          SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Vector3<remove_vector_t<T>>> &ImageView<T>::LUTPalette(
    ImageView<Vector3<remove_vector_t<T>>> &aDst, const mpp::cuda::DevVarView<Vector4A<remove_vector_t<T>>> &aPalette,
    int aBitSize, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    InvokeLutPaletteSrc34A(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aPalette.Pointer(), aBitSize,
                           SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Vector4A<remove_vector_t<T>>> &ImageView<T>::LUTPalette(
    ImageView<Vector4A<remove_vector_t<T>>> &aDst, const mpp::cuda::DevVarView<Vector3<remove_vector_t<T>>> &aPalette,
    int aBitSize, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    InvokeLutPaletteSrc4A3(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aPalette.Pointer(), aBitSize,
                           SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Vector4A<remove_vector_t<T>>> &ImageView<T>::LUTPalette(
    ImageView<Vector4A<remove_vector_t<T>>> &aDst, const mpp::cuda::DevVarView<Vector4A<remove_vector_t<T>>> &aPalette,
    int aBitSize, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    InvokeLutPaletteSrc4A4A(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aPalette.Pointer(), aBitSize,
                            SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Vector4<remove_vector_t<T>>> &ImageView<T>::LUTPalette(
    ImageView<Vector4<remove_vector_t<T>>> &aDst, const mpp::cuda::DevVarView<Vector4<remove_vector_t<T>>> &aPalette,
    int aBitSize, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel8uC1, T> || std::same_as<Pixel16sC1, T> || std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    InvokeLutPaletteSrc44(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aPalette.Pointer(), aBitSize,
                          SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTPalette(
    ImageView<T> &aDst, const mpp::cuda::DevVarView<Vector1<remove_vector_t<T>>> aPalette[vector_active_size_v<T>],
    int aBitSize, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
             std::same_as<remove_vector_t<T>, ushort>) &&
            (vector_active_size_v<T> >= 2)
{
    checkSameSize(ROI(), aDst.ROI());

    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    if constexpr (vector_active_size_v<T> == 2)
    {
        const std::array<const Vector1<remove_vector_t<T>> *, vector_active_size_v<T>> palette = {
            aPalette[0].Pointer(), aPalette[1].Pointer()};

        InvokeLutPaletteSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), palette.data(), aBitSize, SizeRoi(),
                            aStreamCtx);
    }
    else if constexpr (vector_active_size_v<T> == 3)
    {
        const std::array<const Vector1<remove_vector_t<T>> *, vector_active_size_v<T>> palette = {
            aPalette[0].Pointer(), aPalette[1].Pointer(), aPalette[2].Pointer()};

        InvokeLutPaletteSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), palette.data(), aBitSize, SizeRoi(),
                            aStreamCtx);
    }
    else
    {
        const std::array<const Vector1<remove_vector_t<T>> *, vector_active_size_v<T>> palette = {
            aPalette[0].Pointer(), aPalette[1].Pointer(), aPalette[2].Pointer(), aPalette[3].Pointer()};

        InvokeLutPaletteSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), palette.data(), aBitSize, SizeRoi(),
                            aStreamCtx);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTPalette(
    const mpp::cuda::DevVarView<Vector1<remove_vector_t<T>>> aPalette[vector_active_size_v<T>], int aBitSize,
    const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, short> ||
             std::same_as<remove_vector_t<T>, ushort>) &&
            (vector_active_size_v<T> >= 2)
{
    if constexpr (std::same_as<Pixel16sC1, T>)
    {
        if (aBitSize != 16)
        {
            throw INVALIDARGUMENT(
                aBitSize,
                "For images of type Pixel16sC1, only a value of 16 is supported for aBitSize. Provided value is: "
                    << aBitSize);
        }
    }
    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    if constexpr (vector_active_size_v<T> == 2)
    {
        const std::array<const Vector1<remove_vector_t<T>> *, vector_active_size_v<T>> palette = {
            aPalette[0].Pointer(), aPalette[1].Pointer()};

        InvokeLutPaletteInplace(PointerRoi(), Pitch(), palette.data(), aBitSize, SizeRoi(), aStreamCtx);
    }
    else if constexpr (vector_active_size_v<T> == 3)
    {
        const std::array<const Vector1<remove_vector_t<T>> *, vector_active_size_v<T>> palette = {
            aPalette[0].Pointer(), aPalette[1].Pointer(), aPalette[2].Pointer()};

        InvokeLutPaletteInplace(PointerRoi(), Pitch(), palette.data(), aBitSize, SizeRoi(), aStreamCtx);
    }
    else
    {
        const std::array<const Vector1<remove_vector_t<T>> *, vector_active_size_v<T>> palette = {
            aPalette[0].Pointer(), aPalette[1].Pointer(), aPalette[2].Pointer(), aPalette[3].Pointer()};

        InvokeLutPaletteInplace(PointerRoi(), Pitch(), palette.data(), aBitSize, SizeRoi(), aStreamCtx);
    }

    return *this;
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::LUTPalette(ImageView<Pixel8uC1> &aDst,
                                               const mpp::cuda::DevVarView<Pixel8uC1> &aPalette, int aBitSize,
                                               const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    InvokeLutPaletteSrc16u(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aPalette.Pointer(), aBitSize,
                           SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC3> &ImageView<T>::LUTPalette(ImageView<Pixel8uC3> &aDst,
                                               const mpp::cuda::DevVarView<Pixel8uC3> &aPalette, int aBitSize,
                                               const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    InvokeLutPaletteSrc16u(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aPalette.Pointer(), aBitSize,
                           SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC4> &ImageView<T>::LUTPalette(ImageView<Pixel8uC4> &aDst,
                                               const mpp::cuda::DevVarView<Pixel8uC4> &aPalette, int aBitSize,
                                               const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    InvokeLutPaletteSrc16u(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aPalette.Pointer(), aBitSize,
                           SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<Pixel8uC4A> &ImageView<T>::LUTPalette(ImageView<Pixel8uC4A> &aDst,
                                                const mpp::cuda::DevVarView<Pixel8uC4A> &aPalette, int aBitSize,
                                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<Pixel16uC1, T>)
{
    checkSameSize(ROI(), aDst.ROI());

    if (aBitSize < 1 || aBitSize > static_cast<int>(8 * sizeof(remove_vector_t<T>)))
    {
        throw INVALIDARGUMENT(aBitSize, "aBitSize is out of range. Value must be in range [1.."
                                            << 8 * sizeof(remove_vector_t<T>) << "] but is: " << aBitSize);
    }

    InvokeLutPaletteSrc16u(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aPalette.Pointer(), aBitSize,
                           SizeRoi(), aStreamCtx);

    return aDst;
}

#pragma endregion
#pragma region LUT
template <PixelType T>
void ImageView<T>::LUTAccelerator(const Pixel32fC1 *aLevels, int *aAccelerator, int aLUTSize, int aAcceleratorSize)
    requires RealFloatingPoint<remove_vector_t<T>> && (!std::same_as<remove_vector_t<T>, double>)
{
    mpp::LUTAccelerator(reinterpret_cast<const float *>(aLevels), aLUTSize, aAccelerator, aAcceleratorSize);
}

template <PixelType T>
void ImageView<T>::LUTAccelerator(const mpp::cuda::DevVarView<Pixel32fC1> &aLevels,
                                  mpp::cuda::DevVarView<int> &aAccelerator, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealFloatingPoint<remove_vector_t<T>> && (!std::same_as<remove_vector_t<T>, double>)
{
    const int lutSize         = static_cast<int>(aLevels.Size());
    const int acceleratorSize = static_cast<int>(aAccelerator.Size());

    InvokeLutAcceleratorKernelDefault(reinterpret_cast<const float *>(aLevels.Pointer()), lutSize,
                                      aAccelerator.Pointer(), acceleratorSize, aStreamCtx);
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUT(ImageView<T> &aDst, const mpp::cuda::DevVarView<Pixel32fC1> &aLevels,
                                const mpp::cuda::DevVarView<Pixel32fC1> &aValues,
                                const mpp::cuda::DevVarView<int> &aAccelerator, InterpolationMode aInterpolationMode,
                                const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealFloatingPoint<remove_vector_t<T>> && (!std::same_as<remove_vector_t<T>, double>) &&
             (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aDst.ROI());
    if (aLevels.Size() != aValues.Size())
    {
        throw INVALIDARGUMENT(aLevels aValues, "aLevels and aValues must have the same size, but sizes are: "
                                                   << aLevels.Size() << " and " << aValues.Size());
    }

    const int lutSize         = static_cast<int>(aLevels.Size());
    const int acceleratorSize = static_cast<int>(aAccelerator.Size());

    InvokeLutSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aLevels.Pointer(), aValues.Pointer(),
                 aAccelerator.Pointer(), lutSize, acceleratorSize, aInterpolationMode, SizeRoi(), aStreamCtx);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUT(ImageView<T> &aDst,
                                const mpp::cuda::DevVarView<Pixel32fC1> aLevels[vector_active_size_v<T>],
                                const mpp::cuda::DevVarView<Pixel32fC1> aValues[vector_active_size_v<T>],
                                const mpp::cuda::DevVarView<int> aAccelerator[vector_active_size_v<T>],
                                InterpolationMode aInterpolationMode, const mpp::cuda::StreamCtx &aStreamCtx) const
    requires RealFloatingPoint<remove_vector_t<T>> && (!std::same_as<remove_vector_t<T>, double>) &&
             (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aDst.ROI());

    for (size_t channel = 0; channel < vector_active_size_v<T>; channel++)
    {
        if (aLevels[channel].Size() != aValues[channel].Size())
        {
            throw INVALIDARGUMENT(aLevels aValues, "aLevels and aValues must have the same size, but sizes for channel "
                                                       << channel << " are: " << aLevels[channel].Size() << " and "
                                                       << aValues[channel].Size());
        }
    }

    if constexpr (vector_active_size_v<T> == 2)
    {
        const std::array<const Pixel32fC1 *, vector_active_size_v<T>> levels = {aLevels[0].Pointer(),
                                                                                aLevels[1].Pointer()};
        const std::array<const Pixel32fC1 *, vector_active_size_v<T>> values = {aValues[0].Pointer(),
                                                                                aValues[1].Pointer()};
        const std::array<const int *, vector_active_size_v<T>> acc           = {aAccelerator[0].Pointer(),
                                                                                aAccelerator[1].Pointer()};
        const std::array<int, vector_active_size_v<T>> lutSize               = {static_cast<int>(aLevels[0].Size()),
                                                                                static_cast<int>(aLevels[1].Size())};
        const std::array<int, vector_active_size_v<T>> accSize = {static_cast<int>(aAccelerator[0].Size()),
                                                                  static_cast<int>(aAccelerator[1].Size())};

        InvokeLutSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), levels.data(), values.data(), acc.data(),
                     lutSize.data(), accSize.data(), aInterpolationMode, SizeRoi(), aStreamCtx);
    }
    else if constexpr (vector_active_size_v<T> == 3)
    {
        const std::array<const Pixel32fC1 *, vector_active_size_v<T>> levels = {
            aLevels[0].Pointer(), aLevels[1].Pointer(), aLevels[2].Pointer()};
        const std::array<const Pixel32fC1 *, vector_active_size_v<T>> values = {
            aValues[0].Pointer(), aValues[1].Pointer(), aValues[2].Pointer()};
        const std::array<const int *, vector_active_size_v<T>> acc = {
            aAccelerator[0].Pointer(), aAccelerator[1].Pointer(), aAccelerator[2].Pointer()};
        const std::array<int, vector_active_size_v<T>> lutSize = {static_cast<int>(aLevels[0].Size()),
                                                                  static_cast<int>(aLevels[1].Size()),
                                                                  static_cast<int>(aLevels[2].Size())};
        const std::array<int, vector_active_size_v<T>> accSize = {static_cast<int>(aAccelerator[0].Size()),
                                                                  static_cast<int>(aAccelerator[1].Size()),
                                                                  static_cast<int>(aAccelerator[2].Size())};

        InvokeLutSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), levels.data(), values.data(), acc.data(),
                     lutSize.data(), accSize.data(), aInterpolationMode, SizeRoi(), aStreamCtx);
    }
    else
    {
        const std::array<const Pixel32fC1 *, vector_active_size_v<T>> levels = {
            aLevels[0].Pointer(), aLevels[1].Pointer(), aLevels[2].Pointer(), aLevels[3].Pointer()};
        const std::array<const Pixel32fC1 *, vector_active_size_v<T>> values = {
            aValues[0].Pointer(), aValues[1].Pointer(), aValues[2].Pointer(), aValues[3].Pointer()};
        const std::array<const int *, vector_active_size_v<T>> acc = {
            aAccelerator[0].Pointer(), aAccelerator[1].Pointer(), aAccelerator[2].Pointer(), aAccelerator[3].Pointer()};
        const std::array<int, vector_active_size_v<T>> lutSize = {
            static_cast<int>(aLevels[0].Size()), static_cast<int>(aLevels[1].Size()),
            static_cast<int>(aLevels[2].Size()), static_cast<int>(aLevels[3].Size())};
        const std::array<int, vector_active_size_v<T>> accSize = {
            static_cast<int>(aAccelerator[0].Size()), static_cast<int>(aAccelerator[1].Size()),
            static_cast<int>(aAccelerator[2].Size()), static_cast<int>(aAccelerator[3].Size())};

        InvokeLutSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), levels.data(), values.data(), acc.data(),
                     lutSize.data(), accSize.data(), aInterpolationMode, SizeRoi(), aStreamCtx);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUT(const mpp::cuda::DevVarView<Pixel32fC1> &aLevels,
                                const mpp::cuda::DevVarView<Pixel32fC1> &aValues,
                                const mpp::cuda::DevVarView<int> &aAccelerator, InterpolationMode aInterpolationMode,
                                const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealFloatingPoint<remove_vector_t<T>> && (!std::same_as<remove_vector_t<T>, double>) &&
             (vector_active_size_v<T> == 1)
{
    if (aLevels.Size() != aValues.Size())
    {
        throw INVALIDARGUMENT(aLevels aValues, "aLevels and aValues must have the same size, but sizes are: "
                                                   << aLevels.Size() << " and " << aValues.Size());
    }

    const int lutSize         = static_cast<int>(aLevels.Size());
    const int acceleratorSize = static_cast<int>(aAccelerator.Size());

    InvokeLutInplace(PointerRoi(), Pitch(), aLevels.Pointer(), aValues.Pointer(), aAccelerator.Pointer(), lutSize,
                     acceleratorSize, aInterpolationMode, SizeRoi(), aStreamCtx);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUT(const mpp::cuda::DevVarView<Pixel32fC1> aLevels[vector_active_size_v<T>],
                                const mpp::cuda::DevVarView<Pixel32fC1> aValues[vector_active_size_v<T>],
                                const mpp::cuda::DevVarView<int> aAccelerator[vector_active_size_v<T>],
                                InterpolationMode aInterpolationMode, const mpp::cuda::StreamCtx &aStreamCtx)
    requires RealFloatingPoint<remove_vector_t<T>> && (!std::same_as<remove_vector_t<T>, double>) &&
             (vector_active_size_v<T> > 1)
{

    for (size_t channel = 0; channel < vector_active_size_v<T>; channel++)
    {
        if (aLevels[channel].Size() != aValues[channel].Size())
        {
            throw INVALIDARGUMENT(aLevels aValues, "aLevels and aValues must have the same size, but sizes for channel "
                                                       << channel << " are: " << aLevels[channel].Size() << " and "
                                                       << aValues[channel].Size());
        }
    }

    if constexpr (vector_active_size_v<T> == 2)
    {
        const std::array<const Pixel32fC1 *, vector_active_size_v<T>> levels = {aLevels[0].Pointer(),
                                                                                aLevels[1].Pointer()};
        const std::array<const Pixel32fC1 *, vector_active_size_v<T>> values = {aValues[0].Pointer(),
                                                                                aValues[1].Pointer()};
        const std::array<const int *, vector_active_size_v<T>> acc           = {aAccelerator[0].Pointer(),
                                                                                aAccelerator[1].Pointer()};
        const std::array<int, vector_active_size_v<T>> lutSize               = {static_cast<int>(aLevels[0].Size()),
                                                                                static_cast<int>(aLevels[1].Size())};
        const std::array<int, vector_active_size_v<T>> accSize = {static_cast<int>(aAccelerator[0].Size()),
                                                                  static_cast<int>(aAccelerator[1].Size())};

        InvokeLutInplace(PointerRoi(), Pitch(), levels.data(), values.data(), acc.data(), lutSize.data(),
                         accSize.data(), aInterpolationMode, SizeRoi(), aStreamCtx);
    }
    else if constexpr (vector_active_size_v<T> == 3)
    {
        const std::array<const Pixel32fC1 *, vector_active_size_v<T>> levels = {
            aLevels[0].Pointer(), aLevels[1].Pointer(), aLevels[2].Pointer()};
        const std::array<const Pixel32fC1 *, vector_active_size_v<T>> values = {
            aValues[0].Pointer(), aValues[1].Pointer(), aValues[2].Pointer()};
        const std::array<const int *, vector_active_size_v<T>> acc = {
            aAccelerator[0].Pointer(), aAccelerator[1].Pointer(), aAccelerator[2].Pointer()};
        const std::array<int, vector_active_size_v<T>> lutSize = {static_cast<int>(aLevels[0].Size()),
                                                                  static_cast<int>(aLevels[1].Size()),
                                                                  static_cast<int>(aLevels[2].Size())};
        const std::array<int, vector_active_size_v<T>> accSize = {static_cast<int>(aAccelerator[0].Size()),
                                                                  static_cast<int>(aAccelerator[1].Size()),
                                                                  static_cast<int>(aAccelerator[2].Size())};

        InvokeLutInplace(PointerRoi(), Pitch(), levels.data(), values.data(), acc.data(), lutSize.data(),
                         accSize.data(), aInterpolationMode, SizeRoi(), aStreamCtx);
    }
    else
    {
        const std::array<const Pixel32fC1 *, vector_active_size_v<T>> levels = {
            aLevels[0].Pointer(), aLevels[1].Pointer(), aLevels[2].Pointer(), aLevels[3].Pointer()};
        const std::array<const Pixel32fC1 *, vector_active_size_v<T>> values = {
            aValues[0].Pointer(), aValues[1].Pointer(), aValues[2].Pointer(), aValues[3].Pointer()};
        const std::array<const int *, vector_active_size_v<T>> acc = {
            aAccelerator[0].Pointer(), aAccelerator[1].Pointer(), aAccelerator[2].Pointer(), aAccelerator[3].Pointer()};
        const std::array<int, vector_active_size_v<T>> lutSize = {
            static_cast<int>(aLevels[0].Size()), static_cast<int>(aLevels[1].Size()),
            static_cast<int>(aLevels[2].Size()), static_cast<int>(aLevels[3].Size())};
        const std::array<int, vector_active_size_v<T>> accSize = {
            static_cast<int>(aAccelerator[0].Size()), static_cast<int>(aAccelerator[1].Size()),
            static_cast<int>(aAccelerator[2].Size()), static_cast<int>(aAccelerator[3].Size())};

        InvokeLutInplace(PointerRoi(), Pitch(), levels.data(), values.data(), acc.data(), lutSize.data(),
                         accSize.data(), aInterpolationMode, SizeRoi(), aStreamCtx);
    }
    return *this;
}

#pragma endregion
#pragma region Lut3D

template <PixelType T>
ImageView<T> &ImageView<T>::LUTTrilinear(ImageView<T> &aDst,
                                         const mpp::cuda::DevVarView<Vector3<remove_vector_t<T>>> &aLut3D,
                                         const Vector3<remove_vector_t<T>> &aMinLevel,
                                         const Vector3<remove_vector_t<T>> &aMaxLevel, const Pixel32sC3 &aLutSize,
                                         const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> >= 3)
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeLutTrilinearSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aLut3D.Pointer(), aMinLevel,
                          aMaxLevel, aLutSize, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTTrilinear(ImageView<T> &aDst,
                                         const mpp::cuda::DevVarView<Vector4A<remove_vector_t<T>>> &aLut3D,
                                         const Vector3<remove_vector_t<T>> &aMinLevel,
                                         const Vector3<remove_vector_t<T>> &aMaxLevel, const Pixel32sC3 &aLutSize,
                                         const mpp::cuda::StreamCtx &aStreamCtx) const
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> >= 3)
{
    checkSameSize(ROI(), aDst.ROI());

    InvokeLutTrilinearSrc(PointerRoi(), Pitch(), aDst.PointerRoi(), aDst.Pitch(), aLut3D.Pointer(), aMinLevel,
                          aMaxLevel, aLutSize, SizeRoi(), aStreamCtx);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTTrilinear(const mpp::cuda::DevVarView<Vector3<remove_vector_t<T>>> &aLut3D,
                                         const Vector3<remove_vector_t<T>> &aMinLevel,
                                         const Vector3<remove_vector_t<T>> &aMaxLevel, const Pixel32sC3 &aLutSize,
                                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> >= 3)
{
    InvokeLutTrilinearInplace(PointerRoi(), Pitch(), aLut3D.Pointer(), aMinLevel, aMaxLevel, aLutSize, SizeRoi(),
                              aStreamCtx);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LUTTrilinear(const mpp::cuda::DevVarView<Vector4A<remove_vector_t<T>>> &aLut3D,
                                         const Vector3<remove_vector_t<T>> &aMinLevel,
                                         const Vector3<remove_vector_t<T>> &aMaxLevel, const Pixel32sC3 &aLutSize,
                                         const mpp::cuda::StreamCtx &aStreamCtx)
    requires(std::same_as<byte, remove_vector_t<T>> || std::same_as<ushort, remove_vector_t<T>> ||
             std::same_as<short, remove_vector_t<T>> || std::same_as<float, remove_vector_t<T>>) &&
            (vector_active_size_v<T> >= 3)
{
    InvokeLutTrilinearInplace(PointerRoi(), Pitch(), aLut3D.Pointer(), aMinLevel, aMaxLevel, aLutSize, SizeRoi(),
                              aStreamCtx);

    return *this;
}

#pragma endregion
#pragma endregion

} // namespace mpp::image::cuda