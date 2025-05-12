#pragma once
#include <backends/simple_cpu/image/forEachPixel.h>
#include <backends/simple_cpu/image/forEachPixelPlanar.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/affineTransformation.h>
#include <common/image/border.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/functors/inplaceTransformerFunctor.h>
#include <common/image/functors/interpolator.h>
#include <common/image/functors/transformer.h>
#include <common/image/functors/transformerFunctor.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/opp_defs.h>
#include <common/safeCast.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vector1.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace opp::image::cpuSimple
{
#pragma region Rotate
template <PixelType T>
ImageView<T> &ImageView<T>::Rotate(ImageView<T> &aDst, double aAngleInDeg, const Vector2<double> &aShift,
                                   InterpolationMode aInterpolation, BorderType aBorder, Roi aAllowedReadRoi) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Rotate(aDst, aAngleInDeg, aShift, aInterpolation, aBorder, {0}, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::Rotate(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          BorderType aBorder, Roi aAllowedReadRoi)
    requires TwoChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Rotate(aSrc1, aSrc2, aDst1, aDst2, aAngleInDeg, aShift, aInterpolation, aBorder, {0},
                         aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::Rotate(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          BorderType aBorder, Roi aAllowedReadRoi)
    requires ThreeChannel<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Rotate(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, aAngleInDeg, aShift, aInterpolation, aBorder, {0},
                         aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::Rotate(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          BorderType aBorder, Roi aAllowedReadRoi)
    requires FourChannelNoAlpha<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    ImageView<T>::Rotate(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, aAngleInDeg, aShift, aInterpolation,
                         aBorder, {0}, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Rotate(ImageView<T> &aDst, double aAngleInDeg, const Vector2<double> &aShift,
                                   InterpolationMode aInterpolation, BorderType aBorder, T aConstant,
                                   Roi aAllowedReadRoi) const
{
    // The rotation and shift are given from source to destination, with shift applied after rotation. As we compute
    // from destination to source, we have to invert the transformation, the shift being now a pre-rotation shift:

    const AffineTransformation<double> rotate =
        AffineTransformation<double>::GetRotation(-aAngleInDeg) * AffineTransformation<double>::GetTranslation(-aShift);
    return this->WarpAffineBack(aDst, rotate, aInterpolation, aBorder, aConstant, aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::Rotate(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          BorderType aBorder, T aConstant, Roi aAllowedReadRoi)
    requires TwoChannel<T>
{
    // The rotation and shift are given from source to destination, with shift applied after rotation. As we compute
    // from destination to source, we have to invert the transformation, the shift being now a pre-rotation shift:

    const AffineTransformation<double> rotate =
        AffineTransformation<double>::GetRotation(-aAngleInDeg) * AffineTransformation<double>::GetTranslation(-aShift);
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aDst1, aDst2, rotate, aInterpolation, aBorder, aConstant,
                                 aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::Rotate(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aDst1,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst2, ImageView<Vector1<remove_vector_t<T>>> &aDst3,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          BorderType aBorder, T aConstant, Roi aAllowedReadRoi)
    requires ThreeChannel<T>
{
    // The rotation and shift are given from source to destination, with shift applied after rotation. As we compute
    // from destination to source, we have to invert the transformation, the shift being now a pre-rotation shift:

    const AffineTransformation<double> rotate =
        AffineTransformation<double>::GetRotation(-aAngleInDeg) * AffineTransformation<double>::GetTranslation(-aShift);
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aDst1, aDst2, aDst3, rotate, aInterpolation, aBorder, aConstant,
                                 aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::Rotate(ImageView<Vector1<remove_vector_t<T>>> &aSrc1, ImageView<Vector1<remove_vector_t<T>>> &aSrc2,
                          ImageView<Vector1<remove_vector_t<T>>> &aSrc3, ImageView<Vector1<remove_vector_t<T>>> &aSrc4,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst1, ImageView<Vector1<remove_vector_t<T>>> &aDst2,
                          ImageView<Vector1<remove_vector_t<T>>> &aDst3, ImageView<Vector1<remove_vector_t<T>>> &aDst4,
                          double aAngleInDeg, const Vector2<double> &aShift, InterpolationMode aInterpolation,
                          BorderType aBorder, T aConstant, Roi aAllowedReadRoi)
    requires FourChannelNoAlpha<T>
{
    // The rotation and shift are given from source to destination, with shift applied after rotation. As we compute
    // from destination to source, we have to invert the transformation, the shift being now a pre-rotation shift:

    const AffineTransformation<double> rotate =
        AffineTransformation<double>::GetRotation(-aAngleInDeg) * AffineTransformation<double>::GetTranslation(-aShift);
    ImageView<T>::WarpAffineBack(aSrc1, aSrc2, aSrc3, aSrc4, aDst1, aDst2, aDst3, aDst4, rotate, aInterpolation,
                                 aBorder, aConstant, aAllowedReadRoi);
}
#pragma endregion
} // namespace opp::image::cpuSimple