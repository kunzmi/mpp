#pragma once
#include <backends/simple_cpu/image/filterEachPixel_impl.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/filtering/postOperators.h>
#include <common/half_fp16.h>
#include <common/image/border.h>
#include <common/image/channel.h>
#include <common/image/channelList.h>
#include <common/image/filterArea.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/morphology/operators.h>
#include <common/morphology/postOperators.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/safeCast.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vector1.h>
#include <common/vector3.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace mpp::image::cpuSimple
{

#pragma region No mask Erosion/Dilation
template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                     const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Dilation(aDst, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                                     BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    return this->MaxFilter(aDst, aFilterArea, aConstant, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                    const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Erosion(aDst, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                                    BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    return this->MinFilter(aDst, aFilterArea, aConstant, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder) const
    requires RealVector<T>
{
    return this->Dilation(aDst, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                                     BorderType aBorder) const
    requires RealVector<T>
{
    return this->Dilation(aDst, aFilterArea, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder) const
    requires RealVector<T>
{
    return this->Erosion(aDst, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                                    BorderType aBorder) const
    requires RealVector<T>
{
    return this->Erosion(aDst, aFilterArea, aConstant, aBorder, ROI());
}
#pragma endregion

#pragma region Erosion

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                                    BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Erosion(aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                                    const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using FilterT = Pixel8uC1;
    using MorphOp = mpp::Erode<T, FilterT>;
    using PostOp  = mpp::NothingMorph<T>;

    const MorphOp op;
    const PostOp postOp;

    moprhologyEachPixel(*this, aDst, aMask, aFilterArea, aBorder, aConstant, aAllowedReadRoi, op, postOp);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ErosionGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                                        const FilterArea &aFilterArea, BorderType aBorder,
                                        const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->ErosionGray(aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ErosionGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                                        const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                                        const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using FilterT = morph_gray_compute_type_t<T>;
    using MorphOp = mpp::ErodeGray<T, FilterT>;
    using PostOp  = mpp::NothingMorph<T>;

    const MorphOp op;
    const PostOp postOp;

    moprhologyEachPixel(*this, aDst, aMask, aFilterArea, aBorder, aConstant, aAllowedReadRoi, op, postOp);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                                    BorderType aBorder) const
    requires RealVector<T>
{
    return this->Erosion(aDst, aMask, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::Erosion(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                                    const T &aConstant, BorderType aBorder) const
    requires RealVector<T>
{
    return this->Erosion(aDst, aMask, aFilterArea, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::ErosionGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                                        const FilterArea &aFilterArea, BorderType aBorder) const
    requires RealVector<T>
{
    return this->ErosionGray(aDst, aMask, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::ErosionGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                                        const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder) const
    requires RealVector<T>
{
    return this->ErosionGray(aDst, aMask, aFilterArea, aConstant, aBorder, ROI());
}
#pragma endregion

#pragma region Dilation

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                                     BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Dilation(aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                                     const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using FilterT = Pixel8uC1;
    using MorphOp = mpp::Dilate<T, FilterT>;
    using PostOp  = mpp::NothingMorph<T>;

    const MorphOp op;
    const PostOp postOp;

    moprhologyEachPixel(*this, aDst, aMask, aFilterArea, aBorder, aConstant, aAllowedReadRoi, op, postOp);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DilationGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                                         const FilterArea &aFilterArea, BorderType aBorder,
                                         const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->DilationGray(aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::DilationGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                                         const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                                         const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using FilterT = morph_gray_compute_type_t<T>;
    using MorphOp = mpp::DilateGray<T, FilterT>;
    using PostOp  = mpp::NothingMorph<T>;

    const MorphOp op;
    const PostOp postOp;

    moprhologyEachPixel(*this, aDst, aMask, aFilterArea, aBorder, aConstant, aAllowedReadRoi, op, postOp);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                                     BorderType aBorder) const
    requires RealVector<T>
{
    return this->Dilation(aDst, aMask, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::Dilation(ImageView<T> &aDst, const Pixel8uC1 *aMask, const FilterArea &aFilterArea,
                                     const T &aConstant, BorderType aBorder) const
    requires RealVector<T>
{
    return this->Dilation(aDst, aMask, aFilterArea, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::DilationGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                                         const FilterArea &aFilterArea, BorderType aBorder) const
    requires RealVector<T>
{
    return this->DilationGray(aDst, aMask, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::DilationGray(ImageView<T> &aDst, const morph_gray_compute_type_t<T> *aMask,
                                         const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder) const
    requires RealVector<T>
{
    return this->DilationGray(aDst, aMask, aFilterArea, aConstant, aBorder, ROI());
}
#pragma endregion

#pragma region Open

template <PixelType T>
ImageView<T> &ImageView<T>::Open(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                 const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Open(aTemp, aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Open(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                 const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                                 const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    this->Erosion(aTemp, aMask, aFilterArea, aConstant, aBorder, aAllowedReadRoi);
    aTemp.Dilation(aDst, aMask, aFilterArea, aConstant, aBorder, aAllowedReadRoi);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Open(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                 const FilterArea &aFilterArea, BorderType aBorder) const
    requires RealVector<T>
{
    return this->Open(aTemp, aDst, aMask, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::Open(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                 const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder) const
    requires RealVector<T>
{
    return this->Open(aTemp, aDst, aMask, aFilterArea, aConstant, aBorder, ROI());
}
#pragma endregion

#pragma region Close
template <PixelType T>
ImageView<T> &ImageView<T>::Close(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                  const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Close(aTemp, aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Close(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                  const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                                  const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    this->Dilation(aTemp, aMask, aFilterArea, aConstant, aBorder, aAllowedReadRoi);
    aTemp.Erosion(aDst, aMask, aFilterArea, aConstant, aBorder, aAllowedReadRoi);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Close(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                  const FilterArea &aFilterArea, BorderType aBorder) const
    requires RealVector<T>
{
    return this->Close(aTemp, aDst, aMask, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::Close(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                  const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder) const
    requires RealVector<T>
{
    return this->Close(aTemp, aDst, aMask, aFilterArea, aConstant, aBorder, ROI());
}
#pragma endregion

#pragma region TopHat
template <PixelType T>
ImageView<T> &ImageView<T>::TopHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                   const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->TopHat(aTemp, aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::TopHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                   const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                                   const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    this->Erosion(aTemp, aMask, aFilterArea, aConstant, aBorder, aAllowedReadRoi);

    using FilterT  = Pixel8uC1;
    using ComputeT = morph_compute_type_t<T>;
    using MorphOp  = mpp::Dilate<T, FilterT>;
    using PostOp   = mpp::TopHat<T, ComputeT>;

    const MorphOp op;
    const PostOp postOp(PointerRoi(), Pitch());

    moprhologyEachPixel(aTemp, aDst, aMask, aFilterArea, aBorder, aConstant, aAllowedReadRoi, op, postOp);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::TopHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                   const FilterArea &aFilterArea, BorderType aBorder) const
    requires RealVector<T>
{
    return this->TopHat(aTemp, aDst, aMask, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::TopHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                   const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder) const
    requires RealVector<T>
{
    return this->TopHat(aTemp, aDst, aMask, aFilterArea, aConstant, aBorder, ROI());
}
#pragma endregion

#pragma region BlackHat
template <PixelType T>
ImageView<T> &ImageView<T>::BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                     const FilterArea &aFilterArea, BorderType aBorder,
                                     const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->BlackHat(aTemp, aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                     const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                                     const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    this->Dilation(aTemp, aMask, aFilterArea, aConstant, aBorder, aAllowedReadRoi);

    using FilterT  = Pixel8uC1;
    using ComputeT = morph_compute_type_t<T>;
    using MorphOp  = mpp::Erode<T, FilterT>;
    using PostOp   = mpp::BlackHat<T, ComputeT>;

    const MorphOp op;
    const PostOp postOp(PointerRoi(), Pitch());

    moprhologyEachPixel(aTemp, aDst, aMask, aFilterArea, aBorder, aConstant, aAllowedReadRoi, op, postOp);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                     const FilterArea &aFilterArea, BorderType aBorder) const
    requires RealVector<T>
{
    return this->BlackHat(aTemp, aDst, aMask, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::BlackHat(ImageView<T> &aTemp, ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                     const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder) const
    requires RealVector<T>
{
    return this->BlackHat(aTemp, aDst, aMask, aFilterArea, aConstant, aBorder, ROI());
}
#pragma endregion

#pragma region Morphology Gradient
template <PixelType T>
ImageView<T> &ImageView<T>::MorphologyGradient(ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                               const FilterArea &aFilterArea, BorderType aBorder,
                                               const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->MorphologyGradient(aDst, aMask, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::MorphologyGradient(ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                               const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                                               const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    moprhologyGradientEachPixel(*this, aDst, aMask, aFilterArea, aBorder, aConstant, aAllowedReadRoi);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MorphologyGradient(ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                               const FilterArea &aFilterArea, BorderType aBorder) const
    requires RealVector<T>
{
    return this->MorphologyGradient(aDst, aMask, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::MorphologyGradient(ImageView<T> &aDst, const Pixel8uC1 *aMask,
                                               const FilterArea &aFilterArea, const T &aConstant,
                                               BorderType aBorder) const
    requires RealVector<T>
{
    return this->MorphologyGradient(aDst, aMask, aFilterArea, aConstant, aBorder, ROI());
}
#pragma endregion
} // namespace mpp::image::cpuSimple