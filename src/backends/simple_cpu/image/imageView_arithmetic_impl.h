#pragma once
#include <backends/simple_cpu/image/forEachPixel.h>
#include <backends/simple_cpu/image/forEachPixelMasked.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/border.h>
#include <common/image/functors/constantFunctor.h>
#include <common/image/functors/inplaceConstantFunctor.h>
#include <common/image/functors/inplaceConstantScaleFunctor.h>
#include <common/image/functors/inplaceDevConstantFunctor.h>
#include <common/image/functors/inplaceDevConstantScaleFunctor.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/inplaceSrcScaleFunctor.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcConstantScaleFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
#include <common/image/functors/srcDevConstantScaleFunctor.h>
#include <common/image/functors/srcFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/functors/srcSrcScaleFunctor.h>
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
#include <common/vectorTypes.h>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace opp::image::cpuSimple
{
#pragma region Add
template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoundingMode::None>;

    const opp::Add<ComputeT> op;
    const addSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using addSrcSrcScale =
        SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Add<ComputeT> op;
    const addSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, ImageView<T> &aDst)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoundingMode::None>;

    const opp::Add<ComputeT> op;
    const addSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, ImageView<T> &aDst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using addSrcCScale =
        SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Add<ComputeT> op;
    const addSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoundingMode::None>;

    const opp::Add<ComputeT> op;
    const addInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using addInplaceSrcScale =
        InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Add<ComputeT> op;
    const addInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::Add<ComputeT>, RoundingMode::None>;

    const opp::Add<ComputeT> op;
    const addInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using addInplaceCScale =
        InplaceConstantScaleFunctor<1, ComputeT, T, opp::Add<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Add<ComputeT> op;
    const addInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoundingMode::None>;

    const opp::Add<ComputeT> op;
    const addSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using addSrcSrcScale =
        SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Add<ComputeT> op;
    const addSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoundingMode::None>;

    const opp::Add<ComputeT> op;
    const addSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using addSrcCScale =
        SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Add<ComputeT> op;
    const addSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoundingMode::None>;

    const opp::Add<ComputeT> op;
    const addInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using addInplaceSrcScale =
        InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Add<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Add<ComputeT> op;
    const addInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::Add<ComputeT>, RoundingMode::None>;

    const opp::Add<ComputeT> op;
    const addInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using addInplaceCScale =
        InplaceConstantScaleFunctor<1, ComputeT, T, opp::Add<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Add<ComputeT> op;
    const addInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aMask, *this, functor);
    return *this;
}
#pragma endregion

#pragma region Sub
template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::None>;

    const opp::Sub<ComputeT> op;
    const subSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using subSrcSrcScale =
        SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Sub<ComputeT> op;
    const subSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, ImageView<T> &aDst)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::None>;

    const opp::Sub<ComputeT> op;
    const subSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, ImageView<T> &aDst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using subSrcCScale =
        SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Sub<ComputeT> op;
    const subSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::None>;

    const opp::Sub<ComputeT> op;
    const subInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using subInplaceSrcScale =
        InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Sub<ComputeT> op;
    const subInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::None>;

    const opp::Sub<ComputeT> op;
    const subInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using subInplaceCScale =
        InplaceConstantScaleFunctor<1, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Sub<ComputeT> op;
    const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const ImageView<T> &aSrc2)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::SubInv<ComputeT>, RoundingMode::None>;

    const opp::SubInv<ComputeT> op;
    const subInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const ImageView<T> &aSrc2, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using subInplaceSrcScale =
        InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::SubInv<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::SubInv<ComputeT> op;
    const subInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const T &aConst)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::SubInv<ComputeT>, RoundingMode::None>;

    const opp::SubInv<ComputeT> op;
    const subInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const T &aConst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using subInplaceCScale =
        InplaceConstantScaleFunctor<1, ComputeT, T, opp::SubInv<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::SubInv<ComputeT> op;
    const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::None>;

    const opp::Sub<ComputeT> op;
    const subSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using subSrcSrcScale =
        SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Sub<ComputeT> op;
    const subSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::None>;

    const opp::Sub<ComputeT> op;
    const subSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using subSrcCScale =
        SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Sub<ComputeT> op;
    const subSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::None>;

    const opp::Sub<ComputeT> op;
    const subInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using subInplaceSrcScale =
        InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Sub<ComputeT> op;
    const subInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::None>;

    const opp::Sub<ComputeT> op;
    const subInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using subInplaceCScale =
        InplaceConstantScaleFunctor<1, ComputeT, T, opp::Sub<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Sub<ComputeT> op;
    const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::SubInv<ComputeT>, RoundingMode::None>;

    const opp::SubInv<ComputeT> op;
    const subInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using subInplaceSrcScale =
        InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::SubInv<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::SubInv<ComputeT> op;
    const subInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::SubInv<ComputeT>, RoundingMode::None>;

    const opp::SubInv<ComputeT> op;
    const subInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using subInplaceCScale =
        InplaceConstantScaleFunctor<1, ComputeT, T, opp::SubInv<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::SubInv<ComputeT> op;
    const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aMask, *this, functor);
    return *this;
}
#pragma endregion

#pragma region Mul
template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::None>;

    const opp::Mul<ComputeT> op;
    const mulSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using mulSrcSrcScale =
        SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, ImageView<T> &aDst)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::None>;

    const opp::Mul<ComputeT> op;
    const mulSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, ImageView<T> &aDst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using mulSrcCScale =
        SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::None>;

    const opp::Mul<ComputeT> op;
    const mulInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using mulInplaceSrcScale =
        InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::None>;

    const opp::Mul<ComputeT> op;
    const mulInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using mulInplaceCScale =
        InplaceConstantScaleFunctor<1, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::None>;

    const opp::Mul<ComputeT> op;
    const mulSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using mulSrcSrcScale =
        SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::None>;

    const opp::Mul<ComputeT> op;
    const mulSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using mulSrcCScale =
        SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::None>;

    const opp::Mul<ComputeT> op;
    const mulInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using mulInplaceSrcScale =
        InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::None>;

    const opp::Mul<ComputeT> op;
    const mulInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    using mulInplaceCScale =
        InplaceConstantScaleFunctor<1, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aMask, *this, functor);
    return *this;
}
#pragma endregion

#pragma region MulScale
template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    constexpr float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_compute_type_for_t<T>;

    using mulSrcSrcScale =
        SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const T &aConst, ImageView<T> &aDst)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aDst.ROI());

    constexpr float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_compute_type_for_t<T>;

    using mulSrcCScale =
        SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const ImageView<T> &aSrc2)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aSrc2.ROI());

    constexpr float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_compute_type_for_t<T>;

    using mulInplaceSrcScale =
        InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const T &aConst)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    constexpr float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_compute_type_for_t<T>;

    using mulInplaceCScale =
        InplaceConstantScaleFunctor<1, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    constexpr float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_compute_type_for_t<T>;

    using mulSrcSrcScale =
        SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aDst.ROI());

    constexpr float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_compute_type_for_t<T>;

    using mulSrcCScale =
        SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aSrc2.ROI());

    constexpr float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_compute_type_for_t<T>;

    using mulInplaceSrcScale =
        InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    constexpr float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_compute_type_for_t<T>;

    using mulInplaceCScale =
        InplaceConstantScaleFunctor<1, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);

    forEachPixel(aMask, *this, functor);
    return *this;
}
#pragma endregion

#pragma region Div
template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::None>;

    const opp::Div<ComputeT> op;
    const divSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor,
                                RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using divSrcSrcScale =
                SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesToEven>;
            const opp::Div<ComputeT> op;
            const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op,
                                         scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using divSrcSrcScale =
                SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;
            const opp::Div<ComputeT> op;
            const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op,
                                         scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using divSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardZero>;
            const opp::Div<ComputeT> op;
            const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op,
                                         scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using divSrcSrcScale =
                SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardNegativeInfinity>;
            const opp::Div<ComputeT> op;
            const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op,
                                         scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using divSrcSrcScale =
                SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardPositiveInfinity>;
            const opp::Div<ComputeT> op;
            const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op,
                                         scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, ImageView<T> &aDst)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::None>;

    const opp::Div<ComputeT> op;
    const divSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, ImageView<T> &aDst, int aScaleFactor, RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using divSrcCScale =
                SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesToEven>;
            const opp::Div<ComputeT> op;
            const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using divSrcCScale =
                SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;
            const opp::Div<ComputeT> op;
            const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using divSrcCScale =
                SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardZero>;
            const opp::Div<ComputeT> op;
            const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using divSrcCScale =
                SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardNegativeInfinity>;
            const opp::Div<ComputeT> op;
            const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using divSrcCScale =
                SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardPositiveInfinity>;
            const opp::Div<ComputeT> op;
            const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::None>;

    const opp::Div<ComputeT> op;
    const divInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, int aScaleFactor, RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesToEven>;
            const opp::Div<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;
            const opp::Div<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardZero>;
            const opp::Div<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardNegativeInfinity>;
            const opp::Div<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardPositiveInfinity>;
            const opp::Div<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::Div<ComputeT>, RoundingMode::None>;

    const opp::Div<ComputeT> op;
    const divInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, int aScaleFactor, RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesToEven>;
            const opp::Div<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;
            const opp::Div<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardZero>;
            const opp::Div<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardNegativeInfinity>;
            const opp::Div<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardPositiveInfinity>;
            const opp::Div<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const ImageView<T> &aSrc2)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::None>;

    const opp::DivInv<ComputeT> op;
    const divInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const ImageView<T> &aSrc2, int aScaleFactor, RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::NearestTiesToEven>;
            const opp::DivInv<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;
            const opp::DivInv<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::TowardZero>;
            const opp::DivInv<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::TowardNegativeInfinity>;
            const opp::DivInv<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::TowardPositiveInfinity>;
            const opp::DivInv<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const T &aConst)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::None>;

    const opp::DivInv<ComputeT> op;
    const divInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const T &aConst, int aScaleFactor, RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::NearestTiesToEven>;
            const opp::DivInv<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, opp::DivInv<ComputeT>,
                                                                 RoundingMode::NearestTiesAwayFromZero>;
            const opp::DivInv<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::TowardZero>;
            const opp::DivInv<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, opp::DivInv<ComputeT>,
                                                                 RoundingMode::TowardNegativeInfinity>;
            const opp::DivInv<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, opp::DivInv<ComputeT>,
                                                                 RoundingMode::TowardPositiveInfinity>;
            const opp::DivInv<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(*this, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::None>;

    const opp::Div<ComputeT> op;
    const divSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using divSrcSrcScale =
                SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesToEven>;
            const opp::Div<ComputeT> op;
            const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op,
                                         scaleFactorFloat);
            forEachPixel(aMask, aDst, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using divSrcSrcScale =
                SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;
            const opp::Div<ComputeT> op;
            const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op,
                                         scaleFactorFloat);
            forEachPixel(aMask, aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using divSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardZero>;
            const opp::Div<ComputeT> op;
            const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op,
                                         scaleFactorFloat);
            forEachPixel(aMask, aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using divSrcSrcScale =
                SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardNegativeInfinity>;
            const opp::Div<ComputeT> op;
            const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op,
                                         scaleFactorFloat);
            forEachPixel(aMask, aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using divSrcSrcScale =
                SrcSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardPositiveInfinity>;
            const opp::Div<ComputeT> op;
            const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op,
                                         scaleFactorFloat);
            forEachPixel(aMask, aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::None>;

    const opp::Div<ComputeT> op;
    const divSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                int aScaleFactor, RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using divSrcCScale =
                SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesToEven>;
            const opp::Div<ComputeT> op;
            const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, aDst, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using divSrcCScale =
                SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;
            const opp::Div<ComputeT> op;
            const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using divSrcCScale =
                SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardZero>;
            const opp::Div<ComputeT> op;
            const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using divSrcCScale =
                SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardNegativeInfinity>;
            const opp::Div<ComputeT> op;
            const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, aDst, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using divSrcCScale =
                SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardPositiveInfinity>;
            const opp::Div<ComputeT> op;
            const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, aDst, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::None>;

    const opp::Div<ComputeT> op;
    const divInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesToEven>;
            const opp::Div<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;
            const opp::Div<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardZero>;
            const opp::Div<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardNegativeInfinity>;
            const opp::Div<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardPositiveInfinity>;
            const opp::Div<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::Div<ComputeT>, RoundingMode::None>;

    const opp::Div<ComputeT> op;
    const divInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesToEven>;
            const opp::Div<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::Div<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;
            const opp::Div<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardZero>;
            const opp::Div<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardNegativeInfinity>;
            const opp::Div<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::Div<ComputeT>, RoundingMode::TowardPositiveInfinity>;
            const opp::Div<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::None>;

    const opp::DivInv<ComputeT> op;
    const divInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                   RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::NearestTiesToEven>;
            const opp::DivInv<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;
            const opp::DivInv<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::TowardZero>;
            const opp::DivInv<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::TowardNegativeInfinity>;
            const opp::DivInv<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using divInplaceSrcScale =
                InplaceSrcScaleFunctor<1, T, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::TowardPositiveInfinity>;
            const opp::DivInv<ComputeT> op;
            const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Div cannot produce non-integers in computations with ints:
    using divInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::None>;

    const opp::DivInv<ComputeT> op;
    const divInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                   RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    const float scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::NearestTiesToEven>;
            const opp::DivInv<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::NearestTiesAwayFromZero:
        {
            using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, opp::DivInv<ComputeT>,
                                                                 RoundingMode::NearestTiesAwayFromZero>;
            const opp::DivInv<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::TowardZero:
        {
            using divInplaceCScale =
                InplaceConstantScaleFunctor<1, ComputeT, T, opp::DivInv<ComputeT>, RoundingMode::TowardZero>;
            const opp::DivInv<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::TowardNegativeInfinity:
        {
            using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, opp::DivInv<ComputeT>,
                                                                 RoundingMode::TowardNegativeInfinity>;
            const opp::DivInv<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        case opp::RoundingMode::TowardPositiveInfinity:
        {
            using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, opp::DivInv<ComputeT>,
                                                                 RoundingMode::TowardPositiveInfinity>;
            const opp::DivInv<ComputeT> op;
            const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaleFactorFloat);
            forEachPixel(aMask, *this, functor);
        }
        break;
        default:
            throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
    }
    return *this;
}
#pragma endregion

#pragma region Abs
template <PixelType T>
ImageView<T> &ImageView<T>::Abs(ImageView<T> &aDst)
    requires RealSignedVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using ComputeT = default_compute_type_for_t<T>;

    using absSrc = SrcFunctor<1, T, ComputeT, T, opp::Abs<ComputeT>, RoundingMode::None>;

    const opp::Abs<ComputeT> op;
    const absSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Abs()
    requires RealSignedVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using absInplace = InplaceFunctor<1, ComputeT, T, opp::Abs<ComputeT>, RoundingMode::None>;

    const opp::Abs<ComputeT> op;
    const absInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region AbsDiff
template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    requires RealUnsignedVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using AbsDiffSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::AbsDiff<ComputeT>, RoundingMode::None>;

    const opp::AbsDiff<ComputeT> op;
    const AbsDiffSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const T &aConst, ImageView<T> &aDst)
    requires RealUnsignedVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using AbsDiffSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::AbsDiff<ComputeT>, RoundingMode::None>;

    const opp::AbsDiff<ComputeT> op;
    const AbsDiffSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const ImageView<T> &aSrc2)
    requires RealUnsignedVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using AbsDiffInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::AbsDiff<ComputeT>, RoundingMode::None>;

    const opp::AbsDiff<ComputeT> op;
    const AbsDiffInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const T &aConst)
    requires RealUnsignedVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using AbsDiffInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::AbsDiff<ComputeT>, RoundingMode::None>;

    const opp::AbsDiff<ComputeT> op;
    const AbsDiffInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region And
template <PixelType T>
ImageView<T> &ImageView<T>::And(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using AndSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::And<ComputeT>, RoundingMode::None>;

    const opp::And<ComputeT> op;
    const AndSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const T &aConst, ImageView<T> &aDst)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using AndSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::And<ComputeT>, RoundingMode::None>;

    const opp::And<ComputeT> op;
    const AndSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const ImageView<T> &aSrc2)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = T;

    using AndInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::And<ComputeT>, RoundingMode::None>;

    const opp::And<ComputeT> op;
    const AndInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const T &aConst)
    requires RealIntVector<T>
{
    using ComputeT = T;

    using AndInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::And<ComputeT>, RoundingMode::None>;

    const opp::And<ComputeT> op;
    const AndInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Not
template <PixelType T>
ImageView<T> &ImageView<T>::Not(ImageView<T> &aDst)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using ComputeT = T;

    using notSrc = SrcFunctor<1, T, ComputeT, T, opp::Not<ComputeT>, RoundingMode::None>;

    const opp::Not<ComputeT> op;
    const notSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Not()
    requires RealIntVector<T>
{
    using ComputeT = T;

    using notInplace = InplaceFunctor<1, ComputeT, T, opp::Not<ComputeT>, RoundingMode::None>;

    const opp::Not<ComputeT> op;
    const notInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Exp
template <PixelType T>
ImageView<T> &ImageView<T>::Exp(ImageView<T> &aDst)
    requires RealOrComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using ComputeT = default_compute_type_for_t<T>;

    using expSrc = SrcFunctor<1, T, ComputeT, T, opp::Exp<ComputeT>, RoundingMode::None>;

    const opp::Exp<ComputeT> op;
    const expSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Exp()
    requires RealOrComplexVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using expInplace = InplaceFunctor<1, ComputeT, T, opp::Exp<ComputeT>, RoundingMode::None>;

    const opp::Exp<ComputeT> op;
    const expInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Ln
template <PixelType T>
ImageView<T> &ImageView<T>::Ln(ImageView<T> &aDst)
    requires RealOrComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using ComputeT = default_compute_type_for_t<T>;

    using lnSrc = SrcFunctor<1, T, ComputeT, T, opp::Ln<ComputeT>, RoundingMode::None>;

    const opp::Ln<ComputeT> op;
    const lnSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Ln()
    requires RealOrComplexVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using lnInplace = InplaceFunctor<1, ComputeT, T, opp::Ln<ComputeT>, RoundingMode::None>;

    const opp::Ln<ComputeT> op;
    const lnInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region LShift
template <PixelType T>
ImageView<T> &ImageView<T>::LShift(uint aConst, ImageView<T> &aDst)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using LShiftSrcC = SrcFunctor<1, T, ComputeT, T, opp::LShift<ComputeT>, RoundingMode::None>;

    const opp::LShift<ComputeT> op(aConst);
    const LShiftSrcC functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LShift(uint aConst)
    requires RealIntVector<T>
{
    using ComputeT = T;

    using LShiftInplaceC = InplaceFunctor<1, ComputeT, T, opp::LShift<ComputeT>, RoundingMode::None>;

    const opp::LShift<ComputeT> op(aConst);
    const LShiftInplaceC functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Or
template <PixelType T>
ImageView<T> &ImageView<T>::Or(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using OrSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::Or<ComputeT>, RoundingMode::None>;

    const opp::Or<ComputeT> op;
    const OrSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const T &aConst, ImageView<T> &aDst)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using OrSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::Or<ComputeT>, RoundingMode::None>;

    const opp::Or<ComputeT> op;
    const OrSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const ImageView<T> &aSrc2)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = T;

    using OrInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::Or<ComputeT>, RoundingMode::None>;

    const opp::Or<ComputeT> op;
    const OrInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const T &aConst)
    requires RealIntVector<T>
{
    using ComputeT = T;

    using OrInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::Or<ComputeT>, RoundingMode::None>;

    const opp::Or<ComputeT> op;
    const OrInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Sqr
template <PixelType T>
ImageView<T> &ImageView<T>::Sqr(ImageView<T> &aDst)
    requires RealOrComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using ComputeT = default_compute_type_for_t<T>;

    using sqrSrc = SrcFunctor<1, T, ComputeT, T, opp::Sqr<ComputeT>, RoundingMode::None>;

    const opp::Sqr<ComputeT> op;
    const sqrSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sqr()
    requires RealOrComplexVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using sqrInplace = InplaceFunctor<1, ComputeT, T, opp::Sqr<ComputeT>, RoundingMode::None>;

    const opp::Sqr<ComputeT> op;
    const sqrInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Sqrt
template <PixelType T>
ImageView<T> &ImageView<T>::Sqrt(ImageView<T> &aDst)
    requires RealOrComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using ComputeT = default_compute_type_for_t<T>;

    using sqrtSrc = SrcFunctor<1, T, ComputeT, T, opp::Sqrt<ComputeT>, RoundingMode::None>;

    const opp::Sqrt<ComputeT> op;
    const sqrtSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sqrt()
    requires RealOrComplexVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using sqrtInplace = InplaceFunctor<1, ComputeT, T, opp::Sqrt<ComputeT>, RoundingMode::None>;

    const opp::Sqrt<ComputeT> op;
    const sqrtInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region RShift
template <PixelType T>
ImageView<T> &ImageView<T>::RShift(uint aConst, ImageView<T> &aDst)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using RShiftSrcC = SrcFunctor<1, T, ComputeT, T, opp::RShift<ComputeT>, RoundingMode::None>;

    const opp::RShift<ComputeT> op(aConst);
    const RShiftSrcC functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RShift(uint aConst)
    requires RealIntVector<T>
{
    using ComputeT = T;

    using RShiftInplaceC = InplaceFunctor<1, ComputeT, T, opp::RShift<ComputeT>, RoundingMode::None>;

    const opp::RShift<ComputeT> op(aConst);
    const RShiftInplaceC functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Xor
template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using XorSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::Xor<ComputeT>, RoundingMode::None>;

    const opp::Xor<ComputeT> op;
    const XorSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const T &aConst, ImageView<T> &aDst)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using XorSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::Xor<ComputeT>, RoundingMode::None>;

    const opp::Xor<ComputeT> op;
    const XorSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const ImageView<T> &aSrc2)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = T;

    using XorInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::Xor<ComputeT>, RoundingMode::None>;

    const opp::Xor<ComputeT> op;
    const XorInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const T &aConst)
    requires RealIntVector<T>
{
    using ComputeT = T;

    using XorInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::Xor<ComputeT>, RoundingMode::None>;

    const opp::Xor<ComputeT> op;
    const XorInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion

#pragma region AlphaPremul
template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(ImageView<T> &aDst)
    requires FourChannelNoAlpha<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;
    using alphaPremulSrc =
        SrcFunctor<1, T, ComputeT, T, opp::AlphaPremul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    // NOLINTNEXTLINE(misc-const-correctness)
    remove_vector_t<ComputeT> alphaScaleVal = static_cast<remove_vector_t<ComputeT>>(1);
    if constexpr (RealIntVector<T>)
    {
        alphaScaleVal = static_cast<remove_vector_t<ComputeT>>(numeric_limits<remove_vector_t<T>>::max());
    }

    const opp::AlphaPremul<ComputeT> op(alphaScaleVal);

    const alphaPremulSrc functor(PointerRoi(), Pitch(), op);
    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul()
    requires FourChannelNoAlpha<T>
{
    using ComputeT = default_compute_type_for_t<T>;
    using alphaPremulInplace =
        InplaceFunctor<1, ComputeT, T, opp::AlphaPremul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    // NOLINTNEXTLINE(misc-const-correctness)
    remove_vector_t<ComputeT> alphaScaleVal = static_cast<remove_vector_t<ComputeT>>(1);
    if constexpr (RealIntVector<T>)
    {
        alphaScaleVal = static_cast<remove_vector_t<ComputeT>>(numeric_limits<remove_vector_t<T>>::max());
    }

    const opp::AlphaPremul<ComputeT> op(alphaScaleVal);

    const alphaPremulInplace functor(op);
    forEachPixel(*this, functor);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst)
    requires RealFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;
    const T alphaVec(aAlpha);

    using mulSrcC = SrcConstantFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::None>;

    const opp::Mul<ComputeT> op;
    const mulSrcC functor(PointerRoi(), Pitch(), alphaVec, op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst)
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;
    const T alphaVec(aAlpha);
    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    using mulSrcCScale =
        SrcConstantScaleFunctor<1, T, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulSrcCScale functor(PointerRoi(), Pitch(), alphaVec, op, scaleFactorFloat);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha)
    requires RealFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;
    const T alphaVec(aAlpha);

    using mulInplaceC = InplaceConstantFunctor<1, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::None>;

    const opp::Mul<ComputeT> op;
    const mulInplaceC functor(alphaVec, op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha)
    requires RealIntVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;
    const T alphaVec(aAlpha);
    const float scaleFactorFloat = 1.0f / static_cast<float>(numeric_limits<remove_vector_t<T>>::max());

    using mulInplaceCScale =
        InplaceConstantScaleFunctor<1, ComputeT, T, opp::Mul<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Mul<ComputeT> op;
    const mulInplaceCScale functor(alphaVec, op, scaleFactorFloat);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion

#pragma region AlphaComp
template <PixelType T>
ImageView<T> &ImageView<T>::AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, AlphaCompositionOp aAlphaOp)
    requires(!FourChannelAlpha<T>) && RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    constexpr size_t TupelSize = 1;
    using ComputeT             = default_compute_type_for_t<T>;
    using SrcT                 = T;
    using DstT                 = T;

    if constexpr (RealIntVector<SrcT>)
    {
        constexpr remove_vector_t<ComputeT> alphaScaleVal =
            static_cast<remove_vector_t<ComputeT>>(numeric_limits<remove_vector_t<SrcT>>::max());
        constexpr remove_vector_t<ComputeT> alphaScaleValInv =
            static_cast<remove_vector_t<ComputeT>>(1) / alphaScaleVal;

        switch (aAlphaOp)
        {
            case opp::AlphaCompositionOp::Over:
            {
                using AlphaCompOp = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                               opp::AlphaCompositionOp::Over>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesAwayFromZero>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::In:
            {
                using AlphaCompOp =
                    opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv, opp::AlphaCompositionOp::In>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesAwayFromZero>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::Out:
            {
                using AlphaCompOp =
                    opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv, opp::AlphaCompositionOp::Out>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesAwayFromZero>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::ATop:
            {
                using AlphaCompOp = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                               opp::AlphaCompositionOp::ATop>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesAwayFromZero>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::XOr:
            {
                using AlphaCompOp =
                    opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv, opp::AlphaCompositionOp::XOr>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesAwayFromZero>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::Plus:
            {
                using AlphaCompOp = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                               opp::AlphaCompositionOp::Plus>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesAwayFromZero>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::OverPremul:
            {
                using AlphaCompOp = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                               opp::AlphaCompositionOp::OverPremul>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesAwayFromZero>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::InPremul:
            {
                using AlphaCompOp = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                               opp::AlphaCompositionOp::InPremul>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesAwayFromZero>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::OutPremul:
            {
                using AlphaCompOp = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                               opp::AlphaCompositionOp::OutPremul>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesAwayFromZero>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::ATopPremul:
            {
                using AlphaCompOp = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                               opp::AlphaCompositionOp::ATopPremul>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesAwayFromZero>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::XOrPremul:
            {
                using AlphaCompOp = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                               opp::AlphaCompositionOp::XOrPremul>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesAwayFromZero>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::PlusPremul:
            {
                using AlphaCompOp = opp::AlphaCompositionScale<ComputeT, alphaScaleVal, alphaScaleValInv,
                                                               opp::AlphaCompositionOp::PlusPremul>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesAwayFromZero>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
        }
    }
    else
    {
        switch (aAlphaOp)
        {
            case opp::AlphaCompositionOp::Over:
            {
                using AlphaCompOp     = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::Over>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::In:
            {
                using AlphaCompOp     = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::In>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::Out:
            {
                using AlphaCompOp     = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::Out>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::ATop:
            {
                using AlphaCompOp     = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::ATop>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::XOr:
            {
                using AlphaCompOp     = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::XOr>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::Plus:
            {
                using AlphaCompOp     = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::Plus>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::OverPremul:
            {
                using AlphaCompOp     = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::OverPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::InPremul:
            {
                using AlphaCompOp     = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::InPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::OutPremul:
            {
                using AlphaCompOp     = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::OutPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::ATopPremul:
            {
                using AlphaCompOp     = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::ATopPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::XOrPremul:
            {
                using AlphaCompOp     = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::XOrPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case opp::AlphaCompositionOp::PlusPremul:
            {
                using AlphaCompOp     = opp::AlphaComposition<ComputeT, opp::AlphaCompositionOp::PlusPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
        }
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, remove_vector_t<T> aAlpha1,
                                      remove_vector_t<T> aAlpha2, AlphaCompositionOp aAlphaOp)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    constexpr size_t TupelSize = 1;
    using ComputeT             = default_compute_type_for_t<T>;
    using SrcT                 = T;
    using DstT                 = T;

    // NOLINTBEGIN(misc-const-correctness)
    remove_vector_t<ComputeT> alpha1 = static_cast<remove_vector_t<ComputeT>>(aAlpha1);
    remove_vector_t<ComputeT> alpha2 = static_cast<remove_vector_t<ComputeT>>(aAlpha2);
    // NOLINTEND(misc-const-correctness)

    if constexpr (RealIntVector<SrcT>)
    {
        alpha1 /= static_cast<remove_vector_t<ComputeT>>(numeric_limits<remove_vector_t<SrcT>>::max());
        alpha2 /= static_cast<remove_vector_t<ComputeT>>(numeric_limits<remove_vector_t<SrcT>>::max());
    }

    constexpr RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero;

    switch (aAlphaOp)
    {
        case opp::AlphaCompositionOp::Over:
        {
            using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::Over>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::AlphaCompositionOp::In:
        {
            using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::In>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::AlphaCompositionOp::Out:
        {
            using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::Out>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::AlphaCompositionOp::ATop:
        {
            using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::ATop>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::AlphaCompositionOp::XOr:
        {
            using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::XOr>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::AlphaCompositionOp::Plus:
        {
            using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::Plus>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::AlphaCompositionOp::OverPremul:
        {
            using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::OverPremul>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::AlphaCompositionOp::InPremul:
        {
            using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::InPremul>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::AlphaCompositionOp::OutPremul:
        {
            using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::OutPremul>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::AlphaCompositionOp::ATopPremul:
        {
            using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::ATopPremul>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::AlphaCompositionOp::XOrPremul:
        {
            using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::XOrPremul>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case opp::AlphaCompositionOp::PlusPremul:
        {
            using AlphaCompOp     = opp::AlphaCompositionC<ComputeT, opp::AlphaCompositionOp::PlusPremul>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
    }

    return aDst;
}
#pragma endregion

#pragma region Complex
template <PixelType T>
ImageView<T> &ImageView<T>::ConjMul(const ImageView<T> &aSrc2, ImageView<T> &aDst)
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using conjMulSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, opp::ConjMul<ComputeT>, RoundingMode::None>;

    const opp::ConjMul<ComputeT> op;
    const conjMulSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ConjMul(const ImageView<T> &aSrc2)
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using conjMulInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, opp::ConjMul<ComputeT>, RoundingMode::None>;

    const opp::ConjMul<ComputeT> op;
    const conjMulInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Conj(ImageView<T> &aDst)
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using conjSrc = SrcFunctor<1, T, T, T, opp::Conj<T>, RoundingMode::None>;

    const opp::Conj<T> op;
    const conjSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Conj()
    requires ComplexVector<T>
{
    using conjInplace = InplaceFunctor<1, T, T, opp::Conj<T>, RoundingMode::None>;

    const opp::Conj<T> op;
    const conjInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Magnitude(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst)
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using ComputeT = default_compute_type_for_t<T>;
    using DstT     = same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>;

    using magnitudeSrc =
        SrcFunctor<1, SrcT, ComputeT, DstT, opp::Magnitude<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Magnitude<ComputeT> op;

    const magnitudeSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::MagnitudeSqr(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst)
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using ComputeT = default_compute_type_for_t<T>;
    using DstT     = same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>;

    using magnitudeSqrSrc =
        SrcFunctor<1, SrcT, ComputeT, DstT, opp::MagnitudeSqr<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::MagnitudeSqr<ComputeT> op;

    const magnitudeSqrSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Angle(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst)
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using ComputeT = default_compute_type_for_t<T>;
    using DstT     = same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>;

    using angleSrc = SrcFunctor<1, SrcT, ComputeT, DstT, opp::Angle<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Angle<ComputeT> op;

    const angleSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Real(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst)
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using ComputeT = T;
    using DstT     = same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>;

    using realSrc = SrcFunctor<1, SrcT, ComputeT, DstT, opp::Real<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Real<ComputeT> op;

    const realSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Imag(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst)
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using ComputeT = T;
    using DstT     = same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>;

    using imagSrc = SrcFunctor<1, SrcT, ComputeT, DstT, opp::Imag<ComputeT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::Imag<ComputeT> op;

    const imagSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &ImageView<T>::MakeComplex(
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst)
    requires RealSignedVector<T> && (!FourChannelAlpha<T>)
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using ComputeT = T;
    using DstT     = same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>;

    using makeComplexSrc =
        SrcFunctor<1, SrcT, ComputeT, DstT, opp::MakeComplex<DstT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::MakeComplex<DstT> op;

    const makeComplexSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &ImageView<T>::MakeComplex(
    const ImageView<T> &aSrcImag,
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst)
    requires RealSignedVector<T> && (!FourChannelAlpha<T>)
{
    checkSameSize(ROI(), aSrcImag.ROI());
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using ComputeT = T;
    using DstT     = same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>;

    using makeComplexSrcSrc =
        SrcSrcFunctor<1, SrcT, ComputeT, DstT, opp::MakeComplex<DstT>, RoundingMode::NearestTiesAwayFromZero>;

    const opp::MakeComplex<DstT> op;

    const makeComplexSrcSrc functor(PointerRoi(), Pitch(), aSrcImag.PointerRoi(), aSrcImag.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}
#pragma endregion
} // namespace opp::image::cpuSimple