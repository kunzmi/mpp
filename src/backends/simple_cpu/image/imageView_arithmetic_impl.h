#pragma once
#include <backends/simple_cpu/image/addSquareProductWeightedOutputType.h>
#include <backends/simple_cpu/image/forEachPixel.h>
#include <backends/simple_cpu/image/forEachPixelMasked.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/ternary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/image/border.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/constantFunctor.h>
#include <common/image/functors/inplaceConstantFunctor.h>
#include <common/image/functors/inplaceConstantScaleFunctor.h>
#include <common/image/functors/inplaceDevConstantFunctor.h>
#include <common/image/functors/inplaceDevConstantScaleFunctor.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/inplaceSrcScaleFunctor.h>
#include <common/image/functors/inplaceSrcSrcFunctor.h>
#include <common/image/functors/interpolator.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcConstantScaleFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
#include <common/image/functors/srcDevConstantScaleFunctor.h>
#include <common/image/functors/srcFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/functors/srcSrcScaleFunctor.h>
#include <common/image/functors/transformer.h>
#include <common/image/functors/transformerFunctor.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/roiException.h>
#include <common/image/size2D.h>
#include <common/image/sizePitched.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/numeric_limits.h>
#include <common/safeCast.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

// NOLINTBEGIN(clang-diagnostic-double-promotion)

namespace mpp::image::cpuSimple
{
#pragma region Add
template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, RoundingMode::None>;

    const mpp::Add<ComputeT> op;
    const addSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using addSrcSrcScale =
            SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;
        const addSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aDst, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using addSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Add<ComputeT> op;
        const addSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, ImageView<T> &aDst) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, RoundingMode::None>;

    const mpp::Add<ComputeT> op;
    const addSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, ImageView<T> &aDst, int aScaleFactor) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using addSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aDst, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using addSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Add<ComputeT> op;

        const addSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, RoundingMode::None>;

    const mpp::Add<ComputeT> op;
    const addInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const ImageView<T> &aSrc2, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using addInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(*this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using addInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Add<ComputeT> op;

        const addInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(*this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::Add<ComputeT>, RoundingMode::None>;

    const mpp::Add<ComputeT> op;
    const addInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Add(const T &aConst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);

        using addInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(*this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);

        using addInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Add<ComputeT> op;

        const addInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(*this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                      const ImageView<Pixel8uC1> &aMask) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, RoundingMode::None>;

    const mpp::Add<ComputeT> op;
    const addSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using addSrcSrcScale =
            SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using addSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Add<ComputeT> op;

        const addSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, RoundingMode::None>;

    const mpp::Add<ComputeT> op;
    const addSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using addSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using addSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Add<ComputeT> op;

        const addSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, RoundingMode::None>;

    const mpp::Add<ComputeT> op;
    const addInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using addInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using addInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Add<ComputeT> op;

        const addInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Add cannot produce non-integers in computations with ints:
    using addInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::Add<ComputeT>, RoundingMode::None>;

    const mpp::Add<ComputeT> op;
    const addInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AddMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);

        using addInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Add<ComputeT> op;

        const addInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);

        using addInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Add<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Add<ComputeT> op;

        const addInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    return *this;
}
#pragma endregion

#pragma region Sub
template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, RoundingMode::None>;

    const mpp::Sub<ComputeT> op;
    const subSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using subSrcSrcScale =
            SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aDst, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using subSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Sub<ComputeT> op;

        const subSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, ImageView<T> &aDst) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, RoundingMode::None>;

    const mpp::Sub<ComputeT> op;
    const subSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, ImageView<T> &aDst, int aScaleFactor) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using subSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aDst, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using subSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Sub<ComputeT> op;

        const subSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, RoundingMode::None>;

    const mpp::Sub<ComputeT> op;
    const subInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const ImageView<T> &aSrc2, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;
        const subInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(*this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Sub<ComputeT> op;
        const subInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(*this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::Sub<ComputeT>, RoundingMode::None>;

    const mpp::Sub<ComputeT> op;
    const subInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sub(const T &aConst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(*this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Sub<ComputeT> op;

        const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(*this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const ImageView<T> &aSrc2)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::SubInv<ComputeT>, RoundingMode::None>;

    const mpp::SubInv<ComputeT> op;
    const subInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const ImageView<T> &aSrc2, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::SubInv<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(*this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::SubInv<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(*this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const T &aConst)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::SubInv<ComputeT>, RoundingMode::None>;

    const mpp::SubInv<ComputeT> op;
    const subInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInv(const T &aConst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::SubInv<ComputeT>, ScalerT,
                                                             RoundingMode::NearestTiesToEven>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(*this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::SubInv<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(*this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                      const ImageView<Pixel8uC1> &aMask) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, RoundingMode::None>;

    const mpp::Sub<ComputeT> op;
    const subSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using subSrcSrcScale =
            SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;
        const subSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using subSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Sub<ComputeT> op;
        const subSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, RoundingMode::None>;

    const mpp::Sub<ComputeT> op;
    const subSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using subSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;
        const subSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using subSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Sub<ComputeT> op;
        const subSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, RoundingMode::None>;

    const mpp::Sub<ComputeT> op;
    const subInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;
        const subInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Sub<ComputeT> op;
        const subInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::Sub<ComputeT>, RoundingMode::None>;

    const mpp::Sub<ComputeT> op;
    const subInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Sub<ComputeT> op;

        const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Sub<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Sub<ComputeT> op;

        const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::SubInv<ComputeT>, RoundingMode::None>;

    const mpp::SubInv<ComputeT> op;
    const subInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::SubInv<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::SubInv<ComputeT> op;
        const subInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::SubInv<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::SubInv<ComputeT> op;
        const subInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Sub cannot produce non-integers in computations with ints:
    using subInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::SubInv<ComputeT>, RoundingMode::None>;

    const mpp::SubInv<ComputeT> op;
    const subInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::SubInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::SubInv<ComputeT>, ScalerT,
                                                             RoundingMode::NearestTiesToEven>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using subInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::SubInv<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::SubInv<ComputeT> op;

        const subInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    return *this;
}
#pragma endregion

#pragma region Mul
template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, RoundingMode::None>;

    const mpp::Mul<ComputeT> op;
    const mulSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    if (scaleFactorFloat == 1.0)
    {
        using ComputeT = default_ext_int_compute_type_for_t<T>;

        using mulSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

        forEachPixel(aDst, functor);
    }
    else
    {
        using ComputeT = default_ext_int_compute_type_for_t<T>;

        if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
        {
            using ScalerT = mpp::Scale<ComputeT, false>;
            const ScalerT scaler(scaleFactorFloat);
            using mulSrcSrcScale =
                SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

            const mpp::Mul<ComputeT> op;
            const mulSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

            forEachPixel(aDst, functor);
        }
        else
        {
            // Scaler performs NearestTiesToEven rounding:
            using ScalerT = mpp::Scale<ComputeT, true>;
            const ScalerT scaler(scaleFactorFloat);
            using mulSrcSrcScale =
                SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

            const mpp::Mul<ComputeT> op;
            const mulSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

            forEachPixel(aDst, functor);
        }
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, ImageView<T> &aDst) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, RoundingMode::None>;

    const mpp::Mul<ComputeT> op;
    const mulSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, ImageView<T> &aDst, int aScaleFactor) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aDst, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, RoundingMode::None>;

    const mpp::Mul<ComputeT> op;
    const mulInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const ImageView<T> &aSrc2, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;

        const mulInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(*this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;

        const mulInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(*this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_ext_int_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, RoundingMode::None>;

    const mpp::Mul<ComputeT> op;
    const mulInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Mul(const T &aConst, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;

        const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(*this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;

        const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(*this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                      const ImageView<Pixel8uC1> &aMask) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, RoundingMode::None>;

    const mpp::Mul<ComputeT> op;
    const mulSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcSrcScale =
            SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, RoundingMode::None>;

    const mpp::Mul<ComputeT> op;
    const mulSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, RoundingMode::None>;

    const mpp::Mul<ComputeT> op;
    const mulInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_ext_int_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using mulInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, RoundingMode::None>;

    const mpp::Mul<ComputeT> op;
    const mulInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor)
    requires RealOrComplexIntVector<T>
{
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if (scaleFactorFloat >= 1.0 || RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;

        const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    else
    {
        // Scaler performs NearestTiesToEven rounding:
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;

        const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    return *this;
}
#pragma endregion

#pragma region MulScale
template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcSrcScale =
            SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aDst, functor);
    }
    else
    {
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const T &aConst, ImageView<T> &aDst) const
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aDst.ROI());

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aDst, functor);
    }
    else
    {
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const ImageView<T> &aSrc2)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aSrc2.ROI());

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(*this, functor);
    }
    else
    {
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(*this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScale(const T &aConst)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(*this, functor);
    }
    else
    {
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(*this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScaleMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                           const ImageView<Pixel8uC1> &aMask) const
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcSrcScale =
            SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    else
    {
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScaleMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aDst.ROI());

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    else
    {
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScaleMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    checkSameSize(ROI(), aSrc2.ROI());

    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    else
    {
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceSrcScale =
            InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MulScaleMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, ushort>
{
    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    else
    {
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);

        forEachPixel(aMask, *this, functor);
    }
    return *this;
}
#pragma endregion

#pragma region Div
template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, RoundingMode::None>;

    const mpp::Div<ComputeT> op;
    const divSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, ImageView<T> &aDst, int aScaleFactor,
                                RoundingMode aRoundingMode) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divSrcSrcScale =
                    SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT> op;
                const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT> op;
                const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divSrcSrcScale =
                    SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::TowardZero>;
                const mpp::Div<ComputeT> op;
                const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                          RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT> op;
                const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                          RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT> op;
                const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        if (scaleFactorFloat < 1)
        {
            if (aRoundingMode == RoundingMode::NearestTiesToEven)
            {
                using ScalerT = mpp::Scale<ComputeT, true>;
                const ScalerT scaler(scaleFactorFloat);
                using divSrcSrcScale =
                    SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::None>;
                const mpp::Div<ComputeT> op;
                const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aDst, functor);
            }
            else
            {
                throw INVALIDARGUMENT(aRoundingMode,
                                      "Unsupported rounding mode: "
                                          << aRoundingMode << ". Only rounding mode " << RoundingMode::NearestTiesToEven
                                          << " is supported for this source data type and scaling factor < 1.");
            }
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                    using divSrcSrcScale = SrcSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const OpT op(scaleFactorFloat);
                    const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aDst, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    using divSrcSrcScale = SrcSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const OpT op(scaleFactorFloat);
                    const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                    using divSrcSrcScale = SrcSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const OpT op(scaleFactorFloat);
                    const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                    using divSrcSrcScale = SrcSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const OpT op(scaleFactorFloat);
                    const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                    using divSrcSrcScale = SrcSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const OpT op(scaleFactorFloat);
                    const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aDst, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, ImageView<T> &aDst) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, RoundingMode::None>;

    const mpp::Div<ComputeT> op;
    const divSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, ImageView<T> &aDst, int aScaleFactor, RoundingMode aRoundingMode) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divSrcCScale = SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                             RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT> op;
                const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcCScale = SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                             RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT> op;
                const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divSrcCScale =
                    SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::TowardZero>;
                const mpp::Div<ComputeT> op;
                const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcCScale = SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                             RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT> op;
                const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcCScale = SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                             RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT> op;
                const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        if (scaleFactorFloat < 1)
        {
            if (aRoundingMode == RoundingMode::NearestTiesToEven)
            {
                using ScalerT = mpp::Scale<ComputeT, true>;
                const ScalerT scaler(scaleFactorFloat);
                using divSrcCScale =
                    SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::None>;
                const mpp::Div<ComputeT> op;
                const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aDst, functor);
            }
            else
            {
                throw INVALIDARGUMENT(aRoundingMode,
                                      "Unsupported rounding mode: "
                                          << aRoundingMode << ". Only rounding mode " << RoundingMode::NearestTiesToEven
                                          << " is supported for this source data type and scaling factor < 1.");
            }
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                    const OpT op(scaleFactorFloat);
                    using divSrcCScale = SrcConstantFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;

                    const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);
                    forEachPixel(aDst, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    const OpT op(scaleFactorFloat);
                    using divSrcCScale = SrcConstantFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;

                    const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);
                    forEachPixel(aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                    const OpT op(scaleFactorFloat);
                    using divSrcCScale = SrcConstantFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;

                    const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);
                    forEachPixel(aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divSrcCScale = SrcConstantFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;

                    const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);
                    forEachPixel(aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divSrcCScale = SrcConstantFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;

                    const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);
                    forEachPixel(aDst, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, RoundingMode::None>;

    const mpp::Div<ComputeT> op;
    const divInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const ImageView<T> &aSrc2, int aScaleFactor, RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                  RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                  RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::TowardZero>;
                const mpp::Div<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                  RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                  RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        if (scaleFactorFloat < 1)
        {
            if (aRoundingMode == RoundingMode::NearestTiesToEven)
            {
                using ScalerT = mpp::Scale<ComputeT, true>;
                const ScalerT scaler(scaleFactorFloat);
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::None>;
                const mpp::Div<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(*this, functor);
            }
            else
            {
                throw INVALIDARGUMENT(aRoundingMode,
                                      "Unsupported rounding mode: "
                                          << aRoundingMode << ". Only rounding mode " << RoundingMode::NearestTiesToEven
                                          << " is supported for this source data type and scaling factor < 1.");
            }
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(*this, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::Div<ComputeT>, RoundingMode::None>;

    const mpp::Div<ComputeT> op;
    const divInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Div(const T &aConst, int aScaleFactor, RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                     RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                     RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::TowardZero>;
                const mpp::Div<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                     RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                     RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        if (scaleFactorFloat < 1)
        {
            if (aRoundingMode == RoundingMode::NearestTiesToEven)
            {
                using ScalerT = mpp::Scale<ComputeT, true>;
                const ScalerT scaler(scaleFactorFloat);
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::None>;
                const mpp::Div<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(*this, functor);
            }
            else
            {
                throw INVALIDARGUMENT(aRoundingMode,
                                      "Unsupported rounding mode: "
                                          << aRoundingMode << ". Only rounding mode " << RoundingMode::NearestTiesToEven
                                          << " is supported for this source data type and scaling factor < 1.");
            }
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(*this, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const ImageView<T> &aSrc2)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, RoundingMode::None>;

    const mpp::DivInv<ComputeT> op;
    const divInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const ImageView<T> &aSrc2, int aScaleFactor, RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                  RoundingMode::NearestTiesToEven>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                  RoundingMode::NearestTiesAwayFromZero>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT, RoundingMode::TowardZero>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                  RoundingMode::TowardNegativeInfinity>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                  RoundingMode::TowardPositiveInfinity>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        if (scaleFactorFloat < 1)
        {
            if (aRoundingMode == RoundingMode::NearestTiesToEven)
            {
                using ScalerT = mpp::Scale<ComputeT, true>;
                const ScalerT scaler(scaleFactorFloat);
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT, RoundingMode::None>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(*this, functor);
            }
            else
            {
                throw INVALIDARGUMENT(aRoundingMode,
                                      "Unsupported rounding mode: "
                                          << aRoundingMode << ". Only rounding mode " << RoundingMode::NearestTiesToEven
                                          << " is supported for this source data type and scaling factor < 1.");
            }
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(*this, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const T &aConst)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, RoundingMode::None>;

    const mpp::DivInv<ComputeT> op;
    const divInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInv(const T &aConst, int aScaleFactor, RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                     RoundingMode::NearestTiesToEven>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                     RoundingMode::NearestTiesAwayFromZero>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                     RoundingMode::TowardZero>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                     RoundingMode::TowardNegativeInfinity>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                     RoundingMode::TowardPositiveInfinity>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(*this, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        if (scaleFactorFloat < 1)
        {
            if (aRoundingMode == RoundingMode::NearestTiesToEven)
            {
                using ScalerT = mpp::Scale<ComputeT, true>;
                const ScalerT scaler(scaleFactorFloat);
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT, RoundingMode::None>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(*this, functor);
            }
            else
            {
                throw INVALIDARGUMENT(aRoundingMode,
                                      "Unsupported rounding mode: "
                                          << aRoundingMode << ". Only rounding mode " << RoundingMode::NearestTiesToEven
                                          << " is supported for this source data type and scaling factor < 1.");
            }
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(*this, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(*this, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst,
                                      const ImageView<Pixel8uC1> &aMask) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, RoundingMode::None>;

    const mpp::Div<ComputeT> op;
    const divSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const ImageView<T> &aSrc2, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, RoundingMode aRoundingMode) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divSrcSrcScale =
                    SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT> op;
                const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, aDst, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                          RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT> op;
                const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divSrcSrcScale =
                    SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::TowardZero>;
                const mpp::Div<ComputeT> op;
                const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                          RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT> op;
                const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcSrcScale = SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                          RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT> op;
                const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        if (scaleFactorFloat < 1)
        {
            if (aRoundingMode == RoundingMode::NearestTiesToEven)
            {
                using ScalerT = mpp::Scale<ComputeT, true>;
                const ScalerT scaler(scaleFactorFloat);
                using divSrcSrcScale =
                    SrcSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::None>;
                const mpp::Div<ComputeT> op;
                const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, aDst, functor);
            }
            else
            {
                throw INVALIDARGUMENT(aRoundingMode,
                                      "Unsupported rounding mode: "
                                          << aRoundingMode << ". Only rounding mode " << RoundingMode::NearestTiesToEven
                                          << " is supported for this source data type and scaling factor < 1.");
            }
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                    using divSrcSrcScale = SrcSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const OpT op(scaleFactorFloat);
                    const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, aDst, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    using divSrcSrcScale = SrcSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const OpT op(scaleFactorFloat);
                    const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                    using divSrcSrcScale = SrcSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const OpT op(scaleFactorFloat);
                    const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                    using divSrcSrcScale = SrcSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const OpT op(scaleFactorFloat);
                    const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using OpT            = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                    using divSrcSrcScale = SrcSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const OpT op(scaleFactorFloat);
                    const divSrcSrcScale functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, aDst, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask) const
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, RoundingMode::None>;

    const mpp::Div<ComputeT> op;
    const divSrcC functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const T &aConst, ImageView<T> &aDst, const ImageView<Pixel8uC1> &aMask,
                                      int aScaleFactor, RoundingMode aRoundingMode) const
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divSrcCScale = SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                             RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT> op;
                const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, aDst, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divSrcCScale = SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                             RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT> op;
                const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divSrcCScale =
                    SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::TowardZero>;
                const mpp::Div<ComputeT> op;
                const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divSrcCScale = SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                             RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT> op;
                const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, aDst, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divSrcCScale = SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                             RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT> op;
                const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, aDst, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        if (scaleFactorFloat < 1)
        {
            if (aRoundingMode == RoundingMode::NearestTiesToEven)
            {
                using ScalerT = mpp::Scale<ComputeT, true>;
                const ScalerT scaler(scaleFactorFloat);
                using divSrcCScale =
                    SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::None>;
                const mpp::Div<ComputeT> op;
                const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, aDst, functor);
            }
            else
            {
                throw INVALIDARGUMENT(aRoundingMode,
                                      "Unsupported rounding mode: "
                                          << aRoundingMode << ". Only rounding mode " << RoundingMode::NearestTiesToEven
                                          << " is supported for this source data type and scaling factor < 1.");
            }
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                    const OpT op(scaleFactorFloat);
                    using divSrcCScale = SrcConstantFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;

                    const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, aDst, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    const OpT op(scaleFactorFloat);
                    using divSrcCScale = SrcConstantFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;

                    const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                    const OpT op(scaleFactorFloat);
                    using divSrcCScale = SrcConstantFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;

                    const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divSrcCScale = SrcConstantFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;

                    const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, aDst, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divSrcCScale = SrcConstantFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;

                    const divSrcCScale functor(PointerRoi(), Pitch(), static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, aDst, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, RoundingMode::None>;

    const mpp::Div<ComputeT> op;
    const divInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                  RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                  RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::TowardZero>;
                const mpp::Div<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                  RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                  RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        if (scaleFactorFloat < 1)
        {
            if (aRoundingMode == RoundingMode::NearestTiesToEven)
            {
                using ScalerT = mpp::Scale<ComputeT, true>;
                const ScalerT scaler(scaleFactorFloat);
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::None>;
                const mpp::Div<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            else
            {
                throw INVALIDARGUMENT(aRoundingMode,
                                      "Unsupported rounding mode: "
                                          << aRoundingMode << ". Only rounding mode " << RoundingMode::NearestTiesToEven
                                          << " is supported for this source data type and scaling factor < 1.");
            }
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::Div<ComputeT>, RoundingMode::None>;

    const mpp::Div<ComputeT> op;
    const divInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                      RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                     RoundingMode::NearestTiesToEven>;
                const mpp::Div<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                     RoundingMode::NearestTiesAwayFromZero>;
                const mpp::Div<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::TowardZero>;
                const mpp::Div<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                     RoundingMode::TowardNegativeInfinity>;
                const mpp::Div<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Div<ComputeT>, ScalerT,
                                                                     RoundingMode::TowardPositiveInfinity>;
                const mpp::Div<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        if (scaleFactorFloat < 1)
        {
            if (aRoundingMode == RoundingMode::NearestTiesToEven)
            {
                using ScalerT = mpp::Scale<ComputeT, true>;
                const ScalerT scaler(scaleFactorFloat);
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Div<ComputeT>, ScalerT, RoundingMode::None>;
                const mpp::Div<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            else
            {
                throw INVALIDARGUMENT(aRoundingMode,
                                      "Unsupported rounding mode: "
                                          << aRoundingMode << ". Only rounding mode " << RoundingMode::NearestTiesToEven
                                          << " is supported for this source data type and scaling factor < 1.");
            }
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using OpT = DivIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using divInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, RoundingMode::None>;

    const mpp::DivInv<ComputeT> op;
    const divInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInvMasked(const ImageView<T> &aSrc2, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                         RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                  RoundingMode::NearestTiesToEven>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                  RoundingMode::NearestTiesAwayFromZero>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT, RoundingMode::TowardZero>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                  RoundingMode::TowardNegativeInfinity>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceSrcScale = InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                  RoundingMode::TowardPositiveInfinity>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        if (scaleFactorFloat < 1)
        {
            if (aRoundingMode == RoundingMode::NearestTiesToEven)
            {
                using ScalerT = mpp::Scale<ComputeT, true>;
                const ScalerT scaler(scaleFactorFloat);
                using divInplaceSrcScale =
                    InplaceSrcScaleFunctor<1, T, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT, RoundingMode::None>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            else
            {
                throw INVALIDARGUMENT(aRoundingMode,
                                      "Unsupported rounding mode: "
                                          << aRoundingMode << ". Only rounding mode " << RoundingMode::NearestTiesToEven
                                          << " is supported for this source data type and scaling factor < 1.");
            }
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceSrcScale = InplaceSrcFunctor<1, T, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceSrcScale functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask)
    requires RealOrComplexFloatingVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    // set to roundingmode NONE, because Div cannot produce non-integers in computations with ints:
    using divInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, RoundingMode::None>;

    const mpp::DivInv<ComputeT> op;
    const divInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(aMask, *this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::DivInvMasked(const T &aConst, const ImageView<Pixel8uC1> &aMask, int aScaleFactor,
                                         RoundingMode aRoundingMode)
    requires RealOrComplexIntVector<T>
{
    const double scaleFactorFloat = GetScaleFactor(aScaleFactor);

    using ComputeT = default_compute_type_for_t<T>;

    if constexpr (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        switch (aRoundingMode)
        {
            case mpp::RoundingMode::NearestTiesToEven:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                     RoundingMode::NearestTiesToEven>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::NearestTiesAwayFromZero:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                     RoundingMode::NearestTiesAwayFromZero>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::TowardZero:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                     RoundingMode::TowardZero>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::TowardNegativeInfinity:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                     RoundingMode::TowardNegativeInfinity>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            case mpp::RoundingMode::TowardPositiveInfinity:
            {
                using divInplaceCScale = InplaceConstantScaleFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT,
                                                                     RoundingMode::TowardPositiveInfinity>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            break;
            default:
                throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
        }
    }
    else
    {
        if (scaleFactorFloat < 1)
        {
            if (aRoundingMode == RoundingMode::NearestTiesToEven)
            {
                using ScalerT = mpp::Scale<ComputeT, true>;
                const ScalerT scaler(scaleFactorFloat);
                using divInplaceCScale =
                    InplaceConstantScaleFunctor<1, ComputeT, T, mpp::DivInv<ComputeT>, ScalerT, RoundingMode::None>;
                const mpp::DivInv<ComputeT> op;
                const divInplaceCScale functor(static_cast<ComputeT>(aConst), op, scaler);
                forEachPixel(aMask, *this, functor);
            }
            else
            {
                throw INVALIDARGUMENT(aRoundingMode,
                                      "Unsupported rounding mode: "
                                          << aRoundingMode << ". Only rounding mode " << RoundingMode::NearestTiesToEven
                                          << " is supported for this source data type and scaling factor < 1.");
            }
        }
        else
        {
            switch (aRoundingMode)
            {
                case mpp::RoundingMode::NearestTiesToEven:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesToEven>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::NearestTiesAwayFromZero:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::NearestTiesAwayFromZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::TowardZero:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardZero>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::TowardNegativeInfinity:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardNegativeInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                case mpp::RoundingMode::TowardPositiveInfinity:
                {
                    using OpT = DivInvIntScaleUpRound<ComputeT, mpp::RoundingMode::TowardPositiveInfinity>;
                    const OpT op(scaleFactorFloat);
                    using divInplaceCScale = InplaceConstantFunctor<1, ComputeT, T, OpT, RoundingMode::None>;
                    const divInplaceCScale functor(static_cast<ComputeT>(aConst), op);
                    forEachPixel(aMask, *this, functor);
                }
                break;
                default:
                    throw INVALIDARGUMENT(aRoundingMode, "Unsupported rounding mode: " << aRoundingMode);
            }
        }
    }

    return *this;
}
#pragma endregion

#pragma region AddSquare
template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddSquare(ImageView<add_spw_output_for_t<T>> &aSrcDst) const
{
    checkSameSize(ROI(), aSrcDst.ROI());

    using addSqrInplaceSrc = InplaceSrcFunctor<1, T, add_spw_output_for_t<T>, add_spw_output_for_t<T>,
                                               mpp::AddSqr<add_spw_output_for_t<T>>, RoundingMode::None>;

    const mpp::AddSqr<add_spw_output_for_t<T>> op;

    const addSqrInplaceSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aSrcDst, functor);

    return aSrcDst;
}

template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddSquareMasked(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                                  const ImageView<Pixel8uC1> &aMask) const
{
    checkSameSize(ROI(), aSrcDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using addSqrInplaceSrc = InplaceSrcFunctor<1, T, add_spw_output_for_t<T>, add_spw_output_for_t<T>,
                                               mpp::AddSqr<add_spw_output_for_t<T>>, RoundingMode::None>;

    const mpp::AddSqr<add_spw_output_for_t<T>> op;

    const addSqrInplaceSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aMask, aSrcDst, functor);

    return aSrcDst;
}
#pragma endregion

#pragma region AddProduct
template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddProduct(const ImageView<T> &aSrc2,
                                                             ImageView<add_spw_output_for_t<T>> &aSrcDst) const
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aSrcDst.ROI());

    using addProductInplaceSrcSrc = InplaceSrcSrcFunctor<1, T, add_spw_output_for_t<T>, add_spw_output_for_t<T>,
                                                         mpp::AddProduct<add_spw_output_for_t<T>>, RoundingMode::None>;

    const mpp::AddProduct<add_spw_output_for_t<T>> op;

    const addProductInplaceSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aSrcDst, functor);

    return aSrcDst;
}

template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddProductMasked(const ImageView<T> &aSrc2,
                                                                   ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                                   const ImageView<Pixel8uC1> &aMask) const
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aSrcDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using addProductInplaceSrcSrc = InplaceSrcSrcFunctor<1, T, add_spw_output_for_t<T>, add_spw_output_for_t<T>,
                                                         mpp::AddProduct<add_spw_output_for_t<T>>, RoundingMode::None>;

    const mpp::AddProduct<add_spw_output_for_t<T>> op;

    const addProductInplaceSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, aSrcDst, functor);

    return aSrcDst;
}
#pragma endregion

#pragma region AddWeighted
template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddWeighted(const ImageView<T> &aSrc2,
                                                              ImageView<add_spw_output_for_t<T>> &aDst,
                                                              remove_vector_t<add_spw_output_for_t<T>> aAlpha) const
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using addWeightedSrcSrc = SrcSrcFunctor<1, T, add_spw_output_for_t<T>, add_spw_output_for_t<T>,
                                            mpp::AddWeighted<add_spw_output_for_t<T>>, RoundingMode::None>;

    const mpp::AddWeighted<add_spw_output_for_t<T>> op(aAlpha);

    const addWeightedSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddWeightedMasked(const ImageView<T> &aSrc2,
                                                                    ImageView<add_spw_output_for_t<T>> &aDst,
                                                                    remove_vector_t<add_spw_output_for_t<T>> aAlpha,
                                                                    const ImageView<Pixel8uC1> &aMask) const
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using addWeightedSrcSrc = SrcSrcFunctor<1, T, add_spw_output_for_t<T>, add_spw_output_for_t<T>,
                                            mpp::AddWeighted<add_spw_output_for_t<T>>, RoundingMode::None>;

    const mpp::AddWeighted<add_spw_output_for_t<T>> op(aAlpha);

    const addWeightedSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aMask, aDst, functor);

    return aDst;
}
template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddWeighted(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                              remove_vector_t<add_spw_output_for_t<T>> aAlpha) const
{
    checkSameSize(ROI(), aSrcDst.ROI());

    using addWeightedInplaceSrc = InplaceSrcFunctor<1, T, add_spw_output_for_t<T>, add_spw_output_for_t<T>,
                                                    mpp::AddWeighted<add_spw_output_for_t<T>>, RoundingMode::None>;

    const mpp::AddWeighted<add_spw_output_for_t<T>> op(aAlpha);

    const addWeightedInplaceSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aSrcDst, functor);

    return aSrcDst;
}

template <PixelType T>
ImageView<add_spw_output_for_t<T>> &ImageView<T>::AddWeightedMasked(ImageView<add_spw_output_for_t<T>> &aSrcDst,
                                                                    remove_vector_t<add_spw_output_for_t<T>> aAlpha,
                                                                    const ImageView<Pixel8uC1> &aMask) const
{
    checkSameSize(ROI(), aSrcDst.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using addWeightedInplaceSrc = InplaceSrcFunctor<1, T, add_spw_output_for_t<T>, add_spw_output_for_t<T>,
                                                    mpp::AddWeighted<add_spw_output_for_t<T>>, RoundingMode::None>;

    const mpp::AddWeighted<add_spw_output_for_t<T>> op(aAlpha);

    const addWeightedInplaceSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aMask, aSrcDst, functor);

    return aSrcDst;
}
#pragma endregion

#pragma region Abs
template <PixelType T>
ImageView<T> &ImageView<T>::Abs(ImageView<T> &aDst) const
    requires RealSignedVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using ComputeT = default_compute_type_for_t<T>;

    using absSrc = SrcFunctor<1, T, ComputeT, T, mpp::Abs<ComputeT>, RoundingMode::None>;

    const mpp::Abs<ComputeT> op;
    const absSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Abs()
    requires RealSignedVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using absInplace = InplaceFunctor<1, ComputeT, T, mpp::Abs<ComputeT>, RoundingMode::None>;

    const mpp::Abs<ComputeT> op;
    const absInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region AbsDiff
template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
    requires RealUnsignedVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using AbsDiffSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::AbsDiff<ComputeT>, RoundingMode::None>;

    const mpp::AbsDiff<ComputeT> op;
    const AbsDiffSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const T &aConst, ImageView<T> &aDst) const
    requires RealUnsignedVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_compute_type_for_t<T>;

    using AbsDiffSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::AbsDiff<ComputeT>, RoundingMode::None>;

    const mpp::AbsDiff<ComputeT> op;
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

    using AbsDiffInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::AbsDiff<ComputeT>, RoundingMode::None>;

    const mpp::AbsDiff<ComputeT> op;
    const AbsDiffInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AbsDiff(const T &aConst)
    requires RealUnsignedVector<T>
{
    using ComputeT = default_compute_type_for_t<T>;

    using AbsDiffInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::AbsDiff<ComputeT>, RoundingMode::None>;

    const mpp::AbsDiff<ComputeT> op;
    const AbsDiffInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region And
template <PixelType T>
ImageView<T> &ImageView<T>::And(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using AndSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::And<ComputeT>, RoundingMode::None>;

    const mpp::And<ComputeT> op;
    const AndSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const T &aConst, ImageView<T> &aDst) const
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using AndSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::And<ComputeT>, RoundingMode::None>;

    const mpp::And<ComputeT> op;
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

    using AndInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::And<ComputeT>, RoundingMode::None>;

    const mpp::And<ComputeT> op;
    const AndInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::And(const T &aConst)
    requires RealIntVector<T>
{
    using ComputeT = T;

    using AndInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::And<ComputeT>, RoundingMode::None>;

    const mpp::And<ComputeT> op;
    const AndInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Not
template <PixelType T>
ImageView<T> &ImageView<T>::Not(ImageView<T> &aDst) const
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using ComputeT = T;

    using notSrc = SrcFunctor<1, T, ComputeT, T, mpp::Not<ComputeT>, RoundingMode::None>;

    const mpp::Not<ComputeT> op;
    const notSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Not()
    requires RealIntVector<T>
{
    using ComputeT = T;

    using notInplace = InplaceFunctor<1, ComputeT, T, mpp::Not<ComputeT>, RoundingMode::None>;

    const mpp::Not<ComputeT> op;
    const notInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Exp
template <PixelType T>
ImageView<T> &ImageView<T>::Exp(ImageView<T> &aDst) const
    requires RealOrComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using ComputeT = default_floating_compute_type_for_t<T>;

    using expSrc = SrcFunctor<1, T, ComputeT, T, mpp::Exp<ComputeT>, RoundingMode::NearestTiesToEven>;

    const mpp::Exp<ComputeT> op;
    const expSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Exp()
    requires RealOrComplexVector<T>
{
    using ComputeT = default_floating_compute_type_for_t<T>;

    using expInplace = InplaceFunctor<1, ComputeT, T, mpp::Exp<ComputeT>, RoundingMode::NearestTiesToEven>;

    const mpp::Exp<ComputeT> op;
    const expInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Ln
template <PixelType T>
ImageView<T> &ImageView<T>::Ln(ImageView<T> &aDst) const
    requires RealOrComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using ComputeT = default_floating_compute_type_for_t<T>;

    using lnSrc = SrcFunctor<1, T, ComputeT, T, mpp::Ln<ComputeT>, RoundingMode::NearestTiesToEven>;

    const mpp::Ln<ComputeT> op;
    const lnSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Ln()
    requires RealOrComplexVector<T>
{
    using ComputeT = default_floating_compute_type_for_t<T>;

    using lnInplace = InplaceFunctor<1, ComputeT, T, mpp::Ln<ComputeT>, RoundingMode::NearestTiesToEven>;

    const mpp::Ln<ComputeT> op;
    const lnInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region LShift
template <PixelType T>
ImageView<T> &ImageView<T>::LShift(uint aConst, ImageView<T> &aDst) const
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using LShiftSrcC = SrcFunctor<1, T, ComputeT, T, mpp::LShift<ComputeT>, RoundingMode::None>;

    const mpp::LShift<ComputeT> op(aConst);
    const LShiftSrcC functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::LShift(uint aConst)
    requires RealIntVector<T>
{
    using ComputeT = T;

    using LShiftInplaceC = InplaceFunctor<1, ComputeT, T, mpp::LShift<ComputeT>, RoundingMode::None>;

    const mpp::LShift<ComputeT> op(aConst);
    const LShiftInplaceC functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Or
template <PixelType T>
ImageView<T> &ImageView<T>::Or(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using OrSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::Or<ComputeT>, RoundingMode::None>;

    const mpp::Or<ComputeT> op;
    const OrSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const T &aConst, ImageView<T> &aDst) const
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using OrSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::Or<ComputeT>, RoundingMode::None>;

    const mpp::Or<ComputeT> op;
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

    using OrInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::Or<ComputeT>, RoundingMode::None>;

    const mpp::Or<ComputeT> op;
    const OrInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Or(const T &aConst)
    requires RealIntVector<T>
{
    using ComputeT = T;

    using OrInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::Or<ComputeT>, RoundingMode::None>;

    const mpp::Or<ComputeT> op;
    const OrInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Sqr
template <PixelType T>
ImageView<T> &ImageView<T>::Sqr(ImageView<T> &aDst) const
    requires RealOrComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using ComputeT = default_floating_compute_type_for_t<T>;

    using sqrSrc = SrcFunctor<1, T, ComputeT, T, mpp::Sqr<ComputeT>, RoundingMode::None>;

    const mpp::Sqr<ComputeT> op;
    const sqrSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sqr()
    requires RealOrComplexVector<T>
{
    using ComputeT = default_floating_compute_type_for_t<T>;

    using sqrInplace = InplaceFunctor<1, ComputeT, T, mpp::Sqr<ComputeT>, RoundingMode::None>;

    const mpp::Sqr<ComputeT> op;
    const sqrInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Sqrt
template <PixelType T>
ImageView<T> &ImageView<T>::Sqrt(ImageView<T> &aDst) const
    requires RealOrComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using ComputeT = default_floating_compute_type_for_t<T>;

    using sqrtSrc = SrcFunctor<1, T, ComputeT, T, mpp::Sqrt<ComputeT>, RoundingMode::NearestTiesToEven>;

    const mpp::Sqrt<ComputeT> op;
    const sqrtSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Sqrt()
    requires RealOrComplexVector<T>
{
    using ComputeT = default_floating_compute_type_for_t<T>;

    using sqrtInplace = InplaceFunctor<1, ComputeT, T, mpp::Sqrt<ComputeT>, RoundingMode::NearestTiesToEven>;

    const mpp::Sqrt<ComputeT> op;
    const sqrtInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region RShift
template <PixelType T>
ImageView<T> &ImageView<T>::RShift(uint aConst, ImageView<T> &aDst) const
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using RShiftSrcC = SrcFunctor<1, T, ComputeT, T, mpp::RShift<ComputeT>, RoundingMode::None>;

    const mpp::RShift<ComputeT> op(aConst);
    const RShiftSrcC functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::RShift(uint aConst)
    requires RealIntVector<T>
{
    using ComputeT = T;

    using RShiftInplaceC = InplaceFunctor<1, ComputeT, T, mpp::RShift<ComputeT>, RoundingMode::None>;

    const mpp::RShift<ComputeT> op(aConst);
    const RShiftInplaceC functor(op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion
#pragma region Xor
template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using XorSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::Xor<ComputeT>, RoundingMode::None>;

    const mpp::Xor<ComputeT> op;
    const XorSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const T &aConst, ImageView<T> &aDst) const
    requires RealIntVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = T;

    using XorSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::Xor<ComputeT>, RoundingMode::None>;

    const mpp::Xor<ComputeT> op;
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

    using XorInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::Xor<ComputeT>, RoundingMode::None>;

    const mpp::Xor<ComputeT> op;
    const XorInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Xor(const T &aConst)
    requires RealIntVector<T>
{
    using ComputeT = T;

    using XorInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::Xor<ComputeT>, RoundingMode::None>;

    const mpp::Xor<ComputeT> op;
    const XorInplaceC functor(static_cast<ComputeT>(aConst), op);

    forEachPixel(*this, functor);
    return *this;
}
#pragma endregion

#pragma region AlphaPremul
template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(ImageView<T> &aDst) const
    requires FourChannelNoAlpha<T> && RealVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_floating_compute_type_for_t<T>;
    using alphaPremulSrc =
        SrcFunctor<1, T, ComputeT, T, mpp::AlphaPremul<ComputeT, T>, RoundingMode::NearestTiesToEven>;

    const mpp::AlphaPremul<ComputeT, T> op;

    const alphaPremulSrc functor(PointerRoi(), Pitch(), op);
    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul()
    requires FourChannelNoAlpha<T> && RealVector<T>
{
    using ComputeT = default_floating_compute_type_for_t<T>;
    using alphaPremulInplace =
        InplaceFunctor<1, ComputeT, T, mpp::AlphaPremul<ComputeT, T>, RoundingMode::NearestTiesToEven>;

    const mpp::AlphaPremul<ComputeT, T> op;

    const alphaPremulInplace functor(op);
    forEachPixel(*this, functor);

    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst) const
    requires RealFloatingVector<T> && (!FourChannelAlpha<T>)
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_ext_int_compute_type_for_t<T>;
    const T alphaVec(aAlpha);

    using mulSrcC = SrcConstantFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, RoundingMode::None>;

    const mpp::Mul<ComputeT> op;
    const mulSrcC functor(PointerRoi(), Pitch(), alphaVec, op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst) const
    requires RealIntVector<T> && (!FourChannelAlpha<T>)
{
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_ext_int_compute_type_for_t<T>;
    const T alphaVec(aAlpha);
    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    if (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulSrcCScale functor(PointerRoi(), Pitch(), alphaVec, op, scaler);

        forEachPixel(aDst, functor);
    }
    else
    {
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulSrcCScale =
            SrcConstantScaleFunctor<1, T, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulSrcCScale functor(PointerRoi(), Pitch(), alphaVec, op, scaler);

        forEachPixel(aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha)
    requires RealFloatingVector<T> && (!FourChannelAlpha<T>)
{
    using ComputeT = default_ext_int_compute_type_for_t<T>;
    const T alphaVec(aAlpha);

    using mulInplaceC = InplaceConstantFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, RoundingMode::None>;

    const mpp::Mul<ComputeT> op;
    const mulInplaceC functor(alphaVec, op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha)
    requires RealIntVector<T> && (!FourChannelAlpha<T>)
{
    using ComputeT = default_ext_int_compute_type_for_t<T>;
    const T alphaVec(aAlpha);
    constexpr double scaleFactorFloat = 1.0 / static_cast<double>(numeric_limits<remove_vector_t<T>>::max());

    if (RealOrComplexFloatingVector<ComputeT>)
    {
        using ScalerT = mpp::Scale<ComputeT, false>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::NearestTiesToEven>;

        const mpp::Mul<ComputeT> op;
        const mulInplaceCScale functor(alphaVec, op, scaler);

        forEachPixel(*this, functor);
    }
    else
    {
        using ScalerT = mpp::Scale<ComputeT, true>;
        const ScalerT scaler(scaleFactorFloat);
        using mulInplaceCScale =
            InplaceConstantScaleFunctor<1, ComputeT, T, mpp::Mul<ComputeT>, ScalerT, RoundingMode::None>;

        const mpp::Mul<ComputeT> op;
        const mulInplaceCScale functor(alphaVec, op, scaler);

        forEachPixel(*this, functor);
    }
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha, ImageView<T> &aDst) const
    requires FourChannelAlpha<T>
{
    checkSameSize(ROI(), aDst.ROI());

    // use non-alpha Vector4 in order to correctly treat the setted alpha value:
    using ComputeT = Vector4<remove_vector_t<default_floating_compute_type_for_t<T>>>;
    using SrcT     = Vector4<remove_vector_t<T>>;
    using alphaPremulACSrc =
        SrcFunctor<1, SrcT, ComputeT, SrcT, mpp::AlphaPremulAC<ComputeT, SrcT>, RoundingMode::NearestTiesToEven>;

    const mpp::AlphaPremulAC<ComputeT, SrcT> op(aAlpha);

    const alphaPremulACSrc functor(reinterpret_cast<const SrcT *>(PointerRoi()), Pitch(), op);
    ImageView<SrcT> aDstNoAlpha = aDst;
    forEachPixel(aDstNoAlpha, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::AlphaPremul(remove_vector_t<T> aAlpha)
    requires FourChannelAlpha<T>
{
    // use non-alpha Vector4 in order to correctly treat the setted alpha value:
    using ComputeT = Vector4<remove_vector_t<default_floating_compute_type_for_t<T>>>;
    using SrcT     = Vector4<remove_vector_t<T>>;
    using alphaPremulACInplace =
        InplaceFunctor<1, ComputeT, SrcT, mpp::AlphaPremulAC<ComputeT, SrcT>, RoundingMode::NearestTiesToEven>;

    const mpp::AlphaPremulAC<ComputeT, SrcT> op(aAlpha);

    const alphaPremulACInplace functor(op);
    ImageView<SrcT> aDstNoAlpha = *this;
    forEachPixel(aDstNoAlpha, functor);

    return *this;
}
#pragma endregion

#pragma region AlphaComp
template <PixelType T>
ImageView<T> &ImageView<T>::AlphaComp(const ImageView<T> &aSrc2, ImageView<T> &aDst, AlphaCompositionOp aAlphaOp) const
    requires(!FourChannelAlpha<T>) && RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    constexpr size_t TupelSize = 1;
    using ComputeT             = default_floating_compute_type_for_t<T>;
    using SrcT                 = T;
    using DstT                 = T;

    if constexpr (RealIntVector<SrcT>)
    {
        switch (aAlphaOp)
        {
            case mpp::AlphaCompositionOp::Over:
            {
                using AlphaCompOp = mpp::AlphaCompositionScale<ComputeT, SrcT, mpp::AlphaCompositionOp::Over>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesToEven>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::In:
            {
                using AlphaCompOp = mpp::AlphaCompositionScale<ComputeT, SrcT, mpp::AlphaCompositionOp::In>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesToEven>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::Out:
            {
                using AlphaCompOp = mpp::AlphaCompositionScale<ComputeT, SrcT, mpp::AlphaCompositionOp::Out>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesToEven>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::ATop:
            {
                using AlphaCompOp = mpp::AlphaCompositionScale<ComputeT, SrcT, mpp::AlphaCompositionOp::ATop>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesToEven>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::XOr:
            {
                using AlphaCompOp = mpp::AlphaCompositionScale<ComputeT, SrcT, mpp::AlphaCompositionOp::XOr>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesToEven>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::Plus:
            {
                using AlphaCompOp = mpp::AlphaCompositionScale<ComputeT, SrcT, mpp::AlphaCompositionOp::Plus>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesToEven>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::OverPremul:
            {
                using AlphaCompOp = mpp::AlphaCompositionScale<ComputeT, SrcT, mpp::AlphaCompositionOp::OverPremul>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesToEven>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::InPremul:
            {
                using AlphaCompOp = mpp::AlphaCompositionScale<ComputeT, SrcT, mpp::AlphaCompositionOp::InPremul>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesToEven>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::OutPremul:
            {
                using AlphaCompOp = mpp::AlphaCompositionScale<ComputeT, SrcT, mpp::AlphaCompositionOp::OutPremul>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesToEven>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::ATopPremul:
            {
                using AlphaCompOp = mpp::AlphaCompositionScale<ComputeT, SrcT, mpp::AlphaCompositionOp::ATopPremul>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesToEven>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::XOrPremul:
            {
                using AlphaCompOp = mpp::AlphaCompositionScale<ComputeT, SrcT, mpp::AlphaCompositionOp::XOrPremul>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesToEven>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::PlusPremul:
            {
                using AlphaCompOp = mpp::AlphaCompositionScale<ComputeT, SrcT, mpp::AlphaCompositionOp::PlusPremul>;
                using alphaCompSrcSrc =
                    SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::NearestTiesToEven>;
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
            case mpp::AlphaCompositionOp::Over:
            {
                using AlphaCompOp     = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::Over>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::In:
            {
                using AlphaCompOp     = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::In>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::Out:
            {
                using AlphaCompOp     = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::Out>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::ATop:
            {
                using AlphaCompOp     = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::ATop>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::XOr:
            {
                using AlphaCompOp     = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::XOr>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::Plus:
            {
                using AlphaCompOp     = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::Plus>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::OverPremul:
            {
                using AlphaCompOp     = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::OverPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::InPremul:
            {
                using AlphaCompOp     = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::InPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::OutPremul:
            {
                using AlphaCompOp     = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::OutPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::ATopPremul:
            {
                using AlphaCompOp     = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::ATopPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::XOrPremul:
            {
                using AlphaCompOp     = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::XOrPremul>;
                using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, RoundingMode::None>;
                const AlphaCompOp op;
                const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
                forEachPixel(aDst, functor);
            }
            break;
            case mpp::AlphaCompositionOp::PlusPremul:
            {
                using AlphaCompOp     = mpp::AlphaComposition<ComputeT, mpp::AlphaCompositionOp::PlusPremul>;
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
                                      remove_vector_t<T> aAlpha2, AlphaCompositionOp aAlphaOp) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    constexpr size_t TupelSize = 1;
    using ComputeT             = default_floating_compute_type_for_t<T>;
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
        case mpp::AlphaCompositionOp::Over:
        {
            using AlphaCompOp     = mpp::AlphaCompositionC<ComputeT, mpp::AlphaCompositionOp::Over>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::AlphaCompositionOp::In:
        {
            using AlphaCompOp     = mpp::AlphaCompositionC<ComputeT, mpp::AlphaCompositionOp::In>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::AlphaCompositionOp::Out:
        {
            using AlphaCompOp     = mpp::AlphaCompositionC<ComputeT, mpp::AlphaCompositionOp::Out>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::AlphaCompositionOp::ATop:
        {
            using AlphaCompOp     = mpp::AlphaCompositionC<ComputeT, mpp::AlphaCompositionOp::ATop>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::AlphaCompositionOp::XOr:
        {
            using AlphaCompOp     = mpp::AlphaCompositionC<ComputeT, mpp::AlphaCompositionOp::XOr>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::AlphaCompositionOp::Plus:
        {
            using AlphaCompOp     = mpp::AlphaCompositionC<ComputeT, mpp::AlphaCompositionOp::Plus>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::AlphaCompositionOp::OverPremul:
        {
            using AlphaCompOp     = mpp::AlphaCompositionC<ComputeT, mpp::AlphaCompositionOp::OverPremul>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::AlphaCompositionOp::InPremul:
        {
            using AlphaCompOp     = mpp::AlphaCompositionC<ComputeT, mpp::AlphaCompositionOp::InPremul>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::AlphaCompositionOp::OutPremul:
        {
            using AlphaCompOp     = mpp::AlphaCompositionC<ComputeT, mpp::AlphaCompositionOp::OutPremul>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::AlphaCompositionOp::ATopPremul:
        {
            using AlphaCompOp     = mpp::AlphaCompositionC<ComputeT, mpp::AlphaCompositionOp::ATopPremul>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::AlphaCompositionOp::XOrPremul:
        {
            using AlphaCompOp     = mpp::AlphaCompositionC<ComputeT, mpp::AlphaCompositionOp::XOrPremul>;
            using alphaCompSrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, AlphaCompOp, roundingMode>;
            const AlphaCompOp op(alpha1, alpha2);
            const alphaCompSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
            forEachPixel(aDst, functor);
        }
        break;
        case mpp::AlphaCompositionOp::PlusPremul:
        {
            using AlphaCompOp     = mpp::AlphaCompositionC<ComputeT, mpp::AlphaCompositionOp::PlusPremul>;
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
ImageView<T> &ImageView<T>::ConjMul(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using conjMulSrcSrc = SrcSrcFunctor<1, T, ComputeT, T, mpp::ConjMul<ComputeT>, RoundingMode::None>;

    const mpp::ConjMul<ComputeT> op;
    const conjMulSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::ConjMul(const ImageView<T> &aSrc2)
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = default_ext_int_compute_type_for_t<T>;

    // set to roundingmode NONE, because Mul cannot produce non-integers in computations with ints:
    using conjMulInplaceSrc = InplaceSrcFunctor<1, T, ComputeT, T, mpp::ConjMul<ComputeT>, RoundingMode::None>;

    const mpp::ConjMul<ComputeT> op;
    const conjMulInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Conj(ImageView<T> &aDst) const
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());

    using conjSrc = SrcFunctor<1, T, T, T, mpp::Conj<T>, RoundingMode::None>;

    const mpp::Conj<T> op;
    const conjSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::Conj()
    requires ComplexVector<T>
{
    using conjInplace = InplaceFunctor<1, T, T, mpp::Conj<T>, RoundingMode::None>;

    const mpp::Conj<T> op;
    const conjInplace functor(op);

    forEachPixel(*this, functor);
    return *this;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Magnitude(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst) const
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    Size2D sizeDst = aDst.SizeRoi();
    sizeDst.x      = sizeDst.x / 2 + 1;

    using SrcT     = T;
    using ComputeT = default_compute_type_for_t<T>;
    using DstT     = same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>;

    if (SizeRoi() == sizeDst)
    {
        constexpr RoundingMode roundingMode = RoundingMode::None;
        using CoordT                        = int;

        const TransformerPStoFFTW<CoordT> ps2fftw(aDst.SizeRoi());

        using BCType = BorderControl<T, BorderType::None>;
        const BCType bc(PointerRoi(), Pitch(), SizeRoi(), {0});

        const mpp::Magnitude<ComputeT> op;
        using InterpolatorT = Interpolator<T, BCType, CoordT, InterpolationMode::Undefined>;
        const InterpolatorT interpol(bc);
        const TransformerFunctor<1, DstT, CoordT, false, InterpolatorT, TransformerPStoFFTW<CoordT>, roundingMode,
                                 mpp::Magnitude<ComputeT>>
            functor(interpol, ps2fftw, aDst.SizeRoi(), op);

        forEachPixel(aDst, functor);
    }
    else
    {
        checkSameSize(ROI(), aDst.ROI());

        using magnitudeSrc =
            SrcFunctor<1, SrcT, ComputeT, DstT, mpp::Magnitude<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::Magnitude<ComputeT> op;
        const magnitudeSrc functor(PointerRoi(), Pitch(), op);

        forEachPixel(aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::MagnitudeSqr(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst) const
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    Size2D sizeDst = aDst.SizeRoi();
    sizeDst.x      = sizeDst.x / 2 + 1;
    using SrcT     = T;
    using ComputeT = default_compute_type_for_t<T>;
    using DstT     = same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>;

    if (SizeRoi() == sizeDst)
    {
        constexpr RoundingMode roundingMode = RoundingMode::None;
        using CoordT                        = int;

        const TransformerPStoFFTW<CoordT> ps2fftw(aDst.SizeRoi());

        using BCType = BorderControl<T, BorderType::None>;
        const BCType bc(PointerRoi(), Pitch(), SizeRoi(), {0});

        const mpp::MagnitudeSqr<ComputeT> op;
        using InterpolatorT = Interpolator<T, BCType, CoordT, InterpolationMode::Undefined>;
        const InterpolatorT interpol(bc);
        const TransformerFunctor<1, DstT, CoordT, false, InterpolatorT, TransformerPStoFFTW<CoordT>, roundingMode,
                                 mpp::MagnitudeSqr<ComputeT>>
            functor(interpol, ps2fftw, aDst.SizeRoi(), op);

        forEachPixel(aDst, functor);
    }
    else
    {
        checkSameSize(ROI(), aDst.ROI());
        using magnitudeSqrSrc =
            SrcFunctor<1, SrcT, ComputeT, DstT, mpp::MagnitudeSqr<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::MagnitudeSqr<ComputeT> op;

        const magnitudeSqrSrc functor(PointerRoi(), Pitch(), op);

        forEachPixel(aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::MagnitudeLog(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst,
    complex_basetype_t<remove_vector_t<T>> aOffset) const
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    Size2D sizeDst = aDst.SizeRoi();
    sizeDst.x      = sizeDst.x / 2 + 1;

    using SrcT     = T;
    using ComputeT = default_compute_type_for_t<T>;
    using DstT     = same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>;

    if (SizeRoi() == sizeDst)
    {
        constexpr RoundingMode roundingMode = RoundingMode::None;
        using CoordT                        = int;

        const TransformerPStoFFTW<CoordT> ps2fftw(aDst.SizeRoi());

        using BCType = BorderControl<T, BorderType::None>;
        const BCType bc(PointerRoi(), Pitch(), SizeRoi(), {0});

        const mpp::MagnitudeLog<ComputeT> op(aOffset);
        using InterpolatorT = Interpolator<T, BCType, CoordT, InterpolationMode::Undefined>;
        const InterpolatorT interpol(bc);
        const TransformerFunctor<1, DstT, CoordT, false, InterpolatorT, TransformerPStoFFTW<CoordT>, roundingMode,
                                 mpp::MagnitudeLog<ComputeT>>
            functor(interpol, ps2fftw, aDst.SizeRoi(), op);

        forEachPixel(aDst, functor);
    }
    else
    {
        checkSameSize(ROI(), aDst.ROI());

        using magnitudeSrc =
            SrcFunctor<1, SrcT, ComputeT, DstT, mpp::MagnitudeLog<ComputeT>, RoundingMode::NearestTiesToEven>;

        const mpp::MagnitudeLog<ComputeT> op(aOffset);
        const magnitudeSrc functor(PointerRoi(), Pitch(), op);

        forEachPixel(aDst, functor);
    }
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Angle(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst) const
    requires ComplexVector<T> && ComplexFloatingPoint<remove_vector_t<T>>
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using ComputeT = default_compute_type_for_t<T>;
    using DstT     = same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>;

    using angleSrc = SrcFunctor<1, SrcT, ComputeT, DstT, mpp::Angle<ComputeT>, RoundingMode::NearestTiesToEven>;

    const mpp::Angle<ComputeT> op;

    const angleSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Real(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst) const
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using ComputeT = T;
    using DstT     = same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>;

    using realSrc = SrcFunctor<1, SrcT, ComputeT, DstT, mpp::Real<ComputeT>, RoundingMode::NearestTiesToEven>;

    const mpp::Real<ComputeT> op;

    const realSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &ImageView<T>::Imag(
    ImageView<same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>> &aDst) const
    requires ComplexVector<T>
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using ComputeT = T;
    using DstT     = same_vector_size_different_type_t<T, complex_basetype_t<remove_vector_t<T>>>;

    using imagSrc = SrcFunctor<1, SrcT, ComputeT, DstT, mpp::Imag<ComputeT>, RoundingMode::NearestTiesToEven>;

    const mpp::Imag<ComputeT> op;

    const imagSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &ImageView<T>::MakeComplex(
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst) const
    requires RealSignedVector<T> && (!FourChannelAlpha<T>)
{
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using ComputeT = T;
    using DstT     = same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>;

    using makeComplexSrc = SrcFunctor<1, SrcT, ComputeT, DstT, mpp::MakeComplex<DstT>, RoundingMode::NearestTiesToEven>;

    const mpp::MakeComplex<DstT> op;

    const makeComplexSrc functor(PointerRoi(), Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &ImageView<T>::MakeComplex(
    const ImageView<T> &aSrcImag,
    ImageView<same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>> &aDst) const
    requires RealSignedVector<T> && (!FourChannelAlpha<T>)
{
    checkSameSize(ROI(), aSrcImag.ROI());
    checkSameSize(ROI(), aDst.ROI());
    using SrcT     = T;
    using ComputeT = T;
    using DstT     = same_vector_size_different_type_t<T, make_complex_t<remove_vector_t<T>>>;

    using makeComplexSrcSrc =
        SrcSrcFunctor<1, SrcT, ComputeT, DstT, mpp::MakeComplex<DstT>, RoundingMode::NearestTiesToEven>;

    const mpp::MakeComplex<DstT> op;

    const makeComplexSrcSrc functor(PointerRoi(), Pitch(), aSrcImag.PointerRoi(), aSrcImag.Pitch(), op);

    forEachPixel(aDst, functor);
    return aDst;
}
#pragma endregion

// NOLINTEND(clang-diagnostic-double-promotion)
} // namespace mpp::image::cpuSimple