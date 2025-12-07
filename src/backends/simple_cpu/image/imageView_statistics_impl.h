#pragma once
#include <array>
#include <backends/simple_cpu/image/filterEachPixel_impl.h>
#include <backends/simple_cpu/image/forEachPixel.h>
#include <backends/simple_cpu/image/forEachPixelMasked.h>
#include <backends/simple_cpu/image/forEachPixelPlanar.h>
#include <backends/simple_cpu/image/forEachPixelSingleChannel.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/image/reduction.h>
#include <backends/simple_cpu/image/reductionMasked.h>
#include <backends/simple_cpu/operator_random.h>
#include <climits>
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/border.h>
#include <common/image/channel.h>
#include <common/image/channelList.h>
#include <common/image/fixedSizeFilters.h>
#include <common/image/functors/constantFunctor.h>
#include <common/image/functors/convertFunctor.h>
#include <common/image/functors/convertScaleFunctor.h>
#include <common/image/functors/imageFunctors.h>
#include <common/image/functors/inplaceConstantFunctor.h>
#include <common/image/functors/inplaceConstantScaleFunctor.h>
#include <common/image/functors/inplaceDevConstantFunctor.h>
#include <common/image/functors/inplaceDevConstantScaleFunctor.h>
#include <common/image/functors/inplaceFunctor.h>
#include <common/image/functors/inplaceSrcFunctor.h>
#include <common/image/functors/inplaceSrcScaleFunctor.h>
#include <common/image/functors/rectStdDevFunctor.h>
#include <common/image/functors/reductionInitValues.h>
#include <common/image/functors/scaleConversionFunctor.h>
#include <common/image/functors/srcConstantFunctor.h>
#include <common/image/functors/srcConstantScaleFunctor.h>
#include <common/image/functors/srcDevConstantFunctor.h>
#include <common/image/functors/srcDevConstantScaleFunctor.h>
#include <common/image/functors/srcDstAsSrcFunctor.h>
#include <common/image/functors/srcFunctor.h>
#include <common/image/functors/srcPlanar2Functor.h>
#include <common/image/functors/srcPlanar3Functor.h>
#include <common/image/functors/srcPlanar4Functor.h>
#include <common/image/functors/srcReduction2Functor.h>
#include <common/image/functors/srcReductionFunctor.h>
#include <common/image/functors/srcReductionMaxIdxFunctor.h>
#include <common/image/functors/srcReductionMinIdxFunctor.h>
#include <common/image/functors/srcScaleFunctor.h>
#include <common/image/functors/srcSingleChannelFunctor.h>
#include <common/image/functors/srcSrcFunctor.h>
#include <common/image/functors/srcSrcReduction2Functor.h>
#include <common/image/functors/srcSrcReduction5Functor.h>
#include <common/image/functors/srcSrcReductionFunctor.h>
#include <common/image/functors/srcSrcScaleFunctor.h>
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
#include <common/statistics/indexMinMax.h>
#include <common/statistics/operators.h>
#include <common/statistics/postOperators.h>
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
#pragma region AverageError
template <PixelType T>
void ImageView<T>::AverageError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                double &aDstScalar) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL1<SrcT, ComputeT>>;

    const mpp::NormDiffL1<SrcT, ComputeT> op;

    const normL1SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()));
    const mpp::DivScalar<DstT> postOpScalar(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()));

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::AverageErrorMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                      double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL1<SrcT, ComputeT>>;

    const mpp::NormDiffL1<SrcT, ComputeT> op;

    const normL1SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(maskpixels));
    const mpp::DivScalar<DstT> postOpScalar(static_cast<remove_vector_t<DstT>>(maskpixels));

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::AverageError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL1<SrcT, ComputeT>>;

    const mpp::NormDiffL1<SrcT, ComputeT> op;

    const normL1SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()));

    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::AverageErrorMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                      const ImageView<Pixel8uC1> &aMask) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL1<SrcT, ComputeT>>;

    const mpp::NormDiffL1<SrcT, ComputeT> op;

    const normL1SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(maskpixels));

    postOp(aDst);
}
#pragma endregion

#pragma region AverageRelativeError
template <PixelType T>
void ImageView<T>::AverageRelativeError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                        double &aDstScalar) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using avgErrorSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::AverageRelativeError<SrcT, ComputeT>>;

    const mpp::AverageRelativeError<SrcT, ComputeT> op;

    const avgErrorSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()));
    const mpp::DivScalar<DstT> postOpScalar(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()));

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::AverageRelativeErrorMasked(const ImageView<T> &aSrc2,
                                              same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                                              const ImageView<Pixel8uC1> &aMask) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using avgErrorSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::AverageRelativeError<SrcT, ComputeT>>;

    const mpp::AverageRelativeError<SrcT, ComputeT> op;

    const avgErrorSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(maskpixels));
    const mpp::DivScalar<DstT> postOpScalar(static_cast<remove_vector_t<DstT>>(maskpixels));

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::AverageRelativeError(const ImageView<T> &aSrc2,
                                        same_vector_size_different_type_t<T, double> &aDst) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using avgErrorSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::AverageRelativeError<SrcT, ComputeT>>;

    const mpp::AverageRelativeError<SrcT, ComputeT> op;

    const avgErrorSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()));

    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::AverageRelativeErrorMasked(const ImageView<T> &aSrc2,
                                              same_vector_size_different_type_t<T, double> &aDst,
                                              const ImageView<Pixel8uC1> &aMask) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using avgErrorSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::AverageRelativeError<SrcT, ComputeT>>;

    const mpp::AverageRelativeError<SrcT, ComputeT> op;

    const avgErrorSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(maskpixels));

    postOp(aDst);
}
#pragma endregion

#pragma region DotProduct
template <PixelType T>
void ImageView<T>::DotProduct(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                              double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using dotProdSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::DotProduct<SrcT, ComputeT>>;

    const mpp::DotProduct<SrcT, ComputeT> op;

    const dotProdSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::DotProduct(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst,
                              c_double &aDstScalar) const
    requires ComplexVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using dotProdSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::DotProduct<SrcT, ComputeT>>;

    const mpp::DotProduct<SrcT, ComputeT> op;

    const dotProdSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::DotProductMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                    double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using dotProdSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::DotProduct<SrcT, ComputeT>>;

    const mpp::DotProduct<SrcT, ComputeT> op;

    const dotProdSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::DotProductMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst,
                                    c_double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires ComplexVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using dotProdSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::DotProduct<SrcT, ComputeT>>;

    const mpp::DotProduct<SrcT, ComputeT> op;

    const dotProdSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::DotProduct(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using dotProdSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::DotProduct<SrcT, ComputeT>>;

    const mpp::DotProduct<SrcT, ComputeT> op;

    const dotProdSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::DotProduct(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst) const
    requires ComplexVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using dotProdSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::DotProduct<SrcT, ComputeT>>;

    const mpp::DotProduct<SrcT, ComputeT> op;

    const dotProdSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::DotProductMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                    const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using dotProdSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::DotProduct<SrcT, ComputeT>>;

    const mpp::DotProduct<SrcT, ComputeT> op;

    const dotProdSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::DotProductMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst,
                                    const ImageView<Pixel8uC1> &aMask) const
    requires ComplexVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using dotProdSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::DotProduct<SrcT, ComputeT>>;

    const mpp::DotProduct<SrcT, ComputeT> op;

    const dotProdSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}
#pragma endregion
#pragma region MSE
template <PixelType T>
void ImageView<T>::MSE(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                       double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()));
    const mpp::DivScalar<DstT> postOpScalar(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()));

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::MSE(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst,
                       c_double &aDstScalar) const
    requires ComplexVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(SizeRoi().TotalSize()));
    const mpp::DivScalar<DstT> postOpScalar(
        static_cast<complex_basetype_t<remove_vector_t<DstT>>>(SizeRoi().TotalSize()));

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MSEMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                             double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(maskpixels));
    const mpp::DivScalar<DstT> postOpScalar(static_cast<remove_vector_t<DstT>>(maskpixels));

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::MSEMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst,
                             c_double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires ComplexVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(maskpixels));
    const mpp::DivScalar<DstT> postOpScalar(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(maskpixels));

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MSE(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()));

    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::MSE(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst) const
    requires ComplexVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(SizeRoi().TotalSize()));

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MSEMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                             const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<remove_vector_t<DstT>>(maskpixels));

    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::MSEMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, c_double> &aDst,
                             const ImageView<Pixel8uC1> &aMask) const
    requires ComplexVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(maskpixels));

    postOp(aDst);
}
#pragma endregion
#pragma region MaximumError
template <PixelType T>
void ImageView<T>::MaximumError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                double &aDstScalar) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normInfSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffInf<SrcT, ComputeT>>;

    const mpp::NormDiffInf<SrcT, ComputeT> op;

    const normInfSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::MaxScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MaximumErrorMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                      double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normInfSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffInf<SrcT, ComputeT>>;

    const mpp::NormDiffInf<SrcT, ComputeT> op;

    const normInfSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::MaxScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MaximumError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normInfSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffInf<SrcT, ComputeT>>;

    const mpp::NormDiffInf<SrcT, ComputeT> op;

    const normInfSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MaximumErrorMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                      const ImageView<Pixel8uC1> &aMask) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normInfSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffInf<SrcT, ComputeT>>;

    const mpp::NormDiffInf<SrcT, ComputeT> op;

    const normInfSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}
#pragma endregion
#pragma region MaximumRelativeError
template <PixelType T>
void ImageView<T>::MaximumRelativeError(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                        double &aDstScalar) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using avgErrorSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::MaximumRelativeError<SrcT, ComputeT>>;

    const mpp::MaximumRelativeError<SrcT, ComputeT> op;

    const avgErrorSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::MaxScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MaximumRelativeErrorMasked(const ImageView<T> &aSrc2,
                                              same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                                              const ImageView<Pixel8uC1> &aMask) const
    requires(vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using avgErrorSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::MaximumRelativeError<SrcT, ComputeT>>;

    const mpp::MaximumRelativeError<SrcT, ComputeT> op;

    const avgErrorSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::MaxScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MaximumRelativeError(const ImageView<T> &aSrc2,
                                        same_vector_size_different_type_t<T, double> &aDst) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using avgErrorSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::MaximumRelativeError<SrcT, ComputeT>>;

    const mpp::MaximumRelativeError<SrcT, ComputeT> op;

    const avgErrorSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MaximumRelativeErrorMasked(const ImageView<T> &aSrc2,
                                              same_vector_size_different_type_t<T, double> &aDst,
                                              const ImageView<Pixel8uC1> &aMask) const
    requires(vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using avgErrorSrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::MaximumRelativeError<SrcT, ComputeT>>;

    const mpp::MaximumRelativeError<SrcT, ComputeT> op;

    const avgErrorSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}
#pragma endregion
#pragma region NormDiffInf
template <PixelType T>
void ImageView<T>::NormDiffInf(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                               double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    MaximumError(aSrc2, aDst, aDstScalar);
}

template <PixelType T>
void ImageView<T>::NormDiffInfMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                     double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    MaximumErrorMasked(aSrc2, aDst, aDstScalar, aMask);
}

template <PixelType T>
void ImageView<T>::NormDiffInf(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    MaximumError(aSrc2, aDst);
}

template <PixelType T>
void ImageView<T>::NormDiffInfMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                     const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    MaximumErrorMasked(aSrc2, aDst, aMask);
}
#pragma endregion
#pragma region NormDiffL1
template <PixelType T>
void ImageView<T>::NormDiffL1(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                              double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL1<SrcT, ComputeT>>;

    const mpp::NormDiffL1<SrcT, ComputeT> op;

    const normL1SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormDiffL1Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                    double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL1<SrcT, ComputeT>>;

    const mpp::NormDiffL1<SrcT, ComputeT> op;

    const normL1SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormDiffL1(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL1<SrcT, ComputeT>>;

    const mpp::NormDiffL1<SrcT, ComputeT> op;

    const normL1SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormDiffL1Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                    const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL1<SrcT, ComputeT>>;

    const mpp::NormDiffL1<SrcT, ComputeT> op;

    const normL1SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}
#pragma endregion
#pragma region NormDiffL2
template <PixelType T>
void ImageView<T>::NormDiffL2(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                              double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::SqrtPostOp<DstT> postOp;
    const mpp::SumThenSqrtScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormDiffL2Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                    double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::SqrtPostOp<DstT> postOp;
    const mpp::SumThenSqrtScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::NormDiffL2(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::SqrtPostOp<DstT> postOp;
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormDiffL2Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                    const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::SqrtPostOp<DstT> postOp;

    postOp(aDst);
}
#pragma endregion
#pragma region NormRelInf
template <PixelType T>
void ImageView<T>::NormRelInf(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                              double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normInfSrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::NormRelInf<SrcT, ComputeT>,
                                                  mpp::NormInf<SrcT, ComputeT>>;

    const mpp::NormRelInf<SrcT, ComputeT> op1;
    const mpp::NormInf<SrcT, ComputeT> op2;

    const normInfSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op1, op2);

    DstT dst1 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);
    DstT dst2 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), dst1, dst2, functor);

    // ignore result1 as only the second is a meaningful output for NormRel
    const mpp::NormRelInfPost<DstT> postOp2;
    const mpp::NormRelInfPost<DstT> postOpScalar2;

    postOpScalar2(dst1, dst2, aDstScalar);

    postOp2(dst1, dst2, aDst);
}

template <PixelType T>
void ImageView<T>::NormRelInfMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                    double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normInfSrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::NormRelInf<SrcT, ComputeT>,
                                                  mpp::NormInf<SrcT, ComputeT>>;

    const mpp::NormRelInf<SrcT, ComputeT> op1;
    const mpp::NormInf<SrcT, ComputeT> op2;

    const normInfSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op1, op2);

    DstT dst1 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);
    DstT dst2 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), dst1, dst2, functor);

    // ignore result1 as only the second is a meaningful output for NormRel
    const mpp::NormRelInfPost<DstT> postOp2;
    const mpp::NormRelInfPost<DstT> postOpScalar2;

    postOpScalar2(dst1, dst2, aDstScalar);
    postOp2(dst1, dst2, aDst);
}

template <PixelType T>
void ImageView<T>::NormRelInf(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normInfSrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::NormRelInf<SrcT, ComputeT>,
                                                  mpp::NormInf<SrcT, ComputeT>>;

    const mpp::NormRelInf<SrcT, ComputeT> op1;
    const mpp::NormInf<SrcT, ComputeT> op2;

    const normInfSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op1, op2);

    DstT dst1 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);
    DstT dst2 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), dst1, dst2, functor);

    // ignore result1 as only the second is a meaningful output for NormRel
    const mpp::NormRelInfPost<DstT> postOp2;

    postOp2(dst1, dst2, aDst);
}

template <PixelType T>
void ImageView<T>::NormRelInfMasked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                    const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normInfSrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::NormRelInf<SrcT, ComputeT>,
                                                  mpp::NormInf<SrcT, ComputeT>>;

    const mpp::NormRelInf<SrcT, ComputeT> op1;
    const mpp::NormInf<SrcT, ComputeT> op2;

    const normInfSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op1, op2);

    DstT dst1 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);
    DstT dst2 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), dst1, dst2, functor);

    // ignore result1 as only the second is a meaningful output for NormRel
    const mpp::NormRelInfPost<DstT> postOp2;

    postOp2(dst1, dst2, aDst);
}
#pragma endregion
#pragma region NormRelL1
template <PixelType T>
void ImageView<T>::NormRelL1(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                             double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1SrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::NormRelL1<SrcT, ComputeT>,
                                                 mpp::NormL1<SrcT, ComputeT>>;

    const mpp::NormRelL1<SrcT, ComputeT> op1;
    const mpp::NormL1<SrcT, ComputeT> op2;

    const normL1SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op1, op2);

    DstT dst1 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);
    DstT dst2 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), dst1, dst2, functor);

    // ignore result1 as only the second is a meaningful output for NormRel
    const mpp::NormRelL1Post<DstT> postOp2;
    const mpp::NormRelL1Post<DstT> postOpScalar2;

    postOpScalar2(dst1, dst2, aDstScalar);
    postOp2(dst1, dst2, aDst);
}

template <PixelType T>
void ImageView<T>::NormRelL1Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                   double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1SrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::NormRelL1<SrcT, ComputeT>,
                                                 mpp::NormL1<SrcT, ComputeT>>;

    const mpp::NormRelL1<SrcT, ComputeT> op1;
    const mpp::NormL1<SrcT, ComputeT> op2;

    const normL1SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op1, op2);

    DstT dst1 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);
    DstT dst2 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), dst1, dst2, functor);

    // ignore result1 as only the second is a meaningful output for NormRel
    const mpp::NormRelL1Post<DstT> postOp2;
    const mpp::NormRelL1Post<DstT> postOpScalar2;

    postOpScalar2(dst1, dst2, aDstScalar);
    postOp2(dst1, dst2, aDst);
}

template <PixelType T>
void ImageView<T>::NormRelL1(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1SrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::NormRelL1<SrcT, ComputeT>,
                                                 mpp::NormL1<SrcT, ComputeT>>;

    const mpp::NormRelL1<SrcT, ComputeT> op1;
    const mpp::NormL1<SrcT, ComputeT> op2;

    const normL1SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op1, op2);

    DstT dst1 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);
    DstT dst2 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), dst1, dst2, functor);

    // ignore result1 as only the second is a meaningful output for NormRel
    const mpp::NormRelL1Post<DstT> postOp2;

    postOp2(dst1, dst2, aDst);
}

template <PixelType T>
void ImageView<T>::NormRelL1Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                   const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1SrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::NormRelL1<SrcT, ComputeT>,
                                                 mpp::NormL1<SrcT, ComputeT>>;

    const mpp::NormRelL1<SrcT, ComputeT> op1;
    const mpp::NormL1<SrcT, ComputeT> op2;

    const normL1SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op1, op2);

    DstT dst1 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);
    DstT dst2 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), dst1, dst2, functor);

    // ignore result1 as only the second is a meaningful output for NormRel
    const mpp::NormRelL1Post<DstT> postOp2;

    postOp2(dst1, dst2, aDst);
}
#pragma endregion
#pragma region NormRelL2
template <PixelType T>
void ImageView<T>::NormRelL2(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                             double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::NormRelL2<SrcT, ComputeT>,
                                                 mpp::NormL2<SrcT, ComputeT>>;

    const mpp::NormRelL2<SrcT, ComputeT> op1;
    const mpp::NormL2<SrcT, ComputeT> op2;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op1, op2);

    DstT dst1 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);
    DstT dst2 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), dst1, dst2, functor);

    // ignore result1 as only the second is a meaningful output for NormRel
    const mpp::NormRelL2Post<DstT> postOp2;
    const mpp::NormRelL2Post<DstT> postOpScalar2;

    postOpScalar2(dst1, dst2, aDstScalar);
    postOp2(dst1, dst2, aDst);
}

template <PixelType T>
void ImageView<T>::NormRelL2Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                   double &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::NormRelL2<SrcT, ComputeT>,
                                                 mpp::NormL2<SrcT, ComputeT>>;

    const mpp::NormRelL2<SrcT, ComputeT> op1;
    const mpp::NormL2<SrcT, ComputeT> op2;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op1, op2);

    DstT dst1 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);
    DstT dst2 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), dst1, dst2, functor);

    // ignore result1 as only the second is a meaningful output for NormRel
    const mpp::NormRelL2Post<DstT> postOp2;
    const mpp::NormRelL2Post<DstT> postOpScalar2;

    postOpScalar2(dst1, dst2, aDstScalar);
    postOp2(dst1, dst2, aDst);
}

template <PixelType T>
void ImageView<T>::NormRelL2(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::NormRelL2<SrcT, ComputeT>,
                                                 mpp::NormL2<SrcT, ComputeT>>;

    const mpp::NormRelL2<SrcT, ComputeT> op1;
    const mpp::NormL2<SrcT, ComputeT> op2;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op1, op2);

    DstT dst1 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);
    DstT dst2 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), dst1, dst2, functor);

    // ignore result1 as only the second is a meaningful output for NormRel
    const mpp::NormRelL2Post<DstT> postOp2;

    postOp2(dst1, dst2, aDst);
}

template <PixelType T>
void ImageView<T>::NormRelL2Masked(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                                   const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::NormRelL2<SrcT, ComputeT>,
                                                 mpp::NormL2<SrcT, ComputeT>>;

    const mpp::NormRelL2<SrcT, ComputeT> op1;
    const mpp::NormL2<SrcT, ComputeT> op2;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op1, op2);

    DstT dst1 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);
    DstT dst2 = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), dst1, dst2, functor);

    // ignore result1 as only the second is a meaningful output for NormRel
    const mpp::NormRelL2Post<DstT> postOp2;
    postOp2(dst1, dst2, aDst);
}
#pragma endregion
#pragma region PSNR
template <PixelType T>
void ImageView<T>::PSNR(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                        double &aDstScalar, double aValueRange) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::PSNR<DstT> postOp(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()), aValueRange);
    const mpp::PSNRScalar<DstT> postOpScalar(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()), aValueRange);

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::PSNR(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                        double aValueRange) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2SrcSrc = SrcSrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormDiffL2<SrcT, ComputeT>>;

    const mpp::NormDiffL2<SrcT, ComputeT> op;

    const normL2SrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::PSNR<DstT> postOp(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()), aValueRange);

    postOp(aDst);
}
#pragma endregion

#pragma region NormInf
template <PixelType T>
void ImageView<T>::NormInf(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normInfSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormInf<SrcT, ComputeT>>;

    const mpp::NormInf<SrcT, ComputeT> op;

    const normInfSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::MaxScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormInfMasked(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                                 const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normInfSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormInf<SrcT, ComputeT>>;

    const mpp::NormInf<SrcT, ComputeT> op;

    const normInfSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::MaxScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormInf(same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normInfSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormInf<SrcT, ComputeT>>;

    const mpp::NormInf<SrcT, ComputeT> op;

    const normInfSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormInfMasked(same_vector_size_different_type_t<T, double> &aDst,
                                 const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normInfSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormInf<SrcT, ComputeT>>;

    const mpp::NormInf<SrcT, ComputeT> op;

    const normInfSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}

#pragma endregion
#pragma region NormL1
template <PixelType T>
void ImageView<T>::NormL1(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1Src = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormL1<SrcT, ComputeT>>;

    const mpp::NormL1<SrcT, ComputeT> op;

    const normL1Src functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormL1Masked(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                                const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1Src = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormL1<SrcT, ComputeT>>;

    const mpp::NormL1<SrcT, ComputeT> op;

    const normL1Src functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormL1(same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1Src = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormL1<SrcT, ComputeT>>;

    const mpp::NormL1<SrcT, ComputeT> op;

    const normL1Src functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormL1Masked(same_vector_size_different_type_t<T, double> &aDst,
                                const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL1Src = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormL1<SrcT, ComputeT>>;

    const mpp::NormL1<SrcT, ComputeT> op;

    const normL1Src functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}
#pragma endregion
#pragma region NormL2
template <PixelType T>
void ImageView<T>::NormL2(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2Src = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormL2<SrcT, ComputeT>>;

    const mpp::NormL2<SrcT, ComputeT> op;

    const normL2Src functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::SqrtPostOp<DstT> postOp;
    const mpp::SumThenSqrtScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormL2Masked(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                                const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2Src = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormL2<SrcT, ComputeT>>;

    const mpp::NormL2<SrcT, ComputeT> op;

    const normL2Src functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::SqrtPostOp<DstT> postOp;
    const mpp::SumThenSqrtScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormL2(same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2Src = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormL2<SrcT, ComputeT>>;

    const mpp::NormL2<SrcT, ComputeT> op;

    const normL2Src functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::SqrtPostOp<DstT> postOp;

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::NormL2Masked(same_vector_size_different_type_t<T, double> &aDst,
                                const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using normL2Src = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::NormL2<SrcT, ComputeT>>;

    const mpp::NormL2<SrcT, ComputeT> op;

    const normL2Src functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::SqrtPostOp<DstT> postOp;

    postOp(aDst);
}
#pragma endregion

#pragma region Sum
template <PixelType T>
void ImageView<T>::Sum(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::Sum(same_vector_size_different_type_t<T, c_double> &aDst, c_double &aDstScalar) const
    requires ComplexVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::SumMasked(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                             const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::SumMasked(same_vector_size_different_type_t<T, c_double> &aDst, c_double &aDstScalar,
                             const ImageView<Pixel8uC1> &aMask) const
    requires ComplexVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::Sum(same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::Sum(same_vector_size_different_type_t<T, c_double> &aDst) const
    requires ComplexVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::SumMasked(same_vector_size_different_type_t<T, double> &aDst,
                             const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::SumMasked(same_vector_size_different_type_t<T, c_double> &aDst,
                             const ImageView<Pixel8uC1> &aMask) const
    requires ComplexVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}
#pragma endregion

#pragma region Mean
template <PixelType T>
void ImageView<T>::Mean(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(SizeRoi().TotalSize()));
    const mpp::DivScalar<DstT> postOpScalar(
        static_cast<complex_basetype_t<remove_vector_t<DstT>>>(SizeRoi().TotalSize()));

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::Mean(same_vector_size_different_type_t<T, c_double> &aDst, c_double &aDstScalar) const
    requires ComplexVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(SizeRoi().TotalSize()));
    const mpp::DivScalar<DstT> postOpScalar(
        static_cast<complex_basetype_t<remove_vector_t<DstT>>>(SizeRoi().TotalSize()));

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MeanMasked(same_vector_size_different_type_t<T, double> &aDst, double &aDstScalar,
                              const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(maskpixels));
    const mpp::DivScalar<DstT> postOpScalar(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(maskpixels));

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MeanMasked(same_vector_size_different_type_t<T, c_double> &aDst, c_double &aDstScalar,
                              const ImageView<Pixel8uC1> &aMask) const
    requires ComplexVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(maskpixels));
    const mpp::DivScalar<DstT> postOpScalar(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(maskpixels));

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::Mean(same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(SizeRoi().TotalSize()));

    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::Mean(same_vector_size_different_type_t<T, c_double> &aDst) const
    requires ComplexVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(SizeRoi().TotalSize()));

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MeanMasked(same_vector_size_different_type_t<T, double> &aDst,
                              const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(maskpixels));

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MeanMasked(same_vector_size_different_type_t<T, c_double> &aDst,
                              const ImageView<Pixel8uC1> &aMask) const
    requires ComplexVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT                 = same_vector_size_different_type_t<T, c_double>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::Sum<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::DivPostOp<DstT> postOp(static_cast<complex_basetype_t<remove_vector_t<DstT>>>(maskpixels));

    postOp(aDst);
}
#pragma endregion

#pragma region MeanStd
template <PixelType T>
void ImageView<T>::MeanStd(same_vector_size_different_type_t<T, double> &aMean,
                           // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                           same_vector_size_different_type_t<T, double> &aStd, double &aMeanScalar,
                           double &aStdScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT1                = same_vector_size_different_type_t<T, double>;
    using DstT2                = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSumSqrSrc = SrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::Sum<SrcT, ComputeT>,
                                              mpp::SumSqr<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op1;
    const mpp::SumSqr<SrcT, ComputeT> op2;

    const sumSumSqrSrc functor(PointerRoi(), Pitch(), op1, op2);

    aMean = DstT1(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);
    DstT1 sumSqr(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);

    reduction(SizeRoi(), aMean, sumSqr, functor);

    const mpp::DivPostOp<DstT1> postOp1(static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(SizeRoi().TotalSize()));
    const mpp::StdDeviation<DstT2> postOp2(static_cast<remove_vector_t<DstT2>>(SizeRoi().TotalSize()));
    const mpp::DivScalar<DstT1> postOpScalar1(
        static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(SizeRoi().TotalSize()));
    const mpp::StdDeviation<DstT2> postOpScalar2((static_cast<remove_vector_t<DstT2>>(SizeRoi().TotalSize())));

    postOpScalar2(aMean, sumSqr, aStdScalar);
    aMeanScalar = postOpScalar1(aMean);
    postOp2(aMean, sumSqr, aStd);
    postOp1(aMean);
}
template <PixelType T>
void ImageView<T>::MeanStd(same_vector_size_different_type_t<T, c_double> &aMean,
                           same_vector_size_different_type_t<T, double> &aStd, c_double &aMeanScalar,
                           double &aStdScalar) const
    requires ComplexVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT1                = same_vector_size_different_type_t<T, c_double>;
    using DstT2                = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSumSqrSrc = SrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::Sum<SrcT, ComputeT>,
                                              mpp::SumSqr<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op1;
    const mpp::SumSqr<SrcT, ComputeT> op2;

    const sumSumSqrSrc functor(PointerRoi(), Pitch(), op1, op2);

    aMean = DstT1(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);
    DstT1 sumSqr(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);

    reduction(SizeRoi(), aMean, sumSqr, functor);

    const mpp::DivPostOp<DstT1> postOp1(static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(SizeRoi().TotalSize()));
    const mpp::StdDeviation<DstT2> postOp2(static_cast<remove_vector_t<DstT2>>(SizeRoi().TotalSize()));
    const mpp::DivScalar<DstT1> postOpScalar1(
        static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(SizeRoi().TotalSize()));
    const mpp::StdDeviation<DstT2> postOpScalar2((static_cast<remove_vector_t<DstT2>>(SizeRoi().TotalSize())));

    postOpScalar2(aMean, sumSqr, aStdScalar);
    aMeanScalar = postOpScalar1(aMean);
    postOp2(aMean, sumSqr, aStd);
    postOp1(aMean);
}

template <PixelType T>
void ImageView<T>::MeanStdMasked(same_vector_size_different_type_t<T, double> &aMean,
                                 // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                 same_vector_size_different_type_t<T, double> &aStd, double &aMeanScalar,
                                 double &aStdScalar, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT1                = same_vector_size_different_type_t<T, double>;
    using DstT2                = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSumSqrSrc = SrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::Sum<SrcT, ComputeT>,
                                              mpp::SumSqr<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op1;
    const mpp::SumSqr<SrcT, ComputeT> op2;

    const sumSumSqrSrc functor(PointerRoi(), Pitch(), op1, op2);

    aMean = DstT1(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);
    DstT1 sumSqr(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aMean, sumSqr, functor);

    const mpp::DivPostOp<DstT1> postOp1(static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(maskpixels));
    const mpp::StdDeviation<DstT2> postOp2(static_cast<remove_vector_t<DstT2>>(maskpixels));
    const mpp::DivScalar<DstT1> postOpScalar1(static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(maskpixels));
    const mpp::StdDeviation<DstT2> postOpScalar2((static_cast<remove_vector_t<DstT2>>(maskpixels)));

    postOpScalar2(aMean, sumSqr, aStdScalar);
    aMeanScalar = postOpScalar1(aMean);
    postOp2(aMean, sumSqr, aStd);
    postOp1(aMean);
}
template <PixelType T>
void ImageView<T>::MeanStdMasked(same_vector_size_different_type_t<T, c_double> &aMean,
                                 same_vector_size_different_type_t<T, double> &aStd, c_double &aMeanScalar,
                                 double &aStdScalar, const ImageView<Pixel8uC1> &aMask) const
    requires ComplexVector<T> && (vector_active_size_v<T> > 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT1                = same_vector_size_different_type_t<T, c_double>;
    using DstT2                = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSumSqrSrc = SrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::Sum<SrcT, ComputeT>,
                                              mpp::SumSqr<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op1;
    const mpp::SumSqr<SrcT, ComputeT> op2;

    const sumSumSqrSrc functor(PointerRoi(), Pitch(), op1, op2);

    aMean = DstT1(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);
    DstT1 sumSqr(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aMean, sumSqr, functor);

    const mpp::DivPostOp<DstT1> postOp1(static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(maskpixels));
    const mpp::StdDeviation<DstT2> postOp2(static_cast<remove_vector_t<DstT2>>(maskpixels));
    const mpp::DivScalar<DstT1> postOpScalar1(static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(maskpixels));
    const mpp::StdDeviation<DstT2> postOpScalar2((static_cast<remove_vector_t<DstT2>>(maskpixels)));

    postOpScalar2(aMean, sumSqr, aStdScalar);
    aMeanScalar = postOpScalar1(aMean);
    postOp2(aMean, sumSqr, aStd);
    postOp1(aMean);
}

template <PixelType T>
void ImageView<T>::MeanStd(same_vector_size_different_type_t<T, double> &aMean,
                           same_vector_size_different_type_t<T, double> &aStd) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT1                = same_vector_size_different_type_t<T, double>;
    using DstT2                = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSumSqrSrc = SrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::Sum<SrcT, ComputeT>,
                                              mpp::SumSqr<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op1;
    const mpp::SumSqr<SrcT, ComputeT> op2;

    const sumSumSqrSrc functor(PointerRoi(), Pitch(), op1, op2);

    aMean = DstT1(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);
    DstT1 sumSqr(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);

    reduction(SizeRoi(), aMean, sumSqr, functor);

    const mpp::DivPostOp<DstT1> postOp1(static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(SizeRoi().TotalSize()));
    const mpp::StdDeviation<DstT2> postOp2(static_cast<remove_vector_t<DstT2>>(SizeRoi().TotalSize()));

    postOp2(aMean, sumSqr, aStd);
    postOp1(aMean);
}
template <PixelType T>
void ImageView<T>::MeanStd(same_vector_size_different_type_t<T, c_double> &aMean,
                           same_vector_size_different_type_t<T, double> &aStd) const
    requires ComplexVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT1                = same_vector_size_different_type_t<T, c_double>;
    using DstT2                = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSumSqrSrc = SrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::Sum<SrcT, ComputeT>,
                                              mpp::SumSqr<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op1;
    const mpp::SumSqr<SrcT, ComputeT> op2;

    const sumSumSqrSrc functor(PointerRoi(), Pitch(), op1, op2);

    aMean = DstT1(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);
    DstT1 sumSqr(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);

    reduction(SizeRoi(), aMean, sumSqr, functor);

    const mpp::DivPostOp<DstT1> postOp1(static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(SizeRoi().TotalSize()));
    const mpp::StdDeviation<DstT2> postOp2(static_cast<remove_vector_t<DstT2>>(SizeRoi().TotalSize()));

    postOp2(aMean, sumSqr, aStd);
    postOp1(aMean);
}

template <PixelType T>
void ImageView<T>::MeanStdMasked(same_vector_size_different_type_t<T, double> &aMean,
                                 same_vector_size_different_type_t<T, double> &aStd,
                                 const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT1                = same_vector_size_different_type_t<T, double>;
    using DstT2                = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSumSqrSrc = SrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::Sum<SrcT, ComputeT>,
                                              mpp::SumSqr<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op1;
    const mpp::SumSqr<SrcT, ComputeT> op2;

    const sumSumSqrSrc functor(PointerRoi(), Pitch(), op1, op2);

    aMean = DstT1(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);
    DstT1 sumSqr(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aMean, sumSqr, functor);

    const mpp::DivPostOp<DstT1> postOp1(static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(maskpixels));
    const mpp::StdDeviation<DstT2> postOp2(static_cast<remove_vector_t<DstT2>>(maskpixels));

    postOp2(aMean, sumSqr, aStd);
    postOp1(aMean);
}
template <PixelType T>
void ImageView<T>::MeanStdMasked(same_vector_size_different_type_t<T, c_double> &aMean,
                                 same_vector_size_different_type_t<T, double> &aStd,
                                 const ImageView<Pixel8uC1> &aMask) const
    requires ComplexVector<T> && (vector_active_size_v<T> == 1)
{
    checkSameSize(ROI(), aMask.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, c_double>;
    using DstT1                = same_vector_size_different_type_t<T, c_double>;
    using DstT2                = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using sumSumSqrSrc = SrcReduction2Functor<TupelSize, SrcT, ComputeT, ComputeT, mpp::Sum<SrcT, ComputeT>,
                                              mpp::SumSqr<SrcT, ComputeT>>;

    const mpp::Sum<SrcT, ComputeT> op1;
    const mpp::SumSqr<SrcT, ComputeT> op2;

    const sumSumSqrSrc functor(PointerRoi(), Pitch(), op1, op2);

    aMean = DstT1(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);
    DstT1 sumSqr(reduction_init_value_v<ReductionInitValue::Zero, DstT1>);

    const size_t maskpixels = reduction(aMask, SizeRoi(), aMean, sumSqr, functor);

    const mpp::DivPostOp<DstT1> postOp1(static_cast<complex_basetype_t<remove_vector_t<DstT1>>>(maskpixels));
    const mpp::StdDeviation<DstT2> postOp2(static_cast<remove_vector_t<DstT2>>(maskpixels));

    postOp2(aMean, sumSqr, aStd);
    postOp1(aMean);
}
#pragma endregion

#pragma region CountInRange
template <PixelType T>
void ImageView<T>::CountInRange(const T &aLowerLimit, const T &aUpperLimit,
                                same_vector_size_different_type_t<T, size_t> &aDst, size_t &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, size_t>;
    using DstT                 = same_vector_size_different_type_t<T, size_t>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::CountInRange<SrcT>>;

    const mpp::CountInRange<SrcT> op(aLowerLimit, aUpperLimit);

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::CountInRangeMasked(const T &aLowerLimit, const T &aUpperLimit,
                                      same_vector_size_different_type_t<T, size_t> &aDst, size_t &aDstScalar,
                                      const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, size_t>;
    using DstT                 = same_vector_size_different_type_t<T, size_t>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::CountInRange<SrcT>>;

    const mpp::CountInRange<SrcT> op(aLowerLimit, aUpperLimit);

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;
    const mpp::SumScalar<DstT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}
template <PixelType T>
void ImageView<T>::CountInRange(const T &aLowerLimit, const T &aUpperLimit,
                                same_vector_size_different_type_t<T, size_t> &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, size_t>;
    using DstT                 = same_vector_size_different_type_t<T, size_t>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::CountInRange<SrcT>>;

    const mpp::CountInRange<SrcT> op(aLowerLimit, aUpperLimit);

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::CountInRangeMasked(const T &aLowerLimit, const T &aUpperLimit,
                                      same_vector_size_different_type_t<T, size_t> &aDst,
                                      const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, size_t>;
    using DstT                 = same_vector_size_different_type_t<T, size_t>;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::CountInRange<SrcT>>;

    const mpp::CountInRange<SrcT> op(aLowerLimit, aUpperLimit);

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Zero, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<DstT> postOp;

    postOp(aDst);
}
#pragma endregion

#pragma region QualityIndex
template <PixelType T>
void ImageView<T>::QualityIndex(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = same_vector_size_different_type_t<T, double>;
    using DstT                 = same_vector_size_different_type_t<T, double>;
    constexpr size_t TupelSize = 1;

    using qualityIndexSrcSrc =
        SrcSrcReduction5Functor<TupelSize, SrcT, ComputeT, ComputeT, ComputeT, ComputeT, ComputeT,
                                mpp::Sum1Or2<SrcT, ComputeT, 1>, mpp::SumSqr1Or2<SrcT, ComputeT, 1>,
                                mpp::Sum1Or2<SrcT, ComputeT, 2>, mpp::SumSqr1Or2<SrcT, ComputeT, 2>,
                                mpp::DotProduct<SrcT, ComputeT>>;

    const mpp::Sum1Or2<SrcT, ComputeT, 1> opSum1;
    const mpp::SumSqr1Or2<SrcT, ComputeT, 1> opSumSqr1;
    const mpp::Sum1Or2<SrcT, ComputeT, 2> opSum2;
    const mpp::SumSqr1Or2<SrcT, ComputeT, 2> opSumSqr2;
    const mpp::DotProduct<SrcT, ComputeT> opDotProduct;

    const qualityIndexSrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), opSum1, opSumSqr1,
                                     opSum2, opSumSqr2, opDotProduct);
    DstT dst1(0);
    DstT dst2(0);
    DstT dst3(0);
    DstT dst4(0);
    DstT dst5(0);
    reduction(SizeRoi(), dst1, dst2, dst3, dst4, dst5, functor);

    const mpp::QualityIndex<DstT> postOp(static_cast<remove_vector_t<DstT>>(SizeRoi().TotalSize()));

    postOp(dst1, dst2, dst3, dst4, dst5, aDst);
}
#pragma endregion

#pragma region SSIM
template <PixelType T>
void ImageView<T>::SSIM(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                        double aDynamicRange, double aK1, double aK2) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                  = T;
    using ComputeT              = same_vector_size_different_type_t<T, double>;
    using DstT                  = same_vector_size_different_type_t<T, double>;
    constexpr size_t FilterSize = 11;

    const float scaleFactor = std::max(1.0f, std::round(to_float(SizeRoi().Min()) / 256.0f));

    const mpp::SSIM<DstT> postOp(aDynamicRange, aK1, aK2);
    std::vector<double> ssimFilter(FilterSize * FilterSize, 0);

    for (size_t y = 0; y < FilterSize; y++)
    {
        for (size_t x = 0; x < FilterSize; x++)
        {
            const size_t idx = y * FilterSize + x;
            ssimFilter[idx]  = to_double(FixedFilterKernelSSIM::ValuesSeparable[x]) * // NOLINT
                              to_double(FixedFilterKernelSSIM::ValuesSeparable[y]);   // NOLINT
        }
    }

    if (scaleFactor > 1)
    {
        const Size2D scaledRoi = SizeRoi() / to_int(scaleFactor);

        Image<SrcT> resizeSrc1(scaledRoi);
        Image<SrcT> resizeSrc2(scaledRoi);
        Image<DstT> localSSIM(scaledRoi);

        this->Resize(resizeSrc1, 1.0 / to_double(scaleFactor), 0, InterpolationMode::Super, BorderType::Replicate,
                     Roi());
        aSrc2.Resize(resizeSrc2, 1.0 / to_double(scaleFactor), 0, InterpolationMode::Super, BorderType::Replicate,
                     Roi());

        ssimEachPixel<T, ComputeT, DstT, double, mpp::SSIM<DstT>>(resizeSrc1, resizeSrc2, localSSIM, ssimFilter.data(),
                                                                  FilterSize, BorderType::Replicate, resizeSrc1.ROI(),
                                                                  resizeSrc2.ROI(), postOp);
        if constexpr (vector_active_size_v<T> > 1)
        {
            double unused = 0;
            localSSIM.Mean(aDst, unused);
        }
        else
        {
            localSSIM.Mean(aDst);
        }
    }
    else
    {
        Image<DstT> localSSIM(SizeRoi());
        ssimEachPixel<T, ComputeT, DstT, double, mpp::SSIM<DstT>>(
            *this, aSrc2, localSSIM, ssimFilter.data(), FilterSize, BorderType::Replicate, ROI(), aSrc2.ROI(), postOp);

        if constexpr (vector_active_size_v<T> > 1)
        {
            double unused = 0;
            localSSIM.Mean(aDst, unused);
        }
        else
        {
            localSSIM.Mean(aDst);
        }
    }
}
#pragma endregion

#pragma region MSSSIM

template <PixelType T>
void ImageView<T>::MSSSIM(const ImageView<T> &aSrc2, same_vector_size_different_type_t<T, double> &aDst,
                          double aDynamicRange, double aK1, double aK2) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using ComputeT = same_vector_size_different_type_t<T, double>;
    using DstT     = same_vector_size_different_type_t<T, double>;

    constexpr size_t FilterSize = 11;

    const Size2D scaledRoi1 = SizeRoi() / 2;
    const Size2D scaledRoi2 = scaledRoi1 / 2;

    Image<DstT> localSSIM(SizeRoi());
    Image<T> resizeSrc11(scaledRoi1);
    Image<T> resizeSrc12(scaledRoi1);
    Image<T> resizeSrc21(scaledRoi2);
    Image<T> resizeSrc22(scaledRoi2);

    Size2D resizedRoiSize = SizeRoi();

    const mpp::SSIM<DstT> postOpSSIM(aDynamicRange, aK1, aK2);
    const mpp::MSSSIM<DstT> postOpMSSSIM(aDynamicRange, aK2);
    std::vector<double> ssimFilter(FilterSize * FilterSize, 0);

    for (size_t y = 0; y < FilterSize; y++)
    {
        for (size_t x = 0; x < FilterSize; x++)
        {
            const size_t idx = y * FilterSize + x;
            ssimFilter[idx]  = to_double(FixedFilterKernelSSIM::ValuesSeparable[x]) * // NOLINT
                              to_double(FixedFilterKernelSSIM::ValuesSeparable[y]);   // NOLINT
        }
    }

    static constexpr std::array<double, 5> weights = {0.0448, 0.2856, 0.3000, 0.2363, 0.1333}; // sums up to 1
    std::array<DstT, 5> results{};

    // level 0
    ssimEachPixel<T, ComputeT, DstT, double, mpp::MSSSIM<DstT>>(*this, aSrc2, localSSIM, ssimFilter.data(), FilterSize,
                                                                BorderType::Replicate, ROI(), aSrc2.ROI(),
                                                                postOpMSSSIM);

    if constexpr (vector_active_size_v<T> > 1)
    {
        double unused = 0;
        localSSIM.Mean(results[0], unused);
    }
    else
    {
        localSSIM.Mean(results[0]);
    }

    // level 1
    resizedRoiSize /= 2;

    this->Resize(resizeSrc11, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi());
    aSrc2.Resize(resizeSrc12, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi());

    localSSIM.SetRoi(Roi({0}, resizedRoiSize));
    ssimEachPixel<T, ComputeT, DstT, double, mpp::MSSSIM<DstT>>(resizeSrc11, resizeSrc12, localSSIM, ssimFilter.data(),
                                                                FilterSize, BorderType::Replicate, resizeSrc11.ROI(),
                                                                resizeSrc12.ROI(), postOpMSSSIM);

    if constexpr (vector_active_size_v<T> > 1)
    {
        double unused = 0;
        localSSIM.Mean(results[1], unused);
    }
    else
    {
        localSSIM.Mean(results[1]);
    }

    // level 2
    resizedRoiSize /= 2;
    resizeSrc11.Resize(resizeSrc21, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi());
    resizeSrc12.Resize(resizeSrc22, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi());

    localSSIM.SetRoi(Roi({0}, resizedRoiSize));
    ssimEachPixel<T, ComputeT, DstT, double, mpp::MSSSIM<DstT>>(resizeSrc21, resizeSrc22, localSSIM, ssimFilter.data(),
                                                                FilterSize, BorderType::Replicate, resizeSrc21.ROI(),
                                                                resizeSrc22.ROI(), postOpMSSSIM);

    if constexpr (vector_active_size_v<T> > 1)
    {
        double unused = 0;
        localSSIM.Mean(results[2], unused);
    }
    else
    {
        localSSIM.Mean(results[2]);
    }

    // level 3
    resizedRoiSize /= 2;
    resizeSrc11.SetRoi(Roi({0}, resizedRoiSize));
    resizeSrc12.SetRoi(Roi({0}, resizedRoiSize));
    localSSIM.SetRoi(Roi({0}, resizedRoiSize));

    resizeSrc21.Resize(resizeSrc11, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi());
    resizeSrc22.Resize(resizeSrc12, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi());

    ssimEachPixel<T, ComputeT, DstT, double, mpp::MSSSIM<DstT>>(resizeSrc11, resizeSrc12, localSSIM, ssimFilter.data(),
                                                                FilterSize, BorderType::Replicate, resizeSrc11.ROI(),
                                                                resizeSrc12.ROI(), postOpMSSSIM);

    if constexpr (vector_active_size_v<T> > 1)
    {
        double unused = 0;
        localSSIM.Mean(results[3], unused);
    }
    else
    {
        localSSIM.Mean(results[3]);
    }

    // level 4
    resizedRoiSize /= 2;
    resizeSrc21.SetRoi(Roi({0}, resizedRoiSize));
    resizeSrc22.SetRoi(Roi({0}, resizedRoiSize));
    localSSIM.SetRoi(Roi({0}, resizedRoiSize));

    resizeSrc11.Resize(resizeSrc21, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi());
    resizeSrc12.Resize(resizeSrc22, 0.5, 0, InterpolationMode::Super, BorderType::Replicate, Roi());

    ssimEachPixel<T, ComputeT, DstT, double, mpp::SSIM<DstT>>(resizeSrc21, resizeSrc22, localSSIM, ssimFilter.data(),
                                                              FilterSize, BorderType::Replicate, resizeSrc21.ROI(),
                                                              resizeSrc22.ROI(), postOpSSIM);

    if constexpr (vector_active_size_v<T> > 1)
    {
        double unused = 0;
        localSSIM.Mean(results[4], unused);
    }
    else
    {
        localSSIM.Mean(results[4]);
    }

    aDst = DstT(0);
    for (size_t i = 0; i < results.size(); i++)
    {
        aDst += results[i] * weights[i]; // NOLINT
    }
}
#pragma endregion

#pragma region Min
template <PixelType T>
void ImageView<T>::Min(T &aDst, remove_vector_t<T> &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::MinRed<SrcT>>;

    const mpp::MinRed<SrcT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Max, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<SrcT> postOp;
    const mpp::MinScalar<SrcT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MinMasked(T &aDst, remove_vector_t<T> &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::MinRed<SrcT>>;

    const mpp::MinRed<SrcT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Max, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<SrcT> postOp;
    const mpp::MinScalar<SrcT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::Min(T &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::MinRed<SrcT>>;

    const mpp::MinRed<SrcT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Max, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<SrcT> postOp;

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MinMasked(T &aDst, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::MinRed<SrcT>>;

    const mpp::MinRed<SrcT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Max, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<SrcT> postOp;

    postOp(aDst);
}
#pragma endregion
#pragma region Max
template <PixelType T>
void ImageView<T>::Max(T &aDst, remove_vector_t<T> &aDstScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::MaxRed<SrcT>>;

    const mpp::MaxRed<SrcT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Min, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<SrcT> postOp;
    const mpp::MaxScalar<SrcT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MaxMasked(T &aDst, remove_vector_t<T> &aDstScalar, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::MaxRed<SrcT>>;

    const mpp::MaxRed<SrcT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Min, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<SrcT> postOp;
    const mpp::MaxScalar<SrcT> postOpScalar;

    aDstScalar = postOpScalar(aDst);
    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::Max(T &aDst) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::MaxRed<SrcT>>;

    const mpp::MaxRed<SrcT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Min, DstT>);

    reduction(SizeRoi(), aDst, functor);

    const mpp::Nothing<SrcT> postOp;

    postOp(aDst);
}

template <PixelType T>
void ImageView<T>::MaxMasked(T &aDst, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using sumSrc = SrcReductionFunctor<TupelSize, SrcT, ComputeT, mpp::MaxRed<SrcT>>;

    const mpp::MaxRed<SrcT> op;

    const sumSrc functor(PointerRoi(), Pitch(), op);

    aDst = DstT(reduction_init_value_v<ReductionInitValue::Min, DstT>);

    reduction(aMask, SizeRoi(), aDst, functor);

    const mpp::Nothing<SrcT> postOp;

    postOp(aDst);
}
#pragma endregion
#pragma region MinMax
template <PixelType T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ImageView<T>::MinMax(T &aDstMin, T &aDstMax, remove_vector_t<T> &aDstMinScalar,
                          remove_vector_t<T> &aDstMaxScalar) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    constexpr size_t TupelSize = 1;

    using minMaxSrc = SrcReduction2Functor<TupelSize, SrcT, SrcT, SrcT, mpp::MinRed<SrcT>, mpp::MaxRed<SrcT>>;

    const mpp::MinRed<SrcT> op1;
    const mpp::MaxRed<SrcT> op2;

    const minMaxSrc functor(PointerRoi(), Pitch(), op1, op2);

    aDstMin = reduction_init_value_v<ReductionInitValue::Max, T>;
    aDstMax = reduction_init_value_v<ReductionInitValue::Min, T>;

    reduction(SizeRoi(), aDstMin, aDstMax, functor);

    const mpp::Nothing<SrcT> postOp1;
    const mpp::Nothing<SrcT> postOp2;
    const mpp::MinScalar<SrcT> postOpScalar1;
    const mpp::MaxScalar<SrcT> postOpScalar2;

    postOpScalar2(aDstMax, aDstMax, aDstMaxScalar);
    aDstMinScalar = postOpScalar1(aDstMin);
    postOp2(aDstMax);
    postOp1(aDstMin);
}
template <PixelType T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ImageView<T>::MinMaxMasked(T &aDstMin, T &aDstMax, remove_vector_t<T> &aDstMinScalar,
                                remove_vector_t<T> &aDstMaxScalar, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    constexpr size_t TupelSize = 1;

    using minMaxSrc = SrcReduction2Functor<TupelSize, SrcT, SrcT, SrcT, mpp::MinRed<SrcT>, mpp::MaxRed<SrcT>>;

    const mpp::MinRed<SrcT> op1;
    const mpp::MaxRed<SrcT> op2;

    const minMaxSrc functor(PointerRoi(), Pitch(), op1, op2);

    aDstMin = reduction_init_value_v<ReductionInitValue::Max, T>;
    aDstMax = reduction_init_value_v<ReductionInitValue::Min, T>;

    reduction(aMask, SizeRoi(), aDstMin, aDstMax, functor);

    const mpp::Nothing<SrcT> postOp1;
    const mpp::Nothing<SrcT> postOp2;
    const mpp::MinScalar<SrcT> postOpScalar1;
    const mpp::MaxScalar<SrcT> postOpScalar2;

    postOpScalar2(aDstMax, aDstMax, aDstMaxScalar);
    aDstMinScalar = postOpScalar1(aDstMin);
    postOp2(aDstMax);
    postOp1(aDstMin);
}

template <PixelType T>
void ImageView<T>::MinMax(T &aDstMin, T &aDstMax) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    constexpr size_t TupelSize = 1;

    using minMaxSrc = SrcReduction2Functor<TupelSize, SrcT, SrcT, SrcT, mpp::MinRed<SrcT>, mpp::MaxRed<SrcT>>;

    const mpp::MinRed<SrcT> op1;
    const mpp::MaxRed<SrcT> op2;

    const minMaxSrc functor(PointerRoi(), Pitch(), op1, op2);

    aDstMin = reduction_init_value_v<ReductionInitValue::Max, T>;
    aDstMax = reduction_init_value_v<ReductionInitValue::Min, T>;

    reduction(SizeRoi(), aDstMin, aDstMax, functor);

    const mpp::Nothing<SrcT> postOp1;
    const mpp::Nothing<SrcT> postOp2;

    postOp2(aDstMax);
    postOp1(aDstMin);
}
template <PixelType T>
void ImageView<T>::MinMaxMasked(T &aDstMin, T &aDstMax, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    constexpr size_t TupelSize = 1;

    using minMaxSrc = SrcReduction2Functor<TupelSize, SrcT, SrcT, SrcT, mpp::MinRed<SrcT>, mpp::MaxRed<SrcT>>;

    const mpp::MinRed<SrcT> op1;
    const mpp::MaxRed<SrcT> op2;

    const minMaxSrc functor(PointerRoi(), Pitch(), op1, op2);

    aDstMin = reduction_init_value_v<ReductionInitValue::Max, T>;
    aDstMax = reduction_init_value_v<ReductionInitValue::Min, T>;

    reduction(aMask, SizeRoi(), aDstMin, aDstMax, functor);

    const mpp::Nothing<SrcT> postOp1;
    const mpp::Nothing<SrcT> postOp2;

    postOp2(aDstMax);
    postOp1(aDstMin);
}
#pragma endregion
#pragma region MinIndex
template <PixelType T>
void ImageView<T>::MinIndex(T &aDstMin, same_vector_size_different_type_t<T, int> &aDstIndexX,
                            same_vector_size_different_type_t<T, int> &aDstIndexY, remove_vector_t<T> &aDstMinScalar,
                            Vector3<int> &aDstScalarIdx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using idxT                 = same_vector_size_different_type_t<SrcT, int>;
    constexpr size_t TupelSize = 1;

    const mpp::MinIdx<SrcT> redOpMin;
    const SrcReductionMinIdxFunctor<TupelSize, SrcT> functor(PointerRoi(), Pitch());

    std::vector<T> tempVal(to_size_t(SizeRoi().y));
    std::vector<idxT> tempIdxX(to_size_t(SizeRoi().y));

    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        SrcT resultMin(reduction_init_value_v<ReductionInitValue::Max, SrcT>);
        idxT resultMinIdx(INT_MAX);

        for (int pixelX = 0; pixelX < SizeRoi().x; pixelX++)
        {
            functor(pixelX, pixelY, resultMin, resultMinIdx);
        }

        tempVal[to_size_t(pixelY)]  = resultMin;
        tempIdxX[to_size_t(pixelY)] = resultMinIdx;
    }

    SrcT resultMinY(reduction_init_value_v<ReductionInitValue::Max, SrcT>);
    idxT resultMinIdxY(-1);
    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        const SrcT pxMin = tempVal[to_size_t(pixelY)];

        redOpMin(pxMin.x, pixelY, resultMinY.x, resultMinIdxY.x);
        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            redOpMin(pxMin.y, pixelY, resultMinY.y, resultMinIdxY.y);
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            redOpMin(pxMin.z, pixelY, resultMinY.z, resultMinIdxY.z);
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            redOpMin(pxMin.w, pixelY, resultMinY.w, resultMinIdxY.w);
        }
    }
    // fetch X coordinates:
    idxT minIdxX;
    minIdxX.x = tempIdxX[to_size_t(resultMinIdxY.x)].x;
    if constexpr (vector_active_size_v<SrcT> > 1)
    {
        minIdxX.y = tempIdxX[to_size_t(resultMinIdxY.y)].y;
    }
    if constexpr (vector_active_size_v<SrcT> > 2)
    {
        minIdxX.z = tempIdxX[to_size_t(resultMinIdxY.z)].z;
    }
    if constexpr (vector_active_size_v<SrcT> > 3)
    {
        minIdxX.w = tempIdxX[to_size_t(resultMinIdxY.w)].w;
    }

    int minIdxVec                   = 0;
    remove_vector_t<SrcT> minScalar = resultMinY.x;

    if constexpr (vector_active_size_v<SrcT> > 1)
    {
        redOpMin(resultMinY.y, 1, minScalar, minIdxVec);
    }
    if constexpr (vector_active_size_v<SrcT> > 2)
    {
        redOpMin(resultMinY.z, 2, minScalar, minIdxVec);
    }
    if constexpr (vector_active_size_v<SrcT> > 3)
    {
        redOpMin(resultMinY.w, 3, minScalar, minIdxVec);
    }

    const Vector3<int> minIdx(minIdxX[Channel(minIdxVec)], resultMinIdxY[Channel(minIdxVec)], minIdxVec);
    aDstMin       = resultMinY;
    aDstIndexY    = resultMinIdxY;
    aDstIndexX    = minIdxX;
    aDstMinScalar = minScalar;
    aDstScalarIdx = minIdx;
}
template <PixelType T>
void ImageView<T>::MinIndexMasked(T &aDstMin, same_vector_size_different_type_t<T, int> &aDstIndexX,
                                  same_vector_size_different_type_t<T, int> &aDstIndexY,
                                  remove_vector_t<T> &aDstMinScalar, Vector3<int> &aDstScalarIdx,
                                  const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using idxT                 = same_vector_size_different_type_t<SrcT, int>;
    constexpr size_t TupelSize = 1;

    const mpp::MinIdx<SrcT> redOpMin;
    const SrcReductionMinIdxFunctor<TupelSize, SrcT> functor(PointerRoi(), Pitch());

    std::vector<T> tempVal(to_size_t(SizeRoi().y));
    std::vector<idxT> tempIdxX(to_size_t(SizeRoi().y));

    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        SrcT resultMin(reduction_init_value_v<ReductionInitValue::Max, SrcT>);
        idxT resultMinIdx(INT_MAX);

        for (int pixelX = 0; pixelX < SizeRoi().x; pixelX++)
        {
            if (aMask(pixelX, pixelY) > 0)
            {
                functor(pixelX, pixelY, resultMin, resultMinIdx);
            }
        }

        tempVal[to_size_t(pixelY)]  = resultMin;
        tempIdxX[to_size_t(pixelY)] = resultMinIdx;
    }

    SrcT resultMinY(reduction_init_value_v<ReductionInitValue::Max, SrcT>);
    idxT resultMinIdxY(-1);
    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        const SrcT pxMin = tempVal[to_size_t(pixelY)];

        redOpMin(pxMin.x, pixelY, resultMinY.x, resultMinIdxY.x);
        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            redOpMin(pxMin.y, pixelY, resultMinY.y, resultMinIdxY.y);
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            redOpMin(pxMin.z, pixelY, resultMinY.z, resultMinIdxY.z);
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            redOpMin(pxMin.w, pixelY, resultMinY.w, resultMinIdxY.w);
        }
    }
    // fetch X coordinates:
    idxT minIdxX;
    minIdxX.x = tempIdxX[to_size_t(resultMinIdxY.x)].x;
    if constexpr (vector_active_size_v<SrcT> > 1)
    {
        minIdxX.y = tempIdxX[to_size_t(resultMinIdxY.y)].y;
    }
    if constexpr (vector_active_size_v<SrcT> > 2)
    {
        minIdxX.z = tempIdxX[to_size_t(resultMinIdxY.z)].z;
    }
    if constexpr (vector_active_size_v<SrcT> > 3)
    {
        minIdxX.w = tempIdxX[to_size_t(resultMinIdxY.w)].w;
    }

    int minIdxVec                   = 0;
    remove_vector_t<SrcT> minScalar = resultMinY.x;

    if constexpr (vector_active_size_v<SrcT> > 1)
    {
        redOpMin(resultMinY.y, 1, minScalar, minIdxVec);
    }
    if constexpr (vector_active_size_v<SrcT> > 2)
    {
        redOpMin(resultMinY.z, 2, minScalar, minIdxVec);
    }
    if constexpr (vector_active_size_v<SrcT> > 3)
    {
        redOpMin(resultMinY.w, 3, minScalar, minIdxVec);
    }

    const Vector3<int> minIdx(minIdxX[Channel(minIdxVec)], resultMinIdxY[Channel(minIdxVec)], minIdxVec);
    aDstMin       = resultMinY;
    aDstIndexY    = resultMinIdxY;
    aDstIndexX    = minIdxX;
    aDstMinScalar = minScalar;
    aDstScalarIdx = minIdx;
}

template <PixelType T>
void ImageView<T>::MinIndex(T &aDstMin, same_vector_size_different_type_t<T, int> &aDstIndexX,
                            same_vector_size_different_type_t<T, int> &aDstIndexY) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using idxT                 = same_vector_size_different_type_t<SrcT, int>;
    constexpr size_t TupelSize = 1;

    const mpp::MinIdx<SrcT> redOpMin;
    const SrcReductionMinIdxFunctor<TupelSize, SrcT> functor(PointerRoi(), Pitch());

    std::vector<T> tempVal(to_size_t(SizeRoi().y));
    std::vector<idxT> tempIdxX(to_size_t(SizeRoi().y));

    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        SrcT resultMin(reduction_init_value_v<ReductionInitValue::Max, SrcT>);
        idxT resultMinIdx(INT_MAX);

        for (int pixelX = 0; pixelX < SizeRoi().x; pixelX++)
        {
            functor(pixelX, pixelY, resultMin, resultMinIdx);
        }

        tempVal[to_size_t(pixelY)]  = resultMin;
        tempIdxX[to_size_t(pixelY)] = resultMinIdx;
    }

    SrcT resultMinY(reduction_init_value_v<ReductionInitValue::Max, SrcT>);
    idxT resultMinIdxY(-1);
    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        const SrcT pxMin = tempVal[to_size_t(pixelY)];

        redOpMin(pxMin.x, pixelY, resultMinY.x, resultMinIdxY.x);
    }
    // fetch X coordinates:
    idxT minIdxX;
    minIdxX.x = tempIdxX[to_size_t(resultMinIdxY.x)].x;

    aDstMin    = resultMinY;
    aDstIndexY = resultMinIdxY;
    aDstIndexX = minIdxX;
}
template <PixelType T>
void ImageView<T>::MinIndexMasked(T &aDstMin, same_vector_size_different_type_t<T, int> &aDstIndexX,
                                  same_vector_size_different_type_t<T, int> &aDstIndexY,
                                  const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using idxT                 = same_vector_size_different_type_t<SrcT, int>;
    constexpr size_t TupelSize = 1;

    const mpp::MinIdx<SrcT> redOpMin;
    const SrcReductionMinIdxFunctor<TupelSize, SrcT> functor(PointerRoi(), Pitch());

    std::vector<T> tempVal(to_size_t(SizeRoi().y));
    std::vector<idxT> tempIdxX(to_size_t(SizeRoi().y));

    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        SrcT resultMin(reduction_init_value_v<ReductionInitValue::Max, SrcT>);
        idxT resultMinIdx(INT_MAX);

        for (int pixelX = 0; pixelX < SizeRoi().x; pixelX++)
        {
            if (aMask(pixelX, pixelY) > 0)
            {
                functor(pixelX, pixelY, resultMin, resultMinIdx);
            }
        }

        tempVal[to_size_t(pixelY)]  = resultMin;
        tempIdxX[to_size_t(pixelY)] = resultMinIdx;
    }

    SrcT resultMinY(reduction_init_value_v<ReductionInitValue::Max, SrcT>);
    idxT resultMinIdxY(-1);
    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        const SrcT pxMin = tempVal[to_size_t(pixelY)];

        redOpMin(pxMin.x, pixelY, resultMinY.x, resultMinIdxY.x);
    }
    // fetch X coordinates:
    idxT minIdxX;
    minIdxX.x = tempIdxX[to_size_t(resultMinIdxY.x)].x;

    aDstMin    = resultMinY;
    aDstIndexY = resultMinIdxY;
    aDstIndexX = minIdxX;
}
#pragma endregion
#pragma region MaxIndex
template <PixelType T>
void ImageView<T>::MaxIndex(T &aDstMax, same_vector_size_different_type_t<T, int> &aDstIndexX,
                            same_vector_size_different_type_t<T, int> &aDstIndexY, remove_vector_t<T> &aDstMaxScalar,
                            Vector3<int> &aDstScalarIdx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using idxT                 = same_vector_size_different_type_t<SrcT, int>;
    constexpr size_t TupelSize = 1;

    const mpp::MaxIdx<SrcT> redOpMax;
    const SrcReductionMaxIdxFunctor<TupelSize, SrcT> functor(PointerRoi(), Pitch());

    std::vector<T> tempVal(to_size_t(SizeRoi().y));
    std::vector<idxT> tempIdxX(to_size_t(SizeRoi().y));

    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        SrcT resultMax(reduction_init_value_v<ReductionInitValue::Min, SrcT>);
        idxT resultMaxIdx(INT_MAX);

        for (int pixelX = 0; pixelX < SizeRoi().x; pixelX++)
        {
            functor(pixelX, pixelY, resultMax, resultMaxIdx);
        }

        tempVal[to_size_t(pixelY)]  = resultMax;
        tempIdxX[to_size_t(pixelY)] = resultMaxIdx;
    }

    SrcT resultMaxY(reduction_init_value_v<ReductionInitValue::Min, SrcT>);
    idxT resultMaxIdxY(-1);
    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        const SrcT pxMax = tempVal[to_size_t(pixelY)];

        redOpMax(pxMax.x, pixelY, resultMaxY.x, resultMaxIdxY.x);
        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            redOpMax(pxMax.y, pixelY, resultMaxY.y, resultMaxIdxY.y);
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            redOpMax(pxMax.z, pixelY, resultMaxY.z, resultMaxIdxY.z);
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            redOpMax(pxMax.w, pixelY, resultMaxY.w, resultMaxIdxY.w);
        }
    }
    // fetch X coordinates:
    idxT MaxIdxX;
    MaxIdxX.x = tempIdxX[to_size_t(resultMaxIdxY.x)].x;
    if constexpr (vector_active_size_v<SrcT> > 1)
    {
        MaxIdxX.y = tempIdxX[to_size_t(resultMaxIdxY.y)].y;
    }
    if constexpr (vector_active_size_v<SrcT> > 2)
    {
        MaxIdxX.z = tempIdxX[to_size_t(resultMaxIdxY.z)].z;
    }
    if constexpr (vector_active_size_v<SrcT> > 3)
    {
        MaxIdxX.w = tempIdxX[to_size_t(resultMaxIdxY.w)].w;
    }

    int MaxIdxVec                   = 0;
    remove_vector_t<SrcT> MaxScalar = resultMaxY.x;

    if constexpr (vector_active_size_v<SrcT> > 1)
    {
        redOpMax(resultMaxY.y, 1, MaxScalar, MaxIdxVec);
    }
    if constexpr (vector_active_size_v<SrcT> > 2)
    {
        redOpMax(resultMaxY.z, 2, MaxScalar, MaxIdxVec);
    }
    if constexpr (vector_active_size_v<SrcT> > 3)
    {
        redOpMax(resultMaxY.w, 3, MaxScalar, MaxIdxVec);
    }

    const Vector3<int> MaxIdx(MaxIdxX[Channel(MaxIdxVec)], resultMaxIdxY[Channel(MaxIdxVec)], MaxIdxVec);
    aDstMax       = resultMaxY;
    aDstIndexY    = resultMaxIdxY;
    aDstIndexX    = MaxIdxX;
    aDstMaxScalar = MaxScalar;
    aDstScalarIdx = MaxIdx;
}
template <PixelType T>
void ImageView<T>::MaxIndexMasked(T &aDstMax, same_vector_size_different_type_t<T, int> &aDstIndexX,
                                  same_vector_size_different_type_t<T, int> &aDstIndexY,
                                  remove_vector_t<T> &aDstMaxScalar, Vector3<int> &aDstScalarIdx,
                                  const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{
    using SrcT                 = T;
    using idxT                 = same_vector_size_different_type_t<SrcT, int>;
    constexpr size_t TupelSize = 1;

    const mpp::MaxIdx<SrcT> redOpMax;
    const SrcReductionMaxIdxFunctor<TupelSize, SrcT> functor(PointerRoi(), Pitch());

    std::vector<T> tempVal(to_size_t(SizeRoi().y));
    std::vector<idxT> tempIdxX(to_size_t(SizeRoi().y));

    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        SrcT resultMax(reduction_init_value_v<ReductionInitValue::Min, SrcT>);
        idxT resultMaxIdx(INT_MAX);

        for (int pixelX = 0; pixelX < SizeRoi().x; pixelX++)
        {
            if (aMask(pixelX, pixelY) > 0)
            {
                functor(pixelX, pixelY, resultMax, resultMaxIdx);
            }
        }

        tempVal[to_size_t(pixelY)]  = resultMax;
        tempIdxX[to_size_t(pixelY)] = resultMaxIdx;
    }

    SrcT resultMaxY(reduction_init_value_v<ReductionInitValue::Min, SrcT>);
    idxT resultMaxIdxY(-1);
    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        const SrcT pxMax = tempVal[to_size_t(pixelY)];

        redOpMax(pxMax.x, pixelY, resultMaxY.x, resultMaxIdxY.x);
        if constexpr (vector_active_size_v<SrcT> > 1)
        {
            redOpMax(pxMax.y, pixelY, resultMaxY.y, resultMaxIdxY.y);
        }
        if constexpr (vector_active_size_v<SrcT> > 2)
        {
            redOpMax(pxMax.z, pixelY, resultMaxY.z, resultMaxIdxY.z);
        }
        if constexpr (vector_active_size_v<SrcT> > 3)
        {
            redOpMax(pxMax.w, pixelY, resultMaxY.w, resultMaxIdxY.w);
        }
    }
    // fetch X coordinates:
    idxT MaxIdxX;
    MaxIdxX.x = tempIdxX[to_size_t(resultMaxIdxY.x)].x;
    if constexpr (vector_active_size_v<SrcT> > 1)
    {
        MaxIdxX.y = tempIdxX[to_size_t(resultMaxIdxY.y)].y;
    }
    if constexpr (vector_active_size_v<SrcT> > 2)
    {
        MaxIdxX.z = tempIdxX[to_size_t(resultMaxIdxY.z)].z;
    }
    if constexpr (vector_active_size_v<SrcT> > 3)
    {
        MaxIdxX.w = tempIdxX[to_size_t(resultMaxIdxY.w)].w;
    }

    int MaxIdxVec                   = 0;
    remove_vector_t<SrcT> MaxScalar = resultMaxY.x;

    if constexpr (vector_active_size_v<SrcT> > 1)
    {
        redOpMax(resultMaxY.y, 1, MaxScalar, MaxIdxVec);
    }
    if constexpr (vector_active_size_v<SrcT> > 2)
    {
        redOpMax(resultMaxY.z, 2, MaxScalar, MaxIdxVec);
    }
    if constexpr (vector_active_size_v<SrcT> > 3)
    {
        redOpMax(resultMaxY.w, 3, MaxScalar, MaxIdxVec);
    }

    const Vector3<int> MaxIdx(MaxIdxX[Channel(MaxIdxVec)], resultMaxIdxY[Channel(MaxIdxVec)], MaxIdxVec);
    aDstMax       = resultMaxY;
    aDstIndexY    = resultMaxIdxY;
    aDstIndexX    = MaxIdxX;
    aDstMaxScalar = MaxScalar;
    aDstScalarIdx = MaxIdx;
}

template <PixelType T>
void ImageView<T>::MaxIndex(T &aDstMax, same_vector_size_different_type_t<T, int> &aDstIndexX,
                            same_vector_size_different_type_t<T, int> &aDstIndexY) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using idxT                 = same_vector_size_different_type_t<SrcT, int>;
    constexpr size_t TupelSize = 1;

    const mpp::MaxIdx<SrcT> redOpMax;
    const SrcReductionMaxIdxFunctor<TupelSize, SrcT> functor(PointerRoi(), Pitch());

    std::vector<T> tempVal(to_size_t(SizeRoi().y));
    std::vector<idxT> tempIdxX(to_size_t(SizeRoi().y));

    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        SrcT resultMax(reduction_init_value_v<ReductionInitValue::Min, SrcT>);
        idxT resultMaxIdx(INT_MAX);

        for (int pixelX = 0; pixelX < SizeRoi().x; pixelX++)
        {
            functor(pixelX, pixelY, resultMax, resultMaxIdx);
        }

        tempVal[to_size_t(pixelY)]  = resultMax;
        tempIdxX[to_size_t(pixelY)] = resultMaxIdx;
    }

    SrcT resultMaxY(reduction_init_value_v<ReductionInitValue::Min, SrcT>);
    idxT resultMaxIdxY(-1);
    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        const SrcT pxMax = tempVal[to_size_t(pixelY)];

        redOpMax(pxMax.x, pixelY, resultMaxY.x, resultMaxIdxY.x);
    }
    // fetch X coordinates:
    idxT MaxIdxX;
    MaxIdxX.x = tempIdxX[to_size_t(resultMaxIdxY.x)].x;

    aDstMax    = resultMaxY;
    aDstIndexY = resultMaxIdxY;
    aDstIndexX = MaxIdxX;
}
template <PixelType T>
void ImageView<T>::MaxIndexMasked(T &aDstMax, same_vector_size_different_type_t<T, int> &aDstIndexX,
                                  same_vector_size_different_type_t<T, int> &aDstIndexY,
                                  const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{
    using SrcT                 = T;
    using idxT                 = same_vector_size_different_type_t<SrcT, int>;
    constexpr size_t TupelSize = 1;

    const mpp::MaxIdx<SrcT> redOpMax;
    const SrcReductionMaxIdxFunctor<TupelSize, SrcT> functor(PointerRoi(), Pitch());

    std::vector<T> tempVal(to_size_t(SizeRoi().y));
    std::vector<idxT> tempIdxX(to_size_t(SizeRoi().y));

    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        SrcT resultMax(reduction_init_value_v<ReductionInitValue::Min, SrcT>);
        idxT resultMaxIdx(INT_MAX);

        for (int pixelX = 0; pixelX < SizeRoi().x; pixelX++)
        {
            if (aMask(pixelX, pixelY) > 0)
            {
                functor(pixelX, pixelY, resultMax, resultMaxIdx);
            }
        }

        tempVal[to_size_t(pixelY)]  = resultMax;
        tempIdxX[to_size_t(pixelY)] = resultMaxIdx;
    }

    SrcT resultMaxY(reduction_init_value_v<ReductionInitValue::Min, SrcT>);
    idxT resultMaxIdxY(-1);
    for (int pixelY = 0; pixelY < SizeRoi().y; pixelY++)
    {
        const SrcT pxMax = tempVal[to_size_t(pixelY)];

        redOpMax(pxMax.x, pixelY, resultMaxY.x, resultMaxIdxY.x);
    }
    // fetch X coordinates:
    idxT MaxIdxX;
    MaxIdxX.x = tempIdxX[to_size_t(resultMaxIdxY.x)].x;

    aDstMax    = resultMaxY;
    aDstIndexY = resultMaxIdxY;
    aDstIndexX = MaxIdxX;
}
#pragma endregion
#pragma region MinMaxIndex
template <PixelType T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ImageView<T>::MinMaxIndex(T &aDstMin, T &aDstMax, IndexMinMax aDstIdx[vector_active_size_v<T>],
                               // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                               remove_vector_t<T> &aDstMinScalar, remove_vector_t<T> &aDstMaxScalar,
                               IndexMinMaxChannel &aDstScalarIdx) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{

    using idxT = same_vector_size_different_type_t<T, int>;
    idxT idxXMin;
    idxT idxYMin;
    Vec3i idxScalarMin;

    idxT idxXMax;
    idxT idxYMax;
    Vec3i idxScalarMax;

    MinIndex(aDstMin, idxXMin, idxYMin, aDstMinScalar, idxScalarMin);
    MaxIndex(aDstMax, idxXMax, idxYMax, aDstMaxScalar, idxScalarMax);

    aDstScalarIdx.ChannelMax = idxScalarMax.z;
    aDstScalarIdx.ChannelMin = idxScalarMin.z;

    aDstScalarIdx.IndexMax = idxScalarMax.XY();
    aDstScalarIdx.IndexMin = idxScalarMin.XY();

    aDstIdx[0].IndexMax.x = idxXMax.x;
    aDstIdx[0].IndexMax.y = idxYMax.x;
    aDstIdx[0].IndexMin.x = idxXMin.x;
    aDstIdx[0].IndexMin.y = idxYMin.x;

    if constexpr (vector_active_size_v<T> > 1)
    {
        aDstIdx[1].IndexMax.x = idxXMax.y;
        aDstIdx[1].IndexMax.y = idxYMax.y;
        aDstIdx[1].IndexMin.x = idxXMin.y;
        aDstIdx[1].IndexMin.y = idxYMin.y;
    }
    if constexpr (vector_active_size_v<T> > 2)
    {
        aDstIdx[2].IndexMax.x = idxXMax.z;
        aDstIdx[2].IndexMax.y = idxYMax.z;
        aDstIdx[2].IndexMin.x = idxXMin.z;
        aDstIdx[2].IndexMin.y = idxYMin.z;
    }
    if constexpr (vector_active_size_v<T> > 3)
    {
        aDstIdx[3].IndexMax.x = idxXMax.w;
        aDstIdx[3].IndexMax.y = idxYMax.w;
        aDstIdx[3].IndexMin.x = idxXMin.w;
        aDstIdx[3].IndexMin.y = idxYMin.w;
    }
}
template <PixelType T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ImageView<T>::MinMaxIndexMasked(T &aDstMin, T &aDstMax, IndexMinMax aDstIdx[vector_active_size_v<T>],
                                     // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                     remove_vector_t<T> &aDstMinScalar, remove_vector_t<T> &aDstMaxScalar,
                                     IndexMinMaxChannel &aDstScalarIdx, const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> > 1)
{

    using idxT = same_vector_size_different_type_t<T, int>;
    idxT idxXMin;
    idxT idxYMin;
    Vec3i idxScalarMin;

    idxT idxXMax;
    idxT idxYMax;
    Vec3i idxScalarMax;

    MinIndexMasked(aDstMin, idxXMin, idxYMin, aDstMinScalar, idxScalarMin, aMask);
    MaxIndexMasked(aDstMax, idxXMax, idxYMax, aDstMaxScalar, idxScalarMax, aMask);

    aDstScalarIdx.ChannelMax = idxScalarMax.z;
    aDstScalarIdx.ChannelMin = idxScalarMin.z;

    aDstScalarIdx.IndexMax = idxScalarMax.XY();
    aDstScalarIdx.IndexMin = idxScalarMin.XY();

    aDstIdx[0].IndexMax.x = idxXMax.x;
    aDstIdx[0].IndexMax.y = idxYMax.x;
    aDstIdx[0].IndexMin.x = idxXMin.x;
    aDstIdx[0].IndexMin.y = idxYMin.x;

    if constexpr (vector_active_size_v<T> > 1)
    {
        aDstIdx[1].IndexMax.x = idxXMax.y;
        aDstIdx[1].IndexMax.y = idxYMax.y;
        aDstIdx[1].IndexMin.x = idxXMin.y;
        aDstIdx[1].IndexMin.y = idxYMin.y;
    }
    if constexpr (vector_active_size_v<T> > 2)
    {
        aDstIdx[2].IndexMax.x = idxXMax.z;
        aDstIdx[2].IndexMax.y = idxYMax.z;
        aDstIdx[2].IndexMin.x = idxXMin.z;
        aDstIdx[2].IndexMin.y = idxYMin.z;
    }
    if constexpr (vector_active_size_v<T> > 3)
    {
        aDstIdx[3].IndexMax.x = idxXMax.w;
        aDstIdx[3].IndexMax.y = idxYMax.w;
        aDstIdx[3].IndexMin.x = idxXMin.w;
        aDstIdx[3].IndexMin.y = idxYMin.w;
    }
}
template <PixelType T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ImageView<T>::MinMaxIndex(T &aDstMin, T &aDstMax, IndexMinMax &aDstIdx) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{

    using idxT = same_vector_size_different_type_t<T, int>;
    idxT idxXMin;
    idxT idxYMin;

    idxT idxXMax;
    idxT idxYMax;

    MinIndex(aDstMin, idxXMin, idxYMin);
    MaxIndex(aDstMax, idxXMax, idxYMax);

    aDstIdx.IndexMax.x = idxXMax.x;
    aDstIdx.IndexMax.y = idxYMax.x;
    aDstIdx.IndexMin.x = idxXMin.x;
    aDstIdx.IndexMin.y = idxYMin.x;
}
template <PixelType T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ImageView<T>::MinMaxIndexMasked(T &aDstMin, T &aDstMax, IndexMinMax &aDstIdx,
                                     const ImageView<Pixel8uC1> &aMask) const
    requires RealVector<T> && (vector_active_size_v<T> == 1)
{

    using idxT = same_vector_size_different_type_t<T, int>;
    idxT idxXMin;
    idxT idxYMin;

    idxT idxXMax;
    idxT idxYMax;

    MinIndexMasked(aDstMin, idxXMin, idxYMin, aMask);
    MaxIndexMasked(aDstMax, idxXMax, idxYMax, aMask);

    aDstIdx.IndexMax.x = idxXMax.x;
    aDstIdx.IndexMax.y = idxYMax.x;
    aDstIdx.IndexMin.x = idxXMin.x;
    aDstIdx.IndexMin.y = idxYMin.x;
}
#pragma endregion

#pragma region MinEvery
template <PixelType T>
ImageView<T> &ImageView<T>::MinEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using minEverySrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Min<ComputeT>, RoundingMode::None>;
    const mpp::Min<ComputeT> op;
    const minEverySrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MinEvery(const ImageView<T> &aSrc2)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using minEveryInplaceSrc =
        InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Min<ComputeT>, RoundingMode::None>;
    const mpp::Min<ComputeT> op;
    const minEveryInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
    forEachPixel(*this, functor);

    return *this;
}
#pragma endregion

#pragma region MaxEvery
template <PixelType T>
ImageView<T> &ImageView<T>::MaxEvery(const ImageView<T> &aSrc2, ImageView<T> &aDst) const
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());
    checkSameSize(ROI(), aDst.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using maxEverySrcSrc = SrcSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Max<ComputeT>, RoundingMode::None>;
    const mpp::Max<ComputeT> op;
    const maxEverySrcSrc functor(PointerRoi(), Pitch(), aSrc2.PointerRoi(), aSrc2.Pitch(), op);
    forEachPixel(aDst, functor);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MaxEvery(const ImageView<T> &aSrc2)
    requires RealVector<T>
{
    checkSameSize(ROI(), aSrc2.ROI());

    using SrcT                 = T;
    using ComputeT             = T;
    using DstT                 = T;
    constexpr size_t TupelSize = 1;

    using maxEveryInplaceSrc =
        InplaceSrcFunctor<TupelSize, SrcT, ComputeT, DstT, mpp::Max<ComputeT>, RoundingMode::None>;
    const mpp::Max<ComputeT> op;
    const maxEveryInplaceSrc functor(aSrc2.PointerRoi(), aSrc2.Pitch(), op);
    forEachPixel(*this, functor);

    return *this;
}

#pragma region Integral
template <PixelType T>
ImageView<same_vector_size_different_type_t<T, int>> &ImageView<T>::Integral(
    ImageView<same_vector_size_different_type_t<T, int>> &aDst,
    const same_vector_size_different_type_t<T, int> &aVal) const
    requires RealIntVector<T>
{
    if (SizeRoi() != aDst.SizeRoi() - 1)
    {
        throw ROIEXCEPTION(
            "ROI of destination image must be one pixel larger in width and height than original ROI. Original size: "
            << SizeRoi() << " provided destination image size: " << aDst.SizeRoi());
    }

    using DstT = same_vector_size_different_type_t<T, int>;

    // init left/top-border pixels to aVal:
    for (int x = 0; x < aDst.ROI().width; x++)
    {
        aDst(x, 0) = aVal;
    }
    for (int y = 1; y < aDst.ROI().height; y++)
    {
        aDst(0, y) = aVal;
    }

    for (int y = 1; y < aDst.ROI().height; y++)
    {
        for (int x = 1; x < aDst.ROI().width; x++)
        {
            aDst(x, y) = aDst(x, y - 1) + aDst(x - 1, y) - aDst(x - 1, y - 1) + DstT((*this)(x - 1, y - 1));
        }
    }
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::Integral(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst,
    const same_vector_size_different_type_t<T, float> &aVal) const
    requires RealVector<T> && (!std::same_as<double, remove_vector<T>>)
{
    if (SizeRoi() != aDst.SizeRoi() - 1)
    {
        throw ROIEXCEPTION(
            "ROI of destination image must be one pixel larger in width and height than original ROI. Original size: "
            << SizeRoi() << " provided destination image size: " << aDst.SizeRoi());
    }

    using DstT = same_vector_size_different_type_t<T, float>;

    // init left/top-border pixels to aVal:
    for (int x = 0; x < aDst.ROI().width; x++)
    {
        aDst(x, 0) = aVal;
    }
    for (int y = 1; y < aDst.ROI().height; y++)
    {
        aDst(0, y) = aVal;
    }

    for (int y = 1; y < aDst.ROI().height; y++)
    {
        for (int x = 1; x < aDst.ROI().width; x++)
        {
            aDst(x, y) = aDst(x, y - 1) + aDst(x - 1, y) - aDst(x - 1, y - 1) + DstT((*this)(x - 1, y - 1));
        }
    }
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, long64>> &ImageView<T>::Integral(
    ImageView<same_vector_size_different_type_t<T, long64>> &aDst,
    const same_vector_size_different_type_t<T, long64> &aVal) const
    requires RealIntVector<T>
{
    if (SizeRoi() != aDst.SizeRoi() - 1)
    {
        throw ROIEXCEPTION(
            "ROI of destination image must be one pixel larger in width and height than original ROI. Original size: "
            << SizeRoi() << " provided destination image size: " << aDst.SizeRoi());
    }

    using DstT = same_vector_size_different_type_t<T, long64>;

    // init left/top-border pixels to aVal:
    for (int x = 0; x < aDst.ROI().width; x++)
    {
        aDst(x, 0) = aVal;
    }
    for (int y = 1; y < aDst.ROI().height; y++)
    {
        aDst(0, y) = aVal;
    }

    for (int y = 1; y < aDst.ROI().height; y++)
    {
        for (int x = 1; x < aDst.ROI().width; x++)
        {
            aDst(x, y) = aDst(x, y - 1) + aDst(x - 1, y) - aDst(x - 1, y - 1) + DstT((*this)(x - 1, y - 1));
        }
    }
    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, double>> &ImageView<T>::Integral(
    ImageView<same_vector_size_different_type_t<T, double>> &aDst,
    const same_vector_size_different_type_t<T, double> &aVal) const
    requires RealVector<T>
{
    if (SizeRoi() != aDst.SizeRoi() - 1)
    {
        throw ROIEXCEPTION(
            "ROI of destination image must be one pixel larger in width and height than original ROI. Original size: "
            << SizeRoi() << " provided destination image size: " << aDst.SizeRoi());
    }

    using DstT = same_vector_size_different_type_t<T, double>;

    // init left/top-border pixels to aVal:
    for (int x = 0; x < aDst.ROI().width; x++)
    {
        aDst(x, 0) = aVal;
    }
    for (int y = 1; y < aDst.ROI().height; y++)
    {
        aDst(0, y) = aVal;
    }

    for (int y = 1; y < aDst.ROI().height; y++)
    {
        for (int x = 1; x < aDst.ROI().width; x++)
        {
            aDst(x, y) = aDst(x, y - 1) + aDst(x - 1, y) - aDst(x - 1, y - 1) + DstT((*this)(x - 1, y - 1));
        }
    }
    return aDst;
}

template <PixelType T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ImageView<T>::SqrIntegral(ImageView<same_vector_size_different_type_t<T, int>> &aDst,
                               ImageView<same_vector_size_different_type_t<T, int>> &aSqr,
                               // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                               const same_vector_size_different_type_t<T, int> &aVal,
                               const same_vector_size_different_type_t<T, int> &aValSqr) const
    requires RealIntVector<T>
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
    using DstTSqr = same_vector_size_different_type_t<T, int>;

    // init left/top-border pixels to aVal:
    for (int x = 0; x < aDst.ROI().width; x++)
    {
        aDst(x, 0) = aVal;
        aSqr(x, 0) = aValSqr;
    }
    for (int y = 1; y < aDst.ROI().height; y++)
    {
        aDst(0, y) = aVal;
        aSqr(0, y) = aValSqr;
    }

    for (int y = 1; y < aDst.ROI().height; y++)
    {
        for (int x = 1; x < aDst.ROI().width; x++)
        {
            aDst(x, y) = aDst(x, y - 1) + aDst(x - 1, y) - aDst(x - 1, y - 1) + DstT((*this)(x - 1, y - 1));
            aSqr(x, y) = aSqr(x, y - 1) + aSqr(x - 1, y) - aSqr(x - 1, y - 1) +
                         DstTSqr((*this)(x - 1, y - 1)) * DstTSqr((*this)(x - 1, y - 1));
        }
    }
}

template <PixelType T>
void ImageView<T>::SqrIntegral(ImageView<same_vector_size_different_type_t<T, int>> &aDst,
                               ImageView<same_vector_size_different_type_t<T, long64>> &aSqr,
                               const same_vector_size_different_type_t<T, int> &aVal,
                               const same_vector_size_different_type_t<T, long64> &aValSqr) const
    requires RealIntVector<T>
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
    using DstTSqr = same_vector_size_different_type_t<T, long64>;

    // init left/top-border pixels to aVal:
    for (int x = 0; x < aDst.ROI().width; x++)
    {
        aDst(x, 0) = aVal;
        aSqr(x, 0) = aValSqr;
    }
    for (int y = 1; y < aDst.ROI().height; y++)
    {
        aDst(0, y) = aVal;
        aSqr(0, y) = aValSqr;
    }

    for (int y = 1; y < aDst.ROI().height; y++)
    {
        for (int x = 1; x < aDst.ROI().width; x++)
        {
            aDst(x, y) = aDst(x, y - 1) + aDst(x - 1, y) - aDst(x - 1, y - 1) + DstT((*this)(x - 1, y - 1));
            aSqr(x, y) = aSqr(x, y - 1) + aSqr(x - 1, y) - aSqr(x - 1, y - 1) +
                         DstTSqr((*this)(x - 1, y - 1)) * DstTSqr((*this)(x - 1, y - 1));
        }
    }
}

template <PixelType T>
void ImageView<T>::SqrIntegral(ImageView<same_vector_size_different_type_t<T, float>> &aDst,
                               ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                               const same_vector_size_different_type_t<T, float> &aVal,
                               const same_vector_size_different_type_t<T, double> &aValSqr) const
    requires RealVector<T> && (!std::same_as<double, remove_vector<T>>)
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
    using DstTSqr = same_vector_size_different_type_t<T, double>;

    // init left/top-border pixels to aVal:
    for (int x = 0; x < aDst.ROI().width; x++)
    {
        aDst(x, 0) = aVal;
        aSqr(x, 0) = aValSqr;
    }
    for (int y = 1; y < aDst.ROI().height; y++)
    {
        aDst(0, y) = aVal;
        aSqr(0, y) = aValSqr;
    }

    for (int y = 1; y < aDst.ROI().height; y++)
    {
        for (int x = 1; x < aDst.ROI().width; x++)
        {
            aDst(x, y) = aDst(x, y - 1) + aDst(x - 1, y) - aDst(x - 1, y - 1) + DstT((*this)(x - 1, y - 1));
            aSqr(x, y) = aSqr(x, y - 1) + aSqr(x - 1, y) - aSqr(x - 1, y - 1) +
                         DstTSqr((*this)(x - 1, y - 1)) * DstTSqr((*this)(x - 1, y - 1));
        }
    }
}

template <PixelType T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void ImageView<T>::SqrIntegral(ImageView<same_vector_size_different_type_t<T, double>> &aDst,
                               ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                               // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                               const same_vector_size_different_type_t<T, double> &aVal,
                               const same_vector_size_different_type_t<T, double> &aValSqr) const
    requires RealVector<T>
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
    using DstTSqr = same_vector_size_different_type_t<T, double>;

    // init left/top-border pixels to aVal:
    for (int x = 0; x < aDst.ROI().width; x++)
    {
        aDst(x, 0) = aVal;
        aSqr(x, 0) = aValSqr;
    }
    for (int y = 1; y < aDst.ROI().height; y++)
    {
        aDst(0, y) = aVal;
        aSqr(0, y) = aValSqr;
    }

    for (int y = 1; y < aDst.ROI().height; y++)
    {
        for (int x = 1; x < aDst.ROI().width; x++)
        {
            aDst(x, y) = aDst(x, y - 1) + aDst(x - 1, y) - aDst(x - 1, y - 1) + DstT((*this)(x - 1, y - 1));
            aSqr(x, y) = aSqr(x, y - 1) + aSqr(x - 1, y) - aSqr(x - 1, y - 1) +
                         DstTSqr((*this)(x - 1, y - 1)) * DstTSqr((*this)(x - 1, y - 1));
        }
    }
}

template <PixelType T>
void ImageView<T>::RectStdDev(ImageView<same_vector_size_different_type_t<T, int>> &aSqr,
                              ImageView<same_vector_size_different_type_t<T, float>> &aDst,
                              const FilterArea &aFilterArea) const
    requires(std::same_as<remove_vector_t<T>, int>) && NoAlpha<T>
{
    using ComputeT = same_vector_size_different_type_t<T, double>;
    using Src2T    = same_vector_size_different_type_t<T, int>;
    using DstT     = same_vector_size_different_type_t<T, float>;

    checkSameSize(ROI(), aSqr.ROI());
    // aDst roi may differ

    constexpr size_t TupelSize = 1;
    using rectStdDev           = RectStdDevFunctor<TupelSize, T, Src2T, ComputeT, DstT>;
    const rectStdDev functor(PointerRoi(), Pitch(), aSqr.PointerRoi(), aSqr.Pitch(), SizeRoi(), aFilterArea);

    forEachPixel(aDst, functor);
}

template <PixelType T>
void ImageView<T>::RectStdDev(ImageView<same_vector_size_different_type_t<T, long64>> &aSqr,
                              ImageView<same_vector_size_different_type_t<T, float>> &aDst,
                              const FilterArea &aFilterArea) const
    requires(std::same_as<remove_vector_t<T>, int>) && NoAlpha<T>
{
    using ComputeT = same_vector_size_different_type_t<T, double>;
    using Src2T    = same_vector_size_different_type_t<T, long64>;
    using DstT     = same_vector_size_different_type_t<T, float>;

    checkSameSize(ROI(), aSqr.ROI());
    // aDst roi may differ

    constexpr size_t TupelSize = 1;
    using rectStdDev           = RectStdDevFunctor<TupelSize, T, Src2T, ComputeT, DstT>;
    const rectStdDev functor(PointerRoi(), Pitch(), aSqr.PointerRoi(), aSqr.Pitch(), SizeRoi(), aFilterArea);

    forEachPixel(aDst, functor);
}

template <PixelType T>
void ImageView<T>::RectStdDev(ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                              ImageView<same_vector_size_different_type_t<T, float>> &aDst,
                              const FilterArea &aFilterArea) const
    requires(std::same_as<remove_vector_t<T>, float>) && NoAlpha<T>
{
    using ComputeT = same_vector_size_different_type_t<T, double>;
    using Src2T    = same_vector_size_different_type_t<T, double>;
    using DstT     = same_vector_size_different_type_t<T, float>;

    checkSameSize(ROI(), aSqr.ROI());
    // aDst roi may differ

    constexpr size_t TupelSize = 1;
    using rectStdDev           = RectStdDevFunctor<TupelSize, T, Src2T, ComputeT, DstT>;
    const rectStdDev functor(PointerRoi(), Pitch(), aSqr.PointerRoi(), aSqr.Pitch(), SizeRoi(), aFilterArea);

    forEachPixel(aDst, functor);
}

template <PixelType T>
void ImageView<T>::RectStdDev(ImageView<same_vector_size_different_type_t<T, double>> &aSqr,
                              ImageView<same_vector_size_different_type_t<T, double>> &aDst,
                              const FilterArea &aFilterArea) const
    requires(std::same_as<remove_vector_t<T>, double>) && NoAlpha<T>
{
    using ComputeT = same_vector_size_different_type_t<T, double>;
    using Src2T    = same_vector_size_different_type_t<T, double>;
    using DstT     = same_vector_size_different_type_t<T, double>;

    checkSameSize(ROI(), aSqr.ROI());
    // aDst roi may differ

    constexpr size_t TupelSize = 1;
    using rectStdDev           = RectStdDevFunctor<TupelSize, T, Src2T, ComputeT, DstT>;
    const rectStdDev functor(PointerRoi(), Pitch(), aSqr.PointerRoi(), aSqr.Pitch(), SizeRoi(), aFilterArea);

    forEachPixel(aDst, functor);
}
#pragma endregion
#pragma endregion

#pragma region Histogram
template <PixelType T>
void ImageView<T>::EvenLevels(hist_even_level_types_for_t<T> *aHPtrLevels, int aLevels,
                              const hist_even_level_types_for_t<T> &aLowerLevel,
                              const hist_even_level_types_for_t<T> &aUpperLevel)
    requires RealVector<T>
{
    if (aLevels < 2)
    {
        throw INVALIDARGUMENT(aLevels, "aLevels must be at least 2 but provided value is: " << aLevels);
    }
    if (aHPtrLevels == nullptr)
    {
        throw INVALIDARGUMENT(aHPtrLevels, "nullptr");
    }
    if (!(aLowerLevel < aUpperLevel))
    {
        throw INVALIDARGUMENT(aLowerLevel, "aLowerLevel must be smaller than aUpperLevel, but aLowerLevel = "
                                               << aLowerLevel << " and aUpperLevel = " << aUpperLevel);
    }
    using T_Double = same_vector_size_different_type_t<T, double>;

    const T_Double step = (T_Double(aUpperLevel) - T_Double(aLowerLevel)) / (T_Double(aLevels) - 1.0);

    // default mode as in CUB:
    for (int i = 0; i < aLevels; i++)
    {
        const T_Double bin = step * static_cast<double>(i);

        if constexpr (RealIntVector<hist_even_level_types_for_t<T>>)
        {
            aHPtrLevels[i] = hist_even_level_types_for_t<T>(T_Double::Ceil(bin)) + aLowerLevel;
        }
        else
        {
            aHPtrLevels[i] = hist_even_level_types_for_t<T>(bin) + aLowerLevel;
        }
    }
}

template <PixelType T>
void ImageView<T>::HistogramEven(same_vector_size_different_type_t<T, int> *aHist, int aHistBinCount,
                                 const hist_even_level_types_for_t<T> &aLowerLevel,
                                 const hist_even_level_types_for_t<T> &aUpperLevel)
    requires RealVector<T>
{
    std::vector<hist_even_level_types_for_t<T>> levels(to_size_t(aHistBinCount) + 1, 0);

    EvenLevels(levels.data(), aHistBinCount + 1, aLowerLevel, aUpperLevel);

    HistogramRange(aHist, aHistBinCount, levels.data());
}

template <PixelType T>
void ImageView<T>::HistogramRange(same_vector_size_different_type_t<T, int> *aHist, int aHistBinCount,
                                  hist_range_types_for_t<T> *aLevels)
    requires RealVector<T>
{
    for (int i = 0; i < aHistBinCount; i++)
    {
        aHist[i] = same_vector_size_different_type_t<T, int>(0);
    }

    for (const auto &pixelIterator : *this)
    {
        const T &pixel = pixelIterator.Value();

        for (int i = 0; i < aHistBinCount; i++)
        {
            if (aLevels[i].x <= pixel.x && pixel.x < aLevels[i + 1].x)
            {
                aHist[i].x++;
            }
            if constexpr (vector_active_size_v<T> > 1)
            {
                if (aLevels[i].y <= pixel.y && pixel.y < aLevels[i + 1].y)
                {
                    aHist[i].y++;
                }
            }
            if constexpr (vector_active_size_v<T> > 2)
            {
                if (aLevels[i].z <= pixel.z && pixel.z < aLevels[i + 1].z)
                {
                    aHist[i].z++;
                }
            }
            if constexpr (vector_active_size_v<T> > 3)
            {
                if (aLevels[i].w <= pixel.w && pixel.w < aLevels[i + 1].w)
                {
                    aHist[i].w++;
                }
            }
        }
    }
}
#pragma endregion

#pragma region Cross Correlation
template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelation(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                      BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->CrossCorrelation(aTemplate, aDst, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelation(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                      const T &aConstant, BorderType aBorder,
                                                      const Roi &aAllowedReadRoi) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    crossCorrelationEachPixel(*this, aDst, aTemplate, aTemplate.SizeRoi(), aBorder, aConstant, aAllowedReadRoi);

    return aDst;
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationNormalized(const ImageView<T> &aTemplate,
                                                                ImageView<Pixel32fC1> &aDst, BorderType aBorder,
                                                                const Roi &aAllowedReadRoi) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->CrossCorrelationNormalized(aTemplate, aDst, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationNormalized(const ImageView<T> &aTemplate,
                                                                ImageView<Pixel32fC1> &aDst, const T &aConstant,
                                                                BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    crossCorrelationNormalizedEachPixel(*this, aDst, aTemplate, aTemplate.SizeRoi(), aBorder, aConstant,
                                        aAllowedReadRoi);
    return aDst;
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::SquareDistanceNormalized(const ImageView<T> &aTemplate,
                                                              ImageView<Pixel32fC1> &aDst, BorderType aBorder,
                                                              const Roi &aAllowedReadRoi) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->SquareDistanceNormalized(aTemplate, aDst, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::SquareDistanceNormalized(const ImageView<T> &aTemplate,
                                                              ImageView<Pixel32fC1> &aDst, const T &aConstant,
                                                              BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    squareDistanceNormalizedEachPixel(*this, aDst, aTemplate, aTemplate.SizeRoi(), aBorder, aConstant, aAllowedReadRoi);

    return aDst;
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationCoefficient(const ImageView<T> &aTemplate,
                                                                 ImageView<Pixel32fC1> &aDst, BorderType aBorder,
                                                                 const Roi &aAllowedReadRoi) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->CrossCorrelationCoefficient(aTemplate, aDst, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationCoefficient(const ImageView<T> &aTemplate,
                                                                 ImageView<Pixel32fC1> &aDst, const T &aConstant,
                                                                 BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    crossCorrelationCoefficientEachPixel(*this, aDst, aTemplate, aTemplate.SizeRoi(), aBorder, aConstant,
                                         aAllowedReadRoi);

    return aDst;
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelation(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                      BorderType aBorder) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->CrossCorrelation(aTemplate, aDst, aBorder, ROI());
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelation(const ImageView<T> &aTemplate, ImageView<Pixel32fC1> &aDst,
                                                      const T &aConstant, BorderType aBorder) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->CrossCorrelation(aTemplate, aDst, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationNormalized(const ImageView<T> &aTemplate,
                                                                ImageView<Pixel32fC1> &aDst, BorderType aBorder) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->CrossCorrelationNormalized(aTemplate, aDst, aBorder, ROI());
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationNormalized(const ImageView<T> &aTemplate,
                                                                ImageView<Pixel32fC1> &aDst, const T &aConstant,
                                                                BorderType aBorder) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->CrossCorrelationNormalized(aTemplate, aDst, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::SquareDistanceNormalized(const ImageView<T> &aTemplate,
                                                              ImageView<Pixel32fC1> &aDst, BorderType aBorder) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->SquareDistanceNormalized(aTemplate, aDst, aBorder, ROI());
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::SquareDistanceNormalized(const ImageView<T> &aTemplate,
                                                              ImageView<Pixel32fC1> &aDst, const T &aConstant,
                                                              BorderType aBorder) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->SquareDistanceNormalized(aTemplate, aDst, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationCoefficient(const ImageView<T> &aTemplate,
                                                                 ImageView<Pixel32fC1> &aDst, BorderType aBorder) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->CrossCorrelationCoefficient(aTemplate, aDst, aBorder, ROI());
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::CrossCorrelationCoefficient(const ImageView<T> &aTemplate,
                                                                 ImageView<Pixel32fC1> &aDst, const T &aConstant,
                                                                 BorderType aBorder) const
    requires SingleChannel<T> && RealVector<T> && (sizeof(T) < 8)
{
    return this->CrossCorrelationCoefficient(aTemplate, aDst, aConstant, aBorder, ROI());
}
#pragma endregion
} // namespace mpp::image::cpuSimple