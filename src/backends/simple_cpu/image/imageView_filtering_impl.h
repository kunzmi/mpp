#pragma once
#include <backends/simple_cpu/image/filterEachPixel_impl.h>
#include <backends/simple_cpu/image/image.h>
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
inline Size2D GetFilterSize(MaskSize aMaskSize)
{
    switch (aMaskSize)
    {
        case mpp::MaskSize::Mask_1x3:
            return {1, 3};
        case mpp::MaskSize::Mask_1x5:
            return {1, 5};
        case mpp::MaskSize::Mask_3x1:
            return {3, 1};
        case mpp::MaskSize::Mask_5x1:
            return {5, 1};
        case mpp::MaskSize::Mask_3x3:
            return {3, 3};
        case mpp::MaskSize::Mask_5x5:
            return {5, 5};
        case mpp::MaskSize::Mask_7x7:
            return {7, 7};
        case mpp::MaskSize::Mask_9x9:
            return {9, 9};
        case mpp::MaskSize::Mask_11x11:
            return {11, 11};
        case mpp::MaskSize::Mask_13x13:
            return {13, 13};
        case mpp::MaskSize::Mask_15x15:
            return {15, 15};
        default:
            return {-1, -1};
    }
}

inline const float *GetFilter(FixedFilter aFilter, int aSize)
{
    if (aSize == 3)
    {
        switch (aFilter)
        {
            case mpp::FixedFilter::Gauss:
                return reinterpret_cast<const float *>(FixedFilterKernel<mpp::FixedFilter::Gauss, 3, float>::Values);
            case mpp::FixedFilter::HighPass:
                return reinterpret_cast<const float *>(FixedFilterKernel<mpp::FixedFilter::HighPass, 3, float>::Values);
            case mpp::FixedFilter::LowPass:
                return reinterpret_cast<const float *>(FixedFilterKernel<mpp::FixedFilter::LowPass, 3, float>::Values);
            case mpp::FixedFilter::Laplace:
                return reinterpret_cast<const float *>(FixedFilterKernel<mpp::FixedFilter::Laplace, 3, float>::Values);
            case mpp::FixedFilter::PrewittHoriz:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::PrewittHoriz, 3, float>::Values);
            case mpp::FixedFilter::PrewittVert:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::PrewittVert, 3, float>::Values);
            case mpp::FixedFilter::RobertsDown:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::RobertsDown, 3, float>::Values);
            case mpp::FixedFilter::RobertsUp:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::RobertsUp, 3, float>::Values);
            case mpp::FixedFilter::ScharrHoriz:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::ScharrHoriz, 3, float>::Values);
            case mpp::FixedFilter::ScharrVert:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::ScharrVert, 3, float>::Values);
            case mpp::FixedFilter::Sharpen:
                return reinterpret_cast<const float *>(FixedFilterKernel<mpp::FixedFilter::Sharpen, 3, float>::Values);
            case mpp::FixedFilter::SobelCross:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::SobelCross, 3, float>::Values);
            case mpp::FixedFilter::SobelHoriz:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::SobelHoriz, 3, float>::Values);
            case mpp::FixedFilter::SobelVert:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::SobelVert, 3, float>::Values);
            case mpp::FixedFilter::SobelHorizSecond:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::SobelHorizSecond, 3, float>::Values);
            case mpp::FixedFilter::SobelVertSecond:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::SobelVertSecond, 3, float>::Values);
            default:
                break;
        }
    }

    if (aSize == 5)
    {
        switch (aFilter)
        {
            case mpp::FixedFilter::Gauss:
                return reinterpret_cast<const float *>(FixedFilterKernel<mpp::FixedFilter::Gauss, 5, float>::Values);
            case mpp::FixedFilter::HighPass:
                return reinterpret_cast<const float *>(FixedFilterKernel<mpp::FixedFilter::HighPass, 5, float>::Values);
            case mpp::FixedFilter::LowPass:
                return reinterpret_cast<const float *>(FixedFilterKernel<mpp::FixedFilter::LowPass, 5, float>::Values);
            case mpp::FixedFilter::Laplace:
                return reinterpret_cast<const float *>(FixedFilterKernel<mpp::FixedFilter::Laplace, 5, float>::Values);
            case mpp::FixedFilter::SobelCross:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::SobelCross, 5, float>::Values);
            case mpp::FixedFilter::SobelHoriz:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::SobelHoriz, 5, float>::Values);
            case mpp::FixedFilter::SobelVert:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::SobelVert, 5, float>::Values);
            case mpp::FixedFilter::SobelHorizSecond:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::SobelHorizSecond, 5, float>::Values);
            case mpp::FixedFilter::SobelVertSecond:
                return reinterpret_cast<const float *>(
                    FixedFilterKernel<mpp::FixedFilter::SobelVertSecond, 5, float>::Values);
            default:
                break;
        }
    }
    throw INVALIDARGUMENT(aSize aFilter, "Filter is not implemented for this size.");
}

inline const float *GetFilterInv(FixedFilter aFilter, int aSize)
{
    if (aSize == 3)
    {
        switch (aFilter)
        {
            case mpp::FixedFilter::PrewittVert:
                return reinterpret_cast<const float *>(
                    FixedInvertedFilterKernel<mpp::FixedFilter::PrewittVert, 3, float>::Values);
            case mpp::FixedFilter::SobelVert:
                return reinterpret_cast<const float *>(
                    FixedInvertedFilterKernel<mpp::FixedFilter::SobelVert, 3, float>::Values);
            default:
                break;
        }
    }

    if (aSize == 5)
    {
        switch (aFilter)
        {
            case mpp::FixedFilter::SobelVert:
                return reinterpret_cast<const float *>(
                    FixedInvertedFilterKernel<mpp::FixedFilter::SobelVert, 5, float>::Values);
            default:
                break;
        }
    }
    throw INVALIDARGUMENT(aSize aFilter, "Filter is not implemented for this size.");
}

#pragma region FixedFilter

template <PixelType T>
ImageView<T> &ImageView<T>::FixedFilter(ImageView<T> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
                                        BorderType aBorder) const
{
    return this->FixedFilter(aDst, aFilter, aMaskSize, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::FixedFilter(ImageView<T> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
                                        const T &aConstant, BorderType aBorder) const
{
    return this->FixedFilter(aDst, aFilter, aMaskSize, aConstant, aBorder, ROI());
}
template <PixelType T>
ImageView<T> &ImageView<T>::FixedFilter(ImageView<T> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
                                        BorderType aBorder, const Roi &aAllowedReadRoi) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->FixedFilter(aDst, aFilter, aMaskSize, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::FixedFilter(ImageView<T> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
                                        const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const FilterArea filterArea(GetFilterSize(aMaskSize));
    const float *filter = GetFilter(aFilter, filterArea.Size.x);

    using ComputeT = filter_compute_type_for_t<T>;

    filterEachPixel<T, ComputeT, T, float>(*this, aDst, filter, filterArea, aBorder, aConstant, aAllowedReadRoi,
                                           ComputeT(1));

    return aDst;
}

template <PixelType T>
ImageView<alternative_filter_output_type_for_t<T>> &ImageView<T>::FixedFilter(
    ImageView<alternative_filter_output_type_for_t<T>> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
    BorderType aBorder) const
    requires(has_alternative_filter_output_type_for_v<T>)
{
    return this->FixedFilter(aDst, aFilter, aMaskSize, aBorder, ROI());
}

template <PixelType T>
ImageView<alternative_filter_output_type_for_t<T>> &ImageView<T>::FixedFilter(
    ImageView<alternative_filter_output_type_for_t<T>> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
    const T &aConstant, BorderType aBorder) const
    requires(has_alternative_filter_output_type_for_v<T>)
{
    return this->FixedFilter(aDst, aFilter, aMaskSize, aConstant, aBorder, ROI());
}
template <PixelType T>
ImageView<alternative_filter_output_type_for_t<T>> &ImageView<T>::FixedFilter(
    ImageView<alternative_filter_output_type_for_t<T>> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
    BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(has_alternative_filter_output_type_for_v<T>)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->FixedFilter(aDst, aFilter, aMaskSize, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<alternative_filter_output_type_for_t<T>> &ImageView<T>::FixedFilter(
    ImageView<alternative_filter_output_type_for_t<T>> &aDst, mpp::FixedFilter aFilter, MaskSize aMaskSize,
    const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(has_alternative_filter_output_type_for_v<T>)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    const FilterArea filterArea(GetFilterSize(aMaskSize));
    const float *filter = GetFilter(aFilter, filterArea.Size.x);

    using ComputeT = filter_compute_type_for_t<T>;

    filterEachPixel<T, ComputeT, alternative_filter_output_type_for_t<T>, float>(
        *this, aDst, filter, filterArea, aBorder, aConstant, aAllowedReadRoi, ComputeT(1));

    return aDst;
}

#pragma endregion

#pragma region SeparableFilter
template <PixelType T>
ImageView<T> &ImageView<T>::SeparableFilter(ImageView<T> &aDst,
                                            const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                            int aFilterSize, int aFilterCenter, BorderType aBorder) const
{
    return this->SeparableFilter(aDst, aFilter, aFilterSize, aFilterCenter, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::SeparableFilter(ImageView<T> &aDst,
                                            const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                            int aFilterSize, int aFilterCenter, const T &aConstant,
                                            BorderType aBorder) const
{
    return this->SeparableFilter(aDst, aFilter, aFilterSize, aFilterCenter, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::SeparableFilter(ImageView<T> &aDst,
                                            const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                            int aFilterSize, int aFilterCenter, BorderType aBorder,
                                            const Roi &aAllowedReadRoi) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->SeparableFilter(aDst, aFilter, aFilterSize, aFilterCenter, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::SeparableFilter(ImageView<T> &aDst,
                                            const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                            int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder,
                                            const Roi &aAllowedReadRoi) const
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using FilterT  = filtertype_for_t<filter_compute_type_for_t<T>>;
    using ComputeT = filter_compute_type_for_t<T>;

    Image<ComputeT> temp(SizeRoi());

    const FilterArea filterRow({aFilterSize, 1}, {aFilterCenter, 0});
    const FilterArea filterCol({1, aFilterSize}, {0, aFilterCenter});

    filterEachPixel<T, ComputeT, ComputeT, FilterT>(*this, temp, aFilter, filterRow, aBorder, aConstant,
                                                    aAllowedReadRoi, ComputeT(1));
    filterEachPixel<ComputeT, ComputeT, T, FilterT>(temp, aDst, aFilter, filterCol, aBorder, ComputeT(aConstant),
                                                    aAllowedReadRoi, ComputeT(1));

    return aDst;
}

#pragma endregion

#pragma region ColumnFilter
template <PixelType T>
ImageView<T> &ImageView<T>::ColumnFilter(ImageView<T> &aDst,
                                         const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter, int aFilterSize,
                                         int aFilterCenter, BorderType aBorder) const
{
    return this->ColumnFilter(aDst, aFilter, aFilterSize, aFilterCenter, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColumnFilter(ImageView<T> &aDst,
                                         const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter, int aFilterSize,
                                         int aFilterCenter, const T &aConstant, BorderType aBorder) const
{
    return this->ColumnFilter(aDst, aFilter, aFilterSize, aFilterCenter, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColumnFilter(ImageView<T> &aDst,
                                         const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter, int aFilterSize,
                                         int aFilterCenter, BorderType aBorder, const Roi &aAllowedReadRoi) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->ColumnFilter(aDst, aFilter, aFilterSize, aFilterCenter, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ColumnFilter(ImageView<T> &aDst,
                                         const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter, int aFilterSize,
                                         int aFilterCenter, const T &aConstant, BorderType aBorder,
                                         const Roi &aAllowedReadRoi) const
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using FilterT  = filtertype_for_t<filter_compute_type_for_t<T>>;
    using ComputeT = filter_compute_type_for_t<T>;

    const FilterArea filterCol({1, aFilterSize}, {0, aFilterCenter});

    filterEachPixel<T, ComputeT, T, FilterT>(*this, aDst, aFilter, filterCol, aBorder, aConstant, aAllowedReadRoi,
                                             ComputeT(1));

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::ColumnWindowSum(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst,
    complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
    int aFilterCenter, BorderType aBorder) const
    requires RealVector<T>
{
    return this->ColumnWindowSum(aDst, aScalingValue, aFilterSize, aFilterCenter, aBorder, ROI());
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::ColumnWindowSum(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst,
    complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
    int aFilterCenter, const T &aConstant, BorderType aBorder) const
    requires RealVector<T>
{
    return this->ColumnWindowSum(aDst, aScalingValue, aFilterSize, aFilterCenter, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::ColumnWindowSum(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst,
    complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
    int aFilterCenter, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->ColumnWindowSum(aDst, aScalingValue, aFilterSize, aFilterCenter, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::ColumnWindowSum(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst,
    complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
    int aFilterCenter, const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using FilterT  = float;
    using ComputeT = filter_compute_type_for_t<T>;

    const FilterArea filterCol({1, aFilterSize}, {0, aFilterCenter});

    const std::vector<FilterT> filter(to_size_t(aFilterSize), 1.0f);

    filterEachPixel<T, ComputeT, same_vector_size_different_type_t<T, float>, FilterT>(
        *this, aDst, filter.data(), filterCol, aBorder, aConstant, aAllowedReadRoi, ComputeT(1) / aScalingValue);

    return aDst;
}

#pragma endregion

#pragma region RowFilter
template <PixelType T>
ImageView<T> &ImageView<T>::RowFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                      int aFilterSize, int aFilterCenter, BorderType aBorder) const
{
    return this->RowFilter(aDst, aFilter, aFilterSize, aFilterCenter, aBorder, ROI());
}
template <PixelType T>
ImageView<T> &ImageView<T>::RowFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                      int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder) const
{
    return this->RowFilter(aDst, aFilter, aFilterSize, aFilterCenter, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::RowFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                      int aFilterSize, int aFilterCenter, BorderType aBorder,
                                      const Roi &aAllowedReadRoi) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->RowFilter(aDst, aFilter, aFilterSize, aFilterCenter, {0}, aBorder, aAllowedReadRoi);
}
template <PixelType T>
ImageView<T> &ImageView<T>::RowFilter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                      int aFilterSize, int aFilterCenter, const T &aConstant, BorderType aBorder,
                                      const Roi &aAllowedReadRoi) const
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using FilterT  = filtertype_for_t<filter_compute_type_for_t<T>>;
    using ComputeT = filter_compute_type_for_t<T>;

    const FilterArea filterRow({aFilterSize, 1}, {aFilterCenter, 0});

    filterEachPixel<T, ComputeT, T, FilterT>(*this, aDst, aFilter, filterRow, aBorder, aConstant, aAllowedReadRoi,
                                             ComputeT(1));

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::RowWindowSum(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst,
    complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
    int aFilterCenter, BorderType aBorder) const
    requires RealVector<T>
{
    return this->RowWindowSum(aDst, aScalingValue, aFilterSize, aFilterCenter, aBorder, ROI());
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::RowWindowSum(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst,
    complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
    int aFilterCenter, const T &aConstant, BorderType aBorder) const
    requires RealVector<T>
{
    return this->RowWindowSum(aDst, aScalingValue, aFilterSize, aFilterCenter, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::RowWindowSum(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst,
    complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
    int aFilterCenter, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->RowWindowSum(aDst, aScalingValue, aFilterSize, aFilterCenter, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::RowWindowSum(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst,
    complex_basetype_t<remove_vector_t<same_vector_size_different_type_t<T, float>>> aScalingValue, int aFilterSize,
    int aFilterCenter, const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using FilterT  = float;
    using ComputeT = filter_compute_type_for_t<T>;

    const FilterArea filterRow({aFilterSize, 1}, {aFilterCenter, 0});

    const std::vector<FilterT> filter(to_size_t(aFilterSize), 1.0f);

    filterEachPixel<T, ComputeT, same_vector_size_different_type_t<T, float>, FilterT>(
        *this, aDst, filter.data(), filterRow, aBorder, aConstant, aAllowedReadRoi, ComputeT(1) / aScalingValue);

    return aDst;
}

#pragma endregion

#pragma region BoxFilter

template <PixelType T>
ImageView<T> &ImageView<T>::BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder) const
{
    return this->BoxFilter(aDst, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                                      BorderType aBorder) const
{
    return this->BoxFilter(aDst, aFilterArea, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                      const Roi &aAllowedReadRoi) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->BoxFilter(aDst, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                                      BorderType aBorder, const Roi &aAllowedReadRoi) const
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using FilterT  = float;
    using ComputeT = filter_compute_type_for_t<T>;

    const std::vector<FilterT> filter(aFilterArea.Size.TotalSize(), 1.0f);
    const ComputeT scale = ComputeT(1) / ComputeT(to_int(aFilterArea.Size.TotalSize()));

    filterEachPixel<T, ComputeT, T, FilterT>(*this, aDst, filter.data(), aFilterArea, aBorder, aConstant,
                                             aAllowedReadRoi, scale);

    return aDst;
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::BoxFilter(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea,
    BorderType aBorder) const
    requires RealIntVector<T>
{
    return this->BoxFilter(aDst, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::BoxFilter(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea, const T &aConstant,
    BorderType aBorder) const
    requires RealIntVector<T>
{
    return this->BoxFilter(aDst, aFilterArea, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::BoxFilter(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
    const Roi &aAllowedReadRoi) const
    requires RealIntVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->BoxFilter(aDst, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<same_vector_size_different_type_t<T, float>> &ImageView<T>::BoxFilter(
    ImageView<same_vector_size_different_type_t<T, float>> &aDst, const FilterArea &aFilterArea, const T &aConstant,
    BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealIntVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using FilterT  = float;
    using ComputeT = filter_compute_type_for_t<T>;

    const std::vector<FilterT> filter(aFilterArea.Size.TotalSize(), 1.0f);
    const ComputeT scale = 1.0f / to_float(aFilterArea.Size.TotalSize());

    filterEachPixel<T, ComputeT, same_vector_size_different_type_t<T, float>, FilterT>(
        *this, aDst, filter.data(), aFilterArea, aBorder, aConstant, aAllowedReadRoi, scale);

    return aDst;
}

template <PixelType T>
ImageView<Pixel32fC2> &ImageView<T>::BoxAndSumSquareFilter(ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea,
                                                           BorderType aBorder) const
    requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8)
{
    return this->BoxAndSumSquareFilter(aDst, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<Pixel32fC2> &ImageView<T>::BoxAndSumSquareFilter(ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea,
                                                           const T &aConstant, BorderType aBorder) const
    requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8)
{
    return this->BoxAndSumSquareFilter(aDst, aFilterArea, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<Pixel32fC2> &ImageView<T>::BoxAndSumSquareFilter(ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea,
                                                           BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->BoxAndSumSquareFilter(aDst, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<Pixel32fC2> &ImageView<T>::BoxAndSumSquareFilter(ImageView<Pixel32fC2> &aDst, const FilterArea &aFilterArea,
                                                           const T &aConstant, BorderType aBorder,
                                                           const Roi &aAllowedReadRoi) const
    requires RealVector<T> && SingleChannel<T> && (sizeof(T) < 8)
{
    Image<Pixel32fC2> temp(SizeRoi());
    Pixel32fC2 constant;

    constant.x = Pixel32fC1(aConstant).x;
    constant.y = constant.x * constant.x;

    for (const auto &p : SizeRoi())
    {
        temp(p.Pixel.x, p.Pixel.y).x = Pixel32fC1((*this)(p.Pixel.x, p.Pixel.y)).x;
        temp(p.Pixel.x, p.Pixel.y).y = temp(p.Pixel.x, p.Pixel.y).x * temp(p.Pixel.x, p.Pixel.y).x;
    }

    temp.BoxFilter(aDst, aFilterArea, constant, aBorder, aAllowedReadRoi);

    return aDst;
}
#pragma endregion

#pragma region Min/Max Filter
template <PixelType T>
ImageView<T> &ImageView<T>::MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder) const
    requires RealVector<T>
{
    return this->MaxFilter(aDst, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                                      BorderType aBorder) const
    requires RealVector<T>
{
    return this->MaxFilter(aDst, aFilterArea, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                      const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->MaxFilter(aDst, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::MaxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                                      BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    maxFilterEachPixel(*this, aDst, aFilterArea, aBorder, aConstant, aAllowedReadRoi);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder) const
    requires RealVector<T>
{
    return this->MinFilter(aDst, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                                      BorderType aBorder) const
    requires RealVector<T>
{
    return this->MinFilter(aDst, aFilterArea, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                      const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->MinFilter(aDst, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::MinFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, const T &aConstant,
                                      BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    minFilterEachPixel(*this, aDst, aFilterArea, aBorder, aConstant, aAllowedReadRoi);

    return aDst;
}
#pragma endregion

#pragma region Median Filter
template <PixelType T>
ImageView<T> &ImageView<T>::MedianFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder) const
    requires RealVector<T>
{
    return this->MedianFilter(aDst, aFilterArea, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::MedianFilter(ImageView<T> &aDst, const FilterArea &aFilterArea, BorderType aBorder,
                                         const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    medianFilterEachPixel(*this, aDst, aFilterArea, aBorder, aAllowedReadRoi);

    return aDst;
}

#pragma endregion

#pragma region Wiener Filter
template <PixelType T>
ImageView<T> &ImageView<T>::WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                         const filter_compute_type_for_t<T> &aNoise, BorderType aBorder) const
    requires RealVector<T>
{
    return this->WienerFilter(aDst, aFilterArea, aNoise, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                         const filter_compute_type_for_t<T> &aNoise, const T &aConstant,
                                         BorderType aBorder) const
    requires RealVector<T>
{
    return this->WienerFilter(aDst, aFilterArea, aNoise, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                         const filter_compute_type_for_t<T> &aNoise, BorderType aBorder,
                                         const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->WienerFilter(aDst, aFilterArea, aNoise, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::WienerFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                         const filter_compute_type_for_t<T> &aNoise, const T &aConstant,
                                         BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    wienerFilterEachPixel(*this, aDst, aFilterArea, aNoise, aBorder, aConstant, aAllowedReadRoi);

    return aDst;
}

#pragma endregion

#pragma region Threshold Adaptive Box Filter
template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdAdaptiveBoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                       const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
                                                       const T &aValLE, BorderType aBorder) const
    requires RealVector<T>
{
    return this->ThresholdAdaptiveBoxFilter(aDst, aFilterArea, aDelta, aValGT, aValLE, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdAdaptiveBoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                       const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
                                                       const T &aValLE, const T &aConstant, BorderType aBorder) const
    requires RealVector<T>
{
    return this->ThresholdAdaptiveBoxFilter(aDst, aFilterArea, aDelta, aValGT, aValLE, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdAdaptiveBoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                       const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
                                                       const T &aValLE, BorderType aBorder,
                                                       const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->ThresholdAdaptiveBoxFilter(aDst, aFilterArea, aDelta, aValGT, aValLE, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::ThresholdAdaptiveBoxFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                       const filter_compute_type_for_t<T> &aDelta, const T &aValGT,
                                                       const T &aValLE, const T &aConstant, BorderType aBorder,
                                                       const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    thresholdAdaptiveBoxFilterEachPixel(*this, aDst, aFilterArea, aDelta, aValGT, aValLE, aBorder, aConstant,
                                        aAllowedReadRoi);

    return aDst;
}

#pragma endregion

#pragma region Filter
template <PixelType T>
ImageView<T> &ImageView<T>::Filter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                   const FilterArea &aFilterArea, BorderType aBorder) const
{
    return this->Filter(aDst, aFilter, aFilterArea, {0}, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::Filter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                   const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder) const
{
    return this->Filter(aDst, aFilter, aFilterArea, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::Filter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                   const FilterArea &aFilterArea, BorderType aBorder, const Roi &aAllowedReadRoi) const
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->Filter(aDst, aFilter, aFilterArea, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::Filter(ImageView<T> &aDst, const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                   const FilterArea &aFilterArea, const T &aConstant, BorderType aBorder,
                                   const Roi &aAllowedReadRoi) const
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using ComputeT = filter_compute_type_for_t<T>;

    filterEachPixel<T, ComputeT, T, filtertype_for_t<filter_compute_type_for_t<T>>>(
        *this, aDst, aFilter, aFilterArea, aBorder, aConstant, aAllowedReadRoi, ComputeT(1));

    return aDst;
}

#pragma endregion

#pragma region Bilateral Gauss Filter

template <PixelType T>
ImageView<T> &ImageView<T>::BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                 const float *aPreCompGeomDistCoeff, float aValSquareSigma,
                                                 BorderType aBorder) const
    requires SingleChannel<T> && RealVector<T> &&
             (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>)
{
    return this->BilateralGaussFilter(aDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                 const float *aPreCompGeomDistCoeff, float aValSquareSigma,
                                                 const T &aConstant, BorderType aBorder) const
    requires SingleChannel<T> && RealVector<T> &&
             (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>)
{
    return this->BilateralGaussFilter(aDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aConstant, aBorder,
                                      ROI());
}

template <PixelType T>
void ImageView<T>::PrecomputeBilateralGaussFilter(float *aPreCompGeomDistCoeff, const FilterArea &aFilterArea,
                                                  float aPosSquareSigma) const
    requires RealVector<T> && (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>)
{
    if (!aFilterArea.CheckIfValid())
    {
        throw INVALIDARGUMENT(aFilterArea, "Invalid filter area: " << aFilterArea);
    }

    for (const auto &elem : aFilterArea.Size)
    {
        const float idxX    = static_cast<float>(elem.Pixel.x - aFilterArea.Center.x);
        const float idxY    = static_cast<float>(elem.Pixel.y - aFilterArea.Center.y);
        const float distSqr = idxX * idxX + idxY * idxY;

        *aPreCompGeomDistCoeff = std::exp(-distSqr / (2.0f * aPosSquareSigma));
        aPreCompGeomDistCoeff++;
    }
}

template <PixelType T>
ImageView<T> &ImageView<T>::BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                 const float *aPreCompGeomDistCoeff, float aValSquareSigma,
                                                 BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires SingleChannel<T> && RealVector<T> &&
             (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->BilateralGaussFilter(aDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, {0}, aBorder,
                                      aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                 const float *aPreCompGeomDistCoeff, float aValSquareSigma,
                                                 const T &aConstant, BorderType aBorder,
                                                 const Roi &aAllowedReadRoi) const
    requires SingleChannel<T> && RealVector<T> &&
             (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>)
{
    if (!aFilterArea.CheckIfValid())
    {
        throw INVALIDARGUMENT(aFilterArea, "Invalid filter area: " << aFilterArea);
    }

    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using ComputeT = filter_compute_type_for_t<T>;

    bilateralFilterEachPixel<T, ComputeT, T, float>(*this, aDst, aPreCompGeomDistCoeff, aValSquareSigma, aFilterArea,
                                                    Norm::L1, aBorder, aConstant, aAllowedReadRoi);

    return aDst;
}

template <PixelType T>
ImageView<T> &ImageView<T>::BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                 const float *aPreCompGeomDistCoeff, float aValSquareSigma,
                                                 mpp::Norm aNorm, BorderType aBorder) const
    requires(!SingleChannel<T>) && RealVector<T> &&
            (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>)
{
    return this->BilateralGaussFilter(aDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aNorm, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                 const float *aPreCompGeomDistCoeff, float aValSquareSigma,
                                                 mpp::Norm aNorm, const T &aConstant, BorderType aBorder) const
    requires(!SingleChannel<T>) && RealVector<T> &&
            (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>)
{
    return this->BilateralGaussFilter(aDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aNorm, aConstant,
                                      aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                 const float *aPreCompGeomDistCoeff, float aValSquareSigma,
                                                 mpp::Norm aNorm, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(!SingleChannel<T>) && RealVector<T> &&
            (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->BilateralGaussFilter(aDst, aFilterArea, aPreCompGeomDistCoeff, aValSquareSigma, aNorm, {0}, aBorder,
                                      aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::BilateralGaussFilter(ImageView<T> &aDst, const FilterArea &aFilterArea,
                                                 const float *aPreCompGeomDistCoeff, float aValSquareSigma,
                                                 mpp::Norm aNorm, const T &aConstant, BorderType aBorder,
                                                 const Roi &aAllowedReadRoi) const
    requires(!SingleChannel<T>) && RealVector<T> &&
            (sizeof(remove_vector_t<T>) < 4 || std::same_as<remove_vector_t<T>, float>)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using ComputeT = filter_compute_type_for_t<T>;
    bilateralFilterEachPixel<T, ComputeT, T, float>(*this, aDst, aPreCompGeomDistCoeff, aValSquareSigma, aFilterArea,
                                                    aNorm, aBorder, aConstant, aAllowedReadRoi);

    return aDst;
}
#pragma endregion

#pragma region Gradient Vector

template <PixelType T>
void ImageView<T>::GradientVectorSobel(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                       ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                       ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                       BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->GradientVectorSobel(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, {0}, aBorder,
                                     aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::GradientVectorSobel(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                       ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                       ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                       const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    // find first not nullptr output:
    Size2D refSize = aDstX.SizeRoi();
    if (aDstX.Pointer() == nullptr)
    {
        refSize = aDstY.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr)
    {
        refSize = aDstMag.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr)
    {
        refSize = aDstAngle.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr)
    {
        refSize = aDstCovariance.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr && aDstCovariance.Pointer() == nullptr)
    {
        throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle aDstCovariance,
                              "All output images are nullptr, at least one output must be provided.");
    }

    if (aDstX.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstX.SizeRoi());
    }
    if (aDstY.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstY.SizeRoi());
    }
    if (aDstMag.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstMag.SizeRoi());
    }
    if (aDstAngle.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstAngle.SizeRoi());
    }
    if (aDstCovariance.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstCovariance.SizeRoi());
    }

    using ComputeT = filter_compute_type_for_t<T>;

    const FilterArea filterArea(GetFilterSize(aMaskSize));
    const float *filterX = GetFilterInv(mpp::FixedFilter::SobelVert, filterArea.Size.x);
    const float *filterY = GetFilter(mpp::FixedFilter::SobelHoriz, filterArea.Size.x);

    gradientVectorEachPixel<T, ComputeT, Pixel16sC1, float>(*this, aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance,
                                                            filterX, filterY, filterArea, aNorm, aBorder, aConstant,
                                                            aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::GradientVectorSobel(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                       ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                       ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                       BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->GradientVectorSobel(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, {0}, aBorder,
                                     aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::GradientVectorSobel(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                       ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                       ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                       const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    // find first not nullptr output:
    Size2D refSize = aDstX.SizeRoi();
    if (aDstX.Pointer() == nullptr)
    {
        refSize = aDstY.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr)
    {
        refSize = aDstMag.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr)
    {
        refSize = aDstAngle.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr)
    {
        refSize = aDstCovariance.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr && aDstCovariance.Pointer() == nullptr)
    {
        throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle aDstCovariance,
                              "All output images are nullptr, at least one output must be provided.");
    }

    if (aDstX.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstX.SizeRoi());
    }
    if (aDstY.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstY.SizeRoi());
    }
    if (aDstMag.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstMag.SizeRoi());
    }
    if (aDstAngle.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstAngle.SizeRoi());
    }
    if (aDstCovariance.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstCovariance.SizeRoi());
    }

    using ComputeT = filter_compute_type_for_t<T>;

    const FilterArea filterArea(GetFilterSize(aMaskSize));
    const float *filterX = GetFilterInv(mpp::FixedFilter::SobelVert, filterArea.Size.x);
    const float *filterY = GetFilter(mpp::FixedFilter::SobelHoriz, filterArea.Size.x);

    gradientVectorEachPixel<T, ComputeT, Pixel32fC1, float>(*this, aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance,
                                                            filterX, filterY, filterArea, aNorm, aBorder, aConstant,
                                                            aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::GradientVectorScharr(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                        ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                        ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                        BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->GradientVectorScharr(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, {0}, aBorder,
                                      aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::GradientVectorScharr(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                        ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                        ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                        const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    // find first not nullptr output:
    Size2D refSize = aDstX.SizeRoi();
    if (aDstX.Pointer() == nullptr)
    {
        refSize = aDstY.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr)
    {
        refSize = aDstMag.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr)
    {
        refSize = aDstAngle.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr)
    {
        refSize = aDstCovariance.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr && aDstCovariance.Pointer() == nullptr)
    {
        throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle aDstCovariance,
                              "All output images are nullptr, at least one output must be provided.");
    }

    if (aDstX.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstX.SizeRoi());
    }
    if (aDstY.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstY.SizeRoi());
    }
    if (aDstMag.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstMag.SizeRoi());
    }
    if (aDstAngle.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstAngle.SizeRoi());
    }
    if (aDstCovariance.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstCovariance.SizeRoi());
    }

    using ComputeT = filter_compute_type_for_t<T>;

    const FilterArea filterArea(GetFilterSize(aMaskSize));
    const float *filterX = GetFilter(mpp::FixedFilter::ScharrVert, filterArea.Size.x); // not inverted!
    const float *filterY = GetFilter(mpp::FixedFilter::ScharrHoriz, filterArea.Size.x);

    gradientVectorEachPixel<T, ComputeT, Pixel16sC1, float>(*this, aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance,
                                                            filterX, filterY, filterArea, aNorm, aBorder, aConstant,
                                                            aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::GradientVectorScharr(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                        ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                        ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                        BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->GradientVectorScharr(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, {0}, aBorder,
                                      aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::GradientVectorScharr(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                        ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                        ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                        const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    // find first not nullptr output:
    Size2D refSize = aDstX.SizeRoi();
    if (aDstX.Pointer() == nullptr)
    {
        refSize = aDstY.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr)
    {
        refSize = aDstMag.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr)
    {
        refSize = aDstAngle.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr)
    {
        refSize = aDstCovariance.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr && aDstCovariance.Pointer() == nullptr)
    {
        throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle aDstCovariance,
                              "All output images are nullptr, at least one output must be provided.");
    }

    if (aDstX.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstX.SizeRoi());
    }
    if (aDstY.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstY.SizeRoi());
    }
    if (aDstMag.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstMag.SizeRoi());
    }
    if (aDstAngle.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstAngle.SizeRoi());
    }
    if (aDstCovariance.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstCovariance.SizeRoi());
    }

    using ComputeT = filter_compute_type_for_t<T>;

    const FilterArea filterArea(GetFilterSize(aMaskSize));
    const float *filterX = GetFilter(mpp::FixedFilter::ScharrVert, filterArea.Size.x); // not inverted!
    const float *filterY = GetFilter(mpp::FixedFilter::ScharrHoriz, filterArea.Size.x);

    gradientVectorEachPixel<T, ComputeT, Pixel32fC1, float>(*this, aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance,
                                                            filterX, filterY, filterArea, aNorm, aBorder, aConstant,
                                                            aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::GradientVectorPrewitt(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                         ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                         ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                         BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->GradientVectorPrewitt(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, {0}, aBorder,
                                       aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::GradientVectorPrewitt(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                         ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                         ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                         const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    // find first not nullptr output:
    Size2D refSize = aDstX.SizeRoi();
    if (aDstX.Pointer() == nullptr)
    {
        refSize = aDstY.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr)
    {
        refSize = aDstMag.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr)
    {
        refSize = aDstAngle.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr)
    {
        refSize = aDstCovariance.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr && aDstCovariance.Pointer() == nullptr)
    {
        throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle aDstCovariance,
                              "All output images are nullptr, at least one output must be provided.");
    }

    if (aDstX.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstX.SizeRoi());
    }
    if (aDstY.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstY.SizeRoi());
    }
    if (aDstMag.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstMag.SizeRoi());
    }
    if (aDstAngle.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstAngle.SizeRoi());
    }
    if (aDstCovariance.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstCovariance.SizeRoi());
    }

    using ComputeT = filter_compute_type_for_t<T>;

    const FilterArea filterArea(GetFilterSize(aMaskSize));
    const float *filterX = GetFilterInv(mpp::FixedFilter::PrewittVert, filterArea.Size.x);
    const float *filterY = GetFilter(mpp::FixedFilter::PrewittHoriz, filterArea.Size.x);

    gradientVectorEachPixel<T, ComputeT, Pixel16sC1, float>(*this, aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance,
                                                            filterX, filterY, filterArea, aNorm, aBorder, aConstant,
                                                            aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::GradientVectorPrewitt(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                         ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                         ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                         BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->GradientVectorPrewitt(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, {0}, aBorder,
                                       aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::GradientVectorPrewitt(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                         ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                         ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                         const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    // find first not nullptr output:
    Size2D refSize = aDstX.SizeRoi();
    if (aDstX.Pointer() == nullptr)
    {
        refSize = aDstY.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr)
    {
        refSize = aDstMag.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr)
    {
        refSize = aDstAngle.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr)
    {
        refSize = aDstCovariance.SizeRoi();
    }
    if (aDstX.Pointer() == nullptr && aDstY.Pointer() == nullptr && aDstMag.Pointer() == nullptr &&
        aDstAngle.Pointer() == nullptr && aDstCovariance.Pointer() == nullptr)
    {
        throw INVALIDARGUMENT(aDstX aDstY aDstMag aDstAngle aDstCovariance,
                              "All output images are nullptr, at least one output must be provided.");
    }

    if (aDstX.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstX.SizeRoi());
    }
    if (aDstY.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstY.SizeRoi());
    }
    if (aDstMag.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstMag.SizeRoi());
    }
    if (aDstAngle.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstAngle.SizeRoi());
    }
    if (aDstCovariance.Pointer() != nullptr)
    {
        checkSameSize(refSize, aDstCovariance.SizeRoi());
    }

    using ComputeT = filter_compute_type_for_t<T>;

    const FilterArea filterArea(GetFilterSize(aMaskSize));
    const float *filterX = GetFilterInv(mpp::FixedFilter::PrewittVert, filterArea.Size.x);
    const float *filterY = GetFilter(mpp::FixedFilter::PrewittHoriz, filterArea.Size.x);

    gradientVectorEachPixel<T, ComputeT, Pixel32fC1, float>(*this, aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance,
                                                            filterX, filterY, filterArea, aNorm, aBorder, aConstant,
                                                            aAllowedReadRoi);
}

template <PixelType T>
void ImageView<T>::GradientVectorSobel(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                       ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                       ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                       BorderType aBorder) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    return this->GradientVectorSobel(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, aBorder,
                                     ROI());
}

template <PixelType T>
void ImageView<T>::GradientVectorSobel(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                       ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                       ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                       const T &aConstant, BorderType aBorder) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    return this->GradientVectorSobel(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, aConstant,
                                     aBorder, ROI());
}

template <PixelType T>
void ImageView<T>::GradientVectorSobel(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                       ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                       ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                       BorderType aBorder) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    return this->GradientVectorSobel(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, aBorder,
                                     ROI());
}

template <PixelType T>
void ImageView<T>::GradientVectorSobel(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                       ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                       ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                       const T &aConstant, BorderType aBorder) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    return this->GradientVectorSobel(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, aConstant,
                                     aBorder, ROI());
}

template <PixelType T>
void ImageView<T>::GradientVectorScharr(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                        ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                        ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                        BorderType aBorder) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    return this->GradientVectorScharr(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, aBorder,
                                      ROI());
}

template <PixelType T>
void ImageView<T>::GradientVectorScharr(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                        ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                        ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                        const T &aConstant, BorderType aBorder) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    return this->GradientVectorScharr(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, aConstant,
                                      aBorder, ROI());
}

template <PixelType T>
void ImageView<T>::GradientVectorScharr(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                        ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                        ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                        BorderType aBorder) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    return this->GradientVectorScharr(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, aBorder,
                                      ROI());
}

template <PixelType T>
void ImageView<T>::GradientVectorScharr(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                        ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                        ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                        const T &aConstant, BorderType aBorder) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    return this->GradientVectorScharr(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, aConstant,
                                      aBorder, ROI());
}

template <PixelType T>
void ImageView<T>::GradientVectorPrewitt(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                         ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                         ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                         BorderType aBorder) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    return this->GradientVectorPrewitt(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, aBorder,
                                       ROI());
}

template <PixelType T>
void ImageView<T>::GradientVectorPrewitt(ImageView<Pixel16sC1> &aDstX, ImageView<Pixel16sC1> &aDstY,
                                         ImageView<Pixel16sC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                         ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                         const T &aConstant, BorderType aBorder) const
    requires(std::same_as<remove_vector_t<T>, byte> || std::same_as<remove_vector_t<T>, sbyte>)
{
    return this->GradientVectorPrewitt(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, aConstant,
                                       aBorder, ROI());
}

template <PixelType T>
void ImageView<T>::GradientVectorPrewitt(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                         ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                         ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                         BorderType aBorder) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    return this->GradientVectorPrewitt(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, aBorder,
                                       ROI());
}

template <PixelType T>
void ImageView<T>::GradientVectorPrewitt(ImageView<Pixel32fC1> &aDstX, ImageView<Pixel32fC1> &aDstY,
                                         ImageView<Pixel32fC1> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                                         ImageView<Pixel32fC4> &aDstCovariance, Norm aNorm, MaskSize aMaskSize,
                                         const T &aConstant, BorderType aBorder) const
    requires(std::same_as<remove_vector_t<T>, short> || std::same_as<remove_vector_t<T>, ushort> ||
             std::same_as<remove_vector_t<T>, float>)
{
    return this->GradientVectorPrewitt(aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aNorm, aMaskSize, aConstant,
                                       aBorder, ROI());
}
#pragma endregion

#pragma region Unsharp Filter

template <PixelType T>
ImageView<T> &ImageView<T>::UnsharpFilter(ImageView<T> &aDst,
                                          const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                          int aFilterSize, int aFilterCenter,
                                          remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
                                          remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold,
                                          BorderType aBorder) const
    requires RealVector<T>
{
    return this->UnsharpFilter(aDst, aFilter, aFilterSize, aFilterCenter, aWeight, aThreshold, aBorder, ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::UnsharpFilter(ImageView<T> &aDst,
                                          const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                          int aFilterSize, int aFilterCenter,
                                          remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
                                          remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold,
                                          const T &aConstant, BorderType aBorder) const
    requires RealVector<T>
{
    return this->UnsharpFilter(aDst, aFilter, aFilterSize, aFilterCenter, aWeight, aThreshold, aConstant, aBorder,
                               ROI());
}

template <PixelType T>
ImageView<T> &ImageView<T>::UnsharpFilter(ImageView<T> &aDst,
                                          const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                          int aFilterSize, int aFilterCenter,
                                          remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
                                          remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold,
                                          BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->UnsharpFilter(aDst, aFilter, aFilterSize, aFilterCenter, aWeight, aThreshold, {0}, aBorder,
                               aAllowedReadRoi);
}

template <PixelType T>
ImageView<T> &ImageView<T>::UnsharpFilter(ImageView<T> &aDst,
                                          const filtertype_for_t<filter_compute_type_for_t<T>> *aFilter,
                                          int aFilterSize, int aFilterCenter,
                                          remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aWeight,
                                          remove_vector_t<filtertype_for_t<filter_compute_type_for_t<T>>> aThreshold,
                                          const T &aConstant, BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires RealVector<T>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    using FilterT  = filtertype_for_t<filter_compute_type_for_t<T>>;
    using ComputeT = filter_compute_type_for_t<T>;

    Image<ComputeT> temp(SizeRoi());

    const FilterArea filterRow({aFilterSize, 1}, {aFilterCenter, 0});
    const FilterArea filterCol({1, aFilterSize}, {0, aFilterCenter});

    filterEachPixel<T, ComputeT, ComputeT, FilterT>(*this, temp, aFilter, filterRow, aBorder, aConstant,
                                                    aAllowedReadRoi, ComputeT(1));
    unsharpFilterEachPixel<ComputeT, ComputeT, T, FilterT>(temp, *this, aDst, aFilter, aWeight, aThreshold, filterCol,
                                                           aBorder, ComputeT(aConstant), aAllowedReadRoi);

    return aDst;
}
#pragma endregion

#pragma region Harris Corner Response

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::HarrisCornerResponse(ImageView<Pixel32fC1> &aDst, const FilterArea &aAvgWindowSize,
                                                          float aK, float aScale, BorderType aBorder) const
    requires std::same_as<T, Pixel32fC4>
{
    return this->HarrisCornerResponse(aDst, aAvgWindowSize, aK, aScale, aBorder, ROI());
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::HarrisCornerResponse(ImageView<Pixel32fC1> &aDst, const FilterArea &aAvgWindowSize,
                                                          float aK, float aScale, const T &aConstant,
                                                          BorderType aBorder) const
    requires std::same_as<T, Pixel32fC4>
{
    return this->HarrisCornerResponse(aDst, aAvgWindowSize, aK, aScale, aConstant, aBorder, ROI());
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::HarrisCornerResponse(ImageView<Pixel32fC1> &aDst, const FilterArea &aAvgWindowSize,
                                                          float aK, float aScale, BorderType aBorder,
                                                          const Roi &aAllowedReadRoi) const
    requires std::same_as<T, Pixel32fC4>
{
    if (aBorder == BorderType::Constant)
    {
        throw INVALIDARGUMENT(aBorder,
                              "When using BorderType::Constant, the constant value aConstant must be provided.");
    }
    return this->HarrisCornerResponse(aDst, aAvgWindowSize, aK, aScale, {0}, aBorder, aAllowedReadRoi);
}

template <PixelType T>
ImageView<Pixel32fC1> &ImageView<T>::HarrisCornerResponse(ImageView<Pixel32fC1> &aDst, const FilterArea &aAvgWindowSize,
                                                          float aK, float aScale, const T &aConstant,
                                                          BorderType aBorder, const Roi &aAllowedReadRoi) const
    requires std::same_as<T, Pixel32fC4>
{
    checkSameSize(SizeRoi(), aDst.SizeRoi());

    Image<Pixel32fC4> boxfiltered(SizeRoi());

    this->BoxFilter(boxfiltered, aAvgWindowSize, aConstant, aBorder, aAllowedReadRoi);

    using PostOpT = mpp::HarrisCorner;
    const PostOpT postOp(aK, aScale);

    for (const auto &elem : SizeRoi())
    {
        const int x = elem.Pixel.x;
        const int y = elem.Pixel.y;

        postOp(boxfiltered(x, y), aDst(x, y));
    }

    return aDst;
}

#pragma endregion

#pragma region Canny edge
template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CannyEdge(const ImageView<Pixel32fC1> &aSrcAngle, ImageView<Pixel8uC1> &aTemp,
                                              ImageView<Pixel8uC1> &aDst, T aLowThreshold, T aHighThreshold) const
    requires std::same_as<T, Pixel16sC1> || std::same_as<T, Pixel32fC1>
{
    return this->CannyEdge(aSrcAngle, aTemp, aDst, aLowThreshold, aHighThreshold, ROI());
}

template <PixelType T>
ImageView<Pixel8uC1> &ImageView<T>::CannyEdge(const ImageView<Pixel32fC1> &aSrcAngle, ImageView<Pixel8uC1> &aTemp,
                                              ImageView<Pixel8uC1> &aDst, T aLowThreshold, T aHighThreshold,
                                              const Roi &aAllowedReadRoi) const
    requires std::same_as<T, Pixel16sC1> || std::same_as<T, Pixel32fC1>
{
    checkRoiIsInRoi(aAllowedReadRoi, Roi(0, 0, SizeAlloc()));

    cannyEdgeMaxSupressionEachPixel(*this, aSrcAngle.PointerRoi(), aSrcAngle.Pitch(), aTemp, aAllowedReadRoi,
                                    aLowThreshold, aHighThreshold);

    cannyEdgeHysteresisEachPixel(aTemp, aSrcAngle.PointerRoi(), aSrcAngle.Pitch(), aDst, aTemp.ROI());

    return aDst;
}
#pragma endregion
} // namespace mpp::image::cpuSimple