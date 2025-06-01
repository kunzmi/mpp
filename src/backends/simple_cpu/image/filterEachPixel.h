#pragma once
#include <backends/cuda/cudaException.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/image/filterArea.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <common/utilities.h>

#include <iostream>

namespace opp::image::cpuSimple
{
// forward declaration
template <PixelType T> class ImageView;

/// <summary>
/// runs aFilter on every pixel of an image.
/// </summary>
template <typename SrcT, typename ComputeT, typename DstT, typename FilterT>
void filterEachPixel(const ImageView<SrcT> &aSrc, ImageView<DstT> &aDst, const FilterT *aFilter,
                     const FilterArea &aFilterArea, BorderType aBorderType, SrcT aConstant, const Roi &aAllowedReadRoi,
                     ComputeT aScale);

/// <summary>
/// runs Min on every pixel of an image.
/// </summary>
template <typename SrcT>
void minFilterEachPixel(const ImageView<SrcT> &aSrc, ImageView<SrcT> &aDst, const FilterArea &aFilterArea,
                        BorderType aBorderType, SrcT aConstant, const Roi &aAllowedReadRoi);
/// <summary>
/// runs Max on every pixel of an image.
/// </summary>
template <typename SrcT>
void maxFilterEachPixel(const ImageView<SrcT> &aSrc, ImageView<SrcT> &aDst, const FilterArea &aFilterArea,
                        BorderType aBorderType, SrcT aConstant, const Roi &aAllowedReadRoi);

/// <summary>
/// runs wiener filter on every pixel of an image.
/// </summary>
template <typename SrcT>
void wienerFilterEachPixel(const ImageView<SrcT> &aSrc, ImageView<SrcT> &aDst, const FilterArea &aFilterArea,
                           const filter_compute_type_for_t<SrcT> &aNoise, BorderType aBorderType, SrcT aConstant,
                           const Roi &aAllowedReadRoi);
/// <summary>
/// runs wiener filter on every pixel of an image.
/// </summary>
template <typename SrcT>
void thresholdAdaptiveBoxFilterEachPixel(const ImageView<SrcT> &aSrc, ImageView<SrcT> &aDst,
                                         const FilterArea &aFilterArea, const filter_compute_type_for_t<SrcT> &aDelta,
                                         const SrcT &aValGT, const SrcT &aValLE, BorderType aBorderType, SrcT aConstant,
                                         const Roi &aAllowedReadRoi);

template <typename SrcT, typename ComputeT, typename DstT, typename FilterT>
void bilateralFilterEachPixel(const ImageView<SrcT> &aSrc, ImageView<DstT> &aDst, const FilterT *aPreComputedFilter,
                              float aValSquareSigma, const FilterArea &aFilterArea, Norm aNorm, BorderType aBorderType,
                              SrcT aConstant, const Roi &aAllowedReadRoi);

/// <summary>
/// runs aFilter on every pixel of an image.
/// </summary>
template <typename SrcT, typename ComputeT, typename DstT, typename FilterT>
void gradientVectorEachPixel(const ImageView<SrcT> &aSrc, ImageView<DstT> &aDstX, ImageView<DstT> &aDstY,
                             ImageView<DstT> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                             ImageView<Pixel32fC4> &aDstCovariance, const FilterT *aFilterX, const FilterT *aFilterY,
                             const FilterArea &aFilterArea, Norm aNorm, BorderType aBorderType, SrcT aConstant,
                             const Roi &aAllowedReadRoi);

template <typename SrcT, typename ComputeT, typename DstT, typename FilterT>
void unsharpFilterEachPixel(const ImageView<SrcT> &aSrc, ImageView<DstT> &aDst, const FilterT *aFilter, FilterT aWeight,
                            FilterT aThreshold, const FilterArea &aFilterArea, BorderType aBorderType, SrcT aConstant,
                            const Roi &aAllowedReadRoi);
} // namespace opp::image::cpuSimple