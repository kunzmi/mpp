#pragma once
#include "filterEachPixel.h"
#include <backends/cuda/cudaException.h>
#include <backends/simple_cpu/image/imageView.h>
#include <common/image/functors/borderControl.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/image/threadSplit.h>
#include <common/tupel.h>
#include <common/utilities.h>

#include <iostream>

namespace opp::image::cpuSimple
{

template <typename BorderControlT, typename ComputeT, typename DstT, typename FilterT>
void filterEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDst, const FilterT *aFilter,
                     const FilterArea &aFilterArea, ComputeT aScale)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        ComputeT temp(0);
        DstT &pixelOut = pixelIterator.Value();

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            temp += ComputeT(aSrcWithBC(i, j)) * aFilter[f.Pixel.y * aFilterArea.Size.x + f.Pixel.x];
        }
        temp *= aScale;

        if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
        {
            temp.RoundNearest();
        }

        DstT res = DstT(temp);

        // restore alpha channel value:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            res.w = pixelOut.w;
        }

        pixelOut = res;
    }
}

template <typename SrcT, typename ComputeT, typename DstT, typename FilterT>
void filterEachPixel(const ImageView<SrcT> &aSrc, ImageView<DstT> &aDst, const FilterT *aFilter,
                     const FilterArea &aFilterArea, BorderType aBorderType, SrcT aConstant, const Roi &aAllowedReadRoi,
                     ComputeT aScale)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            filterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aDst, aFilter, aFilterArea, aScale);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            filterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aDst, aFilter, aFilterArea, aScale);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            filterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aDst, aFilter, aFilterArea, aScale);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            filterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aDst, aFilter, aFilterArea, aScale);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            filterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aDst, aFilter, aFilterArea, aScale);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            filterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aDst, aFilter, aFilterArea, aScale);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for filter.");
            break;
    }
}

template <typename BorderControlT, typename DstT>
void minFilterEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDst, const FilterArea &aFilterArea)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        DstT res       = numeric_limits<remove_vector_t<DstT>>::max();
        DstT &pixelOut = pixelIterator.Value();

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            res.Min(aSrcWithBC(i, j));
        }

        // restore alpha channel value:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            res.w = pixelOut.w;
        }

        pixelOut = res;
    }
}

template <typename SrcT>
void minFilterEachPixel(const ImageView<SrcT> &aSrc, ImageView<SrcT> &aDst, const FilterArea &aFilterArea,
                        BorderType aBorderType, SrcT aConstant, const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            minFilterEachPixel(bc, aDst, aFilterArea);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            minFilterEachPixel(bc, aDst, aFilterArea);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            minFilterEachPixel(bc, aDst, aFilterArea);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            minFilterEachPixel(bc, aDst, aFilterArea);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            minFilterEachPixel(bc, aDst, aFilterArea);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            minFilterEachPixel(bc, aDst, aFilterArea);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for minFilter.");
            break;
    }
}

template <typename BorderControlT, typename DstT>
void maxFilterEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDst, const FilterArea &aFilterArea)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        DstT res       = numeric_limits<remove_vector_t<DstT>>::min();
        DstT &pixelOut = pixelIterator.Value();

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            res.Max(aSrcWithBC(i, j));
        }

        // restore alpha channel value:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            res.w = pixelOut.w;
        }

        pixelOut = res;
    }
}

template <typename SrcT>
void maxFilterEachPixel(const ImageView<SrcT> &aSrc, ImageView<SrcT> &aDst, const FilterArea &aFilterArea,
                        BorderType aBorderType, SrcT aConstant, const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            maxFilterEachPixel(bc, aDst, aFilterArea);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            maxFilterEachPixel(bc, aDst, aFilterArea);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            maxFilterEachPixel(bc, aDst, aFilterArea);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            maxFilterEachPixel(bc, aDst, aFilterArea);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            maxFilterEachPixel(bc, aDst, aFilterArea);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            maxFilterEachPixel(bc, aDst, aFilterArea);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for minFilter.");
            break;
    }
}

template <typename BorderControlT, typename DstT>
void wienerFilterEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDst, const FilterArea &aFilterArea,
                           const filter_compute_type_for_t<DstT> &aNoise)
{
    using ComputeT = same_vector_size_different_type_t<DstT, remove_vector_t<filter_compute_type_for_t<DstT>>>;
    remove_vector_t<filter_compute_type_for_t<DstT>> maskSizeInv =
        static_cast<remove_vector_t<filter_compute_type_for_t<DstT>>>(1) /
        static_cast<remove_vector_t<filter_compute_type_for_t<DstT>>>(aFilterArea.Size.TotalSize());

    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        ComputeT sum(0);
        ComputeT sumSqr(0);
        DstT &pixelOut    = pixelIterator.Value();
        ComputeT pixelSrc = ComputeT(aSrcWithBC(pixelX, pixelY));

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            ComputeT pixel = ComputeT(aSrcWithBC(i, j));
            sum += pixel;
            sumSqr += pixel * pixel;
        }

        sum *= maskSizeInv;                        // sum --> mean
        sumSqr = sumSqr * maskSizeInv - sum * sum; // sumSqr --> variance (with bias as no N-1)

        // wiener filter:
        sum = sum + ComputeT::Max(sumSqr - aNoise, ComputeT(0)) / ComputeT::Max(sumSqr, aNoise) * (pixelSrc - sum);

        if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
        {
            sum.RoundNearest();
        }

        DstT res = DstT(sum);

        // restore alpha channel value:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            res.w = pixelOut.w;
        }

        pixelOut = res;
    }
}

template <typename SrcT>
void wienerFilterEachPixel(const ImageView<SrcT> &aSrc, ImageView<SrcT> &aDst, const FilterArea &aFilterArea,
                           const filter_compute_type_for_t<SrcT> &aNoise, BorderType aBorderType, SrcT aConstant,
                           const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            wienerFilterEachPixel(bc, aDst, aFilterArea, aNoise);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            wienerFilterEachPixel(bc, aDst, aFilterArea, aNoise);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            wienerFilterEachPixel(bc, aDst, aFilterArea, aNoise);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            wienerFilterEachPixel(bc, aDst, aFilterArea, aNoise);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            wienerFilterEachPixel(bc, aDst, aFilterArea, aNoise);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            wienerFilterEachPixel(bc, aDst, aFilterArea, aNoise);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for filter.");
            break;
    }
}

template <typename BorderControlT, typename DstT>
void thresholdAdaptiveBoxFilterEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDst,
                                         const FilterArea &aFilterArea, const filter_compute_type_for_t<DstT> &aDelta,
                                         const DstT &aValGT, const DstT &aValLE)
{
    using ComputeT = same_vector_size_different_type_t<DstT, remove_vector_t<filter_compute_type_for_t<DstT>>>;
    remove_vector_t<filter_compute_type_for_t<DstT>> maskSizeInv =
        static_cast<remove_vector_t<filter_compute_type_for_t<DstT>>>(1) /
        static_cast<remove_vector_t<filter_compute_type_for_t<DstT>>>(aFilterArea.Size.TotalSize());

    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        ComputeT sum(0);
        DstT &pixelOut    = pixelIterator.Value();
        ComputeT pixelSrc = ComputeT(aSrcWithBC(pixelX, pixelY));

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            ComputeT pixel = ComputeT(aSrcWithBC(i, j));
            sum += pixel;
        }

        sum *= maskSizeInv; // sum --> mean
        sum -= aDelta;

        DstT res = aValLE;
        if (pixelSrc > sum) // for all channels
        {
            res = aValGT;
        }

        // restore alpha channel value:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            res.w = pixelOut.w;
        }

        pixelOut = res;
    }
}

template <typename SrcT>
void thresholdAdaptiveBoxFilterEachPixel(const ImageView<SrcT> &aSrc, ImageView<SrcT> &aDst,
                                         const FilterArea &aFilterArea, const filter_compute_type_for_t<SrcT> &aDelta,
                                         const SrcT &aValGT, const SrcT &aValLE, BorderType aBorderType, SrcT aConstant,
                                         const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            thresholdAdaptiveBoxFilterEachPixel(bc, aDst, aFilterArea, aDelta, aValGT, aValLE);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            thresholdAdaptiveBoxFilterEachPixel(bc, aDst, aFilterArea, aDelta, aValGT, aValLE);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            thresholdAdaptiveBoxFilterEachPixel(bc, aDst, aFilterArea, aDelta, aValGT, aValLE);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            thresholdAdaptiveBoxFilterEachPixel(bc, aDst, aFilterArea, aDelta, aValGT, aValLE);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            thresholdAdaptiveBoxFilterEachPixel(bc, aDst, aFilterArea, aDelta, aValGT, aValLE);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            thresholdAdaptiveBoxFilterEachPixel(bc, aDst, aFilterArea, aDelta, aValGT, aValLE);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for filter.");
            break;
    }
}

template <typename ComputeT>
float getWeightBilateral(const ComputeT &aPixel00, const ComputeT &aPixel, float aValSquareSigma, Norm aNorm)
{
    ComputeT diff = aPixel - aPixel00;
    float dist    = 0;
    if constexpr (vector_active_size_v<ComputeT> > 1)
    {
        if (aNorm == Norm::L2)
        {
            diff.Sqr();
        }
        else
        {
            diff.Abs();
        }
        dist = diff.x;
        dist += diff.y;
        if constexpr (vector_active_size_v<ComputeT> > 2)
        {
            dist += diff.z;
        }
        if constexpr (vector_active_size_v<ComputeT> > 3)
        {
            dist += diff.w;
        }
    }
    else
    {
        dist = diff.x;
    }

    if (aNorm == Norm::L1)
    {
        dist *= dist;
    }

    return std::exp(-dist / (2.0f * aValSquareSigma));
}

template <typename BorderControlT, typename ComputeT, typename DstT, typename FilterT>
void bilateralFilterEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDst, const FilterT *aPreComputedFilter,
                              float aValSquareSigma, const FilterArea &aFilterArea, Norm aNorm)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        ComputeT temp(0);
        DstT &pixelOut   = pixelIterator.Value();
        ComputeT pixel00 = ComputeT(aSrcWithBC(pixelX, pixelY));
        float sumWeights = 0;

        for (const auto &f : aFilterArea.Size)
        {
            const int i             = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j             = pixelY + f.Pixel.y - aFilterArea.Center.y;
            const ComputeT srcPixel = ComputeT(aSrcWithBC(i, j));

            const float wColor = getWeightBilateral(pixel00, srcPixel, aValSquareSigma, aNorm);
            const float wDist  = aPreComputedFilter[f.Pixel.y * aFilterArea.Size.x + f.Pixel.x];
            const float weight = wDist * wColor;
            sumWeights += weight;
            temp += srcPixel * weight;
        }
        temp /= sumWeights;

        if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
        {
            temp.RoundNearest();
        }

        DstT res = DstT(temp);

        // restore alpha channel value:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            res.w = pixelOut.w;
        }

        pixelOut = res;
    }
}

template <typename SrcT, typename ComputeT, typename DstT, typename FilterT>
void bilateralFilterEachPixel(const ImageView<SrcT> &aSrc, ImageView<DstT> &aDst, const FilterT *aPreComputedFilter,
                              float aValSquareSigma, const FilterArea &aFilterArea, Norm aNorm, BorderType aBorderType,
                              SrcT aConstant, const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            bilateralFilterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aDst, aPreComputedFilter, aValSquareSigma,
                                                                      aFilterArea, aNorm);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            bilateralFilterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aDst, aPreComputedFilter, aValSquareSigma,
                                                                      aFilterArea, aNorm);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            bilateralFilterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aDst, aPreComputedFilter, aValSquareSigma,
                                                                      aFilterArea, aNorm);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            bilateralFilterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aDst, aPreComputedFilter, aValSquareSigma,
                                                                      aFilterArea, aNorm);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            bilateralFilterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aDst, aPreComputedFilter, aValSquareSigma,
                                                                      aFilterArea, aNorm);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            bilateralFilterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aDst, aPreComputedFilter, aValSquareSigma,
                                                                      aFilterArea, aNorm);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for filter.");
            break;
    }
}

template <typename BorderControlT, typename ComputeT, typename DstT, typename FilterT>
void gradientVectorEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDstX, ImageView<DstT> &aDstY,
                             ImageView<DstT> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                             ImageView<Pixel32fC4> &aDstCovariance, const FilterT *aFilterX, const FilterT *aFilterY,
                             const FilterArea &aFilterArea, Norm aNorm)
{
    Size2D outSize = aDstX.SizeRoi();
    if (outSize.TotalSize() == 0)
    {
        outSize = aDstY.SizeRoi();
    }
    if (outSize.TotalSize() == 0)
    {
        outSize = aDstMag.SizeRoi();
    }
    if (outSize.TotalSize() == 0)
    {
        outSize = aDstAngle.SizeRoi();
    }
    if (outSize.TotalSize() == 0)
    {
        outSize = aDstCovariance.SizeRoi();
    }

    for (auto &pixelIterator : outSize)
    {
        int pixelX = pixelIterator.Pixel.x;
        int pixelY = pixelIterator.Pixel.y;

        ComputeT gradientX = {0};
        ComputeT gradientY = {0};

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            gradientX += ComputeT(aSrcWithBC(i, j)) * aFilterX[f.Pixel.y * aFilterArea.Size.x + f.Pixel.x];
            gradientY += ComputeT(aSrcWithBC(i, j)) * aFilterY[f.Pixel.y * aFilterArea.Size.x + f.Pixel.x];
        }

        // for multi channel, find channel with largest L2 gradient and store it in first channel:
        if constexpr (vector_active_size_v<ComputeT> > 1)
        {
            remove_vector_t<ComputeT> maxMagSqr = gradientX.x * gradientX.x + gradientY.x * gradientY.x;

            for (int c = 1; c < vector_active_size_v<ComputeT>; c++)
            {
                remove_vector_t<ComputeT> magSqr =
                    gradientX[Channel(c)] * gradientX[Channel(c)] + gradientY[Channel(c)] * gradientY[Channel(c)];

                if (magSqr > maxMagSqr)
                {
                    maxMagSqr   = magSqr;
                    gradientX.x = gradientX[Channel(c)];
                    gradientY.x = gradientY[Channel(c)];
                }
            }
        }

        if (aDstX.Pointer() != nullptr)
        {
            Vector1<remove_vector_t<ComputeT>> temp = gradientX.x;
            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
            {
                temp.RoundNearest();
            }

            aDstX(pixelX, pixelY) = DstT(temp);
        }
        if (aDstY.Pointer() != nullptr)
        {
            Vector1<remove_vector_t<ComputeT>> temp = gradientY.x;
            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
            {
                temp.RoundNearest();
            }

            aDstY(pixelX, pixelY) = DstT(temp);
        }
        if (aDstMag.Pointer() != nullptr)
        {
            Vector1<remove_vector_t<ComputeT>> temp;

            switch (aNorm)
            {
                case Norm::Inf:
                    temp = std::max(gradientX.x, gradientY.x);
                    break;
                case Norm::L1:
                    temp = std::abs(gradientX.x) + std::abs(gradientY.x);
                    break;
                case Norm::L2:
                    temp = std::sqrt(gradientX.x * gradientX.x + gradientY.x * gradientY.x);
                    break;
                default:
                    // well, that shouldn't happen...
                    temp = 0;
                    break;
            }
            if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
            {
                temp.RoundNearest();
            }

            aDstMag(pixelX, pixelY) = DstT(temp);
        }
        if (aDstAngle.Pointer() != nullptr)
        {
            aDstAngle(pixelX, pixelY) = std::atan2(static_cast<float>(gradientY.x), static_cast<float>(gradientX.x));
        }
        if (aDstCovariance.Pointer() != nullptr)
        {
            Pixel32fC4 res;
            res.x                          = static_cast<float>(gradientX.x) * static_cast<float>(gradientX.x);
            res.y                          = static_cast<float>(gradientY.x) * static_cast<float>(gradientY.x);
            res.z                          = static_cast<float>(gradientX.x) * static_cast<float>(gradientY.x);
            res.w                          = res.z;
            aDstCovariance(pixelX, pixelY) = res;
        }
    }
}

template <typename SrcT, typename ComputeT, typename DstT, typename FilterT>
void gradientVectorEachPixel(const ImageView<SrcT> &aSrc, ImageView<DstT> &aDstX, ImageView<DstT> &aDstY,
                             ImageView<DstT> &aDstMag, ImageView<Pixel32fC1> &aDstAngle,
                             ImageView<Pixel32fC4> &aDstCovariance, const FilterT *aFilterX, const FilterT *aFilterY,
                             const FilterArea &aFilterArea, Norm aNorm, BorderType aBorderType, SrcT aConstant,
                             const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            gradientVectorEachPixel<BCType, ComputeT, DstT, FilterT>(
                bc, aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aFilterX, aFilterY, aFilterArea, aNorm);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            gradientVectorEachPixel<BCType, ComputeT, DstT, FilterT>(
                bc, aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aFilterX, aFilterY, aFilterArea, aNorm);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            gradientVectorEachPixel<BCType, ComputeT, DstT, FilterT>(
                bc, aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aFilterX, aFilterY, aFilterArea, aNorm);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            gradientVectorEachPixel<BCType, ComputeT, DstT, FilterT>(
                bc, aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aFilterX, aFilterY, aFilterArea, aNorm);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            gradientVectorEachPixel<BCType, ComputeT, DstT, FilterT>(
                bc, aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aFilterX, aFilterY, aFilterArea, aNorm);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            gradientVectorEachPixel<BCType, ComputeT, DstT, FilterT>(
                bc, aDstX, aDstY, aDstMag, aDstAngle, aDstCovariance, aFilterX, aFilterY, aFilterArea, aNorm);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for filter.");
            break;
    }
}

template <typename BorderControlT, typename ComputeT, typename DstT, typename FilterT>
void unsharpFilterEachPixel(BorderControlT aSrcWithBC, const ImageView<DstT> &aSrcOrig, ImageView<DstT> &aDst,
                            const FilterT *aFilter, FilterT aWeight, FilterT aThreshold, const FilterArea &aFilterArea)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        ComputeT temp(0);
        DstT &pixelOut = pixelIterator.Value();

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            temp += ComputeT(aSrcWithBC(i, j)) * aFilter[f.Pixel.y * aFilterArea.Size.x + f.Pixel.x];
        }

        ComputeT origPixel = ComputeT(aSrcOrig(pixelX, pixelY));
        ComputeT highPass  = origPixel - temp;
        ComputeT activator;
        activator.x = std::abs(highPass.x) >= aThreshold ? 1.0f : 0.0f;
        if constexpr (vector_active_size_v<ComputeT> > 1)
        {
            activator.y = abs(highPass.y) >= aThreshold ? 1.0f : 0.0f;
        }
        if constexpr (vector_active_size_v<ComputeT> > 2)
        {
            activator.z = abs(highPass.z) >= aThreshold ? 1.0f : 0.0f;
        }
        if constexpr (vector_active_size_v<ComputeT> > 3)
        {
            activator.w = abs(highPass.w) >= aThreshold ? 1.0f : 0.0f;
        }
        temp = origPixel + aWeight * highPass * activator;

        if constexpr (RealOrComplexFloatingVector<ComputeT> && RealOrComplexIntVector<DstT>)
        {
            temp.RoundNearest();
        }

        DstT res = DstT(temp);

        // restore alpha channel value:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            res.w = pixelOut.w;
        }

        pixelOut = res;
    }
}

template <typename SrcT, typename ComputeT, typename DstT, typename FilterT>
void unsharpFilterEachPixel(const ImageView<SrcT> &aSrc, const ImageView<DstT> &aSrcOrig, ImageView<DstT> &aDst,
                            const FilterT *aFilter, FilterT aWeight, FilterT aThreshold, const FilterArea &aFilterArea,
                            BorderType aBorderType, SrcT aConstant, const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            unsharpFilterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aSrcOrig, aDst, aFilter, aWeight, aThreshold,
                                                                    aFilterArea);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            unsharpFilterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aSrcOrig, aDst, aFilter, aWeight, aThreshold,
                                                                    aFilterArea);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            unsharpFilterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aSrcOrig, aDst, aFilter, aWeight, aThreshold,
                                                                    aFilterArea);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            unsharpFilterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aSrcOrig, aDst, aFilter, aWeight, aThreshold,
                                                                    aFilterArea);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            unsharpFilterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aSrcOrig, aDst, aFilter, aWeight, aThreshold,
                                                                    aFilterArea);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            unsharpFilterEachPixel<BCType, ComputeT, DstT, FilterT>(bc, aSrcOrig, aDst, aFilter, aWeight, aThreshold,
                                                                    aFilterArea);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for filter.");
            break;
    }
}

template <class SrcT, class DstT, typename BorderControlT>
DstT cannyEdgeMaxSupression(int pixelX, int pixelY, BorderControlT &aSrcWithBC, const Pixel32fC1 *aSrcAngle,
                            size_t aPitchSrcAngle, SrcT aLowThreshold, SrcT aHighThreshold)
{
    constexpr DstT Nothing = 0;
    constexpr DstT Weak    = 1;
    constexpr DstT Strong  = 255;

    const SrcT pixel = aSrcWithBC(pixelX, pixelY);
    if (pixel < aLowThreshold)
    {
        return Nothing;
    }

    Pixel32fC1 angle = *gotoPtr(aSrcAngle, aPitchSrcAngle, pixelX, pixelY);

    // get quantized direction from angle, angles from atan2-function are given in range -pi..pi
    // map this range to 0..3, where 0 = horizontal, 1 = 45deg diagonal, 2 = vertical, 3 = -45deg = 135deg diagonal
    angle.x = round((angle.x / std::numbers::pi_v<float> * 180.0f + 180.0f) / 45.0f);
    int dir = static_cast<int>(angle.x) % 4; // the modulo maps the negative / opposite direction to the positive one

    SrcT pixelMinus;
    SrcT pixelPlus;
    SrcT compare;
    switch (dir)
    {
        case 0:
            // gradient horizontal direction -> check in X direction
            pixelMinus = aSrcWithBC(pixelX - 1, pixelY);
            pixelPlus  = aSrcWithBC(pixelX + 1, pixelY);
            break;
        case 1:
            // gradient +45deg -> check in +45deg direction
            pixelMinus = aSrcWithBC(pixelX - 1, pixelY + 1);
            pixelPlus  = aSrcWithBC(pixelX + 1, pixelY - 1);
            break;
        case 2:
            // gradient vertical direction -> check in Y direction
            pixelMinus = aSrcWithBC(pixelX, pixelY - 1);
            pixelPlus  = aSrcWithBC(pixelX, pixelY + 1);
            break;
        case 3:
            // gradient -45deg=135deg -> check in -45deg direction
            pixelMinus = aSrcWithBC(pixelX - 1, pixelY - 1);
            pixelPlus  = aSrcWithBC(pixelX + 1, pixelY + 1);
            break;
        default:
            break;
    }

    compare = SrcT::Max(pixelMinus, SrcT::Max(pixelPlus, pixel));
    if (pixel == compare)
    {
        return pixel >= aHighThreshold ? Strong : Weak;
    }
    return Nothing;
}

template <class DstT, typename BorderControlT>
DstT cannyEdgeHysteresis(int pixelX, int pixelY, BorderControlT &aSrcWithBC, const Pixel32fC1 *aSrcAngle,
                         size_t aPitchSrcAngle)
{
    constexpr DstT Nothing = 0;
    constexpr DstT Weak    = 1;
    constexpr DstT Strong  = 255;

    DstT pixel = aSrcWithBC(pixelX, pixelY);
    if (pixel == Nothing)
    {
        return Nothing;
    }
    if (pixel == Strong)
    {
        return Strong;
    }

    Pixel32fC1 angle = *gotoPtr(aSrcAngle, aPitchSrcAngle, pixelX, pixelY);

    // get quantized direction from angle, angles from atan2-function are given in range -pi..pi
    // map this range to 0..3, where 0 = horizontal, 1 = 45deg diagonal, 2 = vertical, 3 = -45deg = 135deg diagonal
    angle.x = round((angle.x / std::numbers::pi_v<float> * 180.0f + 180.0f) / 45.0f);
    int dir = static_cast<int>(angle.x) % 4; // the modulo maps the negative / opposite direction to the positive one

    int pixelMinus = 0;
    int pixelPlus  = 0;
    switch (dir)
    {
        case 0:
            // gradient horizontal direction -> check in Y direction
            pixelMinus = aSrcWithBC(pixelX, pixelY - 1).x;
            pixelPlus  = aSrcWithBC(pixelX, pixelY + 1).x;
            break;
        case 1:
            // gradient +45deg -> check in -45deg direction
            pixelMinus = aSrcWithBC(pixelX - 1, pixelY - 1).x;
            pixelPlus  = aSrcWithBC(pixelX + 1, pixelY + 1).x;
            break;
        case 2:
            // gradient vertical direction -> check in X direction
            pixelMinus = aSrcWithBC(pixelX - 1, pixelY).x;
            pixelPlus  = aSrcWithBC(pixelX + 1, pixelY).x;
            break;
        case 3:
            // gradient -45deg=135deg -> check in +45deg direction
            pixelMinus = aSrcWithBC(pixelX - 1, pixelY + 1).x;
            pixelPlus  = aSrcWithBC(pixelX + 1, pixelY - 1).x;
            break;
        default:
            break;
    }

    if (pixelMinus + pixelPlus >= Strong.x)
    {
        // at least one of the neighbor pixels is at least "Strong"
        return Strong;
    }
    return Nothing;
}

template <typename BorderControlT, typename SrcT, typename DstT>
void cannyEdgeMaxSupressionEachPixel(BorderControlT aSrcWithBC, const Pixel32fC1 *aSrcAngle, size_t aPitchSrcAngle,
                                     ImageView<DstT> &aDst, SrcT aLowThreshold, SrcT aHighThreshold)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        DstT &pixelOut = pixelIterator.Value();

        pixelOut = cannyEdgeMaxSupression<SrcT, DstT, BorderControlT>(pixelX, pixelY, aSrcWithBC, aSrcAngle,
                                                                      aPitchSrcAngle, aLowThreshold, aHighThreshold);
    }
}

template <typename SrcT, typename DstT>
void cannyEdgeMaxSupressionEachPixel(const ImageView<SrcT> &aSrc, const Pixel32fC1 *aSrcAngle, size_t aPitchSrcAngle,
                                     ImageView<DstT> &aDst, const Roi &aAllowedReadRoi, SrcT aLowThreshold,
                                     SrcT aHighThreshold)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
    const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

    cannyEdgeMaxSupressionEachPixel<BCType, SrcT, DstT>(bc, aSrcAngle, aPitchSrcAngle, aDst, aLowThreshold,
                                                        aHighThreshold);
}

template <typename BorderControlT, typename DstT>
void cannyEdgeHysteresisEachPixel(BorderControlT aSrcWithBC, const Pixel32fC1 *aSrcAngle, size_t aPitchSrcAngle,
                                  ImageView<DstT> &aDst)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        DstT &pixelOut = pixelIterator.Value();

        pixelOut = cannyEdgeHysteresis<DstT, BorderControlT>(pixelX, pixelY, aSrcWithBC, aSrcAngle, aPitchSrcAngle);
    }
}

template <typename SrcT, typename DstT>
void cannyEdgeHysteresisEachPixel(const ImageView<SrcT> &aSrc, const Pixel32fC1 *aSrcAngle, size_t aPitchSrcAngle,
                                  ImageView<DstT> &aDst, const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
    const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

    cannyEdgeHysteresisEachPixel<BCType, DstT>(bc, aSrcAngle, aPitchSrcAngle, aDst);
}

template <typename BorderControlT, typename DstT, typename SrcT>
void crossCorrelationEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDst, const ImageView<SrcT> &aTemplate,
                               const FilterArea &aFilterArea)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        DstT res(0);
        DstT &pixelOut = pixelIterator.Value();

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            res += DstT(aSrcWithBC(i, j)) * DstT(aTemplate(f.Pixel.x, f.Pixel.y));
        }

        pixelOut = res;
    }
}

template <typename SrcT, typename DstT>
void crossCorrelationEachPixel(const ImageView<SrcT> &aSrc, ImageView<DstT> &aDst, const ImageView<SrcT> &aTemplate,
                               const FilterArea &aFilterArea, BorderType aBorderType, SrcT aConstant,
                               const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            crossCorrelationEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for filter.");
            break;
    }
}

template <typename BorderControlT, typename DstT, typename SrcT>
void crossCorrelationNormalizedEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDst,
                                         const ImageView<SrcT> &aTemplate, const FilterArea &aFilterArea)
{
    DstT sumSqrTpl(0);

    for (auto &pixelIterator : aTemplate)
    {
        DstT pixel = DstT(pixelIterator.Value());
        pixel.Sqr();
        sumSqrTpl += pixel;
    }

    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        DstT res(0);
        DstT imgSqr(0);
        DstT &pixelOut = pixelIterator.Value();

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            res += DstT(aSrcWithBC(i, j)) * DstT(aTemplate(f.Pixel.x, f.Pixel.y));
            imgSqr += DstT(aSrcWithBC(i, j)) * DstT(aSrcWithBC(i, j));
        }

        DstT norm = imgSqr * sumSqrTpl;
        norm.Sqrt();
        res /= norm;
        pixelOut = res;
    }
}

template <typename SrcT, typename DstT>
void crossCorrelationNormalizedEachPixel(const ImageView<SrcT> &aSrc, ImageView<DstT> &aDst,
                                         const ImageView<SrcT> &aTemplate, const FilterArea &aFilterArea,
                                         BorderType aBorderType, SrcT aConstant, const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationNormalizedEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            crossCorrelationNormalizedEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationNormalizedEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationNormalizedEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationNormalizedEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationNormalizedEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for filter.");
            break;
    }
}

template <typename BorderControlT, typename DstT, typename SrcT>
void crossCorrelationCoefficientEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDst,
                                          const ImageView<SrcT> &aTemplate, const FilterArea &aFilterArea)
{
    DstT meanTpl(0);
    DstT sumSqrTpl(0);

    for (auto &pixelIterator : aTemplate)
    {
        DstT pixel = DstT(pixelIterator.Value());
        meanTpl += pixel;
    }
    meanTpl /= DstT(to_float(aFilterArea.Size.TotalSize()));

    for (auto &pixelIterator : aTemplate)
    {
        DstT pixel = DstT(pixelIterator.Value()) - meanTpl;
        pixel.Sqr();
        sumSqrTpl += pixel;
    }

    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        DstT res(0);
        DstT imgMean(0);
        DstT imgSqr(0);
        DstT &pixelOut = pixelIterator.Value();

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            imgMean += DstT(aSrcWithBC(i, j));
        }
        imgMean /= DstT(to_float(aFilterArea.Size.TotalSize()));

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            DstT pixel = DstT(aSrcWithBC(i, j)) - imgMean;

            res += pixel * (DstT(aTemplate(f.Pixel.x, f.Pixel.y)) - meanTpl);
            imgSqr += pixel * pixel;
        }

        DstT norm = imgSqr * sumSqrTpl;
        if (norm <= DstT(0))
        {
            res = DstT(0);
        }
        else
        {
            norm.Sqrt();
            res /= norm;
        }
        pixelOut = res;
    }
}

template <typename SrcT, typename DstT>
void crossCorrelationCoefficientEachPixel(const ImageView<SrcT> &aSrc, ImageView<DstT> &aDst,
                                          const ImageView<SrcT> &aTemplate, const FilterArea &aFilterArea,
                                          BorderType aBorderType, SrcT aConstant, const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationCoefficientEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            crossCorrelationCoefficientEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationCoefficientEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationCoefficientEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationCoefficientEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            crossCorrelationCoefficientEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for filter.");
            break;
    }
}

template <typename BorderControlT, typename DstT, typename SrcT>
void squareDistanceNormalizedEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDst,
                                       const ImageView<SrcT> &aTemplate, const FilterArea &aFilterArea)
{
    DstT sumSqrTpl(0);

    for (auto &pixelIterator : aTemplate)
    {
        DstT pixel = DstT(pixelIterator.Value());
        pixel.Sqr();
        sumSqrTpl += pixel;
    }

    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        DstT res(0);
        DstT imgSqr(0);
        DstT &pixelOut = pixelIterator.Value();

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            DstT diff = DstT(aSrcWithBC(i, j)) - DstT(aTemplate(f.Pixel.x, f.Pixel.y));
            res += diff * diff;
            imgSqr += DstT(aSrcWithBC(i, j)) * DstT(aSrcWithBC(i, j));
        }

        DstT norm = imgSqr * sumSqrTpl;
        norm.Sqrt();
        res /= norm;
        pixelOut = res;
    }
}

template <typename SrcT, typename DstT>
void squareDistanceNormalizedEachPixel(const ImageView<SrcT> &aSrc, ImageView<DstT> &aDst,
                                       const ImageView<SrcT> &aTemplate, const FilterArea &aFilterArea,
                                       BorderType aBorderType, SrcT aConstant, const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            squareDistanceNormalizedEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            squareDistanceNormalizedEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            squareDistanceNormalizedEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            squareDistanceNormalizedEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            squareDistanceNormalizedEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            squareDistanceNormalizedEachPixel<BCType, DstT, SrcT>(bc, aDst, aTemplate, aFilterArea);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for filter.");
            break;
    }
}

template <typename BorderControlT, typename DstT, typename FilterT, typename morphOperation, typename postOp>
void moprhologyEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDst, const FilterT *aMask,
                         const FilterArea &aFilterArea, morphOperation aMorph, postOp aPostOp)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        DstT res(morphOperation::InitValue);
        DstT &pixelOut = pixelIterator.Value();

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            aMorph(aMask[f.Pixel.y * aFilterArea.Size.x + f.Pixel.x], aSrcWithBC(i, j), res);
        }

        aPostOp(pixelX, pixelY, res);

        // restore alpha channel values:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            res.w = pixelOut.w;
        }
        pixelOut = res;
    }
}

template <typename SrcT, typename FilterT, typename morphOperation, typename postOp>
void moprhologyEachPixel(const ImageView<SrcT> &aSrc, ImageView<SrcT> &aDst, const FilterT *aMask,
                         const FilterArea &aFilterArea, BorderType aBorderType, SrcT aConstant,
                         const Roi &aAllowedReadRoi, morphOperation aMorph, postOp aPostOp)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            moprhologyEachPixel<BCType, SrcT>(bc, aDst, aMask, aFilterArea, aMorph, aPostOp);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            moprhologyEachPixel<BCType, SrcT>(bc, aDst, aMask, aFilterArea, aMorph, aPostOp);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            moprhologyEachPixel<BCType, SrcT>(bc, aDst, aMask, aFilterArea, aMorph, aPostOp);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            moprhologyEachPixel<BCType, SrcT>(bc, aDst, aMask, aFilterArea, aMorph, aPostOp);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            moprhologyEachPixel<BCType, SrcT>(bc, aDst, aMask, aFilterArea, aMorph, aPostOp);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            moprhologyEachPixel<BCType, SrcT>(bc, aDst, aMask, aFilterArea, aMorph, aPostOp);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for filter.");
            break;
    }
}

template <typename BorderControlT, typename DstT>
void moprhologyGradientEachPixel(BorderControlT aSrcWithBC, ImageView<DstT> &aDst, const Pixel8uC1 *aMask,
                                 const FilterArea &aFilterArea)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        DstT resultDilate = DstT(numeric_limits<remove_vector_t<DstT>>::min());
        DstT resultErode  = DstT(numeric_limits<remove_vector_t<DstT>>::max());

        DstT &pixelOut = pixelIterator.Value();

        for (const auto &f : aFilterArea.Size)
        {

            if (aMask[f.Pixel.y * aFilterArea.Size.x + f.Pixel.x] > 0)
            {
                const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
                const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

                resultDilate.Max(aSrcWithBC(i, j));
                resultErode.Min(aSrcWithBC(i, j));
            }
        }

        morph_compute_type_t<DstT> erosion  = morph_compute_type_t<DstT>(resultErode);
        morph_compute_type_t<DstT> dilation = morph_compute_type_t<DstT>(resultDilate);

        DstT res = DstT(dilation - erosion);

        // restore alpha channel values:
        if constexpr (has_alpha_channel_v<DstT>)
        {
            res.w = pixelOut.w;
        }
        pixelOut = res;
    }
}

template <typename SrcT>
void moprhologyGradientEachPixel(const ImageView<SrcT> &aSrc, ImageView<SrcT> &aDst, const Pixel8uC1 *aMask,
                                 const FilterArea &aFilterArea, BorderType aBorderType, SrcT aConstant,
                                 const Roi &aAllowedReadRoi)
{

    const Vector2<int> roiOffset = aSrc.ROI().FirstPixel() - aAllowedReadRoi.FirstPixel();
    const SrcT *allowedPtr = gotoPtr(aSrc.Pointer(), aSrc.Pitch(), aAllowedReadRoi.FirstX(), aAllowedReadRoi.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::None:
        {
            using BCType = BorderControl<SrcT, BorderType::None, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            moprhologyGradientEachPixel<BCType, SrcT>(bc, aDst, aMask, aFilterArea);
        }
        break;
        case opp::BorderType::Constant:
        {
            using BCType = BorderControl<SrcT, BorderType::Constant, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset, aConstant);

            moprhologyGradientEachPixel<BCType, SrcT>(bc, aDst, aMask, aFilterArea);
        }
        break;
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            moprhologyGradientEachPixel<BCType, SrcT>(bc, aDst, aMask, aFilterArea);
        }
        break;
        case opp::BorderType::Mirror:
        {
            using BCType = BorderControl<SrcT, BorderType::Mirror, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            moprhologyGradientEachPixel<BCType, SrcT>(bc, aDst, aMask, aFilterArea);
        }
        break;
        case opp::BorderType::MirrorReplicate:
        {
            using BCType = BorderControl<SrcT, BorderType::MirrorReplicate, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            moprhologyGradientEachPixel<BCType, SrcT>(bc, aDst, aMask, aFilterArea);
        }
        break;
        case opp::BorderType::Wrap:
        {
            using BCType = BorderControl<SrcT, BorderType::Wrap, false, false, false, false>;
            const BCType bc(allowedPtr, aSrc.Pitch(), aAllowedReadRoi.Size(), roiOffset);

            moprhologyGradientEachPixel<BCType, SrcT>(bc, aDst, aMask, aFilterArea);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for filter.");
            break;
    }
}

template <typename BorderControlT, typename ComputeT, typename DstT, typename FilterT, typename postOP>
void ssimEachPixel(BorderControlT aSrc1WithBC, BorderControlT aSrc2WithBC, ImageView<DstT> &aDst,
                   const FilterT *aFilter, const FilterArea &aFilterArea, postOP aPostOp)
{
    for (auto &pixelIterator : aDst)
    {
        int pixelX = pixelIterator.Pixel().x - aDst.ROI().x;
        int pixelY = pixelIterator.Pixel().y - aDst.ROI().y;

        ComputeT mean1(0);
        ComputeT mean2(0);
        ComputeT var1Sqr(0);
        ComputeT var2Sqr(0);
        ComputeT crossVarSqr(0);
        DstT &pixelOut = pixelIterator.Value();

        for (const auto &f : aFilterArea.Size)
        {
            const int i = pixelX + f.Pixel.x - aFilterArea.Center.x;
            const int j = pixelY + f.Pixel.y - aFilterArea.Center.y;

            ComputeT pixelSrc1 = ComputeT(aSrc1WithBC(i, j));
            ComputeT pixelSrc2 = ComputeT(aSrc2WithBC(i, j));

            mean1 += pixelSrc1 * aFilter[f.Pixel.y * aFilterArea.Size.x + f.Pixel.x];
            mean2 += pixelSrc2 * aFilter[f.Pixel.y * aFilterArea.Size.x + f.Pixel.x];
            var1Sqr += pixelSrc1 * pixelSrc1 * aFilter[f.Pixel.y * aFilterArea.Size.x + f.Pixel.x];
            var2Sqr += pixelSrc2 * pixelSrc2 * aFilter[f.Pixel.y * aFilterArea.Size.x + f.Pixel.x];
            crossVarSqr += pixelSrc1 * pixelSrc2 * aFilter[f.Pixel.y * aFilterArea.Size.x + f.Pixel.x];
        }

        DstT res;
        aPostOp(mean1, var1Sqr, mean2, var2Sqr, crossVarSqr, res);

        pixelOut = res;
    }
}

template <typename SrcT, typename ComputeT, typename DstT, typename FilterT, typename postOP>
void ssimEachPixel(const ImageView<SrcT> &aSrc1, const ImageView<SrcT> &aSrc2, ImageView<DstT> &aDst,
                   const FilterT *aFilter, const FilterArea &aFilterArea, BorderType aBorderType,
                   const Roi &aAllowedReadRoi1, const Roi &aAllowedReadRoi2, postOP aPostOp)
{

    const Vector2<int> roiOffset1 = aSrc1.ROI().FirstPixel() - aAllowedReadRoi1.FirstPixel();
    const Vector2<int> roiOffset2 = aSrc2.ROI().FirstPixel() - aAllowedReadRoi2.FirstPixel();
    const SrcT *allowedPtr1 =
        gotoPtr(aSrc1.Pointer(), aSrc1.Pitch(), aAllowedReadRoi1.FirstX(), aAllowedReadRoi1.FirstY());
    const SrcT *allowedPtr2 =
        gotoPtr(aSrc2.Pointer(), aSrc2.Pitch(), aAllowedReadRoi2.FirstX(), aAllowedReadRoi2.FirstY());

    switch (aBorderType)
    {
        case opp::BorderType::Replicate:
        {
            using BCType = BorderControl<SrcT, BorderType::Replicate, false, false, false, false>;
            const BCType bc1(allowedPtr1, aSrc1.Pitch(), aAllowedReadRoi1.Size(), roiOffset1);
            const BCType bc2(allowedPtr2, aSrc2.Pitch(), aAllowedReadRoi2.Size(), roiOffset2);

            ssimEachPixel<BCType, ComputeT, DstT, FilterT>(bc1, bc2, aDst, aFilter, aFilterArea, aPostOp);
        }
        break;
        default:
            throw INVALIDARGUMENT(aBorderType, aBorderType << " is not a supported border type mode for ssim.");
            break;
    }
}

} // namespace opp::image::cpuSimple