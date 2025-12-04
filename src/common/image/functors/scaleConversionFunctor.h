#pragma once
#include "imageFunctors.h"
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace mpp::image
{
/// <summary>
/// Computes an output pixel from one src array -&gt; dst pixel by scaling the pixel value from input type value range
/// to ouput type value range using the equation:<para/> dstPixelValue = dstMinRangeValue + scaleFactor * (srcPixelValue
/// - srcMinRangeValue)<para/> whith scaleFactor = (dstMaxRangeValue - dstMinRangeValue) / (srcMaxRangeValue -
/// srcMinRangeValue).
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename ComputeT, typename DstT,
          RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero>
struct ScaleConversionFunctor : public ImageFunctor<false>
{
    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    scalefactor_t<ComputeT> ScaleFactor;
    scalefactor_t<ComputeT> SrcMin;
    scalefactor_t<ComputeT> DstMin;
    scalefactor_t<ComputeT> SrcRange;
    scalefactor_t<ComputeT> DstRange;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

#pragma region Constructors
    ScaleConversionFunctor()
    {
    }

    ScaleConversionFunctor(const SrcT *aSrc1, size_t aSrcPitch1, scalefactor_t<ComputeT> aScaleFactor,
                           scalefactor_t<ComputeT> aSrcMin, scalefactor_t<ComputeT> aDstMin)
        requires RealOrComplexFloatingVector<ComputeT>
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), ScaleFactor(aScaleFactor), SrcMin(aSrcMin), DstMin(aDstMin), SrcRange(0),
          DstRange(0)
    {
    }

    ScaleConversionFunctor(const SrcT *aSrc1, size_t aSrcPitch1, scalefactor_t<ComputeT> aSrcMin,
                           scalefactor_t<ComputeT> aDstMin, scalefactor_t<ComputeT> aSrcRange,
                           scalefactor_t<ComputeT> aDstRange)
        requires RealOrComplexIntVector<ComputeT>
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), ScaleFactor(0), SrcMin(aSrcMin), DstMin(aDstMin * aSrcRange),
          SrcRange(aSrcRange), DstRange(aDstRange)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
        requires RealOrComplexFloatingVector<ComputeT>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        ComputeT temp         = static_cast<ComputeT>(*pixelSrc1);
        temp                  = DstMin + ScaleFactor * (temp - SrcMin);

        if constexpr (RealOrComplexIntegral<pixel_basetype_t<DstT>>)
        {
            round(temp);
        }
        // DstT constructor will clamp temp to value range of DstT
        aDst = static_cast<DstT>(temp);
        return true;
    }

    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
        requires RealOrComplexIntVector<ComputeT>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        ComputeT temp         = static_cast<ComputeT>(*pixelSrc1);
        temp                  = DstMin + DstRange * (temp - SrcMin); // DstMin is scaled by SrcRange

        if constexpr (roundingMode == RoundingMode::NearestTiesToEven)
        {
            temp.DivScaleRoundNearest(SrcRange);
        }
        else if constexpr (roundingMode == RoundingMode::NearestTiesAwayFromZero)
        {
            temp.DivScaleRound(SrcRange);
        }
        else if constexpr (roundingMode == RoundingMode::TowardZero)
        {
            temp.DivScaleRoundZero(SrcRange);
        }
        else if constexpr (roundingMode == RoundingMode::TowardNegativeInfinity)
        {
            temp.DivScaleFloor(SrcRange);
        }
        else if constexpr (roundingMode == RoundingMode::TowardPositiveInfinity)
        {
            temp.DivScaleCeil(SrcRange);
        }
        else
        {
            temp /= SrcRange;
        }

        // DstT constructor will clamp temp to value range of DstT
        aDst = static_cast<DstT>(temp);
        return true;
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires RealOrComplexIntVector<ComputeT>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp = static_cast<ComputeT>(tupelSrc1.value[i]);
            temp          = DstMin + ScaleFactor * (temp - SrcMin);

            if constexpr (RealOrComplexIntegral<pixel_basetype_t<DstT>>)
            {
                round(temp);
            }
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires RealOrComplexFloatingVector<ComputeT>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp = static_cast<ComputeT>(tupelSrc1.value[i]);
            temp          = DstMin + ScaleFactor * (temp - SrcMin);

            if constexpr (RealOrComplexIntegral<pixel_basetype_t<DstT>>)
            {
                round(temp);
            }
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }
};
#pragma endregion
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
