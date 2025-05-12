#pragma once
#include "borderControl.h"
#include "imageFunctors.h"
#include "interpolator.h"
#include "transformer.h"
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
#include <common/opp_defs.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace opp::image
{
/// <summary>
/// Computes an output pixel from src image with geometric coordinate transformation
/// </summary>
template <size_t tupelSize, typename DstT, typename CoordTransformerT, bool checkIfInsideROI, typename InterpolatorT,
          typename TransformerT, RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero>
struct TransformerFunctor : public ImageFunctor<false>
{
    InterpolatorT Interpolator;
    TransformerT Transformer;
    Vector2<CoordTransformerT> SourceRoi;

    [[no_unique_address]] RoundFunctor<roundingMode, typename InterpolatorT::pixel_type> round;

#pragma region Constructors
    TransformerFunctor()
    {
    }

    TransformerFunctor(InterpolatorT aInterpolator, TransformerT aTransformer, const Size2D &aSourceRoi)
        : Interpolator(aInterpolator), Transformer(aTransformer),
          SourceRoi(Vector2<CoordTransformerT>(aSourceRoi) - static_cast<CoordTransformerT>(0.5))
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    /// <summary>
    /// Returns true if the value has been successfully set, false otherwise
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
    {
        // either int2, float2 or double2, depending on transformer type
        const auto coordTransformed = Transformer(aPixelX, aPixelY);

        if constexpr (checkIfInsideROI)
        {
            if (coordTransformed.x < static_cast<CoordTransformerT>(-0.5) || coordTransformed.x >= SourceRoi.x ||
                coordTransformed.y < static_cast<CoordTransformerT>(-0.5) || coordTransformed.y >= SourceRoi.y)
            {
                return false;
            }
        }

        typename InterpolatorT::pixel_type pixel =
            Interpolator(static_cast<InterpolatorT::coordinate_type>(coordTransformed.x),
                         static_cast<InterpolatorT::coordinate_type>(coordTransformed.y));

        if constexpr (RealOrComplexIntVector<DstT>)
        {
            round(pixel); // NOP for integer ComputeT
        }

        aDst = DstT(pixel);
        return true;
    }

#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
    {
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            // either int2, float2 or double2, depending on transformer type
            const auto coordTransformed = Transformer(aPixelX + static_cast<int>(i), aPixelY);

            if constexpr (checkIfInsideROI)
            {
                static_assert(!checkIfInsideROI, "abortIfOutsideRoi is not supported in combination with Tupels.");
            }

            auto pixel = Interpolator(static_cast<InterpolatorT::coordinate_type>(coordTransformed.x),
                                      static_cast<InterpolatorT::coordinate_type>(coordTransformed.y));

            if constexpr (RealOrComplexIntegral<DstT>)
            {
                round(pixel); // NOP for integer ComputeT
            }

            aDst.value[i] = DstT(pixel);
        }
    }
#pragma endregion
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
