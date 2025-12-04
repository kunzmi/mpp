#pragma once
#include "borderControl.h"
#include "imageFunctors.h"
#include "interpolator.h"
#include "transformer.h"
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
/// Computes an output pixel from src image with geometric coordinate transformation
/// </summary>
template <size_t tupelSize, typename DstT, typename CoordTransformerT, bool checkIfInsideROI, typename InterpolatorT,
          typename TransformerT, RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero>
struct TransformerFunctor
    : public ImageFunctor<InterpolatorT::border_control_type::border_type == mpp::BorderType::SmoothEdge>
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

        if constexpr (InterpolatorT::border_control_type::border_type == mpp::BorderType::SmoothEdge)
        {
            static_assert(checkIfInsideROI, "checkIfInsideROI must be enabled for BorderType::smoothEdges.");
        }

        if constexpr (checkIfInsideROI)
        {
            // Note: SourceRoi is integer size - 0.5
            if (coordTransformed.x < static_cast<CoordTransformerT>(-0.5) || coordTransformed.x >= SourceRoi.x ||
                coordTransformed.y < static_cast<CoordTransformerT>(-0.5) || coordTransformed.y >= SourceRoi.y)
            {
                // here we end up for all pixels that are outside the source ROI.
                if constexpr (InterpolatorT::border_control_type::border_type == mpp::BorderType::SmoothEdge)
                {
                    if (coordTransformed.x >= static_cast<CoordTransformerT>(-1.5) &&
                        coordTransformed.x < SourceRoi.x + static_cast<CoordTransformerT>(1) &&
                        coordTransformed.y >= static_cast<CoordTransformerT>(-1.5) &&
                        coordTransformed.y < SourceRoi.y + static_cast<CoordTransformerT>(1))
                    {
                        // here we end up for all pixels that are just next to the ROI, i.e. the one pixel large area
                        // around the ROI.
                        using CT = InterpolatorT::coordinate_type;
                        using PT = typename InterpolatorT::pixel_type;

                        PT pixel = Interpolator(static_cast<InterpolatorT::coordinate_type>(coordTransformed.x),
                                                static_cast<InterpolatorT::coordinate_type>(coordTransformed.y));

                        CT w = 1;
                        // 4 possible cases / edges (corners are weighted twice):
                        if (coordTransformed.x < static_cast<CoordTransformerT>(-0.5))
                        {
                            // left edge
                            w *= static_cast<CT>(1.5) + coordTransformed.x;
                        }
                        else if (coordTransformed.y < static_cast<CoordTransformerT>(-0.5))
                        {
                            // top edge
                            w *= static_cast<CT>(1.5) + coordTransformed.y;
                        }
                        else if (coordTransformed.x >= SourceRoi.x)
                        {
                            // right edge
                            w *= static_cast<CT>(1.0) - (coordTransformed.x - SourceRoi.x);
                        }
                        else // if (coordTransformed.y >= SourceRoi.y)
                        {
                            // lower edge
                            w *= static_cast<CT>(1.0) - (coordTransformed.y - SourceRoi.y);
                        }

                        // blend with destination pixel
                        // for integer type this might not be very precise, but the entire scheme is just to "look good"
                        // so we don't care...
                        pixel = pixel * w + (static_cast<CT>(1.0) - w) * PT(aDst);

                        if constexpr (RealOrComplexIntVector<DstT>)
                        {
                            round(pixel); // NOP for integer ComputeT
                        }

                        aDst = DstT(pixel);
                        return true;
                    }
                }
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
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
