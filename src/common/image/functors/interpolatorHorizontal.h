#pragma once
#include "borderControlHorizontal.h"
#include "interpolator.h"
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <common/mpp_defs.h>
#include <common/numberTypes.h>
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vector2.h>
#include <concepts>

#ifdef IS_HOST_COMPILER
#include <utility>
#define STD std::
#define ROUNDNEAREST std::nearbyint
#else
#define STD
#define ROUNDNEAREST rint
#endif
#include <numbers>

namespace mpp::image
{
template <InterpolationMode Interpol, typename CoordT> struct SuperSamplingParameterHorizontal
{
};

template <typename CoordT> struct SuperSamplingParameterHorizontal<InterpolationMode::Super, CoordT>
{
    const CoordT HalfScalingX;
    const CoordT SumWeightsInv;
};

template <typename PixelT, typename BorderControlT, typename CoordT, InterpolationMode Interpol>
struct InterpolatorHorizontal
{
    using coordinate_type     = CoordT;
    using pixel_type          = PixelT;
    using border_control_type = BorderControlT;

    const BorderControlT borderControl;
    const SuperSamplingParameterHorizontal<Interpol, CoordT> SuperParams;

#pragma region Constructors
    InterpolatorHorizontal(const BorderControlT &aBorderControl)
        requires(Interpol != InterpolationMode::Super)
        : borderControl(aBorderControl), SuperParams({})
    {
    }

    InterpolatorHorizontal(const BorderControlT &aBorderControl, CoordT aDownScalingFactorX)
        requires(Interpol == InterpolationMode::Super)
        : borderControl(aBorderControl),
          SuperParams({static_cast<CoordT>(1) / aDownScalingFactorX * static_cast<CoordT>(0.5), aDownScalingFactorX})
    {
    }
#pragma endregion

    DEVICE_CODE PixelT operator()(CoordT aPixelX, CoordT aPixelY) const
        requires std::same_as<CoordT, int>
    {
        if constexpr (Interpol == InterpolationMode::Undefined)
        {
            return borderControl(aPixelX, aPixelY);
        }
        else if constexpr (Interpol == InterpolationMode::NearestNeighbor)
        {
            return borderControl(aPixelX, aPixelY);
        }
        else
        {
            static_assert(Interpol != InterpolationMode::Undefined,
                          "Unsupported interpolation mode for integer coordinate type.");
        }
    }
    DEVICE_CODE PixelT operator()(CoordT aPixelX, CoordT aPixelY) const
        requires std::floating_point<CoordT>
    {
        if constexpr (Interpol == InterpolationMode::Undefined)
        {
            static_assert(Interpol == InterpolationMode::Undefined,
                          "Unsupported interpolation mode for floating point coordinate type.");
        }
        else if constexpr (Interpol == InterpolationMode::NearestNeighbor)
        {
            // use floor and +0.5 instead of round: a) negative coordinates are correct, b) we save 2 instructions
            const CoordT x0 = STD floor(aPixelX + static_cast<CoordT>(0.5));
            const int ix0   = static_cast<int>(x0);

            return borderControl(ix0, aPixelY);
        }
        else if constexpr (Interpol == InterpolationMode::Linear)
        {
            const CoordT x0 = STD floor(aPixelX);
            const int ix0   = static_cast<int>(x0);

            const CoordT diffX = aPixelX - x0;

            const PixelT pixel00 = borderControl(ix0, aPixelY);
            const PixelT pixel01 = borderControl(ix0 + 1, aPixelY);

            return pixel00 + (pixel01 - pixel00) * diffX;
        }
        else if constexpr (Interpol == InterpolationMode::CubicHermiteSpline)
        {
            const CoordT x0 = STD floor(aPixelX);
            const int ix0   = static_cast<int>(x0) - 1;

            const CoordT diffX  = aPixelX - x0;
            const CoordT diffX2 = diffX * diffX;
            const CoordT diffX3 = diffX2 * diffX;

            const PixelT f0  = borderControl(ix0 + 1, aPixelY);
            const PixelT f1  = borderControl(ix0 + 2, aPixelY);
            const PixelT fx0 = (f1 - PixelT(borderControl(ix0 + 0, aPixelY))) / static_cast<remove_vector_t<PixelT>>(2);
            const PixelT fx1 = (PixelT(borderControl(ix0 + 3, aPixelY)) - f0) / static_cast<remove_vector_t<PixelT>>(2);

            return f0 + fx0 * diffX +
                   (static_cast<remove_vector_t<PixelT>>(-3) * f0 + static_cast<remove_vector_t<PixelT>>(3) * f1 -
                    static_cast<remove_vector_t<PixelT>>(2) * fx0 - fx1) *
                       diffX2 +
                   (static_cast<remove_vector_t<PixelT>>(2) * f0 - static_cast<remove_vector_t<PixelT>>(2) * f1 + fx0 +
                    fx1) *
                       diffX3;
        }
        else if constexpr (Interpol == InterpolationMode::CubicLagrange)
        {
            const CoordT x0 = STD floor(aPixelX);
            const int ix0   = static_cast<int>(x0) - 1;

            const CoordT xx0 = aPixelX - x0 + static_cast<CoordT>(1);
            const CoordT xx1 = aPixelX - x0 + static_cast<CoordT>(1) - static_cast<CoordT>(1);
            const CoordT xx2 = aPixelX - x0 + static_cast<CoordT>(1) - static_cast<CoordT>(2);
            const CoordT xx3 = aPixelX - x0 + static_cast<CoordT>(1) - static_cast<CoordT>(3);

            const CoordT wx0 = xx1 * xx2 * xx3 / static_cast<CoordT>(-6);
            const CoordT wx1 = xx0 * xx2 * xx3 / static_cast<CoordT>(2);
            const CoordT wx2 = xx0 * xx1 * xx3 / static_cast<CoordT>(-2);
            const CoordT wx3 = xx0 * xx1 * xx2 / static_cast<CoordT>(6);

            const PixelT f0 = borderControl(ix0 + 0, aPixelY);
            const PixelT f1 = borderControl(ix0 + 1, aPixelY);
            const PixelT f2 = borderControl(ix0 + 2, aPixelY);
            const PixelT f3 = borderControl(ix0 + 3, aPixelY);

            return f0 * wx0 + f1 * wx1 + f2 * wx2 + f3 * wx3;
        }
        else if constexpr (Interpol == InterpolationMode::Cubic2ParamB05C03)
        {
            return TwoParameterCubic(aPixelX, aPixelY);
        }
        else if constexpr (Interpol == InterpolationMode::Cubic2ParamBSpline)
        {
            return TwoParameterCubic(aPixelX, aPixelY);
        }
        else if constexpr (Interpol == InterpolationMode::Cubic2ParamCatmullRom)
        {
            return TwoParameterCubic(aPixelX, aPixelY);
        }
        else if constexpr (Interpol == InterpolationMode::Lanczos2Lobed)
        {
            const CoordT x0 = STD floor(aPixelX);
            const int ix0   = static_cast<int>(x0) - 1;

            if (x0 == aPixelX)
            {
                // coordinate falls exactly on pixel.
                // avoid sinc(0) and skip all interpolation:
                return borderControl(ix0 + 1, aPixelY);
            }

            const CoordT l0 = LanczosKernel2(aPixelX - x0 + 1);
            const CoordT l1 = LanczosKernel2(aPixelX - x0 + 0);
            const CoordT l2 = LanczosKernel2(aPixelX - x0 - 1);
            const CoordT l3 = LanczosKernel2(aPixelX - x0 - 2);

            const PixelT f0 = borderControl(ix0 + 0, aPixelY);
            const PixelT f1 = borderControl(ix0 + 1, aPixelY);
            const PixelT f2 = borderControl(ix0 + 2, aPixelY);
            const PixelT f3 = borderControl(ix0 + 3, aPixelY);

            const CoordT w = static_cast<CoordT>(1) / (l0 + l1 + l2 + l3);

            return (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3) * w;
        }
        else if constexpr (Interpol == InterpolationMode::Lanczos3Lobed)
        {
            const CoordT x0 = STD floor(aPixelX);
            const int ix0   = static_cast<int>(x0) - 2;

            if (x0 == aPixelX)
            {
                // coordinate falls exactly on pixel.
                // avoid sinc(0) and skip all interpolation:
                return borderControl(ix0 + 2, aPixelY);
            }

            const CoordT l0 = LanczosKernel3(aPixelX - x0 + 2);
            const CoordT l1 = LanczosKernel3(aPixelX - x0 + 1);
            const CoordT l2 = LanczosKernel3(aPixelX - x0 + 0);
            const CoordT l3 = LanczosKernel3(aPixelX - x0 - 1);
            const CoordT l4 = LanczosKernel3(aPixelX - x0 - 2);
            const CoordT l5 = LanczosKernel3(aPixelX - x0 - 3);

            const PixelT f0 = borderControl(ix0 + 0, aPixelY);
            const PixelT f1 = borderControl(ix0 + 1, aPixelY);
            const PixelT f2 = borderControl(ix0 + 2, aPixelY);
            const PixelT f3 = borderControl(ix0 + 3, aPixelY);
            const PixelT f4 = borderControl(ix0 + 4, aPixelY);
            const PixelT f5 = borderControl(ix0 + 5, aPixelY);

            const CoordT w = static_cast<CoordT>(1) / (l0 + l1 + l2 + l3 + l4 + l5);

            return (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3 + f4 * l4 + f5 * l5) * w;
        }
        else if constexpr (Interpol == InterpolationMode::Super)
        {
            const CoordT xMin = aPixelX + static_cast<CoordT>(0.5) - SuperParams.HalfScalingX;
            const CoordT xMax = aPixelX + static_cast<CoordT>(0.5) + SuperParams.HalfScalingX;

            const CoordT xMinFull = STD ceil(xMin);
            const CoordT xMaxFull = STD floor(xMax);

            const CoordT ixMinFull = static_cast<int>(xMinFull);
            const CoordT ixMaxFull = static_cast<int>(xMaxFull);

            const CoordT wxMin = xMinFull - xMin;
            const CoordT wxMax = xMax - xMaxFull;

            PixelT pixelRes(0);
            for (int x = ixMinFull; x < ixMaxFull; x++)
            {
                pixelRes += PixelT(borderControl(x, aPixelY));
            }
            pixelRes += PixelT(borderControl(ixMinFull - 1, aPixelY)) * wxMin;

            pixelRes += PixelT(borderControl(ixMaxFull, aPixelY)) * wxMax;

            return pixelRes * SuperParams.SumWeightsInv;
        }
        else
        {
            static_assert(Interpol != InterpolationMode::Undefined,
                          "Unsupported interpolation mode for floating point coordinate type.");
        }

        // return PixelT(); // dummy
    }

  private:
    DEVICE_CODE PixelT TwoParameterCubic(CoordT aPixelX, CoordT aPixelY) const
    {
        using Param = CubicTwoParams<CoordT, Interpol>;

        const CoordT x0 = STD floor(aPixelX);
        const int ix0   = static_cast<int>(x0) - 1;

        const CoordT diffX  = aPixelX - x0;
        const CoordT diffX2 = diffX * diffX;
        const CoordT diffX3 = diffX2 * diffX;

        const PixelT p0 = borderControl(ix0 + 0, aPixelY);
        const PixelT p1 = borderControl(ix0 + 1, aPixelY);
        const PixelT p2 = borderControl(ix0 + 2, aPixelY);
        const PixelT p3 = borderControl(ix0 + 3, aPixelY);

        PixelT res = (static_cast<CoordT>(-1.0 / 6.0 * Param::B - Param::C) * p0 +
                      static_cast<CoordT>(-3.0 / 2.0 * Param::B - Param::C + 2.0) * p1 +
                      static_cast<CoordT>(3.0 / 2.0 * Param::B + Param::C - 2.0) * p2 +
                      static_cast<CoordT>(1.0 / 6.0 * Param::B + Param::C) * p3) *
                     diffX3;
        res += (static_cast<CoordT>(1.0 / 2.0 * Param::B + 2.0 * Param::C) * p0 +
                static_cast<CoordT>(2.0 * Param::B + Param::C - 3.0) * p1 +
                static_cast<CoordT>(-5.0 / 2.0 * Param::B - 2.0 * Param::C + 3.0) * p2 - Param::C * p3) *
               diffX2;
        res += (static_cast<CoordT>(-1.0 / 2.0 * Param::B - Param::C) * p0 +
                static_cast<CoordT>(1.0 / 2.0 * Param::B + Param::C) * p2) *
               diffX;
        res += static_cast<CoordT>(1.0 / 6.0 * Param::B) * p0 + static_cast<CoordT>(-1.0 / 3.0 * Param::B + 1.0) * p1 +
               static_cast<CoordT>(1.0 / 6.0 * Param::B) * p2;

        return res;
    }

    static DEVICE_CODE CoordT LanczosKernel2(CoordT aX)
    {
        return sinc_never0(aX) * sinc_never0(aX / static_cast<CoordT>(2));
    }

    static DEVICE_CODE CoordT LanczosKernel3(CoordT aX)
    {
        return sinc_never0(aX) * sinc_never0(aX / static_cast<CoordT>(3));
    }
};
} // namespace mpp::image

#undef STD