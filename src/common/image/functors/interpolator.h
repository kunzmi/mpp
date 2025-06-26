#pragma once
#include "borderControl.h"
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <common/numberTypes.h>
#include <common/mpp_defs.h>
#include <common/utilities.h>
#include <common/vector2.h>
#include <common/vector_typetraits.h>
#include <concepts>

#ifdef IS_HOST_COMPILER
#include <utility>
#define STD std::
#else
#define STD
#endif
#include <numbers>

namespace mpp::image
{

template <typename PixelT> struct interpolator_coordinate_type
{
    using type = float;
};
template <> struct interpolator_coordinate_type<double>
{
    using type = double;
};

template <typename PixelT>
using interpolator_coordinate_type_t =
    typename interpolator_coordinate_type<complex_basetype_t<remove_vector_t<PixelT>>>::type;

template <InterpolationMode Interpol, typename CoordT> struct SuperSamplingParameter
{
};

template <typename CoordT> struct SuperSamplingParameter<InterpolationMode::Super, CoordT>
{
    const CoordT HalfScalingX;
    const CoordT HalfScalingY;
    const CoordT SumWeightsInv;
};

template <typename CoordT, InterpolationMode Interpol> struct CubicTwoParams
{
};

template <typename CoordT> struct CubicTwoParams<CoordT, InterpolationMode::Cubic2ParamB05C03>
{
    static constexpr CoordT B = static_cast<CoordT>(0.5);
    static constexpr CoordT C = static_cast<CoordT>(3.0 / 10.0);
};
template <typename CoordT> struct CubicTwoParams<CoordT, InterpolationMode::Cubic2ParamBSpline>
{
    static constexpr CoordT B = static_cast<CoordT>(1);
    static constexpr CoordT C = static_cast<CoordT>(0);
};
template <typename CoordT> struct CubicTwoParams<CoordT, InterpolationMode::Cubic2ParamCatmullRom>
{
    static constexpr CoordT B = static_cast<CoordT>(0);
    static constexpr CoordT C = static_cast<CoordT>(0.5);
};

template <typename PixelT, typename BorderControlT, typename CoordT, InterpolationMode Interpol> struct Interpolator
{
    using coordinate_type = CoordT;
    using pixel_type      = PixelT;

    const BorderControlT borderControl;
    const SuperSamplingParameter<Interpol, CoordT> SuperParams;

#pragma region Constructors
    Interpolator(const BorderControlT &aBorderControl)
        requires(Interpol != InterpolationMode::Super)
        : borderControl(aBorderControl), SuperParams({})
    {
    }

    Interpolator(const BorderControlT &aBorderControl, CoordT aDownScalingFactorX, CoordT aDownScalingFactorY)
        requires(Interpol == InterpolationMode::Super)
        : borderControl(aBorderControl),
          SuperParams({static_cast<CoordT>(1) / aDownScalingFactorX * static_cast<CoordT>(0.5),
                       static_cast<CoordT>(1) / aDownScalingFactorY * static_cast<CoordT>(0.5),
                       aDownScalingFactorX * aDownScalingFactorY})
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
            const CoordT y0 = STD floor(aPixelY + static_cast<CoordT>(0.5));
            const int ix0   = static_cast<int>(x0);
            const int iy0   = static_cast<int>(y0);

            return borderControl(ix0, iy0);
        }
        else if constexpr (Interpol == InterpolationMode::Linear)
        {
            const CoordT x0 = STD floor(aPixelX);
            const CoordT y0 = STD floor(aPixelY);
            const int ix0   = static_cast<int>(x0);
            const int iy0   = static_cast<int>(y0);

            const CoordT diffX = aPixelX - x0;

            PixelT pixel00       = borderControl(ix0, iy0);
            const PixelT pixel01 = borderControl(ix0 + 1, iy0);

            pixel00 = pixel00 + (pixel01 - pixel00) * diffX;

            PixelT pixel10       = borderControl(ix0, iy0 + 1);
            const PixelT pixel11 = borderControl(ix0 + 1, iy0 + 1);

            pixel10 = pixel10 + (pixel11 - pixel10) * diffX;

            const CoordT diffY = aPixelY - y0;

            return pixel00 + (pixel10 - pixel00) * diffY;
        }
        else if constexpr (Interpol == InterpolationMode::CubicHermiteSpline)
        {
            const CoordT x0 = STD floor(aPixelX);
            const CoordT y0 = STD floor(aPixelY);
            const int ix0   = static_cast<int>(x0) - 1;
            const int iy0   = static_cast<int>(y0) - 1;

            const CoordT diffX  = aPixelX - x0;
            const CoordT diffX2 = diffX * diffX;
            const CoordT diffX3 = diffX2 * diffX;

            PixelT f0  = borderControl(ix0 + 1, iy0 + 0);
            PixelT f1  = borderControl(ix0 + 2, iy0 + 0);
            PixelT fx0 = (f1 - PixelT(borderControl(ix0 + 0, iy0 + 0))) / static_cast<remove_vector_t<PixelT>>(2);
            PixelT fx1 = (PixelT(borderControl(ix0 + 3, iy0 + 0)) - f0) / static_cast<remove_vector_t<PixelT>>(2);

            const PixelT py0 =
                f0 + fx0 * diffX +
                (static_cast<remove_vector_t<PixelT>>(-3) * f0 + static_cast<remove_vector_t<PixelT>>(3) * f1 -
                 static_cast<remove_vector_t<PixelT>>(2) * fx0 - fx1) *
                    diffX2 +
                (static_cast<remove_vector_t<PixelT>>(2) * f0 - static_cast<remove_vector_t<PixelT>>(2) * f1 + fx0 +
                 fx1) *
                    diffX3;

            f0  = borderControl(ix0 + 1, iy0 + 1);
            f1  = borderControl(ix0 + 2, iy0 + 1);
            fx0 = (f1 - PixelT(borderControl(ix0 + 0, iy0 + 1))) / static_cast<remove_vector_t<PixelT>>(2);
            fx1 = (PixelT(borderControl(ix0 + 3, iy0 + 1)) - f0) / static_cast<remove_vector_t<PixelT>>(2);

            const PixelT py1 =
                f0 + fx0 * diffX +
                (static_cast<remove_vector_t<PixelT>>(-3) * f0 + static_cast<remove_vector_t<PixelT>>(3) * f1 -
                 static_cast<remove_vector_t<PixelT>>(2) * fx0 - fx1) *
                    diffX2 +
                (static_cast<remove_vector_t<PixelT>>(2) * f0 - static_cast<remove_vector_t<PixelT>>(2) * f1 + fx0 +
                 fx1) *
                    diffX3;

            f0  = borderControl(ix0 + 1, iy0 + 2);
            f1  = borderControl(ix0 + 2, iy0 + 2);
            fx0 = (f1 - PixelT(borderControl(ix0 + 0, iy0 + 2))) / static_cast<remove_vector_t<PixelT>>(2);
            fx1 = (PixelT(borderControl(ix0 + 3, iy0 + 2)) - f0) / static_cast<remove_vector_t<PixelT>>(2);

            const PixelT py2 =
                f0 + fx0 * diffX +
                (static_cast<remove_vector_t<PixelT>>(-3) * f0 + static_cast<remove_vector_t<PixelT>>(3) * f1 -
                 static_cast<remove_vector_t<PixelT>>(2) * fx0 - fx1) *
                    diffX2 +
                (static_cast<remove_vector_t<PixelT>>(2) * f0 - static_cast<remove_vector_t<PixelT>>(2) * f1 + fx0 +
                 fx1) *
                    diffX3;

            f0  = borderControl(ix0 + 1, iy0 + 3);
            f1  = borderControl(ix0 + 2, iy0 + 3);
            fx0 = (f1 - PixelT(borderControl(ix0 + 0, iy0 + 3))) / static_cast<remove_vector_t<PixelT>>(2);
            fx1 = (PixelT(borderControl(ix0 + 3, iy0 + 3)) - f0) / static_cast<remove_vector_t<PixelT>>(2);

            const PixelT py3 =
                f0 + fx0 * diffX +
                (static_cast<remove_vector_t<PixelT>>(-3) * f0 + static_cast<remove_vector_t<PixelT>>(3) * f1 -
                 static_cast<remove_vector_t<PixelT>>(2) * fx0 - fx1) *
                    diffX2 +
                (static_cast<remove_vector_t<PixelT>>(2) * f0 - static_cast<remove_vector_t<PixelT>>(2) * f1 + fx0 +
                 fx1) *
                    diffX3;

            const CoordT diffY  = aPixelY - y0;
            const CoordT diffY2 = diffY * diffY;
            const CoordT diffY3 = diffY2 * diffY;

            f0  = py1;
            f1  = py2;
            fx0 = (f1 - py0) / static_cast<remove_vector_t<PixelT>>(2);
            fx1 = (py3 - f0) / static_cast<remove_vector_t<PixelT>>(2);
            return f0 + fx0 * diffY +
                   (static_cast<remove_vector_t<PixelT>>(-3) * f0 + static_cast<remove_vector_t<PixelT>>(3) * f1 -
                    static_cast<remove_vector_t<PixelT>>(2) * fx0 - fx1) *
                       diffY2 +
                   (static_cast<remove_vector_t<PixelT>>(2) * f0 - static_cast<remove_vector_t<PixelT>>(2) * f1 + fx0 +
                    fx1) *
                       diffY3;
        }
        else if constexpr (Interpol == InterpolationMode::CubicLagrange)
        {
            const CoordT x0 = STD floor(aPixelX);
            const CoordT y0 = STD floor(aPixelY);
            const int ix0   = static_cast<int>(x0) - 1;
            const int iy0   = static_cast<int>(y0) - 1;

            const CoordT xx0 = aPixelX - x0 + static_cast<CoordT>(1);
            const CoordT xx1 = aPixelX - x0 + static_cast<CoordT>(1) - static_cast<CoordT>(1);
            const CoordT xx2 = aPixelX - x0 + static_cast<CoordT>(1) - static_cast<CoordT>(2);
            const CoordT xx3 = aPixelX - x0 + static_cast<CoordT>(1) - static_cast<CoordT>(3);

            const CoordT wx0 = xx1 * xx2 * xx3 / static_cast<CoordT>(-6);
            const CoordT wx1 = xx0 * xx2 * xx3 / static_cast<CoordT>(2);
            const CoordT wx2 = xx0 * xx1 * xx3 / static_cast<CoordT>(-2);
            const CoordT wx3 = xx0 * xx1 * xx2 / static_cast<CoordT>(6);

            PixelT f0 = borderControl(ix0 + 0, iy0 + 0);
            PixelT f1 = borderControl(ix0 + 1, iy0 + 0);
            PixelT f2 = borderControl(ix0 + 2, iy0 + 0);
            PixelT f3 = borderControl(ix0 + 3, iy0 + 0);

            const PixelT py0 = f0 * wx0 + f1 * wx1 + f2 * wx2 + f3 * wx3;

            f0 = borderControl(ix0 + 0, iy0 + 1);
            f1 = borderControl(ix0 + 1, iy0 + 1);
            f2 = borderControl(ix0 + 2, iy0 + 1);
            f3 = borderControl(ix0 + 3, iy0 + 1);

            const PixelT py1 = f0 * wx0 + f1 * wx1 + f2 * wx2 + f3 * wx3;

            f0 = borderControl(ix0 + 0, iy0 + 2);
            f1 = borderControl(ix0 + 1, iy0 + 2);
            f2 = borderControl(ix0 + 2, iy0 + 2);
            f3 = borderControl(ix0 + 3, iy0 + 2);

            const PixelT py2 = f0 * wx0 + f1 * wx1 + f2 * wx2 + f3 * wx3;

            f0 = borderControl(ix0 + 0, iy0 + 3);
            f1 = borderControl(ix0 + 1, iy0 + 3);
            f2 = borderControl(ix0 + 2, iy0 + 3);
            f3 = borderControl(ix0 + 3, iy0 + 3);

            const PixelT py3 = f0 * wx0 + f1 * wx1 + f2 * wx2 + f3 * wx3;

            const CoordT yy0 = aPixelY - y0 + static_cast<CoordT>(1);
            const CoordT yy1 = aPixelY - y0 + static_cast<CoordT>(1) - static_cast<CoordT>(1);
            const CoordT yy2 = aPixelY - y0 + static_cast<CoordT>(1) - static_cast<CoordT>(2);
            const CoordT yy3 = aPixelY - y0 + static_cast<CoordT>(1) - static_cast<CoordT>(3);

            const CoordT wy0 = yy1 * yy2 * yy3 / static_cast<CoordT>(-6);
            const CoordT wy1 = yy0 * yy2 * yy3 / static_cast<CoordT>(2);
            const CoordT wy2 = yy0 * yy1 * yy3 / static_cast<CoordT>(-2);
            const CoordT wy3 = yy0 * yy1 * yy2 / static_cast<CoordT>(6);

            return py0 * wy0 + py1 * wy1 + py2 * wy2 + py3 * wy3;
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
            const CoordT y0 = STD floor(aPixelY);
            const int ix0   = static_cast<int>(x0) - 1;
            const int iy0   = static_cast<int>(y0) - 1;

            if (x0 == aPixelX)
            {
                // coordinate falls exactly on pixel.
                // avoid sinc(0) and skip all interpolation:
                if (y0 == aPixelY)
                {
                    return borderControl(ix0 + 1, iy0 + 1);
                }
                else
                {
                    const PixelT py0 = borderControl(ix0 + 1, iy0 + 0);
                    const PixelT py1 = borderControl(ix0 + 1, iy0 + 1);
                    const PixelT py2 = borderControl(ix0 + 1, iy0 + 2);
                    const PixelT py3 = borderControl(ix0 + 1, iy0 + 3);

                    const CoordT yl0 = LanczosKernel2(aPixelY - y0 + 1);
                    const CoordT yl1 = LanczosKernel2(aPixelY - y0 + 0);
                    const CoordT yl2 = LanczosKernel2(aPixelY - y0 - 1);
                    const CoordT yl3 = LanczosKernel2(aPixelY - y0 - 2);

                    const CoordT yw = static_cast<CoordT>(1) / (yl0 + yl1 + yl2 + yl3);

                    return (py0 * yl0 + py1 * yl1 + py2 * yl2 + py3 * yl3) * yw;
                }
            }

            if (y0 == aPixelY)
            {
                const CoordT l0 = LanczosKernel2(aPixelX - x0 + 1);
                const CoordT l1 = LanczosKernel2(aPixelX - x0 + 0);
                const CoordT l2 = LanczosKernel2(aPixelX - x0 - 1);
                const CoordT l3 = LanczosKernel2(aPixelX - x0 - 2);

                const PixelT f0 = borderControl(ix0 + 0, iy0 + 1);
                const PixelT f1 = borderControl(ix0 + 1, iy0 + 1);
                const PixelT f2 = borderControl(ix0 + 2, iy0 + 1);
                const PixelT f3 = borderControl(ix0 + 3, iy0 + 1);

                const CoordT w = static_cast<CoordT>(1) / (l0 + l1 + l2 + l3);

                return (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3) * w;
            }

            PixelT f0 = borderControl(ix0 + 0, iy0 + 0);
            PixelT f1 = borderControl(ix0 + 1, iy0 + 0);
            PixelT f2 = borderControl(ix0 + 2, iy0 + 0);
            PixelT f3 = borderControl(ix0 + 3, iy0 + 0);

            const CoordT l0 = LanczosKernel2(aPixelX - x0 + 1);
            const CoordT l1 = LanczosKernel2(aPixelX - x0 + 0);
            const CoordT l2 = LanczosKernel2(aPixelX - x0 - 1);
            const CoordT l3 = LanczosKernel2(aPixelX - x0 - 2);

            const CoordT w = static_cast<CoordT>(1) / (l0 + l1 + l2 + l3);

            const PixelT py0 = (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3) * w;

            f0 = borderControl(ix0 + 0, iy0 + 1);
            f1 = borderControl(ix0 + 1, iy0 + 1);
            f2 = borderControl(ix0 + 2, iy0 + 1);
            f3 = borderControl(ix0 + 3, iy0 + 1);

            const PixelT py1 = (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3) * w;

            f0 = borderControl(ix0 + 0, iy0 + 2);
            f1 = borderControl(ix0 + 1, iy0 + 2);
            f2 = borderControl(ix0 + 2, iy0 + 2);
            f3 = borderControl(ix0 + 3, iy0 + 2);

            const PixelT py2 = (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3) * w;

            f0 = borderControl(ix0 + 0, iy0 + 3);
            f1 = borderControl(ix0 + 1, iy0 + 3);
            f2 = borderControl(ix0 + 2, iy0 + 3);
            f3 = borderControl(ix0 + 3, iy0 + 3);

            const PixelT py3 = (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3) * w;

            const CoordT yl0 = LanczosKernel2(aPixelY - y0 + 1);
            const CoordT yl1 = LanczosKernel2(aPixelY - y0 + 0);
            const CoordT yl2 = LanczosKernel2(aPixelY - y0 - 1);
            const CoordT yl3 = LanczosKernel2(aPixelY - y0 - 2);

            const CoordT yw = static_cast<CoordT>(1) / (yl0 + yl1 + yl2 + yl3);

            return (py0 * yl0 + py1 * yl1 + py2 * yl2 + py3 * yl3) * yw;
        }
        else if constexpr (Interpol == InterpolationMode::Lanczos3Lobed)
        {
            const CoordT x0 = STD floor(aPixelX);
            const CoordT y0 = STD floor(aPixelY);
            const int ix0   = static_cast<int>(x0) - 2;
            const int iy0   = static_cast<int>(y0) - 2;

            if (x0 == aPixelX)
            {
                // coordinate falls exactly on pixel.
                // avoid sinc(0) and skip all interpolation:
                if (y0 == aPixelY)
                {
                    return borderControl(ix0 + 2, iy0 + 2);
                }
                else
                {
                    const PixelT py0 = borderControl(ix0 + 2, iy0 + 0);
                    const PixelT py1 = borderControl(ix0 + 2, iy0 + 1);
                    const PixelT py2 = borderControl(ix0 + 2, iy0 + 2);
                    const PixelT py3 = borderControl(ix0 + 2, iy0 + 3);
                    const PixelT py4 = borderControl(ix0 + 2, iy0 + 4);
                    const PixelT py5 = borderControl(ix0 + 2, iy0 + 5);

                    const CoordT yl0 = LanczosKernel3(aPixelY - y0 + 2);
                    const CoordT yl1 = LanczosKernel3(aPixelY - y0 + 1);
                    const CoordT yl2 = LanczosKernel3(aPixelY - y0 + 0);
                    const CoordT yl3 = LanczosKernel3(aPixelY - y0 - 1);
                    const CoordT yl4 = LanczosKernel3(aPixelY - y0 - 2);
                    const CoordT yl5 = LanczosKernel3(aPixelY - y0 - 3);

                    const CoordT yw = yl0 + yl1 + yl2 + yl3 + yl4 + yl5;

                    return (py0 * yl0 + py1 * yl1 + py2 * yl2 + py3 * yl3 + py4 * yl4 + py5 * yl5) / yw;
                }
            }

            if (y0 == aPixelY)
            {
                const CoordT l0 = LanczosKernel3(aPixelX - x0 + 2);
                const CoordT l1 = LanczosKernel3(aPixelX - x0 + 1);
                const CoordT l2 = LanczosKernel3(aPixelX - x0 + 0);
                const CoordT l3 = LanczosKernel3(aPixelX - x0 - 1);
                const CoordT l4 = LanczosKernel3(aPixelX - x0 - 2);
                const CoordT l5 = LanczosKernel3(aPixelX - x0 - 3);

                const PixelT f0 = borderControl(ix0 + 0, iy0 + 2);
                const PixelT f1 = borderControl(ix0 + 1, iy0 + 2);
                const PixelT f2 = borderControl(ix0 + 2, iy0 + 2);
                const PixelT f3 = borderControl(ix0 + 3, iy0 + 2);
                const PixelT f4 = borderControl(ix0 + 4, iy0 + 2);
                const PixelT f5 = borderControl(ix0 + 5, iy0 + 2);

                const CoordT w = static_cast<CoordT>(1) / (l0 + l1 + l2 + l3 + l4 + l5);

                return (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3 + f4 * l4 + f5 * l5) * w;
            }

            PixelT f0 = borderControl(ix0 + 0, iy0 + 0);
            PixelT f1 = borderControl(ix0 + 1, iy0 + 0);
            PixelT f2 = borderControl(ix0 + 2, iy0 + 0);
            PixelT f3 = borderControl(ix0 + 3, iy0 + 0);
            PixelT f4 = borderControl(ix0 + 4, iy0 + 0);
            PixelT f5 = borderControl(ix0 + 5, iy0 + 0);

            const CoordT l0 = LanczosKernel3(aPixelX - x0 + 2);
            const CoordT l1 = LanczosKernel3(aPixelX - x0 + 1);
            const CoordT l2 = LanczosKernel3(aPixelX - x0 + 0);
            const CoordT l3 = LanczosKernel3(aPixelX - x0 - 1);
            const CoordT l4 = LanczosKernel3(aPixelX - x0 - 2);
            const CoordT l5 = LanczosKernel3(aPixelX - x0 - 3);

            const CoordT w = static_cast<CoordT>(1) / (l0 + l1 + l2 + l3 + l4 + l5);

            const PixelT py0 = (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3 + f4 * l4 + f5 * l5) * w;

            f0 = borderControl(ix0 + 0, iy0 + 1);
            f1 = borderControl(ix0 + 1, iy0 + 1);
            f2 = borderControl(ix0 + 2, iy0 + 1);
            f3 = borderControl(ix0 + 3, iy0 + 1);
            f4 = borderControl(ix0 + 4, iy0 + 1);
            f5 = borderControl(ix0 + 5, iy0 + 1);

            const PixelT py1 = (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3 + f4 * l4 + f5 * l5) * w;

            f0 = borderControl(ix0 + 0, iy0 + 2);
            f1 = borderControl(ix0 + 1, iy0 + 2);
            f2 = borderControl(ix0 + 2, iy0 + 2);
            f3 = borderControl(ix0 + 3, iy0 + 2);
            f4 = borderControl(ix0 + 4, iy0 + 2);
            f5 = borderControl(ix0 + 5, iy0 + 2);

            const PixelT py2 = (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3 + f4 * l4 + f5 * l5) * w;

            f0 = borderControl(ix0 + 0, iy0 + 3);
            f1 = borderControl(ix0 + 1, iy0 + 3);
            f2 = borderControl(ix0 + 2, iy0 + 3);
            f3 = borderControl(ix0 + 3, iy0 + 3);
            f4 = borderControl(ix0 + 4, iy0 + 3);
            f5 = borderControl(ix0 + 5, iy0 + 3);

            const PixelT py3 = (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3 + f4 * l4 + f5 * l5) * w;

            f0 = borderControl(ix0 + 0, iy0 + 4);
            f1 = borderControl(ix0 + 1, iy0 + 4);
            f2 = borderControl(ix0 + 2, iy0 + 4);
            f3 = borderControl(ix0 + 3, iy0 + 4);
            f4 = borderControl(ix0 + 4, iy0 + 4);
            f5 = borderControl(ix0 + 5, iy0 + 4);

            const PixelT py4 = (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3 + f4 * l4 + f5 * l5) * w;

            f0 = borderControl(ix0 + 0, iy0 + 5);
            f1 = borderControl(ix0 + 1, iy0 + 5);
            f2 = borderControl(ix0 + 2, iy0 + 5);
            f3 = borderControl(ix0 + 3, iy0 + 5);
            f4 = borderControl(ix0 + 4, iy0 + 5);
            f5 = borderControl(ix0 + 5, iy0 + 5);

            const PixelT py5 = (f0 * l0 + f1 * l1 + f2 * l2 + f3 * l3 + f4 * l4 + f5 * l5) * w;

            const CoordT yl0 = LanczosKernel3(aPixelY - y0 + 2);
            const CoordT yl1 = LanczosKernel3(aPixelY - y0 + 1);
            const CoordT yl2 = LanczosKernel3(aPixelY - y0 + 0);
            const CoordT yl3 = LanczosKernel3(aPixelY - y0 - 1);
            const CoordT yl4 = LanczosKernel3(aPixelY - y0 - 2);
            const CoordT yl5 = LanczosKernel3(aPixelY - y0 - 3);

            const CoordT yw = yl0 + yl1 + yl2 + yl3 + yl4 + yl5;

            return (py0 * yl0 + py1 * yl1 + py2 * yl2 + py3 * yl3 + py4 * yl4 + py5 * yl5) / yw;
        }
        else if constexpr (Interpol == InterpolationMode::Super)
        {
            const CoordT xMin = aPixelX + static_cast<CoordT>(0.5) - SuperParams.HalfScalingX;
            const CoordT xMax = aPixelX + static_cast<CoordT>(0.5) + SuperParams.HalfScalingX;

            const CoordT yMin = aPixelY + static_cast<CoordT>(0.5) - SuperParams.HalfScalingY;
            const CoordT yMax = aPixelY + static_cast<CoordT>(0.5) + SuperParams.HalfScalingY;

            const CoordT xMinFull = STD ceil(xMin);
            const CoordT xMaxFull = STD floor(xMax);

            const CoordT yMinFull = STD ceil(yMin);
            const CoordT yMaxFull = STD floor(yMax);

            const CoordT ixMinFull = static_cast<int>(xMinFull);
            const CoordT ixMaxFull = static_cast<int>(xMaxFull);

            const CoordT iyMinFull = static_cast<int>(yMinFull);
            const CoordT iyMaxFull = static_cast<int>(yMaxFull);

            const CoordT wxMin = xMinFull - xMin;
            const CoordT wxMax = xMax - xMaxFull;
            const CoordT wyMin = yMinFull - yMin;
            const CoordT wyMax = yMax - yMaxFull;

            PixelT pixelRes(0);
            for (int y = iyMinFull; y < iyMaxFull; y++)
            {
                for (int x = ixMinFull; x < ixMaxFull; x++)
                {
                    pixelRes += PixelT(borderControl(x, y));
                }
                pixelRes += PixelT(borderControl(ixMinFull - 1, y)) * wxMin;

                pixelRes += PixelT(borderControl(ixMaxFull, y)) * wxMax;
            }

            for (int x = ixMinFull; x < ixMaxFull; x++)
            {
                pixelRes += PixelT(borderControl(x, iyMinFull - 1)) * wyMin;
            }
            pixelRes += PixelT(borderControl(ixMinFull - 1, iyMinFull - 1)) * wxMin * wyMin;

            pixelRes += PixelT(borderControl(ixMaxFull, iyMinFull - 1)) * wxMax * wyMin;

            for (int x = ixMinFull; x < ixMaxFull; x++)
            {
                pixelRes += PixelT(borderControl(x, iyMaxFull)) * wyMax;
            }
            pixelRes += PixelT(borderControl(ixMinFull - 1, iyMaxFull)) * wxMin * wyMax;

            pixelRes += PixelT(borderControl(ixMaxFull, iyMaxFull)) * wxMax * wyMax;

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
        const CoordT y0 = STD floor(aPixelY);
        const int ix0   = static_cast<int>(x0) - 1;
        const int iy0   = static_cast<int>(y0) - 1;

        const CoordT diffX  = aPixelX - x0;
        const CoordT diffX2 = diffX * diffX;
        const CoordT diffX3 = diffX2 * diffX;

        PixelT p0 = borderControl(ix0 + 0, iy0 + 0);
        PixelT p1 = borderControl(ix0 + 1, iy0 + 0);
        PixelT p2 = borderControl(ix0 + 2, iy0 + 0);
        PixelT p3 = borderControl(ix0 + 3, iy0 + 0);

        PixelT py0 = (static_cast<CoordT>(-1.0 / 6.0 * Param::B - Param::C) * p0 +
                      static_cast<CoordT>(-3.0 / 2.0 * Param::B - Param::C + 2.0) * p1 +
                      static_cast<CoordT>(3.0 / 2.0 * Param::B + Param::C - 2.0) * p2 +
                      static_cast<CoordT>(1.0 / 6.0 * Param::B + Param::C) * p3) *
                     diffX3;
        py0 += (static_cast<CoordT>(1.0 / 2.0 * Param::B + 2.0 * Param::C) * p0 +
                static_cast<CoordT>(2.0 * Param::B + Param::C - 3.0) * p1 +
                static_cast<CoordT>(-5.0 / 2.0 * Param::B - 2.0 * Param::C + 3.0) * p2 - Param::C * p3) *
               diffX2;
        py0 += (static_cast<CoordT>(-1.0 / 2.0 * Param::B - Param::C) * p0 +
                static_cast<CoordT>(1.0 / 2.0 * Param::B + Param::C) * p2) *
               diffX;
        py0 += static_cast<CoordT>(1.0 / 6.0 * Param::B) * p0 + static_cast<CoordT>(-1.0 / 3.0 * Param::B + 1.0) * p1 +
               static_cast<CoordT>(1.0 / 6.0 * Param::B) * p2;

        p0 = borderControl(ix0 + 0, iy0 + 1);
        p1 = borderControl(ix0 + 1, iy0 + 1);
        p2 = borderControl(ix0 + 2, iy0 + 1);
        p3 = borderControl(ix0 + 3, iy0 + 1);

        PixelT py1 = (static_cast<CoordT>(-1.0 / 6.0 * Param::B - Param::C) * p0 +
                      static_cast<CoordT>(-3.0 / 2.0 * Param::B - Param::C + 2.0) * p1 +
                      static_cast<CoordT>(3.0 / 2.0 * Param::B + Param::C - 2.0) * p2 +
                      static_cast<CoordT>(1.0 / 6.0 * Param::B + Param::C) * p3) *
                     diffX3;
        py1 += (static_cast<CoordT>(1.0 / 2.0 * Param::B + 2.0 * Param::C) * p0 +
                static_cast<CoordT>(2.0 * Param::B + Param::C - 3.0) * p1 +
                static_cast<CoordT>(-5.0 / 2.0 * Param::B - 2.0 * Param::C + 3.0) * p2 - Param::C * p3) *
               diffX2;
        py1 += (static_cast<CoordT>(-1.0 / 2.0 * Param::B - Param::C) * p0 +
                static_cast<CoordT>(1.0 / 2.0 * Param::B + Param::C) * p2) *
               diffX;
        py1 += static_cast<CoordT>(1.0 / 6.0 * Param::B) * p0 + static_cast<CoordT>(-1.0 / 3.0 * Param::B + 1.0) * p1 +
               static_cast<CoordT>(1.0 / 6.0 * Param::B) * p2;

        p0 = borderControl(ix0 + 0, iy0 + 2);
        p1 = borderControl(ix0 + 1, iy0 + 2);
        p2 = borderControl(ix0 + 2, iy0 + 2);
        p3 = borderControl(ix0 + 3, iy0 + 2);

        PixelT py2 = (static_cast<CoordT>(-1.0 / 6.0 * Param::B - Param::C) * p0 +
                      static_cast<CoordT>(-3.0 / 2.0 * Param::B - Param::C + 2.0) * p1 +
                      static_cast<CoordT>(3.0 / 2.0 * Param::B + Param::C - 2.0) * p2 +
                      static_cast<CoordT>(1.0 / 6.0 * Param::B + Param::C) * p3) *
                     diffX3;
        py2 += (static_cast<CoordT>(1.0 / 2.0 * Param::B + 2.0 * Param::C) * p0 +
                static_cast<CoordT>(2.0 * Param::B + Param::C - 3.0) * p1 +
                static_cast<CoordT>(-5.0 / 2.0 * Param::B - 2.0 * Param::C + 3.0) * p2 - Param::C * p3) *
               diffX2;
        py2 += (static_cast<CoordT>(-1.0 / 2.0 * Param::B - Param::C) * p0 +
                static_cast<CoordT>(1.0 / 2.0 * Param::B + Param::C) * p2) *
               diffX;
        py2 += static_cast<CoordT>(1.0 / 6.0 * Param::B) * p0 + static_cast<CoordT>(-1.0 / 3.0 * Param::B + 1.0) * p1 +
               static_cast<CoordT>(1.0 / 6.0 * Param::B) * p2;

        p0 = borderControl(ix0 + 0, iy0 + 3);
        p1 = borderControl(ix0 + 1, iy0 + 3);
        p2 = borderControl(ix0 + 2, iy0 + 3);
        p3 = borderControl(ix0 + 3, iy0 + 3);

        PixelT py3 = (static_cast<CoordT>(-1.0 / 6.0 * Param::B - Param::C) * p0 +
                      static_cast<CoordT>(-3.0 / 2.0 * Param::B - Param::C + 2.0) * p1 +
                      static_cast<CoordT>(3.0 / 2.0 * Param::B + Param::C - 2.0) * p2 +
                      static_cast<CoordT>(1.0 / 6.0 * Param::B + Param::C) * p3) *
                     diffX3;
        py3 += (static_cast<CoordT>(1.0 / 2.0 * Param::B + 2.0 * Param::C) * p0 +
                static_cast<CoordT>(2.0 * Param::B + Param::C - 3.0) * p1 +
                static_cast<CoordT>(-5.0 / 2.0 * Param::B - 2.0 * Param::C + 3.0) * p2 - Param::C * p3) *
               diffX2;
        py3 += (static_cast<CoordT>(-1.0 / 2.0 * Param::B - Param::C) * p0 +
                static_cast<CoordT>(1.0 / 2.0 * Param::B + Param::C) * p2) *
               diffX;
        py3 += static_cast<CoordT>(1.0 / 6.0 * Param::B) * p0 + static_cast<CoordT>(-1.0 / 3.0 * Param::B + 1.0) * p1 +
               static_cast<CoordT>(1.0 / 6.0 * Param::B) * p2;

        const CoordT diffY  = aPixelY - y0;
        const CoordT diffY2 = diffY * diffY;
        const CoordT diffY3 = diffY2 * diffY;

        PixelT res = (static_cast<CoordT>(-1.0 / 6.0 * Param::B - Param::C) * py0 +
                      static_cast<CoordT>(-3.0 / 2.0 * Param::B - Param::C + 2.0) * py1 +
                      static_cast<CoordT>(3.0 / 2.0 * Param::B + Param::C - 2.0) * py2 +
                      static_cast<CoordT>(1.0 / 6.0 * Param::B + Param::C) * py3) *
                     diffY3;
        res += (static_cast<CoordT>(1.0 / 2.0 * Param::B + 2.0 * Param::C) * py0 +
                static_cast<CoordT>(2.0 * Param::B + Param::C - 3.0) * py1 +
                static_cast<CoordT>(-5.0 / 2.0 * Param::B - 2.0 * Param::C + 3.0) * py2 - Param::C * py3) *
               diffY2;
        res += (static_cast<CoordT>(-1.0 / 2.0 * Param::B - Param::C) * py0 +
                static_cast<CoordT>(1.0 / 2.0 * Param::B + Param::C) * py2) *
               diffY;
        res += static_cast<CoordT>(1.0 / 6.0 * Param::B) * py0 +
               static_cast<CoordT>(-1.0 / 3.0 * Param::B + 1.0) * py1 + static_cast<CoordT>(1.0 / 6.0 * Param::B) * py2;

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