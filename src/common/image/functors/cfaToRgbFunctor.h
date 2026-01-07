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
#include <common/utilities.h>
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace mpp::image
{
/// <summary>
/// Computes an output pixel from src image with debayering
/// </summary>
template <typename DstT, typename BorderControlT, typename Operation, BayerGridPosition bayerGrid>
struct CFAToRGBFunctor : public ImageFunctor<false>
{
    template <typename T> struct GetComputeT
    {
        using type = uint;
    };
    template <RealSignedIntegral T> struct GetComputeT<T>
    {
        using type = int;
    };
    template <RealFloatingPoint T> struct GetComputeT<T>
    {
        using type = float;
    };

    BorderControlT BorderControl;
    [[no_unique_address]] Operation op;
    using SrcT = Vector1<remove_vector_t<DstT>>;
    using T    = remove_vector_t<DstT>;
    using CT   = typename GetComputeT<T>::type;

#pragma region Constructors
    CFAToRGBFunctor()
    {
    }

    CFAToRGBFunctor(BorderControlT aBorderControl, Operation aOp) : BorderControl(aBorderControl), op(aOp)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    /// <summary>
    /// Returns true if the value has been successfully set, false otherwise
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT aDst[4]) const
    {
        // aPixelX and aPixelY are the coordinate of the upper left corner of a 2x2 pixel block

        if constexpr (bayerGrid == BayerGridPosition::BGGR)
        {
            aDst[0].z = BorderControl(aPixelX + 0, aPixelY + 0).x;
            aDst[1].y = BorderControl(aPixelX + 1, aPixelY + 0).x;
            aDst[2].y = BorderControl(aPixelX + 0, aPixelY + 1).x;
            aDst[3].x = BorderControl(aPixelX + 1, aPixelY + 1).x;

            // for the / 4 cases, NPP returns some weird values, looks more like twice / 2 with rounding down each time.
            // for all divisions we use division with round away from zero, i.e. we add half of the divisor.
            // the results thus differs by up to 1 from NPP, but should be more "correct"
            aDst[0].x = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY - 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY - 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x),
                                           static_cast<CT>(4)));

            aDst[1].x = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 1, aPixelY - 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x),
                                           static_cast<CT>(2)));

            aDst[2].x = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x),
                                           static_cast<CT>(2)));

            aDst[1].z = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 0).x),
                                           static_cast<CT>(2)));

            aDst[2].z = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 2).x),
                                           static_cast<CT>(2)));

            aDst[3].z = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 2).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 2).x),
                                           static_cast<CT>(4)));

            float bdx = (static_cast<float>(BorderControl(aPixelX - 2, aPixelY + 0).x) +
                         static_cast<float>(BorderControl(aPixelX + 2, aPixelY + 0).x)) /
                            2.0f -
                        static_cast<float>(aDst[0].z);

            float bdy = (static_cast<float>(BorderControl(aPixelX + 0, aPixelY - 2).x) +
                         static_cast<float>(BorderControl(aPixelX + 0, aPixelY + 2).x)) /
                            2.0f -
                        static_cast<float>(aDst[0].z);

            bdx = abs(bdx);
            bdy = abs(bdy);

            if (bdx < bdy)
            {
                aDst[0].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x),
                                               static_cast<CT>(2)));
            }
            else if (bdx > bdy)
            {
                aDst[0].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY - 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x),
                                               static_cast<CT>(2)));
            }
            else
            {
                aDst[0].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 0, aPixelY - 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x),
                                               static_cast<CT>(4)));
            }

            float rdx = (static_cast<float>(BorderControl(aPixelX - 1, aPixelY + 1).x) +
                         static_cast<float>(BorderControl(aPixelX + 3, aPixelY + 1).x)) /
                            2.0f -
                        static_cast<float>(aDst[3].x);

            float rdy = (static_cast<float>(BorderControl(aPixelX + 1, aPixelY - 1).x) +
                         static_cast<float>(BorderControl(aPixelX + 1, aPixelY + 3).x)) /
                            2.0f -
                        static_cast<float>(aDst[3].x);

            rdx = abs(rdx);
            rdy = abs(rdy);

            if (rdx < rdy)
            {
                aDst[3].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 1).x),
                                               static_cast<CT>(2)));
            }
            else if (rdx > rdy)
            {
                aDst[3].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 2).x),
                                               static_cast<CT>(2)));
            }
            else
            {
                aDst[3].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 2).x),
                                               static_cast<CT>(4)));
            }
        }
        if constexpr (bayerGrid == BayerGridPosition::GBRG)
        {
            aDst[0].y = BorderControl(aPixelX + 0, aPixelY + 0).x;
            aDst[1].z = BorderControl(aPixelX + 1, aPixelY + 0).x;
            aDst[2].x = BorderControl(aPixelX + 0, aPixelY + 1).x;
            aDst[3].y = BorderControl(aPixelX + 1, aPixelY + 1).x;

            aDst[2].z = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 2).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 2).x),
                                           static_cast<CT>(4)));

            aDst[0].z = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x),
                                           static_cast<CT>(2)));

            aDst[3].z = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 2).x),
                                           static_cast<CT>(2)));

            aDst[0].x = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY - 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x),
                                           static_cast<CT>(2)));

            aDst[3].x = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 1).x),
                                           static_cast<CT>(2)));

            aDst[1].x = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY - 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 2, aPixelY - 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 1).x),
                                           static_cast<CT>(4)));

            float rdx = (static_cast<float>(BorderControl(aPixelX - 2, aPixelY + 1).x) +
                         static_cast<float>(BorderControl(aPixelX + 2, aPixelY + 1).x)) /
                            2.0f -
                        static_cast<float>(aDst[2].x);

            float rdy = (static_cast<float>(BorderControl(aPixelX + 0, aPixelY - 1).x) +
                         static_cast<float>(BorderControl(aPixelX + 0, aPixelY + 3).x)) /
                            2.0f -
                        static_cast<float>(aDst[2].x);

            rdx = abs(rdx);
            rdy = abs(rdy);

            if (rdx < rdy)
            {
                aDst[2].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x),
                                               static_cast<CT>(2)));
            }
            else if (rdx > rdy)
            {
                aDst[2].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 2).x),
                                               static_cast<CT>(2)));
            }
            else
            {
                aDst[2].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 2).x),
                                               static_cast<CT>(4)));
            }

            float bdx = (static_cast<float>(BorderControl(aPixelX - 1, aPixelY + 0).x) +
                         static_cast<float>(BorderControl(aPixelX + 3, aPixelY + 0).x)) /
                            2.0f -
                        static_cast<float>(aDst[1].z);

            float bdy = (static_cast<float>(BorderControl(aPixelX + 1, aPixelY - 2).x) +
                         static_cast<float>(BorderControl(aPixelX + 1, aPixelY + 2).x)) /
                            2.0f -
                        static_cast<float>(aDst[1].z);

            bdx = abs(bdx);
            bdy = abs(bdy);

            if (bdx < bdy)
            {
                aDst[1].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 0).x),
                                               static_cast<CT>(2)));
            }
            else if (bdx > bdy)
            {
                aDst[1].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 1, aPixelY - 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x),
                                               static_cast<CT>(2)));
            }
            else
            {
                aDst[1].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY - 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x),
                                               static_cast<CT>(4)));
            }
        }
        if constexpr (bayerGrid == BayerGridPosition::GRBG)
        {
            aDst[0].y = BorderControl(aPixelX + 0, aPixelY + 0).x;
            aDst[1].x = BorderControl(aPixelX + 1, aPixelY + 0).x;
            aDst[2].z = BorderControl(aPixelX + 0, aPixelY + 1).x;
            aDst[3].y = BorderControl(aPixelX + 1, aPixelY + 1).x;

            aDst[2].x = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 2).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 2).x),
                                           static_cast<CT>(4)));

            aDst[0].x = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x),
                                           static_cast<CT>(2)));

            aDst[3].x = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 2).x),
                                           static_cast<CT>(2)));

            aDst[0].z = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY - 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x),
                                           static_cast<CT>(2)));

            aDst[3].z = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 1).x),
                                           static_cast<CT>(2)));

            aDst[1].z = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY - 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 2, aPixelY - 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 1).x),
                                           static_cast<CT>(4)));

            float bdx = (static_cast<float>(BorderControl(aPixelX - 2, aPixelY + 1).x) +
                         static_cast<float>(BorderControl(aPixelX + 2, aPixelY + 1).x)) /
                            2.0f -
                        static_cast<float>(aDst[2].z);

            float bdy = (static_cast<float>(BorderControl(aPixelX + 0, aPixelY - 1).x) +
                         static_cast<float>(BorderControl(aPixelX + 0, aPixelY + 3).x)) /
                            2.0f -
                        static_cast<float>(aDst[2].z);

            bdx = abs(bdx);
            bdy = abs(bdy);

            if (bdx < bdy)
            {
                aDst[2].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x),
                                               static_cast<CT>(2)));
            }
            else if (bdx > bdy)
            {
                aDst[2].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 2).x),
                                               static_cast<CT>(2)));
            }
            else
            {
                aDst[2].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 2).x),
                                               static_cast<CT>(4)));
            }

            float rdx = (static_cast<float>(BorderControl(aPixelX - 1, aPixelY + 0).x) +
                         static_cast<float>(BorderControl(aPixelX + 3, aPixelY + 0).x)) /
                            2.0f -
                        static_cast<float>(aDst[1].x);

            float rdy = (static_cast<float>(BorderControl(aPixelX + 1, aPixelY - 2).x) +
                         static_cast<float>(BorderControl(aPixelX + 1, aPixelY + 2).x)) /
                            2.0f -
                        static_cast<float>(aDst[1].x);

            rdx = abs(rdx);
            rdy = abs(rdy);

            if (rdx < rdy)
            {
                aDst[1].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 0).x),
                                               static_cast<CT>(2)));
            }
            else if (rdx > rdy)
            {
                aDst[1].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 1, aPixelY - 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x),
                                               static_cast<CT>(2)));
            }
            else
            {
                aDst[1].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY - 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x),
                                               static_cast<CT>(4)));
            }
        }
        if constexpr (bayerGrid == BayerGridPosition::RGGB)
        {
            aDst[0].x = BorderControl(aPixelX + 0, aPixelY + 0).x;
            aDst[1].y = BorderControl(aPixelX + 1, aPixelY + 0).x;
            aDst[2].y = BorderControl(aPixelX + 0, aPixelY + 1).x;
            aDst[3].z = BorderControl(aPixelX + 1, aPixelY + 1).x;

            aDst[0].z = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY - 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY - 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x),
                                           static_cast<CT>(4)));

            aDst[1].z = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 1, aPixelY - 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x),
                                           static_cast<CT>(2)));

            aDst[2].z = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 1).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 1).x),
                                           static_cast<CT>(2)));

            aDst[1].x = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 0).x),
                                           static_cast<CT>(2)));

            aDst[2].x = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 2).x),
                                           static_cast<CT>(2)));

            aDst[3].x = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 0).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 2).x) +
                                               static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 2).x),
                                           static_cast<CT>(4)));

            float rdx = (static_cast<float>(BorderControl(aPixelX - 2, aPixelY + 0).x) +
                         static_cast<float>(BorderControl(aPixelX + 2, aPixelY + 0).x)) /
                            2.0f -
                        static_cast<float>(aDst[0].x);

            float rdy = (static_cast<float>(BorderControl(aPixelX + 0, aPixelY - 2).x) +
                         static_cast<float>(BorderControl(aPixelX + 0, aPixelY + 2).x)) /
                            2.0f -
                        static_cast<float>(aDst[0].x);

            rdx = abs(rdx);
            rdy = abs(rdy);

            if (rdx < rdy)
            {
                aDst[0].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x),
                                               static_cast<CT>(2)));
            }
            else if (rdx > rdy)
            {
                aDst[0].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY - 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x),
                                               static_cast<CT>(2)));
            }
            else
            {
                aDst[0].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX - 1, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 0, aPixelY - 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x),
                                               static_cast<CT>(4)));
            }

            float bdx = (static_cast<float>(BorderControl(aPixelX - 1, aPixelY + 1).x) +
                         static_cast<float>(BorderControl(aPixelX + 3, aPixelY + 1).x)) /
                            2.0f -
                        static_cast<float>(aDst[3].z);

            float bdy = (static_cast<float>(BorderControl(aPixelX + 1, aPixelY - 1).x) +
                         static_cast<float>(BorderControl(aPixelX + 1, aPixelY + 3).x)) /
                            2.0f -
                        static_cast<float>(aDst[3].z);

            bdx = abs(bdx);
            bdy = abs(bdy);

            if (bdx < bdy)
            {
                aDst[3].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 1).x),
                                               static_cast<CT>(2)));
            }
            else if (bdx > bdy)
            {
                aDst[3].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 2).x),
                                               static_cast<CT>(2)));
            }
            else
            {
                aDst[3].y = static_cast<T>(Div(static_cast<CT>(BorderControl(aPixelX + 0, aPixelY + 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 2, aPixelY + 1).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 0).x) +
                                                   static_cast<CT>(BorderControl(aPixelX + 1, aPixelY + 2).x),
                                               static_cast<CT>(4)));
            }
        }

        op(aDst[0]);
        op(aDst[1]);
        op(aDst[2]);
        op(aDst[3]);
        return true;
    }

#pragma endregion

  private:
    template <Number T>
        requires RealIntegral<T>
    DEVICE_CODE static T Div(T aSrc0, T aSrc1)
    {
        return DivScaleRoundTiesAwayFromZero(aSrc0, aSrc1);
    }
    template <Number T>
        requires RealFloatingPoint<T>
    DEVICE_CODE static T Div(T aSrc0, T aSrc1)
    {
        return aSrc0 / aSrc1;
    }
};
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
