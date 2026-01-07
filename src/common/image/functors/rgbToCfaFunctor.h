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
/// Computes an output pixel from src image with bayering
/// </summary>
template <typename DstT, typename BorderControlT, BayerGridPosition bayerGrid>
struct RGBToCFAFunctor : public ImageFunctor<false>
{
    BorderControlT BorderControl;
    using SrcT = Vector3<remove_vector_t<DstT>>;
    using T    = remove_vector_t<DstT>;

#pragma region Constructors
    RGBToCFAFunctor()
    {
    }

    RGBToCFAFunctor(BorderControlT aBorderControl) : BorderControl(aBorderControl)
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
            aDst[0].x = BorderControl(aPixelX + 0, aPixelY + 0).z;
            aDst[1].x = BorderControl(aPixelX + 1, aPixelY + 0).y;
            aDst[2].x = BorderControl(aPixelX + 0, aPixelY + 1).y;
            aDst[3].x = BorderControl(aPixelX + 1, aPixelY + 1).x;
        }
        if constexpr (bayerGrid == BayerGridPosition::GBRG)
        {
            aDst[0].x = BorderControl(aPixelX + 0, aPixelY + 0).y;
            aDst[1].x = BorderControl(aPixelX + 1, aPixelY + 0).z;
            aDst[2].x = BorderControl(aPixelX + 0, aPixelY + 1).x;
            aDst[3].x = BorderControl(aPixelX + 1, aPixelY + 1).y;
        }
        if constexpr (bayerGrid == BayerGridPosition::GRBG)
        {
            aDst[0].x = BorderControl(aPixelX + 0, aPixelY + 0).y;
            aDst[1].x = BorderControl(aPixelX + 1, aPixelY + 0).x;
            aDst[2].x = BorderControl(aPixelX + 0, aPixelY + 1).z;
            aDst[3].x = BorderControl(aPixelX + 1, aPixelY + 1).y;
        }
        if constexpr (bayerGrid == BayerGridPosition::RGGB)
        {
            aDst[0].x = BorderControl(aPixelX + 0, aPixelY + 0).x;
            aDst[1].x = BorderControl(aPixelX + 1, aPixelY + 0).y;
            aDst[2].x = BorderControl(aPixelX + 0, aPixelY + 1).y;
            aDst[3].x = BorderControl(aPixelX + 1, aPixelY + 1).z;
        }

        return true;
    }

#pragma endregion
};
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
