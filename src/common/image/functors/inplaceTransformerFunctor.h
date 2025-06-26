#pragma once
#include "borderControl.h"
#include "imageFunctors.h"
#include "transformer.h"
#include <common/defines.h>
#include <common/image/gotoPtr.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
#include <common/mpp_defs.h>
#include <common/roundFunctor.h>
#include <common/tupel.h>
#include <common/vectorTypes.h>
#include <common/vector_typetraits.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace mpp::image
{
/// <summary>
/// Computes an output pixel from src image with geometric coordinate transformation (inplace)
/// </summary>
template <typename DstT, typename BorderControlT, typename TransformerT>
struct InplaceTransformerFunctor : public ImageFunctor<false>
{
    BorderControlT BorderControl;
    TransformerT Transformer;
    Vector2<int> SourceRoi;

#pragma region Constructors
    InplaceTransformerFunctor()
    {
    }

    InplaceTransformerFunctor(BorderControlT aBorderControl, TransformerT aTransformer, const Size2D &aSourceRoi)
        : BorderControl(aBorderControl), Transformer(aTransformer), SourceRoi(aSourceRoi)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    /// <summary>
    /// Returns true if the value has been successfully set, false otherwise
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, int &aPixelOutX, int &aPixelOutY) const
    {
        Vector2<int> coordTransformed = Transformer(aPixelX, aPixelY);

        BorderControl.AdjustCoordinates(coordTransformed.x, coordTransformed.y);

        aPixelOutX = coordTransformed.x;
        aPixelOutY = coordTransformed.y;
        return true;
    }

#pragma endregion
};
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
