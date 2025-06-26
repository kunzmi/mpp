#pragma once
#include "imageFunctors.h"
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
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
/// Computes an output pixel from one srcDst pixel -&gt; srcDst pixel inplace with float scaling of result
/// </summary>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename ComputeT, typename DstT, typename operation,
          RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero>
struct InplaceScaleFunctor : public ImageFunctor<true>
{
    scalefactor_t<ComputeT> ScaleFactor;

    [[no_unique_address]] operation Op;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

#pragma region Constructors
    InplaceScaleFunctor()
    {
    }

    InplaceScaleFunctor(operation aOp, scalefactor_t<ComputeT> aScaleFactor) : ScaleFactor(aScaleFactor), Op(aOp)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int /*aPixelX*/, int /*aPixelY*/, DstT &aDst) const
        requires RealOrComplexIntegral<pixel_basetype_t<DstT>> && //
                 RealOrComplexFloatingPoint<pixel_basetype_t<ComputeT>>
    {
        ComputeT temp(aDst);
        Op(temp);
        temp *= ScaleFactor;
        round(temp);
        // DstT constructor will clamp temp to value range of DstT
        aDst = static_cast<DstT>(temp);
        return true;
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int /*aPixelX*/, int /*aPixelY*/, Tupel<DstT, tupelSize> &aDst) const
        requires RealOrComplexIntegral<pixel_basetype_t<DstT>> && //
                 RealOrComplexFloatingPoint<pixel_basetype_t<ComputeT>>
    {
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp(aDst.value[i]);
            Op(temp);
            temp *= ScaleFactor;
            round(temp);
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }
#pragma endregion
};
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
