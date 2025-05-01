#pragma once
#include "imageFunctors.h"
#include <common/arithmetic/binary_operators.h>
#include <common/arithmetic/unary_operators.h>
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
/// Computes an output pixel from one device memory constant value -&gt; srcDst pixel (set function)
/// </summary>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
template <size_t tupelSize, typename DstT> struct DevConstantFunctor : public ImageFunctor<false>
{
    const DstT *RESTRICT Constant;

#pragma region Constructors
    DevConstantFunctor()
    {
    }

    DevConstantFunctor(const DstT *aConstant) : Constant(aConstant)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int /*aPixelX*/, int /*aPixelY*/, DstT &aDst) const
    {
        aDst = *Constant;
        return true;
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int /*aPixelX*/, int /*aPixelY*/, Tupel<DstT, tupelSize> &aDst) const
    {
        DstT _constant = *Constant;
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            aDst.value[i] = _constant;
        }
    }
#pragma endregion
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>