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
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace opp::image
{
/// <summary>
/// Computes an output pixel from one srcDst pixel and one device memory constant value -&gt; srcDst pixel inplace with
/// float scaling of result
/// </summary>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename ComputeT, typename DstT, typename operation,
          RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero>
struct InplaceDevConstantScaleFunctor : public ImageFunctor<true>
{
    const DstT *RESTRICT Constant;

    scalefactor_t<ComputeT> ScaleFactor;

    [[no_unique_address]] operation Op;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

#pragma region Constructors
    InplaceDevConstantScaleFunctor()
    {
    }

    InplaceDevConstantScaleFunctor(const DstT *aConstant, operation aOp, scalefactor_t<ComputeT> aScaleFactor)
        : Constant(aConstant), ScaleFactor(aScaleFactor), Op(aOp)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires RealOrComplexIntegral<pixel_basetype_t<DstT>> && //
                 RealOrComplexFloatingPoint<pixel_basetype_t<ComputeT>>
    {
        ComputeT temp(aDst);
        Op(static_cast<ComputeT>(*Constant), temp);
        if constexpr (ComplexVector<ComputeT>)
        {
            temp = temp * ScaleFactor;
        }
        else
        {
            temp *= ScaleFactor;
        }
        round(temp);
        // DstT constructor will clamp temp to value range of DstT
        aDst = static_cast<DstT>(temp);
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires RealOrComplexIntegral<pixel_basetype_t<DstT>> && //
                 RealOrComplexFloatingPoint<pixel_basetype_t<ComputeT>>
    {
        ComputeT _constant = static_cast<ComputeT>(*Constant);
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp(aDst.value[i]);
            Op(_constant, temp);
            if constexpr (ComplexVector<ComputeT>)
            {
                temp = temp * ScaleFactor;
            }
            else
            {
                temp *= ScaleFactor;
            }
            round(temp);
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }
#pragma endregion
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
