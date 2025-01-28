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
/// Computes an output pixel from one src array and one constant value -&gt; dst pixel with float scaling of result
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename ComputeT, typename DstT, typename operation,
          RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero>
struct SrcConstantScaleFunctor : public ImageFunctor<false>
{
    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    ComputeT Constant; // is converted to ComputeT on host

    scalefactor_t<ComputeT> ScaleFactor;

    [[no_unique_address]] operation Op;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

#pragma region Constructors
    SrcConstantScaleFunctor()
    {
    }

    SrcConstantScaleFunctor(const SrcT *aSrc1, size_t aSrcPitch1, ComputeT aConstant, operation aOp,
                            scalefactor_t<ComputeT> aScaleFactor)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), Constant(aConstant), ScaleFactor(aScaleFactor), Op(aOp)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires RealOrComplexIntegral<pixel_basetype_t<DstT>> && //
                 RealOrComplexFloatingPoint<pixel_basetype_t<ComputeT>>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        ComputeT temp;
        Op(static_cast<ComputeT>(*pixelSrc1), Constant, temp);
        temp *= ScaleFactor;
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
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp;
            Op(static_cast<ComputeT>(tupelSrc1.value[i]), Constant, temp);
            temp *= ScaleFactor;
            round(temp);
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }
#pragma endregion
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
