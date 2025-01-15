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
/// Computes an output pixel from one srcDst pixel and one constant value -> srcDst pixel inplace
/// </summary>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename ComputeT, typename DstT, typename operation,
          RoudingMode roundingMode = RoudingMode::NearestTiesAwayFromZero, typename ComputeT_SIMD = voidType,
          typename operation_SIMD = voidType>
struct InplaceConstantFunctor : public ImageFunctor<true>
{
    ComputeT Constant;
    ComputeT_SIMD ConstantSIMD;

    [[no_unique_address]] operation Op;
    [[no_unique_address]] operation_SIMD OpSIMD;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

#pragma region Constructors
    InplaceConstantFunctor()
    {
    }

    InplaceConstantFunctor(ComputeT aConstant, operation aOp) : Constant(aConstant), Op(aOp)
    {
    }

    InplaceConstantFunctor(ComputeT aConstant, operation aOp, ComputeT_SIMD aConstantSIMD, operation_SIMD aOpSIMD)
        : Constant(aConstant), ConstantSIMD(aConstantSIMD), Op(aOp), OpSIMD(aOpSIMD)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires std::same_as<ComputeT, DstT>
    {
        Op(Constant, aDst);
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires(!std::same_as<ComputeT, DstT>)
    {
        ComputeT temp(aDst);
        Op(Constant, temp);
        round(temp); // NOP for integer ComputeT
        // DstT constructor will clamp temp to value range of DstT
        aDst = static_cast<DstT>(temp);
    }
#pragma endregion

#pragma region run SIMD on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires std::same_as<ComputeT_SIMD, Tupel<DstT, tupelSize>>
    {
        static_assert(OpSIMD.has_simd, "Trying to run a SIMD operation that is not implemented for this type.");
        OpSIMD(ConstantSIMD, aDst);
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires std::same_as<ComputeT, DstT> && //
                 std::same_as<ComputeT_SIMD, voidType>
    {
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(Constant, aDst.value[i]);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires(!std::same_as<ComputeT, DstT>) && //
                std::same_as<ComputeT_SIMD, voidType>
    {
#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp(aDst.value[i]);
            Op(Constant, temp);
            round(temp); // NOP for integer ComputeT
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }
#pragma endregion
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
