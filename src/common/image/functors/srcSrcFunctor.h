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
/// Computes an output pixel from two src arrays -&gt; dst pixel
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename ComputeT, typename DstT, typename operation,
          RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero, typename ComputeT_SIMD = voidType,
          typename operation_SIMD = voidType>
struct SrcSrcFunctor : public ImageFunctor<false>
{
    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    const SrcT *RESTRICT Src2;
    size_t SrcPitch2;

    [[no_unique_address]] operation Op;
    [[no_unique_address]] operation_SIMD OpSIMD;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

#pragma region Constructors
    SrcSrcFunctor()
    {
    }

    SrcSrcFunctor(const SrcT *aSrc1, size_t aSrcPitch1, const SrcT *aSrc2, size_t aSrcPitch2, operation aOp)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), Src2(aSrc2), SrcPitch2(aSrcPitch2), Op(aOp)
    {
    }

    SrcSrcFunctor(const SrcT *aSrc1, size_t aSrcPitch1, const SrcT *aSrc2, size_t aSrcPitch2, operation aOp,
                  operation_SIMD aOpSIMD)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), Src2(aSrc2), SrcPitch2(aSrcPitch2), Op(aOp), OpSIMD(aOpSIMD)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    // Case ComputeT==DstT, no conversion, no rounding
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires std::same_as<ComputeT, DstT>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcT *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);
        Op(static_cast<ComputeT>(*pixelSrc1), static_cast<ComputeT>(*pixelSrc2), aDst);
    }

    // Case ComputeT==Floating, DstT==Integral, conversion and rounding of result
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires(!std::same_as<ComputeT, DstT>) && (!(RealVector<ComputeT> && ComplexVector<DstT>))
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcT *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);

        ComputeT temp;
        Op(static_cast<ComputeT>(*pixelSrc1), static_cast<ComputeT>(*pixelSrc2), temp);
        round(temp); // NOP for integer ComputeT
        // DstT constructor will clamp temp to value range of DstT
        aDst = static_cast<DstT>(temp);
    }

    // Needed for the conversion real->Complex
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires(!std::same_as<ComputeT, DstT>) && ((RealVector<ComputeT> && ComplexVector<DstT>))
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcT *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);
        Op(*pixelSrc1, *pixelSrc2, aDst);
    }
#pragma endregion

#pragma region run SIMD on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires std::same_as<ComputeT_SIMD, DstT>
    {
        static_assert(OpSIMD.has_simd, "Trying to run a SIMD operation that is not implemented for this type.");
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcT *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);
        Tupel<SrcT, tupelSize> tupelSrc2 = Tupel<SrcT, tupelSize>::Load(pixelSrc2);

        OpSIMD(tupelSrc1, tupelSrc2, aDst);
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires std::same_as<ComputeT, DstT> && //
                 std::same_as<ComputeT_SIMD, voidType>
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcT *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);
        Tupel<SrcT, tupelSize> tupelSrc2 = Tupel<SrcT, tupelSize>::Load(pixelSrc2);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(static_cast<ComputeT>(tupelSrc1.value[i]), static_cast<ComputeT>(tupelSrc2.value[i]), aDst.value[i]);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires(!std::same_as<ComputeT, DstT>) && //
                std::same_as<ComputeT_SIMD, voidType> && (!(RealVector<ComputeT> && ComplexVector<DstT>))
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcT *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);
        Tupel<SrcT, tupelSize> tupelSrc2 = Tupel<SrcT, tupelSize>::Load(pixelSrc2);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp;
            Op(static_cast<ComputeT>(tupelSrc1.value[i]), static_cast<ComputeT>(tupelSrc2.value[i]), temp);
            round(temp); // NOP for integer ComputeT
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }

    // Needed for the conversion real->Complex
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
        requires(!std::same_as<ComputeT, DstT>) && //
                std::same_as<ComputeT_SIMD, voidType> && ((RealVector<ComputeT> && ComplexVector<DstT>))
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcT *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);
        Tupel<SrcT, tupelSize> tupelSrc2 = Tupel<SrcT, tupelSize>::Load(pixelSrc2);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(tupelSrc1.value[i], tupelSrc2.value[i], aDst.value[i]);
        }
    }
#pragma endregion
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
