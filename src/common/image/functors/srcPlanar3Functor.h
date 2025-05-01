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
#include <common/vector1.h>
#include <common/vectorTypes.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace opp::image
{
/// <summary>
/// Computes an output pixel from three single channel planar src arrays -&gt; dst pixel
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="ComputeT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="operation"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename ComputeT, typename DstT, typename operation,
          RoundingMode roundingMode = RoundingMode::NearestTiesAwayFromZero>
struct SrcPlanar3Functor : public ImageFunctor<false>
{
    using SrcPlane = Vector1<remove_vector_t<SrcT>>;
    const SrcPlane *RESTRICT Src1;
    size_t SrcPitch1;
    const SrcPlane *RESTRICT Src2;
    size_t SrcPitch2;
    const SrcPlane *RESTRICT Src3;
    size_t SrcPitch3;

    [[no_unique_address]] operation Op;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

#pragma region Constructors
    SrcPlanar3Functor()
    {
    }

    SrcPlanar3Functor(const SrcPlane *aSrc1, size_t aSrcPitch1, const SrcPlane *aSrc2, size_t aSrcPitch2,
                      const SrcPlane *aSrc3, size_t aSrcPitch3, operation aOp)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), Src2(aSrc2), SrcPitch2(aSrcPitch2), Src3(aSrc3), SrcPitch3(aSrcPitch3),
          Op(aOp)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
        requires std::same_as<ComputeT, DstT>
    {
        const SrcPlane *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcPlane *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);
        const SrcPlane *pixelSrc3 = gotoPtr(Src3, SrcPitch3, aPixelX, aPixelY);
        const ComputeT pixelSrc(pixelSrc1->x, pixelSrc2->x, pixelSrc3->x);
        Op(pixelSrc, aDst);
        return true;
    }

    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
        requires(!std::same_as<ComputeT, DstT>)
    {
        const SrcPlane *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcPlane *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);
        const SrcPlane *pixelSrc3 = gotoPtr(Src3, SrcPitch3, aPixelX, aPixelY);
        const ComputeT pixelSrc(pixelSrc1->x, pixelSrc2->x, pixelSrc3->x);
        ComputeT temp;
        Op(pixelSrc, temp);
        round(temp); // NOP for integer ComputeT
        // DstT constructor will clamp temp to value range of DstT
        aDst = static_cast<DstT>(temp);
        return true;
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires std::same_as<ComputeT, DstT>
    {
        const SrcPlane *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcPlane *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);
        const SrcPlane *pixelSrc3 = gotoPtr(Src3, SrcPitch3, aPixelX, aPixelY);

        Tupel<SrcPlane, tupelSize> tupelSrc1 = Tupel<SrcPlane, tupelSize>::Load(pixelSrc1);
        Tupel<SrcPlane, tupelSize> tupelSrc2 = Tupel<SrcPlane, tupelSize>::Load(pixelSrc2);
        Tupel<SrcPlane, tupelSize> tupelSrc3 = Tupel<SrcPlane, tupelSize>::Load(pixelSrc3);

        Tupel<ComputeT, tupelSize> tupelSrc;

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            tupelSrc.value[i].x = tupelSrc1.value[i].x;
            tupelSrc.value[i].y = tupelSrc2.value[i].x;
            tupelSrc.value[i].z = tupelSrc3.value[i].x;
        }

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            Op(tupelSrc.value[i], aDst.value[i]);
        }
    }

    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires(!std::same_as<ComputeT, DstT>)
    {
        const SrcPlane *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        const SrcPlane *pixelSrc2 = gotoPtr(Src2, SrcPitch2, aPixelX, aPixelY);
        const SrcPlane *pixelSrc3 = gotoPtr(Src3, SrcPitch3, aPixelX, aPixelY);

        Tupel<SrcPlane, tupelSize> tupelSrc1 = Tupel<SrcPlane, tupelSize>::Load(pixelSrc1);
        Tupel<SrcPlane, tupelSize> tupelSrc2 = Tupel<SrcPlane, tupelSize>::Load(pixelSrc2);
        Tupel<SrcPlane, tupelSize> tupelSrc3 = Tupel<SrcPlane, tupelSize>::Load(pixelSrc3);

        Tupel<ComputeT, tupelSize> tupelSrc;

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            tupelSrc.value[i].x = tupelSrc1.value[i].x;
            tupelSrc.value[i].y = tupelSrc2.value[i].x;
            tupelSrc.value[i].z = tupelSrc3.value[i].x;
        }

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp;
            Op(tupelSrc.value[i], temp);
            round(temp); // NOP for integer ComputeT
            // DstT constructor will clamp temp to value range of DstT
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }
#pragma endregion
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
