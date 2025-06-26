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
#include <common/vector_typetraits.h>
#include <common/vectorTypes.h>
#include <concepts>

// disable warning for pragma unroll when compiling with host compiler:
#include <common/disableWarningsBegin.h>

namespace mpp::image
{
/// <summary>
/// Converts a src pixel of type SrcT to a dst pixel of type DstT pixel
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename DstT, RoundingMode roundingMode, typename operation_SIMD = voidType>
struct ConvertFunctor : public ImageFunctor<false>
{
    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    [[no_unique_address]] operation_SIMD OpSIMD;
    [[no_unique_address]] RoundFunctor<roundingMode, SrcT> round;

#pragma region Constructors
    ConvertFunctor()
    {
    }

    ConvertFunctor(const SrcT *aSrc1, size_t aSrcPitch1) : Src1(aSrc1), SrcPitch1(aSrcPitch1)
    {
    }

    ConvertFunctor(const SrcT *aSrc1, size_t aSrcPitch1, operation_SIMD aOpSIMD)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), OpSIMD(aOpSIMD)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    /// <summary>
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        SrcT pixelSrc         = *pixelSrc1;
        if constexpr (RealOrComplexFloatingPoint<pixel_basetype_t<SrcT>> &&
                      RealOrComplexIntegral<pixel_basetype_t<DstT>>)
        {
            round(pixelSrc);
        }
        aDst = static_cast<DstT>(pixelSrc);
        return true;
    }

    /// <summary>
    /// For float32 to half-float16 or bfloat16 (real and complex numbers), we'll use the special constructor including
    /// the rounding mode to get the correct rounding-conversion.
    /// Returns true if the value has been successfully set
    /// </summary>
    DEVICE_CODE bool operator()(int aPixelX, int aPixelY, DstT &aDst) const
        requires((std::same_as<pixel_basetype_t<SrcT>, float> &&
                  (std::same_as<pixel_basetype_t<DstT>, BFloat16> || std::same_as<pixel_basetype_t<DstT>, HalfFp16>)) ||
                 (std::same_as<pixel_basetype_t<SrcT>, Complex<float>> &&
                  (std::same_as<pixel_basetype_t<DstT>, Complex<BFloat16>> ||
                   std::same_as<pixel_basetype_t<DstT>, Complex<HalfFp16>>))) &&
                (roundingMode != RoundingMode::NearestTiesToEven)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        aDst                  = DstT(*pixelSrc1, roundingMode);
        return true;
    }
#pragma endregion

#pragma region run SIMD on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires std::same_as<SrcT, Vector1<float>> &&
                 (std::same_as<DstT, Vector1<BFloat16>> || std::same_as<DstT, Vector1<HalfFp16>>) &&
                 (roundingMode == RoundingMode::NearestTiesToEven) && (!std::same_as<operation_SIMD, voidType>)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

        OpSIMD(tupelSrc1, aDst);
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            if constexpr (RealOrComplexFloatingPoint<pixel_basetype_t<SrcT>> &&
                          RealOrComplexIntegral<pixel_basetype_t<DstT>>)
            {
                round(tupelSrc1.value[i]);
            }
            aDst.value[i] = static_cast<DstT>(tupelSrc1.value[i]);
        }
    }

    /// <summary>
    /// For float32 to half-float16 or bfloat16 (real and complex numbers), we'll use the special constructor including
    /// the rounding mode to get the correct rounding-conversion.
    /// </summary>
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst) const
        requires((std::same_as<pixel_basetype_t<SrcT>, float> &&
                  (std::same_as<pixel_basetype_t<DstT>, BFloat16> || std::same_as<pixel_basetype_t<DstT>, HalfFp16>)) ||
                 (std::same_as<pixel_basetype_t<SrcT>, Complex<float>> &&
                  (std::same_as<pixel_basetype_t<DstT>, Complex<BFloat16>> ||
                   std::same_as<pixel_basetype_t<DstT>, Complex<HalfFp16>>))) &&
                (roundingMode != RoundingMode::NearestTiesToEven)

    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            aDst.value[i] = DstT(tupelSrc1.value[i], roundingMode);
        }
    }
#pragma endregion
};
} // namespace mpp::image
#include <common/disableWarningsEnd.h>
