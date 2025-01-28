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
template <RealOrComplexVector TVector> struct convert_scale_compute_type
{
};
template <RealVector TVector> struct convert_scale_compute_type<TVector>
{
    using type = same_vector_size_different_type<TVector, float>::type;
};
template <ComplexVector TVector> struct convert_scale_compute_type<TVector>
{
    using type = same_vector_size_different_type<TVector, Complex<float>>::type;
};
template <RealVector TVector>
    requires(std::same_as<remove_vector_t<TVector>, double>)
struct convert_scale_compute_type<TVector>
{
    using type = same_vector_size_different_type<TVector, double>::type;
};
template <RealVector TVector>
    requires(std::same_as<remove_vector_t<TVector>, Complex<double>>)
struct convert_scale_compute_type<TVector>
{
    using type = same_vector_size_different_type<TVector, Complex<double>>::type;
};

template <RealOrComplexVector TVector>
using convert_scale_compute_type_t = typename convert_scale_compute_type<TVector>::type;

/// <summary>
/// Converts a src pixel of type SrcT to float, then scales by scaleFactor and then converts to a dst pixel of type DstT
/// pixel
/// </summary>
/// <typeparam name="SrcT"></typeparam>
/// <typeparam name="DstT"></typeparam>
/// <typeparam name="tupelSize"></typeparam>
/// <typeparam name="roundingMode"></typeparam>
template <size_t tupelSize, typename SrcT, typename DstT, RoundingMode roundingMode>
struct ConvertScaleFunctor : public ImageFunctor<false>
{
    using ComputeT = convert_scale_compute_type_t<SrcT>;

    const SrcT *RESTRICT Src1;
    size_t SrcPitch1;

    scalefactor_t<ComputeT> ScaleFactor;

    [[no_unique_address]] RoundFunctor<roundingMode, ComputeT> round;

#pragma region Constructors
    ConvertScaleFunctor()
    {
    }

    ConvertScaleFunctor(const SrcT *aSrc1, size_t aSrcPitch1, scalefactor_t<ComputeT> aScaleFactor)
        : Src1(aSrc1), SrcPitch1(aSrcPitch1), ScaleFactor(aScaleFactor)
    {
    }
#pragma endregion

#pragma region run naive on one pixel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        ComputeT temp         = static_cast<ComputeT>(*pixelSrc1);
        temp *= ScaleFactor;
        if constexpr (RealOrComplexFloatingPoint<pixel_basetype_t<ComputeT>> &&
                      RealOrComplexIntegral<pixel_basetype_t<DstT>>)
        {
            round(temp);
        }
        aDst = static_cast<DstT>(temp);
    }
    /// <summary>
    /// For float32 to half-float16 or bfloat16 (real and complex numbers), we'll use the special constructor including
    /// the rounding mode to get the correct rounding-conversion.
    /// </summary>
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, DstT &aDst)
        requires((std::same_as<pixel_basetype_t<SrcT>, float> &&
                  (std::same_as<pixel_basetype_t<DstT>, BFloat16> || std::same_as<pixel_basetype_t<DstT>, HalfFp16>)) ||
                 (std::same_as<pixel_basetype_t<SrcT>, Complex<float>> &&
                  (std::same_as<pixel_basetype_t<DstT>, Complex<BFloat16>> ||
                   std::same_as<pixel_basetype_t<DstT>, Complex<HalfFp16>>))) &&
                (roundingMode != RoundingMode::NearestTiesToEven)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);
        ComputeT temp         = *pixelSrc1;
        temp *= ScaleFactor;
        aDst = DstT(temp, roundingMode);
    }
#pragma endregion

#pragma region run sequential on pixel tupel
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
    {
        const SrcT *pixelSrc1 = gotoPtr(Src1, SrcPitch1, aPixelX, aPixelY);

        Tupel<SrcT, tupelSize> tupelSrc1 = Tupel<SrcT, tupelSize>::Load(pixelSrc1);

#pragma unroll
        for (size_t i = 0; i < tupelSize; i++)
        {
            ComputeT temp = static_cast<ComputeT>(tupelSrc1.value[i]);
            temp *= ScaleFactor;
            if constexpr (RealOrComplexFloatingPoint<pixel_basetype_t<SrcT>> &&
                          RealOrComplexIntegral<pixel_basetype_t<DstT>>)
            {
                round(temp);
            }
            aDst.value[i] = static_cast<DstT>(temp);
        }
    }

    /// <summary>
    /// For float32 to half-float16 or bfloat16 (real and complex numbers), we'll use the special constructor including
    /// the rounding mode to get the correct rounding-conversion.
    /// </summary>
    DEVICE_CODE void operator()(int aPixelX, int aPixelY, Tupel<DstT, tupelSize> &aDst)
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
            tupelSrc1.value[i] *= ScaleFactor;
            aDst.value[i] = DstT(tupelSrc1.value[i], roundingMode);
        }
    }
#pragma endregion
};
} // namespace opp::image
#include <common/disableWarningsEnd.h>
