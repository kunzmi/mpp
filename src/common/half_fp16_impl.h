#pragma once
#include "exception.h"
#include "half_fp16.h"
#include "opp_defs.h"
#include <common/defines.h>
#include <common/numeric_limits.h>
#include <concepts>
#ifdef IS_CUDA_COMPILER
#include <cuda_fp16.h>
#endif
#ifdef IS_HOST_COMPILER
#include <bit>
#include <half/half.hpp>
#include <iostream>
#endif

#ifdef IS_HOST_COMPILER
#define INLINE
#else
#define INLINE inline
#endif

// NOLINTBEGIN(misc-definitions-in-headers, performance-unnecessary-value-param)
namespace opp
{

#ifdef IS_HOST_COMPILER

INLINE HalfFp16::HalfFp16(float aFloat) : value(aFloat)
{
}
DEVICE_CODE INLINE HalfFp16::HalfFp16(float aFloat, RoundingMode aRoundingMode)
{
    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
            value = half_float::half(aFloat, std::float_round_style::round_to_nearest);
            break;
        case opp::RoundingMode::TowardZero:
            value = half_float::half(aFloat, std::float_round_style::round_toward_zero);
            break;
        case opp::RoundingMode::TowardNegativeInfinity:
            value = half_float::half(aFloat, std::float_round_style::round_toward_neg_infinity);
            break;
        case opp::RoundingMode::TowardPositiveInfinity:
            value = half_float::half(aFloat, std::float_round_style::round_toward_infinity);
            break;
        default:
            throw INVALIDARGUMENT(aRoundingMode,
                                  "Invalid rounding mode provided: "
                                      << aRoundingMode
                                      << ". Only NearestTiesToEven, TowardZero, TowardNegativeInfinity and "
                                         "TowardPositiveInfinity are supported.");
            break;
    }
}
INLINE HalfFp16::HalfFp16(double aDouble) : value(static_cast<float>(aDouble))
{
}
INLINE HalfFp16::HalfFp16(long64 aLong64) : value(static_cast<float>(aLong64))
{
}
INLINE HalfFp16::HalfFp16(ulong64 aULong64) : value(static_cast<float>(aULong64))
{
}
// INLINE HalfFp16::HalfFp16(int aInt) : value(static_cast<float>(aInt))
//{
// }
INLINE HalfFp16::HalfFp16(uint aUInt) : value(static_cast<float>(aUInt))
{
}
INLINE HalfFp16::HalfFp16(short aShort) : value(static_cast<float>(aShort))
{
}
INLINE HalfFp16::HalfFp16(ushort aUShort) : value(static_cast<float>(aUShort))
{
}
INLINE HalfFp16::HalfFp16(sbyte aSbyte) : value(static_cast<float>(aSbyte))
{
}
INLINE HalfFp16::HalfFp16(byte aByte) : value(static_cast<float>(aByte))
{
}
#endif
#ifdef IS_CUDA_COMPILER
DEVICE_CODE INLINE HalfFp16::HalfFp16(__half aHalf) : value(aHalf)
{
}
DEVICE_CODE INLINE HalfFp16::HalfFp16(float aFloat) : value(__float2half_rn(aFloat))
{
}
DEVICE_CODE INLINE HalfFp16::HalfFp16(float aFloat, RoundingMode aRoundingMode)
{
    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
            value = __float2half_rn(aFloat);
            break;
        case opp::RoundingMode::TowardZero:
            value = __float2half_rz(aFloat);
            break;
        case opp::RoundingMode::TowardNegativeInfinity:
            value = __float2half_rd(aFloat);
            break;
        case opp::RoundingMode::TowardPositiveInfinity:
            value = __float2half_ru(aFloat);
            break;
        default:
            // other rounding modes are not supported and must be catched in host code...
            break;
    }
}
DEVICE_CODE INLINE HalfFp16::HalfFp16(double aDouble) : value(__double2half(aDouble))
{
}
DEVICE_CODE INLINE HalfFp16::HalfFp16(int aInt) : value(__int2half_rn(aInt))
{
}
DEVICE_CODE INLINE HalfFp16::HalfFp16(uint aUInt) : value(__uint2half_rn(aUInt))
{
}
DEVICE_CODE INLINE HalfFp16::HalfFp16(short aShort) : value(__short2half_rn(aShort))
{
}
DEVICE_CODE INLINE HalfFp16::HalfFp16(ushort aUShort) : value(__ushort2half_rn(aUShort))
{
}
DEVICE_CODE INLINE HalfFp16::HalfFp16(sbyte aSbyte) : value(__short2half_rn(aSbyte))
{
}
DEVICE_CODE INLINE HalfFp16::HalfFp16(byte aByte) : value(__ushort2half_rn(aByte))
{
}
#endif

DEVICE_CODE INLINE HalfFp16::operator float() const
{
#ifdef IS_HOST_COMPILER
    return static_cast<float>(value);
#endif
#ifdef IS_CUDA_COMPILER
    return __half2float(value);
#endif
}

#ifdef IS_CUDA_COMPILER
DEVICE_CODE INLINE HalfFp16::operator __half() const
{
    return value;
}
DEVICE_CODE INLINE HalfFp16::operator int() const
{
    return __half2int_rz(value);
}
DEVICE_CODE INLINE HalfFp16::operator uint() const
{
    return __half2uint_rz(value);
}
DEVICE_CODE INLINE HalfFp16::operator short() const
{
    return __half2short_rz(value);
}
DEVICE_CODE INLINE HalfFp16::operator ushort() const
{
    return __half2ushort_rz(value);
}
DEVICE_CODE INLINE HalfFp16::operator byte() const
{
    return __half2uchar_rz(value);
}
DEVICE_CODE INLINE HalfFp16::operator sbyte() const
{
    return __half2char_rz(value);
}
DEVICE_CODE INLINE HalfFp16::operator double() const
{
    return __half2float(value);
}
#endif

/// <summary>
/// </summary>
DEVICE_CODE INLINE bool HalfFp16::operator<(HalfFp16 aOther) const
{
    return value < aOther.value;
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE bool HalfFp16::operator<=(HalfFp16 aOther) const
{
    return value <= aOther.value;
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE bool HalfFp16::operator>(HalfFp16 aOther) const
{
    return value > aOther.value;
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE bool HalfFp16::operator>=(HalfFp16 aOther) const
{
    return value >= aOther.value;
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE bool HalfFp16::operator==(HalfFp16 aOther) const
{
    return value == aOther.value;
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE bool HalfFp16::operator!=(HalfFp16 aOther) const
{
    return value != aOther.value;
}

/// <summary>
/// Negation
/// </summary>
DEVICE_CODE INLINE HalfFp16 HalfFp16::operator-() const
{
    return HalfFp16(-value);
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE HalfFp16 &HalfFp16::operator+=(HalfFp16 aOther)
{
    value += aOther.value;
    return *this;
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE HalfFp16 HalfFp16::operator+(HalfFp16 aOther) const
{
    return HalfFp16(value + aOther.value);
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE HalfFp16 &HalfFp16::operator-=(HalfFp16 aOther)
{
    value -= aOther.value;
    return *this;
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE HalfFp16 HalfFp16::operator-(HalfFp16 aOther) const
{
    return HalfFp16(value - aOther.value);
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE HalfFp16 &HalfFp16::operator*=(HalfFp16 aOther)
{
    value *= aOther.value;
    return *this;
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE HalfFp16 HalfFp16::operator*(HalfFp16 aOther) const
{
    return HalfFp16(value * aOther.value);
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE HalfFp16 &HalfFp16::operator/=(HalfFp16 aOther)
{
    value /= aOther.value;
    return *this;
}

// defined in header...
// DEVICE_CODE INLINE HalfFp16 HalfFp16::operator/(HalfFp16 aOther) const
//{
//     return HalfFp16(value / aOther.value);
// }

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE HalfFp16 &HalfFp16::Exp()
{
    value = half_float::exp(value);
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 &HalfFp16::Exp()
{
    value = hexp(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE HalfFp16 HalfFp16::Exp(HalfFp16 aOther)
{
    return HalfFp16(half_float::exp(aOther.value));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 HalfFp16::Exp(HalfFp16 aOther)
{
    return HalfFp16(hexp(aOther.value));
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE HalfFp16 &HalfFp16::Ln()
{
    value = half_float::log(value);
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 &HalfFp16::Ln()
{
    value = hlog(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE HalfFp16 HalfFp16::Ln(HalfFp16 aOther)
{
    return HalfFp16(half_float::log(aOther.value));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 HalfFp16::Ln(HalfFp16 aOther)
{
    return HalfFp16(hlog(aOther.value));
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE HalfFp16 &HalfFp16::Sqrt()
{
    value = half_float::sqrt(value);
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 &HalfFp16::Sqrt()
{
    value = hsqrt(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE HalfFp16 HalfFp16::Sqrt(HalfFp16 aOther)
{
    return HalfFp16(half_float::sqrt(aOther.value));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 HalfFp16::Sqrt(HalfFp16 aOther)
{
    return HalfFp16(hsqrt(aOther.value));
}
#endif

/// <summary>
/// </summary>
DEVICE_CODE INLINE HalfFp16 &HalfFp16::Abs()
{
#ifdef IS_HOST_COMPILER
    value = half_float::abs(value);
#endif
#ifdef IS_CUDA_COMPILER
    value = __habs(value);
#endif
    return *this;
}

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE HalfFp16 HalfFp16::Abs(HalfFp16 aOther)
{
    return HalfFp16(half_float::abs(aOther.value));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 HalfFp16::Abs(HalfFp16 aOther)
{
    return HalfFp16(__habs(aOther.value));
}
#endif

/// <summary>
/// </summary>
DEVICE_CODE INLINE HalfFp16 &HalfFp16::Min(const HalfFp16 &aOther)
{
#ifdef IS_HOST_COMPILER
    *this = HalfFp16(fmin(value, aOther.value));
#endif
#ifdef IS_CUDA_COMPILER
    value = __hmin(value, aOther.value);
#endif
    return *this;
}

/// <summary>
/// Minimum
/// </summary>
DEVICE_CODE INLINE HalfFp16 HalfFp16::Min(const HalfFp16 &aLeft, const HalfFp16 &aRight)
{
#ifdef IS_HOST_COMPILER
    return HalfFp16(fmin(aLeft.value, aRight.value));
#endif
#ifdef IS_CUDA_COMPILER
    return HalfFp16(__hmin(aLeft.value, aRight.value));
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE HalfFp16 &HalfFp16::Max(const HalfFp16 &aOther)
{
#ifdef IS_HOST_COMPILER
    *this = HalfFp16(fmax(value, aOther.value));
#endif
#ifdef IS_CUDA_COMPILER
    value = __hmax(value, aOther.value);
#endif
    return *this;
}

/// <summary>
/// Maximum
/// </summary>
DEVICE_CODE INLINE HalfFp16 HalfFp16::Max(const HalfFp16 &aLeft, const HalfFp16 &aRight)
{
#ifdef IS_HOST_COMPILER
    return HalfFp16(fmax(aLeft.value, aRight.value));
#endif
#ifdef IS_CUDA_COMPILER
    return HalfFp16(__hmax(aLeft.value, aRight.value));
#endif
}

/// <summary>
/// round()
/// </summary>
DEVICE_CODE INLINE HalfFp16 &HalfFp16::Round()
{
#ifdef IS_HOST_COMPILER
    value = half_float::round(value);
#endif
#ifdef IS_CUDA_COMPILER
    //// it seems there is no "round" function for half floats, so implement it here:
    // constexpr ushort O_Point_5_AS_HFLOAT = 0x3800;
    // constexpr ushort SIGN_BIT            = 0x8000;
    // ushort hfloatbits                    = *reinterpret_cast<ushort *>(&value);
    // hfloatbits &= SIGN_BIT;
    // hfloatbits |= O_Point_5_AS_HFLOAT;

    //// add 0.5 to a positive value, -0.5 to a negative value and then round towards zero:
    // HalfFp16 onehalf = *reinterpret_cast<HalfFp16 *>(&hfloatbits);
    //*this += onehalf;
    // value = htrunc(value);

    // the above code is wrong as the addition is round to nearest even but needs to be round towards zero,
    // which is only available for sm_80 and later. So lets stick to conversion to float32...

    value = round(static_cast<float>(value));
#endif
    return *this;
}

/// <summary>
/// round()
/// </summary>
DEVICE_CODE INLINE HalfFp16 HalfFp16::Round(HalfFp16 aOther)
{
#ifdef IS_HOST_COMPILER
    return HalfFp16(half_float::round(aOther.value));
#endif
#ifdef IS_CUDA_COMPILER
    aOther.Round();
    return aOther;
#endif
}

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE HalfFp16 &HalfFp16::Floor()
{
    value = half_float::floor(value);
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 &HalfFp16::Floor()
{
    value = hfloor(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// floor()
/// </summary>
INLINE HalfFp16 HalfFp16::Floor(HalfFp16 aOther)
{
    return HalfFp16(half_float::floor(aOther.value));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// floor()
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 HalfFp16::Floor(HalfFp16 aOther)
{
    return HalfFp16(hfloor(aOther.value));
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE HalfFp16 &HalfFp16::Ceil()
{
    value = half_float::ceil(value);
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 &HalfFp16::Ceil()
{
    value = hceil(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// ceil()
/// </summary>
INLINE HalfFp16 HalfFp16::Ceil(HalfFp16 aOther)
{
    return HalfFp16(half_float::ceil(aOther.value));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// ceil()
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 HalfFp16::Ceil(HalfFp16 aOther)
{
    return HalfFp16(hceil(aOther.value));
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// Round nearest ties to even<para/>
/// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
INLINE HalfFp16 &HalfFp16::RoundNearest()
{
    value = half_float::nearbyint(value);
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// Round nearest ties to even<para/>
/// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 &HalfFp16::RoundNearest()
{
    value = hrint(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// Round nearest ties to even<para/>
/// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
INLINE HalfFp16 HalfFp16::RoundNearest(HalfFp16 aOther)
{
    return HalfFp16(half_float::nearbyint(aOther.value));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// Round nearest ties to even<para/>
/// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 HalfFp16::RoundNearest(HalfFp16 aOther)
{
    return HalfFp16(hrint(aOther.value));
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// Round toward zero
/// </summary>
INLINE HalfFp16 &HalfFp16::RoundZero()
{
    value = half_float::trunc(value);
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// Round toward zero
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 &HalfFp16::RoundZero()
{
    value = htrunc(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// Round toward zero
/// </summary>
INLINE HalfFp16 HalfFp16::RoundZero(HalfFp16 aOther)
{
    return HalfFp16(half_float::trunc(aOther.value));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// Round toward zero
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 HalfFp16::RoundZero(HalfFp16 aOther)
{
    return HalfFp16(htrunc(aOther.value));
}
#endif

DEVICE_CODE INLINE HalfFp16 HalfFp16::GetSign() const
{
    constexpr ushort ONE_AS_HFLOAT = 0x3c00;
    constexpr ushort SIGN_BIT      = 0x8000;
    ushort hfloatbits              = *reinterpret_cast<const ushort *>(&value);
    hfloatbits &= SIGN_BIT;
    hfloatbits |= ONE_AS_HFLOAT;
    return *reinterpret_cast<HalfFp16 *>(&hfloatbits);
}

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 HalfFp16::Sin(HalfFp16 aOther)
{
    return HalfFp16(hsin(aOther.value));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE HalfFp16 HalfFp16::Cos(HalfFp16 aOther)
{
    return HalfFp16(hcos(aOther.value));
}
#endif

#ifdef IS_HOST_COMPILER
INLINE std::ostream &operator<<(std::ostream &aOs, const opp::HalfFp16 &aHalf)
{
    return aOs << static_cast<float>(aHalf.value);
}
INLINE std::wostream &operator<<(std::wostream &aOs, const opp::HalfFp16 &aHalf)
{
    return aOs << static_cast<float>(aHalf.value);
}
INLINE std::istream &operator>>(std::istream &aIs, opp::HalfFp16 &aHalf)
{
    float temp{};
    aIs >> temp;
    aHalf = opp::HalfFp16(temp);
    return aIs;
}
INLINE std::wistream &operator>>(std::wistream &aIs, opp::HalfFp16 &aHalf)
{
    float temp{};
    aIs >> temp;
    aHalf = opp::HalfFp16(temp);
    return aIs;
}
#endif
} // namespace opp

// NOLINTEND(misc-definitions-in-headers, performance-unnecessary-value-param)

#undef INLINE