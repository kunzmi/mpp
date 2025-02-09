#pragma once
#include "bfloat16.h"
#include "exception.h"
#include "opp_defs.h"
#include <cmath>
#include <common/defines.h>
#include <common/numeric_limits.h>
#include <concepts>
#include <iostream>
#ifdef IS_CUDA_COMPILER
#include <cuda_bf16.h>
#endif

#ifdef IS_HOST_COMPILER
#define INLINE
#else
#define INLINE inline
#endif
// NOLINTBEGIN(misc-definitions-in-headers)

namespace opp
{

#ifdef IS_HOST_COMPILER
DEVICE_CODE INLINE BFloat16::BFloat16(float aFloat, RoundingMode aRoundingMode)
{
    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
            value = FromFloat(aFloat).value;
            break;
        case opp::RoundingMode::TowardZero:
            value = FromFloatTruncate(aFloat).value;
            break;
        default:
            throw INVALIDARGUMENT(aRoundingMode,
                                  "Invalid rounding mode provided: "
                                      << aRoundingMode
                                      << ". Only NearestTiesToEven and TowardZero are supported in host code (on "
                                         "CUDA devices also TowardNegativeInfinity and TowardPositiveInfinity).");
            break;
    }
}
#endif
#ifdef IS_CUDA_COMPILER
DEVICE_CODE INLINE BFloat16::BFloat16(__nv_bfloat16 aBFloat) : value(aBFloat)
{
}
DEVICE_CODE INLINE BFloat16::BFloat16(float aFloat) : value(__float2bfloat16_rn(aFloat))
{
}
DEVICE_CODE INLINE BFloat16::BFloat16(float aFloat, RoundingMode aRoundingMode)
{
    switch (aRoundingMode)
    {
        case opp::RoundingMode::NearestTiesToEven:
            value = __float2bfloat16_rn(aFloat);
            break;
        case opp::RoundingMode::TowardZero:
            value = __float2bfloat16_rz(aFloat);
            break;
        case opp::RoundingMode::TowardNegativeInfinity:
            value = __float2bfloat16_rd(aFloat);
            break;
        case opp::RoundingMode::TowardPositiveInfinity:
            value = __float2bfloat16_ru(aFloat);
            break;
        default:
            // other rounding modes are not supported and must be catched in host code...
            break;
    }
}
DEVICE_CODE INLINE BFloat16::BFloat16(sbyte aVal) : value(__short2bfloat16_rn(aVal))
{
}
DEVICE_CODE INLINE BFloat16::BFloat16(byte aVal) : value(__ushort2bfloat16_rn(aVal))
{
}
DEVICE_CODE INLINE BFloat16::BFloat16(short aVal) : value(__short2bfloat16_rn(aVal))
{
}
DEVICE_CODE INLINE BFloat16::BFloat16(ushort aVal) : value(__ushort2bfloat16_rn(aVal))
{
}
DEVICE_CODE INLINE BFloat16::BFloat16(int aVal) : value(__int2bfloat16_rn(aVal))
{
}
DEVICE_CODE INLINE BFloat16::BFloat16(uint aVal) : value(__uint2bfloat16_rn(aVal))
{
}
DEVICE_ONLY_CODE INLINE BFloat16::BFloat16(long64 aVal) : value(__ll2bfloat16_rn(aVal))
{
}
DEVICE_ONLY_CODE INLINE BFloat16::BFloat16(ulong64 aVal) : value(__ull2bfloat16_rn(aVal))
{
}
DEVICE_ONLY_CODE INLINE BFloat16::BFloat16(double aVal) : value(__double2bfloat16(aVal))
{
}
#endif

#ifdef IS_HOST_COMPILER

/// <summary>
/// Truncate instead of rounding, preserving SNaN
/// </summary>
BFloat16 INLINE BFloat16::FromFloatTruncate(float aFloat)
{
    const union
    {
        float fp32;
        uint int32;
    } u = {aFloat};
    // NOLINTNEXTLINE
    return BFloat16(static_cast<ushort>((u.int32 >> 16) | (!(~u.int32 & 0x7f800000) && (u.int32 & 0xffff))), BINARY);
}
#endif

// zero extend lower 16 bits of bfloat16 to convert to IEEE float
DEVICE_CODE INLINE BFloat16::operator float() const
{
#ifdef IS_HOST_COMPILER
    const union
    {
        uint int32;
        float fp32;
    } u = {static_cast<uint>(value) << 16}; // NOLINT
    return u.fp32;                          // NOLINT
#endif
#ifdef IS_CUDA_COMPILER
    return __bfloat162float(value);
#endif
}

#ifdef IS_CUDA_COMPILER
DEVICE_CODE INLINE BFloat16::operator int() const
{
    return __bfloat162int_rz(value);
}
DEVICE_CODE INLINE BFloat16::operator uint() const
{
    return __bfloat162uint_rz(value);
}
DEVICE_CODE INLINE BFloat16::operator short() const
{
    return __bfloat162short_rz(value);
}
DEVICE_CODE INLINE BFloat16::operator ushort() const
{
    return __bfloat162ushort_rz(value);
}
DEVICE_CODE INLINE BFloat16::operator byte() const
{
    return __bfloat162uchar_rz(value);
}
DEVICE_CODE INLINE BFloat16::operator sbyte() const
{
    return __bfloat162char_rz(value);
}
#endif

/// <summary>
/// </summary>
DEVICE_CODE INLINE bool BFloat16::operator<(BFloat16 aOther) const
{
#ifdef IS_HOST_COMPILER
    return static_cast<float>(*this) < static_cast<float>(aOther);
#endif

#ifdef IS_CUDA_COMPILER
    return value < aOther.value;
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE bool BFloat16::operator<=(BFloat16 aOther) const
{
#ifdef IS_HOST_COMPILER
    return static_cast<float>(*this) <= static_cast<float>(aOther);
#endif

#ifdef IS_CUDA_COMPILER
    return value <= aOther.value;
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE bool BFloat16::operator>(BFloat16 aOther) const
{
#ifdef IS_HOST_COMPILER
    return static_cast<float>(*this) > static_cast<float>(aOther);
#endif

#ifdef IS_CUDA_COMPILER
    return value > aOther.value;
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE bool BFloat16::operator>=(BFloat16 aOther) const
{
#ifdef IS_HOST_COMPILER
    return static_cast<float>(*this) >= static_cast<float>(aOther);
#endif

#ifdef IS_CUDA_COMPILER
    return value >= aOther.value;
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE bool BFloat16::operator==(BFloat16 aOther) const
{
#ifdef IS_HOST_COMPILER
    return static_cast<float>(*this) == static_cast<float>(aOther);
#endif

#ifdef IS_CUDA_COMPILER
    return value == aOther.value;
#endif
}

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
DEVICE_CODE INLINE bool BFloat16::operator==(float aOther) const
{
    return static_cast<float>(*this) == aOther;
}
/// <summary>
/// </summary>
DEVICE_CODE INLINE bool BFloat16::operator==(int aOther) const
{
    return static_cast<float>(*this) == static_cast<float>(aOther);
}
#endif

/// <summary>
/// </summary>
DEVICE_CODE INLINE bool BFloat16::operator!=(BFloat16 aOther) const
{
#ifdef IS_HOST_COMPILER
    return static_cast<float>(*this) != static_cast<float>(aOther);
#endif

#ifdef IS_CUDA_COMPILER
    return value != aOther.value;
#endif
}

#ifdef IS_HOST_COMPILER
/// <summary>
/// Negation
/// </summary>
// constexpr BFloat16 BFloat16::operator-() const;
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// Negation
/// </summary>
DEVICE_CODE INLINE BFloat16 BFloat16::operator-() const
{
    return BFloat16(-value);
}
#endif

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 &BFloat16::operator+=(BFloat16 aOther)
{
#ifdef IS_HOST_COMPILER
    *this = *this + aOther;
    return *this;
#endif

#ifdef IS_CUDA_COMPILER
    value += aOther.value;
    return *this;
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 BFloat16::operator+(BFloat16 aOther) const
{
#ifdef IS_HOST_COMPILER
    return BFloat16(static_cast<float>(*this) + static_cast<float>(aOther));
#endif

#ifdef IS_CUDA_COMPILER
    return BFloat16(value + aOther.value);
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 &BFloat16::operator-=(BFloat16 aOther)
{
#ifdef IS_HOST_COMPILER
    *this = *this - aOther;
    return *this;
#endif

#ifdef IS_CUDA_COMPILER
    value -= aOther.value;
    return *this;
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 BFloat16::operator-(BFloat16 aOther) const
{
#ifdef IS_HOST_COMPILER
    return BFloat16(static_cast<float>(*this) - static_cast<float>(aOther));
#endif

#ifdef IS_CUDA_COMPILER
    return BFloat16(value - aOther.value);
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 &BFloat16::operator*=(BFloat16 aOther)
{
#ifdef IS_HOST_COMPILER
    *this = *this * aOther;
    return *this;
#endif

#ifdef IS_CUDA_COMPILER
    value *= aOther.value;
    return *this;
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 BFloat16::operator*(BFloat16 aOther) const
{
#ifdef IS_HOST_COMPILER
    return BFloat16(static_cast<float>(*this) * static_cast<float>(aOther));
#endif

#ifdef IS_CUDA_COMPILER
    return BFloat16(value * aOther.value);
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 &BFloat16::operator/=(BFloat16 aOther)
{
#ifdef IS_HOST_COMPILER
    *this = *this / aOther;
    return *this;
#endif

#ifdef IS_CUDA_COMPILER
    value /= aOther.value;
    return *this;
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 BFloat16::operator/(BFloat16 aOther) const
{
#ifdef IS_HOST_COMPILER
    return BFloat16(static_cast<float>(*this) / static_cast<float>(aOther));
#endif

#ifdef IS_CUDA_COMPILER
    return BFloat16(value / aOther.value);
#endif
}

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE BFloat16 &BFloat16::Exp()
{
    *this = BFloat16(std::exp(static_cast<float>(*this)));
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 &BFloat16::Exp()
{
    value = hexp(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE BFloat16 BFloat16::Exp(BFloat16 aOther)
{
    return BFloat16(std::exp(static_cast<float>(aOther)));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 BFloat16::Exp(BFloat16 aOther)
{
    return BFloat16(hexp(aOther.value));
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE BFloat16 &BFloat16::Ln()
{
    *this = BFloat16(std::log(static_cast<float>(*this)));
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 &BFloat16::Ln()
{
    value = hlog(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE BFloat16 BFloat16::Ln(BFloat16 aOther)
{
    return BFloat16(std::log(static_cast<float>(aOther)));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 BFloat16::Ln(BFloat16 aOther)
{
    return BFloat16(hlog(aOther.value));
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE BFloat16 &BFloat16::Sqrt()
{
    *this = BFloat16(std::sqrt(static_cast<float>(*this)));
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 &BFloat16::Sqrt()
{
    value = hsqrt(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE BFloat16 BFloat16::Sqrt(BFloat16 aOther)
{
    return BFloat16(std::sqrt(static_cast<float>(aOther)));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 BFloat16::Sqrt(BFloat16 aOther)
{
    return BFloat16(hsqrt(aOther.value));
}
#endif

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 &BFloat16::Abs()
{
#ifdef IS_HOST_COMPILER
    *this = BFloat16(std::abs(static_cast<float>(*this)));
#endif
#ifdef IS_CUDA_COMPILER
    value = __habs(value);
#endif
    return *this;
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 BFloat16::Abs(BFloat16 aOther)
{
#ifdef IS_HOST_COMPILER
    return BFloat16(std::abs(static_cast<float>(aOther)));
#endif
#ifdef IS_CUDA_COMPILER
    return BFloat16(__habs(aOther.value));
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 &BFloat16::Min(const BFloat16 &aOther)
{
#ifdef IS_HOST_COMPILER
    *this = BFloat16(std::min(static_cast<float>(*this), static_cast<float>(aOther)));
#endif
#ifdef IS_CUDA_COMPILER
    value = __hmin(value, aOther.value);
#endif
    return *this;
}

/// <summary>
/// Minimum
/// </summary>
DEVICE_CODE INLINE BFloat16 BFloat16::Min(const BFloat16 &aLeft, const BFloat16 &aRight)
{
#ifdef IS_HOST_COMPILER
    return BFloat16(std::min(static_cast<float>(aLeft), static_cast<float>(aRight)));
#endif
#ifdef IS_CUDA_COMPILER
    return BFloat16(__hmin(aLeft.value, aRight.value));
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 &BFloat16::Max(const BFloat16 &aOther)
{
#ifdef IS_HOST_COMPILER
    *this = BFloat16(std::max(static_cast<float>(*this), static_cast<float>(aOther)));
#endif
#ifdef IS_CUDA_COMPILER
    value = __hmax(value, aOther.value);
#endif
    return *this;
}

/// <summary>
/// Maximum
/// </summary>
DEVICE_CODE INLINE BFloat16 BFloat16::Max(const BFloat16 &aLeft, const BFloat16 &aRight)
{
#ifdef IS_HOST_COMPILER
    return BFloat16(std::max(static_cast<float>(aLeft), static_cast<float>(aRight)));
#endif
#ifdef IS_CUDA_COMPILER
    return BFloat16(__hmax(aLeft.value, aRight.value));
#endif
}

/// <summary>
/// </summary>
DEVICE_CODE INLINE BFloat16 &BFloat16::Round()
{
#ifdef IS_HOST_COMPILER
    *this = BFloat16(std::round(static_cast<float>(*this)));
#endif
#ifdef IS_CUDA_COMPILER
    //// it seems there is no "round" function for bfloats, so implement it here:
    // constexpr ushort O_Point_5_AS_BFLOAT = 0x3f00;
    // constexpr ushort SIGN_BIT            = 0x8000;
    // ushort bfloatbits                    = *reinterpret_cast<ushort *>(&value);
    // bfloatbits &= SIGN_BIT;
    // bfloatbits |= O_Point_5_AS_BFLOAT;

    //// add 0.5 to a positive value, -0.5 to a negative value and then round towards zero:
    // BFloat16 onehalf = *reinterpret_cast<BFloat16 *>(&bfloatbits);
    //*this += onehalf;

    // value = htrunc(value);
    //
    //   the above code is wrong as the addition is round to nearest even but needs to be round towards zero,
    //   which is only available for sm_90 and later. So lets stick to conversion to float32...

    value = round(static_cast<float>(value));
#endif
    return *this;
}

/// <summary>
/// round()
/// </summary>
DEVICE_CODE INLINE BFloat16 BFloat16::Round(BFloat16 aOther)
{
#ifdef IS_HOST_COMPILER
    return BFloat16(std::round(static_cast<float>(aOther)));
#endif
#ifdef IS_CUDA_COMPILER
    aOther.Round();
    return aOther;
#endif
}

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE BFloat16 &BFloat16::Floor()
{
    *this = BFloat16(std::floor(static_cast<float>(*this)));
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 &BFloat16::Floor()
{
    value = hfloor(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// floor()
/// </summary>
INLINE BFloat16 BFloat16::Floor(BFloat16 aOther)
{
    return BFloat16(std::floor(static_cast<float>(aOther)));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// floor()
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 BFloat16::Floor(BFloat16 aOther)
{
    return BFloat16(hfloor(aOther.value));
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// </summary>
INLINE BFloat16 &BFloat16::Ceil()
{
    *this = BFloat16(std::ceil(static_cast<float>(*this)));
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 &BFloat16::Ceil()
{
    value = hceil(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// ceil()
/// </summary>
INLINE BFloat16 BFloat16::Ceil(BFloat16 aOther)
{
    return BFloat16(std::ceil(static_cast<float>(aOther)));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// ceil()
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 BFloat16::Ceil(BFloat16 aOther)
{
    return BFloat16(hceil(aOther.value));
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// Round nearest ties to even<para/>
/// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
INLINE BFloat16 &BFloat16::RoundNearest()
{
    *this = BFloat16(std::nearbyint(static_cast<float>(*this)));
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// Round nearest ties to even<para/>
/// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 &BFloat16::RoundNearest()
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
INLINE BFloat16 BFloat16::RoundNearest(BFloat16 aOther)
{
    return BFloat16(std::nearbyint(static_cast<float>(aOther)));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// Round nearest ties to even<para/>
/// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 BFloat16::RoundNearest(BFloat16 aOther)
{
    return BFloat16(hrint(aOther.value));
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// Round toward zero
/// </summary>
INLINE BFloat16 &BFloat16::RoundZero()
{
    *this = BFloat16(std::trunc(static_cast<float>(*this)));
    return *this;
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// Round toward zero
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 &BFloat16::RoundZero()
{
    value = htrunc(value);
    return *this;
}
#endif

#ifdef IS_HOST_COMPILER
/// <summary>
/// Round toward zero
/// </summary>
INLINE BFloat16 BFloat16::RoundZero(BFloat16 aOther)
{
    return BFloat16(std::trunc(float(aOther)));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// Round toward zero
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 BFloat16::RoundZero(BFloat16 aOther)
{
    return BFloat16(htrunc(aOther.value));
}
#endif

DEVICE_CODE INLINE BFloat16 BFloat16::GetSign() const
{
    constexpr ushort ONE_AS_BFLOAT = 0x3f80;
    constexpr ushort SIGN_BIT      = 0x8000;

#ifdef IS_HOST_COMPILER
    ushort bfloatbits = value;
#endif
#ifdef IS_CUDA_COMPILER
    ushort bfloatbits = *reinterpret_cast<const ushort *>(&value);
#endif
    bfloatbits &= SIGN_BIT;
    bfloatbits |= ONE_AS_BFLOAT;
    return *reinterpret_cast<BFloat16 *>(&bfloatbits);
}

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 BFloat16::Sin(BFloat16 aOther)
{
    return BFloat16(hsin(aOther.value));
}
#endif

#ifdef IS_CUDA_COMPILER
/// <summary>
/// </summary>
DEVICE_ONLY_CODE INLINE BFloat16 BFloat16::Cos(BFloat16 aOther)
{
    return BFloat16(hcos(aOther.value));
}
#endif

INLINE std::ostream &operator<<(std::ostream &aOs, const BFloat16 &aHalf)
{
    return aOs << static_cast<float>(aHalf);
}
INLINE std::wostream &operator<<(std::wostream &aOs, const BFloat16 &aHalf)
{
    return aOs << static_cast<float>(aHalf);
}
INLINE std::istream &operator>>(std::istream &aIs, BFloat16 &aHalf)
{
    float temp{};
    aIs >> temp;
    aHalf = BFloat16(temp);
    return aIs;
}
INLINE std::wistream &operator>>(std::wistream &aIs, BFloat16 &aHalf)
{
    float temp{};
    aIs >> temp;
    aHalf = BFloat16(temp);
    return aIs;
}

} // namespace opp
// NOLINTEND(misc-definitions-in-headers)

#undef INLINE