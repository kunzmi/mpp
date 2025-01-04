#pragma once
#include <cmath>
#include <common/defines.h>
#include <common/numeric_limits.h>
#include <concepts>
#include <iostream>
#ifdef IS_CUDA_COMPILER
#include <cuda_bf16.h>
#endif
#ifdef IS_HOST_COMPILER
#endif

namespace opp
{
/// <summary>
/// We implement our own BFloat16 type, as different devices use different implementations. This is meant to wrap them
/// all together to one type: On CPU we use our own implementation (which is based on amd_hip_bfloat16.h), on Cuda
/// devices the bf16 header from Nvidia and on AMD devices the implementation coming with ROCm. But from an external
/// view, it is always the same opp::BFloat16 datatype.
/// </summary>
class alignas(2) BFloat16
{
  private:
    static constexpr bool BINARY = true; // additional argument for constructor to switch to binary

#ifdef IS_HOST_COMPILER
    ushort value;

    explicit constexpr BFloat16(ushort aUShort, bool /*aBinary*/) : value(aUShort)
    {
    }
#endif
#ifdef IS_CUDA_COMPILER
    __nv_bfloat16 value;

    DEVICE_CODE explicit constexpr BFloat16(ushort aUShort, bool /*aBinary*/) : value(__nv_bfloat16_raw{aUShort})
    {
    }
    friend bool DEVICE_CODE isnan(BFloat16);
    friend bool DEVICE_CODE isinf(BFloat16);
#endif

  public:
    BFloat16() noexcept = default;

#ifdef IS_HOST_COMPILER
    explicit constexpr BFloat16(float aFloat) : value(FromFloat(aFloat).value)
    {
    }
    explicit constexpr BFloat16(sbyte aVal) : value(FromFloat(static_cast<float>(aVal)).value)
    {
    }
    explicit constexpr BFloat16(byte aVal) : value(FromFloat(static_cast<float>(aVal)).value)
    {
    }
    explicit constexpr BFloat16(short aVal) : value(FromFloat(static_cast<float>(aVal)).value)
    {
    }
    explicit constexpr BFloat16(ushort aVal) : value(FromFloat(static_cast<float>(aVal)).value)
    {
    }
    explicit constexpr BFloat16(int aInt) : value(FromFloat(static_cast<float>(aInt)).value)
    {
    }
    explicit constexpr BFloat16(uint aVal) : value(FromFloat(static_cast<float>(aVal)).value)
    {
    }
    explicit constexpr BFloat16(long64 aVal) : value(FromFloat(static_cast<float>(aVal)).value)
    {
    }
    explicit constexpr BFloat16(ulong64 aVal) : value(FromFloat(static_cast<float>(aVal)).value)
    {
    }
    explicit constexpr BFloat16(double aVal) : value(FromFloat(static_cast<float>(aVal)).value)
    {
    }
#endif
#ifdef IS_CUDA_COMPILER
    DEVICE_CODE explicit BFloat16(__nv_bfloat16 aBFloat) : value(aBFloat)
    {
    }
    DEVICE_CODE explicit BFloat16(float aFloat) : value(__float2bfloat16_rn(aFloat))
    {
    }
    DEVICE_CODE explicit BFloat16(sbyte aVal) : value(__short2bfloat16_rn(aVal))
    {
    }
    DEVICE_CODE explicit BFloat16(byte aVal) : value(__ushort2bfloat16_rn(aVal))
    {
    }
    DEVICE_CODE explicit BFloat16(short aVal) : value(__short2bfloat16_rn(aVal))
    {
    }
    DEVICE_CODE explicit BFloat16(ushort aVal) : value(__ushort2bfloat16_rn(aVal))
    {
    }
    DEVICE_CODE explicit BFloat16(int aVal) : value(__int2bfloat16_rn(aVal))
    {
    }
    DEVICE_CODE explicit BFloat16(uint aVal) : value(__uint2bfloat16_rn(aVal))
    {
    }
    DEVICE_ONLY_CODE explicit BFloat16(long64 aVal) : value(__ll2bfloat16_rn(aVal))
    {
    }
    DEVICE_ONLY_CODE explicit BFloat16(ulong64 aVal) : value(__ull2bfloat16_rn(aVal))
    {
    }
    DEVICE_ONLY_CODE explicit BFloat16(double aVal) : value(__double2bfloat16(aVal))
    {
    }
#endif
    ~BFloat16() = default;

    BFloat16(const BFloat16 &)     = default;
    BFloat16(BFloat16 &&) noexcept = default;

    BFloat16 &operator=(const BFloat16 &)     = default;
    BFloat16 &operator=(BFloat16 &&) noexcept = default;

    DEVICE_CODE [[nodiscard]] inline constexpr static BFloat16 FromUShort(ushort aUShort)
    {
        return BFloat16(aUShort, BINARY);
    }

#ifdef IS_HOST_COMPILER
    [[nodiscard]] inline constexpr static BFloat16 FromFloat(float aFloat)
    {
        union
        {
            float fp32;
            uint int32;
        } u = {aFloat};
        if (~u.int32 & 0x7f800000)
        {
            // When the exponent bits are not all 1s, then the value is zero, normal,
            // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
            // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
            // This causes the bfloat16's mantissa to be incremented by 1 if the 16
            // least significant bits of the float mantissa are greater than 0x8000,
            // or if they are equal to 0x8000 and the least significant bit of the
            // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
            // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
            // has the value 0x7f, then incrementing it causes it to become 0x00 and
            // the exponent is incremented by one, which is the next higher FP value
            // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
            // with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
            // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
            // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
            // incrementing it causes it to become an exponent of 0xFF and a mantissa
            // of 0x00, which is Inf, the next higher value to the unrounded value.
            u.int32 += 0x7fff + ((u.int32 >> 16) & 1); // Round to nearest, round to even
        }
        else if (u.int32 & 0xffff)
        {
            // When all of the exponent bits are 1, the value is Inf or NaN.
            // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
            // mantissa bit. Quiet NaN is indicated by the most significant mantissa
            // bit being 1. Signaling NaN is indicated by the most significant
            // mantissa bit being 0 but some other bit(s) being 1. If any of the
            // lower 16 bits of the mantissa are 1, we set the least significant bit
            // of the bfloat16 mantissa, in order to preserve signaling NaN in case
            // the bloat16's mantissa bits are all 0.
            u.int32 |= 0x10000; // Preserve signaling NaN
        }
        return BFloat16(static_cast<ushort>(u.int32 >> 16), BINARY);
    }

    /// <summary>
    /// Truncate instead of rounding, preserving SNaN
    /// </summary>
    [[nodiscard]] inline static BFloat16 FromFloatTruncate(float aFloat)
    {
        union
        {
            float fp32;
            uint int32;
        } u = {aFloat};
        return BFloat16(static_cast<ushort>((u.int32 >> 16) | (!(~u.int32 & 0x7f800000) && (u.int32 & 0xffff))),
                        BINARY);
    }
#endif

    // zero extend lower 16 bits of bfloat16 to convert to IEEE float
    DEVICE_CODE inline operator float() const
    {
#ifdef IS_HOST_COMPILER
        union
        {
            uint int32;
            float fp32;
        } u = {static_cast<uint>(value) << 16};
        return u.fp32;
#endif
#ifdef IS_CUDA_COMPILER
        return __bfloat162float(value);
#endif
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline bool operator<(BFloat16 aOther) const
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
    DEVICE_CODE [[nodiscard]] inline bool operator<=(BFloat16 aOther) const
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
    DEVICE_CODE [[nodiscard]] inline bool operator>(BFloat16 aOther) const
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
    DEVICE_CODE [[nodiscard]] inline bool operator>=(BFloat16 aOther) const
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
    DEVICE_CODE [[nodiscard]] inline bool operator==(BFloat16 aOther) const
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
    DEVICE_CODE [[nodiscard]] inline bool operator==(float aOther) const
    {
        return static_cast<float>(*this) == aOther;
    }
    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline bool operator==(int aOther) const
    {
        return static_cast<float>(*this) == static_cast<float>(aOther);
    }
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline bool operator!=(BFloat16 aOther) const
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
    [[nodiscard]] inline constexpr BFloat16 operator-() const
    {
        BFloat16 ret = *this;
        ret.value ^= 0x8000;
        return ret;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline BFloat16 operator-() const
    {
        return BFloat16(-value);
    }
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE inline BFloat16 &operator+=(BFloat16 aOther)
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
    DEVICE_CODE [[nodiscard]] inline BFloat16 operator+(BFloat16 aOther) const
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
    DEVICE_CODE inline BFloat16 &operator-=(BFloat16 aOther)
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
    DEVICE_CODE [[nodiscard]] inline BFloat16 operator-(BFloat16 aOther) const
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
    DEVICE_CODE inline BFloat16 &operator*=(BFloat16 aOther)
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
    DEVICE_CODE [[nodiscard]] inline BFloat16 operator*(BFloat16 aOther) const
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
    DEVICE_CODE inline BFloat16 &operator/=(BFloat16 aOther)
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
    DEVICE_CODE [[nodiscard]] inline BFloat16 operator/(BFloat16 aOther) const
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
    inline BFloat16 &Exp()
    {
        *this = BFloat16(std::exp(static_cast<float>(*this)));
        return *this;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline BFloat16 &Exp()
    {
        value = hexp(value);
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] inline static BFloat16 Exp(BFloat16 aOther)
    {
        return BFloat16(std::exp(static_cast<float>(aOther)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static BFloat16 Exp(BFloat16 aOther)
    {
        return BFloat16(hexp(aOther.value));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    inline BFloat16 &Ln()
    {
        *this = BFloat16(std::log(static_cast<float>(*this)));
        return *this;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline BFloat16 &Ln()
    {
        value = hlog(value);
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] inline static BFloat16 Ln(BFloat16 aOther)
    {
        return BFloat16(std::log(static_cast<float>(aOther)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static BFloat16 Ln(BFloat16 aOther)
    {
        return BFloat16(hlog(aOther.value));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    inline BFloat16 &Sqrt()
    {
        *this = BFloat16(std::sqrt(static_cast<float>(*this)));
        return *this;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline BFloat16 &Sqrt()
    {
        value = hsqrt(value);
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] inline static BFloat16 Sqrt(BFloat16 aOther)
    {
        return BFloat16(std::sqrt(static_cast<float>(aOther)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static BFloat16 Sqrt(BFloat16 aOther)
    {
        return BFloat16(hsqrt(aOther.value));
    }
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE inline BFloat16 &Abs()
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
    DEVICE_CODE [[nodiscard]] inline static BFloat16 Abs(BFloat16 aOther)
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
    DEVICE_CODE inline BFloat16 &Min(const BFloat16 &aOther)
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
    DEVICE_CODE [[nodiscard]] inline static BFloat16 Min(const BFloat16 &aLeft, const BFloat16 &aRight)
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
    DEVICE_CODE inline BFloat16 &Max(const BFloat16 &aOther)
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
    DEVICE_CODE [[nodiscard]] inline static BFloat16 Max(const BFloat16 &aLeft, const BFloat16 &aRight)
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
    DEVICE_CODE inline BFloat16 &Round()
    {
#ifdef IS_HOST_COMPILER
        *this = BFloat16(std::round(static_cast<float>(*this)));
#endif
#ifdef IS_CUDA_COMPILER
        *this = BFloat16(round(static_cast<float>(value)));
#endif
        return *this;
    }

    /// <summary>
    /// round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline static BFloat16 Round(BFloat16 aOther)
    {
#ifdef IS_HOST_COMPILER
        return BFloat16(std::round(static_cast<float>(aOther)));
#endif
#ifdef IS_CUDA_COMPILER
        return BFloat16(round(static_cast<float>(aOther.value)));
#endif
    }

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    inline BFloat16 &Floor()
    {
        *this = BFloat16(std::floor(static_cast<float>(*this)));
        return *this;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline BFloat16 &Floor()
    {
        value = __int2bfloat16_rd(__bfloat162int_rd(value));
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// floor()
    /// </summary>
    [[nodiscard]] inline static BFloat16 Floor(BFloat16 aOther)
    {
        return BFloat16(std::floor(static_cast<float>(aOther)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// floor()
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static BFloat16 Floor(BFloat16 aOther)
    {
        return BFloat16(__int2bfloat16_rd(__bfloat162int_rd(aOther.value)));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    inline BFloat16 &Ceil()
    {
        *this = BFloat16(std::ceil(static_cast<float>(*this)));
        return *this;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline BFloat16 &Ceil()
    {
        value = __int2bfloat16_ru(__bfloat162int_ru(value));
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// ceil()
    /// </summary>
    [[nodiscard]] inline static BFloat16 Ceil(BFloat16 aOther)
    {
        return BFloat16(std::ceil(static_cast<float>(aOther)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// ceil()
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static BFloat16 Ceil(BFloat16 aOther)
    {
        return BFloat16(__int2bfloat16_ru(__bfloat162int_ru(aOther.value)));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    inline BFloat16 &RoundNearest()
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
    DEVICE_ONLY_CODE inline BFloat16 &RoundNearest()
    {
        value = __int2bfloat16_rn(__bfloat162int_rn(value));
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    [[nodiscard]] inline static BFloat16 RoundNearest(BFloat16 aOther)
    {
        return BFloat16(std::nearbyint(static_cast<float>(aOther)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static BFloat16 RoundNearest(BFloat16 aOther)
    {
        return BFloat16(__int2bfloat16_rn(__bfloat162int_rn(aOther.value)));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    inline BFloat16 &RoundZero()
    {
        *this = BFloat16(std::trunc(static_cast<float>(*this)));
        return *this;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    DEVICE_ONLY_CODE inline BFloat16 &RoundZero()
    {
        value = __int2bfloat16_rz(__bfloat162int_rz(value));
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    [[nodiscard]] inline static BFloat16 RoundZero(BFloat16 aOther)
    {
        return BFloat16(std::trunc(float(aOther)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static BFloat16 RoundZero(BFloat16 aOther)
    {
        return BFloat16(__int2bfloat16_rz(__bfloat162int_rz(aOther.value)));
    }
#endif

    DEVICE_CODE inline BFloat16 GetSign()
    {
        constexpr ushort ONE_AS_BFLOAT = 0x3f80;
        constexpr ushort SIGN_BIT      = 0x8000;
        ushort bfloatbits              = *reinterpret_cast<ushort *>(&value);
        bfloatbits &= SIGN_BIT;
        bfloatbits |= ONE_AS_BFLOAT;
        return *reinterpret_cast<BFloat16 *>(&bfloatbits);
    }

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static BFloat16 Sin(BFloat16 aOther)
    {
        return BFloat16(hsin(aOther.value));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static BFloat16 Cos(BFloat16 aOther)
    {
        return BFloat16(hcos(aOther.value));
    }
#endif
};

inline std::ostream &operator<<(std::ostream &aOs, const BFloat16 &aHalf)
{
    return aOs << static_cast<float>(aHalf);
}
inline std::wostream &operator<<(std::wostream &aOs, const BFloat16 &aHalf)
{
    return aOs << static_cast<float>(aHalf);
}
inline std::istream &operator>>(std::istream &aIs, BFloat16 &aHalf)
{
    float temp;
    aIs >> temp;
    aHalf = BFloat16(temp);
    return aIs;
}
inline std::wistream &operator>>(std::wistream &aIs, BFloat16 &aHalf)
{
    float temp;
    aIs >> temp;
    aHalf = BFloat16(temp);
    return aIs;
}

// bfloat literal: 2.4_bf is read as BFloat16 when opp namespace is used
inline BFloat16 operator"" _bf(long double aValue)
{
    return BFloat16(float(aValue));
}

#ifdef IS_CUDA_COMPILER
DEVICE_CODE bool isnan(BFloat16 aVal)
{
    return __hisnan(aVal.value);
}
DEVICE_CODE bool isinf(BFloat16 aVal)
{
    return __hisinf(aVal.value);
}
#endif

// 16 bit BFloat floating point types
template <> struct numeric_limits<BFloat16>
{
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 min() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0x0400));
    }
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 max() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0x7F7F));
    }
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 lowest() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0xFF7F));
    }
    // minimum exact integer in float = -(2^8) = -256
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 minExact() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0xC380));
    }
    // maximum exact integer in float = 2^9 = 256
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 maxExact() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0x4380));
    }
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 infinity() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0x7f80));
    }
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 quiet_NaN() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0x7fc0));
    }
};

// numeric limits when converting from one type to another, especially 16-Bit floats have some restrictions
template <typename TTo> struct numeric_limits_conversion<BFloat16, TTo>
{
    // Does not have a constexpr constructor on cuda device...
    [[nodiscard]] static const DEVICE_CODE BFloat16 min() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<TTo>::min());
    }
    [[nodiscard]] static const DEVICE_CODE BFloat16 max() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<TTo>::max());
    }
    [[nodiscard]] static const DEVICE_CODE BFloat16 lowest() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<TTo>::lowest());
    }
    [[nodiscard]] static const DEVICE_CODE BFloat16 minExact() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<TTo>::minExact());
    }
    [[nodiscard]] static const DEVICE_CODE BFloat16 maxExact() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<TTo>::maxExact());
    }
};

template <> struct numeric_limits_conversion<BFloat16, short>
{
    // BFloat16 does not have a constexpr constructor on cuda device...
    [[nodiscard]] static const DEVICE_CODE BFloat16 min() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<short>::min());
    }
    [[nodiscard]] static const DEVICE_CODE BFloat16 max() noexcept
    {
        // special case for half floats: the maximum value of short is slightly smaller than the closest exact
        // integer in BFloat16, and as we use round to nearest, the clamping would result in a too large number.
        // Thus for BFloat16 and short, we clamp to the next integer smaller than short::max(), i.e. 32640
        return BFloat16::FromUShort(0x46FF); // = 32640
    }
    [[nodiscard]] static const DEVICE_CODE BFloat16 lowest() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<short>::lowest());
    }
    [[nodiscard]] static const DEVICE_CODE BFloat16 minExact() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<short>::minExact());
    }
    [[nodiscard]] static const DEVICE_CODE BFloat16 maxExact() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<short>::maxExact());
    }
};

template <> struct numeric_limits_conversion<BFloat16, ushort>
{
    [[nodiscard]] static const DEVICE_CODE BFloat16 min() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<ushort>::min());
    }
    [[nodiscard]] static const DEVICE_CODE BFloat16 max() noexcept
    {
        // special case for half floats: the maximum value of ushort is slightly smaller than the closest exact
        // integer in BFloat16, and as we use round to nearest, the clamping would result in a too large number.
        // Thus for BFloat16 and short, we clamp to the next integer smaller than ushort::max(), i.e. 65280
        return BFloat16::FromUShort(0x477f); // = 65280
    }
    [[nodiscard]] static const DEVICE_CODE BFloat16 lowest() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<ushort>::lowest());
    }
    [[nodiscard]] static const DEVICE_CODE BFloat16 minExact() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<ushort>::minExact());
    }
    [[nodiscard]] static const DEVICE_CODE BFloat16 maxExact() noexcept
    {
        return static_cast<BFloat16>(numeric_limits<ushort>::maxExact());
    }
};

} // namespace opp