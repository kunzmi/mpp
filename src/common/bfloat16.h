#pragma once
#include <cmath>
#include <common/defines.h>
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
#ifdef IS_HOST_COMPILER
    ushort value{};

    explicit constexpr BFloat16(ushort aUShort, bool /*aBinary*/) : value(aUShort)
    {
    }
#endif
#ifdef IS_CUDA_COMPILER
    __nv_bfloat16 value{};

    DEVICE_CODE explicit constexpr BFloat16(ushort aUShort, bool /*aBinary*/) : value(__nv_bfloat16_raw{aUShort})
    {
    }
#endif

  public:
    BFloat16() = default;
#ifdef IS_HOST_COMPILER
    explicit constexpr BFloat16(float aFloat) : value(FromFloat(aFloat).value)
    {
    }
    explicit constexpr BFloat16(sbyte aVal) : value(FromFloat(float(aVal)).value)
    {
    }
    explicit constexpr BFloat16(byte aVal) : value(FromFloat(float(aVal)).value)
    {
    }
    explicit constexpr BFloat16(short aVal) : value(FromFloat(float(aVal)).value)
    {
    }
    explicit constexpr BFloat16(ushort aVal) : value(FromFloat(float(aVal)).value)
    {
    }
    explicit constexpr BFloat16(int aInt) : value(FromFloat(float(aInt)).value)
    {
    }
    explicit constexpr BFloat16(uint aVal) : value(FromFloat(float(aVal)).value)
    {
    }
    explicit constexpr BFloat16(long64 aVal) : value(FromFloat(float(aVal)).value)
    {
    }
    explicit constexpr BFloat16(ulong64 aVal) : value(FromFloat(float(aVal)).value)
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
    DEVICE_ONLY_CODE explicit BFloat16(long64 aVal) : value(__ll2bfloat16_ru(aVal))
    {
    }
    DEVICE_ONLY_CODE explicit BFloat16(ulong64 aVal) : value(__ull2bfloat16_ru(aVal))
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
        return BFloat16(aUShort, true);
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
        return BFloat16(ushort(u.int32 >> 16), true);
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
        return BFloat16(ushort((u.int32 >> 16) | (!(~u.int32 & 0x7f800000) && (u.int32 & 0xffff))), true);
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
        } u = {uint(value) << 16};
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
        return float(*this) < float(aOther);
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
        return float(*this) <= float(aOther);
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
        return float(*this) > float(aOther);
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
        return float(*this) >= float(aOther);
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
        return float(*this) == float(aOther);
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
        return float(*this) == aOther;
    }
    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline bool operator==(int aOther) const
    {
        return float(*this) == float(aOther);
    }
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline bool operator!=(BFloat16 aOther) const
    {
#ifdef IS_HOST_COMPILER
        return float(*this) != float(aOther);
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
        return BFloat16(float(*this) + float(aOther));
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
        return BFloat16(float(*this) - float(aOther));
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
        return BFloat16(float(*this) * float(aOther));
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
        return BFloat16(float(*this) / float(aOther));
#endif

#ifdef IS_CUDA_COMPILER
        return BFloat16(value / aOther.value);
#endif
    }

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    inline void Exp()
    {
        *this = BFloat16(std::exp(float(*this)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline void Exp()
    {
        value = hexp(value);
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] inline static BFloat16 Exp(BFloat16 aOther)
    {
        return BFloat16(std::exp(float(aOther)));
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
    inline void Ln()
    {
        *this = BFloat16(std::log(float(*this)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline void Ln()
    {
        value = hlog(value);
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] inline static BFloat16 Ln(BFloat16 aOther)
    {
        return BFloat16(std::log(float(aOther)));
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
    inline void Sqrt()
    {
        *this = BFloat16(std::sqrt(float(*this)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline void Sqrt()
    {
        value = hsqrt(value);
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] inline static BFloat16 Sqrt(BFloat16 aOther)
    {
        return BFloat16(std::sqrt(float(aOther)));
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
    DEVICE_CODE inline void Abs()
    {
#ifdef IS_HOST_COMPILER
        *this = BFloat16(std::abs(float(*this)));
#endif
#ifdef IS_CUDA_COMPILER
        value = __habs(value);
#endif
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline static BFloat16 Abs(BFloat16 aOther)
    {
#ifdef IS_HOST_COMPILER
        return BFloat16(std::abs(float(aOther)));
#endif
#ifdef IS_CUDA_COMPILER
        return BFloat16(__habs(aOther.value));
#endif
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE inline void Min(const BFloat16 &aOther)
    {
#ifdef IS_HOST_COMPILER
        *this = BFloat16(std::min(float(*this), float(aOther)));
#endif
#ifdef IS_CUDA_COMPILER
        value = __hmin(value, aOther.value);
#endif
    }

    /// <summary>
    /// Minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline static BFloat16 Min(const BFloat16 &aLeft, const BFloat16 &aRight)
    {
#ifdef IS_HOST_COMPILER
        return BFloat16(std::min(float(aLeft), float(aRight)));
#endif
#ifdef IS_CUDA_COMPILER
        return BFloat16(__hmin(aLeft.value, aRight.value));
#endif
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE inline void Max(const BFloat16 &aOther)
    {
#ifdef IS_HOST_COMPILER
        *this = BFloat16(std::max(float(*this), float(aOther)));
#endif
#ifdef IS_CUDA_COMPILER
        value = __hmax(value, aOther.value);
#endif
    }

    /// <summary>
    /// Maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline static BFloat16 Max(const BFloat16 &aLeft, const BFloat16 &aRight)
    {
#ifdef IS_HOST_COMPILER
        return BFloat16(std::max(float(aLeft), float(aRight)));
#endif
#ifdef IS_CUDA_COMPILER
        return BFloat16(__hmax(aLeft.value, aRight.value));
#endif
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE inline void Round()
    {
#ifdef IS_HOST_COMPILER
        *this = BFloat16(std::round(float(*this)));
#endif
#ifdef IS_CUDA_COMPILER
        *this = BFloat16(round(float(value)));
#endif
    }

    /// <summary>
    /// round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline static BFloat16 Round(BFloat16 aOther)
    {
#ifdef IS_HOST_COMPILER
        return BFloat16(std::round(float(aOther)));
#endif
#ifdef IS_CUDA_COMPILER
        return BFloat16(round(float(aOther.value)));
#endif
    }

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    inline void Floor()
    {
        *this = BFloat16(std::floor(float(*this)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline void Floor()
    {
        value = __int2bfloat16_rd(__bfloat162int_rd(value));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// floor()
    /// </summary>
    [[nodiscard]] inline static BFloat16 Floor(BFloat16 aOther)
    {
        return BFloat16(std::floor(float(aOther)));
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
    inline void Ceil()
    {
        *this = BFloat16(std::ceil(float(*this)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline void Ceil()
    {
        value = __int2bfloat16_ru(__bfloat162int_ru(value));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// ceil()
    /// </summary>
    [[nodiscard]] inline static BFloat16 Ceil(BFloat16 aOther)
    {
        return BFloat16(std::ceil(float(aOther)));
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
    inline void RoundNearest()
    {
        *this = BFloat16(std::nearbyint(float(*this)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_ONLY_CODE inline void RoundNearest()
    {
        value = __int2bfloat16_rn(__bfloat162int_rn(value));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    [[nodiscard]] inline static BFloat16 RoundNearest(BFloat16 aOther)
    {
        return BFloat16(std::nearbyint(float(aOther)));
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
    inline void RoundZero()
    {
        *this = BFloat16(std::trunc(float(*this)));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    DEVICE_ONLY_CODE inline void RoundZero()
    {
        value = __int2bfloat16_rz(__bfloat162int_rz(value));
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
};

inline std::ostream &operator<<(std::ostream &aOs, const BFloat16 &aHalf)
{
    return aOs << float(aHalf);
}
inline std::wostream &operator<<(std::wostream &aOs, const BFloat16 &aHalf)
{
    return aOs << float(aHalf);
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
} // namespace opp