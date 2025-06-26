#pragma once
#include "mpp_defs.h"
#include <common/defines.h>
#include <common/numeric_limits.h>
#include <concepts>
#include <iostream>
#ifdef IS_CUDA_COMPILER
#include <cuda_bf16.h>
#endif

namespace mpp
{
/// <summary>
/// We implement our own BFloat16 type, as different devices use different implementations. This is meant to wrap them
/// all together to one type: On CPU we use our own implementation (which is based on amd_hip_bfloat16.h), on Cuda
/// devices the bf16 header from Nvidia and on AMD devices the implementation coming with ROCm. But from an external
/// view, it is always the same mpp::BFloat16 datatype.
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
#endif

    friend bool DEVICE_CODE isnan(BFloat16 aVal);
    friend bool DEVICE_CODE isinf(BFloat16 aVal);

  public:
    BFloat16() noexcept = default;

#ifdef IS_HOST_COMPILER
    explicit constexpr BFloat16(float aFloat) : value(FromFloat(aFloat).value)
    {
    }
    DEVICE_CODE explicit BFloat16(float aFloat, RoundingMode aRoundingMode);

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
    DEVICE_CODE BFloat16(__nv_bfloat16 aBFloat);
    DEVICE_CODE explicit BFloat16(float aFloat);
    DEVICE_CODE explicit BFloat16(float aFloat, RoundingMode aRoundingMode);
    DEVICE_CODE explicit BFloat16(sbyte aVal);
    DEVICE_CODE explicit BFloat16(byte aVal);
    DEVICE_CODE explicit BFloat16(short aVal);
    DEVICE_CODE explicit BFloat16(ushort aVal);
    DEVICE_CODE explicit BFloat16(int aVal);
    DEVICE_CODE explicit BFloat16(uint aVal);
    DEVICE_ONLY_CODE explicit BFloat16(long64 aVal);
    DEVICE_ONLY_CODE explicit BFloat16(ulong64 aVal);
    DEVICE_ONLY_CODE explicit BFloat16(double aVal);
#endif
    ~BFloat16() = default;

    BFloat16(const BFloat16 &)     = default;
    BFloat16(BFloat16 &&) noexcept = default;

    BFloat16 &operator=(const BFloat16 &)     = default;
    BFloat16 &operator=(BFloat16 &&) noexcept = default;

    DEVICE_CODE [[nodiscard]] constexpr static BFloat16 FromUShort(ushort aUShort)
    {
        return BFloat16(aUShort, BINARY);
    }

#ifdef IS_HOST_COMPILER
    [[nodiscard]] constexpr static BFloat16 FromFloat(float aFloat)
    {
        union
        {
            float fp32;
            uint int32;
        } u = {aFloat};
        if (~u.int32 & 0x7f800000) // NOLINT
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
            //
            // Round to nearest, round to even
            u.int32 += 0x7fff + ((u.int32 >> 16) & 1); // NOLINT
        }
        else if (u.int32 & 0xffff) // NOLINT
        {
            // When all of the exponent bits are 1, the value is Inf or NaN.
            // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
            // mantissa bit. Quiet NaN is indicated by the most significant mantissa
            // bit being 1. Signaling NaN is indicated by the most significant
            // mantissa bit being 0 but some other bit(s) being 1. If any of the
            // lower 16 bits of the mantissa are 1, we set the least significant bit
            // of the bfloat16 mantissa, in order to preserve signaling NaN in case
            // the bloat16's mantissa bits are all 0.
            //
            // Preserve signaling NaN
            u.int32 |= 0x10000; // NOLINT
        }
        return BFloat16(static_cast<ushort>(u.int32 >> 16), BINARY); // NOLINT
    }

    /// <summary>
    /// Truncate instead of rounding, preserving SNaN
    /// </summary>
    [[nodiscard]] static BFloat16 FromFloatTruncate(float aFloat);

    // zero extend lower 16 bits of bfloat16 to convert to IEEE float
    DEVICE_CODE operator float() const; // NOLINT(hicpp-explicit-conversions)
#endif

#ifdef IS_CUDA_COMPILER
    DEVICE_CODE explicit operator float() const;
    DEVICE_CODE operator __nv_bfloat16() const;
    DEVICE_CODE explicit operator int() const;
    DEVICE_CODE explicit operator uint() const;
    DEVICE_CODE explicit operator short() const;
    DEVICE_CODE explicit operator ushort() const;
    DEVICE_CODE explicit operator byte() const;
    DEVICE_CODE explicit operator sbyte() const;
    DEVICE_CODE explicit operator double() const;
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<(BFloat16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<=(BFloat16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>(BFloat16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>=(BFloat16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator==(BFloat16 aOther) const;

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator==(float aOther) const;
    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator==(int aOther) const;
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator!=(BFloat16 aOther) const;

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Negation
    /// </summary>
    [[nodiscard]] constexpr BFloat16 operator-() const
    {
        BFloat16 ret = *this;
        ret.value ^= 0x8000; // NOLINT
        return ret;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] BFloat16 operator-() const;
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE BFloat16 &operator+=(BFloat16 aOther);

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] BFloat16 operator+(BFloat16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE BFloat16 &operator-=(BFloat16 aOther);

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] BFloat16 operator-(BFloat16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE BFloat16 &operator*=(BFloat16 aOther);

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] BFloat16 operator*(BFloat16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE BFloat16 &operator/=(BFloat16 aOther);

    // defined here in header because of inline/no-inline host/device compiler complexity...
    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] BFloat16 operator/(BFloat16 aOther) const
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
    BFloat16 &Exp();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE BFloat16 &Exp();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] static BFloat16 Exp(BFloat16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static BFloat16 Exp(BFloat16 aOther);
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    BFloat16 &Ln();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE BFloat16 &Ln();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] static BFloat16 Ln(BFloat16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static BFloat16 Ln(BFloat16 aOther);
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    BFloat16 &Sqrt();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE BFloat16 &Sqrt();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] static BFloat16 Sqrt(BFloat16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static BFloat16 Sqrt(BFloat16 aOther);
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE BFloat16 &Abs();

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] static BFloat16 Abs(BFloat16 aOther);

    /// <summary>
    /// </summary>
    DEVICE_CODE BFloat16 &Min(const BFloat16 &aOther);

    /// <summary>
    /// Minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static BFloat16 Min(const BFloat16 &aLeft, const BFloat16 &aRight);

    /// <summary>
    /// </summary>
    DEVICE_CODE BFloat16 &Max(const BFloat16 &aOther);

    /// <summary>
    /// Maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static BFloat16 Max(const BFloat16 &aLeft, const BFloat16 &aRight);

    /// <summary>
    /// </summary>
    DEVICE_CODE BFloat16 &Round();

    /// <summary>
    /// round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static BFloat16 Round(BFloat16 aOther);

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    BFloat16 &Floor();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE BFloat16 &Floor();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// floor()
    /// </summary>
    [[nodiscard]] static BFloat16 Floor(BFloat16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// floor()
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static BFloat16 Floor(BFloat16 aOther);
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    BFloat16 &Ceil();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE BFloat16 &Ceil();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// ceil()
    /// </summary>
    [[nodiscard]] static BFloat16 Ceil(BFloat16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// ceil()
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static BFloat16 Ceil(BFloat16 aOther);
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    BFloat16 &RoundNearest();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_ONLY_CODE BFloat16 &RoundNearest();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    [[nodiscard]] static BFloat16 RoundNearest(BFloat16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static BFloat16 RoundNearest(BFloat16 aOther);
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    BFloat16 &RoundZero();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    DEVICE_ONLY_CODE BFloat16 &RoundZero();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    [[nodiscard]] static BFloat16 RoundZero(BFloat16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static BFloat16 RoundZero(BFloat16 aOther);
#endif

    DEVICE_CODE [[nodiscard]] BFloat16 GetSign() const;

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static BFloat16 Sin(BFloat16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static BFloat16 Cos(BFloat16 aOther);
#endif
};

// bfloat literal: 2.4_bf is read as BFloat16 when mpp namespace is used
inline BFloat16 operator"" _bf(long double aValue)
{
    return BFloat16(float(aValue));
}

#ifdef IS_CUDA_COMPILER
DEVICE_CODE inline bool isnan(BFloat16 aVal)
{
    return __hisnan(aVal.value);
}
DEVICE_CODE inline bool isinf(BFloat16 aVal)
{
    return __hisinf(aVal.value);
}
#endif

#ifdef IS_HOST_COMPILER
DEVICE_CODE inline bool isnan(BFloat16 aVal)
{
    return std::isnan(static_cast<float>(aVal));
}
DEVICE_CODE inline bool isinf(BFloat16 aVal)
{
    return std::isinf(static_cast<float>(aVal));
}
#endif

// 16 bit BFloat floating point types
template <> struct numeric_limits<BFloat16>
{
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 min() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0x0400)); // NOLINT
    }
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 max() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0x7F7F)); // NOLINT
    }
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 lowest() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0xFF7F)); // NOLINT
    }
    // minimum exact integer in float = -(2^8) = -256
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 minExact() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0xC380)); // NOLINT
    }
    // maximum exact integer in float = 2^8 = 256
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 maxExact() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0x4380)); // NOLINT
    }
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 infinity() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0x7f80)); // NOLINT
    }
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 quiet_NaN() noexcept
    {
        return BFloat16::FromUShort(static_cast<ushort>(0x7fc0)); // NOLINT
    }
};

std::ostream &operator<<(std::ostream &aOs, const BFloat16 &aHalf);
std::wostream &operator<<(std::wostream &aOs, const BFloat16 &aHalf);
std::istream &operator>>(std::istream &aIs, BFloat16 &aHalf);
std::wistream &operator>>(std::wistream &aIs, BFloat16 &aHalf);

} // namespace mpp
