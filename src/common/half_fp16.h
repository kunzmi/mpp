#pragma once
#include "mpp_defs.h"
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

namespace mpp
{
/// <summary>
/// We implement our own Half-FP16 type, as different devices use different implementations. This is meant to wrap them
/// all together to one type: On CPU we use the half-library from Christian Rau, on Cuda devices the fp16 header from
/// Nvidia and on AMD devices the implementation coming with ROCm. But from an external view, it is always the same
/// mpp::HalfFp16 datatype.
/// </summary>
class alignas(2) HalfFp16
{
  private:
    static constexpr bool BINARY = true; // additional argument for constructor to switch to binary
#ifdef IS_HOST_COMPILER
    half_float::half value;
#endif
#ifdef IS_CUDA_COMPILER
    __half value;
#endif

  public:
    HalfFp16() noexcept = default;

#ifdef IS_HOST_COMPILER
    explicit constexpr HalfFp16(half_float::half aHalf) : value(aHalf) // NOLINT(performance-unnecessary-value-param)
    {
    }
    explicit HalfFp16(float aFloat);
    DEVICE_CODE explicit HalfFp16(float aFloat, RoundingMode aRoundingMode);
    explicit HalfFp16(double aDouble);
    explicit HalfFp16(long64 aLong64);
    explicit HalfFp16(ulong64 aULong64);
    explicit HalfFp16(int aInt) : value(static_cast<float>(aInt)) {};
    explicit HalfFp16(uint aUInt);
    explicit HalfFp16(short aShort);
    explicit HalfFp16(ushort aUShort);
    explicit HalfFp16(sbyte aSbyte);
    explicit HalfFp16(byte aByte);
#endif
#ifdef IS_CUDA_COMPILER
    DEVICE_CODE HalfFp16(__half aHalf);
    DEVICE_CODE explicit HalfFp16(float aFloat);
    DEVICE_CODE explicit HalfFp16(float aFloat, RoundingMode aRoundingMode);
    DEVICE_CODE explicit HalfFp16(double aDouble);
    DEVICE_CODE explicit HalfFp16(int aInt);
    DEVICE_CODE explicit HalfFp16(uint aUInt);
    DEVICE_CODE explicit HalfFp16(short aShort);
    DEVICE_CODE explicit HalfFp16(ushort aUShort);
    DEVICE_CODE explicit HalfFp16(sbyte aSbyte);
    DEVICE_CODE explicit HalfFp16(byte aByte);
#endif
    ~HalfFp16() = default;

    HalfFp16(const HalfFp16 &)     = default;
    HalfFp16(HalfFp16 &&) noexcept = default;

    HalfFp16 &operator=(const HalfFp16 &)     = default;
    HalfFp16 &operator=(HalfFp16 &&) noexcept = default;

#ifdef IS_CUDA_COMPILER
    // despite being in a "IS_CUDA_COMPILER" section, we mark it as DEVICE_CODE, i.e. including __host__ annotation so
    // that we can use DEVICE_CODE later in calling functions, like in numeric_limits.h which then avoids endles #ifdef
    // switches
    DEVICE_CODE [[nodiscard]] constexpr static HalfFp16 FromUShort(ushort aUShort)
    {
        HalfFp16 ret;
        ret.value = __half(__nv_half_raw(aUShort));
        return ret;
    }

#endif
    friend bool DEVICE_CODE isnan(HalfFp16 aVal);
    friend bool DEVICE_CODE isinf(HalfFp16 aVal);
    friend bool DEVICE_CODE isfinite(HalfFp16 aVal);

#ifdef IS_CUDA_COMPILER
    DEVICE_CODE operator __half() const;
    DEVICE_CODE explicit operator int() const;
    DEVICE_CODE explicit operator uint() const;
    DEVICE_CODE explicit operator short() const;
    DEVICE_CODE explicit operator ushort() const;
    DEVICE_CODE explicit operator byte() const;
    DEVICE_CODE explicit operator sbyte() const;
    DEVICE_CODE explicit operator double() const; // we need that one for the conversion kernel

    DEVICE_CODE explicit operator float() const; // NOLINT(hicpp-explicit-conversions)
#endif

#ifdef IS_HOST_COMPILER
    DEVICE_CODE operator float() const; // NOLINT(hicpp-explicit-conversions)

    [[nodiscard]] constexpr static HalfFp16 FromUShort(ushort aUShort)
    {
        HalfFp16 ret;
        ret.value = half_float::half(aUShort, BINARY);
        return ret;
    }
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<(HalfFp16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator<=(HalfFp16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>(HalfFp16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator>=(HalfFp16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator==(HalfFp16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] bool operator!=(HalfFp16 aOther) const;

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] HalfFp16 operator-() const;

    /// <summary>
    /// </summary>
    DEVICE_CODE HalfFp16 &operator+=(HalfFp16 aOther);

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] HalfFp16 operator+(HalfFp16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE HalfFp16 &operator-=(HalfFp16 aOther);

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] HalfFp16 operator-(HalfFp16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE HalfFp16 &operator*=(HalfFp16 aOther);

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] HalfFp16 operator*(HalfFp16 aOther) const;

    /// <summary>
    /// </summary>
    DEVICE_CODE HalfFp16 &operator/=(HalfFp16 aOther);

    // defined here in header because of inline/no-inline host/device compiler complexity...
    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] HalfFp16 operator/(HalfFp16 aOther) const // NOLINT(performance-unnecessary-value-param)
    {
        return HalfFp16(value / aOther.value);
    }

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    HalfFp16 &Exp();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE HalfFp16 &Exp();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] static HalfFp16 Exp(HalfFp16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static HalfFp16 Exp(HalfFp16 aOther);
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    HalfFp16 &Ln();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE HalfFp16 &Ln();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] static HalfFp16 Ln(HalfFp16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static HalfFp16 Ln(HalfFp16 aOther);
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    HalfFp16 &Sqrt();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE HalfFp16 &Sqrt();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] static HalfFp16 Sqrt(HalfFp16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static HalfFp16 Sqrt(HalfFp16 aOther);
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE HalfFp16 &Abs();

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] static HalfFp16 Abs(HalfFp16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static HalfFp16 Abs(HalfFp16 aOther);
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE HalfFp16 &Min(const HalfFp16 &aOther);

    /// <summary>
    /// Minimum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static HalfFp16 Min(const HalfFp16 &aLeft, const HalfFp16 &aRight);

    /// <summary>
    /// </summary>
    DEVICE_CODE HalfFp16 &Max(const HalfFp16 &aOther);

    /// <summary>
    /// Maximum
    /// </summary>
    DEVICE_CODE [[nodiscard]] static HalfFp16 Max(const HalfFp16 &aLeft, const HalfFp16 &aRight);

    /// <summary>
    /// round()
    /// </summary>
    DEVICE_CODE HalfFp16 &Round();

    /// <summary>
    /// round()
    /// </summary>
    DEVICE_CODE [[nodiscard]] static HalfFp16 Round(HalfFp16 aOther);

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    HalfFp16 &Floor();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE HalfFp16 &Floor();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// floor()
    /// </summary>
    [[nodiscard]] static HalfFp16 Floor(HalfFp16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// floor()
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static HalfFp16 Floor(HalfFp16 aOther);
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    HalfFp16 &Ceil();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE HalfFp16 &Ceil();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// ceil()
    /// </summary>
    [[nodiscard]] static HalfFp16 Ceil(HalfFp16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// ceil()
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static HalfFp16 Ceil(HalfFp16 aOther);
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    HalfFp16 &RoundNearest();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_ONLY_CODE HalfFp16 &RoundNearest();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    [[nodiscard]] static HalfFp16 RoundNearest(HalfFp16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static HalfFp16 RoundNearest(HalfFp16 aOther);
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    HalfFp16 &RoundZero();
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    DEVICE_ONLY_CODE HalfFp16 &RoundZero();
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    [[nodiscard]] static HalfFp16 RoundZero(HalfFp16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static HalfFp16 RoundZero(HalfFp16 aOther);
#endif

    DEVICE_CODE [[nodiscard]] HalfFp16 GetSign() const;

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static HalfFp16 Sin(HalfFp16 aOther);
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] static HalfFp16 Cos(HalfFp16 aOther);
#endif

#ifdef IS_HOST_COMPILER
    friend std::ostream &operator<<(std::ostream &aOs, const mpp::HalfFp16 &aHalf);
    friend std::wostream &operator<<(std::wostream &aOs, const mpp::HalfFp16 &aHalf);
#endif
};

// bfloat literal: 2.4_hf is read as HalfFp16 when mpp namespace is used
inline HalfFp16 operator"" _hf(long double aValue)
{
    return HalfFp16(static_cast<float>(aValue));
}

#ifdef IS_CUDA_COMPILER
DEVICE_CODE inline bool isnan(HalfFp16 aVal)
{
    return __hisnan(aVal.value);
}
DEVICE_CODE inline bool isinf(HalfFp16 aVal)
{
    return __hisinf(aVal.value);
}
DEVICE_CODE inline bool isfinite(HalfFp16 aVal)
{
    return !(__hisinf(aVal.value) || __hisnan(aVal.value));
}
#endif

#ifdef IS_HOST_COMPILER
DEVICE_CODE inline bool isnan(HalfFp16 aVal) // NOLINT(performance-unnecessary-value-param)
{
    return half_float::isnan(aVal.value);
}
DEVICE_CODE inline bool isinf(HalfFp16 aVal) // NOLINT(performance-unnecessary-value-param)
{
    return half_float::isinf(aVal.value);
}
DEVICE_CODE inline bool isfinite(HalfFp16 aVal) // NOLINT(performance-unnecessary-value-param)
{
    return half_float::isfinite(aVal.value);
}

std::ostream &operator<<(std::ostream &aOs, const mpp::HalfFp16 &aHalf);
std::wostream &operator<<(std::wostream &aOs, const mpp::HalfFp16 &aHalf);
std::istream &operator>>(std::istream &aIs, mpp::HalfFp16 &aHalf);
std::wistream &operator>>(std::wistream &aIs, mpp::HalfFp16 &aHalf);
#endif

template <RealNumber T> DEVICE_CODE HalfFp16 operator+(const HalfFp16 &aLeft, T aRight)
{
    return HalfFp16{aLeft + static_cast<HalfFp16>(aRight)};
}
template <RealNumber T> DEVICE_CODE HalfFp16 operator+(T aLeft, const HalfFp16 &aRight)
{
    return HalfFp16{static_cast<HalfFp16>(aLeft) + aRight};
}
template <RealNumber T> DEVICE_CODE HalfFp16 operator-(const HalfFp16 &aLeft, T aRight)
{
    return HalfFp16{aLeft - static_cast<HalfFp16>(aRight)};
}
template <RealNumber T> DEVICE_CODE HalfFp16 operator-(T aLeft, const HalfFp16 &aRight)
{
    return HalfFp16{static_cast<HalfFp16>(aLeft) - aRight};
}
template <RealNumber T> DEVICE_CODE HalfFp16 operator*(const HalfFp16 &aLeft, T aRight)
{
    return HalfFp16{aLeft * static_cast<HalfFp16>(aRight)};
}
template <RealNumber T> DEVICE_CODE HalfFp16 operator*(T aLeft, const HalfFp16 &aRight)
{
    return HalfFp16{static_cast<HalfFp16>(aLeft) * aRight};
}
template <RealNumber T> DEVICE_CODE HalfFp16 operator/(const HalfFp16 &aLeft, T aRight)
{
    return HalfFp16{aLeft / static_cast<HalfFp16>(aRight)};
}
template <RealNumber T> DEVICE_CODE HalfFp16 operator/(T aLeft, const HalfFp16 &aRight)
{
    return HalfFp16{static_cast<HalfFp16>(aLeft) / aRight};
}

// 16 bit half precision floating point types
template <> struct numeric_limits<HalfFp16>
{
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 min() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0x0400)); // NOLINT
    }
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 max() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0x7BFF)); // NOLINT
    }
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 lowest() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0xFBFF)); // NOLINT
    }
    // minimum exact integer in float = -(2^11) = -2048
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 minExact() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0xE800)); // NOLINT
    }
    // maximum exact integer in float = 2^11 = 2048
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 maxExact() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0x6800)); // NOLINT
    }
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 infinity() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0x7c00)); // NOLINT
    }
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 quiet_NaN() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0x7e00)); // NOLINT
    }
};
} // namespace mpp