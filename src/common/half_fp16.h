#pragma once
#include "exception.h"
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

namespace opp
{
/// <summary>
/// We implement our own Half-FP16 type, as different devices use different implementations. This is meant to wrap them
/// all together to one type: On CPU we use the half-library from Christian Rau, on Cuda devices the fp16 header from
/// Nvidia and on AMD devices the implementation coming with ROCm. But from an external view, it is always the same
/// opp::HalfFp16 datatype.
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
    explicit constexpr HalfFp16(half_float::half aHalf) : value(aHalf)
    {
    }
    explicit HalfFp16(float aFloat) : value(aFloat)
    {
    }
    DEVICE_CODE explicit HalfFp16(float aFloat, RoundingMode aRoundingMode)
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
    explicit HalfFp16(double aDouble) : value(static_cast<float>(aDouble))
    {
    }
    explicit HalfFp16(int aInt) : value(static_cast<float>(aInt))
    {
    }
    explicit HalfFp16(uint aUInt) : value(static_cast<float>(aUInt))
    {
    }
    explicit HalfFp16(short aShort) : value(static_cast<float>(aShort))
    {
    }
    explicit HalfFp16(ushort aUShort) : value(static_cast<float>(aUShort))
    {
    }
    explicit HalfFp16(sbyte aSbyte) : value(static_cast<float>(aSbyte))
    {
    }
    explicit HalfFp16(byte aByte) : value(static_cast<float>(aByte))
    {
    }
#endif
#ifdef IS_CUDA_COMPILER
    DEVICE_CODE explicit HalfFp16(__half aHalf) : value(aHalf)
    {
    }
    DEVICE_CODE explicit HalfFp16(float aFloat) : value(__float2half_rn(aFloat))
    {
    }
    DEVICE_CODE explicit HalfFp16(float aFloat, RoundingMode aRoundingMode)
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
    DEVICE_CODE explicit HalfFp16(double aDouble) : value(__double2half(aDouble))
    {
    }
    DEVICE_CODE explicit HalfFp16(int aInt) : value(__int2half_rn(aInt))
    {
    }
    DEVICE_CODE explicit HalfFp16(uint aUInt) : value(__uint2half_rn(aUInt))
    {
    }
    DEVICE_CODE explicit HalfFp16(short aShort) : value(__short2half_rn(aShort))
    {
    }
    DEVICE_CODE explicit HalfFp16(ushort aUShort) : value(__ushort2half_rn(aUShort))
    {
    }
    DEVICE_CODE explicit HalfFp16(sbyte aSbyte) : value(__short2half_rn(aSbyte))
    {
    }
    DEVICE_CODE explicit HalfFp16(byte aByte) : value(__ushort2half_rn(aByte))
    {
    }
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
    DEVICE_CODE [[nodiscard]] inline constexpr static HalfFp16 FromUShort(ushort aUShort)
    {
        HalfFp16 ret;
        ret.value = __half(__nv_half_raw(aUShort));
        return ret;
    }

#endif
    friend bool DEVICE_CODE isnan(HalfFp16);
    friend bool DEVICE_CODE isinf(HalfFp16);

    DEVICE_CODE inline operator float() const
    {
#ifdef IS_HOST_COMPILER
        return static_cast<float>(value);
#endif
#ifdef IS_CUDA_COMPILER
        return __half2float(value);
#endif
    }

#ifdef IS_CUDA_COMPILER
    DEVICE_CODE inline operator int() const
    {
        return __half2int_rz(value);
    }
    DEVICE_CODE inline operator uint() const
    {
        return __half2uint_rz(value);
    }
    DEVICE_CODE inline operator short() const
    {
        return __half2short_rz(value);
    }
    DEVICE_CODE inline operator ushort() const
    {
        return __half2ushort_rz(value);
    }
    DEVICE_CODE inline operator byte() const
    {
        return __half2uchar_rz(value);
    }
    DEVICE_CODE inline operator sbyte() const
    {
        return __half2char_rz(value);
    }
#endif

#ifdef IS_HOST_COMPILER
    [[nodiscard]] inline constexpr static HalfFp16 FromUShort(ushort aUShort)
    {
        HalfFp16 ret;
        ret.value = half_float::half(aUShort, BINARY);
        return ret;
    }
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline bool operator<(HalfFp16 aOther) const
    {
        return value < aOther.value;
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline bool operator<=(HalfFp16 aOther) const
    {
        return value <= aOther.value;
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline bool operator>(HalfFp16 aOther) const
    {
        return value > aOther.value;
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline bool operator>=(HalfFp16 aOther) const
    {
        return value >= aOther.value;
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline bool operator==(HalfFp16 aOther) const
    {
        return value == aOther.value;
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline bool operator!=(HalfFp16 aOther) const
    {
        return value != aOther.value;
    }

    /// <summary>
    /// Negation
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline HalfFp16 operator-() const
    {
        return HalfFp16(-value);
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE inline HalfFp16 &operator+=(HalfFp16 aOther)
    {
        value += aOther.value;
        return *this;
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline HalfFp16 operator+(HalfFp16 aOther) const
    {
        return HalfFp16(value + aOther.value);
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE inline HalfFp16 &operator-=(HalfFp16 aOther)
    {
        value -= aOther.value;
        return *this;
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline HalfFp16 operator-(HalfFp16 aOther) const
    {
        return HalfFp16(value - aOther.value);
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE inline HalfFp16 &operator*=(HalfFp16 aOther)
    {
        value *= aOther.value;
        return *this;
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline HalfFp16 operator*(HalfFp16 aOther) const
    {
        return HalfFp16(value * aOther.value);
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE inline HalfFp16 &operator/=(HalfFp16 aOther)
    {
        value /= aOther.value;
        return *this;
    }

    /// <summary>
    /// </summary>
    DEVICE_CODE [[nodiscard]] inline HalfFp16 operator/(HalfFp16 aOther) const
    {
        return HalfFp16(value / aOther.value);
    }

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    inline HalfFp16 &Exp()
    {
        value = half_float::exp(value);
        return *this;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline HalfFp16 &Exp()
    {
        value = hexp(value);
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] inline static HalfFp16 Exp(HalfFp16 aOther)
    {
        return HalfFp16(half_float::exp(aOther.value));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static HalfFp16 Exp(HalfFp16 aOther)
    {
        return HalfFp16(hexp(aOther.value));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    inline HalfFp16 &Ln()
    {
        value = half_float::log(value);
        return *this;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline HalfFp16 &Ln()
    {
        value = hlog(value);
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] inline static HalfFp16 Ln(HalfFp16 aOther)
    {
        return HalfFp16(half_float::log(aOther.value));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static HalfFp16 Ln(HalfFp16 aOther)
    {
        return HalfFp16(hlog(aOther.value));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    inline HalfFp16 &Sqrt()
    {
        value = half_float::sqrt(value);
        return *this;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline HalfFp16 &Sqrt()
    {
        value = hsqrt(value);
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    [[nodiscard]] inline static HalfFp16 Sqrt(HalfFp16 aOther)
    {
        return HalfFp16(half_float::sqrt(aOther.value));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static HalfFp16 Sqrt(HalfFp16 aOther)
    {
        return HalfFp16(hsqrt(aOther.value));
    }
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE inline HalfFp16 &Abs()
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
    [[nodiscard]] inline static HalfFp16 Abs(HalfFp16 aOther)
    {
        return HalfFp16(half_float::abs(aOther.value));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static HalfFp16 Abs(HalfFp16 aOther)
    {
        return HalfFp16(__habs(aOther.value));
    }
#endif

    /// <summary>
    /// </summary>
    DEVICE_CODE inline HalfFp16 &Min(const HalfFp16 &aOther)
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
    DEVICE_CODE [[nodiscard]] inline static HalfFp16 Min(const HalfFp16 &aLeft, const HalfFp16 &aRight)
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
    DEVICE_CODE inline HalfFp16 &Max(const HalfFp16 &aOther)
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
    DEVICE_CODE [[nodiscard]] inline static HalfFp16 Max(const HalfFp16 &aLeft, const HalfFp16 &aRight)
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
    DEVICE_CODE inline HalfFp16 &Round()
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
    DEVICE_CODE [[nodiscard]] inline static HalfFp16 Round(HalfFp16 aOther)
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
    inline HalfFp16 &Floor()
    {
        value = half_float::floor(value);
        return *this;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline HalfFp16 &Floor()
    {
        value = hfloor(value);
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// floor()
    /// </summary>
    [[nodiscard]] inline static HalfFp16 Floor(HalfFp16 aOther)
    {
        return HalfFp16(half_float::floor(aOther.value));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// floor()
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static HalfFp16 Floor(HalfFp16 aOther)
    {
        return HalfFp16(hfloor(aOther.value));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// </summary>
    inline HalfFp16 &Ceil()
    {
        value = half_float::ceil(value);
        return *this;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE inline HalfFp16 &Ceil()
    {
        value = hceil(value);
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// ceil()
    /// </summary>
    [[nodiscard]] inline static HalfFp16 Ceil(HalfFp16 aOther)
    {
        return HalfFp16(half_float::ceil(aOther.value));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// ceil()
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static HalfFp16 Ceil(HalfFp16 aOther)
    {
        return HalfFp16(hceil(aOther.value));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    inline HalfFp16 &RoundNearest()
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
    DEVICE_ONLY_CODE inline HalfFp16 &RoundNearest()
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
    [[nodiscard]] inline static HalfFp16 RoundNearest(HalfFp16 aOther)
    {
        return HalfFp16(half_float::nearbyint(aOther.value));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round nearest ties to even<para/>
    /// Note: the host function assumes that current rounding mode is set to FE_TONEAREST
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static HalfFp16 RoundNearest(HalfFp16 aOther)
    {
        return HalfFp16(hrint(aOther.value));
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    inline HalfFp16 &RoundZero()
    {
        value = half_float::trunc(value);
        return *this;
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    DEVICE_ONLY_CODE inline HalfFp16 &RoundZero()
    {
        value = htrunc(value);
        return *this;
    }
#endif

#ifdef IS_HOST_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    [[nodiscard]] inline static HalfFp16 RoundZero(HalfFp16 aOther)
    {
        return HalfFp16(half_float::trunc(aOther.value));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// Round toward zero
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static HalfFp16 RoundZero(HalfFp16 aOther)
    {
        return HalfFp16(htrunc(aOther.value));
    }
#endif

    DEVICE_CODE inline HalfFp16 GetSign()
    {
        constexpr ushort ONE_AS_HFLOAT = 0x3c00;
        constexpr ushort SIGN_BIT      = 0x8000;
        ushort hfloatbits              = *reinterpret_cast<ushort *>(&value);
        hfloatbits &= SIGN_BIT;
        hfloatbits |= ONE_AS_HFLOAT;
        return *reinterpret_cast<HalfFp16 *>(&hfloatbits);
    }

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static HalfFp16 Sin(HalfFp16 aOther)
    {
        return HalfFp16(hsin(aOther.value));
    }
#endif

#ifdef IS_CUDA_COMPILER
    /// <summary>
    /// </summary>
    DEVICE_ONLY_CODE [[nodiscard]] inline static HalfFp16 Cos(HalfFp16 aOther)
    {
        return HalfFp16(hcos(aOther.value));
    }
#endif

#ifdef IS_HOST_COMPILER
    friend std::ostream &operator<<(std::ostream &aOs, const HalfFp16 &aHalf);
    friend std::wostream &operator<<(std::wostream &aOs, const HalfFp16 &aHalf);
#endif
};

#ifdef IS_HOST_COMPILER
inline std::ostream &operator<<(std::ostream &aOs, const HalfFp16 &aHalf)
{
    return aOs << static_cast<float>(aHalf.value);
}
inline std::wostream &operator<<(std::wostream &aOs, const HalfFp16 &aHalf)
{
    return aOs << static_cast<float>(aHalf.value);
}
inline std::istream &operator>>(std::istream &aIs, HalfFp16 &aHalf)
{
    float temp;
    aIs >> temp;
    aHalf = HalfFp16(temp);
    return aIs;
}
inline std::wistream &operator>>(std::wistream &aIs, HalfFp16 &aHalf)
{
    float temp;
    aIs >> temp;
    aHalf = HalfFp16(temp);
    return aIs;
}
#endif

// bfloat literal: 2.4_hf is read as HalfFp16 when opp namespace is used
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
#endif

#ifdef IS_HOST_COMPILER
DEVICE_CODE inline bool isnan(HalfFp16 aVal)
{
    return half_float::isnan(aVal.value);
}
DEVICE_CODE inline bool isinf(HalfFp16 aVal)
{
    return half_float::isinf(aVal.value);
}
#endif

// 16 bit half precision floating point types
template <> struct numeric_limits<HalfFp16>
{
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 min() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0x0400));
    }
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 max() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0x7BFF));
    }
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 lowest() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0xFBFF));
    }
    // minimum exact integer in float = -(2^11) = -2048
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 minExact() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0xE800));
    }
    // maximum exact integer in float = 2^11 = 2048
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 maxExact() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0x6800));
    }
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 infinity() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0x7c00));
    }
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 quiet_NaN() noexcept
    {
        return HalfFp16::FromUShort(static_cast<ushort>(0x7e00));
    }
};

//// numeric limits when converting from one type to another, especially 16-Bit floats have some restrictions
// template <typename TTo> struct numeric_limits_conversion<HalfFp16, TTo>
//{
//     // HalfFp16 does not have a constexpr constructor on cuda device...
//     [[nodiscard]] static const DEVICE_CODE HalfFp16 min() noexcept
//     {
//         return static_cast<HalfFp16>(numeric_limits<TTo>::min());
//     }
//     [[nodiscard]] static const DEVICE_CODE HalfFp16 max() noexcept
//     {
//         return static_cast<HalfFp16>(numeric_limits<TTo>::max());
//     }
//     [[nodiscard]] static const DEVICE_CODE HalfFp16 lowest() noexcept
//     {
//         return static_cast<HalfFp16>(numeric_limits<TTo>::lowest());
//     }
//     [[nodiscard]] static const DEVICE_CODE HalfFp16 minExact() noexcept
//     {
//         return static_cast<HalfFp16>(numeric_limits<TTo>::minExact());
//     }
//     [[nodiscard]] static const DEVICE_CODE HalfFp16 maxExact() noexcept
//     {
//         return static_cast<HalfFp16>(numeric_limits<TTo>::maxExact());
//     }
// };
//
// template <> struct numeric_limits_conversion<HalfFp16, short>
//{
//     [[nodiscard]] static const DEVICE_CODE HalfFp16 min() noexcept
//     {
//         return static_cast<HalfFp16>(numeric_limits<short>::min());
//     }
//     [[nodiscard]] static const DEVICE_CODE HalfFp16 max() noexcept
//     {
//         // special case for half floats: the maximum value of short is slightly larger than the closest exact
//         // integer in HalfFp16, and as we use round to nearest, the clamping would result in a too large number.
//         // Thus for HalfFp16 and short, we clamp to the exact integer smaller than short::max(), i.e. 32752
//         return HalfFp16::FromUShort(0x77FF); // = 32752
//     }
//     [[nodiscard]] static const DEVICE_CODE HalfFp16 lowest() noexcept
//     {
//         return static_cast<HalfFp16>(numeric_limits<short>::lowest());
//     }
//     [[nodiscard]] static const DEVICE_CODE HalfFp16 minExact() noexcept
//     {
//         return static_cast<HalfFp16>(numeric_limits<short>::minExact());
//     }
//     [[nodiscard]] static const DEVICE_CODE HalfFp16 maxExact() noexcept
//     {
//         return static_cast<HalfFp16>(numeric_limits<short>::maxExact());
//     }
// };
} // namespace opp