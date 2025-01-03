#pragma once
#include "defines.h"
#include <cfloat>
#include <climits>
#include <common/bfloat16.h>
#include <common/half_fp16.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

// re-define std::numeric_limits min/max for datatypes that we use, as std::limits is not available on device
namespace opp
{
// all types as base, return whatever makes sense...
template <typename T> struct numeric_limits
{
    [[nodiscard]] static constexpr DEVICE_CODE T min() noexcept
    {
        return T();
    }
    [[nodiscard]] static constexpr DEVICE_CODE T max() noexcept
    {
        return T();
    }
    [[nodiscard]] static constexpr DEVICE_CODE T lowest() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE T minExact() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE T maxExact() noexcept
    {
        return max();
    }
};

// unsigned integer types
template <> struct numeric_limits<byte>
{
    [[nodiscard]] static constexpr DEVICE_CODE byte min() noexcept
    {
        return 0;
    }
    [[nodiscard]] static constexpr DEVICE_CODE byte max() noexcept
    {
        return UCHAR_MAX;
    }
    [[nodiscard]] static constexpr DEVICE_CODE byte lowest() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE byte minExact() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE byte maxExact() noexcept
    {
        return max();
    }
};

template <> struct numeric_limits<ushort>
{
    [[nodiscard]] static constexpr DEVICE_CODE ushort min() noexcept
    {
        return 0;
    }
    [[nodiscard]] static constexpr DEVICE_CODE ushort max() noexcept
    {
        return USHRT_MAX;
    }
    [[nodiscard]] static constexpr DEVICE_CODE ushort lowest() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE ushort minExact() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE ushort maxExact() noexcept
    {
        return max();
    }
};

template <> struct numeric_limits<uint>
{
    [[nodiscard]] static constexpr DEVICE_CODE uint min() noexcept
    {
        return 0;
    }
    [[nodiscard]] static constexpr DEVICE_CODE uint max() noexcept
    {
        return UINT_MAX;
    }
    [[nodiscard]] static constexpr DEVICE_CODE uint lowest() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE uint minExact() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE uint maxExact() noexcept
    {
        return max();
    }
};

template <> struct numeric_limits<ulong64>
{
    [[nodiscard]] static constexpr DEVICE_CODE ulong64 min() noexcept
    {
        return 0;
    }
    [[nodiscard]] static constexpr DEVICE_CODE ulong64 max() noexcept
    {
        return ULLONG_MAX;
    }
    [[nodiscard]] static constexpr DEVICE_CODE ulong64 lowest() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE ulong64 minExact() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE ulong64 maxExact() noexcept
    {
        return max();
    }
};

// signed integer types
template <> struct numeric_limits<sbyte>
{
    [[nodiscard]] static constexpr DEVICE_CODE sbyte min() noexcept
    {
        return SCHAR_MIN;
    }
    [[nodiscard]] static constexpr DEVICE_CODE sbyte max() noexcept
    {
        return SCHAR_MAX;
    }
    [[nodiscard]] static constexpr DEVICE_CODE sbyte lowest() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE sbyte minExact() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE sbyte maxExact() noexcept
    {
        return max();
    }
};

template <> struct numeric_limits<short>
{
    [[nodiscard]] static constexpr DEVICE_CODE short min() noexcept
    {
        return SHRT_MIN;
    }
    [[nodiscard]] static constexpr DEVICE_CODE short max() noexcept
    {
        return SHRT_MAX;
    }
    [[nodiscard]] static constexpr DEVICE_CODE short lowest() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE short minExact() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE short maxExact() noexcept
    {
        return max();
    }
};

template <> struct numeric_limits<int>
{
    [[nodiscard]] static constexpr DEVICE_CODE int min() noexcept
    {
        return INT_MIN;
    }
    [[nodiscard]] static constexpr DEVICE_CODE int max() noexcept
    {
        return INT_MAX;
    }
    [[nodiscard]] static constexpr DEVICE_CODE int lowest() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE int minExact() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE int maxExact() noexcept
    {
        return max();
    }
};

template <> struct numeric_limits<long64>
{
    [[nodiscard]] static constexpr DEVICE_CODE long64 min() noexcept
    {
        return LLONG_MIN;
    }
    [[nodiscard]] static constexpr DEVICE_CODE long64 max() noexcept
    {
        return LLONG_MAX;
    }
    [[nodiscard]] static constexpr DEVICE_CODE long64 lowest() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE long64 minExact() noexcept
    {
        return min();
    }
    [[nodiscard]] static constexpr DEVICE_CODE long64 maxExact() noexcept
    {
        return max();
    }
};

// floating point types
template <> struct numeric_limits<float>
{
    [[nodiscard]] static constexpr DEVICE_CODE float min() noexcept
    {
        return FLT_MIN;
    }
    [[nodiscard]] static constexpr DEVICE_CODE float max() noexcept
    {
        return FLT_MAX;
    }
    [[nodiscard]] static constexpr DEVICE_CODE float lowest() noexcept
    {
        return -max();
    }
    // minimum exact integer in float = -(2^24)
    [[nodiscard]] static constexpr DEVICE_CODE float minExact() noexcept
    {
        return -16777216.0f;
    }
    // maximum exact integer in float = 2^24
    [[nodiscard]] static constexpr DEVICE_CODE float maxExact() noexcept
    {
        return 16777216.0f;
    }
    [[nodiscard]] static constexpr DEVICE_CODE float infinity() noexcept
    {
        return std::numeric_limits<float>::infinity();
    }
    [[nodiscard]] static constexpr DEVICE_CODE float quiet_NaN() noexcept
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
};

template <> struct numeric_limits<double>
{
    [[nodiscard]] static constexpr DEVICE_CODE double min() noexcept
    {
        return DBL_MIN;
    }
    [[nodiscard]] static constexpr DEVICE_CODE double max() noexcept
    {
        return DBL_MAX;
    }
    [[nodiscard]] static constexpr DEVICE_CODE double lowest() noexcept
    {
        return -max();
    }
    // minimum exact integer in double = -(2^53)
    [[nodiscard]] static constexpr DEVICE_CODE double minExact() noexcept
    {
        return -9007199254740992.0;
    }
    // maximum exact integer in double = 2^53
    [[nodiscard]] static constexpr DEVICE_CODE double maxExact() noexcept
    {
        return 9007199254740992.0;
    }
    [[nodiscard]] static constexpr DEVICE_CODE double infinity() noexcept
    {
        return std::numeric_limits<double>::infinity();
    }
    [[nodiscard]] static constexpr DEVICE_CODE double quiet_NaN() noexcept
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

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
template <typename TFrom, typename TTo> struct numeric_limits_conversion
{
    [[nodiscard]] static constexpr DEVICE_CODE TFrom min() noexcept
    {
        return static_cast<TFrom>(numeric_limits<TTo>::min());
    }
    [[nodiscard]] static constexpr DEVICE_CODE TFrom max() noexcept
    {
        return static_cast<TFrom>(numeric_limits<TTo>::max());
    }
    [[nodiscard]] static constexpr DEVICE_CODE TFrom lowest() noexcept
    {
        return static_cast<TFrom>(numeric_limits<TTo>::lowest());
    }
    [[nodiscard]] static constexpr DEVICE_CODE TFrom minExact() noexcept
    {
        return static_cast<TFrom>(numeric_limits<TTo>::minExact());
    }
    [[nodiscard]] static constexpr DEVICE_CODE TFrom maxExact() noexcept
    {
        return static_cast<TFrom>(numeric_limits<TTo>::maxExact());
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

template <> struct numeric_limits_conversion<float, int>
{
    // BFloat16 does not have a constexpr constructor on cuda device...
    [[nodiscard]] static constexpr DEVICE_CODE float min() noexcept
    {
        return static_cast<float>(numeric_limits<int>::min());
    }
    [[nodiscard]] static DEVICE_CODE float max() noexcept
    {
        // special case for floats to int: the maximum value of int is slightly smaller than the closest exact
        // integer in float, and as we use round to nearest, the clamping would result in a too large number.
        // Thus for float and int, we clamp to the next integer smaller than int::max(), i.e. 2147483520
        return 2147483520.0f; // = 32640
    }
    [[nodiscard]] static DEVICE_CODE float lowest() noexcept
    {
        return static_cast<float>(numeric_limits<int>::lowest());
    }
    [[nodiscard]] static DEVICE_CODE float minExact() noexcept
    {
        return static_cast<float>(numeric_limits<int>::minExact());
    }
    [[nodiscard]] static DEVICE_CODE float maxExact() noexcept
    {
        return static_cast<float>(numeric_limits<int>::maxExact());
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

// numeric limits when converting from one type to another, especially 16-Bit floats have some restrictions
template <typename TTo> struct numeric_limits_conversion<HalfFp16, TTo>
{
    // HalfFp16 does not have a constexpr constructor on cuda device...
    [[nodiscard]] static const DEVICE_CODE HalfFp16 min() noexcept
    {
        return static_cast<HalfFp16>(numeric_limits<TTo>::min());
    }
    [[nodiscard]] static const DEVICE_CODE HalfFp16 max() noexcept
    {
        return static_cast<HalfFp16>(numeric_limits<TTo>::max());
    }
    [[nodiscard]] static const DEVICE_CODE HalfFp16 lowest() noexcept
    {
        return static_cast<HalfFp16>(numeric_limits<TTo>::lowest());
    }
    [[nodiscard]] static const DEVICE_CODE HalfFp16 minExact() noexcept
    {
        return static_cast<HalfFp16>(numeric_limits<TTo>::minExact());
    }
    [[nodiscard]] static const DEVICE_CODE HalfFp16 maxExact() noexcept
    {
        return static_cast<HalfFp16>(numeric_limits<TTo>::maxExact());
    }
};

template <> struct numeric_limits_conversion<HalfFp16, short>
{
    [[nodiscard]] static const DEVICE_CODE HalfFp16 min() noexcept
    {
        return static_cast<HalfFp16>(numeric_limits<short>::min());
    }
    [[nodiscard]] static const DEVICE_CODE HalfFp16 max() noexcept
    {
        // special case for half floats: the maximum value of short is slightly larger than the closest exact
        // integer in HalfFp16, and as we use round to nearest, the clamping would result in a too large number.
        // Thus for HalfFp16 and short, we clamp to the exact integer smaller than short::max(), i.e. 32752
        return HalfFp16::FromUShort(0x77FF); // = 32752
    }
    [[nodiscard]] static const DEVICE_CODE HalfFp16 lowest() noexcept
    {
        return static_cast<HalfFp16>(numeric_limits<short>::lowest());
    }
    [[nodiscard]] static const DEVICE_CODE HalfFp16 minExact() noexcept
    {
        return static_cast<HalfFp16>(numeric_limits<short>::minExact());
    }
    [[nodiscard]] static const DEVICE_CODE HalfFp16 maxExact() noexcept
    {
        return static_cast<HalfFp16>(numeric_limits<short>::maxExact());
    }
};
} // namespace opp