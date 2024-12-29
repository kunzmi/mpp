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
};

// 16 bit half precision floating point types
template <> struct numeric_limits<HalfFp16>
{
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 min() noexcept
    {
        return HalfFp16::FromUShort(ushort(0x0400));
    }
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 max() noexcept
    {
        return HalfFp16::FromUShort(ushort(0x7BFF));
    }
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 lowest() noexcept
    {
        return HalfFp16::FromUShort(ushort(0xFBFF));
    }
    // minimum exact integer in float = -(2^11) = -2048
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 minExact() noexcept
    {
        return HalfFp16::FromUShort(ushort(0xE800));
    }
    // maximum exact integer in float = 2^11 = 2048
    [[nodiscard]] static constexpr DEVICE_CODE HalfFp16 maxExact() noexcept
    {
        return HalfFp16::FromUShort(ushort(0x6800));
    }
};

// 16 bit BFloat floating point types
template <> struct numeric_limits<BFloat16>
{
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 min() noexcept
    {
        return BFloat16::FromUShort(ushort(0x0400));
    }
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 max() noexcept
    {
        return BFloat16::FromUShort(ushort(0x7F7F));
    }
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 lowest() noexcept
    {
        return BFloat16::FromUShort(ushort(0xFF7F));
    }
    // minimum exact integer in float = -(2^8) = -256
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 minExact() noexcept
    {
        return BFloat16::FromUShort(ushort(0xC380));
    }
    // maximum exact integer in float = 2^9 = 256
    [[nodiscard]] static constexpr DEVICE_CODE BFloat16 maxExact() noexcept
    {
        return BFloat16::FromUShort(ushort(0x4380));
    }
};

} // namespace opp