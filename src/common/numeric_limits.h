#pragma once
#include "defines.h"
#include <cfloat>
#include <climits>
#include <cmath>
#include <common/numberTypes.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

// re-define std::numeric_limits min/max for datatypes that we use, as std::limits is not available on device
namespace mpp
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
        return -16777216.0f; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    }
    // maximum exact integer in float = 2^24
    [[nodiscard]] static constexpr DEVICE_CODE float maxExact() noexcept
    {
        return 16777216.0f; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    }
    [[nodiscard]] static constexpr DEVICE_CODE float infinity() noexcept
    {
        return INFINITY;
        // std::numeric_limits<float>::infinity();
    }
    [[nodiscard]] static DEVICE_CODE float quiet_NaN() noexcept
    {
        return NAN;
        // std::numeric_limits<float>::quiet_NaN();
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
        return -9007199254740992.0; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    }
    // maximum exact integer in double = 2^53
    [[nodiscard]] static constexpr DEVICE_CODE double maxExact() noexcept
    {
        return 9007199254740992.0; // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    }
    [[nodiscard]] static constexpr DEVICE_CODE double infinity() noexcept
    {
        return INFINITY;
        // std::numeric_limits<double>::infinity();
    }
    [[nodiscard]] static DEVICE_CODE double quiet_NaN() noexcept
    {
        return NAN;
        // return std::numeric_limits<double>::quiet_NaN();
    }
};

// numeric limits when converting from one type to another, especially 16-Bit floats have some restrictions
template <typename TFrom, typename TTo> struct numeric_limits_conversion
{
    [[nodiscard]] static constexpr DEVICE_CODE TFrom min() noexcept
    {
        return static_cast<TFrom>(numeric_limits<TTo>::min());
    }
    [[nodiscard]] static constexpr DEVICE_CODE TFrom min() noexcept
        requires RealUnsignedIntegral<TFrom>
    {
        return static_cast<TFrom>(0);
    }
    [[nodiscard]] static constexpr DEVICE_CODE TFrom max() noexcept
    {
        return static_cast<TFrom>(numeric_limits<TTo>::max());
    }
    [[nodiscard]] static constexpr DEVICE_CODE TFrom lowest() noexcept
    {
        return static_cast<TFrom>(numeric_limits<TTo>::lowest());
    }
    [[nodiscard]] static constexpr DEVICE_CODE TFrom lowest() noexcept
        requires RealUnsignedIntegral<TFrom>
    {
        return static_cast<TFrom>(0);
    }
    [[nodiscard]] static constexpr DEVICE_CODE TFrom minExact() noexcept
    {
        return static_cast<TFrom>(numeric_limits<TTo>::minExact());
    }
    [[nodiscard]] static constexpr DEVICE_CODE TFrom minExact() noexcept
        requires RealUnsignedIntegral<TFrom>
    {
        return static_cast<TFrom>(0);
    }
    [[nodiscard]] static constexpr DEVICE_CODE TFrom maxExact() noexcept
    {
        return static_cast<TFrom>(numeric_limits<TTo>::maxExact());
    }
};

template <RealSignedNumber T> struct Complex;

// ignore complex
template <typename TFrom, typename TTo>
struct numeric_limits_conversion<Complex<TFrom>, Complex<TTo>> : numeric_limits_conversion<TFrom, TTo>
{
};

// ignore complex
template <typename TFrom, typename TTo>
struct numeric_limits_conversion<Complex<TFrom>, TTo> : numeric_limits_conversion<TFrom, TTo>
{
};

// ignore complex
template <typename TFrom, typename TTo>
struct numeric_limits_conversion<TFrom, Complex<TTo>> : numeric_limits_conversion<TFrom, TTo>
{
};

// template <> struct numeric_limits_conversion<float, int>
//{
//     [[nodiscard]] static constexpr DEVICE_CODE float min() noexcept
//     {
//         return static_cast<float>(numeric_limits<int>::min());
//     }
//     [[nodiscard]] static DEVICE_CODE float max() noexcept
//     {
//         // special case for floats to int: the maximum value of int is slightly smaller than the closest exact
//         // integer in float, and as we use round to nearest, the clamping would result in a too large number.
//         // Thus for float and int, we clamp to the next integer smaller than int::max(), i.e. 2147483520
//         return 2147483520.0f;
//     }
//     [[nodiscard]] static DEVICE_CODE float lowest() noexcept
//     {
//         return static_cast<float>(numeric_limits<int>::lowest());
//     }
//     [[nodiscard]] static DEVICE_CODE float minExact() noexcept
//     {
//         return static_cast<float>(numeric_limits<int>::minExact());
//     }
//     [[nodiscard]] static DEVICE_CODE float maxExact() noexcept
//     {
//         return static_cast<float>(numeric_limits<int>::maxExact());
//     }
// };

} // namespace mpp