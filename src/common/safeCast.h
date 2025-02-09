#pragma once
#include "defines.h"
#include <cassert>
#include <concepts>
#include <cstddef>
#include <type_traits>

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

namespace opp
{
// template <typename T_toTest, typename T_required>
// concept isSameType = std::is_same<T_toTest, T_required>::value;

template <typename T_From, typename T_FromShouldBe, typename T_To, typename T_ToShouldBe>
concept selectCase = std::same_as<T_FromShouldBe, T_From> && std::same_as<T_ToShouldBe, T_To>;

template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires std::same_as<T_To, T_From>
{
    return true;
}

// Note: we use hard coded values for numeric limits as we need the values in source data type. std::numeric_limits<T>
// would give us the value in target type which then needs to be casted to source type and we risk to introduce errors

#ifdef __APPLE__
// On MAC size_t and ptr_diff are not equivalent to ulong64 and long64
// They are defined as unsigned long int and long int, whereas ulong64 is defined as unsigned long long
// On MAC / clang we have thus to add this new case

// To = float
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long int, T_To, float>
{
    return aValue <= 16777216 && // maximum exact integer in float
           aValue >= -16777216;  // minimum exact integer in float
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, unsigned long int, T_To, float>
{
    return aValue <= 16777216; // maximum exact integer in float
}

// To = double
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long int, T_To, double>
{
    return aValue < 9007199254740993LL && // maximum exact integer in double
           aValue >= -9007199254740992LL; // minimum exact integer in double
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, unsigned long int, T_To, double>
{
    return aValue < 9007199254740993UL; // maximum exact integer in double
}

// To = long int
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, float, T_To, long int>
{
    // ignore rounding errors
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, double, T_To, long int>
{
    // ignore rounding errors
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, unsigned long int, T_To, long int>
{
    return aValue < 9223372036854775808ULL;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, int, T_To, long int>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, uint, T_To, long int>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, short, T_To, long int>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ushort, T_To, long int>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, sbyte, T_To, long int>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, byte, T_To, long int>
{
    return true;
}

// To = unsigned long int
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires std::same_as<T_To, unsigned long int> && std::is_unsigned<T_From>::value
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires std::same_as<T_To, unsigned long int>
{
    return aValue >= 0;
}

// To = int
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long int, T_To, int>
{
    return aValue < 2147483648LL && aValue > -2147483649LL;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, unsigned long int, T_To, int>
{
    return aValue < 2147483648ULL;
}

// To = uint
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long int, T_To, uint>
{
    return aValue < 4294967296LL && aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, unsigned long int, T_To, uint>
{
    return aValue < 4294967296ULL;
}

// To = short
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long int, T_To, short>
{
    return aValue < 32768 && aValue >= -32768;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, unsigned long int, T_To, short>
{
    return aValue < 32768;
}

// To = ushort
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long int, T_To, ushort>
{
    return aValue < 65536 && aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, unsigned long int, T_To, ushort>
{
    return aValue < 65536;
}

// To = sbyte
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long int, T_To, sbyte>
{
    return aValue < 128 && aValue > -129;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, unsigned long int, T_To, sbyte>
{
    return aValue < 128;
}

// To = byte
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long int, T_To, byte>
{
    return aValue < 256 && aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, unsigned long int, T_To, byte>
{
    return aValue < 256;
}
#endif

// To = float
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, double, T_To, float>
{
    // we don't check here and hope the that user knows what he's doing
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long64, T_To, float>
{
    return aValue <= 16777216 && // maximum exact integer in float
           aValue >= -16777216;  // minimum exact integer in float
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ulong64, T_To, float>
{
    return aValue <= 16777216; // maximum exact integer in float
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, int, T_To, float>
{
    return aValue <= 16777216 && // maximum exact integer in float
           aValue >= -16777216;  // minimum exact integer in float
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, uint, T_To, float>
{
    return aValue <= 16777216; // maximum exact integer in float
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, short, T_To, float>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, ushort, T_To, float>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, sbyte, T_To, float>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, byte, T_To, float>
{
    return true;
}

// To = double
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, float, T_To, double>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long64, T_To, double>
{
    return aValue < 9007199254740993LL && // maximum exact integer in double
           aValue >= -9007199254740992LL; // minimum exact integer in double
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ulong64, T_To, double>
{
    return aValue < 9007199254740993UL; // maximum exact integer in double
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, int, T_To, double>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, uint, T_To, double>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, short, T_To, double>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, ushort, T_To, double>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, sbyte, T_To, double>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, byte, T_To, double>
{
    return true;
}

// To = long64
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, float, T_To, long64>
{
    // ignore rounding errors
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, double, T_To, long64>
{
    // ignore rounding errors
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ulong64, T_To, long64>
{
    return aValue < 9223372036854775808ULL;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, int, T_To, long64>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, uint, T_To, long64>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, short, T_To, long64>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, ushort, T_To, long64>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, sbyte, T_To, long64>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, byte, T_To, long64>
{
    return true;
}

// To = ulong64
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires(std::same_as<T_To, ulong64> && std::is_unsigned_v<T_From> && !std::same_as<T_To, T_From>)
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires(std::same_as<T_To, ulong64> && !std::is_unsigned_v<T_From> && !std::same_as<T_To, T_From>)
{
    return aValue >= 0;
}

// To = int
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, float, T_To, int>
{
    return aValue < 2147483648LL && long64(aValue) > -2147483649LL;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, double, T_To, int>
{
    return aValue < 2147483648LL && aValue > -2147483649LL;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long64, T_To, int>
{
    return aValue < 2147483648LL && aValue > -2147483649LL;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ulong64, T_To, int>
{
    return aValue < 2147483648ULL;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, uint, T_To, int>
{
    return aValue < 2147483648UL;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, short, T_To, int>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, ushort, T_To, int>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, sbyte, T_To, int>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, byte, T_To, int>
{
    return true;
}

// To = uint
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, float, T_To, uint>
{
    return aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, double, T_To, uint>
{
    return aValue >= 0 && aValue < 4294967296LL;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long64, T_To, uint>
{
    return aValue < 4294967296LL && aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ulong64, T_To, uint>
{
    return aValue < 4294967296ULL;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, int, T_To, uint>
{
    return aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, short, T_To, uint>
{
    return aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, ushort, T_To, uint>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, sbyte, T_To, uint>
{
    return aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, byte, T_To, uint>
{
    return true;
}

// To = short
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, float, T_To, short>
{
    return aValue < 32768 && aValue >= -32768;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, double, T_To, short>
{
    return aValue < 32768 && aValue >= -32768;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long64, T_To, short>
{
    return aValue < 32768 && aValue >= -32768;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ulong64, T_To, short>
{
    return aValue < 32768;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, int, T_To, short>
{
    return aValue < 32768 && aValue >= -32768;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, uint, T_To, short>
{
    return aValue < 32768;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ushort, T_To, short>
{
    return aValue < 32768;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, sbyte, T_To, short>
{
    return true;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, byte, T_To, short>
{
    return true;
}

// To = ushort
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, float, T_To, ushort>
{
    return aValue >= 0 && aValue < 65536;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, double, T_To, ushort>
{
    return aValue >= 0 && aValue < 65536;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long64, T_To, ushort>
{
    return aValue < 65536 && aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ulong64, T_To, ushort>
{
    return aValue < 65536;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, int, T_To, ushort>
{
    return aValue < 65536 && aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, uint, T_To, ushort>
{
    return aValue < 65536;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, short, T_To, ushort>
{
    return aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, sbyte, T_To, ushort>
{
    return aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/)
    requires selectCase<T_From, byte, T_To, ushort>
{
    return true;
}

// To = sbyte
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, float, T_To, sbyte>
{
    return aValue < 128 && aValue > -129;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, double, T_To, sbyte>
{
    return aValue < 128 && aValue > -129;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long64, T_To, sbyte>
{
    return aValue < 128 && aValue > -129;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ulong64, T_To, sbyte>
{
    return aValue < 128;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, int, T_To, sbyte>
{
    return aValue < 128 && aValue > -129;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, uint, T_To, sbyte>
{
    return aValue < 128;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, short, T_To, sbyte>
{
    return aValue < 128 && aValue > -129;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ushort, T_To, sbyte>
{
    return aValue < 128;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, byte, T_To, sbyte>
{
    return aValue < 128;
}

// To = byte
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, float, T_To, byte>
{
    return aValue < 256 && aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, double, T_To, byte>
{
    return aValue < 256 && aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, long64, T_To, byte>
{
    return aValue < 256 && aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ulong64, T_To, byte>
{
    return aValue < 256;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, int, T_To, byte>
{
    return aValue < 256 && aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, uint, T_To, byte>
{
    return aValue < 256;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, short, T_To, byte>
{
    return aValue < 256 && aValue >= 0;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, ushort, T_To, byte>
{
    return aValue < 256;
}
template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From aValue)
    requires selectCase<T_From, sbyte, T_To, byte>
{
    return aValue >= 0;
}

template <typename T_To, typename T_From>
DEVICE_CODE constexpr bool check_is_safe_cast(T_From /*aValue*/) // all other cases
{
    static_assert(
        AlwaysFalse<T_To>::value,
        "It seems we missed a type combination to check..."); // we shouldn't get here or we missed a combination...
    return false;
}

// checks with assert that a given value can be safely casted to the requested type

// float
template <typename T_From> DEVICE_CODE constexpr float to_float(T_From aValue)
{
    assert(check_is_safe_cast<float>(aValue));
    return static_cast<float>(aValue);
}

// double
template <typename T_From> DEVICE_CODE constexpr double to_double(T_From aValue)
{
    assert(check_is_safe_cast<double>(aValue));
    return static_cast<double>(aValue);
}

// int
template <typename T_From> DEVICE_CODE constexpr int to_int(T_From aValue)
{
    assert(check_is_safe_cast<int>(aValue));
    return static_cast<int>(aValue);
}

// uint
template <typename T_From> DEVICE_CODE constexpr uint to_uint(T_From aValue)
{
    assert(check_is_safe_cast<uint>(aValue));
    return static_cast<uint>(aValue);
}

// ulong64
template <typename T_From> DEVICE_CODE constexpr ulong64 to_ulong64(T_From aValue)
{
    assert(check_is_safe_cast<ulong64>(aValue));
    return static_cast<ulong64>(aValue);
}

// size_t (same as ulong64)
template <typename T_From> DEVICE_CODE constexpr size_t to_size_t(T_From aValue)
{
    static_assert(sizeof(size_t) == 8);
#ifdef __APPLE__
    static_assert(std::is_same<size_t, unsigned long int>::value);
#else
    static_assert(std::is_same_v<size_t, ulong64>);
#endif
    assert(check_is_safe_cast<size_t>(aValue));
    return static_cast<size_t>(aValue);
}

// long64
template <typename T_From> DEVICE_CODE constexpr long64 to_long64(T_From aValue)
{
    assert(check_is_safe_cast<long64>(aValue));
    return static_cast<long64>(aValue);
}

// ushort
template <typename T_From> DEVICE_CODE constexpr ushort to_ushort(T_From aValue)
{
    assert(check_is_safe_cast<ushort>(aValue));
    return static_cast<ushort>(aValue);
}

// short
template <typename T_From> DEVICE_CODE constexpr short to_short(T_From aValue)
{
    assert(check_is_safe_cast<short>(aValue));
    return static_cast<short>(aValue);
}

// byte
template <typename T_From> DEVICE_CODE constexpr byte to_byte(T_From aValue)
{
    assert(check_is_safe_cast<byte>(aValue));
    return static_cast<byte>(aValue);
}

// sbyte
template <typename T_From> DEVICE_CODE constexpr sbyte to_sbyte(T_From aValue)
{
    assert(check_is_safe_cast<sbyte>(aValue));
    return static_cast<sbyte>(aValue);
}

} // namespace opp

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)