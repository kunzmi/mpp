#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/safeCast.h>
#include <cstddef>
#include <cstdint>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#ifndef __APPLE__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#endif // defined(__GNUC__)

#if defined(__clang__)
#pragma clang diagnostic push
#ifndef _MSC_VER
#ifndef __APPLE__
#pragma clang diagnostic ignored "-Wuseless-cast"
#endif
#endif // !_MSC_VER
#endif // defined(__clang__)

using namespace mpp;

TEST_CASE("SafeCast", "[Common]")
{
    CHECK(check_is_safe_cast<byte>(sbyte(-1)) == false);
    CHECK(check_is_safe_cast<byte>(short(-1)) == false);
    CHECK(check_is_safe_cast<byte>(int(-1)) == false);
    CHECK(check_is_safe_cast<byte>(long64(-1)) == false);
    CHECK(check_is_safe_cast<byte>(float(-1)) == false);
    CHECK(check_is_safe_cast<byte>(double(-1)) == false);

    CHECK(check_is_safe_cast<byte>(sbyte(0)) == true);
    CHECK(check_is_safe_cast<byte>(short(0)) == true);
    CHECK(check_is_safe_cast<byte>(int(0)) == true);
    CHECK(check_is_safe_cast<byte>(long64(0)) == true);
    CHECK(check_is_safe_cast<byte>(float(0)) == true);
    CHECK(check_is_safe_cast<byte>(double(0)) == true);

    CHECK(check_is_safe_cast<byte>(sbyte(127)) == true);
    CHECK(check_is_safe_cast<byte>(short(256)) == false);
    CHECK(check_is_safe_cast<byte>(int(256)) == false);
    CHECK(check_is_safe_cast<byte>(long64(256)) == false);
    CHECK(check_is_safe_cast<byte>(float(256)) == false);
    CHECK(check_is_safe_cast<byte>(double(256)) == false);

    CHECK(check_is_safe_cast<byte>(byte(255)) == true);
    CHECK(check_is_safe_cast<byte>(ushort(256)) == false);
    CHECK(check_is_safe_cast<byte>(uint(256)) == false);
    CHECK(check_is_safe_cast<byte>(ulong64(256)) == false);

    CHECK(check_is_safe_cast<ushort>(sbyte(-1)) == false);
    CHECK(check_is_safe_cast<ushort>(short(-1)) == false);
    CHECK(check_is_safe_cast<ushort>(int(-1)) == false);
    CHECK(check_is_safe_cast<ushort>(long64(-1)) == false);
    CHECK(check_is_safe_cast<ushort>(float(-1)) == false);
    CHECK(check_is_safe_cast<ushort>(double(-1)) == false);

    CHECK(check_is_safe_cast<ushort>(sbyte(0)) == true);
    CHECK(check_is_safe_cast<ushort>(short(0)) == true);
    CHECK(check_is_safe_cast<ushort>(int(0)) == true);
    CHECK(check_is_safe_cast<ushort>(long64(0)) == true);
    CHECK(check_is_safe_cast<ushort>(float(0)) == true);
    CHECK(check_is_safe_cast<ushort>(double(0)) == true);

    CHECK(check_is_safe_cast<ushort>(sbyte(127)) == true);
    CHECK(check_is_safe_cast<ushort>(short(32767)) == true);
    CHECK(check_is_safe_cast<ushort>(int(65536)) == false);
    CHECK(check_is_safe_cast<ushort>(long64(65536)) == false);
    CHECK(check_is_safe_cast<ushort>(float(65536)) == false);
    CHECK(check_is_safe_cast<ushort>(double(65536)) == false);

    CHECK(check_is_safe_cast<ushort>(byte(255)) == true);
    CHECK(check_is_safe_cast<ushort>(ushort(65535)) == true);
    CHECK(check_is_safe_cast<ushort>(uint(65536)) == false);
    CHECK(check_is_safe_cast<ushort>(ulong64(65536)) == false);

    CHECK(check_is_safe_cast<uint>(sbyte(-1)) == false);
    CHECK(check_is_safe_cast<uint>(short(-1)) == false);
    CHECK(check_is_safe_cast<uint>(int(-1)) == false);
    CHECK(check_is_safe_cast<uint>(long64(-1)) == false);
    CHECK(check_is_safe_cast<uint>(float(-1)) == false);
    CHECK(check_is_safe_cast<uint>(double(-1)) == false);

    CHECK(check_is_safe_cast<uint>(sbyte(0)) == true);
    CHECK(check_is_safe_cast<uint>(short(0)) == true);
    CHECK(check_is_safe_cast<uint>(int(0)) == true);
    CHECK(check_is_safe_cast<uint>(long64(0)) == true);
    CHECK(check_is_safe_cast<uint>(float(0)) == true);
    CHECK(check_is_safe_cast<uint>(double(0)) == true);

    CHECK(check_is_safe_cast<uint>(sbyte(127)) == true);
    CHECK(check_is_safe_cast<uint>(short(32767)) == true);
    CHECK(check_is_safe_cast<uint>(int(2147483647)) == true);
    CHECK(check_is_safe_cast<uint>(long64(4294967296U)) == false);
    CHECK(check_is_safe_cast<uint>(float(16777216)) == true);
    CHECK(check_is_safe_cast<uint>(double(4294967296U)) == false);

    CHECK(check_is_safe_cast<uint>(byte(255)) == true);
    CHECK(check_is_safe_cast<uint>(ushort(65535)) == true);
    CHECK(check_is_safe_cast<uint>(uint(4294967295U)) == true);
    CHECK(check_is_safe_cast<uint>(ulong64(4294967296U)) == false);

    CHECK(check_is_safe_cast<ulong64>(sbyte(-1)) == false);
    CHECK(check_is_safe_cast<ulong64>(short(-1)) == false);
    CHECK(check_is_safe_cast<ulong64>(int(-1)) == false);
    CHECK(check_is_safe_cast<ulong64>(long64(-1)) == false);
    CHECK(check_is_safe_cast<ulong64>(float(-1)) == false);
    CHECK(check_is_safe_cast<ulong64>(double(-1)) == false);

    CHECK(check_is_safe_cast<ulong64>(sbyte(0)) == true);
    CHECK(check_is_safe_cast<ulong64>(short(0)) == true);
    CHECK(check_is_safe_cast<ulong64>(int(0)) == true);
    CHECK(check_is_safe_cast<ulong64>(long64(0)) == true);
    CHECK(check_is_safe_cast<ulong64>(float(0)) == true);
    CHECK(check_is_safe_cast<ulong64>(double(0)) == true);

    CHECK(check_is_safe_cast<ulong64>(sbyte(127)) == true);
    CHECK(check_is_safe_cast<ulong64>(short(32767)) == true);
    CHECK(check_is_safe_cast<ulong64>(int(2147483647)) == true);
    CHECK(check_is_safe_cast<ulong64>(long64(4294967296U)) == true);
    CHECK(check_is_safe_cast<ulong64>(float(16777216)) == true);
    CHECK(check_is_safe_cast<ulong64>(double(9007199254740993LL)) == true);

    CHECK(check_is_safe_cast<ulong64>(byte(255)) == true);
    CHECK(check_is_safe_cast<ulong64>(ushort(65535)) == true);
    CHECK(check_is_safe_cast<ulong64>(uint(4294967295U)) == true);
    CHECK(check_is_safe_cast<ulong64>(ulong64(4294967296U)) == true);

    CHECK(check_is_safe_cast<sbyte>(sbyte(-1)) == true);
    CHECK(check_is_safe_cast<sbyte>(short(-1)) == true);
    CHECK(check_is_safe_cast<sbyte>(int(-1)) == true);
    CHECK(check_is_safe_cast<sbyte>(long64(-1)) == true);
    CHECK(check_is_safe_cast<sbyte>(float(-1)) == true);
    CHECK(check_is_safe_cast<sbyte>(double(-1)) == true);

    CHECK(check_is_safe_cast<sbyte>(sbyte(-128)) == true);
    CHECK(check_is_safe_cast<sbyte>(short(-129)) == false);
    CHECK(check_is_safe_cast<sbyte>(int(-129)) == false);
    CHECK(check_is_safe_cast<sbyte>(long64(-129)) == false);
    CHECK(check_is_safe_cast<sbyte>(float(-129)) == false);
    CHECK(check_is_safe_cast<sbyte>(double(-129)) == false);

    CHECK(check_is_safe_cast<sbyte>(sbyte(127)) == true);
    CHECK(check_is_safe_cast<sbyte>(short(128)) == false);
    CHECK(check_is_safe_cast<sbyte>(int(128)) == false);
    CHECK(check_is_safe_cast<sbyte>(long64(128)) == false);
    CHECK(check_is_safe_cast<sbyte>(float(128)) == false);
    CHECK(check_is_safe_cast<sbyte>(double(128)) == false);

    CHECK(check_is_safe_cast<sbyte>(byte(128)) == false);
    CHECK(check_is_safe_cast<sbyte>(ushort(128)) == false);
    CHECK(check_is_safe_cast<sbyte>(uint(128)) == false);
    CHECK(check_is_safe_cast<sbyte>(ulong64(128)) == false);

    CHECK(check_is_safe_cast<short>(sbyte(-1)) == true);
    CHECK(check_is_safe_cast<short>(short(-1)) == true);
    CHECK(check_is_safe_cast<short>(int(-1)) == true);
    CHECK(check_is_safe_cast<short>(long64(-1)) == true);
    CHECK(check_is_safe_cast<short>(float(-1)) == true);
    CHECK(check_is_safe_cast<short>(double(-1)) == true);

    CHECK(check_is_safe_cast<short>(sbyte(-128)) == true);
    CHECK(check_is_safe_cast<short>(short(-32768)) == true);
    CHECK(check_is_safe_cast<short>(int(-32769)) == false);
    CHECK(check_is_safe_cast<short>(long64(-32769)) == false);
    CHECK(check_is_safe_cast<short>(float(-32769)) == false);
    CHECK(check_is_safe_cast<short>(double(-32769)) == false);

    CHECK(check_is_safe_cast<short>(sbyte(127)) == true);
    CHECK(check_is_safe_cast<short>(short(32767)) == true);
    CHECK(check_is_safe_cast<short>(int(32768)) == false);
    CHECK(check_is_safe_cast<short>(long64(32768)) == false);
    CHECK(check_is_safe_cast<short>(float(32768)) == false);
    CHECK(check_is_safe_cast<short>(double(32768)) == false);

    CHECK(check_is_safe_cast<short>(byte(255)) == true);
    CHECK(check_is_safe_cast<short>(ushort(32768)) == false);
    CHECK(check_is_safe_cast<short>(uint(32768)) == false);
    CHECK(check_is_safe_cast<short>(ulong64(32768)) == false);

    CHECK(check_is_safe_cast<int>(sbyte(-1)) == true);
    CHECK(check_is_safe_cast<int>(short(-1)) == true);
    CHECK(check_is_safe_cast<int>(int(-1)) == true);
    CHECK(check_is_safe_cast<int>(long64(-1)) == true);
    CHECK(check_is_safe_cast<int>(float(-1)) == true);
    CHECK(check_is_safe_cast<int>(double(-1)) == true);

    CHECK(check_is_safe_cast<int>(sbyte(-128)) == true);
    CHECK(check_is_safe_cast<int>(short(-32768)) == true);
    CHECK(check_is_safe_cast<int>(int(-2147483648)) == true);
    CHECK(check_is_safe_cast<int>(long64(-2147483649)) == false);
    // all numbers larger than -2147483904 in float are casted to -2147483648
    CHECK(check_is_safe_cast<int>(float(-2147483904)) == false);
    CHECK(check_is_safe_cast<int>(double(-2147483649)) == false);

    CHECK(check_is_safe_cast<int>(sbyte(127)) == true);
    CHECK(check_is_safe_cast<int>(short(32767)) == true);
    CHECK(check_is_safe_cast<int>(int(2147483647)) == true);
    CHECK(check_is_safe_cast<int>(long64(2147483648)) == false);
    CHECK(check_is_safe_cast<int>(float(2147483648)) == false);
    CHECK(check_is_safe_cast<int>(double(2147483648)) == false);

    CHECK(check_is_safe_cast<int>(byte(255)) == true);
    CHECK(check_is_safe_cast<int>(ushort(32768)) == true);
    CHECK(check_is_safe_cast<int>(uint(2147483648)) == false);
    CHECK(check_is_safe_cast<int>(ulong64(2147483648)) == false);

    CHECK(check_is_safe_cast<long64>(sbyte(-1)) == true);
    CHECK(check_is_safe_cast<long64>(short(-1)) == true);
    CHECK(check_is_safe_cast<long64>(int(-1)) == true);
    CHECK(check_is_safe_cast<long64>(long64(-1)) == true);
    CHECK(check_is_safe_cast<long64>(float(-1)) == true);
    CHECK(check_is_safe_cast<long64>(double(-1)) == true);

    CHECK(check_is_safe_cast<long64>(sbyte(-128)) == true);
    CHECK(check_is_safe_cast<long64>(short(-32768)) == true);
    CHECK(check_is_safe_cast<long64>(int(-2147483648)) == true);
    CHECK(check_is_safe_cast<long64>(long64(-2147483649)) == true);
    CHECK(check_is_safe_cast<long64>(float(-2147483904)) == true);
    CHECK(check_is_safe_cast<long64>(double(-2147483649)) == true);

    CHECK(check_is_safe_cast<long64>(sbyte(127)) == true);
    CHECK(check_is_safe_cast<long64>(short(32767)) == true);
    CHECK(check_is_safe_cast<long64>(int(2147483647)) == true);
    CHECK(check_is_safe_cast<long64>(long64(9223372036854775807)) == true);
    CHECK(check_is_safe_cast<long64>(float(2147483648)) == true);
    CHECK(check_is_safe_cast<long64>(double(2147483648)) == true);

    CHECK(check_is_safe_cast<long64>(byte(255)) == true);
    CHECK(check_is_safe_cast<long64>(ushort(32768)) == true);
    CHECK(check_is_safe_cast<long64>(uint(2147483648)) == true);
    CHECK(check_is_safe_cast<long64>(ulong64(9223372036854775808ULL)) == false);

    CHECK(check_is_safe_cast<float>(sbyte(-1)) == true);
    CHECK(check_is_safe_cast<float>(short(-1)) == true);
    CHECK(check_is_safe_cast<float>(int(-1)) == true);
    CHECK(check_is_safe_cast<float>(long64(-1)) == true);
    CHECK(check_is_safe_cast<float>(float(-1)) == true);
    CHECK(check_is_safe_cast<float>(double(-1)) == true);

    CHECK(check_is_safe_cast<float>(sbyte(-128)) == true);
    CHECK(check_is_safe_cast<float>(short(-32768)) == true);
    CHECK(check_is_safe_cast<float>(int(-16777217)) == false);
    CHECK(check_is_safe_cast<float>(long64(-16777217)) == false);
    CHECK(check_is_safe_cast<float>(float(-16777216)) == true);
    CHECK(check_is_safe_cast<float>(double(-16777219)) == true);

    CHECK(check_is_safe_cast<float>(sbyte(127)) == true);
    CHECK(check_is_safe_cast<float>(short(32767)) == true);
    CHECK(check_is_safe_cast<float>(int(16777217)) == false);
    CHECK(check_is_safe_cast<float>(long64(16777217)) == false);
    CHECK(check_is_safe_cast<float>(float(16777216)) == true);
    CHECK(check_is_safe_cast<float>(double(16777217)) == true);

    CHECK(check_is_safe_cast<float>(byte(255)) == true);
    CHECK(check_is_safe_cast<float>(ushort(32768)) == true);
    CHECK(check_is_safe_cast<float>(uint(16777217)) == false);
    CHECK(check_is_safe_cast<float>(ulong64(16777217)) == false);

    CHECK(check_is_safe_cast<double>(sbyte(-1)) == true);
    CHECK(check_is_safe_cast<double>(short(-1)) == true);
    CHECK(check_is_safe_cast<double>(int(-1)) == true);
    CHECK(check_is_safe_cast<double>(long64(-1)) == true);
    CHECK(check_is_safe_cast<double>(float(-1)) == true);
    CHECK(check_is_safe_cast<double>(double(-1)) == true);

    CHECK(check_is_safe_cast<double>(sbyte(-128)) == true);
    CHECK(check_is_safe_cast<double>(short(-32768)) == true);
    CHECK(check_is_safe_cast<double>(int(-16777219)) == true);
    CHECK(check_is_safe_cast<double>(long64(-9007199254740995LL)) == false);
    CHECK(check_is_safe_cast<double>(float(-9007199254740995LL)) == true);
    CHECK(check_is_safe_cast<double>(double(-9007199254740995LL)) == true);

    CHECK(check_is_safe_cast<double>(sbyte(127)) == true);
    CHECK(check_is_safe_cast<double>(short(32767)) == true);
    CHECK(check_is_safe_cast<double>(int(16777217)) == true);
    CHECK(check_is_safe_cast<double>(long64(9007199254740993LL)) == false);
    CHECK(check_is_safe_cast<double>(float(9007199254740993LL)) == true);
    CHECK(check_is_safe_cast<double>(double(9007199254740993LL)) == true);

    CHECK(check_is_safe_cast<double>(byte(255)) == true);
    CHECK(check_is_safe_cast<double>(ushort(32768)) == true);
    CHECK(check_is_safe_cast<double>(uint(16777217)) == true);
    CHECK(check_is_safe_cast<double>(ulong64(9007199254740993UL)) == false);
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif // defined(__GNUC__)

#if defined(__clang__)
#pragma clang diagnostic pop
#endif // defined(__clang__)
