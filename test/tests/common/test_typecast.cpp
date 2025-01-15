#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <common/bfloat16.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/numeric_limits.h>
#include <common/utilities.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <cstdint>
#include <math.h>
#include <numeric>
#include <vector>

using namespace opp;
using namespace Catch;

TEST_CASE("Typecast with StaticCast<float> Large", "[Common]")
{
    Vec4f f1(2147483647.0f - 128.0f, 4294967295.0f, NAN, INFINITY);
    Vec4f f2(-2147483648.0f + 128.0f, -2147483648.0f - 1024.0f, NAN, -INFINITY);

    Vector4<int> i1(f1);
    Vector4<int> i2(f2);
    Vector4<uint> ui1(f1);
    Vector4<uint> ui2(f2);

    Vector4<short> s1(f1);
    Vector4<short> s2(f2);
    Vector4<ushort> us1(f1);
    Vector4<ushort> us2(f2);

    Vector4<sbyte> sb1(f1);
    Vector4<sbyte> sb2(f2);
    Vector4<byte> b1(f1);
    Vector4<byte> b2(f2);

    Vector4<BFloat16> bf1(f1);
    Vector4<BFloat16> bf2(f2);
    Vector4<HalfFp16> hf1(f1);
    Vector4<HalfFp16> hf2(f2);

    // to int
    CHECK(i1.x == 2147483520);
    CHECK(i1.y == 2147483647);
    CHECK(i1.z == 0);
    CHECK(i1.w == 2147483647);

    CHECK(i2.x == -2147483520);
    CHECK(i2.y == -2147483648);
    CHECK(i2.z == 0);
    CHECK(i2.w == -2147483648);

    // to uint
    CHECK(ui1.x == 2147483520);
    CHECK(ui1.y == 4294967295);
    CHECK(ui1.z == 0);
    CHECK(ui1.w == 4294967295);

    CHECK(ui2.x == 0);
    CHECK(ui2.y == 0);
    CHECK(ui2.z == 0);
    CHECK(ui2.w == 0);

    // to short
    CHECK(s1.x == 32767);
    CHECK(s1.y == 32767);
    CHECK(s1.z == 0);
    CHECK(s1.w == 32767);

    CHECK(s2.x == -32768);
    CHECK(s2.y == -32768);
    CHECK(s2.z == 0);
    CHECK(s2.w == -32768);

    // to ushort
    CHECK(us1.x == 65535);
    CHECK(us1.y == 65535);
    CHECK(us1.z == 0);
    CHECK(us1.w == 65535);

    CHECK(us2.x == 0);
    CHECK(us2.y == 0);
    CHECK(us2.z == 0);
    CHECK(us2.w == 0);

    // to sbyte
    CHECK(sb1.x == 127);
    CHECK(sb1.y == 127);
    CHECK(sb1.z == 0);
    CHECK(sb1.w == 127);

    CHECK(sb2.x == -128);
    CHECK(sb2.y == -128);
    CHECK(sb2.z == 0);
    CHECK(sb2.w == -128);

    // to byte
    CHECK(b1.x == 255);
    CHECK(b1.y == 255);
    CHECK(b1.z == 0);
    CHECK(b1.w == 255);

    CHECK(b2.x == 0);
    CHECK(b2.y == 0);
    CHECK(b2.z == 0);
    CHECK(b2.w == 0);

    // to bfloat
    CHECK(bf1.x == 2147483648.0f);
    CHECK(bf1.y == 4294967295.0f);
    CHECK(isnan(bf1.z));
    CHECK(bf1.w == INFINITY);

    CHECK(bf2.x == -2147483648.0f);
    CHECK(bf2.y == -2147483648.0f);
    CHECK(isnan(bf2.z));
    CHECK(bf2.w == -INFINITY);

    // to half-float
    CHECK(hf1.x == INFINITY);
    CHECK(hf1.y == INFINITY);
    CHECK(isnan(hf1.z));
    CHECK(hf1.w == INFINITY);

    CHECK(hf2.x == -INFINITY);
    CHECK(hf2.y == -INFINITY);
    CHECK(isnan(hf2.z));
    CHECK(hf2.w == -INFINITY);
}

TEST_CASE("Typecast with StaticCast<float> Small", "[Common]")
{
    Vec4f f1(39.8696189840862f, -118.857810285007f, 89.3771023024069f, 111.102271425933f);
    Vec4f f2(39.3023398854545f, -84.1762079202402f, 52.7477985330198f, -119.850791327380f);

    Vector4<int> i1(f1);
    Vector4<int> i2(f2);
    Vector4<uint> ui1(f1);
    Vector4<uint> ui2(f2);

    Vector4<short> s1(f1);
    Vector4<short> s2(f2);
    Vector4<ushort> us1(f1);
    Vector4<ushort> us2(f2);

    Vector4<sbyte> sb1(f1);
    Vector4<sbyte> sb2(f2);
    Vector4<byte> b1(f1);
    Vector4<byte> b2(f2);

    Vector4<BFloat16> bf1(f1);
    Vector4<BFloat16> bf2(f2);
    Vector4<HalfFp16> hf1(f1);
    Vector4<HalfFp16> hf2(f2);

    // to int
    CHECK(i1.x == 39);
    CHECK(i1.y == -118);
    CHECK(i1.z == 89);
    CHECK(i1.w == 111);

    CHECK(i2.x == 39);
    CHECK(i2.y == -84);
    CHECK(i2.z == 52);
    CHECK(i2.w == -119);

    // to uint
    CHECK(ui1.x == 39);
    CHECK(ui1.y == 0);
    CHECK(ui1.z == 89);
    CHECK(ui1.w == 111);

    CHECK(ui2.x == 39);
    CHECK(ui2.y == 0);
    CHECK(ui2.z == 52);
    CHECK(ui2.w == 0);

    // to short
    CHECK(s1.x == 39);
    CHECK(s1.y == -118);
    CHECK(s1.z == 89);
    CHECK(s1.w == 111);

    CHECK(s2.x == 39);
    CHECK(s2.y == -84);
    CHECK(s2.z == 52);
    CHECK(s2.w == -119);

    // to ushort
    CHECK(us1.x == 39);
    CHECK(us1.y == 0);
    CHECK(us1.z == 89);
    CHECK(us1.w == 111);

    CHECK(us2.x == 39);
    CHECK(us2.y == 0);
    CHECK(us2.z == 52);
    CHECK(us2.w == 0);

    // to sbyte
    CHECK(sb1.x == 39);
    CHECK(sb1.y == -118);
    CHECK(sb1.z == 89);
    CHECK(sb1.w == 111);

    CHECK(sb2.x == 39);
    CHECK(sb2.y == -84);
    CHECK(sb2.z == 52);
    CHECK(sb2.w == -119);

    // to byte
    CHECK(b1.x == 39);
    CHECK(b1.y == 0);
    CHECK(b1.z == 89);
    CHECK(b1.w == 111);

    CHECK(b2.x == 39);
    CHECK(b2.y == 0);
    CHECK(b2.z == 52);
    CHECK(b2.w == 0);

    // to bfloat
    CHECK(bf1.x == 39.75f);
    CHECK(bf1.y == -119.0f);
    CHECK(bf1.z == 89.5f);
    CHECK(bf1.w == 111.0f);

    CHECK(bf2.x == 39.25f);
    CHECK(bf2.y == -84.0f);
    CHECK(bf2.z == 52.75f);
    CHECK(bf2.w == -120.0f);

    // to half-float
    CHECK(hf1.x == 39.875f);
    CHECK(hf1.y == -118.875f);
    CHECK(hf1.z == 89.375f);
    CHECK(hf1.w == 111.125f);

    CHECK(hf2.x == 39.3125f);
    CHECK(hf2.y == -84.1875f);
    CHECK(hf2.z == 52.75f);
    CHECK(hf2.w == -119.875f);
}

TEST_CASE("Typecast with StaticCast<BFloat16>", "[Common]")
{
    Vector4<BFloat16> bf1(2147483647.0_bf - 128.0_bf, 4294967295.0_bf, static_cast<BFloat16>(NAN),
                          static_cast<BFloat16>(INFINITY));
    Vector4<BFloat16> bf2(-2147483648.0_bf + 128.0_bf, -2147483648.0_bf - 1024.0_bf, static_cast<BFloat16>(NAN),
                          static_cast<BFloat16>(-INFINITY));

    Vector4<int> i1(bf1);
    Vector4<int> i2(bf2);
    Vector4<uint> ui1(bf1);
    Vector4<uint> ui2(bf2);

    Vector4<short> s1(bf1);
    Vector4<short> s2(bf2);
    Vector4<ushort> us1(bf1);
    Vector4<ushort> us2(bf2);

    Vector4<sbyte> sb1(bf1);
    Vector4<sbyte> sb2(bf2);
    Vector4<byte> b1(bf1);
    Vector4<byte> b2(bf2);

    Vector4<BFloat16> bf1_(bf1);
    Vector4<BFloat16> bf2_(bf2);
    Vector4<HalfFp16> hf1(bf1);
    Vector4<HalfFp16> hf2(bf2);

    Vector4<float> f1(bf1);
    Vector4<float> f2(bf2);

    // to int
    CHECK(i1.x == 2147483647);
    CHECK(i1.y == 2147483647);
    CHECK(i1.z == 0);
    CHECK(i1.w == 2147483647);

    CHECK(i2.x == -2147483648);
    CHECK(i2.y == -2147483648);
    CHECK(i2.z == 0);
    CHECK(i2.w == -2147483648);

    // to uint
    CHECK(ui1.x == 2147483648);
    CHECK(ui1.y == 4294967295);
    CHECK(ui1.z == 0);
    CHECK(ui1.w == 4294967295);

    CHECK(ui2.x == 0);
    CHECK(ui2.y == 0);
    CHECK(ui2.z == 0);
    CHECK(ui2.w == 0);

    // to short
    CHECK(s1.x == 32767);
    CHECK(s1.y == 32767);
    CHECK(s1.z == 0);
    CHECK(s1.w == 32767);

    CHECK(s2.x == -32768);
    CHECK(s2.y == -32768);
    CHECK(s2.z == 0);
    CHECK(s2.w == -32768);

    // to ushort
    CHECK(us1.x == 65535);
    CHECK(us1.y == 65535);
    CHECK(us1.z == 0);
    CHECK(us1.w == 65535);

    CHECK(us2.x == 0);
    CHECK(us2.y == 0);
    CHECK(us2.z == 0);
    CHECK(us2.w == 0);

    // to sbyte
    CHECK(sb1.x == 127);
    CHECK(sb1.y == 127);
    CHECK(sb1.z == 0);
    CHECK(sb1.w == 127);

    CHECK(sb2.x == -128);
    CHECK(sb2.y == -128);
    CHECK(sb2.z == 0);
    CHECK(sb2.w == -128);

    // to byte
    CHECK(b1.x == 255);
    CHECK(b1.y == 255);
    CHECK(b1.z == 0);
    CHECK(b1.w == 255);

    CHECK(b2.x == 0);
    CHECK(b2.y == 0);
    CHECK(b2.z == 0);
    CHECK(b2.w == 0);

    // to bfloat
    CHECK(bf1_.x == 2147483648.0f);
    CHECK(bf1_.y == 4294967295.0f);
    CHECK(isnan(bf1_.z));
    CHECK(bf1_.w == INFINITY);

    CHECK(bf2_.x == -2147483648.0f);
    CHECK(bf2_.y == -2147483648.0f);
    CHECK(isnan(bf2_.z));
    CHECK(bf2_.w == -INFINITY);

    // to half-float
    CHECK(hf1.x == INFINITY);
    CHECK(hf1.y == INFINITY);
    CHECK(isnan(hf1.z));
    CHECK(hf1.w == INFINITY);

    CHECK(hf2.x == -INFINITY);
    CHECK(hf2.y == -INFINITY);
    CHECK(isnan(hf2.z));
    CHECK(hf2.w == -INFINITY);

    // to float
    CHECK(f1.x == 2147483648.0f);
    CHECK(f1.y == 4294967295.0f);
    CHECK(isnan(f1.z));
    CHECK(f1.w == INFINITY);

    CHECK(f2.x == -2147483648.0f);
    CHECK(f2.y == -2147483648.0f);
    CHECK(isnan(f2.z));
    CHECK(f2.w == -INFINITY);
}

TEST_CASE("Typecast with StaticCast<HalfFp16>", "[Common]")
{
    Vector4<HalfFp16> hf1(65504.0_hf, 65000.0_hf, static_cast<HalfFp16>(NAN), static_cast<HalfFp16>(INFINITY));
    Vector4<HalfFp16> hf2(-65504.0_hf, -65000.0_hf, static_cast<HalfFp16>(NAN), static_cast<HalfFp16>(-INFINITY));

    Vector4<int> i1(hf1);
    Vector4<int> i2(hf2);
    Vector4<uint> ui1(hf1);
    Vector4<uint> ui2(hf2);

    Vector4<short> s1(hf1);
    Vector4<short> s2(hf2);
    Vector4<ushort> us1(hf1);
    Vector4<ushort> us2(hf2);

    Vector4<sbyte> sb1(hf1);
    Vector4<sbyte> sb2(hf2);
    Vector4<byte> b1(hf1);
    Vector4<byte> b2(hf2);

    Vector4<BFloat16> bf1(hf1);
    Vector4<BFloat16> bf2(hf2);
    Vector4<HalfFp16> hf1_(hf1);
    Vector4<HalfFp16> hf2_(hf2);

    Vector4<float> f1(hf1);
    Vector4<float> f2(hf2);

    // to int
    CHECK(i1.x == 65504);
    CHECK(i1.y == 64992);
    CHECK(i1.z == 0);
    CHECK(i1.w == 2147483647);

    CHECK(i2.x == -65504);
    CHECK(i2.y == -64992);
    CHECK(i2.z == 0);
    CHECK(i2.w == -2147483648);

    // to uint
    CHECK(ui1.x == 65504);
    CHECK(ui1.y == 64992);
    CHECK(ui1.z == 0);
    CHECK(ui1.w == 4294967295);

    CHECK(ui2.x == 0);
    CHECK(ui2.y == 0);
    CHECK(ui2.z == 0);
    CHECK(ui2.w == 0);

    // to short
    CHECK(s1.x == 32767);
    CHECK(s1.y == 32767);
    CHECK(s1.z == 0);
    CHECK(s1.w == 32767);

    CHECK(s2.x == -32768);
    CHECK(s2.y == -32768);
    CHECK(s2.z == 0);
    CHECK(s2.w == -32768);

    // to ushort
    CHECK(us1.x == 65504);
    CHECK(us1.y == 64992);
    CHECK(us1.z == 0);
    CHECK(us1.w == 65535);

    CHECK(us2.x == 0);
    CHECK(us2.y == 0);
    CHECK(us2.z == 0);
    CHECK(us2.w == 0);

    // to sbyte
    CHECK(sb1.x == 127);
    CHECK(sb1.y == 127);
    CHECK(sb1.z == 0);
    CHECK(sb1.w == 127);

    CHECK(sb2.x == -128);
    CHECK(sb2.y == -128);
    CHECK(sb2.z == 0);
    CHECK(sb2.w == -128);

    // to byte
    CHECK(b1.x == 255);
    CHECK(b1.y == 255);
    CHECK(b1.z == 0);
    CHECK(b1.w == 255);

    CHECK(b2.x == 0);
    CHECK(b2.y == 0);
    CHECK(b2.z == 0);
    CHECK(b2.w == 0);

    // to bfloat
    CHECK(bf1.x == 65536.0f);
    CHECK(bf1.y == 65024.0f);
    CHECK(isnan(bf1.z));
    CHECK(bf1.w == INFINITY);

    CHECK(bf2.x == -65536.0f);
    CHECK(bf2.y == -65024.0f);
    CHECK(isnan(bf2.z));
    CHECK(bf2.w == -INFINITY);

    // to half-float
    CHECK(hf1_.x == 65504.0f);
    CHECK(hf1_.y == 64992.0f);
    CHECK(isnan(hf1_.z));
    CHECK(hf1_.w == INFINITY);

    CHECK(hf2.x == -65504.0f);
    CHECK(hf2.y == -64992.0f);
    CHECK(isnan(hf2.z));
    CHECK(hf2.w == -INFINITY);

    // to float
    CHECK(f1.x == 65504.0f);
    CHECK(f1.y == 64992.0f);
    CHECK(isnan(f1.z));
    CHECK(f1.w == INFINITY);

    CHECK(f2.x == -65504.0f);
    CHECK(f2.y == -64992.0f);
    CHECK(isnan(f2.z));
    CHECK(f2.w == -INFINITY);
}
