#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/limits.h>
#include <common/safeCast.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <cstdint>

using namespace opp;
using namespace opp::image;
using namespace Catch;

TEST_CASE("Pixel32fC3", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32fC3) == 3 * sizeof(float));

    float arr[3] = {4, 5, 6};
    Pixel32fC3 t0(arr);
    CHECK(t0.x == 4);
    CHECK(t0.y == 5);
    CHECK(t0.z == 6);

    Pixel32fC3 t1(0, 1, 2);
    CHECK(t1.x == 0);
    CHECK(t1.y == 1);
    CHECK(t1.z == 2);

    Pixel32fC3 c(t1);
    CHECK(c.x == 0);
    CHECK(c.y == 1);
    CHECK(c.z == 2);
    CHECK(c == t1);

    Pixel32fC3 c2 = t1;
    CHECK(c2.x == 0);
    CHECK(c2.y == 1);
    CHECK(c2.z == 2);
    CHECK(c2 == t1);

    Pixel32fC3 t2(5);
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(t2.z == 5);
    CHECK(c2 != t2);

    Pixel32fC3 add1 = t1 + t2;
    CHECK(add1.x == 5);
    CHECK(add1.y == 6);
    CHECK(add1.z == 7);

    Pixel32fC3 add2 = 3 + t1;
    CHECK(add2.x == 3);
    CHECK(add2.y == 4);
    CHECK(add2.z == 5);

    Pixel32fC3 add3 = t1 + 4;
    CHECK(add3.x == 4);
    CHECK(add3.y == 5);
    CHECK(add3.z == 6);

    Pixel32fC3 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 4);
    CHECK(add4.y == 6);
    CHECK(add4.z == 8);

    add4 += 3;
    CHECK(add4.x == 7);
    CHECK(add4.y == 9);
    CHECK(add4.z == 11);

    Pixel32fC3 sub1 = t1 - t2;
    CHECK(sub1.x == -5);
    CHECK(sub1.y == -4);
    CHECK(sub1.z == -3);

    Pixel32fC3 sub2 = 3 - t1;
    CHECK(sub2.x == 3);
    CHECK(sub2.y == 2);
    CHECK(sub2.z == 1);

    Pixel32fC3 sub3 = t1 - 4;
    CHECK(sub3.x == -4);
    CHECK(sub3.y == -3);
    CHECK(sub3.z == -2);

    Pixel32fC3 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);
    CHECK(sub4.z == 4);

    sub4 -= 3;
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);
    CHECK(sub4.z == 1);

    t1              = Pixel32fC3(4, 5, 6);
    t2              = Pixel32fC3(6, 7, 8);
    Pixel32fC3 mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);
    CHECK(mul1.z == 48);

    Pixel32fC3 mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);
    CHECK(mul2.z == 18);

    Pixel32fC3 mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);
    CHECK(mul3.z == 24);

    Pixel32fC3 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);
    CHECK(mul4.z == 144);

    mul4 *= 3;
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);
    CHECK(mul4.z == 432);

    Pixel32fC3 div1 = t1 / t2;
    CHECK(div1.x == Approx(0.667).margin(0.001));
    CHECK(div1.y == Approx(0.714).margin(0.001));
    CHECK(div1.z == Approx(0.75).margin(0.001));

    Pixel32fC3 div2 = 3 / t1;
    CHECK(div2.x == Approx(0.75).margin(0.001));
    CHECK(div2.y == Approx(0.6).margin(0.001));
    CHECK(div2.z == Approx(0.5).margin(0.001));

    Pixel32fC3 div3 = t1 / 4;
    CHECK(div3.x == Approx(1.0).margin(0.001));
    CHECK(div3.y == Approx(1.25).margin(0.001));
    CHECK(div3.z == Approx(1.5).margin(0.001));

    Pixel32fC3 div4 = t2;
    div4 /= div3;
    CHECK(div4.x == Approx(6.0).margin(0.001));
    CHECK(div4.y == Approx(5.6).margin(0.001));
    CHECK(div4.z == Approx(5.333).margin(0.001));

    div4 /= 3;
    CHECK(div4.x == Approx(2.0).margin(0.001));
    CHECK(div4.y == Approx(1.867).margin(0.001));
    CHECK(div4.z == Approx(1.778).margin(0.001));

    Pixel32fC3 l(4, 6, -7);
    CHECK(l.MagnitudeSqr() == 101);
    CHECK(l.Magnitude() == Approx(std::sqrt(101)));

    Pixel32fC3 minmax1(10, 20, -10);
    Pixel32fC3 minmax2(-20, 10, 40);

    CHECK(minmax1.Min(minmax2) == Pixel32fC3(-20, 10, -10));
    CHECK(minmax2.Min(minmax1) == Pixel32fC3(-20, 10, -10));

    CHECK(minmax1.Max(minmax2) == Pixel32fC3(10, 20, 40));
    CHECK(minmax2.Max(minmax1) == Pixel32fC3(10, 20, 40));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 20);
}

TEST_CASE("Pixel32fC3_additionalMethods", "[Common]")
{
    CHECK(Pixel32fC3::Dot(Pixel32fC3(4, 5, 6), Pixel32fC3(6, 7, 8)) == 107.0f);
    CHECK(Pixel32fC3(4, 5, 6).Dot(Pixel32fC3(6, 7, 8)) == 107.0f);

    CHECK(Pixel32fC3::Cross(Pixel32fC3(6, 5, 4), Pixel32fC3(7, 8, 11)) == Pixel32fC3(23, -38, 13));
    CHECK(Pixel32fC3(6, 5, 4).Cross(Pixel32fC3(7, 8, 11)) == Pixel32fC3(23, -38, 13));

    Pixel32fC3 norm(4, 5, 6);
    norm.Normalize();
    CHECK(norm.Magnitude() == 1);
    CHECK(norm.x == Approx(0.456).margin(0.001));
    CHECK(norm.y == Approx(0.570).margin(0.001));
    CHECK(norm.z == Approx(0.684).margin(0.001));

    CHECK(Pixel32fC3::Normalize(Pixel32fC3(4, 5, 6)) == norm);

    Pixel32fC3 roundA(0.4f, 0.5f, 0.6f);
    Pixel32fC3 roundB(1.9f, -1.5f, -2.5f);
    Pixel32fC3 round2A = Pixel32fC3::Round(roundA);
    Pixel32fC3 round2B = Pixel32fC3::Round(roundB);
    roundA.Round();
    roundB.Round();
    CHECK(round2A == roundA);
    CHECK(round2B == roundB);
    CHECK(roundA.x == 0.0f);
    CHECK(roundA.y == 1.0f);
    CHECK(roundA.z == 1.0f);
    CHECK(roundB.x == 2.0f);
    CHECK(roundB.y == -2.0f);
    CHECK(roundB.z == -3.0f);

    Pixel32fC3 floorA(0.4f, 0.5f, 0.6f);
    Pixel32fC3 floorB(1.9f, -1.5f, -2.5f);
    Pixel32fC3 floor2A = Pixel32fC3::Floor(floorA);
    Pixel32fC3 floor2B = Pixel32fC3::Floor(floorB);
    floorA.Floor();
    floorB.Floor();
    CHECK(floor2A == floorA);
    CHECK(floor2B == floorB);
    CHECK(floorA.x == 0.0f);
    CHECK(floorA.y == 0.0f);
    CHECK(floorA.z == 0.0f);
    CHECK(floorB.x == 1.0f);
    CHECK(floorB.y == -2.0f);
    CHECK(floorB.z == -3.0f);

    Pixel32fC3 ceilA(0.4f, 0.5f, 0.6f);
    Pixel32fC3 ceilB(1.9f, -1.5f, -2.5f);
    Pixel32fC3 ceil2A = Pixel32fC3::Ceil(ceilA);
    Pixel32fC3 ceil2B = Pixel32fC3::Ceil(ceilB);
    ceilA.Ceil();
    ceilB.Ceil();
    CHECK(ceil2A == ceilA);
    CHECK(ceil2B == ceilB);
    CHECK(ceilA.x == 1.0f);
    CHECK(ceilA.y == 1.0f);
    CHECK(ceilA.z == 1.0f);
    CHECK(ceilB.x == 2.0f);
    CHECK(ceilB.y == -1.0f);
    CHECK(ceilB.z == -2.0f);

    Pixel32fC3 zeroA(0.4f, 0.5f, 0.6f);
    Pixel32fC3 zeroB(1.9f, -1.5f, -2.5f);
    Pixel32fC3 zero2A = Pixel32fC3::RoundZero(zeroA);
    Pixel32fC3 zero2B = Pixel32fC3::RoundZero(zeroB);
    zeroA.RoundZero();
    zeroB.RoundZero();
    CHECK(zero2A == zeroA);
    CHECK(zero2B == zeroB);
    CHECK(zeroA.x == 0.0f);
    CHECK(zeroA.y == 0.0f);
    CHECK(zeroA.z == 0.0f);
    CHECK(zeroB.x == 1.0f);
    CHECK(zeroB.y == -1.0f);
    CHECK(zeroB.z == -2.0f);

    Pixel32fC3 nearestA(0.4f, 0.5f, 0.6f);
    Pixel32fC3 nearestB(1.9f, -1.5f, -2.5f);
    Pixel32fC3 nearest2A = Pixel32fC3::RoundNearest(nearestA);
    Pixel32fC3 nearest2B = Pixel32fC3::RoundNearest(nearestB);
    nearestA.RoundNearest();
    nearestB.RoundNearest();
    CHECK(nearest2A == nearestA);
    CHECK(nearest2B == nearestB);
    CHECK(nearestA.x == 0.0f);
    CHECK(nearestA.y == 0.0f);
    CHECK(nearestA.z == 1.0f);
    CHECK(nearestB.x == 2.0f);
    CHECK(nearestB.y == -2.0f);
    CHECK(nearestB.z == -2.0f);

    Pixel32fC3 exp(2.4f, 12.5f, -14.6f);
    Pixel32fC3 exp2 = Pixel32fC3::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == Approx(11.023176380641601).margin(0.00001));
    CHECK(exp.y == Approx(2.683372865208745e+05).margin(0.00001));
    CHECK(exp.z == Approx(4.563526367903994e-07).margin(0.00001));

    Pixel32fC3 ln(2.4f, 12.5f, 14.6f);
    Pixel32fC3 ln2 = Pixel32fC3::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == Approx(0.875468737353900).margin(0.00001));
    CHECK(ln.y == Approx(2.525728644308256).margin(0.00001));
    CHECK(ln.z == Approx(2.681021528714291).margin(0.00001));

    Pixel32fC3 sqr(2.4f, 12.5f, -14.6f);
    Pixel32fC3 sqr2 = Pixel32fC3::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == Approx(5.76).margin(0.00001));
    CHECK(sqr.y == Approx(156.25).margin(0.00001));
    CHECK(sqr.z == Approx(213.16).margin(0.00001));

    Pixel32fC3 sqrt(2.4f, 12.5f, 14.6f);
    Pixel32fC3 sqrt2 = Pixel32fC3::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == Approx(1.549193338482967).margin(0.00001));
    CHECK(sqrt.y == Approx(3.535533905932738).margin(0.00001));
    CHECK(sqrt.z == Approx(3.820994634908560).margin(0.00001));

    Pixel32fC3 abs(-2.4f, 12.5f, -14.6f);
    Pixel32fC3 abs2 = Pixel32fC3::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == 2.4f);
    CHECK(abs.y == 12.5f);
    CHECK(abs.z == 14.6f);

    Pixel32fC3 absdiffA(13.23592f, -40.24595f, -22.15017f);
    Pixel32fC3 absdiffB(45.75068f, 46.488853f, -34.238691f);
    Pixel32fC3 absdiff2 = Pixel32fC3::AbsDiff(absdiffA, absdiffB);
    Pixel32fC3 absdiff3 = Pixel32fC3::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == Approx(32.514758920888809).margin(0.00001));
    CHECK(absdiffA.y == Approx(86.734813019986703).margin(0.00001));
    CHECK(absdiffA.z == Approx(12.088513718950011).margin(0.00001));

    Pixel32fC3 clampByte(float(numeric_limits<byte>::max()) + 1, float(numeric_limits<byte>::min()) - 1,
                         float(numeric_limits<byte>::min()));
    clampByte.ClampToTargetType(static_cast<byte>(0));
    CHECK(clampByte.x == 255);
    CHECK(clampByte.y == 0);
    CHECK(clampByte.z == 0);

    Pixel32fC3 clampShort(float(numeric_limits<short>::max()) + 1, float(numeric_limits<short>::min()) - 1,
                          float(numeric_limits<short>::min()));
    clampShort.ClampToTargetType(static_cast<short>(0));
    CHECK(clampShort.x == 32767);
    CHECK(clampShort.y == -32768);
    CHECK(clampShort.z == -32768);

    Pixel32fC3 clampSByte(float(numeric_limits<sbyte>::max()) + 1, float(numeric_limits<sbyte>::min()) - 1,
                          float(numeric_limits<sbyte>::min()));
    clampSByte.ClampToTargetType(static_cast<sbyte>(0));
    CHECK(clampSByte.x == 127);
    CHECK(clampSByte.y == -128);
    CHECK(clampSByte.z == -128);

    Pixel32fC3 clampUShort(float(numeric_limits<ushort>::max()) + 1, float(numeric_limits<ushort>::min()) - 1,
                           float(numeric_limits<ushort>::min()));
    clampUShort.ClampToTargetType(static_cast<ushort>(0));
    CHECK(clampUShort.x == 65535);
    CHECK(clampUShort.y == 0);
    CHECK(clampUShort.z == 0);

    Pixel32fC3 clampInt(float(numeric_limits<int>::max()) + 1000, float(numeric_limits<int>::min()) - 1000,
                        float(numeric_limits<int>::min()));
    clampInt.ClampToTargetType(0);
    CHECK(clampInt.x == 2147483647);
    CHECK(clampInt.y == -2147483648);
    CHECK(clampInt.z == -2147483648);

    Pixel32fC3 clampUInt(float(numeric_limits<uint>::max()) + 1000, float(numeric_limits<uint>::min()) - 1000,
                         float(numeric_limits<uint>::min()));
    clampUInt.ClampToTargetType(static_cast<uint>(0));
    CHECK(clampUInt.x == 0xffffffffUL);
    CHECK(clampUInt.y == 0);
    CHECK(clampUInt.z == 0);
}

TEST_CASE("Pixel32sC3", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32sC3) == 3 * sizeof(int));

    Pixel32sC3 t1(100, 200, 300);
    CHECK(t1.x == 100);
    CHECK(t1.y == 200);
    CHECK(t1.z == 300);

    Pixel32sC3 c(t1);
    CHECK(c.x == 100);
    CHECK(c.y == 200);
    CHECK(c.z == 300);
    CHECK(c == t1);

    Pixel32sC3 c2 = t1;
    CHECK(c2.x == 100);
    CHECK(c2.y == 200);
    CHECK(c2.z == 300);
    CHECK(c2 == t1);

    Pixel32sC3 t2(5);
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(t2.z == 5);
    CHECK(c2 != t2);

    Pixel32sC3 add1 = t1 + t2;
    CHECK(add1.x == 105);
    CHECK(add1.y == 205);
    CHECK(add1.z == 305);

    Pixel32sC3 add2 = 3 + t1;
    CHECK(add2.x == 103);
    CHECK(add2.y == 203);
    CHECK(add2.z == 303);

    Pixel32sC3 add3 = t1 + 4;
    CHECK(add3.x == 104);
    CHECK(add3.y == 204);
    CHECK(add3.z == 304);

    Pixel32sC3 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 204);
    CHECK(add4.y == 404);
    CHECK(add4.z == 604);

    add4 += 3;
    CHECK(add4.x == 207);
    CHECK(add4.y == 407);
    CHECK(add4.z == 607);

    Pixel32sC3 sub1 = t1 - t2;
    CHECK(sub1.x == 95);
    CHECK(sub1.y == 195);
    CHECK(sub1.z == 295);

    Pixel32sC3 sub2 = 3 - t1;
    CHECK(sub2.x == -97);
    CHECK(sub2.y == -197);
    CHECK(sub2.z == -297);

    Pixel32sC3 sub3 = t1 - 4;
    CHECK(sub3.x == 96);
    CHECK(sub3.y == 196);
    CHECK(sub3.z == 296);

    Pixel32sC3 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);
    CHECK(sub4.z == 4);

    sub4 -= 3;
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);
    CHECK(sub4.z == 1);

    t1              = Pixel32sC3(4, 5, 6);
    t2              = Pixel32sC3(6, 7, 8);
    Pixel32sC3 mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);
    CHECK(mul1.z == 48);

    Pixel32sC3 mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);
    CHECK(mul2.z == 18);

    Pixel32sC3 mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);
    CHECK(mul3.z == 24);

    Pixel32sC3 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);
    CHECK(mul4.z == 144);

    mul4 *= 3;
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);
    CHECK(mul4.z == 432);

    t1              = Pixel32sC3(1000, 2000, 3000);
    t2              = Pixel32sC3(6, 7, 8);
    Pixel32sC3 div1 = t1 / t2;
    CHECK(div1.x == 166);
    CHECK(div1.y == 285);
    CHECK(div1.z == 375);

    Pixel32sC3 div2 = 30000 / t1;
    CHECK(div2.x == 30);
    CHECK(div2.y == 15);
    CHECK(div2.z == 10);

    Pixel32sC3 div3 = t1 / 4;
    CHECK(div3.x == 250);
    CHECK(div3.y == 500);
    CHECK(div3.z == 750);

    Pixel32sC3 div4 = t2 * 10000;
    div4 /= div3;
    CHECK(div4.x == 240);
    CHECK(div4.y == 140);
    CHECK(div4.z == 106);

    div4 /= 3;
    CHECK(div4.x == 80);
    CHECK(div4.y == 46);
    CHECK(div4.z == 35);

    Pixel32sC3 l(4, 6, -7);
    CHECK(l.MagnitudeSqr() == 101);
    CHECK(l.Magnitude() == Approx(std::sqrt(101)));

    Pixel32sC3 minmax1(10, 20, -10);
    Pixel32sC3 minmax2(-20, 10, 40);

    CHECK(minmax1.Min(minmax2) == Pixel32sC3(-20, 10, -10));
    CHECK(minmax2.Min(minmax1) == Pixel32sC3(-20, 10, -10));

    CHECK(minmax1.Max(minmax2) == Pixel32sC3(10, 20, 40));
    CHECK(minmax2.Max(minmax1) == Pixel32sC3(10, 20, 40));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 20);
}

TEST_CASE("Pixel32sC3_additionalMethods", "[Common]")
{
    Pixel32sC3 norm(4, 5, 6);
    CHECK(norm.Magnitude() == Approx(std::sqrt(77)).margin(0.00001));
    CHECK(norm.MagnitudeSqr() == 77);

    Pixel32sC3 exp(4, 5, 6);
    Pixel32sC3 exp2 = Pixel32sC3::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == 54);
    CHECK(exp.y == 148);
    CHECK(exp.z == 403);

    Pixel32sC3 ln(4, 50, 600);
    Pixel32sC3 ln2 = Pixel32sC3::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == 1);
    CHECK(ln.y == 3);
    CHECK(ln.z == 6);

    Pixel32sC3 sqr(4, 5, 6);
    Pixel32sC3 sqr2 = Pixel32sC3::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == 16);
    CHECK(sqr.y == 25);
    CHECK(sqr.z == 36);

    Pixel32sC3 sqrt(4, 5, 6);
    Pixel32sC3 sqrt2 = Pixel32sC3::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == 2);
    CHECK(sqrt.y == 2);
    CHECK(sqrt.z == 2);

    Pixel32sC3 abs(-2, 12, -14);
    Pixel32sC3 abs2 = Pixel32sC3::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == 2);
    CHECK(abs.y == 12);
    CHECK(abs.z == 14);

    Pixel32sC3 absdiffA(13, -40, -22);
    Pixel32sC3 absdiffB(45, 46, -34);
    Pixel32sC3 absdiff2 = Pixel32sC3::AbsDiff(absdiffA, absdiffB);
    Pixel32sC3 absdiff3 = Pixel32sC3::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == 32);
    CHECK(absdiffA.y == 86);
    CHECK(absdiffA.z == 12);

    Pixel32sC3 clampByte(int(numeric_limits<byte>::max()) + 1, int(numeric_limits<byte>::min()) - 1,
                         int(numeric_limits<byte>::min()));
    clampByte.ClampToTargetType(static_cast<byte>(0));
    CHECK(clampByte.x == 255);
    CHECK(clampByte.y == 0);
    CHECK(clampByte.z == 0);

    Pixel32sC3 clampShort(int(numeric_limits<short>::max()) + 1, int(numeric_limits<short>::min()) - 1,
                          int(numeric_limits<short>::min()));
    clampShort.ClampToTargetType(static_cast<short>(0));
    CHECK(clampShort.x == 32767);
    CHECK(clampShort.y == -32768);
    CHECK(clampShort.z == -32768);

    Pixel32sC3 clampSByte(int(numeric_limits<sbyte>::max()) + 1, int(numeric_limits<sbyte>::min()) - 1,
                          int(numeric_limits<sbyte>::min()));
    clampSByte.ClampToTargetType(static_cast<sbyte>(0));
    CHECK(clampSByte.x == 127);
    CHECK(clampSByte.y == -128);
    CHECK(clampSByte.z == -128);

    Pixel32sC3 clampUShort(int(numeric_limits<ushort>::max()) + 1, int(numeric_limits<ushort>::min()) - 1,
                           int(numeric_limits<ushort>::min()));
    clampUShort.ClampToTargetType(static_cast<ushort>(0));
    CHECK(clampUShort.x == 65535);
    CHECK(clampUShort.y == 0);
    CHECK(clampUShort.z == 0);

    Pixel32sC3 clampInt(numeric_limits<int>::max(), numeric_limits<int>::min(), numeric_limits<int>::min());
    clampInt.ClampToTargetType(0);
    CHECK(clampInt.x == 2147483647);
    CHECK(clampInt.y == -2147483648);
    CHECK(clampInt.z == -2147483648);

    Pixel32sC3 lshift(1024, 2048, 4096);
    Pixel32sC3 lshift2 = Pixel32sC3::LShift(lshift, 2);
    lshift.LShift(2);
    CHECK(lshift == lshift2);
    CHECK(lshift.x == 4096);
    CHECK(lshift.y == 8192);
    CHECK(lshift.z == 16384);

    Pixel32sC3 rshift(1024, 2048, 4096);
    Pixel32sC3 rshift2 = Pixel32sC3::RShift(rshift, 2);
    rshift.RShift(2);
    CHECK(rshift == rshift2);
    CHECK(rshift.x == 256);
    CHECK(rshift.y == 512);
    CHECK(rshift.z == 1024);

    Pixel32sC3 and_(1023, 2047, 4095);
    Pixel32sC3 and_B(512, 1024, 2048);
    Pixel32sC3 and_2 = Pixel32sC3::And(and_, and_B);
    and_.And(and_B);
    CHECK(and_ == and_2);
    CHECK(and_.x == 512);
    CHECK(and_.y == 1024);
    CHECK(and_.z == 2048);

    Pixel32sC3 or_(1023, 2047, 4095);
    Pixel32sC3 or_B(512, 1024, 2048);
    Pixel32sC3 or_2 = Pixel32sC3::Or(or_, or_B);
    or_.Or(or_B);
    CHECK(or_ == or_2);
    CHECK(or_.x == 1023);
    CHECK(or_.y == 2047);
    CHECK(or_.z == 4095);

    Pixel32sC3 xor_(1023, 2047, 4095);
    Pixel32sC3 xor_B(512, 1024, 2048);
    Pixel32sC3 xor_2 = Pixel32sC3::Xor(xor_, xor_B);
    xor_.Xor(xor_B);
    CHECK(xor_ == xor_2);
    CHECK(xor_.x == 511);
    CHECK(xor_.y == 1023);
    CHECK(xor_.z == 2047);

    Pixel32sC3 not_(1023, 2047, 4095);
    Pixel32sC3 not_2 = Pixel32sC3::Not(not_);
    not_.Not();
    CHECK(not_ == not_2);
    CHECK(not_.x == -1024);
    CHECK(not_.y == -2048);
    CHECK(not_.z == -4096);
}

TEST_CASE("Pixel32fC3_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel32fC3 *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel32fC3(float(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->x;
        void *ptrLastMember  = &buffer[i]->z;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<float *>(ptrVector) + 2));
    }

    std::vector<Pixel32fC3> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (3 * sizeof(float))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Pixel32sC3_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel32sC3 *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel32sC3(int(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->x;
        void *ptrLastMember  = &buffer[i]->z;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<int *>(ptrVector) + 2));
    }

    std::vector<Pixel32sC3> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (3 * sizeof(int))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Pixel16uC3_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel16uC3 *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel16uC3(ushort(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->x;
        void *ptrLastMember  = &buffer[i]->z;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<ushort *>(ptrVector) + 2));
    }

    std::vector<Pixel16uC3> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (3 * sizeof(ushort))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Pixel32sC3_streams", "[Common]")
{
    std::string str = "3 4 5";
    std::stringstream ss(str);

    Pixel32sC3 pix;
    ss >> pix;

    CHECK(pix.x == 3);
    CHECK(pix.y == 4);
    CHECK(pix.z == 5);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3, 4, 5)");
}

TEST_CASE("Pixel32fC3_streams", "[Common]")
{
    std::string str = "3.14 2.7 8.9";
    std::stringstream ss(str);

    Pixel32fC3 pix;
    ss >> pix;

    CHECK(pix.x == 3.14f);
    CHECK(pix.y == 2.7f);
    CHECK(pix.z == 8.9f);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3.14, 2.7, 8.9)");
}

TEST_CASE("Axis3D", "[Common]")
{
    Pixel32sC3 pix(2, 3, 4);

    CHECK(pix[Axis3D::X] == 2);
    CHECK(pix[Axis3D::Y] == 3);
    CHECK(pix[Axis3D::Z] == 4);

    CHECK_THROWS_AS(pix[static_cast<Axis3D>(5)], opp::InvalidArgumentException);

    try
    {
        pix[static_cast<Axis3D>(5)] = 12;
    }
    catch (const opp::InvalidArgumentException &ex)
    {
        CHECK(ex.Message() == "Out of range: 5. Must be X, Y or Z (0, 1 or 2).");
    }
}