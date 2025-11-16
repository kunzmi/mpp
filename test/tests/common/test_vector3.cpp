#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <common/complex.h>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/pixelTypes.h>
#include <common/needSaturationClamp.h>
#include <common/numeric_limits.h>
#include <common/vector3.h>
#include <iosfwd>
#include <string>
#include <vector>

using namespace mpp;
using namespace mpp::image;
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

    Pixel32fC3 neg = -t1;
    CHECK(neg.x == 0);
    CHECK(neg.y == -1);
    CHECK(neg.z == -2);
    CHECK(t1 == -neg);

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

    Pixel32fC3 minmax1(10, 20, -10);
    Pixel32fC3 minmax2(-20, 10, 40);

    CHECK(Pixel32fC3::Min(minmax1, minmax2) == Pixel32fC3(-20, 10, -10));
    CHECK(Pixel32fC3::Min(minmax2, minmax1) == Pixel32fC3(-20, 10, -10));

    CHECK(Pixel32fC3::Max(minmax1, minmax2) == Pixel32fC3(10, 20, 40));
    CHECK(Pixel32fC3::Max(minmax2, minmax1) == Pixel32fC3(10, 20, 40));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 20);
}

TEST_CASE("Pixel32fC3_additionalMethods", "[Common]")
{
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
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 256);
    CHECK(clampByte.y == -1);
    CHECK(clampByte.z == 0);

    Pixel32fC3 clampShort(float(numeric_limits<short>::max()) + 1, float(numeric_limits<short>::min()) - 1,
                          float(numeric_limits<short>::min()));
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32768);
    CHECK(clampShort.y == -32769);
    CHECK(clampShort.z == -32768);

    Pixel32fC3 clampSByte(float(numeric_limits<sbyte>::max()) + 1, float(numeric_limits<sbyte>::min()) - 1,
                          float(numeric_limits<sbyte>::min()));
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 128);
    CHECK(clampSByte.y == -129);
    CHECK(clampSByte.z == -128);

    Pixel32fC3 clampUShort(float(numeric_limits<ushort>::max()) + 1, float(numeric_limits<ushort>::min()) - 1,
                           float(numeric_limits<ushort>::min()));
    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 65536);
    CHECK(clampUShort.y == -1);
    CHECK(clampUShort.z == 0);

    Pixel32fC3 clampInt(float(numeric_limits<int>::max()) + 1000, float(numeric_limits<int>::min()) - 1000,
                        float(numeric_limits<int>::min()));
    clampInt.template ClampToTargetType<int>();
    CHECK(clampInt.x == 2147484672.0f);
    CHECK(clampInt.y == -2147484672.0f);
    CHECK(clampInt.z == -2147483648);

    Pixel32fC3 clampUInt(float(numeric_limits<uint>::max()) + 1000, float(numeric_limits<uint>::min()) - 1000,
                         float(numeric_limits<uint>::min()));
    clampUInt.template ClampToTargetType<uint>();
    CHECK(clampUInt.x == 4294968320.0f);
    CHECK(clampUInt.y == -1000.0f);
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

    Pixel32sC3 minmax1(10, 20, -10);
    Pixel32sC3 minmax2(-20, 10, 40);

    CHECK(Pixel32sC3::Min(minmax1, minmax2) == Pixel32sC3(-20, 10, -10));
    CHECK(Pixel32sC3::Min(minmax2, minmax1) == Pixel32sC3(-20, 10, -10));

    CHECK(Pixel32sC3::Max(minmax1, minmax2) == Pixel32sC3(10, 20, 40));
    CHECK(Pixel32sC3::Max(minmax2, minmax1) == Pixel32sC3(10, 20, 40));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 20);

    CHECK(Pixel32sC3(10, 11, -13).DivRound(Pixel32sC3(3, 4, 4)) == Pixel32sC3(3, 3, -3));
    CHECK(Pixel32sC3(10, 11, -13).DivRoundNearest(Pixel32sC3(3, 4, 4)) == Pixel32sC3(3, 3, -3));
    CHECK(Pixel32sC3(10, 11, -13).DivRoundZero(Pixel32sC3(3, 4, 4)) == Pixel32sC3(3, 2, -3));
    CHECK(Pixel32sC3(10, 11, -13).DivFloor(Pixel32sC3(3, 4, 4)) == Pixel32sC3(3, 2, -4));
    CHECK(Pixel32sC3(10, 11, -13).DivCeil(Pixel32sC3(3, 4, 4)) == Pixel32sC3(4, 3, -3));

    CHECK(Pixel32sC3(3, 4, 4).DivInvRound(Pixel32sC3(10, 11, -13)) == Pixel32sC3(3, 3, -3));
    CHECK(Pixel32sC3(3, 4, 4).DivInvRoundNearest(Pixel32sC3(10, 11, -13)) == Pixel32sC3(3, 3, -3));
    CHECK(Pixel32sC3(3, 4, 4).DivInvRoundZero(Pixel32sC3(10, 11, -13)) == Pixel32sC3(3, 2, -3));
    CHECK(Pixel32sC3(3, 4, 4).DivInvFloor(Pixel32sC3(10, 11, -13)) == Pixel32sC3(3, 2, -4));
    CHECK(Pixel32sC3(3, 4, 4).DivInvCeil(Pixel32sC3(10, 11, -13)) == Pixel32sC3(4, 3, -3));

    CHECK(Pixel32sC3::DivRound(Pixel32sC3(10, 11, 13), Pixel32sC3(-3, -4, 4)) == Pixel32sC3(-3, -3, 3));
    CHECK(Pixel32sC3::DivRoundNearest(Pixel32sC3(10, 11, 13), Pixel32sC3(-3, -4, 4)) == Pixel32sC3(-3, -3, 3));
    CHECK(Pixel32sC3::DivRoundZero(Pixel32sC3(10, 11, 13), Pixel32sC3(-3, -4, 4)) == Pixel32sC3(-3, -2, 3));
    CHECK(Pixel32sC3::DivFloor(Pixel32sC3(10, 11, 13), Pixel32sC3(-3, -4, 4)) == Pixel32sC3(-4, -3, 3));
    CHECK(Pixel32sC3::DivCeil(Pixel32sC3(10, 11, 13), Pixel32sC3(-3, -4, 4)) == Pixel32sC3(-3, -2, 4));

    CHECK(Pixel32sC3(-9, 15, -15).DivScaleRoundNearest(10) == Pixel32sC3(-1, 2, -2));
}

TEST_CASE("Pixel32sC3_additionalMethods", "[Common]")
{
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

    Pixel32sC3 clampByte(int(numeric_limits<byte>::max()) + 1, int(numeric_limits<byte>::min()) - 1,
                         int(numeric_limits<byte>::min()));
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 255);
    CHECK(clampByte.y == 0);
    CHECK(clampByte.z == 0);

    Pixel32sC3 clampShort(int(numeric_limits<short>::max()) + 1, int(numeric_limits<short>::min()) - 1,
                          int(numeric_limits<short>::min()));
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32767);
    CHECK(clampShort.y == -32768);
    CHECK(clampShort.z == -32768);

    Pixel32sC3 clampSByte(int(numeric_limits<sbyte>::max()) + 1, int(numeric_limits<sbyte>::min()) - 1,
                          int(numeric_limits<sbyte>::min()));
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 127);
    CHECK(clampSByte.y == -128);
    CHECK(clampSByte.z == -128);

    Pixel32sC3 clampUShort(int(numeric_limits<ushort>::max()) + 1, int(numeric_limits<ushort>::min()) - 1,
                           int(numeric_limits<ushort>::min()));
    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 65535);
    CHECK(clampUShort.y == 0);
    CHECK(clampUShort.z == 0);

    Pixel32sC3 clampInt(numeric_limits<int>::max(), numeric_limits<int>::min(), numeric_limits<int>::min());
    clampInt.template ClampToTargetType<int>();
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

TEST_CASE("Pixel8uC3_streams", "[Common]")
{
    std::string str = "3 4 5";
    std::stringstream ss(str);

    Pixel8uC3 pix;
    ss >> pix;

    CHECK(pix.x == 3);
    CHECK(pix.y == 4);
    CHECK(pix.z == 5);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3, 4, 5)");
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

    CHECK_THROWS_AS(pix[static_cast<Axis3D>(5)], mpp::InvalidArgumentException);

    try
    {
        pix[static_cast<Axis3D>(5)] = 12;
    }
    catch (const mpp::InvalidArgumentException &ex)
    {
        CHECK(ex.Message() == "Out of range: 5. Must be X, Y or Z (0, 1 or 2).");
    }
}

TEST_CASE("Pixel16fC3", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel16fC3) == 3 * sizeof(HalfFp16));

    HalfFp16 arr[4] = {HalfFp16(4), HalfFp16(5), HalfFp16(6)};
    Pixel16fC3 t0(arr);
    CHECK(t0.x == 4);
    CHECK(t0.y == 5);
    CHECK(t0.z == 6);

    Pixel16fC3 t1(HalfFp16(0), HalfFp16(1), HalfFp16(2));
    CHECK(t1.x == 0);
    CHECK(t1.y == 1);
    CHECK(t1.z == 2);

    Pixel16fC3 c(t1);
    CHECK(c.x == 0);
    CHECK(c.y == 1);
    CHECK(c.z == 2);
    CHECK(c == t1);

    Pixel16fC3 c2 = t1;
    CHECK(c2.x == 0);
    CHECK(c2.y == 1);
    CHECK(c2.z == 2);
    CHECK(c2 == t1);

    Pixel16fC3 t2(HalfFp16(5));
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(t2.z == 5);
    CHECK(c2 != t2);

    Pixel16fC3 add1 = t1 + t2;
    CHECK(add1.x == 5);
    CHECK(add1.y == 6);
    CHECK(add1.z == 7);

    Pixel16fC3 add2 = 3 + t1;
    CHECK(add2.x == 3);
    CHECK(add2.y == 4);
    CHECK(add2.z == 5);

    Pixel16fC3 add3 = t1 + 4;
    CHECK(add3.x == 4);
    CHECK(add3.y == 5);
    CHECK(add3.z == 6);

    Pixel16fC3 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 4);
    CHECK(add4.y == 6);
    CHECK(add4.z == 8);

    add4 += HalfFp16(3);
    CHECK(add4.x == 7);
    CHECK(add4.y == 9);
    CHECK(add4.z == 11);

    Pixel16fC3 sub1 = t1 - t2;
    CHECK(sub1.x == -5);
    CHECK(sub1.y == -4);
    CHECK(sub1.z == -3);

    Pixel16fC3 sub2 = 3 - t1;
    CHECK(sub2.x == 3);
    CHECK(sub2.y == 2);
    CHECK(sub2.z == 1);

    Pixel16fC3 sub3 = t1 - 4;
    CHECK(sub3.x == -4);
    CHECK(sub3.y == -3);
    CHECK(sub3.z == -2);

    Pixel16fC3 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);
    CHECK(sub4.z == 4);

    sub4 -= HalfFp16(3);
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);
    CHECK(sub4.z == 1);

    t1              = Pixel16fC3(HalfFp16(4), HalfFp16(5), HalfFp16(6));
    t2              = Pixel16fC3(HalfFp16(6), HalfFp16(7), HalfFp16(8));
    Pixel16fC3 mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);
    CHECK(mul1.z == 48);

    Pixel16fC3 mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);
    CHECK(mul2.z == 18);

    Pixel16fC3 mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);
    CHECK(mul3.z == 24);

    Pixel16fC3 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);
    CHECK(mul4.z == 144);

    mul4 *= HalfFp16(3);
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);
    CHECK(mul4.z == 432);

    Pixel16fC3 div1 = t1 / t2;
    CHECK(div1.x == Approx(0.667).margin(0.001));
    CHECK(div1.y == Approx(0.714).margin(0.001));
    CHECK(div1.z == Approx(0.75).margin(0.001));

    Pixel16fC3 div2 = 3 / t1;
    CHECK(div2.x == Approx(0.75).margin(0.001));
    CHECK(div2.y == Approx(0.6).margin(0.001));
    CHECK(div2.z == Approx(0.5).margin(0.001));

    Pixel16fC3 div3 = t1 / 4;
    CHECK(div3.x == Approx(1.0).margin(0.001));
    CHECK(div3.y == Approx(1.25).margin(0.001));
    CHECK(div3.z == Approx(1.5).margin(0.001));

    Pixel16fC3 div4 = t2;
    div4 /= div3;
    CHECK(div4.x == Approx(6.0).margin(0.001));
    CHECK(div4.y == Approx(5.6).margin(0.01));
    CHECK(div4.z == Approx(5.333).margin(0.001));

    div4 /= HalfFp16(3);
    CHECK(div4.x == Approx(2.0).margin(0.001));
    CHECK(div4.y == Approx(1.867).margin(0.001));
    CHECK(div4.z == Approx(1.778).margin(0.001));

    Pixel16fC3 minmax1(HalfFp16(10), HalfFp16(20), HalfFp16(-10));
    Pixel16fC3 minmax2(HalfFp16(-20), HalfFp16(10), HalfFp16(40));

    CHECK(Pixel16fC3::Min(minmax1, minmax2) == Pixel16fC3(HalfFp16(-20), HalfFp16(10), HalfFp16(-10)));
    CHECK(Pixel16fC3::Min(minmax2, minmax1) == Pixel16fC3(HalfFp16(-20), HalfFp16(10), HalfFp16(-10)));

    CHECK(Pixel16fC3::Max(minmax1, minmax2) == Pixel16fC3(HalfFp16(10), HalfFp16(20), HalfFp16(40)));
    CHECK(Pixel16fC3::Max(minmax2, minmax1) == Pixel16fC3(HalfFp16(10), HalfFp16(20), HalfFp16(40)));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 20);
}

TEST_CASE("Pixel16fC3_additionalMethods", "[Common]")
{
    Pixel16fC3 roundA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f));
    Pixel16fC3 roundB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f));
    Pixel16fC3 round2A = Pixel16fC3::Round(roundA);
    Pixel16fC3 round2B = Pixel16fC3::Round(roundB);
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

    Pixel16fC3 floorA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f));
    Pixel16fC3 floorB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f));
    Pixel16fC3 floor2A = Pixel16fC3::Floor(floorA);
    Pixel16fC3 floor2B = Pixel16fC3::Floor(floorB);
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

    Pixel16fC3 ceilA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f));
    Pixel16fC3 ceilB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f));
    Pixel16fC3 ceil2A = Pixel16fC3::Ceil(ceilA);
    Pixel16fC3 ceil2B = Pixel16fC3::Ceil(ceilB);
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

    Pixel16fC3 zeroA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f));
    Pixel16fC3 zeroB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f));
    Pixel16fC3 zero2A = Pixel16fC3::RoundZero(zeroA);
    Pixel16fC3 zero2B = Pixel16fC3::RoundZero(zeroB);
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

    Pixel16fC3 nearestA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f));
    Pixel16fC3 nearestB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f));
    Pixel16fC3 nearest2A = Pixel16fC3::RoundNearest(nearestA);
    Pixel16fC3 nearest2B = Pixel16fC3::RoundNearest(nearestB);
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

    Pixel16fC3 exp(HalfFp16(2.4f), HalfFp16(12.5f), HalfFp16(-14.6f));
    Pixel16fC3 exp2 = Pixel16fC3::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == Approx(11.023176380641601).margin(0.01));
    CHECK(isinf(exp.y));
    CHECK(exp.z == Approx(4.563526367903994e-07).margin(0.01));

    Pixel16fC3 ln(HalfFp16(2.4f), HalfFp16(12.5f), HalfFp16(14.6f));
    Pixel16fC3 ln2 = Pixel16fC3::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == Approx(0.875468737353900).margin(0.01));
    CHECK(ln.y == Approx(2.525728644308256).margin(0.01));
    CHECK(ln.z == Approx(2.681021528714291).margin(0.01));

    Pixel16fC3 sqr(HalfFp16(2.4f), HalfFp16(12.5f), HalfFp16(-14.6f));
    Pixel16fC3 sqr2 = Pixel16fC3::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == Approx(5.76).margin(0.01));
    CHECK(sqr.y == Approx(156.25).margin(0.01));
    CHECK(sqr.z == Approx(213.16).margin(0.1));

    Pixel16fC3 sqrt(HalfFp16(2.4f), HalfFp16(12.5f), HalfFp16(14.6f));
    Pixel16fC3 sqrt2 = Pixel16fC3::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == Approx(1.549193338482967).margin(0.01));
    CHECK(sqrt.y == Approx(3.535533905932738).margin(0.01));
    CHECK(sqrt.z == Approx(3.820994634908560).margin(0.01));

    Pixel16fC3 abs(HalfFp16(-2.4f), HalfFp16(12.5f), HalfFp16(-14.6f));
    Pixel16fC3 abs2 = Pixel16fC3::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == HalfFp16(2.4f));
    CHECK(abs.y == HalfFp16(12.5f));
    CHECK(abs.z == HalfFp16(14.6f));

    Pixel16fC3 absdiffA(HalfFp16(13.23592f), HalfFp16(-40.24595f), HalfFp16(-22.15017f));
    Pixel16fC3 absdiffB(HalfFp16(45.75068f), HalfFp16(46.488853f), HalfFp16(-34.238691f));
    Pixel16fC3 absdiff2 = Pixel16fC3::AbsDiff(absdiffA, absdiffB);
    Pixel16fC3 absdiff3 = Pixel16fC3::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == Approx(32.514758920888809).margin(0.02));
    CHECK(absdiffA.y == Approx(86.734813019986703).margin(0.02));
    CHECK(absdiffA.z == Approx(12.088513718950011).margin(0.02));

    Pixel16fC3 clampByte(HalfFp16(float(numeric_limits<byte>::max()) + 1),
                         HalfFp16(float(numeric_limits<byte>::min()) - 1),
                         HalfFp16(float(numeric_limits<byte>::min())));
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 256);
    CHECK(clampByte.y == -1);
    CHECK(clampByte.z == 0);

    Pixel16fC3 clampShort(HalfFp16(float(numeric_limits<short>::max()) + 1),
                          HalfFp16(float(numeric_limits<short>::min()) - 1),
                          HalfFp16(float(numeric_limits<short>::min())));
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32768);
    CHECK(clampShort.y == -32768);
    CHECK(clampShort.z == -32768);

    Pixel16fC3 clampSByte(HalfFp16(float(numeric_limits<sbyte>::max()) + 1),
                          HalfFp16(float(numeric_limits<sbyte>::min()) - 1),
                          HalfFp16(float(numeric_limits<sbyte>::min())));
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 128);
    CHECK(clampSByte.y == -129);
    CHECK(clampSByte.z == -128);

    Pixel16fC3 clampUShort(HalfFp16(0.0f), HalfFp16(float(numeric_limits<ushort>::min()) - 1),
                           HalfFp16(float(numeric_limits<ushort>::min())));

    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 0);
    CHECK(clampUShort.y == -1);
    CHECK(clampUShort.z == 0);

    CHECK_FALSE(need_saturation_clamp_v<HalfFp16, int>);

    CHECK_FALSE(need_saturation_clamp_v<HalfFp16, uint>);
    Pixel16fC3 clampUInt(HalfFp16(0), HalfFp16(float(numeric_limits<uint>::min()) - 1000),
                         HalfFp16(float(numeric_limits<uint>::min())));

    clampUInt.template ClampToTargetType<uint>();
    CHECK(clampUInt.x == 0);
    CHECK(clampUInt.y == -1000);
    CHECK(clampUInt.z == 0);

    Pixel32fC3 clampFloatToFp16(float(numeric_limits<int>::min()) - 1000.0f, //
                                float(numeric_limits<int>::max()) + 1000.0f, //
                                float(numeric_limits<short>::min()));

    clampFloatToFp16.template ClampToTargetType<HalfFp16>();
    CHECK(clampFloatToFp16.x == -2147484672.0f);
    CHECK(clampFloatToFp16.y == 2147484672.0f);
    CHECK(clampFloatToFp16.z == -32768);

    Pixel16fC3 fromFromFloat(clampFloatToFp16);
    CHECK(fromFromFloat.x == -INFINITY);
    CHECK(fromFromFloat.y == INFINITY);
    CHECK(fromFromFloat.z == -32768);
}

TEST_CASE("Pixel32fcC3", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32fcC3) == 6 * sizeof(float));

    Complex<float> arr[3] = {Complex<float>(4, 2), Complex<float>(5, -2), Complex<float>(6, 3)};
    Pixel32fcC3 t0(arr);
    CHECK(t0.x == Complex<float>(4, 2));
    CHECK(t0.y == Complex<float>(5, -2));
    CHECK(t0.z == Complex<float>(6, 3));

    Pixel32fcC3 t1(Complex<float>(0, 1), Complex<float>(1, -2), Complex<float>(2, 3));
    CHECK(t1.x == Complex<float>(0, 1));
    CHECK(t1.y == Complex<float>(1, -2));
    CHECK(t1.z == Complex<float>(2, 3));

    Pixel32fcC3 c(t1);
    CHECK(c.x == Complex<float>(0, 1));
    CHECK(c.y == Complex<float>(1, -2));
    CHECK(c.z == Complex<float>(2, 3));
    CHECK(c == t1);

    Pixel32fcC3 c2 = t1;
    CHECK(c2.x == Complex<float>(0, 1));
    CHECK(c2.y == Complex<float>(1, -2));
    CHECK(c2.z == Complex<float>(2, 3));
    CHECK(c2 == t1);

    Pixel32fcC3 t2(5);
    CHECK(t2.x == Complex<float>(5, 0));
    CHECK(t2.y == Complex<float>(5, 0));
    CHECK(t2.z == Complex<float>(5, 0));
    CHECK(c2 != t2);

    Pixel32fcC3 neg = -t1;
    CHECK(neg.x == -1_i);
    CHECK(neg.y == -1 + 2_i);
    CHECK(neg.z == -2 - 3_i);
    CHECK(t1 == -neg);

    Pixel32fcC3 add1 = t1 + t2;
    CHECK(add1.x == Complex<float>(5, 1));
    CHECK(add1.y == Complex<float>(6, -2));
    CHECK(add1.z == Complex<float>(7, 3));

    Pixel32fcC3 add2 = 3 + t1;
    CHECK(add2.x == Complex<float>(3, 1));
    CHECK(add2.y == Complex<float>(4, -2));
    CHECK(add2.z == Complex<float>(5, 3));

    Pixel32fcC3 add3 = t1 + 4;
    CHECK(add3.x == Complex<float>(4, 1));
    CHECK(add3.y == Complex<float>(5, -2));
    CHECK(add3.z == Complex<float>(6, 3));

    Pixel32fcC3 add4 = t1;
    add4 += add3;
    CHECK(add4.x == Complex<float>(4, 2));
    CHECK(add4.y == Complex<float>(6, -4));
    CHECK(add4.z == Complex<float>(8, 6));

    add4 += 3.0f + 0_i;
    CHECK(add4.x == Complex<float>(7, 2));
    CHECK(add4.y == Complex<float>(9, -4));
    CHECK(add4.z == Complex<float>(11, 6));

    Pixel32fcC3 sub1 = t1 - t2;
    CHECK(sub1.x == Complex<float>(-5, 1));
    CHECK(sub1.y == Complex<float>(-4, -2));
    CHECK(sub1.z == Complex<float>(-3, 3));

    Pixel32fcC3 sub2 = 3 - t1;
    CHECK(sub2.x == Complex<float>(3, -1));
    CHECK(sub2.y == Complex<float>(2, +2));
    CHECK(sub2.z == Complex<float>(1, -3));

    Pixel32fcC3 sub3 = t1 - 4;
    CHECK(sub3.x == Complex<float>(-4, 1));
    CHECK(sub3.y == Complex<float>(-3, -2));
    CHECK(sub3.z == Complex<float>(-2, 3));

    Pixel32fcC3 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == Complex<float>(4, 0));
    CHECK(sub4.y == Complex<float>(4, 0));
    CHECK(sub4.z == Complex<float>(4, 0));

    sub4 -= 3.0f + 0_i;
    CHECK(sub4.x == Complex<float>(1, 0));
    CHECK(sub4.y == Complex<float>(1, 0));
    CHECK(sub4.z == Complex<float>(1, 0));

    Pixel32fcC3 sub5 = t1;
    Pixel32fcC3 sub6(9, 8, 7);
    sub5.SubInv(sub6);
    CHECK(sub5.x == Complex<float>(9, -1));
    CHECK(sub5.y == Complex<float>(7, 2));
    CHECK(sub5.z == Complex<float>(5, -3));

    t1               = Pixel32fcC3(Complex<float>(4, 5), Complex<float>(6, 7), Complex<float>(8, 9));
    t2               = Pixel32fcC3(Complex<float>(5, 6), Complex<float>(7, -8), Complex<float>(9, -5));
    Pixel32fcC3 mul1 = t1 * t2;
    CHECK(mul1.x == Complex<float>(-10, 49));
    CHECK(mul1.y == Complex<float>(98, 1));
    CHECK(mul1.z == Complex<float>(117, 41));

    Pixel32fcC3 mul2 = 3 * t1;
    CHECK(mul2.x == Complex<float>(12, 15));
    CHECK(mul2.y == Complex<float>(18, 21));
    CHECK(mul2.z == Complex<float>(24, 27));

    Pixel32fcC3 mul3 = t1 * 4;
    CHECK(mul3.x == Complex<float>(16, 20));
    CHECK(mul3.y == Complex<float>(24, 28));
    CHECK(mul3.z == Complex<float>(32, 36));

    Pixel32fcC3 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == Complex<float>(-36, 160));
    CHECK(mul4.y == Complex<float>(-52, 336));
    CHECK(mul4.z == Complex<float>(-68, 576));

    mul4 *= 3.0f + 0_i;
    CHECK(mul4.x == Complex<float>(-108, 480));
    CHECK(mul4.y == Complex<float>(-156, 1008));
    CHECK(mul4.z == Complex<float>(-204, 1728));

    Pixel32fcC3 div1 = t1 / t2;
    CHECK(div1.x.real == Approx(0.819672108f).margin(0.001));
    CHECK(div1.x.imag == Approx(0.016393442f).margin(0.001));
    CHECK(div1.y.real == Approx(-0.123893805f).margin(0.001));
    CHECK(div1.y.imag == Approx(0.85840708f).margin(0.001));
    CHECK(div1.z.real == Approx(0.254716992f).margin(0.001));
    CHECK(div1.z.imag == Approx(1.141509414f).margin(0.001));

    Pixel32fcC3 div2 = 3 / t1;
    CHECK(div2.x.real == Approx(0.292682916f).margin(0.001));
    CHECK(div2.x.imag == Approx(-0.365853667f).margin(0.001));
    CHECK(div2.y.real == Approx(0.211764708f).margin(0.001));
    CHECK(div2.y.imag == Approx(-0.247058824f).margin(0.001));
    CHECK(div2.z.real == Approx(0.165517241f).margin(0.001));
    CHECK(div2.z.imag == Approx(-0.186206892f).margin(0.001));

    Pixel32fcC3 div3 = t1 / 4;
    CHECK(div3.x.real == Approx(1).margin(0.001));
    CHECK(div3.x.imag == Approx(1.25).margin(0.001));
    CHECK(div3.y.real == Approx(1.5).margin(0.001));
    CHECK(div3.y.imag == Approx(1.75).margin(0.001));
    CHECK(div3.z.real == Approx(2).margin(0.001));
    CHECK(div3.z.imag == Approx(2.25).margin(0.001));

    Pixel32fcC3 div4 = t2;
    div4 /= div3;
    CHECK(div4.x.real == Approx(4.878048897f).margin(0.001));
    CHECK(div4.x.imag == Approx(-0.097560972f).margin(0.001));
    CHECK(div4.y.real == Approx(-0.65882355f).margin(0.001));
    CHECK(div4.y.imag == Approx(-4.564705849f).margin(0.001));
    CHECK(div4.z.real == Approx(0.744827569f).margin(0.001));
    CHECK(div4.z.imag == Approx(-3.337930918f).margin(0.001));

    div4 /= 3.0f + 0_i;
    CHECK(div4.x.real == Approx(1.626016259f).margin(0.001));
    CHECK(div4.x.imag == Approx(-0.032520324f).margin(0.001));
    CHECK(div4.y.real == Approx(-0.21960786f).margin(0.001));
    CHECK(div4.y.imag == Approx(-1.521568656f).margin(0.001));
    CHECK(div4.z.real == Approx(0.248275861f).margin(0.001));
    CHECK(div4.z.imag == Approx(-1.112643719f).margin(0.001));

    Pixel32fcC3 div5 = t1;
    Pixel32fcC3 div6(9, 8, 7);
    Pixel32fcC3 difv7 = div6 / div5;
    div5.DivInv(div6);
    CHECK(div5.x == difv7.x);
    CHECK(div5.y == difv7.y);
    CHECK(div5.z == difv7.z);

    Pixel32fcC3 conj1 = t1;
    Pixel32fcC3 conj2 = Pixel32fcC3::Conj(t1);
    conj1.Conj();
    CHECK(conj1.x == 4 - 5_i);
    CHECK(conj1.y == 6 - 7_i);
    CHECK(conj1.z == 8 - 9_i);
    CHECK(conj2.x == 4 - 5_i);
    CHECK(conj2.y == 6 - 7_i);
    CHECK(conj2.z == 8 - 9_i);

    conj1 = t1;
    conj1.ConjMul(t2);
    conj2 = Pixel32fcC3::ConjMul(t1, t2);
    CHECK(conj1.x == 50 + 1_i);
    CHECK(conj1.y == -14 + 97_i);
    CHECK(conj1.z == 27 + 121_i);
    CHECK(conj2.x == 50 + 1_i);
    CHECK(conj2.y == -14 + 97_i);
    CHECK(conj2.z == 27 + 121_i);

    Pixel32fcC3 a = t1;
    Pixel32fcC3 s = t1;
    Pixel32fcC3 m = t1;
    Pixel32fcC3 d = t1;

    a += 5.0f;
    s -= 5.0f;
    m *= 5.0f;
    d /= 5.0f;
    CHECK(a == t1 + 5.0f);
    CHECK(s == t1 - 5.0f);
    CHECK(m == t1 * 5.0f);
    CHECK(d == t1 / 5.0f);
}

TEST_CASE("Pixel32fcC3_additionalMethods", "[Common]")
{
    Pixel32fcC3 roundA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f), Complex<float>(1.9f, -1.5f));
    Pixel32fcC3 round2A = Pixel32fcC3::Round(roundA);
    roundA.Round();
    CHECK(round2A == roundA);
    CHECK(roundA.x.real == 0.0f);
    CHECK(roundA.x.imag == 1.0f);
    CHECK(roundA.y.real == 1.0f);
    CHECK(roundA.y.imag == 2.0f);
    CHECK(roundA.z.real == 2.0f);
    CHECK(roundA.z.imag == -2.0f);

    Pixel32fcC3 floorA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f), Complex<float>(1.9f, -1.5f));
    Pixel32fcC3 floor2A = Pixel32fcC3::Floor(floorA);
    floorA.Floor();
    CHECK(floor2A == floorA);
    CHECK(floorA.x.real == 0.0f);
    CHECK(floorA.x.imag == 0.0f);
    CHECK(floorA.y.real == 0.0f);
    CHECK(floorA.y.imag == 1.0f);
    CHECK(floorA.z.real == 1.0f);
    CHECK(floorA.z.imag == -2.0f);

    Pixel32fcC3 ceilA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f), Complex<float>(1.9f, -1.5f));
    Pixel32fcC3 ceil2A = Pixel32fcC3::Ceil(ceilA);
    ceilA.Ceil();
    CHECK(ceil2A == ceilA);
    CHECK(ceilA.x.real == 1.0f);
    CHECK(ceilA.x.imag == 1.0f);
    CHECK(ceilA.y.real == 1.0f);
    CHECK(ceilA.y.imag == 2.0f);
    CHECK(ceilA.z.real == 2.0f);
    CHECK(ceilA.z.imag == -1.0f);

    Pixel32fcC3 zeroA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f), Complex<float>(1.9f, -1.5f));
    Pixel32fcC3 zero2A = Pixel32fcC3::RoundZero(zeroA);
    zeroA.RoundZero();
    CHECK(zero2A == zeroA);
    CHECK(zeroA.x.real == 0.0f);
    CHECK(zeroA.x.imag == 0.0f);
    CHECK(zeroA.y.real == 0.0f);
    CHECK(zeroA.y.imag == 1.0f);
    CHECK(zeroA.z.real == 1.0f);
    CHECK(zeroA.z.imag == -1.0f);

    Pixel32fcC3 nearestA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f), Complex<float>(1.9f, -1.5f));
    Pixel32fcC3 nearest2A = Pixel32fcC3::RoundNearest(nearestA);
    nearestA.RoundNearest();
    CHECK(nearest2A == nearestA);
    CHECK(nearestA.x.real == 0.0f);
    CHECK(nearestA.x.imag == 0.0f);
    CHECK(nearestA.y.real == 1.0f);
    CHECK(nearestA.y.imag == 2.0f);
    CHECK(nearestA.z.real == 2.0f);
    CHECK(nearestA.z.imag == -2.0f);

    // Not vector does the value clampling but the complex type!
    Pixel32fcC3 clampToShort(float(numeric_limits<short>::max()) + 1, float(numeric_limits<short>::min()) - 1,
                             float(numeric_limits<short>::min()));
    Pixel16scC3 clampShort(clampToShort);
    CHECK(clampShort.x.real == 32767);
    CHECK(clampShort.y.real == -32768);
    CHECK(clampShort.z.real == -32768);

    Pixel32fcC3 clampToInt(float(numeric_limits<int>::max()) + 1000, float(numeric_limits<int>::min()) - 1000,
                           float(numeric_limits<int>::min()));

    Pixel32scC3 clampInt(clampToInt);
    CHECK(clampInt.x.real == 2147483647);
    CHECK(clampInt.y.real == -2147483648);
    CHECK(clampInt.z.real == -2147483648);
}

TEST_CASE("Pixel64sC3", "[Common]")
{
    CHECK(Pixel64sC3(10, 11, -13).DivRound(Pixel64sC3(3, 4, 4)) == Pixel64sC3(3, 3, -3));
    CHECK(Pixel64sC3(10, 11, -13).DivRoundNearest(Pixel64sC3(3, 4, 4)) == Pixel64sC3(3, 3, -3));
    CHECK(Pixel64sC3(10, 11, -13).DivRoundZero(Pixel64sC3(3, 4, 4)) == Pixel64sC3(3, 2, -3));
    CHECK(Pixel64sC3(10, 11, -13).DivFloor(Pixel64sC3(3, 4, 4)) == Pixel64sC3(3, 2, -4));
    CHECK(Pixel64sC3(10, 11, -13).DivCeil(Pixel64sC3(3, 4, 4)) == Pixel64sC3(4, 3, -3));

    CHECK(Pixel64sC3::DivRound(Pixel64sC3(10, 11, 13), Pixel64sC3(-3, -4, 4)) == Pixel64sC3(-3, -3, 3));
    CHECK(Pixel64sC3::DivRoundNearest(Pixel64sC3(10, 11, 13), Pixel64sC3(-3, -4, 4)) == Pixel64sC3(-3, -3, 3));
    CHECK(Pixel64sC3::DivRoundZero(Pixel64sC3(10, 11, 13), Pixel64sC3(-3, -4, 4)) == Pixel64sC3(-3, -2, 3));
    CHECK(Pixel64sC3::DivFloor(Pixel64sC3(10, 11, 13), Pixel64sC3(-3, -4, 4)) == Pixel64sC3(-4, -3, 3));
    CHECK(Pixel64sC3::DivCeil(Pixel64sC3(10, 11, 13), Pixel64sC3(-3, -4, 4)) == Pixel64sC3(-3, -2, 4));

    CHECK(Pixel64sC3(-9, 15, -15).DivScaleRoundNearest(10) == Pixel64sC3(-1, 2, -2));
}

TEST_CASE("Pixel64scC3", "[Common]")
{
    CHECK(Pixel64scC3(10 - 4_i, 11 + 4_i, -13 - 4_i).DivRound(Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i)) ==
          Pixel64scC3(2 - 2_i, 3, -2 - 2_i));
    CHECK(Pixel64scC3(10 - 4_i, 11 + 4_i, -13 - 4_i).DivRoundNearest(Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i)) ==
          Pixel64scC3(2 - 2_i, 3, -2 - 2_i));
    CHECK(Pixel64scC3(10 - 4_i, 11 + 4_i, -13 - 4_i).DivRoundZero(Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i)) ==
          Pixel64scC3(1 - 2_i, 2, -2 - 2_i));
    CHECK(Pixel64scC3(10 - 4_i, 11 + 4_i, -13 - 4_i).DivFloor(Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i)) ==
          Pixel64scC3(1 - 3_i, 2 - 1_i, -3 - 3_i));
    CHECK(Pixel64scC3(10 - 4_i, 11 + 4_i, -13 - 4_i).DivCeil(Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i)) ==
          Pixel64scC3(2 - 2_i, 3, -2 - 2_i));

    CHECK(Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i).DivInvRound(Pixel64scC3(10 - 4_i, 11 + 4_i, -13 - 4_i)) ==
          Pixel64scC3(2 - 2_i, 3, -2 - 2_i));
    CHECK(Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i).DivInvRoundNearest(Pixel64scC3(10 - 4_i, 11 + 4_i, -13 - 4_i)) ==
          Pixel64scC3(2 - 2_i, 3, -2 - 2_i));
    CHECK(Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i).DivInvRoundZero(Pixel64scC3(10 - 4_i, 11 + 4_i, -13 - 4_i)) ==
          Pixel64scC3(1 - 2_i, 2, -2 - 2_i));
    CHECK(Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i).DivInvFloor(Pixel64scC3(10 - 4_i, 11 + 4_i, -13 - 4_i)) ==
          Pixel64scC3(1 - 3_i, 2 - 1_i, -3 - 3_i));
    CHECK(Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i).DivInvCeil(Pixel64scC3(10 - 4_i, 11 + 4_i, -13 - 4_i)) ==
          Pixel64scC3(2 - 2_i, 3, -2 - 2_i));

    CHECK(Pixel64scC3::DivRound(Pixel64scC3(-10 + 4_i, -11 - 4_i, +13 + 4_i), Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i)) ==
          Pixel64scC3(-2 + 2_i, -3, 2 + 2_i));
    CHECK(Pixel64scC3::DivRoundNearest(Pixel64scC3(-10 + 4_i, -11 - 4_i, +13 + 4_i),
                                       Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i)) == Pixel64scC3(-2 + 2_i, -3, 2 + 2_i));
    CHECK(Pixel64scC3::DivRoundZero(Pixel64scC3(-10 + 4_i, -11 - 4_i, +13 + 4_i),
                                    Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i)) == Pixel64scC3(-1 + 2_i, -2, 2 + 2_i));
    CHECK(Pixel64scC3::DivFloor(Pixel64scC3(-10 + 4_i, -11 - 4_i, +13 + 4_i), Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i)) ==
          Pixel64scC3(-2 + 2_i, -3, 2 + 2_i));
    CHECK(Pixel64scC3::DivCeil(Pixel64scC3(-10 + 4_i, -11 - 4_i, +13 + 4_i), Pixel64scC3(3 + 2_i, 4 + 2_i, 4 - 2_i)) ==
          Pixel64scC3(-1 + 3_i, -2 + 1_i, 3 + 3_i));

    CHECK(Pixel64scC3(-9 + 11_i, 15 - 5_i, -15 + 5_i).DivScaleRoundNearest(10) == Pixel64scC3(-1 + 1_i, 2, -2));
}