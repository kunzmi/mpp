#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/numeric_limits.h>
#include <common/safeCast.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <cstdint>
#include <math.h>

using namespace opp;
using namespace opp::image;
using namespace Catch;

TEST_CASE("Pixel32fC4", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32fC4) == 4 * sizeof(float));

    float arr[4] = {4, 5, 6, 7};
    Pixel32fC4 t0(arr);
    CHECK(t0.x == 4);
    CHECK(t0.y == 5);
    CHECK(t0.z == 6);
    CHECK(t0.w == 7);

    Pixel32fC4 t1(0, 1, 2, 3);
    CHECK(t1.x == 0);
    CHECK(t1.y == 1);
    CHECK(t1.z == 2);
    CHECK(t1.w == 3);

    Pixel32fC4 c(t1);
    CHECK(c.x == 0);
    CHECK(c.y == 1);
    CHECK(c.z == 2);
    CHECK(c.w == 3);
    CHECK(c == t1);

    Pixel32fC4 c2 = t1;
    CHECK(c2.x == 0);
    CHECK(c2.y == 1);
    CHECK(c2.z == 2);
    CHECK(c2.w == 3);
    CHECK(c2 == t1);

    Pixel32fC4 t2(5);
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(t2.z == 5);
    CHECK(t2.w == 5);
    CHECK(c2 != t2);

    Pixel32fC4 add1 = t1 + t2;
    CHECK(add1.x == 5);
    CHECK(add1.y == 6);
    CHECK(add1.z == 7);
    CHECK(add1.w == 8);

    Pixel32fC4 add2 = 3 + t1;
    CHECK(add2.x == 3);
    CHECK(add2.y == 4);
    CHECK(add2.z == 5);
    CHECK(add2.w == 6);

    Pixel32fC4 add3 = t1 + 4;
    CHECK(add3.x == 4);
    CHECK(add3.y == 5);
    CHECK(add3.z == 6);
    CHECK(add3.w == 7);

    Pixel32fC4 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 4);
    CHECK(add4.y == 6);
    CHECK(add4.z == 8);
    CHECK(add4.w == 10);

    add4 += 3;
    CHECK(add4.x == 7);
    CHECK(add4.y == 9);
    CHECK(add4.z == 11);
    CHECK(add4.w == 13);

    Pixel32fC4 sub1 = t1 - t2;
    CHECK(sub1.x == -5);
    CHECK(sub1.y == -4);
    CHECK(sub1.z == -3);
    CHECK(sub1.w == -2);

    Pixel32fC4 sub2 = 3 - t1;
    CHECK(sub2.x == 3);
    CHECK(sub2.y == 2);
    CHECK(sub2.z == 1);
    CHECK(sub2.w == 0);

    Pixel32fC4 sub3 = t1 - 4;
    CHECK(sub3.x == -4);
    CHECK(sub3.y == -3);
    CHECK(sub3.z == -2);
    CHECK(sub3.w == -1);

    Pixel32fC4 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);
    CHECK(sub4.z == 4);
    CHECK(sub4.w == 4);

    sub4 -= 3;
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);
    CHECK(sub4.z == 1);
    CHECK(sub4.w == 1);

    Pixel32fC4 sub5 = t1;
    Pixel32fC4 sub6(9, 8, 7, 6);
    sub5.SubInv(sub6);
    CHECK(sub5.x == 9);
    CHECK(sub5.y == 7);
    CHECK(sub5.z == 5);
    CHECK(sub5.w == 3);

    t1              = Pixel32fC4(4, 5, 6, 7);
    t2              = Pixel32fC4(6, 7, 8, 9);
    Pixel32fC4 mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);
    CHECK(mul1.z == 48);
    CHECK(mul1.w == 63);

    Pixel32fC4 mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);
    CHECK(mul2.z == 18);
    CHECK(mul2.w == 21);

    Pixel32fC4 mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);
    CHECK(mul3.z == 24);
    CHECK(mul3.w == 28);

    Pixel32fC4 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);
    CHECK(mul4.z == 144);
    CHECK(mul4.w == 196);

    mul4 *= 3;
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);
    CHECK(mul4.z == 432);
    CHECK(mul4.w == 588);

    Pixel32fC4 div1 = t1 / t2;
    CHECK(div1.x == Approx(0.667).margin(0.001));
    CHECK(div1.y == Approx(0.714).margin(0.001));
    CHECK(div1.z == Approx(0.75).margin(0.001));
    CHECK(div1.w == Approx(0.778).margin(0.001));

    Pixel32fC4 div2 = 3 / t1;
    CHECK(div2.x == Approx(0.75).margin(0.001));
    CHECK(div2.y == Approx(0.6).margin(0.001));
    CHECK(div2.z == Approx(0.5).margin(0.001));
    CHECK(div2.w == Approx(0.429).margin(0.001));

    Pixel32fC4 div3 = t1 / 4;
    CHECK(div3.x == Approx(1.0).margin(0.001));
    CHECK(div3.y == Approx(1.25).margin(0.001));
    CHECK(div3.z == Approx(1.5).margin(0.001));
    CHECK(div3.w == Approx(1.75).margin(0.001));

    Pixel32fC4 div4 = t2;
    div4 /= div3;
    CHECK(div4.x == Approx(6.0).margin(0.001));
    CHECK(div4.y == Approx(5.6).margin(0.001));
    CHECK(div4.z == Approx(5.333).margin(0.001));
    CHECK(div4.w == Approx(5.143).margin(0.001));

    div4 /= 3;
    CHECK(div4.x == Approx(2.0).margin(0.001));
    CHECK(div4.y == Approx(1.867).margin(0.001));
    CHECK(div4.z == Approx(1.778).margin(0.001));
    CHECK(div4.w == Approx(1.714).margin(0.001));

    Pixel32fC4 div5 = t1;
    Pixel32fC4 div6(9, 8, 7, 6);
    Pixel32fC4 difv7 = div6 / div5;
    div5.DivInv(div6);
    CHECK(div5.x == difv7.x);
    CHECK(div5.y == difv7.y);
    CHECK(div5.z == difv7.z);
    CHECK(div5.w == difv7.w);

    Pixel32fC4 minmax1(10, 20, -10, 50);
    Pixel32fC4 minmax2(-20, 10, 40, 30);

    CHECK(Pixel32fC4::Min(minmax1, minmax2) == Pixel32fC4(-20, 10, -10, 30));
    CHECK(Pixel32fC4::Min(minmax2, minmax1) == Pixel32fC4(-20, 10, -10, 30));

    CHECK(Pixel32fC4::Max(minmax1, minmax2) == Pixel32fC4(10, 20, 40, 50));
    CHECK(Pixel32fC4::Max(minmax2, minmax1) == Pixel32fC4(10, 20, 40, 50));

    minmax1 = Pixel32fC4(10, 20, -10, 50);
    minmax2 = Pixel32fC4(-20, 10, 40, 30);
    CHECK(minmax1.Min(minmax2) == Pixel32fC4(-20, 10, -10, 30));
    CHECK(minmax2.Min(minmax1) == Pixel32fC4(-20, 10, -10, 30));

    minmax1 = Pixel32fC4(10, 20, -10, 50);
    minmax2 = Pixel32fC4(-20, 10, 40, 30);
    CHECK(minmax1.Max(minmax2) == Pixel32fC4(10, 20, 40, 50));
    CHECK(minmax2.Max(minmax1) == Pixel32fC4(10, 20, 40, 50));

    minmax1 = Pixel32fC4(10, 20, -10, 50);
    minmax2 = Pixel32fC4(-20, 10, 40, 30);
    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 50);
}

TEST_CASE("Pixel32fC4_additionalMethods", "[Common]")
{
    Pixel32fC4 roundA(0.4f, 0.5f, 0.6f, 1.5f);
    Pixel32fC4 roundB(1.9f, -1.5f, -2.5f, numeric_limits<float>::maxExact() - 0.5f);
    Pixel32fC4 round2A = Pixel32fC4::Round(roundA);
    Pixel32fC4 round2B = Pixel32fC4::Round(roundB);
    roundA.Round();
    roundB.Round();
    CHECK(round2A == roundA);
    CHECK(round2B == roundB);
    CHECK(roundA.x == 0.0f);
    CHECK(roundA.y == 1.0f);
    CHECK(roundA.z == 1.0f);
    CHECK(roundA.w == 2.0f);
    CHECK(roundB.x == 2.0f);
    CHECK(roundB.y == -2.0f);
    CHECK(roundB.z == -3.0f);
    CHECK(roundB.w == 16777216.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel32fC4 floorA(0.4f, 0.5f, 0.6f, 1.5f);
    Pixel32fC4 floorB(1.9f, -1.5f, -2.5f, numeric_limits<float>::maxExact() - 0.5f);
    Pixel32fC4 floor2A = Pixel32fC4::Floor(floorA);
    Pixel32fC4 floor2B = Pixel32fC4::Floor(floorB);
    floorA.Floor();
    floorB.Floor();
    CHECK(floor2A == floorA);
    CHECK(floor2B == floorB);
    CHECK(floorA.x == 0.0f);
    CHECK(floorA.y == 0.0f);
    CHECK(floorA.z == 0.0f);
    CHECK(floorA.w == 1.0f);
    CHECK(floorB.x == 1.0f);
    CHECK(floorB.y == -2.0f);
    CHECK(floorB.z == -3.0f);
    CHECK(floorB.w == 16777216.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel32fC4 ceilA(0.4f, 0.5f, 0.6f, 1.5f);
    Pixel32fC4 ceilB(1.9f, -1.5f, -2.5f, numeric_limits<float>::maxExact() - 0.5f);
    Pixel32fC4 ceil2A = Pixel32fC4::Ceil(ceilA);
    Pixel32fC4 ceil2B = Pixel32fC4::Ceil(ceilB);
    ceilA.Ceil();
    ceilB.Ceil();
    CHECK(ceil2A == ceilA);
    CHECK(ceil2B == ceilB);
    CHECK(ceilA.x == 1.0f);
    CHECK(ceilA.y == 1.0f);
    CHECK(ceilA.z == 1.0f);
    CHECK(ceilA.w == 2.0f);
    CHECK(ceilB.x == 2.0f);
    CHECK(ceilB.y == -1.0f);
    CHECK(ceilB.z == -2.0f);
    CHECK(ceilB.w == 16777216.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel32fC4 zeroA(0.4f, 0.5f, 0.6f, 1.5f);
    Pixel32fC4 zeroB(1.9f, -1.5f, -2.5f, numeric_limits<float>::maxExact() - 0.5f);
    Pixel32fC4 zero2A = Pixel32fC4::RoundZero(zeroA);
    Pixel32fC4 zero2B = Pixel32fC4::RoundZero(zeroB);
    zeroA.RoundZero();
    zeroB.RoundZero();
    CHECK(zero2A == zeroA);
    CHECK(zero2B == zeroB);
    CHECK(zeroA.x == 0.0f);
    CHECK(zeroA.y == 0.0f);
    CHECK(zeroA.z == 0.0f);
    CHECK(zeroA.w == 1.0f);
    CHECK(zeroB.x == 1.0f);
    CHECK(zeroB.y == -1.0f);
    CHECK(zeroB.z == -2.0f);
    CHECK(zeroB.w == 16777216.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel32fC4 nearestA(0.4f, 0.5f, 0.6f, 1.5f);
    Pixel32fC4 nearestB(1.9f, -1.5f, -2.5f, numeric_limits<float>::maxExact() - 0.5f);
    Pixel32fC4 nearest2A = Pixel32fC4::RoundNearest(nearestA);
    Pixel32fC4 nearest2B = Pixel32fC4::RoundNearest(nearestB);
    nearestA.RoundNearest();
    nearestB.RoundNearest();
    CHECK(nearest2A == nearestA);
    CHECK(nearest2B == nearestB);
    CHECK(nearestA.x == 0.0f);
    CHECK(nearestA.y == 0.0f);
    CHECK(nearestA.z == 1.0f);
    CHECK(nearestA.w == 2.0f);
    CHECK(nearestB.x == 2.0f);
    CHECK(nearestB.y == -2.0f);
    CHECK(nearestB.z == -2.0f);
    CHECK(nearestB.w == 16777216.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel32fC4 exp(2.4f, 12.5f, -14.6f, 20.0f);
    Pixel32fC4 exp2 = Pixel32fC4::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == Approx(11.023176380641601).margin(0.00001));
    CHECK(exp.y == Approx(2.683372865208745e+05).margin(0.00001));
    CHECK(exp.z == Approx(4.563526367903994e-07).margin(0.00001));
    CHECK(exp.w == Approx(4.851651954097903e+08).margin(0.00001));

    Pixel32fC4 ln(2.4f, 12.5f, 14.6f, 100.0f);
    Pixel32fC4 ln2 = Pixel32fC4::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == Approx(0.875468737353900).margin(0.00001));
    CHECK(ln.y == Approx(2.525728644308256).margin(0.00001));
    CHECK(ln.z == Approx(2.681021528714291).margin(0.00001));
    CHECK(ln.w == Approx(4.605170185988092).margin(0.00001));

    Pixel32fC4 sqr(2.4f, 12.5f, -14.6f, 20.0f);
    Pixel32fC4 sqr2 = Pixel32fC4::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == Approx(5.76).margin(0.00001));
    CHECK(sqr.y == Approx(156.25).margin(0.00001));
    CHECK(sqr.z == Approx(213.16).margin(0.00001));
    CHECK(sqr.w == Approx(400).margin(0.00001));

    Pixel32fC4 sqrt(2.4f, 12.5f, 14.6f, 100.0f);
    Pixel32fC4 sqrt2 = Pixel32fC4::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == Approx(1.549193338482967).margin(0.00001));
    CHECK(sqrt.y == Approx(3.535533905932738).margin(0.00001));
    CHECK(sqrt.z == Approx(3.820994634908560).margin(0.00001));
    CHECK(sqrt.w == Approx(10).margin(0.00001));

    Pixel32fC4 abs(-2.4f, 12.5f, -14.6f, 0.0f);
    Pixel32fC4 abs2 = Pixel32fC4::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == 2.4f);
    CHECK(abs.y == 12.5f);
    CHECK(abs.z == 14.6f);
    CHECK(abs.w == 0.0f);

    Pixel32fC4 absdiffA(13.23592f, -40.24595f, -22.15017f, 4.68815f);
    Pixel32fC4 absdiffB(45.75068f, 46.488853f, -34.238691f, -47.0592f);
    Pixel32fC4 absdiff2 = Pixel32fC4::AbsDiff(absdiffA, absdiffB);
    Pixel32fC4 absdiff3 = Pixel32fC4::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == Approx(32.514758920888809).margin(0.00001));
    CHECK(absdiffA.y == Approx(86.734813019986703).margin(0.00001));
    CHECK(absdiffA.z == Approx(12.088513718950011).margin(0.00001));
    CHECK(absdiffA.w == Approx(51.747430096559953).margin(0.00001));

    Pixel32fC4 clampByte(float(numeric_limits<byte>::max()) + 1, float(numeric_limits<byte>::min()) - 1,
                         float(numeric_limits<byte>::min()), float(numeric_limits<byte>::max()));
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 255);
    CHECK(clampByte.y == 0);
    CHECK(clampByte.z == 0);
    CHECK(clampByte.w == 255);

    Pixel32fC4 clampShort(float(numeric_limits<short>::max()) + 1, float(numeric_limits<short>::min()) - 1,
                          float(numeric_limits<short>::min()), float(numeric_limits<short>::max()));
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32767);
    CHECK(clampShort.y == -32768);
    CHECK(clampShort.z == -32768);
    CHECK(clampShort.w == 32767);

    Pixel32fC4 clampSByte(float(numeric_limits<sbyte>::max()) + 1, float(numeric_limits<sbyte>::min()) - 1,
                          float(numeric_limits<sbyte>::min()), float(numeric_limits<sbyte>::max()));
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 127);
    CHECK(clampSByte.y == -128);
    CHECK(clampSByte.z == -128);
    CHECK(clampSByte.w == 127);

    Pixel32fC4 clampUShort(float(numeric_limits<ushort>::max()) + 1, float(numeric_limits<ushort>::min()) - 1,
                           float(numeric_limits<ushort>::min()), float(numeric_limits<ushort>::max()));
    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 65535);
    CHECK(clampUShort.y == 0);
    CHECK(clampUShort.z == 0);
    CHECK(clampUShort.w == 65535);

    Pixel32fC4 clampInt(float(numeric_limits<int>::max()) + 1000, float(numeric_limits<int>::min()) - 1000,
                        float(numeric_limits<int>::min()), float(numeric_limits<int>::max()));
    clampInt.template ClampToTargetType<int>();
    CHECK(clampInt.x == 2147483520.0f);
    CHECK(clampInt.y == -2147483648.0f);
    CHECK(clampInt.z == -2147483648.0f);
    CHECK(clampInt.w == 2147483520.0f);

    Pixel32sC4 floatToInt(clampInt);
    CHECK(floatToInt.x == 2147483520);
    CHECK(floatToInt.y == -2147483648);
    CHECK(floatToInt.z == -2147483648);
    CHECK(floatToInt.w == 2147483520);

    Pixel32fC4 clampUInt(float(numeric_limits<uint>::max()) + 1000, float(numeric_limits<uint>::min()) - 1000,
                         float(numeric_limits<uint>::min()), float(numeric_limits<uint>::max()));
    clampUInt.template ClampToTargetType<uint>();
    CHECK(clampUInt.x == 0xffffffffUL);
    CHECK(clampUInt.y == 0);
    CHECK(clampUInt.z == 0);
    CHECK(clampUInt.w == 0xffffffffUL);
}

TEST_CASE("Pixel32sC4", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32sC4) == 4 * sizeof(int));

    Pixel32sC4 t1(100, 200, 300, 400);
    CHECK(t1.x == 100);
    CHECK(t1.y == 200);
    CHECK(t1.z == 300);
    CHECK(t1.w == 400);

    Pixel32sC4 c(t1);
    CHECK(c.x == 100);
    CHECK(c.y == 200);
    CHECK(c.z == 300);
    CHECK(c.w == 400);
    CHECK(c == t1);

    Pixel32sC4 c2 = t1;
    CHECK(c2.x == 100);
    CHECK(c2.y == 200);
    CHECK(c2.z == 300);
    CHECK(c2.w == 400);
    CHECK(c2 == t1);

    Pixel32sC4 t2(5);
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(t2.z == 5);
    CHECK(t2.w == 5);
    CHECK(c2 != t2);

    Pixel32sC4 add1 = t1 + t2;
    CHECK(add1.x == 105);
    CHECK(add1.y == 205);
    CHECK(add1.z == 305);
    CHECK(add1.w == 405);

    Pixel32sC4 add2 = 3 + t1;
    CHECK(add2.x == 103);
    CHECK(add2.y == 203);
    CHECK(add2.z == 303);
    CHECK(add2.w == 403);

    Pixel32sC4 add3 = t1 + 4;
    CHECK(add3.x == 104);
    CHECK(add3.y == 204);
    CHECK(add3.z == 304);
    CHECK(add3.w == 404);

    Pixel32sC4 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 204);
    CHECK(add4.y == 404);
    CHECK(add4.z == 604);
    CHECK(add4.w == 804);

    add4 += 3;
    CHECK(add4.x == 207);
    CHECK(add4.y == 407);
    CHECK(add4.z == 607);
    CHECK(add4.w == 807);

    Pixel32sC4 sub1 = t1 - t2;
    CHECK(sub1.x == 95);
    CHECK(sub1.y == 195);
    CHECK(sub1.z == 295);
    CHECK(sub1.w == 395);

    Pixel32sC4 sub2 = 3 - t1;
    CHECK(sub2.x == -97);
    CHECK(sub2.y == -197);
    CHECK(sub2.z == -297);
    CHECK(sub2.w == -397);

    Pixel32sC4 sub3 = t1 - 4;
    CHECK(sub3.x == 96);
    CHECK(sub3.y == 196);
    CHECK(sub3.z == 296);
    CHECK(sub3.w == 396);

    Pixel32sC4 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);
    CHECK(sub4.z == 4);
    CHECK(sub4.w == 4);

    sub4 -= 3;
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);
    CHECK(sub4.z == 1);
    CHECK(sub4.w == 1);

    t1              = Pixel32sC4(4, 5, 6, 7);
    t2              = Pixel32sC4(6, 7, 8, 9);
    Pixel32sC4 mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);
    CHECK(mul1.z == 48);
    CHECK(mul1.w == 63);

    Pixel32sC4 mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);
    CHECK(mul2.z == 18);
    CHECK(mul2.w == 21);

    Pixel32sC4 mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);
    CHECK(mul3.z == 24);
    CHECK(mul3.w == 28);

    Pixel32sC4 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);
    CHECK(mul4.z == 144);
    CHECK(mul4.w == 196);

    mul4 *= 3;
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);
    CHECK(mul4.z == 432);
    CHECK(mul4.w == 588);

    t1              = Pixel32sC4(1000, 2000, 3000, 4000);
    t2              = Pixel32sC4(6, 7, 8, 9);
    Pixel32sC4 div1 = t1 / t2;
    CHECK(div1.x == 166);
    CHECK(div1.y == 285);
    CHECK(div1.z == 375);
    CHECK(div1.w == 444);

    Pixel32sC4 div2 = 30000 / t1;
    CHECK(div2.x == 30);
    CHECK(div2.y == 15);
    CHECK(div2.z == 10);
    CHECK(div2.w == 7);

    Pixel32sC4 div3 = t1 / 4;
    CHECK(div3.x == 250);
    CHECK(div3.y == 500);
    CHECK(div3.z == 750);
    CHECK(div3.w == 1000);

    Pixel32sC4 div4 = t2 * 10000;
    div4 /= div3;
    CHECK(div4.x == 240);
    CHECK(div4.y == 140);
    CHECK(div4.z == 106);
    CHECK(div4.w == 90);

    div4 /= 3;
    CHECK(div4.x == 80);
    CHECK(div4.y == 46);
    CHECK(div4.z == 35);
    CHECK(div4.w == 30);

    Pixel32sC4 minmax1(10, 20, -10, 50);
    Pixel32sC4 minmax2(-20, 10, 40, 30);

    CHECK(Pixel32sC4::Min(minmax1, minmax2) == Pixel32sC4(-20, 10, -10, 30));
    CHECK(Pixel32sC4::Min(minmax2, minmax1) == Pixel32sC4(-20, 10, -10, 30));

    CHECK(Pixel32sC4::Max(minmax1, minmax2) == Pixel32sC4(10, 20, 40, 50));
    CHECK(Pixel32sC4::Max(minmax2, minmax1) == Pixel32sC4(10, 20, 40, 50));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 50);
}

TEST_CASE("Pixel32sC4_additionalMethods", "[Common]")
{
    Pixel32sC4 exp(4, 5, 6, 7);
    Pixel32sC4 exp2 = Pixel32sC4::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == 54);
    CHECK(exp.y == 148);
    CHECK(exp.z == 403);
    CHECK(exp.w == 1096);

    Pixel32sC4 ln(4, 50, 600, 700);
    Pixel32sC4 ln2 = Pixel32sC4::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == 1);
    CHECK(ln.y == 3);
    CHECK(ln.z == 6);
    CHECK(ln.w == 6);

    Pixel32sC4 sqr(4, 5, 6, 7);
    Pixel32sC4 sqr2 = Pixel32sC4::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == 16);
    CHECK(sqr.y == 25);
    CHECK(sqr.z == 36);
    CHECK(sqr.w == 49);

    Pixel32sC4 sqrt(4, 5, 6, 9);
    Pixel32sC4 sqrt2 = Pixel32sC4::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == 2);
    CHECK(sqrt.y == 2);
    CHECK(sqrt.z == 2);
    CHECK(sqrt.w == 3);

    Pixel32sC4 abs(-2, 12, -14, 0);
    Pixel32sC4 abs2 = Pixel32sC4::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == 2);
    CHECK(abs.y == 12);
    CHECK(abs.z == 14);
    CHECK(abs.w == 0);

    Pixel32sC4 clampByte(int(numeric_limits<byte>::max()) + 1, int(numeric_limits<byte>::min()) - 1,
                         int(numeric_limits<byte>::min()), int(numeric_limits<byte>::max()));
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 255);
    CHECK(clampByte.y == 0);
    CHECK(clampByte.z == 0);
    CHECK(clampByte.w == 255);

    Pixel32sC4 clampShort(int(numeric_limits<short>::max()) + 1, int(numeric_limits<short>::min()) - 1,
                          int(numeric_limits<short>::min()), int(numeric_limits<short>::max()));
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32767);
    CHECK(clampShort.y == -32768);
    CHECK(clampShort.z == -32768);
    CHECK(clampShort.w == 32767);

    Pixel32sC4 clampSByte(int(numeric_limits<sbyte>::max()) + 1, int(numeric_limits<sbyte>::min()) - 1,
                          int(numeric_limits<sbyte>::min()), int(numeric_limits<sbyte>::max()));
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 127);
    CHECK(clampSByte.y == -128);
    CHECK(clampSByte.z == -128);
    CHECK(clampSByte.w == 127);

    Pixel32sC4 clampUShort(int(numeric_limits<ushort>::max()) + 1, int(numeric_limits<ushort>::min()) - 1,
                           int(numeric_limits<ushort>::min()), int(numeric_limits<ushort>::max()));
    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 65535);
    CHECK(clampUShort.y == 0);
    CHECK(clampUShort.z == 0);
    CHECK(clampUShort.w == 65535);

    Pixel32sC4 clampInt(numeric_limits<int>::max(), numeric_limits<int>::min(), numeric_limits<int>::min(),
                        numeric_limits<int>::max());
    clampInt.template ClampToTargetType<int>();
    CHECK(clampInt.x == 2147483647);
    CHECK(clampInt.y == -2147483648);
    CHECK(clampInt.z == -2147483648);
    CHECK(clampInt.w == 2147483647);

    Pixel32sC4 lshift(1024, 2048, 4096, 8192);
    Pixel32sC4 lshift2 = Pixel32sC4::LShift(lshift, 2);
    lshift.LShift(2);
    CHECK(lshift == lshift2);
    CHECK(lshift.x == 4096);
    CHECK(lshift.y == 8192);
    CHECK(lshift.z == 16384);
    CHECK(lshift.w == 32768);

    Pixel32sC4 rshift(1024, 2048, 4096, 8192);
    Pixel32sC4 rshift2 = Pixel32sC4::RShift(rshift, 2);
    rshift.RShift(2);
    CHECK(rshift == rshift2);
    CHECK(rshift.x == 256);
    CHECK(rshift.y == 512);
    CHECK(rshift.z == 1024);
    CHECK(rshift.w == 2048);

    Pixel32sC4 and_(1023, 2047, 4095, 8191);
    Pixel32sC4 and_B(512, 1024, 2048, 4096);
    Pixel32sC4 and_2 = Pixel32sC4::And(and_, and_B);
    and_.And(and_B);
    CHECK(and_ == and_2);
    CHECK(and_.x == 512);
    CHECK(and_.y == 1024);
    CHECK(and_.z == 2048);
    CHECK(and_.w == 4096);

    Pixel32sC4 or_(1023, 2047, 4095, 8191);
    Pixel32sC4 or_B(512, 1024, 2048, 4096);
    Pixel32sC4 or_2 = Pixel32sC4::Or(or_, or_B);
    or_.Or(or_B);
    CHECK(or_ == or_2);
    CHECK(or_.x == 1023);
    CHECK(or_.y == 2047);
    CHECK(or_.z == 4095);
    CHECK(or_.w == 8191);

    Pixel32sC4 xor_(1023, 2047, 4095, 8191);
    Pixel32sC4 xor_B(512, 1024, 2048, 4096);
    Pixel32sC4 xor_2 = Pixel32sC4::Xor(xor_, xor_B);
    xor_.Xor(xor_B);
    CHECK(xor_ == xor_2);
    CHECK(xor_.x == 511);
    CHECK(xor_.y == 1023);
    CHECK(xor_.z == 2047);
    CHECK(xor_.w == 4095);

    Pixel32sC4 not_(1023, 2047, 4095, 8191);
    Pixel32sC4 not_2 = Pixel32sC4::Not(not_);
    not_.Not();
    CHECK(not_ == not_2);
    CHECK(not_.x == -1024);
    CHECK(not_.y == -2048);
    CHECK(not_.z == -4096);
    CHECK(not_.w == -8192);
}

TEST_CASE("Pixel32fC4_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel32fC4 *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel32fC4(float(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->x;
        void *ptrLastMember  = &buffer[i]->w;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<float *>(ptrVector) + 3));

        // vector must be 16 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 16 == 0);
    }

    std::vector<Pixel32fC4> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (4 * sizeof(float))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Pixel32sC4_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel32sC4 *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel32sC4(int(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->x;
        void *ptrLastMember  = &buffer[i]->w;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<int *>(ptrVector) + 3));

        // vector must be 16 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 16 == 0);
    }

    std::vector<Pixel32sC4> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (4 * sizeof(int))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Pixel16uC4_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel16uC4 *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel16uC4(ushort(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->x;
        void *ptrLastMember  = &buffer[i]->w;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<ushort *>(ptrVector) + 3));

        // vector must be 8 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 8 == 0);
    }

    std::vector<Pixel16uC4> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (4 * sizeof(ushort))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Pixel32sC4_streams", "[Common]")
{
    std::string str = "3 4 5 6";
    std::stringstream ss(str);

    Pixel32sC4 pix;
    ss >> pix;

    CHECK(pix.x == 3);
    CHECK(pix.y == 4);
    CHECK(pix.z == 5);
    CHECK(pix.w == 6);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3, 4, 5, 6)");
}

TEST_CASE("Pixel32fC4_streams", "[Common]")
{
    std::string str = "3.14 2.7 8.9 -12.6";
    std::stringstream ss(str);

    Pixel32fC4 pix;
    ss >> pix;

    CHECK(pix.x == 3.14f);
    CHECK(pix.y == 2.7f);
    CHECK(pix.z == 8.9f);
    CHECK(pix.w == -12.6f);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3.14, 2.7, 8.9, -12.6)");
}

TEST_CASE("Pixel16fC4_streams", "[Common]")
{
    std::string str = "3.14 2.7 8.9 -12.6";
    std::stringstream ss(str);

    Pixel16fC4 pix;
    ss >> pix;

    CHECK(pix.x == Approx(3.14f).margin(0.01));
    CHECK(pix.y == Approx(2.7f).margin(0.01));
    CHECK(pix.z == Approx(8.9f).margin(0.01));
    CHECK(pix.w == Approx(-12.6f).margin(0.01));

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3.14062, 2.69922, 8.89844, -12.6016)");
}

TEST_CASE("Pixel16bfC4_streams", "[Common]")
{
    std::string str = "3.14 2.7 8.9 -12.6";
    std::stringstream ss(str);

    Pixel16bfC4 pix;
    ss >> pix;

    CHECK(pix.x == Approx(3.14f).margin(0.01));
    CHECK(pix.y == Approx(2.7f).margin(0.01));
    CHECK(pix.z == Approx(8.9f).margin(0.1));
    CHECK(pix.w == Approx(-12.6f).margin(0.1));

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3.14062, 2.70312, 8.875, -12.625)");
}

TEST_CASE("Axis4D", "[Common]")
{
    Pixel32sC4 pix(2, 3, 4, 5);

    CHECK(pix[Axis4D::X] == 2);
    CHECK(pix[Axis4D::Y] == 3);
    CHECK(pix[Axis4D::Z] == 4);
    CHECK(pix[Axis4D::W] == 5);

    CHECK_THROWS_AS(pix[static_cast<Axis4D>(5)], opp::InvalidArgumentException);

    try
    {
        pix[static_cast<Axis4D>(5)] = 12;
    }
    catch (const opp::InvalidArgumentException &ex)
    {
        CHECK(ex.Message() == "Out of range: 5. Must be X, Y, Z or W (0, 1, 2 or 3).");
    }
}

TEST_CASE("Pixel16fC4", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel16fC4) == 4 * sizeof(HalfFp16));

    HalfFp16 arr[4] = {HalfFp16(4), HalfFp16(5), HalfFp16(6), HalfFp16(7)};
    Pixel16fC4 t0(arr);
    CHECK(t0.x == 4);
    CHECK(t0.y == 5);
    CHECK(t0.z == 6);
    CHECK(t0.w == 7);

    Pixel16fC4 t1(HalfFp16(0), HalfFp16(1), HalfFp16(2), HalfFp16(3));
    CHECK(t1.x == 0);
    CHECK(t1.y == 1);
    CHECK(t1.z == 2);
    CHECK(t1.w == 3);

    Pixel16fC4 c(t1);
    CHECK(c.x == 0);
    CHECK(c.y == 1);
    CHECK(c.z == 2);
    CHECK(c.w == 3);
    CHECK(c == t1);

    Pixel16fC4 c2 = t1;
    CHECK(c2.x == 0);
    CHECK(c2.y == 1);
    CHECK(c2.z == 2);
    CHECK(c2.w == 3);
    CHECK(c2 == t1);

    Pixel16fC4 t2(HalfFp16(5));
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(t2.z == 5);
    CHECK(t2.w == 5);
    CHECK(c2 != t2);

    Pixel16fC4 add1 = t1 + t2;
    CHECK(add1.x == 5);
    CHECK(add1.y == 6);
    CHECK(add1.z == 7);
    CHECK(add1.w == 8);

    Pixel16fC4 add2 = 3 + t1;
    CHECK(add2.x == 3);
    CHECK(add2.y == 4);
    CHECK(add2.z == 5);
    CHECK(add2.w == 6);

    Pixel16fC4 add3 = t1 + 4;
    CHECK(add3.x == 4);
    CHECK(add3.y == 5);
    CHECK(add3.z == 6);
    CHECK(add3.w == 7);

    Pixel16fC4 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 4);
    CHECK(add4.y == 6);
    CHECK(add4.z == 8);
    CHECK(add4.w == 10);

    add4 += HalfFp16(3);
    CHECK(add4.x == 7);
    CHECK(add4.y == 9);
    CHECK(add4.z == 11);
    CHECK(add4.w == 13);

    Pixel16fC4 sub1 = t1 - t2;
    CHECK(sub1.x == -5);
    CHECK(sub1.y == -4);
    CHECK(sub1.z == -3);
    CHECK(sub1.w == -2);

    Pixel16fC4 sub2 = 3 - t1;
    CHECK(sub2.x == 3);
    CHECK(sub2.y == 2);
    CHECK(sub2.z == 1);
    CHECK(sub2.w == 0);

    Pixel16fC4 sub3 = t1 - 4;
    CHECK(sub3.x == -4);
    CHECK(sub3.y == -3);
    CHECK(sub3.z == -2);
    CHECK(sub3.w == -1);

    Pixel16fC4 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);
    CHECK(sub4.z == 4);
    CHECK(sub4.w == 4);

    sub4 -= HalfFp16(3);
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);
    CHECK(sub4.z == 1);
    CHECK(sub4.w == 1);

    t1              = Pixel16fC4(HalfFp16(4), HalfFp16(5), HalfFp16(6), HalfFp16(7));
    t2              = Pixel16fC4(HalfFp16(6), HalfFp16(7), HalfFp16(8), HalfFp16(9));
    Pixel16fC4 mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);
    CHECK(mul1.z == 48);
    CHECK(mul1.w == 63);

    Pixel16fC4 mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);
    CHECK(mul2.z == 18);
    CHECK(mul2.w == 21);

    Pixel16fC4 mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);
    CHECK(mul3.z == 24);
    CHECK(mul3.w == 28);

    Pixel16fC4 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);
    CHECK(mul4.z == 144);
    CHECK(mul4.w == 196);

    mul4 *= HalfFp16(3);
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);
    CHECK(mul4.z == 432);
    CHECK(mul4.w == 588);

    Pixel16fC4 div1 = t1 / t2;
    CHECK(div1.x == Approx(0.667).margin(0.001));
    CHECK(div1.y == Approx(0.714).margin(0.001));
    CHECK(div1.z == Approx(0.75).margin(0.001));
    CHECK(div1.w == Approx(0.778).margin(0.001));

    Pixel16fC4 div2 = 3 / t1;
    CHECK(div2.x == Approx(0.75).margin(0.001));
    CHECK(div2.y == Approx(0.6).margin(0.001));
    CHECK(div2.z == Approx(0.5).margin(0.001));
    CHECK(div2.w == Approx(0.429).margin(0.001));

    Pixel16fC4 div3 = t1 / 4;
    CHECK(div3.x == Approx(1.0).margin(0.001));
    CHECK(div3.y == Approx(1.25).margin(0.001));
    CHECK(div3.z == Approx(1.5).margin(0.001));
    CHECK(div3.w == Approx(1.75).margin(0.001));

    Pixel16fC4 div4 = t2;
    div4 /= div3;
    CHECK(div4.x == Approx(6.0).margin(0.001));
    CHECK(div4.y == Approx(5.6).margin(0.01));
    CHECK(div4.z == Approx(5.333).margin(0.001));
    CHECK(div4.w == Approx(5.143).margin(0.01));

    div4 /= HalfFp16(3);
    CHECK(div4.x == Approx(2.0).margin(0.001));
    CHECK(div4.y == Approx(1.867).margin(0.001));
    CHECK(div4.z == Approx(1.778).margin(0.001));
    CHECK(div4.w == Approx(1.714).margin(0.001));

    Pixel16fC4 minmax1(HalfFp16(10), HalfFp16(20), HalfFp16(-10), HalfFp16(50));
    Pixel16fC4 minmax2(HalfFp16(-20), HalfFp16(10), HalfFp16(40), HalfFp16(30));

    CHECK(Pixel16fC4::Min(minmax1, minmax2) == Pixel16fC4(HalfFp16(-20), HalfFp16(10), HalfFp16(-10), HalfFp16(30)));
    CHECK(Pixel16fC4::Min(minmax2, minmax1) == Pixel16fC4(HalfFp16(-20), HalfFp16(10), HalfFp16(-10), HalfFp16(30)));

    CHECK(Pixel16fC4::Max(minmax1, minmax2) == Pixel16fC4(HalfFp16(10), HalfFp16(20), HalfFp16(40), HalfFp16(50)));
    CHECK(Pixel16fC4::Max(minmax2, minmax1) == Pixel16fC4(HalfFp16(10), HalfFp16(20), HalfFp16(40), HalfFp16(50)));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 50);
}

TEST_CASE("Pixel16fC4_additionalMethods", "[Common]")
{
    Pixel16fC4 roundA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f), HalfFp16(1.5f));
    Pixel16fC4 roundB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f),
                      HalfFp16(numeric_limits<HalfFp16>::maxExact() - 0.5f));
    Pixel16fC4 round2A = Pixel16fC4::Round(roundA);
    Pixel16fC4 round2B = Pixel16fC4::Round(roundB);
    roundA.Round();
    roundB.Round();
    CHECK(round2A == roundA);
    CHECK(round2B == roundB);
    CHECK(roundA.x == 0.0f);
    CHECK(roundA.y == 1.0f);
    CHECK(roundA.z == 1.0f);
    CHECK(roundA.w == 2.0f);
    CHECK(roundB.x == 2.0f);
    CHECK(roundB.y == -2.0f);
    CHECK(roundB.z == -3.0f);
    CHECK(roundB.w == 2048.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel16fC4 floorA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f), HalfFp16(1.5f));
    Pixel16fC4 floorB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f),
                      HalfFp16(numeric_limits<HalfFp16>::maxExact() - 0.5f));
    Pixel16fC4 floor2A = Pixel16fC4::Floor(floorA);
    Pixel16fC4 floor2B = Pixel16fC4::Floor(floorB);
    floorA.Floor();
    floorB.Floor();
    CHECK(floor2A == floorA);
    CHECK(floor2B == floorB);
    CHECK(floorA.x == 0.0f);
    CHECK(floorA.y == 0.0f);
    CHECK(floorA.z == 0.0f);
    CHECK(floorA.w == 1.0f);
    CHECK(floorB.x == 1.0f);
    CHECK(floorB.y == -2.0f);
    CHECK(floorB.z == -3.0f);
    CHECK(floorB.w == 2048.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel16fC4 ceilA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f), HalfFp16(1.5f));
    Pixel16fC4 ceilB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f),
                     HalfFp16(numeric_limits<HalfFp16>::maxExact() - 0.5f));
    Pixel16fC4 ceil2A = Pixel16fC4::Ceil(ceilA);
    Pixel16fC4 ceil2B = Pixel16fC4::Ceil(ceilB);
    ceilA.Ceil();
    ceilB.Ceil();
    CHECK(ceil2A == ceilA);
    CHECK(ceil2B == ceilB);
    CHECK(ceilA.x == 1.0f);
    CHECK(ceilA.y == 1.0f);
    CHECK(ceilA.z == 1.0f);
    CHECK(ceilA.w == 2.0f);
    CHECK(ceilB.x == 2.0f);
    CHECK(ceilB.y == -1.0f);
    CHECK(ceilB.z == -2.0f);
    CHECK(ceilB.w == 2048.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel16fC4 zeroA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f), HalfFp16(1.5f));
    Pixel16fC4 zeroB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f),
                     HalfFp16(numeric_limits<HalfFp16>::maxExact() - 0.5f));
    Pixel16fC4 zero2A = Pixel16fC4::RoundZero(zeroA);
    Pixel16fC4 zero2B = Pixel16fC4::RoundZero(zeroB);
    zeroA.RoundZero();
    zeroB.RoundZero();
    CHECK(zero2A == zeroA);
    CHECK(zero2B == zeroB);
    CHECK(zeroA.x == 0.0f);
    CHECK(zeroA.y == 0.0f);
    CHECK(zeroA.z == 0.0f);
    CHECK(zeroA.w == 1.0f);
    CHECK(zeroB.x == 1.0f);
    CHECK(zeroB.y == -1.0f);
    CHECK(zeroB.z == -2.0f);
    CHECK(zeroB.w == 2048.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel16fC4 nearestA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f), HalfFp16(1.5f));
    Pixel16fC4 nearestB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f),
                        HalfFp16(numeric_limits<HalfFp16>::maxExact() - 0.5f));
    Pixel16fC4 nearest2A = Pixel16fC4::RoundNearest(nearestA);
    Pixel16fC4 nearest2B = Pixel16fC4::RoundNearest(nearestB);
    nearestA.RoundNearest();
    nearestB.RoundNearest();
    CHECK(nearest2A == nearestA);
    CHECK(nearest2B == nearestB);
    CHECK(nearestA.x == 0.0f);
    CHECK(nearestA.y == 0.0f);
    CHECK(nearestA.z == 1.0f);
    CHECK(nearestA.w == 2.0f);
    CHECK(nearestB.x == 2.0f);
    CHECK(nearestB.y == -2.0f);
    CHECK(nearestB.z == -2.0f);
    CHECK(nearestB.w == 2048.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel16fC4 exp(HalfFp16(2.4f), HalfFp16(12.5f), HalfFp16(-14.6f), HalfFp16(20.0f));
    Pixel16fC4 exp2 = Pixel16fC4::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == Approx(11.023176380641601).margin(0.01));
    CHECK(isinf(exp.y));
    CHECK(exp.z == Approx(4.563526367903994e-07).margin(0.01));
    CHECK(isinf(exp.w));

    Pixel16fC4 ln(HalfFp16(2.4f), HalfFp16(12.5f), HalfFp16(14.6f), HalfFp16(100.0f));
    Pixel16fC4 ln2 = Pixel16fC4::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == Approx(0.875468737353900).margin(0.01));
    CHECK(ln.y == Approx(2.525728644308256).margin(0.01));
    CHECK(ln.z == Approx(2.681021528714291).margin(0.01));
    CHECK(ln.w == Approx(4.605170185988092).margin(0.01));

    Pixel16fC4 sqr(HalfFp16(2.4f), HalfFp16(12.5f), HalfFp16(-14.6f), HalfFp16(20.0f));
    Pixel16fC4 sqr2 = Pixel16fC4::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == Approx(5.76).margin(0.01));
    CHECK(sqr.y == Approx(156.25).margin(0.01));
    CHECK(sqr.z == Approx(213.16).margin(0.1));
    CHECK(sqr.w == Approx(400).margin(0.01));

    Pixel16fC4 sqrt(HalfFp16(2.4f), HalfFp16(12.5f), HalfFp16(14.6f), HalfFp16(100.0f));
    Pixel16fC4 sqrt2 = Pixel16fC4::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == Approx(1.549193338482967).margin(0.01));
    CHECK(sqrt.y == Approx(3.535533905932738).margin(0.01));
    CHECK(sqrt.z == Approx(3.820994634908560).margin(0.01));
    CHECK(sqrt.w == Approx(10).margin(0.01));

    Pixel16fC4 abs(HalfFp16(-2.4f), HalfFp16(12.5f), HalfFp16(-14.6f), HalfFp16(0.0f));
    Pixel16fC4 abs2 = Pixel16fC4::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == HalfFp16(2.4f));
    CHECK(abs.y == HalfFp16(12.5f));
    CHECK(abs.z == HalfFp16(14.6f));
    CHECK(abs.w == 0.0f);

    Pixel16fC4 absdiffA(HalfFp16(13.23592f), HalfFp16(-40.24595f), HalfFp16(-22.15017f), HalfFp16(4.68815f));
    Pixel16fC4 absdiffB(HalfFp16(45.75068f), HalfFp16(46.488853f), HalfFp16(-34.238691f), HalfFp16(-47.0592f));
    Pixel16fC4 absdiff2 = Pixel16fC4::AbsDiff(absdiffA, absdiffB);
    Pixel16fC4 absdiff3 = Pixel16fC4::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == Approx(32.514758920888809).margin(0.02));
    CHECK(absdiffA.y == Approx(86.734813019986703).margin(0.02));
    CHECK(absdiffA.z == Approx(12.088513718950011).margin(0.02));
    CHECK(absdiffA.w == Approx(51.747430096559953).margin(0.02));

    Pixel16fC4 clampByte(HalfFp16(float(numeric_limits<byte>::max()) + 1),
                         HalfFp16(float(numeric_limits<byte>::min()) - 1), HalfFp16(float(numeric_limits<byte>::min())),
                         HalfFp16(float(numeric_limits<byte>::max())));
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 255);
    CHECK(clampByte.y == 0);
    CHECK(clampByte.z == 0);
    CHECK(clampByte.w == 255);

    Pixel16fC4 clampShort(HalfFp16(float(numeric_limits<short>::max()) + 1),
                          HalfFp16(float(numeric_limits<short>::min()) - 1),
                          HalfFp16(float(numeric_limits<short>::min())), HalfFp16(float(numeric_limits<short>::max())));
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32752);
    CHECK(clampShort.y == -32768);
    CHECK(clampShort.z == -32768);
    CHECK(clampShort.w == 32752);

    Pixel16fC4 clampSByte(HalfFp16(float(numeric_limits<sbyte>::max()) + 1),
                          HalfFp16(float(numeric_limits<sbyte>::min()) - 1),
                          HalfFp16(float(numeric_limits<sbyte>::min())), HalfFp16(float(numeric_limits<sbyte>::max())));
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 127);
    CHECK(clampSByte.y == -128);
    CHECK(clampSByte.z == -128);
    CHECK(clampSByte.w == 127);

    Pixel16fC4 clampUShort(
        HalfFp16(0.0f), HalfFp16(float(numeric_limits<ushort>::min()) - 1),
        HalfFp16(float(numeric_limits<ushort>::min())),
        HalfFp16(float(numeric_limits<HalfFp16>::max()))); // UShort::max() is larger than HalfFp16::max() and would
                                                           // result in an inf
    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 0);
    CHECK(clampUShort.y == 0);
    CHECK(clampUShort.z == 0);
    CHECK(clampUShort.w == 65504); // = HalfFp16::max()

    CHECK_FALSE(need_saturation_clamp_v<HalfFp16, int>);

    CHECK(need_saturation_clamp_v<HalfFp16, uint>);
    Pixel16fC4 clampUInt(
        HalfFp16(0), HalfFp16(float(numeric_limits<uint>::min()) - 1000), HalfFp16(float(numeric_limits<uint>::min())),
        HalfFp16(float(numeric_limits<HalfFp16>::max()))); // UInt::max() is larger than HalfFp16::max() and would
                                                           // result in an inf
    clampUInt.template ClampToTargetType<uint>();
    CHECK(clampUInt.x == 0);
    CHECK(clampUInt.y == 0);
    CHECK(clampUInt.z == 0);
    CHECK(clampUInt.w == 65504); // = HalfFp16::max()

    Pixel32fC4 clampFloatToFp16(float(numeric_limits<int>::min()) - 1000.0f, //
                                float(numeric_limits<int>::max()) + 1000.0f, //
                                float(numeric_limits<short>::min()),         //
                                float(numeric_limits<short>::max()));

    clampFloatToFp16.template ClampToTargetType<HalfFp16>();
    CHECK(clampFloatToFp16.x == -65504);
    CHECK(clampFloatToFp16.y == 65504);
    CHECK(clampFloatToFp16.z == -32768);
    CHECK(clampFloatToFp16.w == 32767);

    Pixel16fC4 fromFromFloat(clampFloatToFp16);
    CHECK(fromFromFloat.x == -65504);
    CHECK(fromFromFloat.y == 65504);
    CHECK(fromFromFloat.z == -32768);
    CHECK(fromFromFloat.w == 32768); // 32767 gets rounded to 32768
}

TEST_CASE("Pixel16bfC4", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel16bfC4) == 2 * sizeof(float));

    BFloat16 arr[4] = {BFloat16(4), BFloat16(5), BFloat16(6), BFloat16(7)};
    Pixel16bfC4 t0(arr);
    CHECK(t0.x == 4);
    CHECK(t0.y == 5);
    CHECK(t0.z == 6);
    CHECK(t0.w == 7);

    Pixel16bfC4 t1(BFloat16(0), BFloat16(1), BFloat16(2), BFloat16(3));
    CHECK(t1.x == 0);
    CHECK(t1.y == 1);
    CHECK(t1.z == 2);
    CHECK(t1.w == 3);

    Pixel16bfC4 c(t1);
    CHECK(c.x == 0);
    CHECK(c.y == 1);
    CHECK(c.z == 2);
    CHECK(c.w == 3);
    CHECK(c == t1);

    Pixel16bfC4 c2 = t1;
    CHECK(c2.x == 0);
    CHECK(c2.y == 1);
    CHECK(c2.z == 2);
    CHECK(c2.w == 3);
    CHECK(c2 == t1);

    Pixel16bfC4 t2(BFloat16(5));
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(t2.z == 5);
    CHECK(t2.w == 5);
    CHECK(c2 != t2);

    Pixel16bfC4 add1 = t1 + t2;
    CHECK(add1.x == 5);
    CHECK(add1.y == 6);
    CHECK(add1.z == 7);
    CHECK(add1.w == 8);

    Pixel16bfC4 add2 = 3 + t1;
    CHECK(add2.x == 3);
    CHECK(add2.y == 4);
    CHECK(add2.z == 5);
    CHECK(add2.w == 6);

    Pixel16bfC4 add3 = t1 + 4;
    CHECK(add3.x == 4);
    CHECK(add3.y == 5);
    CHECK(add3.z == 6);
    CHECK(add3.w == 7);

    Pixel16bfC4 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 4);
    CHECK(add4.y == 6);
    CHECK(add4.z == 8);
    CHECK(add4.w == 10);

    add4 += BFloat16(3);
    CHECK(add4.x == 7);
    CHECK(add4.y == 9);
    CHECK(add4.z == 11);
    CHECK(add4.w == 13);

    Pixel16bfC4 sub1 = t1 - t2;
    CHECK(sub1.x == -5);
    CHECK(sub1.y == -4);
    CHECK(sub1.z == -3);
    CHECK(sub1.w == -2);

    Pixel16bfC4 sub2 = 3 - t1;
    CHECK(sub2.x == 3);
    CHECK(sub2.y == 2);
    CHECK(sub2.z == 1);
    CHECK(sub2.w == 0);

    Pixel16bfC4 sub3 = t1 - 4;
    CHECK(sub3.x == -4);
    CHECK(sub3.y == -3);
    CHECK(sub3.z == -2);
    CHECK(sub3.w == -1);

    Pixel16bfC4 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);
    CHECK(sub4.z == 4);
    CHECK(sub4.w == 4);

    sub4 -= BFloat16(3);
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);
    CHECK(sub4.z == 1);
    CHECK(sub4.w == 1);

    t1               = Pixel16bfC4(BFloat16(4), BFloat16(5), BFloat16(6), BFloat16(7));
    t2               = Pixel16bfC4(BFloat16(6), BFloat16(7), BFloat16(8), BFloat16(9));
    Pixel16bfC4 mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);
    CHECK(mul1.z == 48);
    CHECK(mul1.w == 63);

    Pixel16bfC4 mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);
    CHECK(mul2.z == 18);
    CHECK(mul2.w == 21);

    Pixel16bfC4 mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);
    CHECK(mul3.z == 24);
    CHECK(mul3.w == 28);

    Pixel16bfC4 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);
    CHECK(mul4.z == 144);
    CHECK(mul4.w == 196);

    mul4 *= BFloat16(3);
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);
    CHECK(mul4.z == 432);
    CHECK(mul4.w == 588);

    Pixel16bfC4 div1 = t1 / t2;
    CHECK(div1.x == Approx(0.667).margin(0.001));
    CHECK(div1.y == Approx(0.714).margin(0.001));
    CHECK(div1.z == Approx(0.75).margin(0.001));
    CHECK(div1.w == Approx(0.778).margin(0.001));

    Pixel16bfC4 div2 = 3 / t1;
    CHECK(div2.x == Approx(0.75).margin(0.001));
    CHECK(div2.y == Approx(0.6).margin(0.01));
    CHECK(div2.z == Approx(0.5).margin(0.001));
    CHECK(div2.w == Approx(0.429).margin(0.01));

    Pixel16bfC4 div3 = t1 / 4;
    CHECK(div3.x == Approx(1.0).margin(0.001));
    CHECK(div3.y == Approx(1.25).margin(0.001));
    CHECK(div3.z == Approx(1.5).margin(0.001));
    CHECK(div3.w == Approx(1.75).margin(0.001));

    Pixel16bfC4 div4 = t2;
    div4 /= div3;
    CHECK(div4.x == Approx(6.0).margin(0.001));
    CHECK(div4.y == Approx(5.6).margin(0.01));
    CHECK(div4.z == Approx(5.333).margin(0.02));
    CHECK(div4.w == Approx(5.143).margin(0.02));

    div4 /= BFloat16(3);
    CHECK(div4.x == Approx(2.0).margin(0.001));
    CHECK(div4.y == Approx(1.867).margin(0.001));
    CHECK(div4.z == Approx(1.778).margin(0.01));
    CHECK(div4.w == Approx(1.714).margin(0.01));

    Pixel16bfC4 minmax1(BFloat16(10), BFloat16(20), BFloat16(-10), BFloat16(50));
    Pixel16bfC4 minmax2(BFloat16(-20), BFloat16(10), BFloat16(40), BFloat16(30));

    CHECK(Pixel16bfC4::Min(minmax1, minmax2) == Pixel16bfC4(BFloat16(-20), BFloat16(10), BFloat16(-10), BFloat16(30)));
    CHECK(Pixel16bfC4::Min(minmax2, minmax1) == Pixel16bfC4(BFloat16(-20), BFloat16(10), BFloat16(-10), BFloat16(30)));

    CHECK(Pixel16bfC4::Max(minmax1, minmax2) == Pixel16bfC4(BFloat16(10), BFloat16(20), BFloat16(40), BFloat16(50)));
    CHECK(Pixel16bfC4::Max(minmax2, minmax1) == Pixel16bfC4(BFloat16(10), BFloat16(20), BFloat16(40), BFloat16(50)));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 50);
}

TEST_CASE("Pixel16bfC4_additionalMethods", "[Common]")
{
    Pixel16bfC4 roundA(BFloat16(0.4f), BFloat16(0.5f), BFloat16(0.6f), BFloat16(1.5f));
    Pixel16bfC4 roundB(BFloat16(1.9f), BFloat16(-1.5f), BFloat16(-2.5f),
                       BFloat16(numeric_limits<BFloat16>::maxExact() - 0.5f));
    Pixel16bfC4 round2A = Pixel16bfC4::Round(roundA);
    Pixel16bfC4 round2B = Pixel16bfC4::Round(roundB);
    roundA.Round();
    roundB.Round();
    CHECK(round2A == roundA);
    CHECK(round2B == roundB);
    CHECK(roundA.x == 0.0f);
    CHECK(roundA.y == 1.0f);
    CHECK(roundA.z == 1.0f);
    CHECK(roundA.w == 2.0f);
    CHECK(roundB.x == 2.0f);
    CHECK(roundB.y == -2.0f);
    CHECK(roundB.z == -3.0f);
    CHECK(roundB.w == 256.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel16bfC4 floorA(BFloat16(0.4f), BFloat16(0.5f), BFloat16(0.6f), BFloat16(1.5f));
    Pixel16bfC4 floorB(BFloat16(1.9f), BFloat16(-1.5f), BFloat16(-2.5f),
                       BFloat16(numeric_limits<BFloat16>::maxExact() - 0.5f));
    Pixel16bfC4 floor2A = Pixel16bfC4::Floor(floorA);
    Pixel16bfC4 floor2B = Pixel16bfC4::Floor(floorB);
    floorA.Floor();
    floorB.Floor();
    CHECK(floor2A == floorA);
    CHECK(floor2B == floorB);
    CHECK(floorA.x == 0.0f);
    CHECK(floorA.y == 0.0f);
    CHECK(floorA.z == 0.0f);
    CHECK(floorA.w == 1.0f);
    CHECK(floorB.x == 1.0f);
    CHECK(floorB.y == -2.0f);
    CHECK(floorB.z == -3.0f);
    CHECK(floorB.w == 256.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel16bfC4 ceilA(BFloat16(0.4f), BFloat16(0.5f), BFloat16(0.6f), BFloat16(1.5f));
    Pixel16bfC4 ceilB(BFloat16(1.9f), BFloat16(-1.5f), BFloat16(-2.5f),
                      BFloat16(numeric_limits<BFloat16>::maxExact() - 0.5f));
    Pixel16bfC4 ceil2A = Pixel16bfC4::Ceil(ceilA);
    Pixel16bfC4 ceil2B = Pixel16bfC4::Ceil(ceilB);
    ceilA.Ceil();
    ceilB.Ceil();
    CHECK(ceil2A == ceilA);
    CHECK(ceil2B == ceilB);
    CHECK(ceilA.x == 1.0f);
    CHECK(ceilA.y == 1.0f);
    CHECK(ceilA.z == 1.0f);
    CHECK(ceilA.w == 2.0f);
    CHECK(ceilB.x == 2.0f);
    CHECK(ceilB.y == -1.0f);
    CHECK(ceilB.z == -2.0f);
    CHECK(ceilB.w == 256.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel16bfC4 zeroA(BFloat16(0.4f), BFloat16(0.5f), BFloat16(0.6f), BFloat16(1.5f));
    Pixel16bfC4 zeroB(BFloat16(1.9f), BFloat16(-1.5f), BFloat16(-2.5f),
                      BFloat16(numeric_limits<BFloat16>::maxExact() - 0.5f));
    Pixel16bfC4 zero2A = Pixel16bfC4::RoundZero(zeroA);
    Pixel16bfC4 zero2B = Pixel16bfC4::RoundZero(zeroB);
    zeroA.RoundZero();
    zeroB.RoundZero();
    CHECK(zero2A == zeroA);
    CHECK(zero2B == zeroB);
    CHECK(zeroA.x == 0.0f);
    CHECK(zeroA.y == 0.0f);
    CHECK(zeroA.z == 0.0f);
    CHECK(zeroA.w == 1.0f);
    CHECK(zeroB.x == 1.0f);
    CHECK(zeroB.y == -1.0f);
    CHECK(zeroB.z == -2.0f);
    CHECK(zeroB.w == 256.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel16bfC4 nearestA(BFloat16(0.4f), BFloat16(0.5f), BFloat16(0.6f), BFloat16(1.5f));
    Pixel16bfC4 nearestB(BFloat16(1.9f), BFloat16(-1.5f), BFloat16(-2.5f),
                         BFloat16(numeric_limits<BFloat16>::maxExact() - 0.5f));
    Pixel16bfC4 nearest2A = Pixel16bfC4::RoundNearest(nearestA);
    Pixel16bfC4 nearest2B = Pixel16bfC4::RoundNearest(nearestB);
    nearestA.RoundNearest();
    nearestB.RoundNearest();
    CHECK(nearest2A == nearestA);
    CHECK(nearest2B == nearestB);
    CHECK(nearestA.x == 0.0f);
    CHECK(nearestA.y == 0.0f);
    CHECK(nearestA.z == 1.0f);
    CHECK(nearestA.w == 2.0f);
    CHECK(nearestB.x == 2.0f);
    CHECK(nearestB.y == -2.0f);
    CHECK(nearestB.z == -2.0f);
    CHECK(nearestB.w == 256.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel16bfC4 exp(BFloat16(2.4f), BFloat16(12.5f), BFloat16(-14.6f), BFloat16(20.0f));
    Pixel16bfC4 exp2 = Pixel16bfC4::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == Approx(11.023176380641601).margin(0.04));
    CHECK(exp.y == 268288);
    CHECK(exp.z == Approx(4.563526367903994e-07).margin(0.00001));
    CHECK(exp.w == Approx(484441984.0f).margin(1000));

    Pixel16bfC4 ln(BFloat16(2.4f), BFloat16(12.5f), BFloat16(14.6f), BFloat16(100.0f));
    Pixel16bfC4 ln2 = Pixel16bfC4::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == Approx(0.875468737353900).margin(0.01));
    CHECK(ln.y == Approx(2.525728644308256).margin(0.01));
    CHECK(ln.z == Approx(2.681021528714291).margin(0.01));
    CHECK(ln.w == Approx(4.605170185988092).margin(0.1));

    Pixel16bfC4 sqr(BFloat16(2.4f), BFloat16(12.5f), BFloat16(-14.6f), BFloat16(20.0f));
    Pixel16bfC4 sqr2 = Pixel16bfC4::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == Approx(5.76).margin(0.05));
    CHECK(sqr.y == Approx(156.25).margin(0.5));
    CHECK(sqr.z == Approx(213.16).margin(1));
    CHECK(sqr.w == Approx(400).margin(0.00001));

    Pixel16bfC4 sqrt(BFloat16(2.4f), BFloat16(12.5f), BFloat16(14.6f), BFloat16(100.0f));
    Pixel16bfC4 sqrt2 = Pixel16bfC4::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == Approx(1.549193338482967).margin(0.01));
    CHECK(sqrt.y == Approx(3.535533905932738).margin(0.01));
    CHECK(sqrt.z == Approx(3.820994634908560).margin(0.01));
    CHECK(sqrt.w == Approx(10).margin(0.01));

    Pixel16bfC4 abs(BFloat16(-2.4f), BFloat16(12.5f), BFloat16(-14.6f), BFloat16(0.0f));
    Pixel16bfC4 abs2 = Pixel16bfC4::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == 2.40625f);
    CHECK(abs.y == 12.5f);
    CHECK(abs.z == 14.625f);
    CHECK(abs.w == 0.0f);

    Pixel16bfC4 absdiffA(BFloat16(13.23592f), BFloat16(-40.24595f), BFloat16(-22.15017f), BFloat16(4.68815f));
    Pixel16bfC4 absdiffB(BFloat16(45.75068f), BFloat16(46.488853f), BFloat16(-34.238691f), BFloat16(-47.0592f));
    Pixel16bfC4 absdiff2 = Pixel16bfC4::AbsDiff(absdiffA, absdiffB);
    Pixel16bfC4 absdiff3 = Pixel16bfC4::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == Approx(32.514758920888809).margin(0.1));
    CHECK(absdiffA.y == Approx(86.734813019986703).margin(0.5));
    CHECK(absdiffA.z == Approx(12.088513718950011).margin(0.1));
    CHECK(absdiffA.w == Approx(51.747430096559953).margin(0.1));

    Pixel16bfC4 clampByte(BFloat16(float(numeric_limits<byte>::max()) + 1),
                          BFloat16(float(numeric_limits<byte>::min()) - 1),
                          BFloat16(float(numeric_limits<byte>::min())), BFloat16(float(numeric_limits<byte>::max())));
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 255.0f);
    CHECK(clampByte.y == 0.0f);
    CHECK(clampByte.z == 0.0f);
    CHECK(clampByte.w == 255.0f);

    Pixel16bfC4 clampShort(
        BFloat16(float(numeric_limits<short>::max()) + 1), BFloat16(float(numeric_limits<short>::min()) - 1),
        BFloat16(float(numeric_limits<short>::min())), BFloat16(float(numeric_limits<short>::max())));
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32640); // max value that we clamp BFloat to for shorts
    CHECK(clampShort.y == -32768.0f);
    CHECK(clampShort.z == -32768.0f);
    CHECK(clampShort.w == 32640);

    Pixel16bfC4 clampSByte(
        BFloat16(float(numeric_limits<sbyte>::max()) + 1), BFloat16(float(numeric_limits<sbyte>::min()) - 1),
        BFloat16(float(numeric_limits<sbyte>::min())), BFloat16(float(numeric_limits<sbyte>::max())));
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 127.0f);
    CHECK(clampSByte.y == -128.0f);
    CHECK(clampSByte.z == -128.0f);
    CHECK(clampSByte.w == 127.0f);

    Pixel16bfC4 clampUShort(
        BFloat16(float(numeric_limits<ushort>::max()) + 1), BFloat16(float(numeric_limits<ushort>::min()) - 1),
        BFloat16(float(numeric_limits<ushort>::min())), BFloat16(float(numeric_limits<ushort>::max())));
    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 65280); // max value that we clamp BFloat to for ushorts
    CHECK(clampUShort.y == 0.0f);
    CHECK(clampUShort.z == 0.0f);
    CHECK(clampUShort.w == 65280);

    Pixel16bfC4 clampInt(BFloat16(float(numeric_limits<int>::max()) + 16777216),
                         BFloat16(float(numeric_limits<int>::min()) - 16777216),
                         BFloat16(float(numeric_limits<int>::min())), BFloat16(float(numeric_limits<int>::max())));
    clampInt.template ClampToTargetType<int>();

    // The following values are even out of exact range for 32bit-floats and have no real meaning...
    float cech1 = 2147483639.0f;
    float cech2 = 2147483648.0f;
    CHECK(cech1 == cech2);
    CHECK(clampInt.x == 2147483647.0f);
    CHECK(clampInt.y == -2147483648.0f);
    CHECK(clampInt.z == -2147483648.0f);
    CHECK(clampInt.w == 2147483647.0f);

    Pixel16bfC4 clampUInt(BFloat16(float(numeric_limits<uint>::max()) + 16777216 * 2),
                          BFloat16(float(numeric_limits<uint>::min()) - 16777216 * 2),
                          BFloat16(float(numeric_limits<uint>::min())), BFloat16(float(numeric_limits<uint>::max())));
    clampUInt.template ClampToTargetType<uint>();
    CHECK(clampUInt.x == float(0xffffffffUL));
    CHECK(clampUInt.y == 0.0f);
    CHECK(clampUInt.z == 0.0f);
    CHECK(clampUInt.w == float(0xffffffffUL));
}

TEST_CASE("Pixel32fcC4", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32fcC4) == 8 * sizeof(float));

    Complex<float> arr[4] = {Complex<float>(4, 2), Complex<float>(5, -2), Complex<float>(6, 3), Complex<float>(7, -12)};
    Pixel32fcC4 t0(arr);
    CHECK(t0.x == Complex<float>(4, 2));
    CHECK(t0.y == Complex<float>(5, -2));
    CHECK(t0.z == Complex<float>(6, 3));
    CHECK(t0.w == Complex<float>(7, -12));

    Pixel32fcC4 t1(Complex<float>(0, 1), Complex<float>(1, -2), Complex<float>(2, 3), Complex<float>(3, -4));
    CHECK(t1.x == Complex<float>(0, 1));
    CHECK(t1.y == Complex<float>(1, -2));
    CHECK(t1.z == Complex<float>(2, 3));
    CHECK(t1.w == Complex<float>(3, -4));

    Pixel32fcC4 c(t1);
    CHECK(c.x == Complex<float>(0, 1));
    CHECK(c.y == Complex<float>(1, -2));
    CHECK(c.z == Complex<float>(2, 3));
    CHECK(c.w == Complex<float>(3, -4));
    CHECK(c == t1);

    Pixel32fcC4 c2 = t1;
    CHECK(c2.x == Complex<float>(0, 1));
    CHECK(c2.y == Complex<float>(1, -2));
    CHECK(c2.z == Complex<float>(2, 3));
    CHECK(c2.w == Complex<float>(3, -4));
    CHECK(c2 == t1);

    Pixel32fcC4 t2(5);
    CHECK(t2.x == Complex<float>(5, 0));
    CHECK(t2.y == Complex<float>(5, 0));
    CHECK(t2.z == Complex<float>(5, 0));
    CHECK(t2.w == Complex<float>(5, 0));
    CHECK(c2 != t2);

    Pixel32fcC4 add1 = t1 + t2;
    CHECK(add1.x == Complex<float>(5, 1));
    CHECK(add1.y == Complex<float>(6, -2));
    CHECK(add1.z == Complex<float>(7, 3));
    CHECK(add1.w == Complex<float>(8, -4));

    Pixel32fcC4 add2 = 3 + t1;
    CHECK(add2.x == Complex<float>(3, 1));
    CHECK(add2.y == Complex<float>(4, -2));
    CHECK(add2.z == Complex<float>(5, 3));
    CHECK(add2.w == Complex<float>(6, -4));

    Pixel32fcC4 add3 = t1 + 4;
    CHECK(add3.x == Complex<float>(4, 1));
    CHECK(add3.y == Complex<float>(5, -2));
    CHECK(add3.z == Complex<float>(6, 3));
    CHECK(add3.w == Complex<float>(7, -4));

    Pixel32fcC4 add4 = t1;
    add4 += add3;
    CHECK(add4.x == Complex<float>(4, 2));
    CHECK(add4.y == Complex<float>(6, -4));
    CHECK(add4.z == Complex<float>(8, 6));
    CHECK(add4.w == Complex<float>(10, -8));

    add4 += 3;
    CHECK(add4.x == Complex<float>(7, 2));
    CHECK(add4.y == Complex<float>(9, -4));
    CHECK(add4.z == Complex<float>(11, 6));
    CHECK(add4.w == Complex<float>(13, -8));

    Pixel32fcC4 sub1 = t1 - t2;
    CHECK(sub1.x == Complex<float>(-5, 1));
    CHECK(sub1.y == Complex<float>(-4, -2));
    CHECK(sub1.z == Complex<float>(-3, 3));
    CHECK(sub1.w == Complex<float>(-2, -4));

    Pixel32fcC4 sub2 = 3 - t1;
    CHECK(sub2.x == Complex<float>(3, 1));
    CHECK(sub2.y == Complex<float>(2, -2));
    CHECK(sub2.z == Complex<float>(1, 3));
    CHECK(sub2.w == Complex<float>(0, -4));

    Pixel32fcC4 sub3 = t1 - 4;
    CHECK(sub3.x == Complex<float>(-4, 1));
    CHECK(sub3.y == Complex<float>(-3, -2));
    CHECK(sub3.z == Complex<float>(-2, 3));
    CHECK(sub3.w == Complex<float>(-1, -4));

    Pixel32fcC4 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == Complex<float>(4, 0));
    CHECK(sub4.y == Complex<float>(4, 0));
    CHECK(sub4.z == Complex<float>(4, 0));
    CHECK(sub4.w == Complex<float>(4, 0));

    sub4 -= 3;
    CHECK(sub4.x == Complex<float>(1, 0));
    CHECK(sub4.y == Complex<float>(1, 0));
    CHECK(sub4.z == Complex<float>(1, 0));
    CHECK(sub4.w == Complex<float>(1, 0));

    Pixel32fcC4 sub5 = t1;
    Pixel32fcC4 sub6(9, 8, 7, 6);
    sub5.SubInv(sub6);
    CHECK(sub5.x == Complex<float>(9, -1));
    CHECK(sub5.y == Complex<float>(7, 2));
    CHECK(sub5.z == Complex<float>(5, -3));
    CHECK(sub5.w == Complex<float>(3, 4));

    t1 = Pixel32fcC4(Complex<float>(4, 5), Complex<float>(6, 7), Complex<float>(8, 9), Complex<float>(-3, -4));
    t2 = Pixel32fcC4(Complex<float>(5, 6), Complex<float>(7, -8), Complex<float>(9, -5), Complex<float>(3, 2));
    Pixel32fcC4 mul1 = t1 * t2;
    CHECK(mul1.x == Complex<float>(-10, 49));
    CHECK(mul1.y == Complex<float>(98, 1));
    CHECK(mul1.z == Complex<float>(117, 41));
    CHECK(mul1.w == Complex<float>(-1, -18));

    Pixel32fcC4 mul2 = 3 * t1;
    CHECK(mul2.x == Complex<float>(12, 15));
    CHECK(mul2.y == Complex<float>(18, 21));
    CHECK(mul2.z == Complex<float>(24, 27));
    CHECK(mul2.w == Complex<float>(-9, -12));

    Pixel32fcC4 mul3 = t1 * 4;
    CHECK(mul3.x == Complex<float>(16, 20));
    CHECK(mul3.y == Complex<float>(24, 28));
    CHECK(mul3.z == Complex<float>(32, 36));
    CHECK(mul3.w == Complex<float>(-12, -16));

    Pixel32fcC4 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == Complex<float>(-36, 160));
    CHECK(mul4.y == Complex<float>(-52, 336));
    CHECK(mul4.z == Complex<float>(-68, 576));
    CHECK(mul4.w == Complex<float>(-28, 96));

    mul4 *= 3;
    CHECK(mul4.x == Complex<float>(-108, 480));
    CHECK(mul4.y == Complex<float>(-156, 1008));
    CHECK(mul4.z == Complex<float>(-204, 1728));
    CHECK(mul4.w == Complex<float>(-84, 288));

    Pixel32fcC4 div1 = t1 / t2;
    CHECK(div1.x.real == Approx(0.819672108f).margin(0.001));
    CHECK(div1.x.imag == Approx(0.016393442f).margin(0.001));
    CHECK(div1.y.real == Approx(-0.123893805f).margin(0.001));
    CHECK(div1.y.imag == Approx(0.85840708f).margin(0.001));
    CHECK(div1.z.real == Approx(0.254716992f).margin(0.001));
    CHECK(div1.z.imag == Approx(1.141509414f).margin(0.001));
    CHECK(div1.w.real == Approx(-1.307692289f).margin(0.001));
    CHECK(div1.w.imag == Approx(-0.461538464f).margin(0.001));

    Pixel32fcC4 div2 = 3 / t1;
    CHECK(div2.x.real == Approx(0.292682916f).margin(0.001));
    CHECK(div2.x.imag == Approx(-0.365853667f).margin(0.001));
    CHECK(div2.y.real == Approx(0.211764708f).margin(0.001));
    CHECK(div2.y.imag == Approx(-0.247058824f).margin(0.001));
    CHECK(div2.z.real == Approx(0.165517241f).margin(0.001));
    CHECK(div2.z.imag == Approx(-0.186206892f).margin(0.001));
    CHECK(div2.w.real == Approx(-0.360000014f).margin(0.001));
    CHECK(div2.w.imag == Approx(0.479999989f).margin(0.001));

    Pixel32fcC4 div3 = t1 / 4;
    CHECK(div3.x.real == Approx(1).margin(0.001));
    CHECK(div3.x.imag == Approx(1.25).margin(0.001));
    CHECK(div3.y.real == Approx(1.5).margin(0.001));
    CHECK(div3.y.imag == Approx(1.75).margin(0.001));
    CHECK(div3.z.real == Approx(2).margin(0.001));
    CHECK(div3.z.imag == Approx(2.25).margin(0.001));
    CHECK(div3.w.real == Approx(-0.75).margin(0.001));
    CHECK(div3.w.imag == Approx(-1).margin(0.001));

    Pixel32fcC4 div4 = t2;
    div4 /= div3;
    CHECK(div4.x.real == Approx(4.878048897f).margin(0.001));
    CHECK(div4.x.imag == Approx(-0.097560972f).margin(0.001));
    CHECK(div4.y.real == Approx(-0.65882355f).margin(0.001));
    CHECK(div4.y.imag == Approx(-4.564705849f).margin(0.001));
    CHECK(div4.z.real == Approx(0.744827569f).margin(0.001));
    CHECK(div4.z.imag == Approx(-3.337930918f).margin(0.001));
    CHECK(div4.w.real == Approx(-2.720000029f).margin(0.001));
    CHECK(div4.w.imag == Approx(0.959999979f).margin(0.001));

    div4 /= 3;
    CHECK(div4.x.real == Approx(1.626016259f).margin(0.001));
    CHECK(div4.x.imag == Approx(-0.032520324f).margin(0.001));
    CHECK(div4.y.real == Approx(-0.21960786f).margin(0.001));
    CHECK(div4.y.imag == Approx(-1.521568656f).margin(0.001));
    CHECK(div4.z.real == Approx(0.248275861f).margin(0.001));
    CHECK(div4.z.imag == Approx(-1.112643719f).margin(0.001));
    CHECK(div4.w.real == Approx(-0.906666636f).margin(0.001));
    CHECK(div4.w.imag == Approx(0.319999993f).margin(0.001));

    Pixel32fcC4 div5 = t1;
    Pixel32fcC4 div6(9, 8, 7, 6);
    Pixel32fcC4 difv7 = div6 / div5;
    div5.DivInv(div6);
    CHECK(div5.x == difv7.x);
    CHECK(div5.y == difv7.y);
    CHECK(div5.z == difv7.z);
    CHECK(div5.w == difv7.w);

    Pixel32fcC4 minmax1(10, 20, -10, 50);
    Pixel32fcC4 minmax2(-20, 10, 40, 30);

    CHECK(Pixel32fcC4::Min(minmax1, minmax2) == Pixel32fcC4(-20, 10, -10, 30));
    CHECK(Pixel32fcC4::Min(minmax2, minmax1) == Pixel32fcC4(-20, 10, -10, 30));

    CHECK(Pixel32fcC4::Max(minmax1, minmax2) == Pixel32fcC4(10, 20, 40, 50));
    CHECK(Pixel32fcC4::Max(minmax2, minmax1) == Pixel32fcC4(10, 20, 40, 50));

    minmax1 = Pixel32fcC4(10, 20, -10, 50);
    minmax2 = Pixel32fcC4(-20, 10, 40, 30);
    CHECK(minmax1.Min(minmax2) == Pixel32fcC4(-20, 10, -10, 30));
    CHECK(minmax2.Min(minmax1) == Pixel32fcC4(-20, 10, -10, 30));

    minmax1 = Pixel32fcC4(10, 20, -10, 50);
    minmax2 = Pixel32fcC4(-20, 10, 40, 30);
    CHECK(minmax1.Max(minmax2) == Pixel32fcC4(10, 20, 40, 50));
    CHECK(minmax2.Max(minmax1) == Pixel32fcC4(10, 20, 40, 50));

    minmax1 = Pixel32fcC4(10, 20, -10, 50);
    minmax2 = Pixel32fcC4(-20, 10, 40, 30);
    CHECK(minmax2.Min().real == -20);
    CHECK(minmax1.Max().real == 50);
}

TEST_CASE("Pixel32fcC4_additionalMethods", "[Common]")
{
    Pixel32fcC4 roundA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f), Complex<float>(1.9f, -1.5f),
                       Complex<float>(-2.5f, numeric_limits<float>::maxExact() - 0.5f));
    Pixel32fcC4 round2A = Pixel32fcC4::Round(roundA);
    roundA.Round();
    CHECK(round2A == roundA);
    CHECK(roundA.x.real == 0.0f);
    CHECK(roundA.x.imag == 1.0f);
    CHECK(roundA.y.real == 1.0f);
    CHECK(roundA.y.imag == 2.0f);
    CHECK(roundA.z.real == 2.0f);
    CHECK(roundA.z.imag == -2.0f);
    CHECK(roundA.w.real == -3.0f);
    CHECK(roundA.w.imag == 16777216.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel32fcC4 floorA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f), Complex<float>(1.9f, -1.5f),
                       Complex<float>(-2.5f, numeric_limits<float>::maxExact() - 0.5f));
    Pixel32fcC4 floor2A = Pixel32fcC4::Floor(floorA);
    floorA.Floor();
    CHECK(floor2A == floorA);
    CHECK(floorA.x.real == 0.0f);
    CHECK(floorA.x.imag == 0.0f);
    CHECK(floorA.y.real == 0.0f);
    CHECK(floorA.y.imag == 1.0f);
    CHECK(floorA.z.real == 1.0f);
    CHECK(floorA.z.imag == -2.0f);
    CHECK(floorA.w.real == -3.0f);
    CHECK(floorA.w.imag == 16777216.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel32fcC4 ceilA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f), Complex<float>(1.9f, -1.5f),
                      Complex<float>(-2.5f, numeric_limits<float>::maxExact() - 0.5f));
    Pixel32fcC4 ceil2A = Pixel32fcC4::Ceil(ceilA);
    ceilA.Ceil();
    CHECK(ceil2A == ceilA);
    CHECK(ceilA.x.real == 1.0f);
    CHECK(ceilA.x.imag == 1.0f);
    CHECK(ceilA.y.real == 1.0f);
    CHECK(ceilA.y.imag == 2.0f);
    CHECK(ceilA.z.real == 2.0f);
    CHECK(ceilA.z.imag == -1.0f);
    CHECK(ceilA.w.real == -2.0f);
    CHECK(ceilA.w.imag == 16777216.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel32fcC4 zeroA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f), Complex<float>(1.9f, -1.5f),
                      Complex<float>(-2.5f, numeric_limits<float>::maxExact() - 0.5f));
    Pixel32fcC4 zero2A = Pixel32fcC4::RoundZero(zeroA);
    zeroA.RoundZero();
    CHECK(zero2A == zeroA);
    CHECK(zeroA.x.real == 0.0f);
    CHECK(zeroA.x.imag == 0.0f);
    CHECK(zeroA.y.real == 0.0f);
    CHECK(zeroA.y.imag == 1.0f);
    CHECK(zeroA.z.real == 1.0f);
    CHECK(zeroA.z.imag == -1.0f);
    CHECK(zeroA.w.real == -2.0f);
    CHECK(zeroA.w.imag == 16777216.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    Pixel32fcC4 nearestA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f), Complex<float>(1.9f, -1.5f),
                         Complex<float>(-2.5f, numeric_limits<float>::maxExact() - 0.5f));
    Pixel32fcC4 nearest2A = Pixel32fcC4::RoundNearest(nearestA);
    nearestA.RoundNearest();
    CHECK(nearest2A == nearestA);
    CHECK(nearestA.x.real == 0.0f);
    CHECK(nearestA.x.imag == 0.0f);
    CHECK(nearestA.y.real == 1.0f);
    CHECK(nearestA.y.imag == 2.0f);
    CHECK(nearestA.z.real == 2.0f);
    CHECK(nearestA.z.imag == -2.0f);
    CHECK(nearestA.w.real == -2.0f);
    CHECK(nearestA.w.imag == 16777216.0f); // this will always be exact integer as the -0.5f cannot be stored in float

    // Not vector does the value clampling but the complex type!
    Pixel32fcC4 clampToShort(float(numeric_limits<short>::max()) + 1, float(numeric_limits<short>::min()) - 1,
                             float(numeric_limits<short>::min()), float(numeric_limits<short>::max()));
    Pixel16scC4 clampShort(clampToShort);
    CHECK(clampShort.x.real == 32767);
    CHECK(clampShort.y.real == -32768);
    CHECK(clampShort.z.real == -32768);
    CHECK(clampShort.w.real == 32767);

    Pixel32fcC4 clampToInt(float(numeric_limits<int>::max()) + 1000, float(numeric_limits<int>::min()) - 1000,
                           float(numeric_limits<int>::min()), float(numeric_limits<int>::max()));

    Pixel32scC4 clampInt(clampToInt);
    CHECK(clampInt.x.real == 2147483520);
    CHECK(clampInt.y.real == -2147483648);
    CHECK(clampInt.z.real == -2147483648);
    CHECK(clampInt.w.real == 2147483520);
}