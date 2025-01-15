#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <common/defines.h>
#include <common/exception.h>
#include <common/half_fp16.h>
#include <common/image/pixelTypes.h>
#include <common/needSaturationClamp.h>
#include <common/numeric_limits.h>
#include <common/vector4.h>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

using namespace opp;
using namespace opp::image;
using namespace Catch;

TEST_CASE("Pixel32fC4A", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32fC4A) == 4 * sizeof(float));

    float arr[4] = {4, 5, 6};
    Pixel32fC4A t0(arr);
    CHECK(t0.x == 4);
    CHECK(t0.y == 5);
    CHECK(t0.z == 6);

    Pixel32fC4A t1(0, 1, 2);
    CHECK(t1.x == 0);
    CHECK(t1.y == 1);
    CHECK(t1.z == 2);

    Pixel32fC4A c(t1);
    CHECK(c.x == 0);
    CHECK(c.y == 1);
    CHECK(c.z == 2);
    CHECK(c == t1);

    Pixel32fC4A c2 = t1;
    CHECK(c2.x == 0);
    CHECK(c2.y == 1);
    CHECK(c2.z == 2);
    CHECK(c2 == t1);

    Pixel32fC4A t2(5);
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(t2.z == 5);
    CHECK(c2 != t2);

    Pixel32fC4A neg = -t1;
    CHECK(neg.x == 0);
    CHECK(neg.y == -1);
    CHECK(neg.z == -2);
    CHECK(t1 == -neg);

    Pixel32fC4A add1 = t1 + t2;
    CHECK(add1.x == 5);
    CHECK(add1.y == 6);
    CHECK(add1.z == 7);

    Pixel32fC4A add2 = 3 + t1;
    CHECK(add2.x == 3);
    CHECK(add2.y == 4);
    CHECK(add2.z == 5);

    Pixel32fC4A add3 = t1 + 4;
    CHECK(add3.x == 4);
    CHECK(add3.y == 5);
    CHECK(add3.z == 6);

    Pixel32fC4A add4 = t1;
    add4 += add3;
    CHECK(add4.x == 4);
    CHECK(add4.y == 6);
    CHECK(add4.z == 8);

    add4 += 3;
    CHECK(add4.x == 7);
    CHECK(add4.y == 9);
    CHECK(add4.z == 11);

    Pixel32fC4A sub1 = t1 - t2;
    CHECK(sub1.x == -5);
    CHECK(sub1.y == -4);
    CHECK(sub1.z == -3);

    Pixel32fC4A sub2 = 3 - t1;
    CHECK(sub2.x == 3);
    CHECK(sub2.y == 2);
    CHECK(sub2.z == 1);

    Pixel32fC4A sub3 = t1 - 4;
    CHECK(sub3.x == -4);
    CHECK(sub3.y == -3);
    CHECK(sub3.z == -2);

    Pixel32fC4A sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);
    CHECK(sub4.z == 4);

    sub4 -= 3;
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);
    CHECK(sub4.z == 1);

    t1               = Pixel32fC4A(4, 5, 6);
    t2               = Pixel32fC4A(6, 7, 8);
    Pixel32fC4A mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);
    CHECK(mul1.z == 48);

    Pixel32fC4A mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);
    CHECK(mul2.z == 18);

    Pixel32fC4A mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);
    CHECK(mul3.z == 24);

    Pixel32fC4A mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);
    CHECK(mul4.z == 144);

    mul4 *= 3;
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);
    CHECK(mul4.z == 432);

    Pixel32fC4A div1 = t1 / t2;
    CHECK(div1.x == Approx(0.667).margin(0.001));
    CHECK(div1.y == Approx(0.714).margin(0.001));
    CHECK(div1.z == Approx(0.75).margin(0.001));

    Pixel32fC4A div2 = 3 / t1;
    CHECK(div2.x == Approx(0.75).margin(0.001));
    CHECK(div2.y == Approx(0.6).margin(0.001));
    CHECK(div2.z == Approx(0.5).margin(0.001));

    Pixel32fC4A div3 = t1 / 4;
    CHECK(div3.x == Approx(1.0).margin(0.001));
    CHECK(div3.y == Approx(1.25).margin(0.001));
    CHECK(div3.z == Approx(1.5).margin(0.001));

    Pixel32fC4A div4 = t2;
    div4 /= div3;
    CHECK(div4.x == Approx(6.0).margin(0.001));
    CHECK(div4.y == Approx(5.6).margin(0.001));
    CHECK(div4.z == Approx(5.333).margin(0.001));

    div4 /= 3;
    CHECK(div4.x == Approx(2.0).margin(0.001));
    CHECK(div4.y == Approx(1.867).margin(0.001));
    CHECK(div4.z == Approx(1.778).margin(0.001));

    Pixel32fC4A minmax1(10, 20, -10);
    Pixel32fC4A minmax2(-20, 10, 40);

    CHECK((Pixel32fC4A::Min(minmax1, minmax2) == Pixel32fC4A(-20, 10, -10)));
    CHECK((Pixel32fC4A::Min(minmax2, minmax1) == Pixel32fC4A(-20, 10, -10)));

    minmax1 = Pixel32fC4A(10, 20, -10);
    minmax2 = Pixel32fC4A(-20, 10, 40);
    CHECK((Pixel32fC4A::Max(minmax1, minmax2) == Pixel32fC4A(10, 20, 40)));
    CHECK((Pixel32fC4A::Max(minmax2, minmax1) == Pixel32fC4A(10, 20, 40)));

    minmax1 = Pixel32fC4A(10, 20, -10);
    minmax2 = Pixel32fC4A(-20, 10, 40);
    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 20);
}

TEST_CASE("Pixel32fC4A_additionalMethods", "[Common]")
{
    Pixel32fC4A roundA(0.4f, 0.5f, 0.6f);
    Pixel32fC4A roundB(1.9f, -1.5f, -2.5f);
    Pixel32fC4A round2A = Pixel32fC4A::Round(roundA);
    Pixel32fC4A round2B = Pixel32fC4A::Round(roundB);
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

    Pixel32fC4A floorA(0.4f, 0.5f, 0.6f);
    Pixel32fC4A floorB(1.9f, -1.5f, -2.5f);
    Pixel32fC4A floor2A = Pixel32fC4A::Floor(floorA);
    Pixel32fC4A floor2B = Pixel32fC4A::Floor(floorB);
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

    Pixel32fC4A ceilA(0.4f, 0.5f, 0.6f);
    Pixel32fC4A ceilB(1.9f, -1.5f, -2.5f);
    Pixel32fC4A ceil2A = Pixel32fC4A::Ceil(ceilA);
    Pixel32fC4A ceil2B = Pixel32fC4A::Ceil(ceilB);
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

    Pixel32fC4A zeroA(0.4f, 0.5f, 0.6f);
    Pixel32fC4A zeroB(1.9f, -1.5f, -2.5f);
    Pixel32fC4A zero2A = Pixel32fC4A::RoundZero(zeroA);
    Pixel32fC4A zero2B = Pixel32fC4A::RoundZero(zeroB);
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

    Pixel32fC4A nearestA(0.4f, 0.5f, 0.6f);
    Pixel32fC4A nearestB(1.9f, -1.5f, -2.5f);
    Pixel32fC4A nearest2A = Pixel32fC4A::RoundNearest(nearestA);
    Pixel32fC4A nearest2B = Pixel32fC4A::RoundNearest(nearestB);
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

    Pixel32fC4A exp(2.4f, 12.5f, -14.6f);
    Pixel32fC4A exp2 = Pixel32fC4A::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == Approx(11.023176380641601).margin(0.00001));
    CHECK(exp.y == Approx(2.683372865208745e+05).margin(0.00001));
    CHECK(exp.z == Approx(4.563526367903994e-07).margin(0.00001));

    Pixel32fC4A ln(2.4f, 12.5f, 14.6f);
    Pixel32fC4A ln2 = Pixel32fC4A::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == Approx(0.875468737353900).margin(0.00001));
    CHECK(ln.y == Approx(2.525728644308256).margin(0.00001));
    CHECK(ln.z == Approx(2.681021528714291).margin(0.00001));

    Pixel32fC4A sqr(2.4f, 12.5f, -14.6f);
    Pixel32fC4A sqr2 = Pixel32fC4A::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == Approx(5.76).margin(0.00001));
    CHECK(sqr.y == Approx(156.25).margin(0.00001));
    CHECK(sqr.z == Approx(213.16).margin(0.00001));

    Pixel32fC4A sqrt(2.4f, 12.5f, 14.6f);
    Pixel32fC4A sqrt2 = Pixel32fC4A::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == Approx(1.549193338482967).margin(0.00001));
    CHECK(sqrt.y == Approx(3.535533905932738).margin(0.00001));
    CHECK(sqrt.z == Approx(3.820994634908560).margin(0.00001));

    Pixel32fC4A abs(-2.4f, 12.5f, -14.6f);
    Pixel32fC4A abs2 = Pixel32fC4A::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == 2.4f);
    CHECK(abs.y == 12.5f);
    CHECK(abs.z == 14.6f);

    Pixel32fC4A absdiffA(13.23592f, -40.24595f, -22.15017f);
    Pixel32fC4A absdiffB(45.75068f, 46.488853f, -34.238691f);
    Pixel32fC4A absdiff2 = Pixel32fC4A::AbsDiff(absdiffA, absdiffB);
    Pixel32fC4A absdiff3 = Pixel32fC4A::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == Approx(32.514758920888809).margin(0.00001));
    CHECK(absdiffA.y == Approx(86.734813019986703).margin(0.00001));
    CHECK(absdiffA.z == Approx(12.088513718950011).margin(0.00001));

    Pixel32fC4A clampByte(float(numeric_limits<byte>::max()) + 1, float(numeric_limits<byte>::min()) - 1,
                          float(numeric_limits<byte>::min()));
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 256);
    CHECK(clampByte.y == -1);
    CHECK(clampByte.z == 0);

    Pixel32fC4A clampShort(float(numeric_limits<short>::max()) + 1, float(numeric_limits<short>::min()) - 1,
                           float(numeric_limits<short>::min()));
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32768);
    CHECK(clampShort.y == -32769);
    CHECK(clampShort.z == -32768);

    Pixel32fC4A clampSByte(float(numeric_limits<sbyte>::max()) + 1, float(numeric_limits<sbyte>::min()) - 1,
                           float(numeric_limits<sbyte>::min()));
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 128);
    CHECK(clampSByte.y == -129);
    CHECK(clampSByte.z == -128);

    Pixel32fC4A clampUShort(float(numeric_limits<ushort>::max()) + 1, float(numeric_limits<ushort>::min()) - 1,
                            float(numeric_limits<ushort>::min()));
    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 65536);
    CHECK(clampUShort.y == -1);
    CHECK(clampUShort.z == 0);

    Pixel32fC4A clampInt(float(numeric_limits<int>::max()) + 1000, float(numeric_limits<int>::min()) - 1000,
                         float(numeric_limits<int>::min()));
    clampInt.template ClampToTargetType<int>();
    CHECK(clampInt.x == 2147484672.0f);
    CHECK(clampInt.y == -2147484672.0f);
    CHECK(clampInt.z == -2147483648);

    Pixel32fC4A clampUInt(float(numeric_limits<uint>::max()) + 1000, float(numeric_limits<uint>::min()) - 1000,
                          float(numeric_limits<uint>::min()));
    clampUInt.template ClampToTargetType<uint>();
    CHECK(clampUInt.x == 4294968320.0f);
    CHECK(clampUInt.y == -1000.0f);
    CHECK(clampUInt.z == 0);
}

TEST_CASE("Pixel32sC4A", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32sC4A) == 4 * sizeof(int));

    Pixel32sC4A t1(100, 200, 300);
    CHECK(t1.x == 100);
    CHECK(t1.y == 200);
    CHECK(t1.z == 300);

    Pixel32sC4A c(t1);
    CHECK(c.x == 100);
    CHECK(c.y == 200);
    CHECK(c.z == 300);
    CHECK(c == t1);

    Pixel32sC4A c2 = t1;
    CHECK(c2.x == 100);
    CHECK(c2.y == 200);
    CHECK(c2.z == 300);
    CHECK(c2 == t1);

    Pixel32sC4A t2(5);
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(t2.z == 5);
    CHECK(c2 != t2);

    Pixel32sC4A add1 = t1 + t2;
    CHECK(add1.x == 105);
    CHECK(add1.y == 205);
    CHECK(add1.z == 305);

    Pixel32sC4A add2 = 3 + t1;
    CHECK(add2.x == 103);
    CHECK(add2.y == 203);
    CHECK(add2.z == 303);

    Pixel32sC4A add3 = t1 + 4;
    CHECK(add3.x == 104);
    CHECK(add3.y == 204);
    CHECK(add3.z == 304);

    Pixel32sC4A add4 = t1;
    add4 += add3;
    CHECK(add4.x == 204);
    CHECK(add4.y == 404);
    CHECK(add4.z == 604);

    add4 += 3;
    CHECK(add4.x == 207);
    CHECK(add4.y == 407);
    CHECK(add4.z == 607);

    Pixel32sC4A sub1 = t1 - t2;
    CHECK(sub1.x == 95);
    CHECK(sub1.y == 195);
    CHECK(sub1.z == 295);

    Pixel32sC4A sub2 = 3 - t1;
    CHECK(sub2.x == -97);
    CHECK(sub2.y == -197);
    CHECK(sub2.z == -297);

    Pixel32sC4A sub3 = t1 - 4;
    CHECK(sub3.x == 96);
    CHECK(sub3.y == 196);
    CHECK(sub3.z == 296);

    Pixel32sC4A sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);
    CHECK(sub4.z == 4);

    sub4 -= 3;
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);
    CHECK(sub4.z == 1);

    t1               = Pixel32sC4A(4, 5, 6);
    t2               = Pixel32sC4A(6, 7, 8);
    Pixel32sC4A mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);
    CHECK(mul1.z == 48);

    Pixel32sC4A mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);
    CHECK(mul2.z == 18);

    Pixel32sC4A mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);
    CHECK(mul3.z == 24);

    Pixel32sC4A mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);
    CHECK(mul4.z == 144);

    mul4 *= 3;
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);
    CHECK(mul4.z == 432);

    t1               = Pixel32sC4A(1000, 2000, 3000);
    t2               = Pixel32sC4A(6, 7, 8);
    Pixel32sC4A div1 = t1 / t2;
    CHECK(div1.x == 166);
    CHECK(div1.y == 285);
    CHECK(div1.z == 375);

    Pixel32sC4A div2 = 30000 / t1;
    CHECK(div2.x == 30);
    CHECK(div2.y == 15);
    CHECK(div2.z == 10);

    Pixel32sC4A div3 = t1 / 4;
    CHECK(div3.x == 250);
    CHECK(div3.y == 500);
    CHECK(div3.z == 750);

    Pixel32sC4A div4 = t2 * 10000;
    div4 /= div3;
    CHECK(div4.x == 240);
    CHECK(div4.y == 140);
    CHECK(div4.z == 106);

    div4 /= 3;
    CHECK(div4.x == 80);
    CHECK(div4.y == 46);
    CHECK(div4.z == 35);

    Pixel32sC4A minmax1(10, 20, -10);
    Pixel32sC4A minmax2(-20, 10, 40);

    CHECK((Pixel32sC4A::Min(minmax1, minmax2) == Pixel32sC4A(-20, 10, -10)));
    CHECK((Pixel32sC4A::Min(minmax2, minmax1) == Pixel32sC4A(-20, 10, -10)));

    minmax1 = Pixel32sC4A(10, 20, -10);
    minmax2 = Pixel32sC4A(-20, 10, 40);
    CHECK((Pixel32sC4A::Max(minmax1, minmax2) == Pixel32sC4A(10, 20, 40)));
    CHECK((Pixel32sC4A::Max(minmax2, minmax1) == Pixel32sC4A(10, 20, 40)));

    minmax1 = Pixel32sC4A(10, 20, -10);
    minmax2 = Pixel32sC4A(-20, 10, 40);
    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 20);
}

TEST_CASE("Pixel32sC4A_additionalMethods", "[Common]")
{
    Pixel32sC4A exp(4, 5, 6);
    Pixel32sC4A exp2 = Pixel32sC4A::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == 54);
    CHECK(exp.y == 148);
    CHECK(exp.z == 403);

    Pixel32sC4A ln(4, 50, 600);
    Pixel32sC4A ln2 = Pixel32sC4A::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == 1);
    CHECK(ln.y == 3);
    CHECK(ln.z == 6);

    Pixel32sC4A sqr(4, 5, 6);
    Pixel32sC4A sqr2 = Pixel32sC4A::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == 16);
    CHECK(sqr.y == 25);
    CHECK(sqr.z == 36);

    Pixel32sC4A sqrt(4, 5, 6);
    Pixel32sC4A sqrt2 = Pixel32sC4A::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == 2);
    CHECK(sqrt.y == 2);
    CHECK(sqrt.z == 2);

    Pixel32sC4A abs(-2, 12, -14);
    Pixel32sC4A abs2 = Pixel32sC4A::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == 2);
    CHECK(abs.y == 12);
    CHECK(abs.z == 14);

    Pixel32sC4A clampByte(int(numeric_limits<byte>::max()) + 1, int(numeric_limits<byte>::min()) - 1,
                          int(numeric_limits<byte>::min()));
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 255);
    CHECK(clampByte.y == 0);
    CHECK(clampByte.z == 0);

    Pixel32sC4A clampShort(int(numeric_limits<short>::max()) + 1, int(numeric_limits<short>::min()) - 1,
                           int(numeric_limits<short>::min()));
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32767);
    CHECK(clampShort.y == -32768);
    CHECK(clampShort.z == -32768);

    Pixel32sC4A clampSByte(int(numeric_limits<sbyte>::max()) + 1, int(numeric_limits<sbyte>::min()) - 1,
                           int(numeric_limits<sbyte>::min()));
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 127);
    CHECK(clampSByte.y == -128);
    CHECK(clampSByte.z == -128);

    Pixel32sC4A clampUShort(int(numeric_limits<ushort>::max()) + 1, int(numeric_limits<ushort>::min()) - 1,
                            int(numeric_limits<ushort>::min()));
    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 65535);
    CHECK(clampUShort.y == 0);
    CHECK(clampUShort.z == 0);

    Pixel32sC4A clampInt(numeric_limits<int>::max(), numeric_limits<int>::min(), numeric_limits<int>::min());
    clampInt.template ClampToTargetType<int>();
    CHECK(clampInt.x == 2147483647);
    CHECK(clampInt.y == -2147483648);
    CHECK(clampInt.z == -2147483648);

    Pixel32sC4A lshift(1024, 2048, 4096);
    Pixel32sC4A lshift2 = Pixel32sC4A::LShift(lshift, 2);
    lshift.LShift(2);
    CHECK(lshift == lshift2);
    CHECK(lshift.x == 4096);
    CHECK(lshift.y == 8192);
    CHECK(lshift.z == 16384);

    Pixel32sC4A rshift(1024, 2048, 4096);
    Pixel32sC4A rshift2 = Pixel32sC4A::RShift(rshift, 2);
    rshift.RShift(2);
    CHECK(rshift == rshift2);
    CHECK(rshift.x == 256);
    CHECK(rshift.y == 512);
    CHECK(rshift.z == 1024);

    Pixel32sC4A and_(1023, 2047, 4095);
    Pixel32sC4A and_B(512, 1024, 2048);
    Pixel32sC4A and_2 = Pixel32sC4A::And(and_, and_B);
    and_.And(and_B);
    CHECK(and_ == and_2);
    CHECK(and_.x == 512);
    CHECK(and_.y == 1024);
    CHECK(and_.z == 2048);

    Pixel32sC4A or_(1023, 2047, 4095);
    Pixel32sC4A or_B(512, 1024, 2048);
    Pixel32sC4A or_2 = Pixel32sC4A::Or(or_, or_B);
    or_.Or(or_B);
    CHECK(or_ == or_2);
    CHECK(or_.x == 1023);
    CHECK(or_.y == 2047);
    CHECK(or_.z == 4095);

    Pixel32sC4A xor_(1023, 2047, 4095);
    Pixel32sC4A xor_B(512, 1024, 2048);
    Pixel32sC4A xor_2 = Pixel32sC4A::Xor(xor_, xor_B);
    xor_.Xor(xor_B);
    CHECK(xor_ == xor_2);
    CHECK(xor_.x == 511);
    CHECK(xor_.y == 1023);
    CHECK(xor_.z == 2047);

    Pixel32sC4A not_(1023, 2047, 4095);
    Pixel32sC4A not_2 = Pixel32sC4A::Not(not_);
    not_.Not();
    CHECK(not_ == not_2);
    CHECK(not_.x == -1024);
    CHECK(not_.y == -2048);
    CHECK(not_.z == -4096);
}

TEST_CASE("Pixel32fC4A_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel32fC4A *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel32fC4A(float(i));
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

    std::vector<Pixel32fC4A> buffer2(count);
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

TEST_CASE("Pixel32sC4A_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel32sC4A *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel32sC4A(int(i));
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

    std::vector<Pixel32sC4A> buffer2(count);
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

TEST_CASE("Pixel8uC4A_streams", "[Common]")
{
    std::string str = "3 4 5";
    std::stringstream ss(str);

    Pixel8uC4A pix;
    ss >> pix;

    CHECK(pix.x == 3);
    CHECK(pix.y == 4);
    CHECK(pix.z == 5);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3, 4, 5, A)");
}

TEST_CASE("Pixel32sC4A_streams", "[Common]")
{
    std::string str = "3 4 5";
    std::stringstream ss(str);

    Pixel32sC4A pix;
    ss >> pix;

    CHECK(pix.x == 3);
    CHECK(pix.y == 4);
    CHECK(pix.z == 5);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3, 4, 5, A)");
}

TEST_CASE("Pixel32fC4A_streams", "[Common]")
{
    std::string str = "3.14 2.7 8.9";
    std::stringstream ss(str);

    Pixel32fC4A pix;
    ss >> pix;

    CHECK(pix.x == 3.14f);
    CHECK(pix.y == 2.7f);
    CHECK(pix.z == 8.9f);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3.14, 2.7, 8.9, A)");
}

TEST_CASE("Axis4D", "[Common]")
{
    Pixel32sC4A pix(2, 3, 4);

    CHECK(pix[Axis4D::X] == 2);
    CHECK(pix[Axis4D::Y] == 3);
    CHECK(pix[Axis4D::Z] == 4);

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

TEST_CASE("Pixel16fC4A", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel16fC4A) == 4 * sizeof(HalfFp16));

    HalfFp16 arr[4] = {HalfFp16(4), HalfFp16(5), HalfFp16(6)};
    Pixel16fC4A t0(arr);
    CHECK(t0.x == 4);
    CHECK(t0.y == 5);
    CHECK(t0.z == 6);

    Pixel16fC4A t1(HalfFp16(0), HalfFp16(1), HalfFp16(2));
    CHECK(t1.x == 0);
    CHECK(t1.y == 1);
    CHECK(t1.z == 2);

    Pixel16fC4A c(t1);
    CHECK(c.x == 0);
    CHECK(c.y == 1);
    CHECK(c.z == 2);
    CHECK(c == t1);

    Pixel16fC4A c2 = t1;
    CHECK(c2.x == 0);
    CHECK(c2.y == 1);
    CHECK(c2.z == 2);
    CHECK(c2 == t1);

    Pixel16fC4A t2(HalfFp16(5));
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(t2.z == 5);
    CHECK(c2 != t2);

    Pixel16fC4A add1 = t1 + t2;
    CHECK(add1.x == 5);
    CHECK(add1.y == 6);
    CHECK(add1.z == 7);

    Pixel16fC4A add2 = 3 + t1;
    CHECK(add2.x == 3);
    CHECK(add2.y == 4);
    CHECK(add2.z == 5);

    Pixel16fC4A add3 = t1 + 4;
    CHECK(add3.x == 4);
    CHECK(add3.y == 5);
    CHECK(add3.z == 6);

    Pixel16fC4A add4 = t1;
    add4 += add3;
    CHECK(add4.x == 4);
    CHECK(add4.y == 6);
    CHECK(add4.z == 8);

    add4 += HalfFp16(3);
    CHECK(add4.x == 7);
    CHECK(add4.y == 9);
    CHECK(add4.z == 11);

    Pixel16fC4A sub1 = t1 - t2;
    CHECK(sub1.x == -5);
    CHECK(sub1.y == -4);
    CHECK(sub1.z == -3);

    Pixel16fC4A sub2 = 3 - t1;
    CHECK(sub2.x == 3);
    CHECK(sub2.y == 2);
    CHECK(sub2.z == 1);

    Pixel16fC4A sub3 = t1 - 4;
    CHECK(sub3.x == -4);
    CHECK(sub3.y == -3);
    CHECK(sub3.z == -2);

    Pixel16fC4A sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);
    CHECK(sub4.z == 4);

    sub4 -= HalfFp16(3);
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);
    CHECK(sub4.z == 1);

    t1               = Pixel16fC4A(HalfFp16(4), HalfFp16(5), HalfFp16(6));
    t2               = Pixel16fC4A(HalfFp16(6), HalfFp16(7), HalfFp16(8));
    Pixel16fC4A mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);
    CHECK(mul1.z == 48);

    Pixel16fC4A mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);
    CHECK(mul2.z == 18);

    Pixel16fC4A mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);
    CHECK(mul3.z == 24);

    Pixel16fC4A mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);
    CHECK(mul4.z == 144);

    mul4 *= HalfFp16(3);
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);
    CHECK(mul4.z == 432);

    Pixel16fC4A div1 = t1 / t2;
    CHECK(div1.x == Approx(0.667).margin(0.001));
    CHECK(div1.y == Approx(0.714).margin(0.001));
    CHECK(div1.z == Approx(0.75).margin(0.001));

    Pixel16fC4A div2 = 3 / t1;
    CHECK(div2.x == Approx(0.75).margin(0.001));
    CHECK(div2.y == Approx(0.6).margin(0.001));
    CHECK(div2.z == Approx(0.5).margin(0.001));

    Pixel16fC4A div3 = t1 / 4;
    CHECK(div3.x == Approx(1.0).margin(0.001));
    CHECK(div3.y == Approx(1.25).margin(0.001));
    CHECK(div3.z == Approx(1.5).margin(0.001));

    Pixel16fC4A div4 = t2;
    div4 /= div3;
    CHECK(div4.x == Approx(6.0).margin(0.001));
    CHECK(div4.y == Approx(5.6).margin(0.01));
    CHECK(div4.z == Approx(5.333).margin(0.001));

    div4 /= HalfFp16(3);
    CHECK(div4.x == Approx(2.0).margin(0.001));
    CHECK(div4.y == Approx(1.867).margin(0.001));
    CHECK(div4.z == Approx(1.778).margin(0.001));

    Pixel16fC4A minmax1(HalfFp16(10), HalfFp16(20), HalfFp16(-10));
    Pixel16fC4A minmax2(HalfFp16(-20), HalfFp16(10), HalfFp16(40));

    CHECK(Pixel16fC4A::Min(minmax1, minmax2) == Pixel16fC4A(HalfFp16(-20), HalfFp16(10), HalfFp16(-10)));
    CHECK(Pixel16fC4A::Min(minmax2, minmax1) == Pixel16fC4A(HalfFp16(-20), HalfFp16(10), HalfFp16(-10)));

    CHECK(Pixel16fC4A::Max(minmax1, minmax2) == Pixel16fC4A(HalfFp16(10), HalfFp16(20), HalfFp16(40)));
    CHECK(Pixel16fC4A::Max(minmax2, minmax1) == Pixel16fC4A(HalfFp16(10), HalfFp16(20), HalfFp16(40)));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 20);
}

TEST_CASE("Pixel16fC4A_additionalMethods", "[Common]")
{
    Pixel16fC4A roundA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f));
    Pixel16fC4A roundB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f));
    Pixel16fC4A round2A = Pixel16fC4A::Round(roundA);
    Pixel16fC4A round2B = Pixel16fC4A::Round(roundB);
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

    Pixel16fC4A floorA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f));
    Pixel16fC4A floorB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f));
    Pixel16fC4A floor2A = Pixel16fC4A::Floor(floorA);
    Pixel16fC4A floor2B = Pixel16fC4A::Floor(floorB);
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

    Pixel16fC4A ceilA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f));
    Pixel16fC4A ceilB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f));
    Pixel16fC4A ceil2A = Pixel16fC4A::Ceil(ceilA);
    Pixel16fC4A ceil2B = Pixel16fC4A::Ceil(ceilB);
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

    Pixel16fC4A zeroA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f));
    Pixel16fC4A zeroB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f));
    Pixel16fC4A zero2A = Pixel16fC4A::RoundZero(zeroA);
    Pixel16fC4A zero2B = Pixel16fC4A::RoundZero(zeroB);
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

    Pixel16fC4A nearestA(HalfFp16(0.4f), HalfFp16(0.5f), HalfFp16(0.6f));
    Pixel16fC4A nearestB(HalfFp16(1.9f), HalfFp16(-1.5f), HalfFp16(-2.5f));
    Pixel16fC4A nearest2A = Pixel16fC4A::RoundNearest(nearestA);
    Pixel16fC4A nearest2B = Pixel16fC4A::RoundNearest(nearestB);
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

    Pixel16fC4A exp(HalfFp16(2.4f), HalfFp16(12.5f), HalfFp16(-14.6f));
    Pixel16fC4A exp2 = Pixel16fC4A::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == Approx(11.023176380641601).margin(0.01));
    CHECK(isinf(exp.y));
    CHECK(exp.z == Approx(4.563526367903994e-07).margin(0.01));

    Pixel16fC4A ln(HalfFp16(2.4f), HalfFp16(12.5f), HalfFp16(14.6f));
    Pixel16fC4A ln2 = Pixel16fC4A::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == Approx(0.875468737353900).margin(0.01));
    CHECK(ln.y == Approx(2.525728644308256).margin(0.01));
    CHECK(ln.z == Approx(2.681021528714291).margin(0.01));

    Pixel16fC4A sqr(HalfFp16(2.4f), HalfFp16(12.5f), HalfFp16(-14.6f));
    Pixel16fC4A sqr2 = Pixel16fC4A::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == Approx(5.76).margin(0.01));
    CHECK(sqr.y == Approx(156.25).margin(0.01));
    CHECK(sqr.z == Approx(213.16).margin(0.1));

    Pixel16fC4A sqrt(HalfFp16(2.4f), HalfFp16(12.5f), HalfFp16(14.6f));
    Pixel16fC4A sqrt2 = Pixel16fC4A::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == Approx(1.549193338482967).margin(0.01));
    CHECK(sqrt.y == Approx(3.535533905932738).margin(0.01));
    CHECK(sqrt.z == Approx(3.820994634908560).margin(0.01));

    Pixel16fC4A abs(HalfFp16(-2.4f), HalfFp16(12.5f), HalfFp16(-14.6f));
    Pixel16fC4A abs2 = Pixel16fC4A::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == HalfFp16(2.4f));
    CHECK(abs.y == HalfFp16(12.5f));
    CHECK(abs.z == HalfFp16(14.6f));

    Pixel16fC4A absdiffA(HalfFp16(13.23592f), HalfFp16(-40.24595f), HalfFp16(-22.15017f));
    Pixel16fC4A absdiffB(HalfFp16(45.75068f), HalfFp16(46.488853f), HalfFp16(-34.238691f));
    Pixel16fC4A absdiff2 = Pixel16fC4A::AbsDiff(absdiffA, absdiffB);
    Pixel16fC4A absdiff3 = Pixel16fC4A::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == Approx(32.514758920888809).margin(0.02));
    CHECK(absdiffA.y == Approx(86.734813019986703).margin(0.02));
    CHECK(absdiffA.z == Approx(12.088513718950011).margin(0.02));

    Pixel16fC4A clampByte(HalfFp16(float(numeric_limits<byte>::max()) + 1),
                          HalfFp16(float(numeric_limits<byte>::min()) - 1),
                          HalfFp16(float(numeric_limits<byte>::min())));
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 256);
    CHECK(clampByte.y == -1);
    CHECK(clampByte.z == 0);

    Pixel16fC4A clampShort(HalfFp16(float(numeric_limits<short>::max()) + 1),
                           HalfFp16(float(numeric_limits<short>::min()) - 1),
                           HalfFp16(float(numeric_limits<short>::min())));
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32768);
    CHECK(clampShort.y == -32768);
    CHECK(clampShort.z == -32768);

    Pixel16fC4A clampSByte(HalfFp16(float(numeric_limits<sbyte>::max()) + 1),
                           HalfFp16(float(numeric_limits<sbyte>::min()) - 1),
                           HalfFp16(float(numeric_limits<sbyte>::min())));
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 128);
    CHECK(clampSByte.y == -129);
    CHECK(clampSByte.z == -128);

    Pixel16fC4A clampUShort(HalfFp16(0.0f), HalfFp16(float(numeric_limits<ushort>::min()) - 1),
                            HalfFp16(float(numeric_limits<ushort>::min())));

    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 0);
    CHECK(clampUShort.y == -1);
    CHECK(clampUShort.z == 0);

    CHECK_FALSE(need_saturation_clamp_v<HalfFp16, int>);

    CHECK_FALSE(need_saturation_clamp_v<HalfFp16, uint>);
    Pixel16fC4A clampUInt(HalfFp16(0), HalfFp16(float(numeric_limits<uint>::min()) - 1000),
                          HalfFp16(float(numeric_limits<uint>::min())));

    clampUInt.template ClampToTargetType<uint>();
    CHECK(clampUInt.x == 0);
    CHECK(clampUInt.y == -1000);
    CHECK(clampUInt.z == 0);

    Pixel32fC4A clampFloatToFp16(float(numeric_limits<int>::min()) - 1000.0f, //
                                 float(numeric_limits<int>::max()) + 1000.0f, //
                                 float(numeric_limits<short>::min()));

    clampFloatToFp16.template ClampToTargetType<HalfFp16>();
    CHECK(clampFloatToFp16.x == -2147484672.0f);
    CHECK(clampFloatToFp16.y == 2147484672.0f);
    CHECK(clampFloatToFp16.z == -32768);

    Pixel16fC4A fromFromFloat(clampFloatToFp16);
    CHECK(fromFromFloat.x == -INFINITY);
    CHECK(fromFromFloat.y == INFINITY);
    CHECK(fromFromFloat.z == -32768);
}