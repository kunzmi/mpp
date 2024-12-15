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

    Pixel32fC4 l(4, 6, -7, -2);
    CHECK(l.MagnitudeSqr() == 105);
    CHECK(l.Magnitude() == Approx(std::sqrt(105)));

    Pixel32fC4 minmax1(10, 20, -10, 50);
    Pixel32fC4 minmax2(-20, 10, 40, 30);

    CHECK(Pixel32fC4::Min(minmax1, minmax2) == Pixel32fC4(-20, 10, -10, 30));
    CHECK(Pixel32fC4::Min(minmax2, minmax1) == Pixel32fC4(-20, 10, -10, 30));

    CHECK(Pixel32fC4::Max(minmax1, minmax2) == Pixel32fC4(10, 20, 40, 50));
    CHECK(Pixel32fC4::Max(minmax2, minmax1) == Pixel32fC4(10, 20, 40, 50));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 50);
}

TEST_CASE("Pixel32fC4_additionalMethods", "[Common]")
{
    CHECK(Pixel32fC4::Dot(Pixel32fC4(4, 5, 6, 7), Pixel32fC4(6, 7, 8, 9)) == 170.0f);
    CHECK(Pixel32fC4(4, 5, 6, 7).Dot(Pixel32fC4(6, 7, 8, 9)) == 170.0f);

    Pixel32fC4 norm(4, 5, 6, 7);
    norm.Normalize();
    CHECK(norm.Magnitude() == 1);
    CHECK(norm.x == Approx(0.3563).margin(0.001));
    CHECK(norm.y == Approx(0.4454).margin(0.001));
    CHECK(norm.z == Approx(0.5345).margin(0.001));
    CHECK(norm.w == Approx(0.6236).margin(0.001));

    CHECK((Pixel32fC4::Normalize(Pixel32fC4(4, 5, 6, 7)) == norm));

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
    CHECK(clampInt.x == 2147483647);
    CHECK(clampInt.y == -2147483648);
    CHECK(clampInt.z == -2147483648);
    CHECK(clampInt.w == 2147483647);

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

    Pixel32sC4 l(4, 6, -7, -2);
    CHECK(l.MagnitudeSqr() == 105);
    CHECK(l.Magnitude() == Approx(std::sqrt(105)));

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
    Pixel32sC4 norm(4, 5, 6, 7);
    CHECK(norm.Magnitude() == Approx(std::sqrt(126.0)).margin(0.00001));
    CHECK(norm.MagnitudeSqr() == 126);

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

    Pixel32sC4 absdiffA(13, -40, -22, 4);
    Pixel32sC4 absdiffB(45, 46, -34, -47);
    Pixel32sC4 absdiff2 = Pixel32sC4::AbsDiff(absdiffA, absdiffB);
    Pixel32sC4 absdiff3 = Pixel32sC4::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == 32);
    CHECK(absdiffA.y == 86);
    CHECK(absdiffA.z == 12);
    CHECK(absdiffA.w == 51);

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