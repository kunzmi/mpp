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
#include <common/vector2.h>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;

TEST_CASE("Pixel32fC2", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32fC2) == 2 * sizeof(float));

    float arr[2] = {4, 5};
    Pixel32fC2 t0(arr);
    CHECK(t0.x == 4);
    CHECK(t0.y == 5);

    Pixel32fC2 t1(0, 1);
    CHECK(t1.x == 0);
    CHECK(t1.y == 1);

    Pixel32fC2 c(t1);
    CHECK(c.x == 0);
    CHECK(c.y == 1);
    CHECK(c == t1);

    Pixel32fC2 c2 = t1;
    CHECK(c2.x == 0);
    CHECK(c2.y == 1);
    CHECK(c2 == t1);

    Pixel32fC2 t2(5.0f);
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(c2 != t2);

    Pixel32fC2 neg = -t1;
    CHECK(neg.x == 0);
    CHECK(neg.y == -1);
    CHECK(t1 == -neg);

    Pixel32fC2 add1 = t1 + t2;
    CHECK(add1.x == 5);
    CHECK(add1.y == 6);

    Pixel32fC2 add2 = 3 + t1;
    CHECK(add2.x == 3);
    CHECK(add2.y == 4);

    Pixel32fC2 add3 = t1 + 4;
    CHECK(add3.x == 4);
    CHECK(add3.y == 5);

    Pixel32fC2 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 4);
    CHECK(add4.y == 6);

    add4 += 3;
    CHECK(add4.x == 7);
    CHECK(add4.y == 9);

    Pixel32fC2 sub1 = t1 - t2;
    CHECK(sub1.x == -5);
    CHECK(sub1.y == -4);

    Pixel32fC2 sub2 = 3 - t1;
    CHECK(sub2.x == 3);
    CHECK(sub2.y == 2);

    Pixel32fC2 sub3 = t1 - 4;
    CHECK(sub3.x == -4);
    CHECK(sub3.y == -3);

    Pixel32fC2 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);

    sub4 -= 3;
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);

    t1              = Pixel32fC2(4, 5);
    t2              = Pixel32fC2(6, 7);
    Pixel32fC2 mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);

    Pixel32fC2 mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);

    Pixel32fC2 mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);

    Pixel32fC2 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);

    mul4 *= 3;
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);

    Pixel32fC2 div1 = t1 / t2;
    CHECK(div1.x == Approx(0.667).margin(0.001));
    CHECK(div1.y == Approx(0.714).margin(0.001));

    Pixel32fC2 div2 = 3 / t1;
    CHECK(div2.x == Approx(0.75).margin(0.001));
    CHECK(div2.y == Approx(0.6).margin(0.001));

    Pixel32fC2 div3 = t1 / 4;
    CHECK(div3.x == Approx(1.0).margin(0.001));
    CHECK(div3.y == Approx(1.25).margin(0.001));

    Pixel32fC2 div4 = t2;
    div4 /= div3;
    CHECK(div4.x == Approx(6.0).margin(0.001));
    CHECK(div4.y == Approx(5.6).margin(0.001));

    div4 /= 3;
    CHECK(div4.x == Approx(2.0).margin(0.001));
    CHECK(div4.y == Approx(1.867).margin(0.001));

    Pixel32fC2 minmax1(10, 20);
    Pixel32fC2 minmax2(-20, 10);

    CHECK(Pixel32fC2::Min(minmax1, minmax2) == Pixel32fC2(-20, 10));
    CHECK(Pixel32fC2::Min(minmax2, minmax1) == Pixel32fC2(-20, 10));

    CHECK(Pixel32fC2::Max(minmax1, minmax2) == Pixel32fC2(10, 20));
    CHECK(Pixel32fC2::Max(minmax2, minmax1) == Pixel32fC2(10, 20));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 20);
}

TEST_CASE("Pixel32fC2_additionalMethods", "[Common]")
{
    Pixel32fC2 roundA(0.4f, 0.5f);
    Pixel32fC2 roundB(1.9f, -1.5f);
    Pixel32fC2 round2A = Pixel32fC2::Round(roundA);
    Pixel32fC2 round2B = Pixel32fC2::Round(roundB);
    roundA.Round();
    roundB.Round();
    CHECK(round2A == roundA);
    CHECK(round2B == roundB);
    CHECK(roundA.x == 0.0f);
    CHECK(roundA.y == 1.0f);
    CHECK(roundB.x == 2.0f);
    CHECK(roundB.y == -2.0f);

    Pixel32fC2 floorA(0.4f, 0.5f);
    Pixel32fC2 floorB(1.9f, -1.5f);
    Pixel32fC2 floor2A = Pixel32fC2::Floor(floorA);
    Pixel32fC2 floor2B = Pixel32fC2::Floor(floorB);
    floorA.Floor();
    floorB.Floor();
    CHECK(floor2A == floorA);
    CHECK(floor2B == floorB);
    CHECK(floorA.x == 0.0f);
    CHECK(floorA.y == 0.0f);
    CHECK(floorB.x == 1.0f);
    CHECK(floorB.y == -2.0f);

    Pixel32fC2 ceilA(0.4f, 0.5f);
    Pixel32fC2 ceilB(1.9f, -1.5f);
    Pixel32fC2 ceil2A = Pixel32fC2::Ceil(ceilA);
    Pixel32fC2 ceil2B = Pixel32fC2::Ceil(ceilB);
    ceilA.Ceil();
    ceilB.Ceil();
    CHECK(ceil2A == ceilA);
    CHECK(ceil2B == ceilB);
    CHECK(ceilA.x == 1.0f);
    CHECK(ceilA.y == 1.0f);
    CHECK(ceilB.x == 2.0f);
    CHECK(ceilB.y == -1.0f);

    Pixel32fC2 zeroA(0.4f, 0.5f);
    Pixel32fC2 zeroB(1.9f, -1.5f);
    Pixel32fC2 zero2A = Pixel32fC2::RoundZero(zeroA);
    Pixel32fC2 zero2B = Pixel32fC2::RoundZero(zeroB);
    zeroA.RoundZero();
    zeroB.RoundZero();
    CHECK(zero2A == zeroA);
    CHECK(zero2B == zeroB);
    CHECK(zeroA.x == 0.0f);
    CHECK(zeroA.y == 0.0f);
    CHECK(zeroB.x == 1.0f);
    CHECK(zeroB.y == -1.0f);

    Pixel32fC2 nearestA(0.4f, 0.5f);
    Pixel32fC2 nearestB(1.9f, -1.5f);
    Pixel32fC2 nearest2A = Pixel32fC2::RoundNearest(nearestA);
    Pixel32fC2 nearest2B = Pixel32fC2::RoundNearest(nearestB);
    nearestA.RoundNearest();
    nearestB.RoundNearest();
    CHECK(nearest2A == nearestA);
    CHECK(nearest2B == nearestB);
    CHECK(nearestA.x == 0.0f);
    CHECK(nearestA.y == 0.0f);
    CHECK(nearestB.x == 2.0f);
    CHECK(nearestB.y == -2.0f);

    Pixel32fC2 exp(2.4f, 12.5f);
    Pixel32fC2 exp2 = Pixel32fC2::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == Approx(11.023176380641601).margin(0.00001));
    CHECK(exp.y == Approx(2.683372865208745e+05).margin(0.00001));

    Pixel32fC2 ln(2.4f, 12.5f);
    Pixel32fC2 ln2 = Pixel32fC2::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == Approx(0.875468737353900).margin(0.00001));
    CHECK(ln.y == Approx(2.525728644308256).margin(0.00001));

    Pixel32fC2 sqr(2.4f, 12.5f);
    Pixel32fC2 sqr2 = Pixel32fC2::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == Approx(5.76).margin(0.00001));
    CHECK(sqr.y == Approx(156.25).margin(0.00001));

    Pixel32fC2 sqrt(2.4f, 12.5f);
    Pixel32fC2 sqrt2 = Pixel32fC2::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == Approx(1.549193338482967).margin(0.00001));
    CHECK(sqrt.y == Approx(3.535533905932738).margin(0.00001));

    Pixel32fC2 abs(-2.4f, 12.5f);
    Pixel32fC2 abs2 = Pixel32fC2::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == 2.4f);
    CHECK(abs.y == 12.5f);

    Pixel32fC2 absdiffA(13.23592f, -40.24595f);
    Pixel32fC2 absdiffB(45.75068f, 46.488853f);
    Pixel32fC2 absdiff2 = Pixel32fC2::AbsDiff(absdiffA, absdiffB);
    Pixel32fC2 absdiff3 = Pixel32fC2::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == Approx(32.514758920888809).margin(0.00001));
    CHECK(absdiffA.y == Approx(86.734813019986703).margin(0.00001));

    Pixel32fC2 clampByte(float(numeric_limits<byte>::max()) + 1, float(numeric_limits<byte>::min()) - 1);
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 256);
    CHECK(clampByte.y == -1);

    Pixel32fC2 clampShort(float(numeric_limits<short>::max()) + 1, float(numeric_limits<short>::min()) - 1);
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32768);
    CHECK(clampShort.y == -32769);

    Pixel32fC2 clampSByte(float(numeric_limits<sbyte>::max()) + 1, float(numeric_limits<sbyte>::min()) - 1);
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 128);
    CHECK(clampSByte.y == -129);

    Pixel32fC2 clampUShort(float(numeric_limits<ushort>::max()) + 1, float(numeric_limits<ushort>::min()) - 1);
    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 65536);
    CHECK(clampUShort.y == -1);

    Pixel32fC2 clampInt(float(numeric_limits<int>::max()) + 1000, float(numeric_limits<int>::min()) - 1000);
    clampInt.template ClampToTargetType<int>();
    CHECK(clampInt.x == 2147484672.0f);
    CHECK(clampInt.y == -2147484672.0f);

    Pixel32fC2 clampUInt(float(numeric_limits<uint>::max()) + 1000, float(numeric_limits<uint>::min()) - 1000);
    clampUInt.template ClampToTargetType<uint>();
    CHECK(clampUInt.x == 4294968320.0f);
    CHECK(clampUInt.y == -1000.0f);
}

TEST_CASE("Pixel32sC2", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32sC2) == 2 * sizeof(int));

    Pixel32sC2 t1(100, 200);
    CHECK(t1.x == 100);
    CHECK(t1.y == 200);

    Pixel32sC2 c(t1);
    CHECK(c.x == 100);
    CHECK(c.y == 200);
    CHECK(c == t1);

    Pixel32sC2 c2 = t1;
    CHECK(c2.x == 100);
    CHECK(c2.y == 200);
    CHECK(c2 == t1);

    Pixel32sC2 t2(5);
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(c2 != t2);

    Pixel32sC2 add1 = t1 + t2;
    CHECK(add1.x == 105);
    CHECK(add1.y == 205);

    Pixel32sC2 add2 = 3 + t1;
    CHECK(add2.x == 103);
    CHECK(add2.y == 203);

    Pixel32sC2 add3 = t1 + 4;
    CHECK(add3.x == 104);
    CHECK(add3.y == 204);

    Pixel32sC2 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 204);
    CHECK(add4.y == 404);

    add4 += 3;
    CHECK(add4.x == 207);
    CHECK(add4.y == 407);

    Pixel32sC2 sub1 = t1 - t2;
    CHECK(sub1.x == 95);
    CHECK(sub1.y == 195);

    Pixel32sC2 sub2 = 3 - t1;
    CHECK(sub2.x == -97);
    CHECK(sub2.y == -197);

    Pixel32sC2 sub3 = t1 - 4;
    CHECK(sub3.x == 96);
    CHECK(sub3.y == 196);

    Pixel32sC2 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);

    sub4 -= 3;
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);

    t1              = Pixel32sC2(4, 5);
    t2              = Pixel32sC2(6, 7);
    Pixel32sC2 mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);

    Pixel32sC2 mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);

    Pixel32sC2 mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);

    Pixel32sC2 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);

    mul4 *= 3;
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);

    t1              = Pixel32sC2(1000, 2000);
    t2              = Pixel32sC2(6, 7);
    Pixel32sC2 div1 = t1 / t2;
    CHECK(div1.x == 166);
    CHECK(div1.y == 285);

    Pixel32sC2 div2 = 30000 / t1;
    CHECK(div2.x == 30);
    CHECK(div2.y == 15);

    Pixel32sC2 div3 = t1 / 4;
    CHECK(div3.x == 250);
    CHECK(div3.y == 500);

    Pixel32sC2 div4 = t2 * 10000;
    div4 /= div3;
    CHECK(div4.x == 240);
    CHECK(div4.y == 140);

    div4 /= 3;
    CHECK(div4.x == 80);
    CHECK(div4.y == 46);

    Pixel32sC2 minmax1(10, 20);
    Pixel32sC2 minmax2(-20, 10);

    CHECK(Pixel32sC2::Min(minmax1, minmax2) == Pixel32sC2(-20, 10));
    CHECK(Pixel32sC2::Min(minmax2, minmax1) == Pixel32sC2(-20, 10));

    CHECK(Pixel32sC2::Max(minmax1, minmax2) == Pixel32sC2(10, 20));
    CHECK(Pixel32sC2::Max(minmax2, minmax1) == Pixel32sC2(10, 20));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 20);

    CHECK(Pixel32sC2(10, 11).DivRound(Pixel32sC2(3, 4)) == Pixel32sC2(3, 3));
    CHECK(Pixel32sC2(10, 11).DivRoundNearest(Pixel32sC2(3, 4)) == Pixel32sC2(3, 3));
    CHECK(Pixel32sC2(10, 11).DivRoundZero(Pixel32sC2(3, 4)) == Pixel32sC2(3, 2));
    CHECK(Pixel32sC2(10, 11).DivFloor(Pixel32sC2(3, 4)) == Pixel32sC2(3, 2));
    CHECK(Pixel32sC2(10, 11).DivCeil(Pixel32sC2(3, 4)) == Pixel32sC2(4, 3));

    CHECK(Pixel32sC2(3, 4).DivInvRound(Pixel32sC2(10, 11)) == Pixel32sC2(3, 3));
    CHECK(Pixel32sC2(3, 4).DivInvRoundNearest(Pixel32sC2(10, 11)) == Pixel32sC2(3, 3));
    CHECK(Pixel32sC2(3, 4).DivInvRoundZero(Pixel32sC2(10, 11)) == Pixel32sC2(3, 2));
    CHECK(Pixel32sC2(3, 4).DivInvFloor(Pixel32sC2(10, 11)) == Pixel32sC2(3, 2));
    CHECK(Pixel32sC2(3, 4).DivInvCeil(Pixel32sC2(10, 11)) == Pixel32sC2(4, 3));

    CHECK(Pixel32sC2::DivRound(Pixel32sC2(10, 11), Pixel32sC2(-3, -4)) == Pixel32sC2(-3, -3));
    CHECK(Pixel32sC2::DivRoundNearest(Pixel32sC2(10, 11), Pixel32sC2(-3, -4)) == Pixel32sC2(-3, -3));
    CHECK(Pixel32sC2::DivRoundZero(Pixel32sC2(10, 11), Pixel32sC2(-3, -4)) == Pixel32sC2(-3, -2));
    CHECK(Pixel32sC2::DivFloor(Pixel32sC2(10, 11), Pixel32sC2(-3, -4)) == Pixel32sC2(-4, -3));
    CHECK(Pixel32sC2::DivCeil(Pixel32sC2(10, 11), Pixel32sC2(-3, -4)) == Pixel32sC2(-3, -2));

    CHECK(Pixel32sC2(-9, 15).DivScaleRoundNearest(10) == Pixel32sC2(-1, 2));
}

TEST_CASE("Pixel32sC2_additionalMethods", "[Common]")
{
    Pixel32sC2 exp(4, 5);
    Pixel32sC2 exp2 = Pixel32sC2::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == 54);
    CHECK(exp.y == 148);

    Pixel32sC2 ln(4, 50);
    Pixel32sC2 ln2 = Pixel32sC2::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == 1);
    CHECK(ln.y == 3);

    Pixel32sC2 sqr(4, 5);
    Pixel32sC2 sqr2 = Pixel32sC2::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == 16);
    CHECK(sqr.y == 25);

    Pixel32sC2 sqrt(4, 5);
    Pixel32sC2 sqrt2 = Pixel32sC2::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == 2);
    CHECK(sqrt.y == 2);

    Pixel32sC2 abs(-2, 12);
    Pixel32sC2 abs2 = Pixel32sC2::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == 2);
    CHECK(abs.y == 12);

    Pixel32sC2 clampByte(int(numeric_limits<byte>::max()) + 1, int(numeric_limits<byte>::min()) - 1);
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 255);
    CHECK(clampByte.y == 0);

    Pixel32sC2 clampShort(int(numeric_limits<short>::max()) + 1, int(numeric_limits<short>::min()) - 1);
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32767);
    CHECK(clampShort.y == -32768);

    Pixel32sC2 clampSByte(int(numeric_limits<sbyte>::max()) + 1, int(numeric_limits<sbyte>::min()) - 1);
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 127);
    CHECK(clampSByte.y == -128);

    Pixel32sC2 clampUShort(int(numeric_limits<ushort>::max()) + 1, int(numeric_limits<ushort>::min()) - 1);
    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 65535);
    CHECK(clampUShort.y == 0);

    Pixel32sC2 clampInt(numeric_limits<int>::max(), numeric_limits<int>::min());
    clampInt.template ClampToTargetType<int>();
    CHECK(clampInt.x == 2147483647);
    CHECK(clampInt.y == -2147483648);

    Pixel32sC2 lshift(1024, 2048);
    Pixel32sC2 lshift2 = Pixel32sC2::LShift(lshift, 2);
    lshift.LShift(2);
    CHECK(lshift == lshift2);
    CHECK(lshift.x == 4096);
    CHECK(lshift.y == 8192);

    Pixel32sC2 rshift(1024, 2048);
    Pixel32sC2 rshift2 = Pixel32sC2::RShift(rshift, 2);
    rshift.RShift(2);
    CHECK(rshift == rshift2);
    CHECK(rshift.x == 256);
    CHECK(rshift.y == 512);

    Pixel32sC2 and_(1023, 2047);
    Pixel32sC2 and_B(512, 1024);
    Pixel32sC2 and_2 = Pixel32sC2::And(and_, and_B);
    and_.And(and_B);
    CHECK(and_ == and_2);
    CHECK(and_.x == 512);
    CHECK(and_.y == 1024);

    Pixel32sC2 or_(1023, 2047);
    Pixel32sC2 or_B(512, 1024);
    Pixel32sC2 or_2 = Pixel32sC2::Or(or_, or_B);
    or_.Or(or_B);
    CHECK(or_ == or_2);
    CHECK(or_.x == 1023);
    CHECK(or_.y == 2047);

    Pixel32sC2 xor_(1023, 2047);
    Pixel32sC2 xor_B(512, 1024);
    Pixel32sC2 xor_2 = Pixel32sC2::Xor(xor_, xor_B);
    xor_.Xor(xor_B);
    CHECK(xor_ == xor_2);
    CHECK(xor_.x == 511);
    CHECK(xor_.y == 1023);

    Pixel32sC2 not_(1023, 2047);
    Pixel32sC2 not_2 = Pixel32sC2::Not(not_);
    not_.Not();
    CHECK(not_ == not_2);
    CHECK(not_.x == -1024);
    CHECK(not_.y == -2048);
}

TEST_CASE("Pixel32fC2_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel32fC2 *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel32fC2(float(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->x;
        void *ptrLastMember  = &buffer[i]->y;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<float *>(ptrVector) + 1));

        // vector must be 8 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 8 == 0);
    }

    std::vector<Pixel32fC2> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (2 * sizeof(float))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Pixel32sC2_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel32sC2 *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel32sC2(int(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->x;
        void *ptrLastMember  = &buffer[i]->y;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<int *>(ptrVector) + 1));

        // vector must be 8 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 8 == 0);
    }

    std::vector<Pixel32sC2> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (2 * sizeof(int))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Pixel16uC2_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel16uC2 *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel16uC2(ushort(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->x;
        void *ptrLastMember  = &buffer[i]->y;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<ushort *>(ptrVector) + 1));

        // vector must be 4 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 4 == 0);
    }

    std::vector<Pixel16uC2> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (2 * sizeof(ushort))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Pixel8uC2_streams", "[Common]")
{
    std::string str = "3 4";
    std::stringstream ss(str);

    Pixel8uC2 pix;
    ss >> pix;

    CHECK(pix.x == 3);
    CHECK(pix.y == 4);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3, 4)");
}

TEST_CASE("Pixel32sC2_streams", "[Common]")
{
    std::string str = "3 4";
    std::stringstream ss(str);

    Pixel32sC2 pix;
    ss >> pix;

    CHECK(pix.x == 3);
    CHECK(pix.y == 4);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3, 4)");
}

TEST_CASE("Pixel32fC2_streams", "[Common]")
{
    std::string str = "3.14 2.7";
    std::stringstream ss(str);

    Pixel32fC2 pix;
    ss >> pix;

    CHECK(pix.x == 3.14f);
    CHECK(pix.y == 2.7f);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3.14, 2.7)");
}

TEST_CASE("Axis2D", "[Common]")
{
    Pixel32sC2 pix(2, 3);

    CHECK(pix[Axis2D::X] == 2);
    CHECK(pix[Axis2D::Y] == 3);

    CHECK_THROWS_AS(pix[static_cast<Axis2D>(5)], mpp::InvalidArgumentException);

    try
    {
        pix[static_cast<Axis2D>(5)] = 12;
    }
    catch (const mpp::InvalidArgumentException &ex)
    {
        CHECK(ex.Message() == "Out of range: 5. Must be X or Y (0 or 1).");
    }
}

TEST_CASE("Pixel16fC2", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel16fC2) == 2 * sizeof(HalfFp16));

    HalfFp16 arr[4] = {4.0_hf, 5.0_hf};
    Pixel16fC2 t0(arr);
    CHECK(t0.x == 4);
    CHECK(t0.y == 5);

    Pixel16fC2 t1(0.0_hf, 1.0_hf);
    CHECK(t1.x == 0);
    CHECK(t1.y == 1);

    Pixel16fC2 c(t1);
    CHECK(c.x == 0);
    CHECK(c.y == 1);
    CHECK(c == t1);

    Pixel16fC2 c2 = t1;
    CHECK(c2.x == 0);
    CHECK(c2.y == 1);
    CHECK(c2 == t1);

    Pixel16fC2 t2(5.0_hf);
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(c2 != t2);

    Pixel16fC2 add1 = t1 + t2;
    CHECK(add1.x == 5);
    CHECK(add1.y == 6);

    Pixel16fC2 add2 = 3 + t1;
    CHECK(add2.x == 3);
    CHECK(add2.y == 4);

    Pixel16fC2 add3 = t1 + 4;
    CHECK(add3.x == 4);
    CHECK(add3.y == 5);

    Pixel16fC2 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 4);
    CHECK(add4.y == 6);

    add4 += 3.0_hf;
    CHECK(add4.x == 7);
    CHECK(add4.y == 9);

    Pixel16fC2 sub1 = t1 - t2;
    CHECK(sub1.x == -5);
    CHECK(sub1.y == -4);

    Pixel16fC2 sub2 = 3 - t1;
    CHECK(sub2.x == 3);
    CHECK(sub2.y == 2);

    Pixel16fC2 sub3 = t1 - 4;
    CHECK(sub3.x == -4);
    CHECK(sub3.y == -3);

    Pixel16fC2 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);

    sub4 -= 3.0_hf;
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);

    t1              = Pixel16fC2(4.0_hf, 5.0_hf);
    t2              = Pixel16fC2(6.0_hf, 7.0_hf);
    Pixel16fC2 mul1 = t1 * t2;
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);

    Pixel16fC2 mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);

    Pixel16fC2 mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);

    Pixel16fC2 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);

    mul4 *= 3.0_hf;
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);

    Pixel16fC2 div1 = t1 / t2;
    CHECK(div1.x == Approx(0.667).margin(0.001));
    CHECK(div1.y == Approx(0.714).margin(0.001));

    Pixel16fC2 div2 = 3 / t1;
    CHECK(div2.x == Approx(0.75).margin(0.001));
    CHECK(div2.y == Approx(0.6).margin(0.001));

    Pixel16fC2 div3 = t1 / 4;
    CHECK(div3.x == Approx(1.0).margin(0.001));
    CHECK(div3.y == Approx(1.25).margin(0.001));

    Pixel16fC2 div4 = t2;
    div4 /= div3;
    CHECK(div4.x == Approx(6.0).margin(0.001));
    CHECK(div4.y == Approx(5.6).margin(0.01));

    div4 /= 3.0_hf;
    CHECK(div4.x == Approx(2.0).margin(0.001));
    CHECK(div4.y == Approx(1.867).margin(0.001));

    Pixel16fC2 minmax1(10.0_hf, 20.0_hf);
    Pixel16fC2 minmax2(-20.0_hf, 10.0_hf);

    CHECK(Pixel16fC2::Min(minmax1, minmax2) == Pixel16fC2(-20.0_hf, 10.0_hf));
    CHECK(Pixel16fC2::Min(minmax2, minmax1) == Pixel16fC2(-20.0_hf, 10.0_hf));

    CHECK(Pixel16fC2::Max(minmax1, minmax2) == Pixel16fC2(10.0_hf, 20.0_hf));
    CHECK(Pixel16fC2::Max(minmax2, minmax1) == Pixel16fC2(10.0_hf, 20.0_hf));

    CHECK(minmax2.Min() == -20);
    CHECK(minmax1.Max() == 20);
}

TEST_CASE("Pixel16fC2_additionalMethods", "[Common]")
{
    Pixel16fC2 roundA(0.4_hf, 0.5_hf);
    Pixel16fC2 roundB(1.9_hf, -1.5_hf);
    Pixel16fC2 round2A = Pixel16fC2::Round(roundA);
    Pixel16fC2 round2B = Pixel16fC2::Round(roundB);
    roundA.Round();
    roundB.Round();
    CHECK(round2A == roundA);
    CHECK(round2B == roundB);
    CHECK(roundA.x == 0.0f);
    CHECK(roundA.y == 1.0f);
    CHECK(roundB.x == 2.0f);
    CHECK(roundB.y == -2.0f);

    Pixel16fC2 floorA(0.4_hf, 0.5_hf);
    Pixel16fC2 floorB(1.9_hf, -1.5_hf);
    Pixel16fC2 floor2A = Pixel16fC2::Floor(floorA);
    Pixel16fC2 floor2B = Pixel16fC2::Floor(floorB);
    floorA.Floor();
    floorB.Floor();
    CHECK(floor2A == floorA);
    CHECK(floor2B == floorB);
    CHECK(floorA.x == 0.0f);
    CHECK(floorA.y == 0.0f);
    CHECK(floorB.x == 1.0f);
    CHECK(floorB.y == -2.0f);

    Pixel16fC2 ceilA(0.4_hf, 0.5_hf);
    Pixel16fC2 ceilB(1.9_hf, -1.5_hf);
    Pixel16fC2 ceil2A = Pixel16fC2::Ceil(ceilA);
    Pixel16fC2 ceil2B = Pixel16fC2::Ceil(ceilB);
    ceilA.Ceil();
    ceilB.Ceil();
    CHECK(ceil2A == ceilA);
    CHECK(ceil2B == ceilB);
    CHECK(ceilA.x == 1.0f);
    CHECK(ceilA.y == 1.0f);
    CHECK(ceilB.x == 2.0f);
    CHECK(ceilB.y == -1.0f);

    Pixel16fC2 zeroA(0.4_hf, 0.5_hf);
    Pixel16fC2 zeroB(1.9_hf, -1.5_hf);
    Pixel16fC2 zero2A = Pixel16fC2::RoundZero(zeroA);
    Pixel16fC2 zero2B = Pixel16fC2::RoundZero(zeroB);
    zeroA.RoundZero();
    zeroB.RoundZero();
    CHECK(zero2A == zeroA);
    CHECK(zero2B == zeroB);
    CHECK(zeroA.x == 0.0f);
    CHECK(zeroA.y == 0.0f);
    CHECK(zeroB.x == 1.0f);
    CHECK(zeroB.y == -1.0f);

    Pixel16fC2 nearestA(0.4_hf, 0.5_hf);
    Pixel16fC2 nearestB(1.9_hf, -1.5_hf);
    Pixel16fC2 nearest2A = Pixel16fC2::RoundNearest(nearestA);
    Pixel16fC2 nearest2B = Pixel16fC2::RoundNearest(nearestB);
    nearestA.RoundNearest();
    nearestB.RoundNearest();
    CHECK(nearest2A == nearestA);
    CHECK(nearest2B == nearestB);
    CHECK(nearestA.x == 0.0f);
    CHECK(nearestA.y == 0.0f);
    CHECK(nearestB.x == 2.0f);
    CHECK(nearestB.y == -2.0f);

    Pixel16fC2 exp(2.4_hf, 12.5_hf);
    Pixel16fC2 exp2 = Pixel16fC2::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == Approx(11.023176380641601).margin(0.01));
    CHECK(isinf(exp.y));

    Pixel16fC2 ln(2.4_hf, 12.5_hf);
    Pixel16fC2 ln2 = Pixel16fC2::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == Approx(0.875468737353900).margin(0.01));
    CHECK(ln.y == Approx(2.525728644308256).margin(0.01));

    Pixel16fC2 sqr(2.4_hf, 12.5_hf);
    Pixel16fC2 sqr2 = Pixel16fC2::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == Approx(5.76).margin(0.01));
    CHECK(sqr.y == Approx(156.25).margin(0.01));

    Pixel16fC2 sqrt(2.4_hf, 12.5_hf);
    Pixel16fC2 sqrt2 = Pixel16fC2::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == Approx(1.549193338482967).margin(0.01));
    CHECK(sqrt.y == Approx(3.535533905932738).margin(0.01));

    Pixel16fC2 abs(-2.4_hf, 12.5_hf);
    Pixel16fC2 abs2 = Pixel16fC2::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == 2.4_hf);
    CHECK(abs.y == 12.5_hf);

    Pixel16fC2 absdiffA(13.23592_hf, -40.24595_hf);
    Pixel16fC2 absdiffB(45.75068_hf, 46.488853_hf);
    Pixel16fC2 absdiff2 = Pixel16fC2::AbsDiff(absdiffA, absdiffB);
    Pixel16fC2 absdiff3 = Pixel16fC2::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == Approx(32.514758920888809).margin(0.02));
    CHECK(absdiffA.y == Approx(86.734813019986703).margin(0.02));

    Pixel16fC2 clampByte(HalfFp16(float(numeric_limits<byte>::max()) + 1),
                         HalfFp16(float(numeric_limits<byte>::min()) - 1));
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 256);
    CHECK(clampByte.y == -1);

    Pixel16fC2 clampShort(HalfFp16(float(numeric_limits<short>::max()) + 1),
                          HalfFp16(float(numeric_limits<short>::min()) - 1));
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32768);
    CHECK(clampShort.y == -32768);

    Pixel16fC2 clampSByte(HalfFp16(float(numeric_limits<sbyte>::max()) + 1),
                          HalfFp16(float(numeric_limits<sbyte>::min()) - 1));
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 128);
    CHECK(clampSByte.y == -129);

    Pixel16fC2 clampUShort(HalfFp16(0.0f), HalfFp16(float(numeric_limits<ushort>::min()) - 1));

    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 0);
    CHECK(clampUShort.y == -1);

    CHECK_FALSE(need_saturation_clamp_v<HalfFp16, int>);

    CHECK_FALSE(need_saturation_clamp_v<HalfFp16, uint>);
    Pixel16fC2 clampUInt(HalfFp16(0), HalfFp16(float(numeric_limits<uint>::min()) - 1000));

    clampUInt.template ClampToTargetType<uint>();
    CHECK(clampUInt.x == 0);
    CHECK(clampUInt.y == -1000);

    Pixel32fC2 clampFloatToFp16(float(numeric_limits<int>::min()) - 1000.0f, //
                                float(numeric_limits<int>::max()) + 1000.0f);

    clampFloatToFp16.template ClampToTargetType<HalfFp16>();
    CHECK(clampFloatToFp16.x == -2147484672.0f);
    CHECK(clampFloatToFp16.y == 2147484672.0f);

    Pixel16fC2 fromFromFloat(clampFloatToFp16);
    CHECK(fromFromFloat.x == -INFINITY);
    CHECK(fromFromFloat.y == INFINITY);
}

TEST_CASE("Pixel32fcC2", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32fcC2) == 4 * sizeof(float));

    Complex<float> arr[2] = {Complex<float>(4, 2), Complex<float>(5, -2)};
    Pixel32fcC2 t0(arr);
    CHECK(t0.x == Complex<float>(4, 2));
    CHECK(t0.y == Complex<float>(5, -2));

    Pixel32fcC2 t1(Complex<float>(0, 1), Complex<float>(1, -2));
    CHECK(t1.x == Complex<float>(0, 1));
    CHECK(t1.y == Complex<float>(1, -2));

    Pixel32fcC2 c(t1);
    CHECK(c.x == Complex<float>(0, 1));
    CHECK(c.y == Complex<float>(1, -2));
    CHECK(c == t1);

    Pixel32fcC2 c2 = t1;
    CHECK(c2.x == Complex<float>(0, 1));
    CHECK(c2.y == Complex<float>(1, -2));
    CHECK(c2 == t1);

    Pixel32fcC2 t2(5);
    CHECK(t2.x == Complex<float>(5, 0));
    CHECK(t2.y == Complex<float>(5, 0));
    CHECK(c2 != t2);

    Pixel32fcC2 neg = -t1;
    CHECK(neg.x == -1_i);
    CHECK(neg.y == -1 + 2_i);
    CHECK(t1 == -neg);

    Pixel32fcC2 add1 = t1 + t2;
    CHECK(add1.x == Complex<float>(5, 1));
    CHECK(add1.y == Complex<float>(6, -2));

    Pixel32fcC2 add2 = 3 + t1;
    CHECK(add2.x == Complex<float>(3, 1));
    CHECK(add2.y == Complex<float>(4, -2));

    Pixel32fcC2 add3 = t1 + 4;
    CHECK(add3.x == Complex<float>(4, 1));
    CHECK(add3.y == Complex<float>(5, -2));

    Pixel32fcC2 add4 = t1;
    add4 += add3;
    CHECK(add4.x == Complex<float>(4, 2));
    CHECK(add4.y == Complex<float>(6, -4));

    add4 += 3.0f + 0_i;
    CHECK(add4.x == Complex<float>(7, 2));
    CHECK(add4.y == Complex<float>(9, -4));

    Pixel32fcC2 sub1 = t1 - t2;
    CHECK(sub1.x == Complex<float>(-5, 1));
    CHECK(sub1.y == Complex<float>(-4, -2));

    Pixel32fcC2 sub2 = 3 - t1;
    CHECK(sub2.x == Complex<float>(3, -1));
    CHECK(sub2.y == Complex<float>(2, +2));

    Pixel32fcC2 sub3 = t1 - 4;
    CHECK(sub3.x == Complex<float>(-4, 1));
    CHECK(sub3.y == Complex<float>(-3, -2));

    Pixel32fcC2 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == Complex<float>(4, 0));
    CHECK(sub4.y == Complex<float>(4, 0));

    sub4 -= 3.0f + 0_i;
    CHECK(sub4.x == Complex<float>(1, 0));
    CHECK(sub4.y == Complex<float>(1, 0));

    Pixel32fcC2 sub5 = t1;
    Pixel32fcC2 sub6(9, 8);
    sub5.SubInv(sub6);
    CHECK(sub5.x == Complex<float>(9, -1));
    CHECK(sub5.y == Complex<float>(7, 2));

    t1               = Pixel32fcC2(Complex<float>(4, 5), Complex<float>(6, 7));
    t2               = Pixel32fcC2(Complex<float>(5, 6), Complex<float>(7, -8));
    Pixel32fcC2 mul1 = t1 * t2;
    CHECK(mul1.x == Complex<float>(-10, 49));
    CHECK(mul1.y == Complex<float>(98, 1));

    Pixel32fcC2 mul2 = 3 * t1;
    CHECK(mul2.x == Complex<float>(12, 15));
    CHECK(mul2.y == Complex<float>(18, 21));

    Pixel32fcC2 mul3 = t1 * 4;
    CHECK(mul3.x == Complex<float>(16, 20));
    CHECK(mul3.y == Complex<float>(24, 28));

    Pixel32fcC2 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == Complex<float>(-36, 160));
    CHECK(mul4.y == Complex<float>(-52, 336));

    mul4 *= 3.0f + 0_i;
    CHECK(mul4.x == Complex<float>(-108, 480));
    CHECK(mul4.y == Complex<float>(-156, 1008));

    Pixel32fcC2 div1 = t1 / t2;
    CHECK(div1.x.real == Approx(0.819672108f).margin(0.001));
    CHECK(div1.x.imag == Approx(0.016393442f).margin(0.001));
    CHECK(div1.y.real == Approx(-0.123893805f).margin(0.001));
    CHECK(div1.y.imag == Approx(0.85840708f).margin(0.001));

    Pixel32fcC2 div2 = 3 / t1;
    CHECK(div2.x.real == Approx(0.292682916f).margin(0.001));
    CHECK(div2.x.imag == Approx(-0.365853667f).margin(0.001));
    CHECK(div2.y.real == Approx(0.211764708f).margin(0.001));
    CHECK(div2.y.imag == Approx(-0.247058824f).margin(0.001));

    Pixel32fcC2 div3 = t1 / 4;
    CHECK(div3.x.real == Approx(1).margin(0.001));
    CHECK(div3.x.imag == Approx(1.25).margin(0.001));
    CHECK(div3.y.real == Approx(1.5).margin(0.001));
    CHECK(div3.y.imag == Approx(1.75).margin(0.001));

    Pixel32fcC2 div4 = t2;
    div4 /= div3;
    CHECK(div4.x.real == Approx(4.878048897f).margin(0.001));
    CHECK(div4.x.imag == Approx(-0.097560972f).margin(0.001));
    CHECK(div4.y.real == Approx(-0.65882355f).margin(0.001));
    CHECK(div4.y.imag == Approx(-4.564705849f).margin(0.001));

    div4 /= 3.0f + 0_i;
    CHECK(div4.x.real == Approx(1.626016259f).margin(0.001));
    CHECK(div4.x.imag == Approx(-0.032520324f).margin(0.001));
    CHECK(div4.y.real == Approx(-0.21960786f).margin(0.001));
    CHECK(div4.y.imag == Approx(-1.521568656f).margin(0.001));

    Pixel32fcC2 div5 = t1;
    Pixel32fcC2 div6(9, 8);
    Pixel32fcC2 difv7 = div6 / div5;
    div5.DivInv(div6);
    CHECK(div5.x == difv7.x);
    CHECK(div5.y == difv7.y);

    Pixel32fcC2 conj1 = t1;
    Pixel32fcC2 conj2 = Pixel32fcC2::Conj(t1);
    conj1.Conj();
    CHECK(conj1.x == 4 - 5_i);
    CHECK(conj1.y == 6 - 7_i);
    CHECK(conj2.x == 4 - 5_i);
    CHECK(conj2.y == 6 - 7_i);

    conj1 = t1;
    conj1.ConjMul(t2);
    conj2 = Pixel32fcC2::ConjMul(t1, t2);
    CHECK(conj1.x == 50 + 1_i);
    CHECK(conj1.y == -14 + 97_i);
    CHECK(conj2.x == 50 + 1_i);
    CHECK(conj2.y == -14 + 97_i);

    Pixel32fcC2 a = t1;
    Pixel32fcC2 s = t1;
    Pixel32fcC2 m = t1;
    Pixel32fcC2 d = t1;

    a += 5.0f;
    s -= 5.0f;
    m *= 5.0f;
    d /= 5.0f;
    CHECK(a == t1 + 5.0f);
    CHECK(s == t1 - 5.0f);
    CHECK(m == t1 * 5.0f);
    CHECK(d == t1 / 5.0f);
}

TEST_CASE("Pixel32fcC2_additionalMethods", "[Common]")
{
    Pixel32fcC2 roundA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f));
    Pixel32fcC2 round2A = Pixel32fcC2::Round(roundA);
    roundA.Round();
    CHECK(round2A == roundA);
    CHECK(roundA.x.real == 0.0f);
    CHECK(roundA.x.imag == 1.0f);
    CHECK(roundA.y.real == 1.0f);
    CHECK(roundA.y.imag == 2.0f);

    Pixel32fcC2 floorA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f));
    Pixel32fcC2 floor2A = Pixel32fcC2::Floor(floorA);
    floorA.Floor();
    CHECK(floor2A == floorA);
    CHECK(floorA.x.real == 0.0f);
    CHECK(floorA.x.imag == 0.0f);
    CHECK(floorA.y.real == 0.0f);
    CHECK(floorA.y.imag == 1.0f);

    Pixel32fcC2 ceilA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f));
    Pixel32fcC2 ceil2A = Pixel32fcC2::Ceil(ceilA);
    ceilA.Ceil();
    CHECK(ceil2A == ceilA);
    CHECK(ceilA.x.real == 1.0f);
    CHECK(ceilA.x.imag == 1.0f);
    CHECK(ceilA.y.real == 1.0f);
    CHECK(ceilA.y.imag == 2.0f);

    Pixel32fcC2 zeroA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f));
    Pixel32fcC2 zero2A = Pixel32fcC2::RoundZero(zeroA);
    zeroA.RoundZero();
    CHECK(zero2A == zeroA);
    CHECK(zeroA.x.real == 0.0f);
    CHECK(zeroA.x.imag == 0.0f);
    CHECK(zeroA.y.real == 0.0f);
    CHECK(zeroA.y.imag == 1.0f);

    Pixel32fcC2 nearestA(Complex<float>(0.4f, 0.5f), Complex<float>(0.6f, 1.5f));
    Pixel32fcC2 nearest2A = Pixel32fcC2::RoundNearest(nearestA);
    nearestA.RoundNearest();
    CHECK(nearest2A == nearestA);
    CHECK(nearestA.x.real == 0.0f);
    CHECK(nearestA.x.imag == 0.0f);
    CHECK(nearestA.y.real == 1.0f);
    CHECK(nearestA.y.imag == 2.0f);

    // Not vector does the value clampling but the complex type!
    Pixel32fcC2 clampToShort(float(numeric_limits<short>::max()) + 1, float(numeric_limits<short>::min()) - 1);
    Pixel16scC2 clampShort(clampToShort);
    CHECK(clampShort.x.real == 32767);
    CHECK(clampShort.y.real == -32768);

    Pixel32fcC2 clampToInt(float(numeric_limits<int>::max()) + 1000, float(numeric_limits<int>::min()) - 1000);

    Pixel32scC2 clampInt(clampToInt);
    CHECK(clampInt.x.real == 2147483647);
    CHECK(clampInt.y.real == -2147483648);
}

TEST_CASE("Pixel64sC2", "[Common]")
{
    CHECK(Pixel64sC2(10, 11).DivRound(Pixel64sC2(3, 4)) == Pixel64sC2(3, 3));
    CHECK(Pixel64sC2(10, 11).DivRoundNearest(Pixel64sC2(3, 4)) == Pixel64sC2(3, 3));
    CHECK(Pixel64sC2(10, 11).DivRoundZero(Pixel64sC2(3, 4)) == Pixel64sC2(3, 2));
    CHECK(Pixel64sC2(10, 11).DivFloor(Pixel64sC2(3, 4)) == Pixel64sC2(3, 2));
    CHECK(Pixel64sC2(10, 11).DivCeil(Pixel64sC2(3, 4)) == Pixel64sC2(4, 3));

    CHECK(Pixel64sC2::DivRound(Pixel64sC2(10, 11), Pixel64sC2(-3, -4)) == Pixel64sC2(-3, -3));
    CHECK(Pixel64sC2::DivRoundNearest(Pixel64sC2(10, 11), Pixel64sC2(-3, -4)) == Pixel64sC2(-3, -3));
    CHECK(Pixel64sC2::DivRoundZero(Pixel64sC2(10, 11), Pixel64sC2(-3, -4)) == Pixel64sC2(-3, -2));
    CHECK(Pixel64sC2::DivFloor(Pixel64sC2(10, 11), Pixel64sC2(-3, -4)) == Pixel64sC2(-4, -3));
    CHECK(Pixel64sC2::DivCeil(Pixel64sC2(10, 11), Pixel64sC2(-3, -4)) == Pixel64sC2(-3, -2));

    CHECK(Pixel64sC2(-9, 15).DivScaleRoundNearest(10) == Pixel64sC2(-1, 2));
}

TEST_CASE("Pixel64scC2", "[Common]")
{
    CHECK(Pixel64scC2(10 - 4_i, 11 + 4_i).DivRound(Pixel64scC2(3 + 2_i, 4 + 2_i)) == Pixel64scC2(2 - 2_i, 3));
    CHECK(Pixel64scC2(10 - 4_i, 11 + 4_i).DivRoundNearest(Pixel64scC2(3 + 2_i, 4 + 2_i)) == Pixel64scC2(2 - 2_i, 3));
    CHECK(Pixel64scC2(10 - 4_i, 11 + 4_i).DivRoundZero(Pixel64scC2(3 + 2_i, 4 + 2_i)) == Pixel64scC2(1 - 2_i, 2));
    CHECK(Pixel64scC2(10 - 4_i, 11 + 4_i).DivFloor(Pixel64scC2(3 + 2_i, 4 + 2_i)) == Pixel64scC2(1 - 3_i, 2 - 1_i));
    CHECK(Pixel64scC2(10 - 4_i, 11 + 4_i).DivCeil(Pixel64scC2(3 + 2_i, 4 + 2_i)) == Pixel64scC2(2 - 2_i, 3));

    CHECK(Pixel64scC2(3 + 2_i, 4 + 2_i).DivInvRound(Pixel64scC2(10 - 4_i, 11 + 4_i)) == Pixel64scC2(2 - 2_i, 3));
    CHECK(Pixel64scC2(3 + 2_i, 4 + 2_i).DivInvRoundNearest(Pixel64scC2(10 - 4_i, 11 + 4_i)) == Pixel64scC2(2 - 2_i, 3));
    CHECK(Pixel64scC2(3 + 2_i, 4 + 2_i).DivInvRoundZero(Pixel64scC2(10 - 4_i, 11 + 4_i)) == Pixel64scC2(1 - 2_i, 2));
    CHECK(Pixel64scC2(3 + 2_i, 4 + 2_i).DivInvFloor(Pixel64scC2(10 - 4_i, 11 + 4_i)) == Pixel64scC2(1 - 3_i, 2 - 1_i));
    CHECK(Pixel64scC2(3 + 2_i, 4 + 2_i).DivInvCeil(Pixel64scC2(10 - 4_i, 11 + 4_i)) == Pixel64scC2(2 - 2_i, 3));

    CHECK(Pixel64scC2::DivRound(Pixel64scC2(-10 + 4_i, -11 - 4_i), Pixel64scC2(3 + 2_i, 4 + 2_i)) ==
          Pixel64scC2(-2 + 2_i, -3));
    CHECK(Pixel64scC2::DivRoundNearest(Pixel64scC2(-10 + 4_i, -11 - 4_i), Pixel64scC2(3 + 2_i, 4 + 2_i)) ==
          Pixel64scC2(-2 + 2_i, -3));
    CHECK(Pixel64scC2::DivRoundZero(Pixel64scC2(-10 + 4_i, -11 - 4_i), Pixel64scC2(3 + 2_i, 4 + 2_i)) ==
          Pixel64scC2(-1 + 2_i, -2));
    CHECK(Pixel64scC2::DivFloor(Pixel64scC2(-10 + 4_i, -11 - 4_i), Pixel64scC2(3 + 2_i, 4 + 2_i)) ==
          Pixel64scC2(-2 + 2_i, -3));
    CHECK(Pixel64scC2::DivCeil(Pixel64scC2(-10 + 4_i, -11 - 4_i), Pixel64scC2(3 + 2_i, 4 + 2_i)) ==
          Pixel64scC2(-1 + 3_i, -2 + 1_i));

    CHECK(Pixel64scC2(-9 + 11_i, 15 - 5_i).DivScaleRoundNearest(10) == Pixel64scC2(-1 + 1_i, 2));
}