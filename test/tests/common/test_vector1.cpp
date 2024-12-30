#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/numeric_limits.h>
#include <common/safeCast.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <cstdint>

using namespace opp;
using namespace opp::image;
using namespace Catch;

TEST_CASE("Pixel32fC1", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32fC1) == 1 * sizeof(float));

    float arr[1] = {4};
    Pixel32fC1 t0(arr);
    CHECK(t0.x == 4);

    Pixel32fC1 t1(1);
    CHECK(t1.x == 1);

    Pixel32fC1 c(t1);
    CHECK(c.x == 1);
    CHECK(c == t1);

    Pixel32fC1 c2 = t1;
    CHECK(c2.x == 1);
    CHECK(c2 == t1);

    Pixel32fC1 t2(5);
    CHECK(t2.x == 5);
    CHECK(c2 != t2);

    Pixel32fC1 add1 = t1 + t2;
    CHECK(add1.x == 6);

    Pixel32fC1 add2 = 3 + t1;
    CHECK(add2.x == 4);

    Pixel32fC1 add3 = t1 + 4;
    CHECK(add3.x == 5);

    Pixel32fC1 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 6);

    add4 += 3;
    CHECK(add4.x == 9);

    Pixel32fC1 sub1 = t1 - t2;
    CHECK(sub1.x == -4);

    Pixel32fC1 sub2 = 3 - t1;
    CHECK(sub2.x == 2);

    Pixel32fC1 sub3 = t1 - 4;
    CHECK(sub3.x == -3);

    Pixel32fC1 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);

    sub4 -= 3;
    CHECK(sub4.x == 1);

    t1              = Pixel32fC1(4);
    t2              = Pixel32fC1(6);
    Pixel32fC1 mul1 = t1 * t2;
    CHECK(mul1.x == 24);

    Pixel32fC1 mul2 = 3 * t1;
    CHECK(mul2.x == 12);

    Pixel32fC1 mul3 = t1 * 4;
    CHECK(mul3.x == 16);

    Pixel32fC1 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);

    mul4 *= 3;
    CHECK(mul4.x == 192);

    Pixel32fC1 div1 = t1 / t2;
    CHECK(div1.x == Approx(0.667).margin(0.001));

    Pixel32fC1 div2 = 3 / t1;
    CHECK(div2.x == Approx(0.75).margin(0.001));

    Pixel32fC1 div3 = t1 / 4;
    CHECK(div3.x == Approx(1.0).margin(0.001));

    Pixel32fC1 div4 = t2;
    div4 /= div3;
    CHECK(div4.x == Approx(6.0).margin(0.001));

    div4 /= 3;
    CHECK(div4.x == Approx(2.0).margin(0.001));

    Pixel32fC1 l(4);
    CHECK(l.MagnitudeSqr() == 16);
    CHECK(l.Magnitude() == Approx(std::sqrt(16)));

    Pixel32fC1 minmax1(10);
    Pixel32fC1 minmax2(-20);

    CHECK(Pixel32fC1::Min(minmax1, minmax2) == Pixel32fC1(-20));
    CHECK(Pixel32fC1::Min(minmax2, minmax1) == Pixel32fC1(-20));

    CHECK(Pixel32fC1::Max(minmax1, minmax2) == Pixel32fC1(10));
    CHECK(Pixel32fC1::Max(minmax2, minmax1) == Pixel32fC1(10));

    CHECK(Pixel32fC1::CompareEQ(Pixel32fC1(12.0f), Pixel32fC1(12.0f)) == Pixel8uC1(0xFF));
    CHECK(Pixel32fC1::CompareEQ(Pixel32fC1(12.0f), Pixel32fC1(13.0f)) == Pixel8uC1(0x00));

    CHECK(Pixel32fC1::CompareNEQ(Pixel32fC1(12.0f), Pixel32fC1(13.0f)) == Pixel8uC1(0xFF));
    CHECK(Pixel32fC1::CompareNEQ(Pixel32fC1(12.0f), Pixel32fC1(12.0f)) == Pixel8uC1(0x00));

    CHECK(Pixel32fC1::CompareGE(Pixel32fC1(13.0f), Pixel32fC1(12.0f)) == Pixel8uC1(0xFF));
    CHECK(Pixel32fC1::CompareGE(Pixel32fC1(12.0f), Pixel32fC1(13.0f)) == Pixel8uC1(0x00));
    CHECK(Pixel32fC1::CompareGE(Pixel32fC1(12.0f), Pixel32fC1(12.0f)) == Pixel8uC1(0xFF));
    CHECK(Pixel32fC1::CompareGE(Pixel32fC1(12.0f), Pixel32fC1(13.0f)) == Pixel8uC1(0x00));

    CHECK(Pixel32fC1::CompareGT(Pixel32fC1(13.0f), Pixel32fC1(12.0f)) == Pixel8uC1(0xFF));
    CHECK(Pixel32fC1::CompareGT(Pixel32fC1(12.0f), Pixel32fC1(13.0f)) == Pixel8uC1(0x00));
    CHECK(Pixel32fC1::CompareGT(Pixel32fC1(12.0f), Pixel32fC1(12.0f)) == Pixel8uC1(0x00));

    CHECK(Pixel32fC1::CompareLE(Pixel32fC1(12.0f), Pixel32fC1(13.0f)) == Pixel8uC1(0xFF));
    CHECK(Pixel32fC1::CompareLE(Pixel32fC1(13.0f), Pixel32fC1(12.0f)) == Pixel8uC1(0x00));
    CHECK(Pixel32fC1::CompareLE(Pixel32fC1(12.0f), Pixel32fC1(12.0f)) == Pixel8uC1(0xFF));
    CHECK(Pixel32fC1::CompareLE(Pixel32fC1(13.0f), Pixel32fC1(12.0f)) == Pixel8uC1(0x00));

    CHECK(Pixel32fC1::CompareLT(Pixel32fC1(12.0f), Pixel32fC1(13.0f)) == Pixel8uC1(0xFF));
    CHECK(Pixel32fC1::CompareLT(Pixel32fC1(13.0f), Pixel32fC1(12.0f)) == Pixel8uC1(0x00));
    CHECK(Pixel32fC1::CompareLT(Pixel32fC1(12.0f), Pixel32fC1(12.0f)) == Pixel8uC1(0x00));
}

TEST_CASE("Pixel32fC1_additionalMethods", "[Common]")
{
    Pixel32fC1 norm(4);
    norm.Normalize();
    CHECK(norm.Magnitude() == 1);
    CHECK(norm.x == Approx(1).margin(0.001));

    CHECK(Pixel32fC1::Normalize(Pixel32fC1(4)) == norm);

    Pixel32fC1 roundA(0.4f);
    Pixel32fC1 roundB(1.9f);
    Pixel32fC1 round2A = Pixel32fC1::Round(roundA);
    Pixel32fC1 round2B = Pixel32fC1::Round(roundB);
    roundA.Round();
    roundB.Round();
    CHECK(round2A == roundA);
    CHECK(round2B == roundB);
    CHECK(roundA.x == 0.0f);
    CHECK(roundB.x == 2.0f);

    Pixel32fC1 floorA(0.4f);
    Pixel32fC1 floorB(1.9f);
    Pixel32fC1 floor2A = Pixel32fC1::Floor(floorA);
    Pixel32fC1 floor2B = Pixel32fC1::Floor(floorB);
    floorA.Floor();
    floorB.Floor();
    CHECK(floor2A == floorA);
    CHECK(floor2B == floorB);
    CHECK(floorA.x == 0.0f);
    CHECK(floorB.x == 1.0f);

    Pixel32fC1 ceilA(0.4f);
    Pixel32fC1 ceilB(1.9f);
    Pixel32fC1 ceil2A = Pixel32fC1::Ceil(ceilA);
    Pixel32fC1 ceil2B = Pixel32fC1::Ceil(ceilB);
    ceilA.Ceil();
    ceilB.Ceil();
    CHECK(ceil2A == ceilA);
    CHECK(ceil2B == ceilB);
    CHECK(ceilA.x == 1.0f);
    CHECK(ceilB.x == 2.0f);

    Pixel32fC1 zeroA(0.4f);
    Pixel32fC1 zeroB(1.9f);
    Pixel32fC1 zero2A = Pixel32fC1::RoundZero(zeroA);
    Pixel32fC1 zero2B = Pixel32fC1::RoundZero(zeroB);
    zeroA.RoundZero();
    zeroB.RoundZero();
    CHECK(zero2A == zeroA);
    CHECK(zero2B == zeroB);
    CHECK(zeroA.x == 0.0f);
    CHECK(zeroB.x == 1.0f);

    Pixel32fC1 nearestA(0.4f);
    Pixel32fC1 nearestB(1.9f);
    Pixel32fC1 nearest2A = Pixel32fC1::RoundNearest(nearestA);
    Pixel32fC1 nearest2B = Pixel32fC1::RoundNearest(nearestB);
    nearestA.RoundNearest();
    nearestB.RoundNearest();
    CHECK(nearest2A == nearestA);
    CHECK(nearest2B == nearestB);
    CHECK(nearestA.x == 0.0f);
    CHECK(nearestB.x == 2.0f);

    Pixel32fC1 exp(2.4f);
    Pixel32fC1 exp2 = Pixel32fC1::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == Approx(11.023176380641601).margin(0.00001));

    Pixel32fC1 ln(2.4f);
    Pixel32fC1 ln2 = Pixel32fC1::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == Approx(0.875468737353900).margin(0.00001));

    Pixel32fC1 sqr(2.4f);
    Pixel32fC1 sqr2 = Pixel32fC1::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == Approx(5.76).margin(0.00001));

    Pixel32fC1 sqrt(2.4f);
    Pixel32fC1 sqrt2 = Pixel32fC1::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == Approx(1.549193338482967).margin(0.00001));

    Pixel32fC1 abs(-2.4f);
    Pixel32fC1 abs2 = Pixel32fC1::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == 2.4f);

    Pixel32fC1 absdiffA(13.23592f);
    Pixel32fC1 absdiffB(45.75068f);
    Pixel32fC1 absdiff2 = Pixel32fC1::AbsDiff(absdiffA, absdiffB);
    Pixel32fC1 absdiff3 = Pixel32fC1::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == Approx(32.514758920888809).margin(0.00001));

    Pixel32fC1 clampByte(float(numeric_limits<byte>::max()) + 1);
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 255);

    Pixel32fC1 clampShort(float(numeric_limits<short>::max()) + 1);
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32767);

    Pixel32fC1 clampSByte(float(numeric_limits<sbyte>::max()) + 1);
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 127);

    Pixel32fC1 clampUShort(float(numeric_limits<ushort>::max()) + 1);
    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 65535);

    Pixel32fC1 clampInt(float(numeric_limits<int>::max()) + 1000);
    clampInt.template ClampToTargetType<int>();
    CHECK(clampInt.x == 2147483647);

    Pixel32fC1 clampUInt(float(numeric_limits<uint>::max()) + 1000);
    clampUInt.template ClampToTargetType<uint>();
    CHECK(clampUInt.x == 0xffffffffUL);
}

TEST_CASE("Pixel32sC1", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel32sC1) == 1 * sizeof(int));

    Pixel32sC1 t1(100);
    CHECK(t1.x == 100);

    Pixel32sC1 c(t1);
    CHECK(c.x == 100);
    CHECK(c == t1);

    Pixel32sC1 c2 = t1;
    CHECK(c2.x == 100);
    CHECK(c2 == t1);

    Pixel32sC1 t2(5);
    CHECK(t2.x == 5);
    CHECK(c2 != t2);

    Pixel32sC1 add1 = t1 + t2;
    CHECK(add1.x == 105);

    Pixel32sC1 add2 = 3 + t1;
    CHECK(add2.x == 103);

    Pixel32sC1 add3 = t1 + 4;
    CHECK(add3.x == 104);

    Pixel32sC1 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 204);

    add4 += 3;
    CHECK(add4.x == 207);

    Pixel32sC1 sub1 = t1 - t2;
    CHECK(sub1.x == 95);

    Pixel32sC1 sub2 = 3 - t1;
    CHECK(sub2.x == -97);

    Pixel32sC1 sub3 = t1 - 4;
    CHECK(sub3.x == 96);

    Pixel32sC1 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);

    sub4 -= 3;
    CHECK(sub4.x == 1);

    t1              = Pixel32sC1(4);
    t2              = Pixel32sC1(6);
    Pixel32sC1 mul1 = t1 * t2;
    CHECK(mul1.x == 24);

    Pixel32sC1 mul2 = 3 * t1;
    CHECK(mul2.x == 12);

    Pixel32sC1 mul3 = t1 * 4;
    CHECK(mul3.x == 16);

    Pixel32sC1 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);

    mul4 *= 3;
    CHECK(mul4.x == 192);

    t1              = Pixel32sC1(1000);
    t2              = Pixel32sC1(6);
    Pixel32sC1 div1 = t1 / t2;
    CHECK(div1.x == 166);

    Pixel32sC1 div2 = 30000 / t1;
    CHECK(div2.x == 30);

    Pixel32sC1 div3 = t1 / 4;
    CHECK(div3.x == 250);

    Pixel32sC1 div4 = t2 * 10000;
    div4 /= div3;
    CHECK(div4.x == 240);

    div4 /= 3;
    CHECK(div4.x == 80);

    Pixel32sC1 l(4);
    CHECK(l.MagnitudeSqr() == 16);
    CHECK(l.Magnitude() == 4);

    Pixel32sC1 minmax1(10);
    Pixel32sC1 minmax2(-20);

    CHECK(Pixel32sC1::Min(minmax1, minmax2) == Pixel32sC1(-20));
    CHECK(Pixel32sC1::Min(minmax2, minmax1) == Pixel32sC1(-20));

    CHECK(Pixel32sC1::Max(minmax1, minmax2) == Pixel32sC1(10));
    CHECK(Pixel32sC1::Max(minmax2, minmax1) == Pixel32sC1(10));
}

TEST_CASE("Pixel32sC1_additionalMethods", "[Common]")
{
    Pixel32sC1 norm(4);
    CHECK(norm.Magnitude() == 4);
    CHECK(norm.MagnitudeSqr() == 16);

    Pixel32sC1 exp(4);
    Pixel32sC1 exp2 = Pixel32sC1::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == 54);

    Pixel32sC1 ln(4);
    Pixel32sC1 ln2 = Pixel32sC1::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == 1);

    Pixel32sC1 sqr(4);
    Pixel32sC1 sqr2 = Pixel32sC1::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == 16);

    Pixel32sC1 sqrt(4);
    Pixel32sC1 sqrt2 = Pixel32sC1::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == 2);

    Pixel32sC1 abs(-2);
    Pixel32sC1 abs2 = Pixel32sC1::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == 2);

    Pixel32sC1 absdiffA(13);
    Pixel32sC1 absdiffB(45);
    Pixel32sC1 absdiff2 = Pixel32sC1::AbsDiff(absdiffA, absdiffB);
    Pixel32sC1 absdiff3 = Pixel32sC1::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == 32);

    Pixel32sC1 clampByte(int(numeric_limits<byte>::max()) + 1);
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 255);

    Pixel32sC1 clampShort(int(numeric_limits<short>::max()) + 1);
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32767);

    Pixel32sC1 clampSByte(int(numeric_limits<sbyte>::max()) + 1);
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 127);

    Pixel32sC1 clampUShort(int(numeric_limits<ushort>::max()) + 1);
    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 65535);

    Pixel32sC1 clampInt(numeric_limits<int>::max());
    clampInt.template ClampToTargetType<int>();
    CHECK(clampInt.x == 2147483647);

    Pixel32sC1 lshift(1024);
    Pixel32sC1 lshift2 = Pixel32sC1::LShift(lshift, 2);
    lshift.LShift(2);
    CHECK(lshift == lshift2);
    CHECK(lshift.x == 4096);

    Pixel32sC1 rshift(1024);
    Pixel32sC1 rshift2 = Pixel32sC1::RShift(rshift, 2);
    rshift.RShift(2);
    CHECK(rshift == rshift2);
    CHECK(rshift.x == 256);

    Pixel32sC1 and_(1023);
    Pixel32sC1 and_B(512);
    Pixel32sC1 and_2 = Pixel32sC1::And(and_, and_B);
    and_.And(and_B);
    CHECK(and_ == and_2);
    CHECK(and_.x == 512);

    Pixel32sC1 or_(1023);
    Pixel32sC1 or_B(512);
    Pixel32sC1 or_2 = Pixel32sC1::Or(or_, or_B);
    or_.Or(or_B);
    CHECK(or_ == or_2);
    CHECK(or_.x == 1023);

    Pixel32sC1 xor_(1023);
    Pixel32sC1 xor_B(512);
    Pixel32sC1 xor_2 = Pixel32sC1::Xor(xor_, xor_B);
    xor_.Xor(xor_B);
    CHECK(xor_ == xor_2);
    CHECK(xor_.x == 511);

    Pixel32sC1 not_(1023);
    Pixel32sC1 not_2 = Pixel32sC1::Not(not_);
    not_.Not();
    CHECK(not_ == not_2);
    CHECK(not_.x == -1024);
}

TEST_CASE("Pixel32fC1_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel32fC1 *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel32fC1(float(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->x;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);

        // vector must be 4 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 4 == 0);
    }

    std::vector<Pixel32fC1> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (1 * sizeof(float))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Pixel32sC1_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel32sC1 *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel32sC1(int(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->x;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);

        // vector must be 4 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 4 == 0);
    }

    std::vector<Pixel32sC1> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (1 * sizeof(int))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Pixel16uC1_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Pixel16uC1 *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Pixel16uC1(ushort(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->x;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);

        // vector must be 2 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 2 == 0);
    }

    std::vector<Pixel16uC1> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (1 * sizeof(ushort))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Pixel32sC1_streams", "[Common]")
{
    std::string str = "3";
    std::stringstream ss(str);

    Pixel32sC1 pix;
    ss >> pix;

    CHECK(pix.x == 3);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3)");
}

TEST_CASE("Pixel32fC1_streams", "[Common]")
{
    std::string str = "3.14";
    std::stringstream ss(str);

    Pixel32fC1 pix;
    ss >> pix;

    CHECK(pix.x == 3.14f);

    std::stringstream ss2;
    ss2 << pix;

    CHECK(ss2.str() == "(3.14)");
}

TEST_CASE("Axis1D", "[Common]")
{
    Pixel32sC1 pix(2);

    CHECK(pix[Axis1D::X] == 2);

    CHECK_THROWS_AS(pix[static_cast<Axis1D>(5)], opp::InvalidArgumentException);

    try
    {
        pix[static_cast<Axis1D>(5)] = 12;
    }
    catch (const opp::InvalidArgumentException &ex)
    {
        CHECK(ex.Message() == "Out of range: 5. Must be X (0).");
    }
}

TEST_CASE("Pixel16fC1", "[Common]")
{
    // check size:
    CHECK(sizeof(Pixel16fC1) == 1 * sizeof(HalfFp16));

    HalfFp16 arr[4] = {HalfFp16(4)};
    Pixel16fC1 t0(arr);
    CHECK(t0.x == 4);

    Pixel16fC1 t1(HalfFp16(0));
    CHECK(t1.x == 0);

    Pixel16fC1 c(t1);
    CHECK(c.x == 0);
    CHECK(c == t1);

    Pixel16fC1 c2 = t1;
    CHECK(c2.x == 0);
    CHECK(c2 == t1);

    Pixel16fC1 t2(HalfFp16(5));
    CHECK(t2.x == 5);
    CHECK(c2 != t2);

    Pixel16fC1 add1 = t1 + t2;
    CHECK(add1.x == 5);

    Pixel16fC1 add2 = 3 + t1;
    CHECK(add2.x == 3);

    Pixel16fC1 add3 = t1 + 4;
    CHECK(add3.x == 4);

    Pixel16fC1 add4 = t1;
    add4 += add3;
    CHECK(add4.x == 4);

    add4 += HalfFp16(3);
    CHECK(add4.x == 7);

    Pixel16fC1 sub1 = t1 - t2;
    CHECK(sub1.x == -5);

    Pixel16fC1 sub2 = 3 - t1;
    CHECK(sub2.x == 3);

    Pixel16fC1 sub3 = t1 - 4;
    CHECK(sub3.x == -4);

    Pixel16fC1 sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);

    sub4 -= HalfFp16(3);
    CHECK(sub4.x == 1);

    t1              = Pixel16fC1(HalfFp16(4));
    t2              = Pixel16fC1(HalfFp16(6));
    Pixel16fC1 mul1 = t1 * t2;
    CHECK(mul1.x == 24);

    Pixel16fC1 mul2 = 3 * t1;
    CHECK(mul2.x == 12);

    Pixel16fC1 mul3 = t1 * 4;
    CHECK(mul3.x == 16);

    Pixel16fC1 mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);

    mul4 *= HalfFp16(3);
    CHECK(mul4.x == 192);

    Pixel16fC1 div1 = t1 / t2;
    CHECK(div1.x == Approx(0.667).margin(0.001));

    Pixel16fC1 div2 = 3 / t1;
    CHECK(div2.x == Approx(0.75).margin(0.001));

    Pixel16fC1 div3 = t1 / 4;
    CHECK(div3.x == Approx(1.0).margin(0.001));

    Pixel16fC1 div4 = t2;
    div4 /= div3;
    CHECK(div4.x == Approx(6.0).margin(0.001));

    div4 /= HalfFp16(3);
    CHECK(div4.x == Approx(2.0).margin(0.001));

    Pixel16fC1 l(HalfFp16(4));
    CHECK(l.MagnitudeSqr() == 16);
    CHECK(l.Magnitude() == Approx(std::sqrt(16)).margin(0.01));

    Pixel16fC1 minmax1(HalfFp16(10));
    Pixel16fC1 minmax2(HalfFp16(-20));

    CHECK(Pixel16fC1::Min(minmax1, minmax2) == Pixel16fC1(HalfFp16(-20)));
    CHECK(Pixel16fC1::Min(minmax2, minmax1) == Pixel16fC1(HalfFp16(-20)));

    CHECK(Pixel16fC1::Max(minmax1, minmax2) == Pixel16fC1(HalfFp16(10)));
    CHECK(Pixel16fC1::Max(minmax2, minmax1) == Pixel16fC1(HalfFp16(10)));
}

TEST_CASE("Pixel16fC1_additionalMethods", "[Common]")
{
    Pixel16fC1 norm(HalfFp16(4));
    norm.Normalize();
    CHECK(norm.Magnitude() == 1);
    CHECK(norm.x == Approx(1).margin(0.001));

    CHECK((Pixel16fC1::Normalize(Pixel16fC1(HalfFp16(4))) == norm));

    Pixel16fC1 roundA(HalfFp16(0.4f));
    Pixel16fC1 roundB(HalfFp16(1.9f));
    Pixel16fC1 round2A = Pixel16fC1::Round(roundA);
    Pixel16fC1 round2B = Pixel16fC1::Round(roundB);
    roundA.Round();
    roundB.Round();
    CHECK(round2A == roundA);
    CHECK(round2B == roundB);
    CHECK(roundA.x == 0.0f);
    CHECK(roundB.x == 2.0f);

    Pixel16fC1 floorA(HalfFp16(0.4f));
    Pixel16fC1 floorB(HalfFp16(1.9f));
    Pixel16fC1 floor2A = Pixel16fC1::Floor(floorA);
    Pixel16fC1 floor2B = Pixel16fC1::Floor(floorB);
    floorA.Floor();
    floorB.Floor();
    CHECK(floor2A == floorA);
    CHECK(floor2B == floorB);
    CHECK(floorA.x == 0.0f);
    CHECK(floorB.x == 1.0f);

    Pixel16fC1 ceilA(HalfFp16(0.4f));
    Pixel16fC1 ceilB(HalfFp16(1.9f));
    Pixel16fC1 ceil2A = Pixel16fC1::Ceil(ceilA);
    Pixel16fC1 ceil2B = Pixel16fC1::Ceil(ceilB);
    ceilA.Ceil();
    ceilB.Ceil();
    CHECK(ceil2A == ceilA);
    CHECK(ceil2B == ceilB);
    CHECK(ceilA.x == 1.0f);
    CHECK(ceilB.x == 2.0f);

    Pixel16fC1 zeroA(HalfFp16(0.4f));
    Pixel16fC1 zeroB(HalfFp16(1.9f));
    Pixel16fC1 zero2A = Pixel16fC1::RoundZero(zeroA);
    Pixel16fC1 zero2B = Pixel16fC1::RoundZero(zeroB);
    zeroA.RoundZero();
    zeroB.RoundZero();
    CHECK(zero2A == zeroA);
    CHECK(zero2B == zeroB);
    CHECK(zeroA.x == 0.0f);
    CHECK(zeroB.x == 1.0f);

    Pixel16fC1 nearestA(HalfFp16(0.4f));
    Pixel16fC1 nearestB(HalfFp16(1.9f));
    Pixel16fC1 nearest2A = Pixel16fC1::RoundNearest(nearestA);
    Pixel16fC1 nearest2B = Pixel16fC1::RoundNearest(nearestB);
    nearestA.RoundNearest();
    nearestB.RoundNearest();
    CHECK(nearest2A == nearestA);
    CHECK(nearest2B == nearestB);
    CHECK(nearestA.x == 0.0f);
    CHECK(nearestB.x == 2.0f);

    Pixel16fC1 exp(HalfFp16(2.4f));
    Pixel16fC1 exp2 = Pixel16fC1::Exp(exp);
    exp.Exp();
    CHECK(exp == exp2);
    CHECK(exp.x == Approx(11.023176380641601).margin(0.01));

    Pixel16fC1 ln(HalfFp16(2.4f));
    Pixel16fC1 ln2 = Pixel16fC1::Ln(ln);
    ln.Ln();
    CHECK(ln == ln2);
    CHECK(ln.x == Approx(0.875468737353900).margin(0.01));

    Pixel16fC1 sqr(HalfFp16(2.4f));
    Pixel16fC1 sqr2 = Pixel16fC1::Sqr(sqr);
    sqr.Sqr();
    CHECK(sqr == sqr2);
    CHECK(sqr.x == Approx(5.76).margin(0.01));

    Pixel16fC1 sqrt(HalfFp16(2.4f));
    Pixel16fC1 sqrt2 = Pixel16fC1::Sqrt(sqrt);
    sqrt.Sqrt();
    CHECK(sqrt == sqrt2);
    CHECK(sqrt.x == Approx(1.549193338482967).margin(0.01));

    Pixel16fC1 abs(HalfFp16(-2.4f));
    Pixel16fC1 abs2 = Pixel16fC1::Abs(abs);
    abs.Abs();
    CHECK(abs == abs2);
    CHECK(abs.x == HalfFp16(2.4f));

    Pixel16fC1 absdiffA(HalfFp16(13.23592f));
    Pixel16fC1 absdiffB(HalfFp16(45.75068f));
    Pixel16fC1 absdiff2 = Pixel16fC1::AbsDiff(absdiffA, absdiffB);
    Pixel16fC1 absdiff3 = Pixel16fC1::AbsDiff(absdiffB, absdiffA);
    absdiffA.AbsDiff(absdiffB);
    CHECK(absdiffA == absdiff2);
    CHECK(absdiffA == absdiff3);
    CHECK(absdiffA.x == Approx(32.514758920888809).margin(0.02));

    Pixel16fC1 clampByte(HalfFp16(float(numeric_limits<byte>::max()) + 1));
    clampByte.template ClampToTargetType<byte>();
    CHECK(clampByte.x == 255);

    Pixel16fC1 clampShort(HalfFp16(float(numeric_limits<short>::max()) + 1));
    clampShort.template ClampToTargetType<short>();
    CHECK(clampShort.x == 32752);

    Pixel16fC1 clampSByte(HalfFp16(float(numeric_limits<sbyte>::max()) + 1));
    clampSByte.template ClampToTargetType<sbyte>();
    CHECK(clampSByte.x == 127);

    Pixel16fC1 clampUShort(HalfFp16(-10.0f));

    clampUShort.template ClampToTargetType<ushort>();
    CHECK(clampUShort.x == 0);

    CHECK_FALSE(need_saturation_clamp_v<HalfFp16, int>);

    CHECK(need_saturation_clamp_v<HalfFp16, uint>);
    Pixel16fC1 clampUInt(HalfFp16(-10));

    clampUInt.template ClampToTargetType<uint>();
    CHECK(clampUInt.x == 0);

    Pixel32fC1 clampFloatToFp16(float(numeric_limits<int>::min()) - 1000.0f);

    clampFloatToFp16.template ClampToTargetType<HalfFp16>();
    CHECK(clampFloatToFp16.x == -65504);

    Pixel16fC1 fromFromFloat(clampFloatToFp16);
    CHECK(fromFromFloat.x == -65504);
}
