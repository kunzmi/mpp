#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/complex.h>
#include <common/defines.h>
#include <common/safeCast.h>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

using namespace opp;
using namespace Catch;

TEST_CASE("Complex<float>", "[Common]")
{
    // check size:
    CHECK(sizeof(Complex<float>) == 2 * sizeof(float));

    float arr[2] = {4, 5};
    Complex<float> t0(arr);
    CHECK(t0.real == 4);
    CHECK(t0.imag == 5);

    Complex<float> t1(0, 1);
    CHECK(t1.real == 0);
    CHECK(t1.imag == 1);

    Complex<float> c(t1);
    CHECK(c.real == 0);
    CHECK(c.imag == 1);
    CHECK(c == t1);

    Complex<float> c2 = t1;
    CHECK(c2.real == 0);
    CHECK(c2.imag == 1);
    CHECK(c2 == t1);

    Complex<float> t2(5);
    CHECK(t2.real == 5);
    CHECK(t2.imag == 0);
    CHECK(c2 != t2);

    Complex<float> add1 = t1 + t2;
    CHECK(add1.real == 5);
    CHECK(add1.imag == 1);

    Complex<float> add2 = 3 + t1;
    CHECK(add2.real == 3);
    CHECK(add2.imag == 1);

    Complex<float> add3 = t1 + 4;
    CHECK(add3.real == 4);
    CHECK(add3.imag == 1);

    Complex<float> add4 = t1;
    add4 += add3;
    CHECK(add4.real == 4);
    CHECK(add4.imag == 2);

    add4 += 3;
    CHECK(add4.real == 7);
    CHECK(add4.imag == 2);

    Complex<float> sub1 = t1 - t2;
    CHECK(sub1.real == -5);
    CHECK(sub1.imag == 1);

    Complex<float> sub2 = 3 - t1;
    CHECK(sub2.real == 3);
    CHECK(sub2.imag == 1);

    Complex<float> sub3 = t1 - 4;
    CHECK(sub3.real == -4);
    CHECK(sub3.imag == 1);

    Complex<float> sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.real == 4);
    CHECK(sub4.imag == 0);

    sub4 -= 3;
    CHECK(sub4.real == 1);
    CHECK(sub4.imag == 0);

    std::complex<float> ref1(4, 5);
    std::complex<float> ref2(6, 7);
    std::complex<float> refmul1 = ref1 * ref2;
    t1                          = Complex<float>(4, 5);
    t2                          = Complex<float>(6, 7);
    Complex<float> mul1         = t1 * t2;
    CHECK(mul1.real == refmul1.real());
    CHECK(mul1.imag == refmul1.imag());

    std::complex<float> refmul2 = ref1;
    refmul2 *= 3;
    Complex<float> mul2 = 3 * t1;
    CHECK(mul2.real == refmul2.real());
    CHECK(mul2.imag == refmul2.imag());

    std::complex<float> refmul3 = ref1;
    refmul3 *= 4;
    Complex<float> mul3 = t1 * 4;
    CHECK(mul3.real == refmul3.real());
    CHECK(mul3.imag == refmul3.imag());

    std::complex<float> refmul4 = ref1;
    refmul4 *= refmul3;
    Complex<float> mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.real == refmul4.real());
    CHECK(mul4.imag == refmul4.imag());

    refmul4 *= 3;
    mul4 *= 3;
    CHECK(mul4.real == refmul4.real());
    CHECK(mul4.imag == refmul4.imag());

    std::complex<float> refdiv1 = ref1;
    refdiv1 /= ref2;
    Complex<float> div1 = t1 / t2;
    CHECK(div1.real == Approx(refdiv1.real()).margin(0.00001));
    CHECK(div1.imag == Approx(refdiv1.imag()).margin(0.00001));

    std::complex<float> refdiv2 = 3;
    refdiv2 /= ref1;
    Complex<float> div2 = 3 / t1;
    CHECK(div2.real == Approx(refdiv2.real()).margin(0.00001));
    CHECK(div2.imag == Approx(refdiv2.imag()).margin(0.00001));

    std::complex<float> refdiv3 = ref1;
    refdiv3 /= 4;
    Complex<float> div3 = t1 / 4;
    CHECK(div3.real == Approx(refdiv3.real()).margin(0.00001));
    CHECK(div3.imag == Approx(refdiv3.imag()).margin(0.00001));

    std::complex<float> refdiv4 = ref2;
    refdiv4 /= refdiv3;
    Complex<float> div4 = t2;
    div4 /= div3;
    CHECK(div4.real == Approx(refdiv4.real()).margin(0.00001));
    CHECK(div4.imag == Approx(refdiv4.imag()).margin(0.00001));

    refdiv4 /= 3;
    div4 /= 3;
    CHECK(div4.real == Approx(refdiv4.real()).margin(0.00001));
    CHECK(div4.imag == Approx(refdiv4.imag()).margin(0.00001));

    Complex<float> l(4, 6);
    CHECK(l.MagnitudeSqr() == 52);
    CHECK(l.Magnitude() == Approx(std::sqrt(52)));

    Complex<float> minmax1(10, 20);
    Complex<float> minmax2(-20, 10);

    CHECK(minmax1.Min(minmax2) == Complex<float>(-20, 10));
    CHECK(minmax2.Min(minmax1) == Complex<float>(-20, 10));

    CHECK(minmax1.Max(minmax2) == Complex<float>(10, 20));
    CHECK(minmax2.Max(minmax1) == Complex<float>(10, 20));
}

TEST_CASE("Complex<float>_additionalMethods", "[Common]")
{
    Complex<float> norm(4, 5);
    norm.Normalize();
    CHECK(norm.Magnitude() == 1);
    CHECK(norm.real == Approx(0.6247).margin(0.001));
    CHECK(norm.imag == Approx(0.7809).margin(0.001));

    CHECK(Complex<float>::Normalize(Complex<float>(4, 5)) == norm);
}

TEST_CASE("Complex<int>", "[Common]")
{
    // check size:
    CHECK(sizeof(Complex<int>) == 2 * sizeof(int));

    Complex<int> t1(100, 200);
    CHECK(t1.real == 100);
    CHECK(t1.imag == 200);

    Complex<int> c(t1);
    CHECK(c.real == 100);
    CHECK(c.imag == 200);
    CHECK(c == t1);

    Complex<int> c2 = t1;
    CHECK(c2.real == 100);
    CHECK(c2.imag == 200);
    CHECK(c2 == t1);

    Complex<int> t2(5);
    CHECK(t2.real == 5);
    CHECK(t2.imag == 0);
    CHECK(c2 != t2);

    Complex<int> add1 = t1 + t2;
    CHECK(add1.real == 105);
    CHECK(add1.imag == 200);

    Complex<int> add2 = 3 + t1;
    CHECK(add2.real == 103);
    CHECK(add2.imag == 200);

    Complex<int> add3 = t1 + 4;
    CHECK(add3.real == 104);
    CHECK(add3.imag == 200);

    Complex<int> add4 = t1;
    add4 += add3;
    CHECK(add4.real == 204);
    CHECK(add4.imag == 400);

    add4 += 3;
    CHECK(add4.real == 207);
    CHECK(add4.imag == 400);

    Complex<int> sub1 = t1 - t2;
    CHECK(sub1.real == 95);
    CHECK(sub1.imag == 200);

    Complex<int> sub2 = 3 - t1;
    CHECK(sub2.real == -97);
    CHECK(sub2.imag == 200);

    Complex<int> sub3 = t1 - 4;
    CHECK(sub3.real == 96);
    CHECK(sub3.imag == 200);

    Complex<int> sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.real == 4);
    CHECK(sub4.imag == 0);

    sub4 -= 3;
    CHECK(sub4.real == 1);
    CHECK(sub4.imag == 0);

    t1                = Complex<int>(4, 5);
    t2                = Complex<int>(6, 7);
    Complex<int> mul1 = t1 * t2;
    CHECK(mul1.real == -11);
    CHECK(mul1.imag == 58);

    Complex<int> mul2 = 3 * t1;
    CHECK(mul2.real == 12);
    CHECK(mul2.imag == 15);

    Complex<int> mul3 = t1 * 4;
    CHECK(mul3.real == 16);
    CHECK(mul3.imag == 20);

    Complex<int> mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.real == -36);
    CHECK(mul4.imag == 160);

    mul4 *= 3;
    CHECK(mul4.real == -108);
    CHECK(mul4.imag == 480);

    t1                = Complex<int>(1000, 2000);
    t2                = Complex<int>(6, 7);
    Complex<int> div1 = t1 / t2;
    CHECK(div1.real == 235);
    CHECK(div1.imag == 58);

    Complex<int> div2 = 30000 / t1;
    CHECK(div2.real == 6);
    CHECK(div2.imag == -12);

    Complex<int> div3 = t1 / 4;
    CHECK(div3.real == 250);
    CHECK(div3.imag == 500);

    Complex<int> div4 = t2 * 10000;
    div4 /= div3;
    CHECK(div4.real == 160);
    CHECK(div4.imag == -40);

    div4 /= 3;
    CHECK(div4.real == 53);
    CHECK(div4.imag == -13);

    Complex<int> minmax1(10, 20);
    Complex<int> minmax2(-20, 10);

    CHECK(minmax1.Min(minmax2) == Complex<int>(-20, 10));
    CHECK(minmax2.Min(minmax1) == Complex<int>(-20, 10));

    CHECK(minmax1.Max(minmax2) == Complex<int>(10, 20));
    CHECK(minmax2.Max(minmax1) == Complex<int>(10, 20));
}

TEST_CASE("Complex<float>_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Complex<float> *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Complex<float>(float(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->real;
        void *ptrLastMember  = &buffer[i]->imag;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<float *>(ptrVector) + 1));

        // vector must be 8 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 8 == 0);
    }

    std::vector<Complex<float>> buffer2(count);
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

TEST_CASE("Complex<int>_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Complex<int> *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Complex<int>(int(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->real;
        void *ptrLastMember  = &buffer[i]->imag;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<int *>(ptrVector) + 1));

        // vector must be 8 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 8 == 0);
    }

    std::vector<Complex<int>> buffer2(count);
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

TEST_CASE("Complex<ushort>_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Complex<ushort> *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Complex<ushort>(ushort(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->real;
        void *ptrLastMember  = &buffer[i]->imag;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<ushort *>(ptrVector) + 1));

        // vector must be 4 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 4 == 0);
    }

    std::vector<Complex<ushort>> buffer2(count);
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

TEST_CASE("Complex<int>_streams", "[Common]")
{
    std::string str = "3 4";
    std::stringstream ss(str);

    Complex<int> cmplx;
    ss >> cmplx;

    CHECK(cmplx.real == 3);
    CHECK(cmplx.imag == 4);

    std::stringstream ss2;
    ss2 << cmplx;

    CHECK(ss2.str() == "3 + 4i");
}

TEST_CASE("Complex<float>_streams", "[Common]")
{
    std::string str = "3.14 2.7";
    std::stringstream ss(str);

    Complex<float> cmplx;
    ss >> cmplx;

    CHECK(cmplx.real == 3.14f);
    CHECK(cmplx.imag == 2.7f);

    std::stringstream ss2;
    ss2 << cmplx;

    CHECK(ss2.str() == "3.14 + 2.7i");
}
