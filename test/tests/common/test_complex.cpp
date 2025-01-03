#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/bfloat16.h>
#include <common/complex.h>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/safeCast.h>
#include <common/Vector2.h>
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

    Complex<float> c3 = Vector2<float>(10, 20);
    CHECK(c3.real == 10);
    CHECK(c3.imag == 20);

    Complex<float> c4 = std::complex<float>(10, 20);
    CHECK(c4.real == 10);
    CHECK(c4.imag == 20);

    Complex<float> cEps1(1.0f, -4.0f);
    Complex<float> cEps2(1.001f, -4.001f);
    CHECK(cEps1 != cEps2);
    CHECK(Complex<float>::EqEps(cEps1, cEps2, 0.1f));
    cEps1 = Complex<float>(1.0f, -4.0f);
    cEps2 = Complex<float>(1.001f, -4.001f);
    CHECK(cEps1 != cEps2);
    CHECK_FALSE(Complex<float>::EqEps(cEps1, cEps2, 0.0001f));
    cEps1 = Complex<float>(HUGE_VALF, -4.0f);
    cEps2 = Complex<float>(-HUGE_VALF, -4.001f);
    CHECK(cEps1 != cEps2);
    CHECK(Complex<float>::EqEps(cEps1, cEps2, 0.1f));
    cEps1 = Complex<float>(HUGE_VALF, -4.0f);
    cEps2 = Complex<float>(1.0f, -4.001f);
    CHECK(cEps1 != cEps2);
    CHECK_FALSE(Complex<float>::EqEps(cEps1, cEps2, 0.1f));
    cEps1 = Complex<float>(1.0f, std::numeric_limits<float>::quiet_NaN());
    cEps2 = Complex<float>(1.001f, std::numeric_limits<float>::quiet_NaN());
    CHECK(cEps1 != cEps2);
    CHECK(Complex<float>::EqEps(cEps1, cEps2, 0.1f));
    cEps1 = Complex<float>(1.0f, 4.0f);
    cEps2 = Complex<float>(1.001f, std::numeric_limits<float>::quiet_NaN());
    CHECK(cEps1 != cEps2);
    CHECK_FALSE(Complex<float>::EqEps(cEps1, cEps2, 0.1f));

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
    CHECK(sub2.imag == -1);

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

    std::complex<float> refexp = std::exp(ref1);
    t1                         = Complex<float>(4, 5);
    Complex<float> exp1        = Complex<float>::Exp(t1);
    Complex<float> exp2        = t1.Exp();
    CHECK(exp1.real == refexp.real());
    CHECK(exp1.imag == refexp.imag());
    CHECK(exp2.real == refexp.real());
    CHECK(exp2.imag == refexp.imag());

    std::complex<float> reflog = std::log(ref1);
    t1                         = Complex<float>(4, 5);
    Complex<float> log1        = Complex<float>::Ln(t1);
    Complex<float> log2        = t1.Ln();
    CHECK(log1.real == reflog.real());
    CHECK(log1.imag == reflog.imag());
    CHECK(log2.real == reflog.real());
    CHECK(log2.imag == reflog.imag());

    std::complex<float> refsqrt = std::sqrt(ref1);
    t1                          = Complex<float>(4, 5);
    Complex<float> sqrt1        = Complex<float>::Sqrt(t1);
    Complex<float> sqrt2        = t1.Sqrt();
    CHECK(sqrt1.real == refsqrt.real());
    CHECK(sqrt1.imag == refsqrt.imag());
    CHECK(sqrt2.real == refsqrt.real());
    CHECK(sqrt2.imag == refsqrt.imag());

    Complex<float> minmax1(10, 20);
    Complex<float> minmax2(-20, 10);

    CHECK(minmax1.Min(minmax2) == Complex<float>(-20, 10));
    CHECK(minmax2.Min(minmax1) == Complex<float>(-20, 10));

    minmax1 = Complex<float>(10, 20);
    minmax2 = Complex<float>(-20, 10);
    CHECK(minmax1.Max(minmax2) == Complex<float>(10, 20));
    CHECK(minmax2.Max(minmax1) == Complex<float>(10, 20));

    Complex<float> conj(10, 20);
    Complex<float> conj2 = Complex<float>::Conj(conj);
    conj.Conj();
    CHECK(conj == Complex<float>(10, -20));
    CHECK(conj == conj2);

    Complex<float> conjMulA(10, 20);
    Complex<float> conjMulB(3, -2);
    Complex<float> conjMul = Complex<float>::ConjMul(conjMulA, conjMulB);
    conjMulA.ConjMul(conjMulB);
    CHECK(conjMul == Complex<float>(-10, 80));
    CHECK(conjMulA == conjMul);

    Complex<float> fromInt = Complex<int>(10, 20);
    CHECK(fromInt.real == 10);
    CHECK(fromInt.imag == 20);

    Complex<float> fromLiteral = 10.0f + 20.0_i;
    CHECK(fromLiteral.real == 10);
    CHECK(fromLiteral.imag == 20);

    Complex<float> fromLiteral2 = 10 - 20.0_i;
    CHECK(fromLiteral2.real == 10);
    CHECK(fromLiteral2.imag == -20);

    Complex<float> fromLiteralInt = 10 + 20_i;
    CHECK(fromLiteralInt.real == 10);
    CHECK(fromLiteralInt.imag == 20);

    Complex<float> fromLiteralInt2 = 10.0f + 20_i;
    CHECK(fromLiteralInt2.real == 10);
    CHECK(fromLiteralInt2.imag == 20);

    Complex<float> fromLiteralInt3 = 10 + 20.0_i;
    CHECK(fromLiteralInt3.real == 10);
    CHECK(fromLiteralInt3.imag == 20);
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
    CHECK(sub2.imag == -200);

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

    minmax1 = Complex<int>(10, 20);
    minmax2 = Complex<int>(-20, 10);
    CHECK(minmax1.Max(minmax2) == Complex<int>(10, 20));
    CHECK(minmax2.Max(minmax1) == Complex<int>(10, 20));

    Complex<int> fromFloat = Complex<float>(10.2f, 20.3f);
    CHECK(fromFloat.real == 10);
    CHECK(fromFloat.imag == 20);

    Complex<int> fromLiteral = 10.0f + 20.0_i;
    CHECK(fromLiteral.real == 10);
    CHECK(fromLiteral.imag == 20);

    Complex<int> fromLiteralFloat = 10 + 20_i;
    CHECK(fromLiteralFloat.real == 10);
    CHECK(fromLiteralFloat.imag == 20);

    Complex<int> fromLiteralFloat2 = 10.0f + 20_i;
    CHECK(fromLiteralFloat2.real == 10);
    CHECK(fromLiteralFloat2.imag == 20);

    Complex<int> fromLiteralFloat3 = 10 + 20.0_i;
    CHECK(fromLiteralFloat3.real == 10);
    CHECK(fromLiteralFloat3.imag == 20);
}

TEST_CASE("Complex<short>", "[Common]")
{
    // check size:
    CHECK(sizeof(Complex<short>) == 2 * sizeof(short));

    Complex<short> t1(100, 200);
    CHECK(t1.real == 100);
    CHECK(t1.imag == 200);

    Complex<short> c(t1);
    CHECK(c.real == 100);
    CHECK(c.imag == 200);
    CHECK(c == t1);

    Complex<short> c2 = t1;
    CHECK(c2.real == 100);
    CHECK(c2.imag == 200);
    CHECK(c2 == t1);

    Complex<short> t2(5);
    CHECK(t2.real == 5);
    CHECK(t2.imag == 0);
    CHECK(c2 != t2);

    Complex<short> add1 = t1 + t2;
    CHECK(add1.real == 105);
    CHECK(add1.imag == 200);

    Complex<short> add2 = 3 + t1;
    CHECK(add2.real == 103);
    CHECK(add2.imag == 200);

    Complex<short> add3 = t1 + 4;
    CHECK(add3.real == 104);
    CHECK(add3.imag == 200);

    Complex<short> add4 = t1;
    add4 += add3;
    CHECK(add4.real == 204);
    CHECK(add4.imag == 400);
    Complex<short> aaaa = 3;
    add4 += to_short(3);
    CHECK(add4.real == 207);
    CHECK(add4.imag == 400);

    Complex<short> sub1 = t1 - t2;
    CHECK(sub1.real == 95);
    CHECK(sub1.imag == 200);

    Complex<short> sub2 = 3 - t1;
    CHECK(sub2.real == -97);
    CHECK(sub2.imag == -200);

    Complex<short> sub3 = t1 - 4;
    CHECK(sub3.real == 96);
    CHECK(sub3.imag == 200);

    Complex<short> sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.real == 4);
    CHECK(sub4.imag == 0);

    sub4 -= to_short(3);
    CHECK(sub4.real == 1);
    CHECK(sub4.imag == 0);

    t1                  = Complex<short>(4, 5);
    t2                  = Complex<short>(6, 7);
    Complex<short> mul1 = t1 * t2;
    CHECK(mul1.real == -11);
    CHECK(mul1.imag == 58);

    Complex<short> mul2 = 3 * t1;
    CHECK(mul2.real == 12);
    CHECK(mul2.imag == 15);

    Complex<short> mul3 = t1 * 4;
    CHECK(mul3.real == 16);
    CHECK(mul3.imag == 20);

    Complex<short> mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.real == -36);
    CHECK(mul4.imag == 160);

    mul4 *= to_short(3);
    CHECK(mul4.real == -108);
    CHECK(mul4.imag == 480);

    t1                  = Complex<short>(10, 20);
    t2                  = Complex<short>(6, 7);
    Complex<short> div1 = t1 / t2;
    CHECK(div1.real == 2);
    CHECK(div1.imag == 0);

    Complex<short> div2 = 300 / t1;
    CHECK(div2.real == 6);
    CHECK(div2.imag == -12);

    Complex<short> div3 = t1 / 4;
    CHECK(div3.real == 2);
    CHECK(div3.imag == 5);

    Complex<short> div4 = t2 * 100;
    div4 /= div3;
    CHECK(div4.real == 162);
    CHECK(div4.imag == -55);

    div4 /= to_short(3);
    CHECK(div4.real == 54);
    CHECK(div4.imag == -18);

    Complex<short> minmax1(10, 20);
    Complex<short> minmax2(-20, 10);

    CHECK(minmax1.Min(minmax2) == Complex<short>(-20, 10));
    CHECK(minmax2.Min(minmax1) == Complex<short>(-20, 10));

    minmax1 = Complex<short>(10, 20);
    minmax2 = Complex<short>(-20, 10);
    CHECK(minmax1.Max(minmax2) == Complex<short>(10, 20));
    CHECK(minmax2.Max(minmax1) == Complex<short>(10, 20));

    Complex<short> fromFloat = Complex<float>(10.4f, 20.9f);
    CHECK(fromFloat.real == 10);
    CHECK(fromFloat.imag == 20);

    Complex<short> fromFloatClamp = Complex<float>(SHRT_MAX + 1000, SHRT_MIN - 1000);
    CHECK(fromFloatClamp.real == SHRT_MAX);
    CHECK(fromFloatClamp.imag == SHRT_MIN);

    Complex<short> fromLiteral = 10.0f + 20.0_i;
    CHECK(fromLiteral.real == 10);
    CHECK(fromLiteral.imag == 20);

    Complex<short> fromLiteralFloat = 10 + 20_i;
    CHECK(fromLiteralFloat.real == 10);
    CHECK(fromLiteralFloat.imag == 20);

    Complex<short> fromLiteralFloat2 = 10.0f + 20_i;
    CHECK(fromLiteralFloat2.real == 10);
    CHECK(fromLiteralFloat2.imag == 20);

    Complex<short> fromLiteralFloat3 = 10 + 20.0_i;
    CHECK(fromLiteralFloat3.real == 10);
    CHECK(fromLiteralFloat3.imag == 20);
}

TEST_CASE("Complex<BFloat16>", "[Common]")
{
    // check size:
    CHECK(sizeof(Complex<BFloat16>) == 2 * sizeof(BFloat16));

    BFloat16 arr[2] = {4.0_bf, 5.0_bf};
    Complex<BFloat16> t0(arr);
    CHECK(t0.real == 4);
    CHECK(t0.imag == 5);

    Complex<BFloat16> t1(0.0_bf, 1.0_bf);
    CHECK(t1.real == 0);
    CHECK(t1.imag == 1);

    Complex<BFloat16> c(t1);
    CHECK(c.real == 0);
    CHECK(c.imag == 1);
    CHECK(c == t1);

    Complex<BFloat16> c2 = t1;
    CHECK(c2.real == 0);
    CHECK(c2.imag == 1);
    CHECK(c2 == t1);

    Complex<BFloat16> cEps1(1.0_bf, -4.0_bf);
    Complex<BFloat16> cEps2(1.1_bf, -4.1_bf);
    CHECK(cEps1 != cEps2);
    CHECK(Complex<BFloat16>::EqEps(cEps1, cEps2, 0.2_bf));
    cEps1 = Complex<BFloat16>(1.0_bf, -4.0_bf);
    cEps2 = Complex<BFloat16>(1.1_bf, -4.5_bf);
    CHECK(cEps1 != cEps2);
    CHECK_FALSE(Complex<BFloat16>::EqEps(cEps1, cEps2, 0.2_bf));
    cEps1 = Complex<BFloat16>(numeric_limits<BFloat16>::infinity(), -4.0_bf);
    cEps2 = Complex<BFloat16>(-numeric_limits<BFloat16>::infinity(), -4.1_bf);
    CHECK(cEps1 != cEps2);
    CHECK(Complex<BFloat16>::EqEps(cEps1, cEps2, 0.2_bf));
    cEps1 = Complex<BFloat16>(numeric_limits<BFloat16>::infinity(), -4.0_bf);
    cEps2 = Complex<BFloat16>(1.0_bf, -4.1_bf);
    CHECK(cEps1 != cEps2);
    CHECK_FALSE(Complex<BFloat16>::EqEps(cEps1, cEps2, 0.2_bf));
    cEps1 = Complex<BFloat16>(1.0_bf, numeric_limits<BFloat16>::quiet_NaN());
    cEps2 = Complex<BFloat16>(1.1_bf, numeric_limits<BFloat16>::quiet_NaN());
    CHECK(cEps1 != cEps2);
    CHECK(Complex<BFloat16>::EqEps(cEps1, cEps2, 0.2_bf));
    cEps1 = Complex<BFloat16>(1.0_bf, 4.0_bf);
    cEps2 = Complex<BFloat16>(1.1_bf, numeric_limits<BFloat16>::quiet_NaN());
    CHECK(cEps1 != cEps2);
    CHECK_FALSE(Complex<BFloat16>::EqEps(cEps1, cEps2, 0.2_bf));

    Complex<BFloat16> c3 = Vector2<BFloat16>(10.0_bf, 20.0_bf);
    CHECK(c3.real == 10);
    CHECK(c3.imag == 20);

    Complex<BFloat16> t2(5.0_bf);
    CHECK(t2.real == 5);
    CHECK(t2.imag == 0);
    CHECK(c2 != t2);

    Complex<BFloat16> add1 = t1 + t2;
    CHECK(add1.real == 5);
    CHECK(add1.imag == 1);

    Complex<BFloat16> add2 = 3 + t1;
    CHECK(add2.real == 3);
    CHECK(add2.imag == 1);

    Complex<BFloat16> add3 = t1 + 4;
    CHECK(add3.real == 4);
    CHECK(add3.imag == 1);

    Complex<BFloat16> add4 = t1;
    add4 += add3;
    CHECK(add4.real == 4);
    CHECK(add4.imag == 2);

    add4 += 3.0_bf;
    CHECK(add4.real == 7);
    CHECK(add4.imag == 2);

    Complex<BFloat16> sub1 = t1 - t2;
    CHECK(sub1.real == -5);
    CHECK(sub1.imag == 1);

    Complex<BFloat16> sub2 = 3 - t1;
    CHECK(sub2.real == 3);
    CHECK(sub2.imag == -1);

    Complex<BFloat16> sub3 = t1 - 4;
    CHECK(sub3.real == -4);
    CHECK(sub3.imag == 1);

    Complex<BFloat16> sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.real == 4);
    CHECK(sub4.imag == 0);

    sub4 -= 3.0_bf;
    CHECK(sub4.real == 1);
    CHECK(sub4.imag == 0);

    std::complex<float> ref1(4.0f, 5.0f);
    std::complex<float> ref2(6.0f, 7.0f);
    std::complex<float> refmul1 = ref1 * ref2;
    t1                          = Complex<BFloat16>(4.0_bf, 5.0_bf);
    t2                          = Complex<BFloat16>(6.0_bf, 7.0_bf);
    Complex<BFloat16> mul1      = t1 * t2;
    CHECK(mul1.real == refmul1.real());
    CHECK(mul1.imag == refmul1.imag());

    std::complex<float> refmul2 = ref1;
    refmul2 *= 3.0_bf;
    Complex<BFloat16> mul2 = 3 * t1;
    CHECK(mul2.real == refmul2.real());
    CHECK(mul2.imag == refmul2.imag());

    std::complex<float> refmul3 = ref1;
    refmul3 *= 4.0_bf;
    Complex<BFloat16> mul3 = t1 * 4;
    CHECK(mul3.real == refmul3.real());
    CHECK(mul3.imag == refmul3.imag());

    std::complex<float> refmul4 = ref1;
    refmul4 *= refmul3;
    Complex<BFloat16> mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.real == refmul4.real());
    CHECK(mul4.imag == refmul4.imag());

    refmul4 *= 3.0_bf;
    mul4 *= 3.0_bf;
    CHECK(mul4.real == refmul4.real());
    CHECK(mul4.imag == refmul4.imag());

    std::complex<float> refdiv1 = ref1;
    refdiv1 /= ref2;
    Complex<BFloat16> div1 = t1 / t2;
    CHECK(div1.real == Approx(refdiv1.real()).margin(0.005));
    CHECK(div1.imag == Approx(refdiv1.imag()).margin(0.001));

    std::complex<float> refdiv2 = 3.0f;
    refdiv2 /= ref1;
    Complex<BFloat16> div2 = 3.0_bf / t1;
    CHECK(div2.real == Approx(refdiv2.real()).margin(0.001));
    CHECK(div2.imag == Approx(refdiv2.imag()).margin(0.001));

    std::complex<float> refdiv3 = ref1;
    refdiv3 /= 4.0_bf;
    Complex<BFloat16> div3 = t1 / 4;
    CHECK(div3.real == Approx(refdiv3.real()).margin(0.001));
    CHECK(div3.imag == Approx(refdiv3.imag()).margin(0.001));

    std::complex<float> refdiv4 = ref2;
    refdiv4 /= refdiv3;
    Complex<BFloat16> div4 = t2;
    div4 /= div3;
    CHECK(div4.real == Approx(refdiv4.real()).margin(0.01));
    CHECK(div4.imag == Approx(refdiv4.imag()).margin(0.001));

    refdiv4 /= 3.0_bf;
    div4 /= 3.0_bf;
    CHECK(div4.real == Approx(refdiv4.real()).margin(0.005));
    CHECK(div4.imag == Approx(refdiv4.imag()).margin(0.001));

    Complex<BFloat16> l(4.0_bf, 6.0_bf);
    CHECK(l.MagnitudeSqr() == 52);
    CHECK(l.Magnitude() == Approx(std::sqrt(52)).margin(0.01));

    Complex<BFloat16> minmax1(10.0_bf, 20.0_bf);
    Complex<BFloat16> minmax2(-20.0_bf, 10.0_bf);

    CHECK((minmax1.Min(minmax2)) == (Complex<BFloat16>(-20.0_bf, 10.0_bf)));
    CHECK(minmax2.Min(minmax1) == Complex<BFloat16>(-20.0_bf, 10.0_bf));

    minmax1 = Complex<BFloat16>(10.0_bf, 20.0_bf);
    minmax2 = Complex<BFloat16>(-20.0_bf, 10.0_bf);
    CHECK(minmax1.Max(minmax2) == Complex<BFloat16>(10.0_bf, 20.0_bf));
    CHECK(minmax2.Max(minmax1) == Complex<BFloat16>(10.0_bf, 20.0_bf));

    Complex<BFloat16> conj(10.0_bf, 20.0_bf);
    Complex<BFloat16> conj2 = Complex<BFloat16>::Conj(conj);
    conj.Conj();
    CHECK(conj == Complex<BFloat16>(10.0_bf, -20.0_bf));
    CHECK(conj == conj2);

    Complex<BFloat16> conjMulA(10.0_bf, 20.0_bf);
    Complex<BFloat16> conjMulB(3.0_bf, -2.0_bf);
    Complex<BFloat16> conjMul = Complex<BFloat16>::ConjMul(conjMulA, conjMulB);
    conjMulA.ConjMul(conjMulB);
    CHECK(conjMul == Complex<BFloat16>(-10.0_bf, 80.0_bf));
    CHECK(conjMulA == conjMul);

    Complex<BFloat16> fromInt = Complex<int>(10, 20);
    CHECK(fromInt.real == 10);
    CHECK(fromInt.imag == 20);

    Complex<BFloat16> fromLiteral = 10.0f + 20.0_ib;
    CHECK(fromLiteral.real == 10);
    CHECK(fromLiteral.imag == 20);

    Complex<BFloat16> fromLiteral2 = 10 + 20.0_ib;
    CHECK(fromLiteral2.real == 10);
    CHECK(fromLiteral2.imag == 20);

    Complex<BFloat16> fromLiteral3 = 10 - 20.0_ib;
    CHECK(fromLiteral3.real == 10);
    CHECK(fromLiteral3.imag == -20);

    Complex<BFloat16> fromLiteralFloat = 10 + 20_i;
    CHECK(fromLiteralFloat.real == 10);
    CHECK(fromLiteralFloat.imag == 20);

    Complex<BFloat16> fromLiteralFloat2 = 10.0f + 20_i;
    CHECK(fromLiteralFloat2.real == 10);
    CHECK(fromLiteralFloat2.imag == 20);

    Complex<BFloat16> fromLiteralFloat3 = 10 + 20.0_i;
    CHECK(fromLiteralFloat3.real == 10);
    CHECK(fromLiteralFloat3.imag == 20);
}

TEST_CASE("Complex<HalfFp16>", "[Common]")
{
    // check size:
    CHECK(sizeof(Complex<HalfFp16>) == 2 * sizeof(HalfFp16));

    HalfFp16 arr[2] = {4.0_hf, 5.0_hf};
    Complex<HalfFp16> t0(arr);
    CHECK(t0.real == 4);
    CHECK(t0.imag == 5);

    Complex<HalfFp16> t1(0.0_hf, 1.0_hf);
    CHECK(t1.real == 0);
    CHECK(t1.imag == 1);

    Complex<HalfFp16> c(t1);
    CHECK(c.real == 0);
    CHECK(c.imag == 1);
    CHECK(c == t1);

    Complex<HalfFp16> c2 = t1;
    CHECK(c2.real == 0);
    CHECK(c2.imag == 1);
    CHECK(c2 == t1);

    Complex<HalfFp16> c3 = Vector2<HalfFp16>(10.0_hf, 20.0_hf);
    CHECK(c3.real == 10);
    CHECK(c3.imag == 20);

    Complex<HalfFp16> cEps1(1.0_hf, -4.0_hf);
    Complex<HalfFp16> cEps2(1.001_hf, -4.001_hf);
    CHECK(cEps1 != cEps2);
    CHECK(Complex<HalfFp16>::EqEps(cEps1, cEps2, 0.1_hf));
    cEps1 = Complex<HalfFp16>(1.0_hf, -4.0_hf);
    cEps2 = Complex<HalfFp16>(1.001_hf, -4.001_hf);
    CHECK(cEps1 != cEps2);
    CHECK_FALSE(Complex<HalfFp16>::EqEps(cEps1, cEps2, 0.0001_hf));
    cEps1 = Complex<HalfFp16>(numeric_limits<HalfFp16>::infinity(), -4.0_hf);
    cEps2 = Complex<HalfFp16>(-numeric_limits<HalfFp16>::infinity(), -4.001_hf);
    CHECK(cEps1 != cEps2);
    CHECK(Complex<HalfFp16>::EqEps(cEps1, cEps2, 0.1_hf));
    cEps1 = Complex<HalfFp16>(numeric_limits<HalfFp16>::infinity(), -4.0_hf);
    cEps2 = Complex<HalfFp16>(1.0_hf, -4.001_hf);
    CHECK(cEps1 != cEps2);
    CHECK_FALSE(Complex<HalfFp16>::EqEps(cEps1, cEps2, 0.1_hf));
    cEps1 = Complex<HalfFp16>(1.0_hf, numeric_limits<HalfFp16>::quiet_NaN());
    cEps2 = Complex<HalfFp16>(1.001_hf, numeric_limits<HalfFp16>::quiet_NaN());
    CHECK(cEps1 != cEps2);
    CHECK(Complex<HalfFp16>::EqEps(cEps1, cEps2, 0.1_hf));
    cEps1 = Complex<HalfFp16>(1.0_hf, 4.0_hf);
    cEps2 = Complex<HalfFp16>(1.001_hf, numeric_limits<HalfFp16>::quiet_NaN());
    CHECK(cEps1 != cEps2);
    CHECK_FALSE(Complex<HalfFp16>::EqEps(cEps1, cEps2, 0.1_hf));

    Complex<HalfFp16> t2(5.0_hf);
    CHECK(t2.real == 5);
    CHECK(t2.imag == 0);
    CHECK(c2 != t2);

    Complex<HalfFp16> add1 = t1 + t2;
    CHECK(add1.real == 5);
    CHECK(add1.imag == 1);

    Complex<HalfFp16> add2 = 3.0_hf + t1;
    CHECK(add2.real == 3);
    CHECK(add2.imag == 1);

    Complex<HalfFp16> add3 = t1 + 4;
    CHECK(add3.real == 4);
    CHECK(add3.imag == 1);

    Complex<HalfFp16> add4 = t1;
    add4 += add3;
    CHECK(add4.real == 4);
    CHECK(add4.imag == 2);

    add4 += 3.0_hf;
    CHECK(add4.real == 7);
    CHECK(add4.imag == 2);

    Complex<HalfFp16> sub1 = t1 - t2;
    CHECK(sub1.real == -5);
    CHECK(sub1.imag == 1);

    Complex<HalfFp16> sub2 = 3 - t1;
    CHECK(sub2.real == 3);
    CHECK(sub2.imag == -1);

    Complex<HalfFp16> sub3 = t1 - 4;
    CHECK(sub3.real == -4);
    CHECK(sub3.imag == 1);

    Complex<HalfFp16> sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.real == 4);
    CHECK(sub4.imag == 0);

    sub4 -= 3.0_hf;
    CHECK(sub4.real == 1);
    CHECK(sub4.imag == 0);

    std::complex<float> ref1(4, 5);
    std::complex<float> ref2(6, 7);
    std::complex<float> refmul1 = ref1 * ref2;
    t1                          = Complex<HalfFp16>(4.0_hf, 5.0_hf);
    t2                          = Complex<HalfFp16>(6.0_hf, 7.0_hf);
    Complex<HalfFp16> mul1      = t1 * t2;
    CHECK(mul1.real == refmul1.real());
    CHECK(mul1.imag == refmul1.imag());

    std::complex<float> refmul2 = ref1;
    refmul2 *= 3.0_hf;
    Complex<HalfFp16> mul2 = 3 * t1;
    CHECK(mul2.real == refmul2.real());
    CHECK(mul2.imag == refmul2.imag());

    std::complex<float> refmul3 = ref1;
    refmul3 *= 4.0_hf;
    Complex<HalfFp16> mul3 = t1 * 4;
    CHECK(mul3.real == refmul3.real());
    CHECK(mul3.imag == refmul3.imag());

    std::complex<float> refmul4 = ref1;
    refmul4 *= refmul3;
    Complex<HalfFp16> mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.real == refmul4.real());
    CHECK(mul4.imag == refmul4.imag());

    refmul4 *= 3.0_hf;
    mul4 *= 3.0_hf;
    CHECK(mul4.real == refmul4.real());
    CHECK(mul4.imag == refmul4.imag());

    std::complex<float> refdiv1 = ref1;
    refdiv1 /= ref2;
    Complex<HalfFp16> div1 = t1 / t2;
    CHECK(div1.real == Approx(refdiv1.real()).margin(0.001));
    CHECK(div1.imag == Approx(refdiv1.imag()).margin(0.001));

    std::complex<float> refdiv2 = 3.0f;
    refdiv2 /= ref1;
    Complex<HalfFp16> div2 = 3 / t1;
    CHECK(div2.real == Approx(refdiv2.real()).margin(0.001));
    CHECK(div2.imag == Approx(refdiv2.imag()).margin(0.001));

    std::complex<float> refdiv3 = ref1;
    refdiv3 /= 4.0_hf;
    Complex<HalfFp16> div3 = t1 / 4;
    CHECK(div3.real == Approx(refdiv3.real()).margin(0.001));
    CHECK(div3.imag == Approx(refdiv3.imag()).margin(0.001));

    std::complex<float> refdiv4 = ref2;
    refdiv4 /= refdiv3;
    Complex<HalfFp16> div4 = t2;
    div4 /= div3;
    CHECK(div4.real == Approx(refdiv4.real()).margin(0.002));
    CHECK(div4.imag == Approx(refdiv4.imag()).margin(0.001));

    refdiv4 /= 3.0_hf;
    div4 /= 3.0_hf;
    CHECK(div4.real == Approx(refdiv4.real()).margin(0.001));
    CHECK(div4.imag == Approx(refdiv4.imag()).margin(0.001));

    Complex<HalfFp16> l(4.0_hf, 6.0_hf);
    CHECK(l.MagnitudeSqr() == 52);
    CHECK(l.Magnitude() == Approx(std::sqrt(52)).margin(0.002));

    Complex<HalfFp16> minmax1(10.0_hf, 20.0_hf);
    Complex<HalfFp16> minmax2(-20.0_hf, 10.0_hf);

    CHECK(minmax1.Min(minmax2) == Complex<HalfFp16>(-20.0_hf, 10.0_hf));
    CHECK(minmax2.Min(minmax1) == Complex<HalfFp16>(-20.0_hf, 10.0_hf));

    minmax1 = Complex<HalfFp16>(10.0_hf, 20.0_hf);
    minmax2 = Complex<HalfFp16>(-20.0_hf, 10.0_hf);
    CHECK(minmax1.Max(minmax2) == Complex<HalfFp16>(10.0_hf, 20.0_hf));
    CHECK(minmax2.Max(minmax1) == Complex<HalfFp16>(10.0_hf, 20.0_hf));

    Complex<HalfFp16> conj(10.0_hf, 20.0_hf);
    Complex<HalfFp16> conj2 = Complex<HalfFp16>::Conj(conj);
    conj.Conj();
    CHECK(conj == Complex<HalfFp16>(10.0_hf, -20.0_hf));
    CHECK(conj == conj2);

    Complex<HalfFp16> conjMulA(10.0_hf, 20.0_hf);
    Complex<HalfFp16> conjMulB(3.0_hf, -2.0_hf);
    Complex<HalfFp16> conjMul = Complex<HalfFp16>::ConjMul(conjMulA, conjMulB);
    conjMulA.ConjMul(conjMulB);
    CHECK(conjMul == Complex<HalfFp16>(-10.0_hf, 80.0_hf));
    CHECK(conjMulA == conjMul);

    Complex<HalfFp16> fromInt = Complex<int>(10, 20);
    CHECK(fromInt.real == 10);
    CHECK(fromInt.imag == 20);

    Complex<HalfFp16> fromLiteral = 10.0f + 20.0_ih;
    CHECK(fromLiteral.real == 10);
    CHECK(fromLiteral.imag == 20);

    Complex<HalfFp16> fromLiteral2 = 10 + 20.0_ih;
    CHECK(fromLiteral2.real == 10);
    CHECK(fromLiteral2.imag == 20);

    Complex<HalfFp16> fromLiteral3 = 10 - 20.0_ih;
    CHECK(fromLiteral3.real == 10);
    CHECK(fromLiteral3.imag == -20);

    Complex<HalfFp16> fromLiteralFloat = 10 + 20_i;
    CHECK(fromLiteralFloat.real == 10);
    CHECK(fromLiteralFloat.imag == 20);

    Complex<HalfFp16> fromLiteralFloat2 = 10.0f + 20_i;
    CHECK(fromLiteralFloat2.real == 10);
    CHECK(fromLiteralFloat2.imag == 20);

    Complex<HalfFp16> fromLiteralFloat3 = 10 + 20.0_i;
    CHECK(fromLiteralFloat3.real == 10);
    CHECK(fromLiteralFloat3.imag == 20);
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

TEST_CASE("Complex<short>_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Complex<short> *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Complex<short>(short(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->real;
        void *ptrLastMember  = &buffer[i]->imag;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<short *>(ptrVector) + 1));

        // vector must be 4 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 4 == 0);
    }

    std::vector<Complex<short>> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (2 * sizeof(short))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Complex<BFloat16>_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Complex<BFloat16> *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Complex<BFloat16>(BFloat16(i));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->real;
        void *ptrLastMember  = &buffer[i]->imag;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<BFloat16 *>(ptrVector) + 1));

        // vector must be 4 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 4 == 0);
    }

    std::vector<Complex<BFloat16>> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (2 * sizeof(BFloat16))));
    }

    for (size_t i = 0; i < count; i++)
    {
        delete buffer[i];
    }
}

TEST_CASE("Complex<HalfFp16>_alignment", "[Common]")
{
    constexpr int count = 10;
    std::vector<Complex<HalfFp16> *> buffer(count);

    for (size_t i = 0; i < count; i++)
    {
        buffer[i] = new Complex<HalfFp16>(HalfFp16(to_int(i)));
    }

    for (size_t i = 0; i < count; i++)
    {
        void *ptrFirstMember = &buffer[i]->real;
        void *ptrLastMember  = &buffer[i]->imag;
        void *ptrVector      = buffer[i];

        CHECK(ptrFirstMember == ptrVector);
        CHECK(ptrLastMember == (reinterpret_cast<HalfFp16 *>(ptrVector) + 1));

        // vector must be 4 byte memory aligned:
        CHECK(std::int64_t(ptrVector) % 4 == 0);
    }

    std::vector<Complex<HalfFp16>> buffer2(count);
    for (size_t i = 0; i < count - 1; i++)
    {
        void *ptrVector1 = &buffer2[i];
        void *ptrVector2 = &buffer2[i + 1];

        CHECK(ptrVector2 == reinterpret_cast<void *>(reinterpret_cast<char *>(ptrVector1) + (2 * sizeof(HalfFp16))));
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

    CHECK(ss2.str() == "(3 + 4i)");
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

    CHECK(ss2.str() == "(3.14 + 2.7i)");
}

TEST_CASE("Complex<BFloat16>_streams", "[Common]")
{
    std::string str = "3.14 2.7";
    std::stringstream ss(str);

    Complex<BFloat16> cmplx;
    ss >> cmplx;

    CHECK(cmplx.real == Approx(3.14).margin(0.005));
    CHECK(cmplx.imag == Approx(2.7).margin(0.005));

    std::stringstream ss2;
    ss2 << cmplx;

    CHECK(ss2.str() == "(3.14062 + 2.70312i)");
}

TEST_CASE("Complex<HalfFp16>_streams", "[Common]")
{
    std::string str = "3.14 2.7";
    std::stringstream ss(str);

    Complex<HalfFp16> cmplx;
    ss >> cmplx;

    CHECK(cmplx.real == Approx(3.14).margin(0.005));
    CHECK(cmplx.imag == Approx(2.7).margin(0.005));

    std::stringstream ss2;
    ss2 << cmplx;

    CHECK(ss2.str() == "(3.14062 + 2.69922i)");
}
