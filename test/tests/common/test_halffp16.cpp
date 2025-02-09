#include <catch2/catch_test_macros.hpp>
#include <cfloat>
#include <cmath>
#include <common/defines.h>
#include <common/half_fp16.h>
#include <common/numeric_limits.h>
#include <cstddef>
#include <cstdint>
#include <math.h>
#include <numeric>
#include <sstream>

using namespace opp;
using namespace Catch;

// avoid compiler error "division by zero" by making it non-constant...
float getZero(float aVal)
{
    if (aVal == 0.0f)
    {
        return 0.0f;
    }
    return 0.0f;
}

TEST_CASE("HalfFp16", "[Common]")
{
    HalfFp16 bf;

    bf = HalfFp16(0.4f);
    bf.Round();
    CHECK(bf == HalfFp16(0.0f));
    bf = HalfFp16(0.4f);
    bf.Floor();
    CHECK(bf == HalfFp16(0.0f));
    bf = HalfFp16(0.4f);
    bf.Ceil();
    CHECK(bf == HalfFp16(1.0f));
    bf = HalfFp16(0.4f);
    bf.RoundNearest();
    CHECK(bf == HalfFp16(0.0f));
    bf = HalfFp16(0.4f);
    bf.RoundZero();
    CHECK(bf == HalfFp16(0.0f));

    bf = HalfFp16(0.5f);
    bf.Round();
    CHECK(bf == HalfFp16(1.0f));
    bf = HalfFp16(0.5f);
    bf.Floor();
    CHECK(bf == HalfFp16(0.0f));
    bf = HalfFp16(0.5f);
    bf.Ceil();
    CHECK(bf == HalfFp16(1.0f));
    bf = HalfFp16(0.5f);
    bf.RoundNearest();
    CHECK(bf == HalfFp16(0.0f));
    bf = HalfFp16(0.5f);
    bf.RoundZero();
    CHECK(bf == HalfFp16(0.0f));

    bf = HalfFp16(0.6f);
    bf.Round();
    CHECK(bf == HalfFp16(1.0f));
    bf = HalfFp16(0.6f);
    bf.Floor();
    CHECK(bf == HalfFp16(0.0f));
    bf = HalfFp16(0.6f);
    bf.Ceil();
    CHECK(bf == HalfFp16(1.0f));
    bf = HalfFp16(0.6f);
    bf.RoundNearest();
    CHECK(bf == HalfFp16(1.0f));
    bf = HalfFp16(0.6f);
    bf.RoundZero();
    CHECK(bf == HalfFp16(0.0f));

    bf = HalfFp16(1.5f);
    bf.Round();
    CHECK(bf == HalfFp16(2.0f));
    bf = HalfFp16(1.5f);
    bf.Floor();
    CHECK(bf == HalfFp16(1.0f));
    bf = HalfFp16(1.5f);
    bf.Ceil();
    CHECK(bf == HalfFp16(2.0f));
    bf = HalfFp16(1.5f);
    bf.RoundNearest();
    CHECK(bf == HalfFp16(2.0f));
    bf = HalfFp16(1.5f);
    bf.RoundZero();
    CHECK(bf == HalfFp16(1.0f));

    bf = HalfFp16(1.9f);
    bf.Round();
    CHECK(bf == HalfFp16(2.0f));
    bf = HalfFp16(1.9f);
    bf.Floor();
    CHECK(bf == HalfFp16(1.0f));
    bf = HalfFp16(1.9f);
    bf.Ceil();
    CHECK(bf == HalfFp16(2.0f));
    bf = HalfFp16(1.9f);
    bf.RoundNearest();
    CHECK(bf == HalfFp16(2.0f));
    bf = HalfFp16(1.9f);
    bf.RoundZero();
    CHECK(bf == HalfFp16(1.0f));

    bf = HalfFp16(-1.5f);
    bf.Round();
    CHECK(bf == HalfFp16(-2.0f));
    bf = HalfFp16(-1.5f);
    bf.Floor();
    CHECK(bf == HalfFp16(-2.0f));
    bf = HalfFp16(-1.5f);
    bf.Ceil();
    CHECK(bf == HalfFp16(-1.0f));
    bf = HalfFp16(-1.5f);
    bf.RoundNearest();
    CHECK(bf == HalfFp16(-2.0f));
    bf = HalfFp16(-1.5f);
    bf.RoundZero();
    CHECK(bf == HalfFp16(-1.0f));

    bf = HalfFp16(-2.5f);
    bf.Round();
    CHECK(bf == HalfFp16(-3.0f));
    bf = HalfFp16(-2.5f);
    bf.Floor();
    CHECK(bf == HalfFp16(-3.0f));
    bf = HalfFp16(-2.5f);
    bf.Ceil();
    CHECK(bf == HalfFp16(-2.0f));
    bf = HalfFp16(-2.5f);
    bf.RoundNearest();
    CHECK(bf == HalfFp16(-2.0f));
    bf = HalfFp16(-2.5f);
    bf.RoundZero();
    CHECK(bf == HalfFp16(-2.0f));

    CHECK_FALSE(HalfFp16(0.4f) > HalfFp16(0.5f));
    CHECK(HalfFp16(0.5f) > HalfFp16(0.4f));

    CHECK(HalfFp16(0.4f) < HalfFp16(0.5f));
    CHECK_FALSE(HalfFp16(0.5f) < HalfFp16(0.4f));

    CHECK(HalfFp16(0.4f) <= HalfFp16(0.5f));
    CHECK_FALSE(HalfFp16(0.5f) <= HalfFp16(0.4f));
    CHECK(HalfFp16(0.4f) <= HalfFp16(0.4f));

    CHECK_FALSE(HalfFp16(0.4f) >= HalfFp16(0.5f));
    CHECK(HalfFp16(0.5f) >= HalfFp16(0.4f));
    CHECK(HalfFp16(0.4f) >= HalfFp16(0.4f));

    CHECK_FALSE(HalfFp16(0.4f) == HalfFp16(0.5f));
    CHECK(HalfFp16(0.4f) == HalfFp16(0.4f));

    CHECK(HalfFp16(0.4f) != HalfFp16(0.5f));
    CHECK_FALSE(HalfFp16(0.4f) != HalfFp16(0.4f));

    // negate
    bf = -HalfFp16(4.0f);
    CHECK(bf == HalfFp16(-4.0f));
    bf = -HalfFp16(-4.0f);
    CHECK(bf == HalfFp16(4.0f));

    // +
    bf = HalfFp16(0.4f) + HalfFp16(0.5f);
    CHECK(bf == HalfFp16(0.9f));
    // -
    bf = HalfFp16(0.4f) - HalfFp16(0.5f);
    CHECK(bf == HalfFp16(-0.100098f));
    // *
    bf = HalfFp16(0.4f) * HalfFp16(0.5f);
    CHECK(bf == HalfFp16(0.2f));
    // /
    bf = HalfFp16(0.4f) / HalfFp16(0.5f);
    CHECK(bf == HalfFp16(0.8f));

    // +=
    bf = HalfFp16(0.4f);
    bf += HalfFp16(1.4f);
    CHECK(bf == HalfFp16(1.80078f));
    // -=
    bf = HalfFp16(0.4f);
    bf -= HalfFp16(1.4f);
    CHECK(bf == HalfFp16(-1.0f));
    // *=
    bf = HalfFp16(0.4f);
    bf *= HalfFp16(2.0f);
    CHECK(bf == HalfFp16(0.8f));
    // /=
    bf = HalfFp16(0.4f);
    bf /= HalfFp16(2.0f);
    CHECK(bf == HalfFp16(0.2f));

    bf = HalfFp16::Exp(HalfFp16(16.0f));
    CHECK(bf == HalfFp16(std::exp(16.0f)));
    bf = HalfFp16::Ln(HalfFp16(16.0f));
    CHECK(bf == HalfFp16(std::log(16.0f)));
    bf = HalfFp16::Sqrt(HalfFp16(16.0f));
    CHECK(bf == HalfFp16(std::sqrt(16.0f)));
    bf = HalfFp16::Abs(HalfFp16(-16.0f));
    CHECK(bf == HalfFp16(16.0f));
    bf = HalfFp16::Abs(HalfFp16(16.0f));
    CHECK(bf == HalfFp16(16.0f));

    bf = HalfFp16(16.0f);
    bf.Exp();
    CHECK(bf == HalfFp16(std::exp(16.0f)));
    bf = HalfFp16(16.0f);
    bf.Ln();
    CHECK(bf == HalfFp16(std::log(16.0f)));
    bf = HalfFp16(16.0f);
    bf.Sqrt();
    CHECK(bf == HalfFp16(std::sqrt(16.0f)));
    bf = HalfFp16(-16.0f);
    bf.Abs();
    CHECK(bf == HalfFp16(16.0f));
    bf = HalfFp16(16.0f);
    bf.Abs();
    CHECK(bf == HalfFp16(16.0f));

    bf = HalfFp16(16.0f);
    bf = bf.GetSign();
    CHECK(bf == HalfFp16(1.0f));
    bf = HalfFp16(-16.0f);
    bf = bf.GetSign();
    CHECK(bf == HalfFp16(-1.0f));

    bf = HalfFp16(0.0f);
    CHECK(bf == 0.0f);

    bf = HalfFp16(10.0f);
    CHECK(bf == 10.0f);

    bf = HalfFp16(-10.0f);
    CHECK(bf == -10.0f);

    bf = HalfFp16(FLT_MAX);
    CHECK(isinf(bf));

    bf = HalfFp16(-FLT_MAX);
    CHECK(isinf(bf));

    bf = HalfFp16(1.0f / getZero(0.0f));
    CHECK(isinf(bf));

    bf = HalfFp16(sqrtf(-1.0f));
    CHECK(isnan(bf));

    bf = HalfFp16(INFINITY);
    CHECK(isinf(bf));

    bf = HalfFp16(-INFINITY);
    CHECK(isinf(bf));

    bf = HalfFp16(FLT_MIN);
    CHECK(bf == 0.0f);

    bf = HalfFp16(-0.0f);
    CHECK(bf == -0.0f);

    std::stringstream ss;
    ss << HalfFp16(13.256f);
    CHECK(ss.str() == "13.2578");
}

TEST_CASE("HalfFp16 - limits", "[Common]")
{
    CHECK(numeric_limits<HalfFp16>::min() == 6.103515625e-05f);
    CHECK(numeric_limits<HalfFp16>::lowest() == -65504);
    CHECK(numeric_limits<HalfFp16>::max() == 65504);
    CHECK(numeric_limits<HalfFp16>::minExact() == -2048.0f);
    CHECK(numeric_limits<HalfFp16>::maxExact() == 2048.0f);

    // halfFp16 can't store 2049 exactly:
    HalfFp16 moreThanMaxExact(2049.0f);
    HalfFp16 lessThanMinExact(-2049.0f);
    CHECK(moreThanMaxExact == 2048.0f);
    CHECK(lessThanMinExact == -2048.0f);

    // but halfFp16 can store 2050 (now smallest increment is 2):
    moreThanMaxExact = HalfFp16(2050.0f);
    lessThanMinExact = HalfFp16(-2050.0f);
    CHECK(moreThanMaxExact == 2050.0f);
    CHECK(lessThanMinExact == -2050.0f);
}
