#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/bfloat16.h>
#include <common/defines.h>
#include <common/numeric_limits.h>
#include <cstddef>
#include <cstdint>
#include <math.h>
#include <numeric>
#include <vector>

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

TEST_CASE("BFloat16", "[Common]")
{
    BFloat16 bf;

    bf = BFloat16(0.4f);
    bf.Round();
    CHECK(bf == BFloat16(0.0f));
    bf = BFloat16(0.4f);
    bf.Floor();
    CHECK(bf == BFloat16(0.0f));
    bf = BFloat16(0.4f);
    bf.Ceil();
    CHECK(bf == BFloat16(1.0f));
    bf = BFloat16(0.4f);
    bf.RoundNearest();
    CHECK(bf == BFloat16(0.0f));
    bf = BFloat16(0.4f);
    bf.RoundZero();
    CHECK(bf == BFloat16(0.0f));

    bf = BFloat16(0.5f);
    bf.Round();
    CHECK(bf == BFloat16(1.0f));
    bf = BFloat16(0.5f);
    bf.Floor();
    CHECK(bf == BFloat16(0.0f));
    bf = BFloat16(0.5f);
    bf.Ceil();
    CHECK(bf == BFloat16(1.0f));
    bf = BFloat16(0.5f);
    bf.RoundNearest();
    CHECK(bf == BFloat16(0.0f));
    bf = BFloat16(0.5f);
    bf.RoundZero();
    CHECK(bf == BFloat16(0.0f));

    bf = BFloat16(0.6f);
    bf.Round();
    CHECK(bf == BFloat16(1.0f));
    bf = BFloat16(0.6f);
    bf.Floor();
    CHECK(bf == BFloat16(0.0f));
    bf = BFloat16(0.6f);
    bf.Ceil();
    CHECK(bf == BFloat16(1.0f));
    bf = BFloat16(0.6f);
    bf.RoundNearest();
    CHECK(bf == BFloat16(1.0f));
    bf = BFloat16(0.6f);
    bf.RoundZero();
    CHECK(bf == BFloat16(0.0f));

    bf = BFloat16(1.5f);
    bf.Round();
    CHECK(bf == BFloat16(2.0f));
    bf = BFloat16(1.5f);
    bf.Floor();
    CHECK(bf == BFloat16(1.0f));
    bf = BFloat16(1.5f);
    bf.Ceil();
    CHECK(bf == BFloat16(2.0f));
    bf = BFloat16(1.5f);
    bf.RoundNearest();
    CHECK(bf == BFloat16(2.0f));
    bf = BFloat16(1.5f);
    bf.RoundZero();
    CHECK(bf == BFloat16(1.0f));

    bf = BFloat16(1.9f);
    bf.Round();
    CHECK(bf == BFloat16(2.0f));
    bf = BFloat16(1.9f);
    bf.Floor();
    CHECK(bf == BFloat16(1.0f));
    bf = BFloat16(1.9f);
    bf.Ceil();
    CHECK(bf == BFloat16(2.0f));
    bf = BFloat16(1.9f);
    bf.RoundNearest();
    CHECK(bf == BFloat16(2.0f));
    bf = BFloat16(1.9f);
    bf.RoundZero();
    CHECK(bf == BFloat16(1.0f));

    bf = BFloat16(-1.5f);
    bf.Round();
    CHECK(bf == BFloat16(-2.0f));
    bf = BFloat16(-1.5f);
    bf.Floor();
    CHECK(bf == BFloat16(-2.0f));
    bf = BFloat16(-1.5f);
    bf.Ceil();
    CHECK(bf == BFloat16(-1.0f));
    bf = BFloat16(-1.5f);
    bf.RoundNearest();
    CHECK(bf == BFloat16(-2.0f));
    bf = BFloat16(-1.5f);
    bf.RoundZero();
    CHECK(bf == BFloat16(-1.0f));

    bf = BFloat16(-2.5f);
    bf.Round();
    CHECK(bf == BFloat16(-3.0f));
    bf = BFloat16(-2.5f);
    bf.Floor();
    CHECK(bf == BFloat16(-3.0f));
    bf = BFloat16(-2.5f);
    bf.Ceil();
    CHECK(bf == BFloat16(-2.0f));
    bf = BFloat16(-2.5f);
    bf.RoundNearest();
    CHECK(bf == BFloat16(-2.0f));
    bf = BFloat16(-2.5f);
    bf.RoundZero();
    CHECK(bf == BFloat16(-2.0f));

    CHECK_FALSE(BFloat16(0.4f) > BFloat16(0.5f));
    CHECK(BFloat16(0.5f) > BFloat16(0.4f));

    CHECK(BFloat16(0.4f) < BFloat16(0.5f));
    CHECK_FALSE(BFloat16(0.5f) < BFloat16(0.4f));

    CHECK(BFloat16(0.4f) <= BFloat16(0.5f));
    CHECK_FALSE(BFloat16(0.5f) <= BFloat16(0.4f));
    CHECK(BFloat16(0.4f) <= BFloat16(0.4f));

    CHECK_FALSE(BFloat16(0.4f) >= BFloat16(0.5f));
    CHECK(BFloat16(0.5f) >= BFloat16(0.4f));
    CHECK(BFloat16(0.4f) >= BFloat16(0.4f));

    CHECK_FALSE(BFloat16(0.4f) == BFloat16(0.5f));
    CHECK(BFloat16(0.4f) == BFloat16(0.4f));

    CHECK(BFloat16(0.4f) != BFloat16(0.5f));
    CHECK_FALSE(BFloat16(0.4f) != BFloat16(0.4f));

    // negate
    bf = -BFloat16(4.0f);
    CHECK(bf == BFloat16(-4.0f));
    bf = -BFloat16(-4.0f);
    CHECK(bf == BFloat16(4.0f));

    // +
    bf = BFloat16(0.4f) + BFloat16(0.5f);
    CHECK(bf == BFloat16(0.9f));
    // -
    bf = BFloat16(0.4f) - BFloat16(0.5f);
    CHECK(BFloat16::Abs(bf - BFloat16(-0.1f)) < BFloat16(0.0005f));
    // *
    bf = BFloat16(0.4f) * BFloat16(0.5f);
    CHECK(bf == BFloat16(0.2f));
    // /
    bf = BFloat16(0.4f) / BFloat16(0.5f);
    CHECK(bf == BFloat16(0.8f));

    // +=
    bf = BFloat16(0.4f);
    bf += BFloat16(1.4f);
    CHECK(bf == BFloat16(1.8f));
    // -=
    bf = BFloat16(0.4f);
    bf -= BFloat16(1.4f);
    CHECK(bf == BFloat16(-1.0f));
    // *=
    bf = BFloat16(0.4f);
    bf *= BFloat16(2.0f);
    CHECK(bf == BFloat16(0.8f));
    // /=
    bf = BFloat16(0.4f);
    bf /= BFloat16(2.0f);
    CHECK(bf == BFloat16(0.2f));

    bf = BFloat16::Exp(BFloat16(16.0f));
    CHECK(bf == BFloat16(std::exp(16.0f)));
    bf = BFloat16::Ln(BFloat16(16.0f));
    CHECK(bf == BFloat16(std::log(16.0f)));
    bf = BFloat16::Sqrt(BFloat16(16.0f));
    CHECK(bf == BFloat16(std::sqrt(16.0f)));
    bf = BFloat16::Abs(BFloat16(-16.0f));
    CHECK(bf == BFloat16(16.0f));
    bf = BFloat16::Abs(BFloat16(16.0f));
    CHECK(bf == BFloat16(16.0f));

    bf = BFloat16(16.0f);
    bf.Exp();
    CHECK(bf == BFloat16(std::exp(16.0f)));
    bf = BFloat16(16.0f);
    bf.Ln();
    CHECK(bf == BFloat16(std::log(16.0f)));
    bf = BFloat16(16.0f);
    bf.Sqrt();
    CHECK(bf == BFloat16(std::sqrt(16.0f)));
    bf = BFloat16(-16.0f);
    bf.Abs();
    CHECK(bf == BFloat16(16.0f));
    bf = BFloat16(16.0f);
    bf.Abs();
    CHECK(bf == BFloat16(16.0f));

    bf = BFloat16(0.0f);
    CHECK(bf == 0.0f);

    bf = BFloat16(10.0f);
    CHECK(bf == 10.0f);

    bf = BFloat16(-10.0f);
    CHECK(bf == -10.0f);

    bf = BFloat16(FLT_MAX);
    CHECK(isinf(bf));

    bf = BFloat16(-FLT_MAX);
    CHECK(isinf(bf));

    bf = BFloat16(1.0f / getZero(0.0f));
    CHECK(isinf(bf));

    bf = BFloat16(sqrtf(-1.0f));
    CHECK(isnan(bf));

    bf = BFloat16(INFINITY);
    CHECK(isinf(bf));

    bf = BFloat16(-INFINITY);
    CHECK(isinf(bf));

    bf = BFloat16(FLT_MIN);
    CHECK(bf == FLT_MIN);

    bf = BFloat16(-0.0f);
    CHECK(bf == -0.0f);

    std::stringstream ss;
    ss << BFloat16(13.256f);
    CHECK(ss.str() == "13.25");
}

TEST_CASE("BFloat16 - limits", "[Common]")
{
    CHECK(numeric_limits<BFloat16>::min() == 1.5046327690525280102e-36f);
    CHECK(numeric_limits<BFloat16>::lowest() == -3.3895313892515354759e+38f);
    CHECK(numeric_limits<BFloat16>::max() == 3.3895313892515354759e+38f);
    CHECK(numeric_limits<BFloat16>::minExact() == -256.0f);
    CHECK(numeric_limits<BFloat16>::maxExact() == 256.0f);

    // BFloat16 can't store 257 exactly:
    BFloat16 moreThanMaxExact(257.0f);
    BFloat16 lessThanMinExact(-257.0f);
    CHECK(moreThanMaxExact == 256.0f);
    CHECK(lessThanMinExact == -256.0f);

    // but BFloat16 can store 258 (now smallest inrcement is 2):
    moreThanMaxExact = BFloat16(258.0f);
    lessThanMinExact = BFloat16(-258.0f);
    CHECK(moreThanMaxExact == 258.0f);
    CHECK(lessThanMinExact == -258.0f);
}
