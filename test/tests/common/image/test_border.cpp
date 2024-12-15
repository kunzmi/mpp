#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/border.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <sstream>
#include <vector>

using namespace opp;
using namespace opp::image;
using namespace Catch;

TEST_CASE("Border", "[Common.Image]")
{
    int vals[] = {10, 20, 30, 40};
    Border b1(vals);

    Vec4i vec4(10, 20, 30, 40);
    Border b2(vec4);

    Border b3(10);

    Border b4(10, 20);

    Border b5(10, 20, 30, 40);

    Border b6(b1);

    CHECK(b1.lowerX == 10);
    CHECK(b1.lowerY == 20);
    CHECK(b1.higherX == 30);
    CHECK(b1.higherY == 40);

    CHECK(b1 == b2);
    CHECK(b1 != b3);

    CHECK(b3.lowerX == 10);
    CHECK(b3.higherX == 10);
    CHECK(b3.lowerY == 10);
    CHECK(b3.higherY == 10);

    CHECK(b4.lowerX == 10);
    CHECK(b4.higherX == 10);
    CHECK(b4.lowerY == 20);
    CHECK(b4.higherY == 20);

    CHECK(b1 == b5);

    CHECK(b1 == b6);

    b3 += 5;
    CHECK(b3.lowerX == 15);
    CHECK(b3.lowerY == 15);
    CHECK(b3.higherX == 15);
    CHECK(b3.higherY == 15);

    b3 += b1;
    CHECK(b3.lowerX == 25);
    CHECK(b3.lowerY == 35);
    CHECK(b3.higherX == 45);
    CHECK(b3.higherY == 55);

    b3 += Vec2i(20, 10);
    CHECK(b3.lowerX == 45);
    CHECK(b3.lowerY == 45);
    CHECK(b3.higherX == 65);
    CHECK(b3.higherY == 65);

    b3 -= Vec2i(10, 20);
    CHECK(b3.lowerX == 35);
    CHECK(b3.lowerY == 25);
    CHECK(b3.higherX == 55);
    CHECK(b3.higherY == 45);

    b3 -= 20;
    CHECK(b3.lowerX == 15);
    CHECK(b3.lowerY == 5);
    CHECK(b3.higherX == 35);
    CHECK(b3.higherY == 25);

    b3 *= 10;
    CHECK(b3.lowerX == 150);
    CHECK(b3.lowerY == 50);
    CHECK(b3.higherX == 350);
    CHECK(b3.higherY == 250);

    b3 /= 10;
    CHECK(b3.lowerX == 15);
    CHECK(b3.lowerY == 5);
    CHECK(b3.higherX == 35);
    CHECK(b3.higherY == 25);

    Border b7 = b3 + Vec2i(10, 20);
    CHECK(b7.lowerX == 25);
    CHECK(b7.lowerY == 25);
    CHECK(b7.higherX == 45);
    CHECK(b7.higherY == 45);

    Border b8 = b7 + b3;
    CHECK(b8.lowerX == 40);
    CHECK(b8.lowerY == 30);
    CHECK(b8.higherX == 80);
    CHECK(b8.higherY == 70);

    Border b9 = b8 + 10;
    CHECK(b9.lowerX == 50);
    CHECK(b9.lowerY == 40);
    CHECK(b9.higherX == 90);
    CHECK(b9.higherY == 80);

    Border b10 = 5 + b9;
    CHECK(b10.lowerX == 55);
    CHECK(b10.lowerY == 45);
    CHECK(b10.higherX == 95);
    CHECK(b10.higherY == 85);

    b7 = b3 - Vec2i(10, 20);
    CHECK(b7.lowerX == 5);
    CHECK(b7.lowerY == -15);
    CHECK(b7.higherX == 25);
    CHECK(b7.higherY == 5);

    b8 = b7 - b3;
    CHECK(b8.lowerX == -10);
    CHECK(b8.lowerY == -20);
    CHECK(b8.higherX == -10);
    CHECK(b8.higherY == -20);

    b9 = b8 - 10;
    CHECK(b9.lowerX == -20);
    CHECK(b9.lowerY == -30);
    CHECK(b9.higherX == -20);
    CHECK(b9.higherY == -30);

    b10 = 5 - b9;
    CHECK(b10.lowerX == 25);
    CHECK(b10.lowerY == 35);
    CHECK(b10.higherX == 25);
    CHECK(b10.higherY == 35);

    Border m1(20, 30, 40, 50);
    Border m2 = m1 * Vec2i(10, 20);
    CHECK(m2.lowerX == 200);
    CHECK(m2.lowerY == 600);
    CHECK(m2.higherX == 400);
    CHECK(m2.higherY == 1000);

    Border m3 = m1 * 10;
    CHECK(m3.lowerX == 200);
    CHECK(m3.lowerY == 300);
    CHECK(m3.higherX == 400);
    CHECK(m3.higherY == 500);

    Border m4 = 10 * m1;
    CHECK(m4.lowerX == 200);
    CHECK(m4.lowerY == 300);
    CHECK(m4.higherX == 400);
    CHECK(m4.higherY == 500);

    Border m5 = m4 * m1;
    CHECK(m5.lowerX == 4000);
    CHECK(m5.lowerY == 9000);
    CHECK(m5.higherX == 16000);
    CHECK(m5.higherY == 25000);

    Border d1(20, 30, 40, 50);
    Border d2 = d1 / Vec2i(10, 20);
    CHECK(d2.lowerX == 2);
    CHECK(d2.lowerY == 1);
    CHECK(d2.higherX == 4);
    CHECK(d2.higherY == 2);

    Border d3 = d1 / 10;
    CHECK(d3.lowerX == 2);
    CHECK(d3.lowerY == 3);
    CHECK(d3.higherX == 4);
    CHECK(d3.higherY == 5);

    Border d4 = 1000 / d1;
    CHECK(d4.lowerX == 50);
    CHECK(d4.lowerY == 33);
    CHECK(d4.higherX == 25);
    CHECK(d4.higherY == 20);

    Border d5 = d4 / d3;
    CHECK(d5.lowerX == 25);
    CHECK(d5.lowerY == 11);
    CHECK(d5.higherX == 6);
    CHECK(d5.higherY == 4);

    std::stringstream ss;
    ss << d5;

    CHECK(ss.str() == "(25, 11, 6, 4)");
}