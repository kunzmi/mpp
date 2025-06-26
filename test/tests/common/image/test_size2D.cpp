#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/size2D.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <sstream>
#include <vector>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;

TEST_CASE("Size2D", "[Common.Image]")
{
    // check size:
    CHECK(sizeof(Size2D) == 2 * sizeof(int));

    int sizes[] = {10, 20};
    Size2D t0(sizes);
    CHECK(t0.x == 10);
    CHECK(t0.y == 20);

    Size2D t1(100, 200);
    CHECK(t1.x == 100);
    CHECK(t1.y == 200);

    Size2D c(t1);
    CHECK(c.x == 100);
    CHECK(c.y == 200);
    CHECK(c == t1);

    Size2D c2 = t1;
    CHECK(c2.x == 100);
    CHECK(c2.y == 200);
    CHECK(c2 == t1);

    Size2D t2(5, 5);
    CHECK(t2.x == 5);
    CHECK(t2.y == 5);
    CHECK(c2 != t2);

    Size2D add1 = t1 + Vec2i(t2);
    CHECK(add1.x == 105);
    CHECK(add1.y == 205);

    Size2D add2 = 3 + t1;
    CHECK(add2.x == 103);
    CHECK(add2.y == 203);

    Size2D add3 = t1 + 4;
    CHECK(add3.x == 104);
    CHECK(add3.y == 204);

    Size2D add4 = t1;
    add4 += Vec2i(add3);
    CHECK(add4.x == 204);
    CHECK(add4.y == 404);

    add4 += 3;
    CHECK(add4.x == 207);
    CHECK(add4.y == 407);

    Size2D sub1 = t1 - Vec2i(t2);
    CHECK(sub1.x == 95);
    CHECK(sub1.y == 195);

    Size2D sub2 = 3 - t1;
    CHECK(sub2.x == -97);
    CHECK(sub2.y == -197);

    Size2D sub3 = t1 - 4;
    CHECK(sub3.x == 96);
    CHECK(sub3.y == 196);

    Size2D sub4 = t1;
    sub4 -= sub3;
    CHECK(sub4.x == 4);
    CHECK(sub4.y == 4);

    sub4 -= 3;
    CHECK(sub4.x == 1);
    CHECK(sub4.y == 1);

    t1          = Size2D(4, 5);
    t2          = Size2D(6, 7);
    Size2D mul1 = t1 * Vec2i(t2);
    CHECK(mul1.x == 24);
    CHECK(mul1.y == 35);

    Size2D mul2 = 3 * t1;
    CHECK(mul2.x == 12);
    CHECK(mul2.y == 15);

    Size2D mul3 = t1 * 4;
    CHECK(mul3.x == 16);
    CHECK(mul3.y == 20);

    Size2D mul4 = t1;
    mul4 *= mul3;
    CHECK(mul4.x == 64);
    CHECK(mul4.y == 100);

    mul4 *= 3;
    CHECK(mul4.x == 192);
    CHECK(mul4.y == 300);

    t1          = Size2D(1000, 2000);
    t2          = Size2D(6, 7);
    Size2D div1 = t1 / Vec2i(t2);
    CHECK(div1.x == 166);
    CHECK(div1.y == 285);

    Size2D div2 = 30000 / t1;
    CHECK(div2.x == 30);
    CHECK(div2.y == 15);

    Size2D div3 = t1 / 4;
    CHECK(div3.x == 250);
    CHECK(div3.y == 500);

    Size2D div4 = t2 * 10000;
    div4 /= div3;
    CHECK(div4.x == 240);
    CHECK(div4.y == 140);

    div4 /= 3;
    CHECK(div4.x == 80);
    CHECK(div4.y == 46);

    Size2D size_idx(1023, 1025);
    size_t idx = size_idx.GetFlatIndex(456, 789);
    CHECK(idx == 807603ULL);
    CHECK(idx == size_idx(456, 789));
    Vec2i coord = size_idx.GetCoordinates(idx);
    CHECK(coord == Vec2i(456, 789));

    std::stringstream ss;
    ss << size_idx;
    CHECK(ss.str() == "(1023 x 1025)");
}