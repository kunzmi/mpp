#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/border.h>
#include <common/image/roi.h>
#include <common/image/size2D.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <sstream>
#include <vector>

using namespace opp;
using namespace opp::image;
using namespace Catch;

TEST_CASE("Roi", "[Common.Image]")
{
    Roi roi0;
    CHECK(roi0.x == 0);
    CHECK(roi0.y == 0);
    CHECK(roi0.width == 0);
    CHECK(roi0.height == 0);

    int vals[] = {10, 20, 100, 200};
    Roi roi1(vals);
    CHECK(roi1.x == 10);
    CHECK(roi1.y == 20);
    CHECK(roi1.width == 100);
    CHECK(roi1.height == 200);

    CHECK(roi1.FirstX() == 10);
    CHECK(roi1.LastX() == 10 + 100 - 1);
    CHECK(roi1.FirstY() == 20);
    CHECK(roi1.LastY() == 20 + 200 - 1);
    CHECK(roi1.FirstPixel() == Vec2i(10, 20));
    CHECK(roi1.LastPixel() == Vec2i(109, 219));
    CHECK(roi1.BoundingBoxMin() == Vec2i(10, 20));
    CHECK(roi1.BoundingBoxMax() == Vec2i(110, 220));
    CHECK(roi1.Size() == Size2D(100, 200));

    Vec4i vec4(11, 21, 100, 200);
    Roi roi2(vec4);
    CHECK(roi2.x == 11);
    CHECK(roi2.y == 21);
    CHECK(roi2.width == 100);
    CHECK(roi2.height == 200);

    Roi roi3(Vec2i(12, 13), 100, 200);
    CHECK(roi3.x == 12);
    CHECK(roi3.y == 13);
    CHECK(roi3.width == 100);
    CHECK(roi3.height == 200);

    Roi roi4(Vec2i(13, 14), Size2D(100, 200));
    CHECK(roi4.x == 13);
    CHECK(roi4.y == 14);
    CHECK(roi4.width == 100);
    CHECK(roi4.height == 200);

    Roi roi5(14, 15, Size2D(100, 200));
    CHECK(roi5.x == 14);
    CHECK(roi5.y == 15);
    CHECK(roi5.width == 100);
    CHECK(roi5.height == 200);

    Roi roi6(15, 16, 100, 200);
    CHECK(roi6.x == 15);
    CHECK(roi6.y == 16);
    CHECK(roi6.width == 100);
    CHECK(roi6.height == 200);

    std::stringstream ss;
    ss << roi1;

    CHECK(ss.str() == "ROI: X = [10..109] Y = [20..219]");

    Roi a = Roi(572, 444, Size2D(256, 512));
    CHECK(a.x == 572);
    CHECK(a.y == 444);
    CHECK(a.width == 256);
    CHECK(a.height == 512);

    CHECK(a.Contains(Vec2i(750, 750)));
    CHECK(a.Contains(Vec2i(572, 444)));
    CHECK(a.Contains(Vec2i(572 + 256 - 1, 444 + 512 - 1)));

    CHECK(!a.Contains(Vec2i(572 + 256, 444 + 512)));
    Roi b = a + 1;
    CHECK(b.Contains(Vec2i(750, 750)));
    CHECK(b.Contains(Vec2i(572 - 1, 444 - 1)));
    CHECK(b.Contains(Vec2i(572 + 256, 444 + 512)));

    CHECK(b.Contains(a));
    CHECK(!a.Contains(b));

    Roi big   = Roi(256, 256, 128, 128);
    Roi small = Roi(360, 340, 80, 80);

    CHECK(!big.Contains(small));
    Roi fit = big.ShiftUntilFit(small);

    CHECK(big.Contains(fit));
    CHECK(big.LastX() == fit.LastX());
    CHECK(big.LastY() == fit.LastY());
    CHECK(small.width == fit.width);
    CHECK(small.height == fit.height);

    big   = Roi(256, 256, 128, 128);
    small = Roi(100, 100, 80, 80);

    CHECK(!big.Contains(small));
    fit = big.ShiftUntilFit(small);

    CHECK(big.Contains(fit));
    CHECK(big.FirstX() == fit.FirstX());
    CHECK(big.FirstY() == fit.FirstY());
    CHECK(small.width == fit.width);
    CHECK(small.height == fit.height);

    fit = small.ShiftUntilFit(big);

    CHECK(big.Contains(fit));
    CHECK(big.FirstX() == fit.FirstX());
    CHECK(big.FirstY() == fit.FirstY());
    CHECK(small.width == fit.width);
    CHECK(small.height == fit.height);

    small.x = -200;
    small.y = 20000;
    fit     = big.ShiftUntilFit(small);

    CHECK(big.Contains(fit));

    small = Roi(-50, -25, Size2D(100, 50));
    big   = Roi(-20, -40, Size2D(40, 80));
    fit   = small.ShiftUntilFit(big);
    CHECK(small == fit);

    Roi r1(100, 100, 1000, 1000);
    Roi r2 = r1;
    CHECK(r1 == r2);

    r1 += 1;
    CHECK(r1 != r2);
    CHECK(r1.x == 99);
    CHECK(r1.y == 99);
    CHECK(r1.width == 1002);
    CHECK(r1.height == 1002);

    r1 += Vec2i(10, 20);
    CHECK(r1.x == 89);
    CHECK(r1.y == 79);
    CHECK(r1.width == 1022);
    CHECK(r1.height == 1042);

    r1 += Border(10, 20, 30, 40);
    CHECK(r1.x == 79);
    CHECK(r1.y == 59);
    CHECK(r1.width == 1062);
    CHECK(r1.height == 1102);

    r1 = Roi(0, 0, 100, 200);
    r2 = Roi(1000, 1000, 100, 200);

    r1 += r2;
    CHECK(r1.x == 0);
    CHECK(r1.y == 0);
    CHECK(r1.width == 1100);
    CHECK(r1.height == 1200);

    r1     = Roi(0, 0, 100, 200);
    r2     = Roi(1000, 1000, 100, 200);
    Roi r3 = r1.Union(r2);
    CHECK(r3.x == 0);
    CHECK(r3.y == 0);
    CHECK(r3.width == 1100);
    CHECK(r3.height == 1200);
    CHECK(r3.Center() == Vec2i(550, 600));

    r1 = Roi(100, 100, 1000, 1000);
    r2 = r1;

    r1 -= 1;
    CHECK(r1 != r2);
    CHECK(r1.x == 101);
    CHECK(r1.y == 101);
    CHECK(r1.width == 998);
    CHECK(r1.height == 998);

    r1 -= Vec2i(10, 20);
    CHECK(r1.x == 111);
    CHECK(r1.y == 121);
    CHECK(r1.width == 978);
    CHECK(r1.height == 958);

    r1 -= Border(10, 20, 30, 40);
    CHECK(r1.x == 121);
    CHECK(r1.y == 141);
    CHECK(r1.width == 938);
    CHECK(r1.height == 898);

    r1 *= 2;
    CHECK(r1.x == 121 * 2);
    CHECK(r1.y == 141 * 2);
    CHECK(r1.width == 938 * 2);
    CHECK(r1.height == 898 * 2);

    r1 /= 2;
    CHECK(r1.x == 121);
    CHECK(r1.y == 141);
    CHECK(r1.width == 938);
    CHECK(r1.height == 898);

    r1 = Roi(0, 0, 10, 20);
    r2 = Roi(5, 6, 10, 10);
    r3 = Roi(100, 200, 10, 20);

    CHECK(r1.IntersectsWith(r2));
    CHECK(!r1.IntersectsWith(r3));

    CHECK(r1.Intersect(r2) == Roi(5, 6, 5, 10));
    CHECK(r2.Intersect(r1) == Roi(5, 6, 5, 10));
    CHECK(r1.Intersect(r3) == Roi());

    CHECK(r1 + Vec2i(3, 4) == Roi(-3, -4, 16, 28));
    CHECK(r1 + Border(3, 4, 5, 6) == Roi(-3, -4, 18, 30));
    CHECK(Border(3, 4, 5, 6) + r1 == Roi(-3, -4, 18, 30));
    CHECK(r1 + 10 == Roi(-10, -10, 30, 40));
    CHECK(10 + r1 == Roi(-10, -10, 30, 40));

    CHECK(r1 - Vec2i(3, 4) == Roi(3, 4, 4, 12));
    CHECK(r1 - Border(3, 4, 5, 6) == Roi(3, 4, 2, 10));
    CHECK(Border(3, 4, 5, 6) + r1 == Roi(-3, -4, 18, 30));
    CHECK(r1 - 1 == Roi(1, 1, 8, 18));

    Border b1(3, 4, 5, 6);
    r2        = r1 + b1;
    Border b2 = r2 - r1;
    Border b3 = r1 - r2;
    Border b4 = r1 - r1;
    CHECK(b1 == b2);
    CHECK(b3 == Border(-3, -4, -5, -6));
    CHECK(b4 == Border(0));

    r1 = Roi(100, 200, 100, 200);
    r2 = r1 + b1;
    b2 = r2 - r1;
    CHECK(b1 == b2);

    r1 = Roi(10, 20, 100, 200);
    r2 = r1 * Vec2i(3, 2);
    CHECK(r2 == Roi(10 * 3, 20 * 2, 100 * 3, 200 * 2));

    r2 = r1 * 2;
    CHECK(r2 == Roi(10 * 2, 20 * 2, 100 * 2, 200 * 2));
    r2 = 2 * r1;
    CHECK(r2 == Roi(10 * 2, 20 * 2, 100 * 2, 200 * 2));

    r2 = r1 / Vec2i(3, 2);
    CHECK(r2 == Roi(10 / 3, 20 / 2, 100 / 3, 200 / 2));

    r2 = r1 / 2;
    CHECK(r2 == Roi(10 / 2, 20 / 2, 100 / 2, 200 / 2));
}