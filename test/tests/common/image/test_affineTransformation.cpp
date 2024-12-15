#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/matrix.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <sstream>
#include <vector>

using namespace opp;
using namespace opp::image;
using namespace Catch;

TEST_CASE("AffineTransformation", "[Common.Image]")
{
    // check size:
    CHECK(sizeof(AffineTransformation) == 6 * sizeof(double));

    AffineTransformation b(10);
    CHECK(b[0] == 10);
    CHECK(b[1] == 10);
    CHECK(b[2] == 10);
    CHECK(b[3] == 10);
    CHECK(b[4] == 10);
    CHECK(b[5] == 10);

    double values[] = {0, 1, 2, 3, 4, 5};
    AffineTransformation b0(values);
    CHECK(b0[0] == 0);
    CHECK(b0[1] == 1);
    CHECK(b0[2] == 2);
    CHECK(b0[3] == 3);
    CHECK(b0[4] == 4);
    CHECK(b0[5] == 5);

    AffineTransformation c;
    CHECK(c[0] == 1);
    CHECK(c[1] == 0);
    CHECK(c[2] == 0);
    CHECK(c[3] == 0);
    CHECK(c[4] == 1);
    CHECK(c[5] == 0);

    AffineTransformation d(0, 1, 2, 3, 4, 5);
    CHECK(d[0] == 0);
    CHECK(d[1] == 1);
    CHECK(d[2] == 2);
    CHECK(d[3] == 3);
    CHECK(d[4] == 4);
    CHECK(d[5] == 5);

    CHECK(c != d);
    CHECK(d == d);

    CHECK(d[0] == d(0, 0));
    CHECK(d[1] == d(0, 1));
    CHECK(d[2] == d(0, 2));
    CHECK(d[3] == d(1, 0));
    CHECK(d[4] == d(1, 1));
    CHECK(d[5] == d(1, 2));

    AffineTransformation e = d + b;
    CHECK(e[0] == 10);
    CHECK(e[5] == 15);

    e = d - b;
    CHECK(e[0] == -10);
    CHECK(e[5] == -5);

    e = AffineTransformation(1, 2, 3, 4, 5, 6) * AffineTransformation(9, 8, 7, 6, 5, 4);
    CHECK(e == AffineTransformation(21, 18, 18, 66, 57, 54));

    CHECK(AffineTransformation(4, 5, 6, 7, 8, 9) * Vec2d(10, 20) == Vec2d(146, 239));

    CHECK(AffineTransformation(7, 8, 8, 4, 5, 6).Det() == Approx(3).margin(0.0000001));
    d = AffineTransformation(7, 8, 8, 4, 5, 6).Inverse();

    CHECK(d[0] == Approx(1.666666666666666).margin(0.000001));
    CHECK(d[1] == Approx(-2.666666666666665).margin(0.000001));
    CHECK(d[2] == Approx(2.666666666666665).margin(0.000001));
    CHECK(d[3] == Approx(-1.333333333333333).margin(0.000001));
    CHECK(d[4] == Approx(2.333333333333332).margin(0.000001));
    CHECK(d[5] == Approx(-3.333333333333332).margin(0.000001));

    d = d.Inverse();
    CHECK(d[0] == Approx(7).margin(0.00001));
    CHECK(d[1] == Approx(8).margin(0.00001));
    CHECK(d[2] == Approx(8).margin(0.00001));
    CHECK(d[3] == Approx(4).margin(0.00001));
    CHECK(d[4] == Approx(5).margin(0.00001));
    CHECK(d[5] == Approx(6).margin(0.00001));

    std::stringstream ss;
    ss << AffineTransformation(2, 4, 6, 8, 10, 12);

    CHECK(ss.str() ==
          "( 2.000000  4.000000  6.000000)\n( 8.000000 10.000000 12.000000)\n( 0.000000  0.000000  1.000000)\n");

    AffineTransformation transform(0.8147, 0.9134, 100, 0.9058, 0.6324, 200);
    Vec2d p0(100, 100);
    Vec2d p1(1000, 1000);
    Vec2d p2(20, 800);
    Vec2d p3(900, 10);

    Vec2d trans0 = transform * p0;
    Vec2d trans1 = transform * p1;
    Vec2d trans2 = transform * p2;
    Vec2d trans3 = transform * p3;

    CHECK(trans0 == Vec2d(272.81, 353.82));
    CHECK(trans1 == Vec2d(1828.1, 1738.2));
    CHECK(trans2 == Vec2d(847.014, 724.036));

    std::pair<Vec2d, Vec2d> pair0(p0, trans0);
    std::pair<Vec2d, Vec2d> pair1(p1, trans1);
    std::pair<Vec2d, Vec2d> pair2(p2, trans2);
    std::pair<Vec2d, Vec2d> pair3(p3, trans3);

    AffineTransformation transform2(pair0, pair1, pair2);
    Matrix<double> transformAffine(pair0, pair1, pair2, pair3);

    CHECK(transform[0] == Approx(transform2[0]).margin(0.00001));
    CHECK(transform[1] == Approx(transform2[1]).margin(0.00001));
    CHECK(transform[2] == Approx(transform2[2]).margin(0.00001));
    CHECK(transform[3] == Approx(transform2[3]).margin(0.00001));
    CHECK(transform[4] == Approx(transform2[4]).margin(0.00001));
    CHECK(transform[5] == Approx(transform2[5]).margin(0.00001));

    CHECK(transform[0] == Approx(transformAffine[0]).margin(0.00001));
    CHECK(transform[1] == Approx(transformAffine[1]).margin(0.00001));
    CHECK(transform[2] == Approx(transformAffine[2]).margin(0.00001));
    CHECK(transform[3] == Approx(transformAffine[3]).margin(0.00001));
    CHECK(transform[4] == Approx(transformAffine[4]).margin(0.00001));
    CHECK(transform[5] == Approx(transformAffine[5]).margin(0.00001));
    CHECK(0 == Approx(transformAffine[6]).margin(0.00001));
    CHECK(0 == Approx(transformAffine[7]).margin(0.00001));
    CHECK(1 == Approx(transformAffine[8]).margin(0.00001));

    AffineTransformation rot45 = AffineTransformation::GetRotation(45);
    AffineTransformation rot90 = AffineTransformation::GetRotation(90);
    AffineTransformation shift = AffineTransformation::GetTranslation(Vec2d(100, 200));
    AffineTransformation shear = AffineTransformation::GetShear(Vec2d(0.2, 0.3));
    AffineTransformation scale = AffineTransformation::GetScale(Vec2d(1.1, 1.2));

    CHECK(rot45[0] == Approx(0.707106781186547).margin(0.00001));
    CHECK(rot45[1] == Approx(-0.707106781186547).margin(0.00001));
    CHECK(rot45[2] == Approx(0).margin(0.00001));
    CHECK(rot45[3] == Approx(0.707106781186547).margin(0.00001));
    CHECK(rot45[4] == Approx(0.707106781186547).margin(0.00001));
    CHECK(rot45[5] == Approx(0).margin(0.00001));

    CHECK(rot90[0] == Approx(0).margin(0.00001));
    CHECK(rot90[1] == Approx(-1).margin(0.00001));
    CHECK(rot90[2] == Approx(0).margin(0.00001));
    CHECK(rot90[3] == Approx(1).margin(0.00001));
    CHECK(rot90[4] == Approx(0).margin(0.00001));
    CHECK(rot90[5] == Approx(0).margin(0.00001));

    CHECK(shift[0] == Approx(1).margin(0.00001));
    CHECK(shift[1] == Approx(0).margin(0.00001));
    CHECK(shift[2] == Approx(100).margin(0.00001));
    CHECK(shift[3] == Approx(0).margin(0.00001));
    CHECK(shift[4] == Approx(1).margin(0.00001));
    CHECK(shift[5] == Approx(200).margin(0.00001));

    CHECK(shear[0] == Approx(1).margin(0.00001));
    CHECK(shear[1] == Approx(0.2).margin(0.00001));
    CHECK(shear[2] == Approx(0).margin(0.00001));
    CHECK(shear[3] == Approx(0.3).margin(0.00001));
    CHECK(shear[4] == Approx(1).margin(0.00001));
    CHECK(shear[5] == Approx(0).margin(0.00001));

    CHECK(scale[0] == Approx(1.1).margin(0.00001));
    CHECK(scale[1] == Approx(0).margin(0.00001));
    CHECK(scale[2] == Approx(0).margin(0.00001));
    CHECK(scale[3] == Approx(0).margin(0.00001));
    CHECK(scale[4] == Approx(1.2).margin(0.00001));
    CHECK(scale[5] == Approx(0).margin(0.00001));
}