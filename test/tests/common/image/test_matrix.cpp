#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/matrix.h>
#include <common/image/quad.h>
#include <common/image/roi.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <sstream>
#include <vector>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;

TEST_CASE("Matrix<float>", "[Common.Image]")
{
    // check size:
    CHECK(sizeof(Matrix<float>) == 9 * sizeof(float));

    Matrix<float> b(10);
    CHECK(b[0] == 10);
    CHECK(b[1] == 10);
    CHECK(b[2] == 10);
    CHECK(b[3] == 10);
    CHECK(b[4] == 10);
    CHECK(b[5] == 10);
    CHECK(b[6] == 10);
    CHECK(b[7] == 10);
    CHECK(b[8] == 10);

    Matrix<float> c;
    CHECK(c[0] == 1);
    CHECK(c[1] == 0);
    CHECK(c[2] == 0);
    CHECK(c[3] == 0);
    CHECK(c[4] == 1);
    CHECK(c[5] == 0);
    CHECK(c[6] == 0);
    CHECK(c[7] == 0);
    CHECK(c[8] == 1);

    Matrix<float> d(0, 1, 2, 3, 4, 5, 6, 7, 8);
    CHECK(d[0] == 0);
    CHECK(d[1] == 1);
    CHECK(d[2] == 2);
    CHECK(d[3] == 3);
    CHECK(d[4] == 4);
    CHECK(d[5] == 5);
    CHECK(d[6] == 6);
    CHECK(d[7] == 7);
    CHECK(d[8] == 8);

    CHECK(c != d);
    CHECK(d == d);

    CHECK(d[0] == d(0, 0));
    CHECK(d[1] == d(0, 1));
    CHECK(d[2] == d(0, 2));
    CHECK(d[3] == d(1, 0));
    CHECK(d[4] == d(1, 1));
    CHECK(d[5] == d(1, 2));
    CHECK(d[6] == d(2, 0));
    CHECK(d[7] == d(2, 1));
    CHECK(d[8] == d(2, 2));

    Matrix<float> e = d + b;
    CHECK(e[0] == 10);
    CHECK(e[5] == 15);
    CHECK(e[8] == 18);

    e = d - b;
    CHECK(e[0] == -10);
    CHECK(e[5] == -5);
    CHECK(e[8] == -2);

    e = Matrix<float>(1, 2, 3, 4, 5, 6, 7, 8, 9) * Matrix<float>(9, 8, 7, 6, 5, 4, 3, 2, 1);
    CHECK(e == Matrix<float>(30, 24, 18, 84, 69, 54, 138, 114, 90));

    CHECK(Matrix<float>(1, 2, 3, 4, 5, 6, 7, 8, 9) * Vec3f(10, 20, 30) == Vec3f(140, 320, 500));
    CHECK(Vec3f(10, 20, 30) * Matrix<float>(1, 2, 3, 4, 5, 6, 7, 8, 9) == Vec3f(300, 360, 420));

    CHECK(Matrix<float>(7, 8, 8, 4, 5, 6, 7, 8, 9).Det() == 3.0f);
    d = Matrix<float>(7, 8, 8, 4, 5, 6, 7, 8, 9).Inverse();

    CHECK(d[0] == Approx(-1).margin(0.001));
    CHECK(d[1] == Approx(-2.667).margin(0.001));
    CHECK(d[2] == Approx(2.667).margin(0.001));
    CHECK(d[3] == Approx(2).margin(0.001));
    CHECK(d[4] == Approx(2.3333).margin(0.001));
    CHECK(d[5] == Approx(-3.3333).margin(0.001));
    CHECK(d[6] == Approx(-1).margin(0.001));
    CHECK(d[7] == Approx(0).margin(0.001));
    CHECK(d[8] == Approx(1).margin(0.001));

    d = d.Inverse();
    CHECK(d[0] == Approx(7).margin(0.001));
    CHECK(d[1] == Approx(8).margin(0.001));
    CHECK(d[2] == Approx(8).margin(0.001));
    CHECK(d[3] == Approx(4).margin(0.001));
    CHECK(d[4] == Approx(5).margin(0.001));
    CHECK(d[5] == Approx(6).margin(0.001));
    CHECK(d[6] == Approx(7).margin(0.001));
    CHECK(d[7] == Approx(8).margin(0.001));
    CHECK(d[8] == Approx(9).margin(0.001));

    std::stringstream ss;
    ss << Matrix<float>(2, 4, 6, 8, 10, 12, 14, 16, 18);

    CHECK(ss.str() ==
          "( 2.000000  4.000000  6.000000)\n( 8.000000 10.000000 12.000000)\n(14.000000 16.000000 18.000000)\n");

    Roi roi(0, 0, 512, 512);
    Quad<float> quadRot30({162.297501f, -93.702515f}, {604.836487f, 161.797485f}, {349.336487f, 604.336426f},
                          {-93.202499f, 348.836456f});

    AffineTransformation<float> shift1 = AffineTransformation<float>::GetTranslation({-256, -256});
    AffineTransformation<float> rot    = AffineTransformation<float>::GetRotation(-30);
    AffineTransformation<float> shift2 = AffineTransformation<float>::GetTranslation({256, 256});
    AffineTransformation<float> affine = shift2 * rot * shift1;
    Matrix<float> perspective(affine);

    Matrix<float> perspectiveFromQuad(roi, quadRot30);

    Matrix<float> diffPerspective = perspective - perspectiveFromQuad;
    float diff                    = 0;
    for (size_t i = 0; i < 9; i++)
    {
        diff += std::abs(diffPerspective.Data()[i]);
    }
    CHECK(diff < 1e-7f);
}

TEST_CASE("Matrix<double>", "[Common.Image]")
{
    // check size:
    CHECK(sizeof(Matrix<double>) == 9 * sizeof(double));

    Matrix<double> b(10);
    CHECK(b[0] == 10);
    CHECK(b[1] == 10);
    CHECK(b[2] == 10);
    CHECK(b[3] == 10);
    CHECK(b[4] == 10);
    CHECK(b[5] == 10);
    CHECK(b[6] == 10);
    CHECK(b[7] == 10);
    CHECK(b[8] == 10);

    double values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    Matrix<double> b0(values);
    CHECK(b0[0] == 0);
    CHECK(b0[1] == 1);
    CHECK(b0[2] == 2);
    CHECK(b0[3] == 3);
    CHECK(b0[4] == 4);
    CHECK(b0[5] == 5);
    CHECK(b0[6] == 6);
    CHECK(b0[7] == 7);
    CHECK(b0[8] == 8);

    Matrix<double> c;
    CHECK(c[0] == 1);
    CHECK(c[1] == 0);
    CHECK(c[2] == 0);
    CHECK(c[3] == 0);
    CHECK(c[4] == 1);
    CHECK(c[5] == 0);
    CHECK(c[6] == 0);
    CHECK(c[7] == 0);
    CHECK(c[8] == 1);

    Matrix<double> d(0, 1, 2, 3, 4, 5, 6, 7, 8);
    CHECK(d[0] == 0);
    CHECK(d[1] == 1);
    CHECK(d[2] == 2);
    CHECK(d[3] == 3);
    CHECK(d[4] == 4);
    CHECK(d[5] == 5);
    CHECK(d[6] == 6);
    CHECK(d[7] == 7);
    CHECK(d[8] == 8);

    CHECK(c != d);
    CHECK(d == d);

    CHECK(d[0] == d(0, 0));
    CHECK(d[1] == d(0, 1));
    CHECK(d[2] == d(0, 2));
    CHECK(d[3] == d(1, 0));
    CHECK(d[4] == d(1, 1));
    CHECK(d[5] == d(1, 2));
    CHECK(d[6] == d(2, 0));
    CHECK(d[7] == d(2, 1));
    CHECK(d[8] == d(2, 2));

    Matrix<double> e = d + b;
    CHECK(e[0] == 10);
    CHECK(e[5] == 15);
    CHECK(e[8] == 18);

    e = d - b;
    CHECK(e[0] == -10);
    CHECK(e[5] == -5);
    CHECK(e[8] == -2);

    e = Matrix<double>(1, 2, 3, 4, 5, 6, 7, 8, 9) * Matrix<double>(9, 8, 7, 6, 5, 4, 3, 2, 1);
    CHECK(e == Matrix<double>(30, 24, 18, 84, 69, 54, 138, 114, 90));

    CHECK(Matrix<double>(1, 2, 3, 4, 5, 6, 7, 8, 9) * Vec3d(10, 20, 30) == Vec3d(140, 320, 500));
    CHECK(Vec3d(10, 20, 30) * Matrix<double>(1, 2, 3, 4, 5, 6, 7, 8, 9) == Vec3d(300, 360, 420));

    CHECK(Matrix<double>(7, 8, 8, 4, 5, 6, 7, 8, 9).Det() == 3.0);
    d = Matrix<double>(7, 8, 8, 4, 5, 6, 7, 8, 9).Inverse();

    CHECK(d[0] == Approx(-1).margin(0.001));
    CHECK(d[1] == Approx(-2.667).margin(0.001));
    CHECK(d[2] == Approx(2.667).margin(0.001));
    CHECK(d[3] == Approx(2).margin(0.001));
    CHECK(d[4] == Approx(2.3333).margin(0.001));
    CHECK(d[5] == Approx(-3.3333).margin(0.001));
    CHECK(d[6] == Approx(-1).margin(0.001));
    CHECK(d[7] == Approx(0).margin(0.001));
    CHECK(d[8] == Approx(1).margin(0.001));

    d = d.Inverse();
    CHECK(d[0] == Approx(7).margin(0.001));
    CHECK(d[1] == Approx(8).margin(0.001));
    CHECK(d[2] == Approx(8).margin(0.001));
    CHECK(d[3] == Approx(4).margin(0.001));
    CHECK(d[4] == Approx(5).margin(0.001));
    CHECK(d[5] == Approx(6).margin(0.001));
    CHECK(d[6] == Approx(7).margin(0.001));
    CHECK(d[7] == Approx(8).margin(0.001));
    CHECK(d[8] == Approx(9).margin(0.001));

    std::stringstream ss;
    ss << Matrix<double>(2, 4, 6, 8, 10, 12, 14, 16, 18);

    CHECK(ss.str() ==
          "( 2.000000  4.000000  6.000000)\n( 8.000000 10.000000 12.000000)\n(14.000000 16.000000 18.000000)\n");

    AffineTransformation<double> affine(2, 3, 4, 5, 6, 7);
    Matrix<double> matAffine(affine);
    CHECK(matAffine[0] == 2);
    CHECK(matAffine[1] == 3);
    CHECK(matAffine[2] == 4);
    CHECK(matAffine[3] == 5);
    CHECK(matAffine[4] == 6);
    CHECK(matAffine[5] == 7);
    CHECK(matAffine[6] == 0);
    CHECK(matAffine[7] == 0);
    CHECK(matAffine[8] == 1);

    Matrix<double> transform(0.8147, 0.9134, 100, 0.9058, 0.6324, 200, 0.1270, 0.0975, 1);
    Vec2d p0(100, 100);
    Vec2d p1(1000, 1000);
    Vec2d p2(20, 800);
    Vec2d p3(900, 10);

    Vec2d trans0 = transform * p0;
    Vec2d trans1 = transform * p1;
    Vec2d trans2 = transform * p2;
    Vec2d trans3 = transform * p3;

    CHECK(trans0 == Vec2d(11.633688699360341, 15.088272921108743));
    CHECK(trans1 == Vec2d(8.106873614190686, 7.708203991130820));
    CHECK(trans2 == Vec2d(10.387711552612215, 8.879519254353690));
    CHECK(trans3 == Vec2d(7.244583960438616, 8.785585895506344));

    std::pair<Vec2d, Vec2d> pair0(p0, trans0);
    std::pair<Vec2d, Vec2d> pair1(p1, trans1);
    std::pair<Vec2d, Vec2d> pair2(p2, trans2);
    std::pair<Vec2d, Vec2d> pair3(p3, trans3);

    Matrix<double> transform2(pair0, pair1, pair2, pair3);

    CHECK(transform[0] == Approx(transform2[0]).margin(0.00001));
    CHECK(transform[1] == Approx(transform2[1]).margin(0.00001));
    CHECK(transform[2] == Approx(transform2[2]).margin(0.00001));
    CHECK(transform[3] == Approx(transform2[3]).margin(0.00001));
    CHECK(transform[4] == Approx(transform2[4]).margin(0.00001));
    CHECK(transform[5] == Approx(transform2[5]).margin(0.00001));
    CHECK(transform[6] == Approx(transform2[6]).margin(0.00001));
    CHECK(transform[7] == Approx(transform2[7]).margin(0.00001));
    CHECK(transform2[8] == 1);

    Roi roi(0, 0, 512, 512);
    Quad<double> quadRot30(
        {162.29749663118366243, -93.702503368816280727}, {604.83647796503191785, 161.79749663118369085},
        {349.33647796503191785, 604.33647796503191785}, {-93.202503368816309148, 348.83647796503191785});

    AffineTransformation<double> shift1  = AffineTransformation<double>::GetTranslation({-256, -256});
    AffineTransformation<double> rot     = AffineTransformation<double>::GetRotation(-30);
    AffineTransformation<double> shift2  = AffineTransformation<double>::GetTranslation({256, 256});
    AffineTransformation<double> affine2 = shift2 * rot * shift1;
    Matrix<double> perspective(affine2);

    Matrix<double> perspectiveFromQuad(roi, quadRot30);

    Matrix<double> diffPerspective = perspective - perspectiveFromQuad;
    double diff                    = 0;
    for (size_t i = 0; i < 9; i++)
    {
        diff += std::abs(diffPerspective.Data()[i]);
    }
    CHECK(diff < 1e-15);
}