#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/matrix.h>
#include <common/image/quad.h>
#include <common/image/roi.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <sstream>
#include <vector>

using namespace opp;
using namespace opp::image;
using namespace Catch;

TEST_CASE("Quad<float>", "[Common.Image]")
{
    {
        AffineTransformation<float> shift1 = AffineTransformation<float>::GetTranslation({-256, -256});
        AffineTransformation<float> rot    = AffineTransformation<float>::GetRotation(-30);
        AffineTransformation<float> shift2 = AffineTransformation<float>::GetTranslation({256, 256});
        AffineTransformation<float> affine = shift2 * rot * shift1;

        Roi roi{0, 0, 512, 512};

        Quad<float> quad = affine * roi;
        Quad<float> check({162.297501f, -93.702515f}, {604.836487f, 161.797485f}, {349.336487f, 604.336426f},
                          {-93.202499f, 348.836456f});

        CHECK(check == quad);
    }

    {
        AffineTransformation<float> shift1 = AffineTransformation<float>::GetTranslation({-256, -256});
        AffineTransformation<float> rot    = AffineTransformation<float>::GetRotation(-30);
        AffineTransformation<float> shift2 = AffineTransformation<float>::GetTranslation({256, 256});
        AffineTransformation<float> affine = shift2 * rot * shift1;
        Matrix<float> perspective(affine);

        Roi roi{0, 0, 512, 512};

        Quad<float> quad = perspective * roi;
        Quad<float> check({162.297501f, -93.702515f}, {604.836487f, 161.797485f}, {349.336487f, 604.336426f},
                          {-93.202499f, 348.836456f});

        CHECK(check == quad);
    }
}

TEST_CASE("Quad<double>", "[Common.Image]")
{
    {
        AffineTransformation<double> shift1 = AffineTransformation<double>::GetTranslation({-256, -256});
        AffineTransformation<double> rot    = AffineTransformation<double>::GetRotation(-30);
        AffineTransformation<double> shift2 = AffineTransformation<double>::GetTranslation({256, 256});
        AffineTransformation<double> affine = shift2 * rot * shift1;

        Roi roi{0, 0, 512, 512};

        Quad<double> quad = affine * roi;
        Quad<double> check(
            {162.29749663118366243, -93.702503368816280727}, {604.83647796503191785, 161.79749663118369085},
            {349.33647796503191785, 604.33647796503191785}, {-93.202503368816309148, 348.83647796503191785});
        CHECK(check == quad);
    }
    {
        AffineTransformation<double> shift1 = AffineTransformation<double>::GetTranslation({-256, -256});
        AffineTransformation<double> rot    = AffineTransformation<double>::GetRotation(-30);
        AffineTransformation<double> shift2 = AffineTransformation<double>::GetTranslation({256, 256});
        AffineTransformation<double> affine = shift2 * rot * shift1;
        Matrix<double> perspective(affine);

        Roi roi{0, 0, 512, 512};

        Quad<double> quad = perspective * roi;
        Quad<double> check(
            {162.29749663118366243, -93.702503368816280727}, {604.83647796503191785, 161.79749663118369085},
            {349.33647796503191785, 604.33647796503191785}, {-93.202503368816309148, 348.83647796503191785});
        CHECK(check == quad);
    }
}