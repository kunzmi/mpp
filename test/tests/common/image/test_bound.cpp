#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/affineTransformation.h>
#include <common/image/bound.h>
#include <common/image/quad.h>
#include <common/image/roi.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <sstream>
#include <vector>

using namespace opp;
using namespace opp::image;
using namespace Catch;

TEST_CASE("Bound<float>", "[Common.Image]")
{
    {
        Quad<float> quad{{10, 16}, {15, 12}, {17, 18}, {19, 20}};
        Bound<float> bound(quad);
        Bound<float> check({10, 12}, {19, 20});
        CHECK(check == bound);
    }
    {
        AffineTransformation<float> shift1 = AffineTransformation<float>::GetTranslation({-256, -256});
        AffineTransformation<float> rot    = AffineTransformation<float>::GetRotation(-30);
        AffineTransformation<float> shift2 = AffineTransformation<float>::GetTranslation({256, 256});
        AffineTransformation<float> affine = shift2 * rot * shift1;

        Roi roi{0, 0, 512, 512};

        Quad<float> quad = affine * roi;

        Bound<float> bound(quad);

        Bound<float> check({-93.20249939f, -93.70251465f}, {604.8364868f, 604.3364258f});

        CHECK(check == bound);
    }
}

TEST_CASE("Bound<double>", "[Common.Image]")
{
    {
        Quad<double> quad{{10, 16}, {15, 12}, {17, 18}, {19, 20}};
        Bound<double> bound(quad);
        Bound<double> check({10, 12}, {19, 20});
        CHECK(check == bound);
    }
    {
        AffineTransformation<double> shift1 = AffineTransformation<double>::GetTranslation({-256, -256});
        AffineTransformation<double> rot    = AffineTransformation<double>::GetRotation(-30);
        AffineTransformation<double> shift2 = AffineTransformation<double>::GetTranslation({256, 256});
        AffineTransformation<double> affine = shift2 * rot * shift1;

        Roi roi{0, 0, 512, 512};

        Quad<double> quad = affine * roi;

        Bound<double> bound(quad);

        Bound<double> check({-93.202503368816309148, -93.702503368816280727},
                            {604.83647796503191785, 604.33647796503191785});

        CHECK(check == bound);
    }
}