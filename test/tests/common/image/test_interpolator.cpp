#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/functors/borderControl.h>
#include <common/image/functors/interpolator.h>
#include <common/image/pixelTypes.h>
#include <common/mpp_defs.h>
#include <common/safeCast.h>
#include <common/vectorTypes.h>
#include <cstddef>
#include <numeric>
#include <sstream>
#include <vector>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;

TEST_CASE("Interpolator - NearestNeighbor", "[Common.Image]")
{
    constexpr int widthSrc         = 4;
    constexpr int heightSrc        = 4;
    std::vector<Pixel32fC1> srcImg = {0.3517f, 0.8308f, 0.5853f, 0.5497f, 0.9172f, 0.2858f, 0.7572f, 0.7537f,
                                      0.3804f, 0.5678f, 0.0759f, 0.0540f, 0.5308f, 0.7792f, 0.9340f, 0.1299f};

    BorderControl<Pixel32fC1, BorderType::None> bc(srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc},
                                                   {0, 0});

    Interpolator<Pixel32fC1, BorderControl<Pixel32fC1, BorderType::None>, float, InterpolationMode::NearestNeighbor>
        interpol(bc);

    std::vector<Pixel32fC2> checkpoints = {{1.4f, 1.1f}, {1.2f, 1.3f}, {1.5f, 1.5f}, {1.11f, 1.64f}, {1.9f, 1.8f}};
    std::vector<Pixel32fC1> checkvalues = {0.2858f, 0.2858f, 0.0759f, 0.5678f, 0.0759f};

    for (size_t i = 0; i < checkpoints.size(); i++)
    {
        CHECK(Pixel32fC1::Abs(interpol(checkpoints[i].x, checkpoints[i].y) - checkvalues[i]) < 0.000001f);
    }
}

TEST_CASE("Interpolator - Linear", "[Common.Image]")
{
    constexpr int widthSrc         = 4;
    constexpr int heightSrc        = 4;
    std::vector<Pixel32fC1> srcImg = {0.3517f, 0.8308f, 0.5853f, 0.5497f, 0.9172f, 0.2858f, 0.7572f, 0.7537f,
                                      0.3804f, 0.5678f, 0.0759f, 0.0540f, 0.5308f, 0.7792f, 0.9340f, 0.1299f};

    BorderControl<Pixel32fC1, BorderType::None> bc(srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc},
                                                   {0, 0});

    Interpolator<Pixel32fC1, BorderControl<Pixel32fC1, BorderType::None>, float, InterpolationMode::Linear> interpol(
        bc);

    std::vector<Pixel32fC2> checkpoints = {{1.4f, 1.1f}, {1.2f, 1.3f}, {1.5f, 1.5f}, {1.11f, 1.64f}, {1.9f, 1.8f}};
    std::vector<Pixel32fC1> checkvalues = {0.464028f, 0.406882f, 0.421675f, 0.45031768f, 0.242084f};

    for (size_t i = 0; i < checkpoints.size(); i++)
    {
        CHECK(Pixel32fC1::Abs(interpol(checkpoints[i].x, checkpoints[i].y) - checkvalues[i]) < 0.000001f);
    }
}

TEST_CASE("Interpolator - CubicHermiteSpline", "[Common.Image]")
{
    constexpr int widthSrc         = 4;
    constexpr int heightSrc        = 4;
    std::vector<Pixel32fC1> srcImg = {0.3517f, 0.8308f, 0.5853f, 0.5497f, 0.9172f, 0.2858f, 0.7572f, 0.7537f,
                                      0.3804f, 0.5678f, 0.0759f, 0.0540f, 0.5308f, 0.7792f, 0.9340f, 0.1299f};

    BorderControl<Pixel32fC1, BorderType::None> bc(srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc},
                                                   {0, 0});

    Interpolator<Pixel32fC1, BorderControl<Pixel32fC1, BorderType::None>, float, InterpolationMode::CubicHermiteSpline>
        interpol(bc);

    std::vector<Pixel32fC2> checkpoints = {{1.4f, 1.1f}, {1.2f, 1.3f}, {1.5f, 1.5f}, {1.11f, 1.64f}, {1.9f, 1.8f}};
    std::vector<Pixel32fC1> checkvalues = {0.3997599012f, 0.3180918688f, 0.35575546875f, 0.417283935458947f,
                                           0.1502261284f};

    for (size_t i = 0; i < checkpoints.size(); i++)
    {
        CHECK(Pixel32fC1::Abs(interpol(checkpoints[i].x, checkpoints[i].y) - checkvalues[i]) < 0.000001f);
    }
}

TEST_CASE("Interpolator - CubicLagrange", "[Common.Image]")
{
    constexpr int widthSrc         = 4;
    constexpr int heightSrc        = 4;
    std::vector<Pixel32fC1> srcImg = {0.3517f, 0.8308f, 0.5853f, 0.5497f, 0.9172f, 0.2858f, 0.7572f, 0.7537f,
                                      0.3804f, 0.5678f, 0.0759f, 0.0540f, 0.5308f, 0.7792f, 0.9340f, 0.1299f};

    BorderControl<Pixel32fC1, BorderType::None> bc(srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc},
                                                   {0, 0});

    Interpolator<Pixel32fC1, BorderControl<Pixel32fC1, BorderType::None>, float, InterpolationMode::CubicLagrange>
        interpol(bc);

    std::vector<Pixel32fC2> checkpoints = {{1.4f, 1.1f}, {1.2f, 1.3f}, {1.5f, 1.5f}, {1.11f, 1.64f}, {1.9f, 1.8f}};
    std::vector<Pixel32fC1> checkvalues = {0.4086051583f, 0.3395940363f, 0.3557554483f, 0.4065053761f, 0.1933734417f};

    for (size_t i = 0; i < checkpoints.size(); i++)
    {
        CHECK(Pixel32fC1::Abs(interpol(checkpoints[i].x, checkpoints[i].y) - checkvalues[i]) < 0.000001f);
    }
}

TEST_CASE("Interpolator - Cubic2ParamB05C03", "[Common.Image]")
{
    constexpr int widthSrc         = 4;
    constexpr int heightSrc        = 4;
    std::vector<Pixel32fC1> srcImg = {0.3517f, 0.8308f, 0.5853f, 0.5497f, 0.9172f, 0.2858f, 0.7572f, 0.7537f,
                                      0.3804f, 0.5678f, 0.0759f, 0.0540f, 0.5308f, 0.7792f, 0.9340f, 0.1299f};

    BorderControl<Pixel32fC1, BorderType::None> bc(srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc},
                                                   {0, 0});

    Interpolator<Pixel32fC1, BorderControl<Pixel32fC1, BorderType::None>, float, InterpolationMode::Cubic2ParamB05C03>
        interpol(bc);

    std::vector<Pixel32fC2> checkpoints = {{1.4f, 1.1f}, {1.2f, 1.3f}, {1.5f, 1.5f}, {1.11f, 1.64f}, {1.9f, 1.8f}};
    std::vector<Pixel32fC1> checkvalues = {0.456605196f, 0.4014370143f, 0.395013243f, 0.4316885769f, 0.2713333666f};

    for (size_t i = 0; i < checkpoints.size(); i++)
    {
        CHECK(Pixel32fC1::Abs(interpol(checkpoints[i].x, checkpoints[i].y) - checkvalues[i]) < 0.000002f);
    }
}

TEST_CASE("Interpolator - Cubic2ParamBSpline", "[Common.Image]")
{
    constexpr int widthSrc         = 4;
    constexpr int heightSrc        = 4;
    std::vector<Pixel32fC1> srcImg = {0.3517f, 0.8308f, 0.5853f, 0.5497f, 0.9172f, 0.2858f, 0.7572f, 0.7537f,
                                      0.3804f, 0.5678f, 0.0759f, 0.0540f, 0.5308f, 0.7792f, 0.9340f, 0.1299f};

    BorderControl<Pixel32fC1, BorderType::None> bc(srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc},
                                                   {0, 0});

    Interpolator<Pixel32fC1, BorderControl<Pixel32fC1, BorderType::None>, float, InterpolationMode::Cubic2ParamBSpline>
        interpol(bc);

    std::vector<Pixel32fC2> checkpoints = {{1.4f, 1.1f}, {1.2f, 1.3f}, {1.5f, 1.5f}, {1.11f, 1.64f}, {1.9f, 1.8f}};
    std::vector<Pixel32fC1> checkvalues = {0.4985078871f, 0.4677197039f, 0.440200597f, 0.4643953443f, 0.3685692251f};

    for (size_t i = 0; i < checkpoints.size(); i++)
    {
        CHECK(Pixel32fC1::Abs(interpol(checkpoints[i].x, checkpoints[i].y) - checkvalues[i]) < 0.000001f);
    }
}

TEST_CASE("Interpolator - Cubic2ParamCatmullRom", "[Common.Image]")
{
    constexpr int widthSrc         = 4;
    constexpr int heightSrc        = 4;
    std::vector<Pixel32fC1> srcImg = {0.3517f, 0.8308f, 0.5853f, 0.5497f, 0.9172f, 0.2858f, 0.7572f, 0.7537f,
                                      0.3804f, 0.5678f, 0.0759f, 0.0540f, 0.5308f, 0.7792f, 0.9340f, 0.1299f};

    BorderControl<Pixel32fC1, BorderType::None> bc(srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc},
                                                   {0, 0});

    Interpolator<Pixel32fC1, BorderControl<Pixel32fC1, BorderType::None>, float,
                 InterpolationMode::Cubic2ParamCatmullRom>
        interpol(bc);

    std::vector<Pixel32fC2> checkpoints = {{1.4f, 1.1f}, {1.2f, 1.3f}, {1.5f, 1.5f}, {1.11f, 1.64f}, {1.9f, 1.8f}};
    std::vector<Pixel32fC1> checkvalues = {0.3997599781f, 0.3180919588f, 0.3557554781f, 0.4172839224f, 0.1502262652f};

    for (size_t i = 0; i < checkpoints.size(); i++)
    {
        CHECK(Pixel32fC1::Abs(interpol(checkpoints[i].x, checkpoints[i].y) - checkvalues[i]) < 0.000001f);
    }
}

// TEST_CASE("Interpolator - Lanczos2Lobed", "[Common.Image]")
//{
//     constexpr int widthSrc         = 4;
//     constexpr int heightSrc        = 4;
//     std::vector<Pixel32fC1> srcImg = {0.3517f, 0.8308f, 0.5853f, 0.5497f, 0.9172f, 0.2858f, 0.7572f, 0.7537f,
//                                       0.3804f, 0.5678f, 0.0759f, 0.0540f, 0.5308f, 0.7792f, 0.9340f, 0.1299f};
//
//     BorderControl<Pixel32fC1, BorderType::None> bc(srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc});
//
//     Interpolator<Pixel32fC1, BorderControl<Pixel32fC1, BorderType::None>, float, InterpolationMode::Lanczos2Lobed>
//         interpol(bc);
//
//     std::vector<Pixel32fC2> checkpoints = {{1.4f, 1.1f}, {1.2f, 1.3f}, {1.5f, 1.5f}, {1.11f, 1.64f}, {1.9f, 1.8f}};
//     std::vector<Pixel32fC1> checkvalues = {0.3574733138f, 0.2720996737f, 0.2933445871f, 0.3743664622f,
//     0.1381813735f};
//
//     for (size_t i = 0; i < checkpoints.size(); i++)
//     {
//         //CHECK(Pixel32fC1::Abs(interpol(checkpoints[i].x, checkpoints[i].y) - checkvalues[i]) < 0.000001f);
//     }
// }

TEST_CASE("Interpolator - Lanczos3Lobed", "[Common.Image]")
{
    constexpr int widthSrc         = 6;
    constexpr int heightSrc        = 6;
    std::vector<Pixel32fC1> srcImg = {0.5688f, 0.4694f, 0.0119f, 0.3371f, 0.1622f, 0.7943f, 0.3112f, 0.5285f, 0.1656f,
                                      0.6020f, 0.2630f, 0.6541f, 0.6892f, 0.7482f, 0.4505f, 0.0838f, 0.2290f, 0.9133f,
                                      0.1524f, 0.8258f, 0.5383f, 0.9961f, 0.0782f, 0.4427f, 0.1067f, 0.9619f, 0.0046f,
                                      0.7749f, 0.8173f, 0.8687f, 0.0844f, 0.3998f, 0.2599f, 0.8001f, 0.4314f, 0.9106f};

    BorderControl<Pixel32fC1, BorderType::None> bc(srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc},
                                                   {0, 0});

    Interpolator<Pixel32fC1, BorderControl<Pixel32fC1, BorderType::None>, float, InterpolationMode::Lanczos3Lobed>
        interpol(bc);

    std::vector<Pixel32fC2> checkpoints = {{2.4f, 2.1f}, {2.2f, 2.3f}, {2.5f, 2.5f}, {2.11f, 2.64f}, {2.9f, 2.8f}};
    std::vector<Pixel32fC1> checkvalues = {0.3146157861f, 0.4839749336f, 0.5800235868f, 0.6061018705f, 0.8738811016f};

    for (size_t i = 0; i < checkpoints.size(); i++)
    {
        CHECK(Pixel32fC1::Abs(interpol(checkpoints[i].x, checkpoints[i].y) - checkvalues[i]) < 0.000001f);
    }
}

TEST_CASE("Interpolator - Super", "[Common.Image]")
{
    constexpr int widthSrc         = 6;
    constexpr int heightSrc        = 6;
    std::vector<Pixel32fC1> srcImg = {0.5688f, 0.4694f, 0.0119f, 0.3371f, 0.1622f, 0.7943f, 0.3112f, 0.5285f, 0.1656f,
                                      0.6020f, 0.2630f, 0.6541f, 0.6892f, 0.7482f, 0.4505f, 0.0838f, 0.2290f, 0.9133f,
                                      0.1524f, 0.8258f, 0.5383f, 0.9961f, 0.0782f, 0.4427f, 0.1067f, 0.9619f, 0.0046f,
                                      0.7749f, 0.8173f, 0.8687f, 0.0844f, 0.3998f, 0.2599f, 0.8001f, 0.4314f, 0.9106f};

    BorderControl<Pixel32fC1, BorderType::None> bc(srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc},
                                                   {0, 0});

    Interpolator<Pixel32fC1, BorderControl<Pixel32fC1, BorderType::None>, float, InterpolationMode::Super>
        interpolSuper3(bc, 1 / 3.0f, 1 / 3.0f);

    CHECK(interpolSuper3(2.5f, 2.5f) == 0.491725057f);

    Interpolator<Pixel32fC1, BorderControl<Pixel32fC1, BorderType::None>, float, InterpolationMode::Super>
        interpolSuper2(bc, 1 / 2.0f, 1 / 2.0f);

    CHECK(interpolSuper2(2.5f, 2.5f) == (0.4505f + 0.0838f + 0.5383f + 0.9961f) / 4.0f);
    CHECK(interpolSuper2(2.0f, 2.0f) == 1.95655f / 4.0f);
}