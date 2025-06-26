#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>
#include <common/image/functors/borderControl.h>
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

TEST_CASE("BorderControl - Constant", "[Common.Image]")
{
    constexpr int widthSrc  = 4;
    constexpr int heightSrc = 5;
    std::vector<Pixel32sC1> srcImg(widthSrc * heightSrc);
    std::vector<Pixel32sC1> dstImg((widthSrc + 4) * (heightSrc + 4));
    std::iota(srcImg.begin(), srcImg.end(), 0);

    BorderControl<Pixel32sC1, BorderType::Constant, false, false, false, false> bc(
        srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc}, {0, 0}, 100);

    for (int y = 0; y < heightSrc + 4; y++)
    {
        for (int x = 0; x < widthSrc + 4; x++)
        {
            int srcX = x - 2;
            int srcY = y - 2;

            dstImg[to_size_t(y * (widthSrc + 4) + x)] = bc(srcX, srcY);
        }
    }

    std::vector<Pixel32sC1> expected({100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                                      100, 100, 100, 0,   1,   2,   3,   100, 100, 100, 100, 4,   5,   6,   7,
                                      100, 100, 100, 100, 8,   9,   10,  11,  100, 100, 100, 100, 12,  13,  14,
                                      15,  100, 100, 100, 100, 16,  17,  18,  19,  100, 100, 100, 100, 100, 100,
                                      100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100});

    bool isIdentical = true;
    for (size_t i = 0; i < expected.size(); i++)
    {
        isIdentical &= expected[i] == dstImg[i];
    }
    CHECK(isIdentical);
}

TEST_CASE("BorderControl - Replicate", "[Common.Image]")
{
    constexpr int widthSrc  = 4;
    constexpr int heightSrc = 5;
    std::vector<Pixel32sC1> srcImg(widthSrc * heightSrc);
    std::vector<Pixel32sC1> dstImg((widthSrc + 4) * (heightSrc + 4));
    std::iota(srcImg.begin(), srcImg.end(), 0);

    BorderControl<Pixel32sC1, BorderType::Replicate, false, false, false, false> bc(
        srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc}, {0, 0});

    for (int y = 0; y < heightSrc + 4; y++)
    {
        for (int x = 0; x < widthSrc + 4; x++)
        {
            int srcX = x - 2;
            int srcY = y - 2;

            dstImg[to_size_t(y * (widthSrc + 4) + x)] = bc(srcX, srcY);
        }
    }

    std::vector<Pixel32sC1> expected({0,  0,  0,  1,  2,  3,  3,  3,  0,  0,  0,  1,  2,  3,  3,  3,  0,  0,
                                      0,  1,  2,  3,  3,  3,  4,  4,  4,  5,  6,  7,  7,  7,  8,  8,  8,  9,
                                      10, 11, 11, 11, 12, 12, 12, 13, 14, 15, 15, 15, 16, 16, 16, 17, 18, 19,
                                      19, 19, 16, 16, 16, 17, 18, 19, 19, 19, 16, 16, 16, 17, 18, 19, 19, 19});

    bool isIdentical = true;
    for (size_t i = 0; i < expected.size(); i++)
    {
        isIdentical &= expected[i] == dstImg[i];
    }
    CHECK(isIdentical);
}

TEST_CASE("BorderControl - Wrap", "[Common.Image]")
{
    constexpr int widthSrc  = 4;
    constexpr int heightSrc = 5;
    std::vector<Pixel32sC1> srcImg(widthSrc * heightSrc);
    std::vector<Pixel32sC1> dstImg((widthSrc + 4) * (heightSrc + 4));
    std::iota(srcImg.begin(), srcImg.end(), 0);

    BorderControl<Pixel32sC1, BorderType::Wrap, false, false, false, false> bc(srcImg.data(), widthSrc * sizeof(int),
                                                                               {widthSrc, heightSrc}, {0, 0});

    for (int y = 0; y < heightSrc + 4; y++)
    {
        for (int x = 0; x < widthSrc + 4; x++)
        {
            int srcX = x - 2;
            int srcY = y - 2;

            dstImg[to_size_t(y * (widthSrc + 4) + x)] = bc(srcX, srcY);
        }
    }

    std::vector<Pixel32sC1> expected({14, 15, 12, 13, 14, 15, 12, 13, 18, 19, 16, 17, 18, 19, 16, 17, 2,  3,
                                      0,  1,  2,  3,  0,  1,  6,  7,  4,  5,  6,  7,  4,  5,  10, 11, 8,  9,
                                      10, 11, 8,  9,  14, 15, 12, 13, 14, 15, 12, 13, 18, 19, 16, 17, 18, 19,
                                      16, 17, 2,  3,  0,  1,  2,  3,  0,  1,  6,  7,  4,  5,  6,  7,  4,  5});

    bool isIdentical = true;
    for (size_t i = 0; i < expected.size(); i++)
    {
        isIdentical &= expected[i] == dstImg[i];
    }
    CHECK(isIdentical);
}

TEST_CASE("BorderControl - Mirror", "[Common.Image]")
{
    constexpr int widthSrc  = 4;
    constexpr int heightSrc = 5;
    std::vector<Pixel32sC1> srcImg(widthSrc * heightSrc);
    std::vector<Pixel32sC1> dstImg((widthSrc + 4) * (heightSrc + 4));
    std::iota(srcImg.begin(), srcImg.end(), 0);

    BorderControl<Pixel32sC1, BorderType::Mirror, false, false, false, false> bc(srcImg.data(), widthSrc * sizeof(int),
                                                                                 {widthSrc, heightSrc}, {0, 0});

    for (int y = 0; y < heightSrc + 4; y++)
    {
        for (int x = 0; x < widthSrc + 4; x++)
        {
            int srcX = x - 2;
            int srcY = y - 2;

            dstImg[to_size_t(y * (widthSrc + 4) + x)] = bc(srcX, srcY);
        }
    }

    std::vector<Pixel32sC1> expected({10, 9,  8,  9,  10, 11, 10, 9,  6,  5,  4,  5,  6,  7,  6,  5,  2,  1,
                                      0,  1,  2,  3,  2,  1,  6,  5,  4,  5,  6,  7,  6,  5,  10, 9,  8,  9,
                                      10, 11, 10, 9,  14, 13, 12, 13, 14, 15, 14, 13, 18, 17, 16, 17, 18, 19,
                                      18, 17, 14, 13, 12, 13, 14, 15, 14, 13, 10, 9,  8,  9,  10, 11, 10, 9});

    bool isIdentical = true;
    for (size_t i = 0; i < expected.size(); i++)
    {
        isIdentical &= expected[i] == dstImg[i];
    }
    CHECK(isIdentical);
}

TEST_CASE("BorderControl - MirrorReplicate", "[Common.Image]")
{
    constexpr int widthSrc  = 4;
    constexpr int heightSrc = 5;
    std::vector<Pixel32sC1> srcImg(widthSrc * heightSrc);
    std::vector<Pixel32sC1> dstImg((widthSrc + 4) * (heightSrc + 4));
    std::iota(srcImg.begin(), srcImg.end(), 0);

    BorderControl<Pixel32sC1, BorderType::MirrorReplicate, false, false, false, false> bc(
        srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc}, {0, 0});

    for (int y = 0; y < heightSrc + 4; y++)
    {
        for (int x = 0; x < widthSrc + 4; x++)
        {
            int srcX = x - 2;
            int srcY = y - 2;

            dstImg[to_size_t(y * (widthSrc + 4) + x)] = bc(srcX, srcY);
        }
    }

    std::vector<Pixel32sC1> expected({5,  4,  4,  5,  6,  7,  7,  6,  1,  0,  0,  1,  2,  3,  3,  2,  1,  0,
                                      0,  1,  2,  3,  3,  2,  5,  4,  4,  5,  6,  7,  7,  6,  9,  8,  8,  9,
                                      10, 11, 11, 10, 13, 12, 12, 13, 14, 15, 15, 14, 17, 16, 16, 17, 18, 19,
                                      19, 18, 17, 16, 16, 17, 18, 19, 19, 18, 13, 12, 12, 13, 14, 15, 15, 14});

    bool isIdentical = true;
    for (size_t i = 0; i < expected.size(); i++)
    {
        isIdentical &= expected[i] == dstImg[i];
    }
    CHECK(isIdentical);
}

TEST_CASE("BorderControl - Mirror (avoid branching)", "[Common.Image]")
{
    constexpr int widthSrc  = 4;
    constexpr int heightSrc = 5;
    std::vector<Pixel32sC1> srcImg(widthSrc * heightSrc);
    std::vector<Pixel32sC1> dstImg((widthSrc + 4) * (heightSrc + 4));
    std::iota(srcImg.begin(), srcImg.end(), 0);

    BorderControl<Pixel32sC1, BorderType::Mirror, false, false, true, false> bc(srcImg.data(), widthSrc * sizeof(int),
                                                                                {widthSrc, heightSrc}, {0, 0});

    for (int y = 0; y < heightSrc + 4; y++)
    {
        for (int x = 0; x < widthSrc + 4; x++)
        {
            int srcX = x - 2;
            int srcY = y - 2;

            dstImg[to_size_t(y * (widthSrc + 4) + x)] = bc(srcX, srcY);
        }
    }

    std::vector<Pixel32sC1> expected({10, 9,  8,  9,  10, 11, 10, 9,  6,  5,  4,  5,  6,  7,  6,  5,  2,  1,
                                      0,  1,  2,  3,  2,  1,  6,  5,  4,  5,  6,  7,  6,  5,  10, 9,  8,  9,
                                      10, 11, 10, 9,  14, 13, 12, 13, 14, 15, 14, 13, 18, 17, 16, 17, 18, 19,
                                      18, 17, 14, 13, 12, 13, 14, 15, 14, 13, 10, 9,  8,  9,  10, 11, 10, 9});

    bool isIdentical = true;
    for (size_t i = 0; i < expected.size(); i++)
    {
        isIdentical &= expected[i] == dstImg[i];
    }
    CHECK(isIdentical);
}

TEST_CASE("BorderControl - MirrorReplicate (avoid branching)", "[Common.Image]")
{
    constexpr int widthSrc  = 4;
    constexpr int heightSrc = 5;
    std::vector<Pixel32sC1> srcImg(widthSrc * heightSrc);
    std::vector<Pixel32sC1> dstImg((widthSrc + 4) * (heightSrc + 4));
    std::iota(srcImg.begin(), srcImg.end(), 0);

    BorderControl<Pixel32sC1, BorderType::MirrorReplicate, false, false, true, false> bc(
        srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc}, {0, 0});

    for (int y = 0; y < heightSrc + 4; y++)
    {
        for (int x = 0; x < widthSrc + 4; x++)
        {
            int srcX = x - 2;
            int srcY = y - 2;

            dstImg[to_size_t(y * (widthSrc + 4) + x)] = bc(srcX, srcY);
        }
    }

    std::vector<Pixel32sC1> expected({5,  4,  4,  5,  6,  7,  7,  6,  1,  0,  0,  1,  2,  3,  3,  2,  1,  0,
                                      0,  1,  2,  3,  3,  2,  5,  4,  4,  5,  6,  7,  7,  6,  9,  8,  8,  9,
                                      10, 11, 11, 10, 13, 12, 12, 13, 14, 15, 15, 14, 17, 16, 16, 17, 18, 19,
                                      19, 18, 17, 16, 16, 17, 18, 19, 19, 18, 13, 12, 12, 13, 14, 15, 15, 14});

    bool isIdentical = true;
    for (size_t i = 0; i < expected.size(); i++)
    {
        isIdentical &= expected[i] == dstImg[i];
    }
    CHECK(isIdentical);
}

TEST_CASE("BorderControl - Wrap (iterative)", "[Common.Image]")
{
    constexpr int widthSrc  = 4;
    constexpr int heightSrc = 5;
    std::vector<Pixel32sC1> srcImg(widthSrc * heightSrc);
    std::vector<Pixel32sC1> dstImg((widthSrc * 4) * (heightSrc * 4));
    std::iota(srcImg.begin(), srcImg.end(), 0);

    BorderControl<Pixel32sC1, BorderType::Wrap, false, true, false, false> bc(srcImg.data(), widthSrc * sizeof(int),
                                                                              {widthSrc, heightSrc}, {0, 0});

    for (int y = 0; y < heightSrc * 4; y++)
    {
        for (int x = 0; x < widthSrc * 4; x++)
        {
            int srcX = x - widthSrc;
            int srcY = y - heightSrc;

            dstImg[to_size_t(y * (widthSrc * 4) + x)] = bc(srcX, srcY);
        }
    }

    std::vector<Pixel32sC1> expected(
        {0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  4,  5,  6,  7,  4,  5,  6,  7,  4,  5,  6,
         7,  4,  5,  6,  7,  8,  9,  10, 11, 8,  9,  10, 11, 8,  9,  10, 11, 8,  9,  10, 11, 12, 13, 14, 15, 12, 13,
         14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19, 0,
         1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  4,  5,  6,  7,  4,  5,  6,  7,  4,  5,  6,  7,
         4,  5,  6,  7,  8,  9,  10, 11, 8,  9,  10, 11, 8,  9,  10, 11, 8,  9,  10, 11, 12, 13, 14, 15, 12, 13, 14,
         15, 12, 13, 14, 15, 12, 13, 14, 15, 16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19, 0,  1,
         2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  4,  5,  6,  7,  4,  5,  6,  7,  4,  5,  6,  7,  4,
         5,  6,  7,  8,  9,  10, 11, 8,  9,  10, 11, 8,  9,  10, 11, 8,  9,  10, 11, 12, 13, 14, 15, 12, 13, 14, 15,
         12, 13, 14, 15, 12, 13, 14, 15, 16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19, 0,  1,  2,
         3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  4,  5,  6,  7,  4,  5,  6,  7,  4,  5,  6,  7,  4,  5,
         6,  7,  8,  9,  10, 11, 8,  9,  10, 11, 8,  9,  10, 11, 8,  9,  10, 11, 12, 13, 14, 15, 12, 13, 14, 15, 12,
         13, 14, 15, 12, 13, 14, 15, 16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19, 16, 17, 18, 19});

    bool isIdentical = true;
    for (size_t i = 0; i < expected.size(); i++)
    {
        isIdentical &= expected[i] == dstImg[i];
    }
    CHECK(isIdentical);
}

TEST_CASE("BorderControl - Mirror (iterative)", "[Common.Image]")
{
    constexpr int widthSrc  = 4;
    constexpr int heightSrc = 5;
    std::vector<Pixel32sC1> srcImg(widthSrc * heightSrc);
    std::vector<Pixel32sC1> dstImg((widthSrc * 4) * (heightSrc * 4));
    std::iota(srcImg.begin(), srcImg.end(), 0);

    BorderControl<Pixel32sC1, BorderType::Mirror, false, true, false, false> bc(srcImg.data(), widthSrc * sizeof(int),
                                                                                {widthSrc, heightSrc}, {0, 0});

    for (int y = 0; y < heightSrc * 4; y++)
    {
        for (int x = 0; x < widthSrc * 4; x++)
        {
            int srcX = x - widthSrc;
            int srcY = y - heightSrc;

            dstImg[to_size_t(y * (widthSrc * 4) + x)] = bc(srcX, srcY);
        }
    }

    std::vector<Pixel32sC1> expected(
        {14, 15, 14, 13, 12, 13, 14, 15, 14, 13, 12, 13, 14, 15, 14, 13, 18, 19, 18, 17, 16, 17, 18, 19, 18, 17, 16,
         17, 18, 19, 18, 17, 14, 15, 14, 13, 12, 13, 14, 15, 14, 13, 12, 13, 14, 15, 14, 13, 10, 11, 10, 9,  8,  9,
         10, 11, 10, 9,  8,  9,  10, 11, 10, 9,  6,  7,  6,  5,  4,  5,  6,  7,  6,  5,  4,  5,  6,  7,  6,  5,  2,
         3,  2,  1,  0,  1,  2,  3,  2,  1,  0,  1,  2,  3,  2,  1,  6,  7,  6,  5,  4,  5,  6,  7,  6,  5,  4,  5,
         6,  7,  6,  5,  10, 11, 10, 9,  8,  9,  10, 11, 10, 9,  8,  9,  10, 11, 10, 9,  14, 15, 14, 13, 12, 13, 14,
         15, 14, 13, 12, 13, 14, 15, 14, 13, 18, 19, 18, 17, 16, 17, 18, 19, 18, 17, 16, 17, 18, 19, 18, 17, 14, 15,
         14, 13, 12, 13, 14, 15, 14, 13, 12, 13, 14, 15, 14, 13, 10, 11, 10, 9,  8,  9,  10, 11, 10, 9,  8,  9,  10,
         11, 10, 9,  6,  7,  6,  5,  4,  5,  6,  7,  6,  5,  4,  5,  6,  7,  6,  5,  2,  3,  2,  1,  0,  1,  2,  3,
         2,  1,  0,  1,  2,  3,  2,  1,  6,  7,  6,  5,  4,  5,  6,  7,  6,  5,  4,  5,  6,  7,  6,  5,  10, 11, 10,
         9,  8,  9,  10, 11, 10, 9,  8,  9,  10, 11, 10, 9,  14, 15, 14, 13, 12, 13, 14, 15, 14, 13, 12, 13, 14, 15,
         14, 13, 18, 19, 18, 17, 16, 17, 18, 19, 18, 17, 16, 17, 18, 19, 18, 17, 14, 15, 14, 13, 12, 13, 14, 15, 14,
         13, 12, 13, 14, 15, 14, 13, 10, 11, 10, 9,  8,  9,  10, 11, 10, 9,  8,  9,  10, 11, 10, 9});

    bool isIdentical = true;
    for (size_t i = 0; i < expected.size(); i++)
    {
        isIdentical &= expected[i] == dstImg[i];
    }
    CHECK(isIdentical);
}

TEST_CASE("BorderControl - MirrorReplicate (iterative)", "[Common.Image]")
{
    constexpr int widthSrc  = 4;
    constexpr int heightSrc = 5;
    std::vector<Pixel32sC1> srcImg(widthSrc * heightSrc);
    std::vector<Pixel32sC1> dstImg((widthSrc * 4) * (heightSrc * 4));
    std::iota(srcImg.begin(), srcImg.end(), 0);

    BorderControl<Pixel32sC1, BorderType::MirrorReplicate, false, true, false, false> bc(
        srcImg.data(), widthSrc * sizeof(int), {widthSrc, heightSrc}, {0, 0});

    for (int y = 0; y < heightSrc * 4; y++)
    {
        for (int x = 0; x < widthSrc * 4; x++)
        {
            int srcX = x - widthSrc;
            int srcY = y - heightSrc;

            dstImg[to_size_t(y * (widthSrc * 4) + x)] = bc(srcX, srcY);
        }
    }

    std::vector<Pixel32sC1> expected(
        {19, 18, 17, 16, 16, 17, 18, 19, 19, 18, 17, 16, 16, 17, 18, 19, 15, 14, 13, 12, 12, 13, 14, 15, 15, 14, 13,
         12, 12, 13, 14, 15, 11, 10, 9,  8,  8,  9,  10, 11, 11, 10, 9,  8,  8,  9,  10, 11, 7,  6,  5,  4,  4,  5,
         6,  7,  7,  6,  5,  4,  4,  5,  6,  7,  3,  2,  1,  0,  0,  1,  2,  3,  3,  2,  1,  0,  0,  1,  2,  3,  3,
         2,  1,  0,  0,  1,  2,  3,  3,  2,  1,  0,  0,  1,  2,  3,  7,  6,  5,  4,  4,  5,  6,  7,  7,  6,  5,  4,
         4,  5,  6,  7,  11, 10, 9,  8,  8,  9,  10, 11, 11, 10, 9,  8,  8,  9,  10, 11, 15, 14, 13, 12, 12, 13, 14,
         15, 15, 14, 13, 12, 12, 13, 14, 15, 19, 18, 17, 16, 16, 17, 18, 19, 19, 18, 17, 16, 16, 17, 18, 19, 19, 18,
         17, 16, 16, 17, 18, 19, 19, 18, 17, 16, 16, 17, 18, 19, 15, 14, 13, 12, 12, 13, 14, 15, 15, 14, 13, 12, 12,
         13, 14, 15, 11, 10, 9,  8,  8,  9,  10, 11, 11, 10, 9,  8,  8,  9,  10, 11, 7,  6,  5,  4,  4,  5,  6,  7,
         7,  6,  5,  4,  4,  5,  6,  7,  3,  2,  1,  0,  0,  1,  2,  3,  3,  2,  1,  0,  0,  1,  2,  3,  3,  2,  1,
         0,  0,  1,  2,  3,  3,  2,  1,  0,  0,  1,  2,  3,  7,  6,  5,  4,  4,  5,  6,  7,  7,  6,  5,  4,  4,  5,
         6,  7,  11, 10, 9,  8,  8,  9,  10, 11, 11, 10, 9,  8,  8,  9,  10, 11, 15, 14, 13, 12, 12, 13, 14, 15, 15,
         14, 13, 12, 12, 13, 14, 15, 19, 18, 17, 16, 16, 17, 18, 19, 19, 18, 17, 16, 16, 17, 18, 19});

    bool isIdentical = true;
    for (size_t i = 0; i < expected.size(); i++)
    {
        isIdentical &= expected[i] == dstImg[i];
    }
    CHECK(isIdentical);
}

TEST_CASE("BorderControl - Planar", "[Common.Image]")
{
    constexpr int widthSrc  = 4;
    constexpr int heightSrc = 5;
    std::vector<Pixel32sC1> srcImg(widthSrc * heightSrc);
    std::vector<Pixel32sC1> dstImg((widthSrc * 4) * (heightSrc * 4));
    std::iota(srcImg.begin(), srcImg.end(), 0);

    BorderControl<Pixel32sC4, BorderType::MirrorReplicate, false, true, false, true> bc(
        srcImg.data(), widthSrc * sizeof(int), srcImg.data() + 1, widthSrc * sizeof(int), srcImg.data() + 2,
        widthSrc * sizeof(int), srcImg.data() + 3, widthSrc * sizeof(int), {widthSrc, heightSrc}, {0, 0});

    CHECK(bc(1, 1) == Pixel32sC4(5, 6, 7, 8));
}

TEST_CASE("BorderControl - Inplace", "[Common.Image]")
{
    constexpr int widthSrc  = 4;
    constexpr int heightSrc = 5;

    BorderControl<Pixel32sC1, BorderType::MirrorReplicate, false, false, false, false> bc({widthSrc, heightSrc},
                                                                                          {3, 3});

    int x = 1;
    int y = 2;

    bc.AdjustCoordinates(x, y);
    CHECK(x == 3);
    CHECK(y == 4);
}